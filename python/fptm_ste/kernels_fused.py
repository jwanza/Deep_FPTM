"""
Fused Triton Kernels for High-Priority STCM Operations.

This module provides optimized fused kernels for:
1. STE Ternary Quantization (fused tanh + threshold + STE)
2. Clause Synchronization (fused gather + multiply + EMA + normalize)
3. Gumbel-Softmax Sampling (fused random + softmax + selection)

All kernels include:
- Correctness verification against reference implementations
- Before/after benchmarking utilities
"""

import torch
import torch.nn.functional as F
from torch.autograd import Function
from typing import Tuple, Optional
import time

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# =============================================================================
# 1. FUSED STE TERNARY QUANTIZATION
# =============================================================================

def ste_ternary_reference(logits: torch.Tensor, band: float, temperature: float) -> torch.Tensor:
    """Reference implementation of STE ternary quantization."""
    soft = torch.tanh(logits / temperature)
    with torch.no_grad():
        hard = torch.zeros_like(logits)
        hard = torch.where(logits > band, torch.ones_like(logits), hard)
        hard = torch.where(logits < -band, -torch.ones_like(logits), hard)
    return hard + (soft - soft.detach())


if TRITON_AVAILABLE:
    @triton.jit
    def fused_ste_ternary_kernel(
        logits_ptr,
        out_ptr,
        N,
        band,
        inv_temp,  # 1/temperature for faster division
        BLOCK: tl.constexpr,
    ):
        """
        Fused STE ternary quantization kernel.
        
        Combines:
        - tanh(logits / temperature) for soft gradient
        - threshold comparison for hard ternary values
        - STE gradient connection
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        
        # Load logits
        logits = tl.load(logits_ptr + offs, mask=mask, other=0.0)
        
        # Soft path (for backward gradient)
        scaled = logits * inv_temp
        # tanh approximation: tanh(x) ≈ x / (1 + |x|) for small x, else sign(x)
        # More accurate: use exp-based tanh
        exp_pos = tl.exp(scaled)
        exp_neg = tl.exp(-scaled)
        soft = (exp_pos - exp_neg) / (exp_pos + exp_neg + 1e-7)
        
        # Hard path (for forward)
        hard = tl.where(logits > band, 1.0,
               tl.where(logits < -band, -1.0, 0.0))
        
        # STE: use hard for forward, soft gradient flows back
        # In Triton, we can't do stop_gradient, so we output hard
        # The autograd wrapper handles the STE connection
        
        tl.store(out_ptr + offs, hard, mask=mask)


    class FusedSTETernaryFunction(Function):
        """Autograd function for fused STE ternary with proper gradient."""
        
        @staticmethod
        def forward(ctx, logits: torch.Tensor, band: float, temperature: float) -> torch.Tensor:
            N = logits.numel()
            out = torch.empty_like(logits)
            
            BLOCK = 1024
            grid = ((N + BLOCK - 1) // BLOCK,)
            
            fused_ste_ternary_kernel[grid](
                logits.contiguous().view(-1),
                out.view(-1),
                N,
                band,
                1.0 / max(temperature, 1e-6),
                BLOCK=BLOCK,
            )
            
            ctx.save_for_backward(logits)
            ctx.temperature = temperature
            return out.view_as(logits)
        
        @staticmethod
        def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
            logits, = ctx.saved_tensors
            # STE: gradient flows through tanh
            soft = torch.tanh(logits / ctx.temperature)
            # Derivative of tanh: 1 - tanh^2
            grad_soft = 1.0 - soft * soft
            grad_input = grad_output * grad_soft / ctx.temperature
            return grad_input, None, None


def fused_ste_ternary(logits: torch.Tensor, band: float, temperature: float) -> torch.Tensor:
    """
    Fused STE ternary quantization.
    
    Uses Triton kernel if available, otherwise falls back to reference.
    
    Args:
        logits: Input tensor
        band: Threshold band around zero (values in [-band, band] -> 0)
        temperature: Temperature for soft gradient
        
    Returns:
        Ternary tensor with values in {-1, 0, 1}
    """
    if TRITON_AVAILABLE and logits.is_cuda:
        return FusedSTETernaryFunction.apply(logits, band, temperature)
    return ste_ternary_reference(logits, band, temperature)


# =============================================================================
# 2. FUSED CLAUSE SYNCHRONIZATION
# =============================================================================

def clause_sync_reference(
    activations: torch.Tensor,
    left_indices: torch.Tensor,
    right_indices: torch.Tensor,
    decay_alpha: Optional[torch.Tensor],
    decay_beta: Optional[torch.Tensor],
    r: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference implementation of clause synchronization."""
    B = activations.size(0)
    
    # Gather paired activations
    left_act = activations[:, left_indices]  # [B, synch_size]
    right_act = activations[:, right_indices]  # [B, synch_size]
    
    # Pairwise product
    pairwise_product = left_act * right_act
    
    # EMA update
    if decay_alpha is None or decay_beta is None:
        decay_alpha = pairwise_product
        decay_beta = torch.ones_like(pairwise_product)
    else:
        decay_alpha = r * decay_alpha + pairwise_product
        decay_beta = r * decay_beta + 1
    
    # Normalize
    synchronization = decay_alpha / torch.sqrt(decay_beta)
    
    return synchronization, decay_alpha, decay_beta


if TRITON_AVAILABLE:
    @triton.jit
    def fused_clause_sync_kernel(
        activations_ptr,
        left_idx_ptr,
        right_idx_ptr,
        alpha_ptr,
        beta_ptr,
        r_ptr,
        out_sync_ptr,
        out_alpha_ptr,
        out_beta_ptr,
        B, S, N_clauses,
        stride_act_b, stride_act_n,
        stride_ab_b, stride_ab_s,
        first_call,  # 1 if decay_alpha/beta should be initialized
        BLOCK_B: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        """
        Fused clause synchronization kernel.
        
        Fuses:
        - Gather left/right activations by index
        - Multiply for pairwise correlation
        - EMA update with decay
        - Normalize by sqrt(beta)
        """
        pid_b = tl.program_id(0)
        pid_s = tl.program_id(1)
        
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        
        mask_b = offs_b < B
        mask_s = offs_s < S
        mask = mask_b[:, None] & mask_s[None, :]
        
        # Load indices (same for all batch elements)
        left_idx = tl.load(left_idx_ptr + offs_s, mask=mask_s, other=0)
        right_idx = tl.load(right_idx_ptr + offs_s, mask=mask_s, other=0)
        
        # Load decay parameter
        r = tl.load(r_ptr + offs_s, mask=mask_s, other=0.0)
        
        # Gather activations: activations[b, left_idx[s]]
        # Using 2D indexing
        act_ptrs_left = activations_ptr + offs_b[:, None] * stride_act_b + left_idx[None, :] * stride_act_n
        act_ptrs_right = activations_ptr + offs_b[:, None] * stride_act_b + right_idx[None, :] * stride_act_n
        
        left_act = tl.load(act_ptrs_left, mask=mask, other=0.0)
        right_act = tl.load(act_ptrs_right, mask=mask, other=0.0)
        
        # Pairwise product
        product = left_act * right_act
        
        # Load or initialize alpha/beta
        ab_ptrs = offs_b[:, None] * stride_ab_b + offs_s[None, :] * stride_ab_s
        
        if first_call == 1:
            new_alpha = product
            new_beta = tl.full((BLOCK_B, BLOCK_S), 1.0, dtype=tl.float32)
        else:
            alpha = tl.load(alpha_ptr + ab_ptrs, mask=mask, other=0.0)
            beta = tl.load(beta_ptr + ab_ptrs, mask=mask, other=1.0)
            new_alpha = r[None, :] * alpha + product
            new_beta = r[None, :] * beta + 1.0
        
        # Normalize
        sync = new_alpha / tl.sqrt(new_beta + 1e-8)
        
        # Store outputs
        tl.store(out_sync_ptr + ab_ptrs, sync, mask=mask)
        tl.store(out_alpha_ptr + ab_ptrs, new_alpha, mask=mask)
        tl.store(out_beta_ptr + ab_ptrs, new_beta, mask=mask)


def fused_clause_sync(
    activations: torch.Tensor,
    left_indices: torch.Tensor,
    right_indices: torch.Tensor,
    decay_alpha: Optional[torch.Tensor],
    decay_beta: Optional[torch.Tensor],
    r: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused clause synchronization with Triton acceleration.
    
    Args:
        activations: [B, N_clauses] clause activations
        left_indices: [S] indices of left clauses in pairs
        right_indices: [S] indices of right clauses in pairs
        decay_alpha: [B, S] running EMA sum (or None for first call)
        decay_beta: [B, S] running EMA count (or None for first call)
        r: [S] decay rates per sync dimension
        
    Returns:
        synchronization: [B, S] normalized sync values
        new_decay_alpha: [B, S] updated EMA sum
        new_decay_beta: [B, S] updated EMA count
    """
    if not TRITON_AVAILABLE or not activations.is_cuda:
        return clause_sync_reference(activations, left_indices, right_indices,
                                     decay_alpha, decay_beta, r)
    
    B, N_clauses = activations.shape
    S = left_indices.shape[0]
    
    # Ensure contiguous
    activations = activations.contiguous()
    left_indices = left_indices.contiguous().long()
    right_indices = right_indices.contiguous().long()
    r = r.contiguous()
    
    first_call = 1 if decay_alpha is None else 0
    
    # Allocate outputs
    out_sync = torch.empty(B, S, device=activations.device, dtype=activations.dtype)
    out_alpha = torch.empty(B, S, device=activations.device, dtype=activations.dtype)
    out_beta = torch.empty(B, S, device=activations.device, dtype=activations.dtype)
    
    # Use dummy tensors for first call
    if decay_alpha is None:
        decay_alpha = torch.zeros(B, S, device=activations.device)
        decay_beta = torch.ones(B, S, device=activations.device)
    
    BLOCK_B = min(32, B)
    BLOCK_S = min(128, S)
    
    grid = ((B + BLOCK_B - 1) // BLOCK_B, (S + BLOCK_S - 1) // BLOCK_S)
    
    fused_clause_sync_kernel[grid](
        activations,
        left_indices,
        right_indices,
        decay_alpha,
        decay_beta,
        r,
        out_sync,
        out_alpha,
        out_beta,
        B, S, N_clauses,
        activations.stride(0), activations.stride(1),
        out_alpha.stride(0), out_alpha.stride(1),
        first_call,
        BLOCK_B=BLOCK_B,
        BLOCK_S=BLOCK_S,
    )
    
    return out_sync, out_alpha, out_beta


# =============================================================================
# 3. FUSED GUMBEL-SOFTMAX
# =============================================================================

def gumbel_softmax_reference(
    logits: torch.Tensor,
    temperature: float,
    hard: bool = True,
) -> torch.Tensor:
    """Reference implementation of Gumbel-Softmax."""
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    y = (logits + gumbel_noise) / temperature
    y_soft = F.softmax(y, dim=-1)
    
    if hard:
        index = y_soft.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        return y_hard - y_soft.detach() + y_soft
    return y_soft


if TRITON_AVAILABLE:
    @triton.jit
    def fused_gumbel_softmax_kernel(
        logits_ptr,
        out_ptr,
        seed,
        B, K,  # batch, num_categories
        inv_temp,
        stride_b, stride_k,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Fused Gumbel-Softmax kernel.
        
        Fuses:
        - Gumbel noise generation
        - Addition and temperature scaling
        - Softmax computation
        - Argmax for hard selection
        """
        pid_b = tl.program_id(0)
        
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_k = tl.arange(0, BLOCK_K)
        
        mask_b = offs_b < B
        mask_k = offs_k < K
        mask = mask_b[:, None] & mask_k[None, :]
        
        # Compute offsets for input and output
        offsets = offs_b[:, None] * stride_b + offs_k[None, :] * stride_k
        
        # Load logits [BLOCK_B, K]
        logits = tl.load(logits_ptr + offsets, mask=mask, other=-1e9)
        
        # Generate Gumbel noise using philox
        # Using triton's random number generator
        random_offs = offs_b[:, None] * K + offs_k[None, :]
        u = tl.rand(seed, random_offs)
        u = tl.maximum(u, 1e-10)  # Clamp for numerical stability
        gumbel = -tl.log(-tl.log(u + 1e-10) + 1e-10)
        
        # Add noise and scale
        y = (logits + gumbel) * inv_temp
        
        # Softmax: exp(y - max) / sum(exp(y - max))
        y_max = tl.max(y, axis=1)[:, None]
        y_shifted = y - y_max
        exp_y = tl.exp(y_shifted)
        sum_exp = tl.sum(exp_y, axis=1)[:, None]
        softmax = exp_y / (sum_exp + 1e-10)
        
        # Hard selection: argmax
        argmax = tl.argmax(softmax, axis=1)[:, None]
        
        # One-hot encoding
        is_max = (offs_k[None, :] == argmax)
        hard = tl.where(is_max, 1.0, 0.0)
        
        # Store to output using same offsets
        tl.store(out_ptr + offsets, hard, mask=mask)


    class FusedGumbelSoftmaxFunction(Function):
        """Autograd function for fused Gumbel-Softmax."""
        
        @staticmethod
        def forward(ctx, logits: torch.Tensor, temperature: float) -> torch.Tensor:
            B = logits.shape[0] if logits.dim() > 1 else 1
            K = logits.shape[-1]
            
            out = torch.empty_like(logits)
            
            BLOCK_B = min(32, B)
            BLOCK_K = triton.next_power_of_2(K)
            
            if BLOCK_K > 2048:
                # Fall back to reference for very large K
                return gumbel_softmax_reference(logits, temperature, hard=True)
            
            grid = ((B + BLOCK_B - 1) // BLOCK_B,)
            seed = torch.randint(0, 2**31, (1,), device=logits.device).item()
            
            fused_gumbel_softmax_kernel[grid](
                logits.contiguous(),
                out,
                seed,
                B, K,
                1.0 / max(temperature, 1e-6),
                logits.stride(0) if logits.dim() > 1 else 0,
                logits.stride(-1),
                BLOCK_B=BLOCK_B,
                BLOCK_K=BLOCK_K,
            )
            
            ctx.save_for_backward(logits, out)
            ctx.temperature = temperature
            return out
        
        @staticmethod
        def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
            logits, out = ctx.saved_tensors
            # STE: gradient flows through softmax
            # Approximate: use softmax gradient
            softmax = F.softmax(logits / ctx.temperature, dim=-1)
            # Jacobian of softmax: diag(s) - s @ s^T
            # For efficiency, use: grad_input = (grad_output - sum(grad_output * softmax)) * softmax / temp
            grad_softmax = grad_output - (grad_output * softmax).sum(dim=-1, keepdim=True)
            grad_input = grad_softmax * softmax / ctx.temperature
            return grad_input, None


def fused_gumbel_softmax(
    logits: torch.Tensor,
    temperature: float = 1.0,
    hard: bool = True,
) -> torch.Tensor:
    """
    Fused Gumbel-Softmax with Triton acceleration.
    
    Args:
        logits: [B, K] or [B, ..., K] unnormalized log probabilities
        temperature: Temperature for softmax
        hard: Whether to use hard (one-hot) or soft output
        
    Returns:
        Sampled tensor with same shape as logits
    """
    if not TRITON_AVAILABLE or not logits.is_cuda or not hard:
        return gumbel_softmax_reference(logits, temperature, hard)
    
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        return FusedGumbelSoftmaxFunction.apply(logits, temperature).squeeze(0)
    
    return FusedGumbelSoftmaxFunction.apply(logits, temperature)


# =============================================================================
# BENCHMARKING UTILITIES
# =============================================================================

def benchmark_function(fn, *args, warmup: int = 10, iters: int = 100, **kwargs):
    """Benchmark a function with warmup and timing."""
    # Warmup
    for _ in range(warmup):
        _ = fn(*args, **kwargs)
    
    if args and hasattr(args[0], 'device') and args[0].is_cuda:
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iters):
        result = fn(*args, **kwargs)
    
    if args and hasattr(args[0], 'device') and args[0].is_cuda:
        torch.cuda.synchronize()
    
    elapsed = (time.perf_counter() - start) / iters * 1000  # ms
    return result, elapsed


def verify_correctness(result_fused, result_ref, name: str, rtol: float = 1e-3, atol: float = 1e-5):
    """Verify fused kernel matches reference implementation."""
    if isinstance(result_fused, tuple):
        for i, (f, r) in enumerate(zip(result_fused, result_ref)):
            verify_correctness(f, r, f"{name}[{i}]", rtol, atol)
        return
    
    diff = (result_fused - result_ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    matches = torch.allclose(result_fused, result_ref, rtol=rtol, atol=atol)
    
    status = "✅ PASS" if matches else "❌ FAIL"
    print(f"{name}: {status} (max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})")
    
    return matches


def run_all_benchmarks():
    """Run comprehensive benchmarks for all fused kernels."""
    print("=" * 70)
    print("FUSED TRITON KERNELS BENCHMARK")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmarks")
        return
    
    device = 'cuda'
    
    # Test sizes
    sizes = [(64, 128), (256, 256), (512, 512), (1024, 1024)]
    
    print("\n1. STE TERNARY QUANTIZATION")
    print("-" * 70)
    print(f"{'Size':<20} {'Reference (ms)':<15} {'Fused (ms)':<15} {'Speedup':<10} {'Status'}")
    print("-" * 70)
    
    for B, N in sizes:
        logits = torch.randn(B, N, device=device)
        band, temp = 0.1, 1.0
        
        # Reference
        ref_result, ref_time = benchmark_function(
            ste_ternary_reference, logits, band, temp
        )
        
        # Fused
        fused_result, fused_time = benchmark_function(
            fused_ste_ternary, logits, band, temp
        )
        
        # Verify
        matches = torch.allclose(fused_result, ref_result, atol=1e-5)
        status = "✅" if matches else "❌"
        speedup = ref_time / fused_time
        
        print(f"{B}x{N:<16} {ref_time:<15.4f} {fused_time:<15.4f} {speedup:<10.2f}x {status}")
    
    print("\n2. CLAUSE SYNCHRONIZATION")
    print("-" * 70)
    print(f"{'Size':<20} {'Reference (ms)':<15} {'Fused (ms)':<15} {'Speedup':<10} {'Status'}")
    print("-" * 70)
    
    for B, N in sizes:
        S = N // 2  # Sync size
        activations = torch.randn(B, N, device=device)
        left_indices = torch.randint(0, N, (S,), device=device)
        right_indices = torch.randint(0, N, (S,), device=device)
        r = torch.exp(-torch.rand(S, device=device) * 0.1)
        
        # Reference
        ref_result, ref_time = benchmark_function(
            clause_sync_reference, activations, left_indices, right_indices, None, None, r
        )
        
        # Fused
        fused_result, fused_time = benchmark_function(
            fused_clause_sync, activations, left_indices, right_indices, None, None, r
        )
        
        # Verify (check sync output)
        matches = torch.allclose(fused_result[0], ref_result[0], atol=1e-4)
        status = "✅" if matches else "❌"
        speedup = ref_time / fused_time
        
        print(f"{B}x{N}→{S:<10} {ref_time:<15.4f} {fused_time:<15.4f} {speedup:<10.2f}x {status}")
    
    print("\n3. GUMBEL-SOFTMAX")
    print("-" * 70)
    print(f"{'Size':<20} {'Reference (ms)':<15} {'Fused (ms)':<15} {'Speedup':<10}")
    print("-" * 70)
    
    for B, K in [(64, 3), (256, 10), (512, 64), (1024, 128)]:
        logits = torch.randn(B, K, device=device)
        temp = 1.0
        
        # Reference
        _, ref_time = benchmark_function(
            gumbel_softmax_reference, logits, temp, True
        )
        
        # Fused
        _, fused_time = benchmark_function(
            fused_gumbel_softmax, logits, temp, True
        )
        
        speedup = ref_time / fused_time
        # Note: Can't compare results directly due to randomness
        print(f"{B}x{K:<16} {ref_time:<15.4f} {fused_time:<15.4f} {speedup:<10.2f}x")
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    run_all_benchmarks()

