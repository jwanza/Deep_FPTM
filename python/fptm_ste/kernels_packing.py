"""
GPU-Accelerated Weight Packing Kernels.

This module provides Triton kernels for fast packing/unpacking of:
1. Boolean tensors into int32 (32 bools per int)
2. Ternary weights into int16 (8 ternary values per int16)
3. Ternary weights into int32 (16 ternary values per int32)

All operations are GPU-accelerated for maximum throughput.
"""
import torch
from typing import Tuple

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

if TRITON_AVAILABLE:
    @triton.jit
    def pack_bool_to_int32_kernel(
        x_ptr,      # [N] boolean/float input
        out_ptr,    # [N // 32] int32 output
        n_elements,
        BLOCK_SIZE: tl.constexpr
    ):
        """Pack boolean tensor into int32."""
        pid = tl.program_id(0)
        out_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        
        in_base = out_idx * 32
        mask = in_base < n_elements
        
        packed = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
        
        for i in range(32):
            val_ptr = x_ptr + in_base + i
            val = tl.load(val_ptr, mask=mask, other=0.0)
            bit = (val > 0.5).to(tl.int32)
            packed |= (bit << i)
            
        tl.store(out_ptr + out_idx, packed, mask=mask)

    @triton.jit
    def pack_ternary_to_int16_kernel(
        w_ptr,      # [N, K] ternary weights
        out_ptr,    # [N, K_packed] int16 output
        N, K, K_packed,
        stride_wn, stride_wk,
        stride_on, stride_ok,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Pack ternary weights (-1, 0, 1) into int16.
        Each int16 stores 8 ternary values (2 bits each).
        Encoding: +1 = 01, -1 = 10, 0 = 00
        """
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        
        n_mask = offs_n < N
        k_mask = offs_k < K_packed
        
        # Initialize packed output
        packed = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.int32)
        
        # Pack 8 ternary values into each int16
        for i in range(8):
            # Load ternary value
            w_col = offs_k * 8 + i
            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + w_col[None, :] * stride_wk
            w_mask = n_mask[:, None] & (w_col[None, :] < K)
            w = tl.load(w_ptrs, mask=w_mask, other=0.0)
            
            # Encode: +1 -> 01, -1 -> 10, 0 -> 00
            pos_bit = (w > 0.5).to(tl.int32)
            neg_bit = (w < -0.5).to(tl.int32)
            
            packed |= pos_bit << (2 * i)
            packed |= neg_bit << (2 * i + 1)
        
        # Store as int16
        out_ptrs = out_ptr + offs_n[:, None] * stride_on + offs_k[None, :] * stride_ok
        out_mask = n_mask[:, None] & k_mask[None, :]
        tl.store(out_ptrs, packed.to(tl.int16), mask=out_mask)

    @triton.jit
    def unpack_ternary_from_int16_kernel(
        packed_ptr,  # [N, K_packed] int16 input
        out_ptr,     # [N, K] float output
        N, K, K_packed,
        stride_pn, stride_pk,
        stride_on, stride_ok,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Unpack int16 packed ternary weights back to float."""
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k_packed = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        
        n_mask = offs_n < N
        k_mask = offs_k_packed < K_packed
        
        # Load packed values
        packed_ptrs = packed_ptr + offs_n[:, None] * stride_pn + offs_k_packed[None, :] * stride_pk
        packed_mask = n_mask[:, None] & k_mask[None, :]
        packed = tl.load(packed_ptrs, mask=packed_mask, other=0).to(tl.int32)
        
        # Unpack 8 values
        for i in range(8):
            out_col = offs_k_packed * 8 + i
            out_mask = n_mask[:, None] & (out_col[None, :] < K)
            
            pos_bit = (packed >> (2 * i)) & 1
            neg_bit = (packed >> (2 * i + 1)) & 1
            
            # Decode: 01 -> +1, 10 -> -1, 00 -> 0
            value = pos_bit.to(tl.float32) - neg_bit.to(tl.float32)
            
            out_ptrs = out_ptr + offs_n[:, None] * stride_on + out_col[None, :] * stride_ok
            tl.store(out_ptrs, value, mask=out_mask)


def pack_bool_fused(x: torch.Tensor) -> torch.Tensor:
    """
    Fused kernel to pack float/bool tensor into int32.
    Input: [..., N]
    Output: [..., N // 32]
    """
    if not TRITON_AVAILABLE or not x.is_cuda:
        # CPU fallback
        x_flat = x.flatten()
        n = x_flat.numel()
        n_packed = (n + 31) // 32
        x_pad = torch.zeros(n_packed * 32, device=x.device, dtype=x.dtype)
        x_pad[:n] = x_flat
        x_pad = x_pad.view(n_packed, 32)
        w = (1 << torch.arange(32, device=x.device, dtype=torch.int64))
        packed = ((x_pad > 0.5).to(torch.int64) * w).sum(dim=-1).to(torch.int32)
        return packed
    
    x_flat = x.flatten()
    n = x_flat.numel()
    n_packed = (n + 31) // 32
    
    out = torch.empty(n_packed, dtype=torch.int32, device=x.device)
    
    grid = lambda meta: (triton.cdiv(n_packed, meta['BLOCK_SIZE']),)
    
    pack_bool_to_int32_kernel[grid](
        x_flat, out, n,
        BLOCK_SIZE=256
    )
    
    return out


def pack_ternary_int16_fused(w: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    GPU-accelerated packing of ternary weights into int16.
    
    Args:
        w: [N, K] tensor with values in {-1, 0, 1}
        
    Returns:
        packed: [N, K_packed] int16 tensor
        original_shape: (N, K) tuple
    """
    N, K = w.shape
    K_packed = (K + 7) // 8
    
    if not TRITON_AVAILABLE or not w.is_cuda:
        # CPU fallback - use existing implementation
        from .kernels_bitplane16 import pack_ternary_int16
        return pack_ternary_int16(w)
    
    w = w.contiguous()
    out = torch.empty((N, K_packed), dtype=torch.int16, device=w.device)
    
    BLOCK_N = 32
    BLOCK_K = 32
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K_packed, BLOCK_K))
    
    pack_ternary_to_int16_kernel[grid](
        w, out,
        N, K, K_packed,
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    
    return out, (N, K)


def unpack_ternary_int16_fused(packed: torch.Tensor, original_shape: Tuple[int, int]) -> torch.Tensor:
    """
    GPU-accelerated unpacking of int16 to ternary weights.
    
    Args:
        packed: [N, K_packed] int16 tensor
        original_shape: (N, K) tuple
        
    Returns:
        w: [N, K] float tensor with values in {-1, 0, 1}
    """
    N, K = original_shape
    K_packed = packed.shape[1]
    
    if not TRITON_AVAILABLE or not packed.is_cuda:
        # CPU fallback
        from .kernels_bitplane16 import unpack_ternary_int16
        return unpack_ternary_int16(packed, original_shape)
    
    out = torch.empty((N, K), dtype=torch.float32, device=packed.device)
    
    BLOCK_N = 32
    BLOCK_K = 32
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K_packed, BLOCK_K))
    
    unpack_ternary_from_int16_kernel[grid](
        packed.contiguous(), out,
        N, K, K_packed,
        packed.stride(0), packed.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    
    return out


# =============================================================================
# BENCHMARKING
# =============================================================================

def benchmark_packing(N=512, K=1024):
    """Benchmark GPU vs CPU packing."""
    import time
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.device('cuda')
    w = torch.randint(-1, 2, (N, K), device=device).float()
    
    # Warmup
    for _ in range(5):
        pack_ternary_int16_fused(w)
    torch.cuda.synchronize()
    
    print(f"\nTernary Packing Benchmark (N={N}, K={K})")
    print("=" * 50)
    
    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        pack_ternary_int16_fused(w)
    torch.cuda.synchronize()
    pack_time = (time.perf_counter() - t0) / 100 * 1000
    
    print(f"Pack time:   {pack_time:.3f} ms")
    print(f"Memory reduction: 16x (float32 -> 2 bits)")


if __name__ == "__main__":
    benchmark_packing()
