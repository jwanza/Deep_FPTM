"""
Optimized Triton kernels for STCM and L-SSM.

Key features:
1. Packed Ternary Matmul (Int32): Packs 16 weights per int32 for 16x memory compression.
2. Arithmetic Unpacking: Branchless decoding of ternary weights.
3. PopCount Kernel: Efficient boolean matrix multiplication for L-SSM.
4. Fused Packing: Direct GPU packing of bools.
"""

import torch
import torch.nn.functional as F
from torch.autograd import Function
from typing import Tuple

try:
    import triton
    import triton.language as tl
    from .kernels_packing import pack_bool_fused
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def pack_ternary_int32(w: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Packs a float tensor of ternary weights {-1, 0, 1} into int32.
    """
    if w.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {w.dim()}D")
    
    N, K = w.shape
    original_shape = (N, K)
    
    # Pad K to multiple of 16
    K_padded = (K + 15) // 16 * 16
    if K_padded != K:
        w = F.pad(w, (0, K_padded - K))
    
    # Reshape to [N, K_packed, 16]
    K_packed = K_padded // 16
    w_reshaped = w.view(N, K_packed, 16)
    
    # Convert values to codes: -1 -> 2, 0 -> 0, 1 -> 1
    codes = torch.zeros_like(w_reshaped, dtype=torch.int32)
    codes[w_reshaped > 0.5] = 1
    codes[w_reshaped < -0.5] = 2
    
    # Pack 16 codes into 1 int32
    packed = torch.zeros((N, K_packed), dtype=torch.int32, device=w.device)
    
    for i in range(16):
        packed |= (codes[:, :, i] << (2 * i))
        
    return packed, original_shape


def unpack_ternary_int32(w_packed: torch.Tensor, original_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Unpacks int32 tensor back to float ternary.
    """
    N, K = original_shape
    device = w_packed.device
    
    w_packed_expanded = w_packed.unsqueeze(-1) # [N, K_packed, 1]
    shifts = torch.arange(0, 32, 2, device=device) # [16]
    
    codes = (w_packed_expanded >> shifts) & 3
    codes = codes.view(N, -1)
    codes = codes[:, :K]
    
    val = (codes & 1).float() - ((codes >> 1) & 1).float()
    
    return val


if TRITON_AVAILABLE:
    @triton.jit
    def swar_popc(v):
        v = v - ((v >> 1) & 0x55555555)
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
        v = (v + (v >> 4)) & 0x0F0F0F0F
        v = (v * 0x01010101) >> 24
        return v

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        ],
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def ternary_matmul_kernel_v2(
        x_ptr, w_packed_ptr, out_ptr,
        M, N, K, K_packed,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_om, stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Computes Y = X @ W.T where W is packed 16 weights/int32.
        """
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m
        
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm)
        w_ptrs = w_packed_ptr + (offs_n[None, :] * stride_wn)
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # Calculate packed block size at compile time
        BLOCK_K_PACKED: tl.constexpr = BLOCK_K // 16

        for k_idx in range(0, K, BLOCK_K):
            # 1. Load X [BLOCK_M, BLOCK_K]
            offs_k = k_idx + tl.arange(0, BLOCK_K)
            x_ptrs_k = x_ptrs + (offs_k[None, :] * stride_xk)
            x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            x = tl.load(x_ptrs_k, mask=x_mask, other=0.0)
            
            # 2. Load W_packed [BLOCK_N, BLOCK_K_PACKED]
            k_pack_start = k_idx // 16
            
            # Use the constexpr directly
            offs_k_packed = k_pack_start + tl.arange(0, BLOCK_K_PACKED)
            w_ptrs_k = w_ptrs + (offs_k_packed[None, :] * stride_wk)
            w_mask = (offs_n[:, None] < N) & (offs_k_packed[None, :] < K_packed)
            
            w_packed = tl.load(w_ptrs_k, mask=w_mask, other=0) 
            
            # Fallback: Just loop p in range BLOCK_K_PACKED and load W column-wise
            for p in range(BLOCK_K_PACKED):
                # Load one column of packed W: [BLOCK_N, 1]
                offs_p = k_pack_start + p
                w_col_ptr = w_ptrs + (offs_p * stride_wk)
                w_col = tl.load(w_col_ptr, mask=(offs_n[:, None] < N), other=0)
                
                # Unpack to 16 weights [BLOCK_N, 16]
                w_col_broad = w_col[:, None]
                shifts = tl.arange(0, 16) * 2
                shifts = shifts[None, :]
                
                codes = (w_col_broad >> shifts) & 3
                w_chunk = (codes & 1).to(tl.float32) - ((codes >> 1) & 1).to(tl.float32)
                
                # Load corresponding X chunk [BLOCK_M, 16]
                offs_k_chunk = (k_idx + p * 16) + tl.arange(0, 16)
                x_chunk_ptr = x_ptrs + (offs_k_chunk[None, :] * stride_xk)
                x_chunk_mask = (offs_m[:, None] < M) & (offs_k_chunk[None, :] < K)
                x_chunk = tl.load(x_chunk_ptr, mask=x_chunk_mask, other=0.0)
                
                acc += tl.dot(x_chunk, tl.trans(w_chunk))

        out_ptrs = out_ptr + (offs_m[:, None] * stride_om) + (offs_n[None, :] * stride_on)
        out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptrs, acc, mask=out_mask)


class TernaryLinearFunctionV2(Function):
    @staticmethod
    def forward(ctx, x, w_packed, original_shape):
        x = x.contiguous()
        w_packed = w_packed.contiguous()
        
        M, K = x.shape
        N, _ = original_shape
        _, K_packed = w_packed.shape
        
        out = torch.empty((M, N), device=x.device, dtype=torch.float32)
        
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        )
        
        ternary_matmul_kernel_v2[grid](
            x, w_packed, out,
            M, N, K, K_packed,
            x.stride(0), x.stride(1),
            w_packed.stride(0), w_packed.stride(1),
            out.stride(0), out.stride(1),
        )
        
        ctx.save_for_backward(x, w_packed)
        ctx.original_shape = original_shape
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, w_packed = ctx.saved_tensors
        original_shape = ctx.original_shape
        w = unpack_ternary_int32(w_packed, original_shape)
        grad_input = grad_output.matmul(w)
        return grad_input, None, None
        

def ternary_linear_v2(x: torch.Tensor, w_packed: torch.Tensor, original_shape: Tuple[int, int]) -> torch.Tensor:
    if TRITON_AVAILABLE and x.is_cuda and w_packed.is_cuda:
        return TernaryLinearFunctionV2.apply(x, w_packed, original_shape)
    else:
        return F.linear(x, unpack_ternary_int32(w_packed, original_shape))


# ============================================================================
# L-SSM Primitives (PopCount)
# ============================================================================

if TRITON_AVAILABLE:
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        ],
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def popcount_matmul_kernel(
        x_ptr, w_ptr, out_ptr,
        M, N, K_packed,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_om, stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Computes Y = PopCount(X & W.T)
        """
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m
        
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        
        for k in range(0, K_packed, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K_packed
            
            x_ptrs = x_ptr + (offs_m[:, None] * stride_xm) + (offs_k[None, :] * stride_xk)
            x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0)
            
            w_ptrs = w_ptr + (offs_n[:, None] * stride_wn) + (offs_k[None, :] * stride_wk)
            w = tl.load(w_ptrs, mask=(offs_n[:, None] < N) & k_mask[None, :], other=0)
            
            x_broad = x[:, None, :]
            w_broad = w[None, :, :]
            
            bitwise = x_broad & w_broad
            counts = swar_popc(bitwise)
            block_sum = tl.sum(counts, axis=2)
            acc += block_sum
            
        out_ptrs = out_ptr + (offs_m[:, None] * stride_om) + (offs_n[None, :] * stride_on)
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptrs, acc, mask=mask)

def popcount_matmul(x_packed: torch.Tensor, w_packed: torch.Tensor) -> torch.Tensor:
    if TRITON_AVAILABLE and x_packed.is_cuda:
        M, K_packed = x_packed.shape
        N, _ = w_packed.shape
        out = torch.empty((M, N), device=x_packed.device, dtype=torch.int32)
        
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        )
        
        popcount_matmul_kernel[grid](
            x_packed, w_packed, out,
            M, N, K_packed,
            x_packed.stride(0), x_packed.stride(1),
            w_packed.stride(0), w_packed.stride(1),
            out.stride(0), out.stride(1),
        )
        return out
    else:
        # Fallback
        M, K_packed = x_packed.shape
        N, _ = w_packed.shape
        out = torch.zeros((M, N), dtype=torch.int32, device=x_packed.device)
        
        chunk_size = 32
        for i in range(0, M, chunk_size):
            x_chunk = x_packed[i:i+chunk_size]
            res = (x_chunk.unsqueeze(1) & w_packed.unsqueeze(0))
            
            c = res
            c = c - ((c >> 1) & 0x55555555)
            c = (c & 0x33333333) + ((c >> 2) & 0x33333333)
            c = (c + (c >> 4)) & 0x0F0F0F0F
            c = (c * 0x01010101) >> 24
            
            out[i:i+chunk_size] = c.sum(dim=-1).int()
            
        return out
