import torch
import triton
import triton.language as tl

@triton.jit
def block_popcount_matmul_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K_packed,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    
    # Block-wise iteration
    for k in range(0, K_packed, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K_packed
        
        # Load X and W
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm) + (offs_k[None, :] * stride_xk)
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0)
        
        w_ptrs = w_ptr + (offs_n[:, None] * stride_wn) + (offs_k[None, :] * stride_wk)
        w = tl.load(w_ptrs, mask=(offs_n[:, None] < N) & k_mask[None, :], other=0)
        
        # Broadcasting for POPC (memory intensive but simple)
        # To optimize, we can loop inside the block to reduce register pressure
        # But for now, let's trust the compiler
        x_broad = x[:, None, :]
        w_broad = w[None, :, :]
        
        bitwise = x_broad & w_broad
        
        # PopCount
        # Check for tl.popc support
        # We can implement a SWAR popcount here for safety
        v = bitwise
        v = v - ((v >> 1) & 0x55555555)
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
        v = (v + (v >> 4)) & 0x0F0F0F0F
        v = (v * 0x01010101) >> 24
        
        counts = v
        block_sum = tl.sum(counts, axis=2)
        acc += block_sum
        
    out_ptrs = out_ptr + (offs_m[:, None] * stride_om) + (offs_n[None, :] * stride_on)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=mask)

def block_popcount_matmul(x_packed: torch.Tensor, w_packed: torch.Tensor) -> torch.Tensor:
    M, K_packed = x_packed.shape
    N, _ = w_packed.shape
    out = torch.empty((M, N), device=x_packed.device, dtype=torch.int32)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    
    block_popcount_matmul_kernel[grid](
        x_packed, w_packed, out,
        M, N, K_packed,
        x_packed.stride(0), x_packed.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
    )
    return out





