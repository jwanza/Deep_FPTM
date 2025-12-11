import torch
import triton
import triton.language as tl

@triton.jit
def bit_to_int8_tensor_core_matmul_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K_packed,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_PACKED: tl.constexpr,
):
    """
    Computes Y = PopCount(X & W.T) using Tensor Cores.
    
    Strategy:
    1. Load packed int32 blocks.
    2. Unpack to bits (0 or 1).
    3. Use tl.dot() which maps to MMA (Tensor Core).
       Matmul(A, B) = Sum(A*B). Since A,B in {0,1}, A*B = A&B.
       So Sum(A*B) = PopCount(A&B).
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Accumulator in float32 for safety (compatible with any dot output)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Iterate over K_packed
    for k in range(0, K_packed, BLOCK_K_PACKED):
        offs_k = k + tl.arange(0, BLOCK_K_PACKED)
        k_mask = offs_k < K_packed
        
        # Load Packed Data [BLOCK_M, BLOCK_K_PACKED]
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm) + (offs_k[None, :] * stride_xk)
        x_packed = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0)
        
        # W: [BLOCK_N, BLOCK_K_PACKED]
        w_ptrs = w_ptr + (offs_n[:, None] * stride_wn) + (offs_k[None, :] * stride_wk)
        w_packed = tl.load(w_ptrs, mask=(offs_n[:, None] < N) & k_mask[None, :], other=0)
        
        # Unpack bits loop
        for b in range(32):
            # Extract bit b from all packed integers
            # Convert to float16 to ensure compatibility with all Tensor Cores
            x_bit = ((x_packed >> b) & 1).to(tl.float16) # [BLOCK_M, BLOCK_K_PACKED]
            w_bit = ((w_packed >> b) & 1).to(tl.float16) # [BLOCK_N, BLOCK_K_PACKED]
            
            # Dot Product: [BLOCK_M, K] @ [K, BLOCK_N]
            # K = BLOCK_K_PACKED (must be >= 16)
            acc += tl.dot(x_bit, tl.trans(w_bit))
            
    out_ptrs = out_ptr + (offs_m[:, None] * stride_om) + (offs_n[None, :] * stride_on)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.int32), mask=mask)

def bit_int8_matmul(x_packed: torch.Tensor, w_packed: torch.Tensor) -> torch.Tensor:
    M, K_packed = x_packed.shape
    N, _ = w_packed.shape
    out = torch.empty((M, N), device=x_packed.device, dtype=torch.int32)
    
    # Must be >= 16 for float16 MMA
    BLOCK_K_PACKED = 32 
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    
    bit_to_int8_tensor_core_matmul_kernel[grid](
        x_packed, w_packed, out,
        M, N, K_packed,
        x_packed.stride(0), x_packed.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_K_PACKED=BLOCK_K_PACKED,
        BLOCK_M=64, BLOCK_N=64 
    )
    return out
