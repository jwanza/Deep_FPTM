import torch
import triton
import triton.language as tl

@triton.jit
def debug_sum_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K_packed,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_PACKED: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid % (M // BLOCK_M)
    pid_n = pid // (M // BLOCK_M) # Simple grid assumption
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    
    for k in range(0, K_packed, BLOCK_K_PACKED):
        offs_k = k + tl.arange(0, BLOCK_K_PACKED)
        k_mask = offs_k < K_packed
        
        # Load X
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm) + (offs_k[None, :] * stride_xk)
        x_packed = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0)
        
        # Just check if we loaded anything non-zero
        # acc += tl.sum(x_packed, axis=1)[:, None] # Sum rows
        
        # Check extraction
        for b in range(1): # Check bit 0 only
            x_bit = ((x_packed >> b) & 1).to(tl.int8)
            # Sum x_bit
            # Broadcast to BLOCK_N
            acc += x_bit[:, 0:1] # Just take first column of K? No.
            # This is hard to debug with matrix output.
            pass
    
    # Store 1 if X loaded something
    # tl.store(out_ptr..., acc)
    pass

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
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    
    for k in range(0, K_packed, BLOCK_K_PACKED):
        offs_k = k + tl.arange(0, BLOCK_K_PACKED)
        k_mask = offs_k < K_packed
        
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm) + (offs_k[None, :] * stride_xk)
        x_packed = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0)
        
        w_ptrs = w_ptr + (offs_n[:, None] * stride_wn) + (offs_k[None, :] * stride_wk)
        w_packed = tl.load(w_ptrs, mask=(offs_n[:, None] < N) & k_mask[None, :], other=0)
        
        for b in range(32):
            x_bit = ((x_packed >> b) & 1).to(tl.int8)
            w_bit = ((w_packed >> b) & 1).to(tl.int8)
            acc += tl.dot(x_bit, tl.trans(w_bit), out_dtype=tl.int32)
            
    out_ptrs = out_ptr + (offs_m[:, None] * stride_om) + (offs_n[None, :] * stride_on)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=mask)





