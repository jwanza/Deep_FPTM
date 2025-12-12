"""
Activation Packing Kernels for Clause Synchronization.

This module provides kernels for packing clause activations into compact
binary format for efficient clause synchronization operations.

Benefits:
- 32x memory reduction (float32 -> 1 bit per activation)
- Faster bitwise operations for sync
- Reduced memory bandwidth
"""
import torch
from typing import Tuple, Optional

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# =============================================================================
# REFERENCE IMPLEMENTATIONS
# =============================================================================

def pack_activations_reference(
    activations: torch.Tensor,
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Pack clause activations into uint32 for fast sync.
    
    Args:
        activations: [B, C] clause activations
        threshold: Binarization threshold
        
    Returns:
        packed: [B, C_packed] uint32 where C_packed = ceil(C / 32)
    """
    B, C = activations.shape
    C_packed = (C + 31) // 32
    C_pad = C_packed * 32
    
    # Pad if necessary
    if C_pad != C:
        act_pad = torch.zeros((B, C_pad), device=activations.device, dtype=activations.dtype)
        act_pad[:, :C] = activations
        activations = act_pad
    
    # Binarize
    binary = (activations > threshold).to(torch.int64)
    
    # Reshape and pack
    binary = binary.view(B, C_packed, 32)
    w = (1 << torch.arange(32, device=activations.device, dtype=torch.int64))
    packed = (binary * w).sum(dim=-1).to(torch.int32)
    
    return packed


def bitwise_sync_reference(
    packed_left: torch.Tensor,
    packed_right: torch.Tensor,
) -> torch.Tensor:
    """
    Compute clause synchronization using bitwise operations.
    
    Sync = popcount(left AND right)
    
    Args:
        packed_left: [B, C_packed] packed left clause activations
        packed_right: [B, C_packed] packed right clause activations
        
    Returns:
        sync: [B, C_packed] sync counts
    """
    # Bitwise AND
    anded = packed_left & packed_right
    
    # Popcount using SWAR algorithm
    x = anded.to(torch.int64)
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    x = (x * 0x0101010101010101) >> 56
    
    return x.to(torch.int32)


# =============================================================================
# TRITON KERNELS
# =============================================================================

if TRITON_AVAILABLE:
    @triton.jit
    def pack_activations_kernel(
        act_ptr, out_ptr,
        B, C, C_packed,
        stride_ab, stride_ac,
        stride_ob, stride_oc,
        threshold,
        BLOCK_B: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """Pack activations into uint32."""
        pid_b = tl.program_id(0)
        pid_c = tl.program_id(1)
        
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_c_packed = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        
        b_mask = offs_b < B
        c_mask = offs_c_packed < C_packed
        
        # Pack 32 activations into each int32
        packed = tl.zeros((BLOCK_B, BLOCK_C), dtype=tl.int32)
        
        for i in range(32):
            c_idx = offs_c_packed * 32 + i
            act_ptrs = act_ptr + offs_b[:, None] * stride_ab + c_idx[None, :] * stride_ac
            act_mask = b_mask[:, None] & (c_idx[None, :] < C)
            act = tl.load(act_ptrs, mask=act_mask, other=0.0)
            
            bit = (act > threshold).to(tl.int32)
            packed |= bit << i
        
        out_ptrs = out_ptr + offs_b[:, None] * stride_ob + offs_c_packed[None, :] * stride_oc
        out_mask = b_mask[:, None] & c_mask[None, :]
        tl.store(out_ptrs, packed, mask=out_mask)

    @triton.jit
    def bitwise_sync_kernel(
        left_ptr, right_ptr, out_ptr,
        B, C_packed,
        stride_b, stride_c,
        BLOCK_B: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """Compute clause sync using bitwise AND + popcount."""
        pid_b = tl.program_id(0)
        pid_c = tl.program_id(1)
        
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        
        b_mask = offs_b < B
        c_mask = offs_c < C_packed
        mask = b_mask[:, None] & c_mask[None, :]
        
        left_ptrs = left_ptr + offs_b[:, None] * stride_b + offs_c[None, :] * stride_c
        right_ptrs = right_ptr + offs_b[:, None] * stride_b + offs_c[None, :] * stride_c
        
        left = tl.load(left_ptrs, mask=mask, other=0)
        right = tl.load(right_ptrs, mask=mask, other=0)
        
        # Bitwise AND
        anded = left & right
        
        # SWAR popcount for 32-bit
        x = anded.to(tl.int64)
        x = x - ((x >> 1) & 0x55555555)
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
        x = (x + (x >> 4)) & 0x0F0F0F0F
        x = (x * 0x01010101) >> 24
        
        out_ptrs = out_ptr + offs_b[:, None] * stride_b + offs_c[None, :] * stride_c
        tl.store(out_ptrs, x.to(tl.int32), mask=mask)


# =============================================================================
# HIGH-LEVEL INTERFACE
# =============================================================================

def pack_clause_activations(
    activations: torch.Tensor,
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Pack boolean clause activations into uint32 for fast sync.
    
    Args:
        activations: [B, C] clause activations
        threshold: Binarization threshold
        
    Returns:
        packed: [B, C_packed] int32 where C_packed = ceil(C / 32)
    """
    B, C = activations.shape
    C_packed = (C + 31) // 32
    
    if not TRITON_AVAILABLE or not activations.is_cuda:
        return pack_activations_reference(activations, threshold)
    
    out = torch.empty((B, C_packed), dtype=torch.int32, device=activations.device)
    
    BLOCK_B = min(32, B)
    BLOCK_C = min(32, C_packed)
    grid = (triton.cdiv(B, BLOCK_B), triton.cdiv(C_packed, BLOCK_C))
    
    pack_activations_kernel[grid](
        activations.contiguous(), out,
        B, C, C_packed,
        activations.stride(0), activations.stride(1),
        out.stride(0), out.stride(1),
        threshold,
        BLOCK_B=BLOCK_B, BLOCK_C=BLOCK_C,
    )
    
    return out


def bitwise_clause_sync(
    packed_left: torch.Tensor,
    packed_right: torch.Tensor,
) -> torch.Tensor:
    """
    Compute clause synchronization using bitwise operations.
    
    Sync = popcount(left AND right)
    
    Args:
        packed_left: [B, C_packed] packed left clause activations
        packed_right: [B, C_packed] packed right clause activations
        
    Returns:
        sync: [B, C_packed] sync counts
    """
    B, C_packed = packed_left.shape
    
    if not TRITON_AVAILABLE or not packed_left.is_cuda:
        return bitwise_sync_reference(packed_left, packed_right)
    
    out = torch.empty_like(packed_left)
    
    BLOCK_B = min(32, B)
    BLOCK_C = min(32, C_packed)
    grid = (triton.cdiv(B, BLOCK_B), triton.cdiv(C_packed, BLOCK_C))
    
    bitwise_sync_kernel[grid](
        packed_left.contiguous(), packed_right.contiguous(), out,
        B, C_packed,
        packed_left.stride(0), packed_left.stride(1),
        BLOCK_B=BLOCK_B, BLOCK_C=BLOCK_C,
    )
    
    return out


# =============================================================================
# BENCHMARKING
# =============================================================================

def benchmark_activation_packing(B=128, C=256):
    """Benchmark activation packing."""
    import time
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.device('cuda')
    activations = torch.rand(B, C, device=device)
    
    # Warmup
    for _ in range(5):
        pack_clause_activations(activations)
    torch.cuda.synchronize()
    
    print(f"\nActivation Packing Benchmark (B={B}, C={C})")
    print("=" * 50)
    
    # Benchmark pack
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        pack_clause_activations(activations)
    torch.cuda.synchronize()
    pack_time = (time.perf_counter() - t0) / 100 * 1000
    
    print(f"Pack time:   {pack_time:.3f} ms")
    
    # Benchmark sync
    packed_left = pack_clause_activations(activations)
    packed_right = pack_clause_activations(activations)
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        bitwise_clause_sync(packed_left, packed_right)
    torch.cuda.synchronize()
    sync_time = (time.perf_counter() - t0) / 100 * 1000
    
    print(f"Sync time:   {sync_time:.3f} ms")
    print(f"Memory reduction: 32x (float32 -> 1 bit)")


if __name__ == "__main__":
    benchmark_activation_packing()





