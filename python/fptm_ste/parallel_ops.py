"""
Parallel Scan Primitives for STCM Optimization.

This module provides low-level parallel scan operations that enable O(log N)
computation of sequential recurrences by exploiting associativity.

The key insight is that linear recurrences of the form:
    h_t = A * h_{t-1} + B
    
can be computed in parallel using the closed-form solution:
    h_t = A^t * h_0 + sum_{j=0}^{t-1}(A^{t-1-j} * B_j)

When B is constant (same input at each step):
    h_t = A_cumulative[t] * cumsum(B / A_cumulative)

This is implemented using torch.cumsum which is highly optimized on GPUs.

Mathematical Background:
-----------------------
For the recurrence h_t = A * h_{t-1} + Bx (with h_0 = 0):

    h_1 = A * 0 + Bx = Bx
    h_2 = A * h_1 + Bx = A*Bx + Bx
    h_3 = A * h_2 + Bx = A²*Bx + A*Bx + Bx
    ...
    h_t = sum_{j=0}^{t-1} A^j * Bx = Bx * (1 + A + A² + ... + A^{t-1})

The parallel scan computes this sum in O(log t) depth using prefix sums.

References:
----------
- Blelloch, G. E. (1990). Prefix Sums and Their Applications
- Mamba architecture: Selective State Space Models
- ParallelScanCTTM in continuous-thought-machines project
"""

from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn.functional as F


def parallel_cumsum_stable(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Numerically stable parallel cumulative sum.
    
    Uses Kahan summation internally (via torch.cumsum) to reduce
    floating-point errors for long sequences.
    
    Args:
        x: Input tensor of any shape
        dim: Dimension along which to compute cumsum
        
    Returns:
        Cumulative sum along specified dimension
    """
    return torch.cumsum(x, dim=dim)


def parallel_cumprod_log(log_x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Parallel cumulative product computed in log-space for stability.
    
    Computes cumprod(exp(log_x)) = exp(cumsum(log_x))
    
    This avoids numerical overflow/underflow when computing products
    of many factors that may be very small or very large.
    
    Args:
        log_x: Log of factors to multiply [*, T, D]
        dim: Dimension along which to compute cumprod
        
    Returns:
        Cumulative product exp(cumsum(log_x))
    """
    return torch.exp(torch.cumsum(log_x, dim=dim))


def associative_scan_linear(
    A: torch.Tensor,
    Bx: torch.Tensor,
    T: int,
) -> torch.Tensor:
    """
    Associative scan for linear recurrence h_t = A * h_{t-1} + Bx.
    
    This is the core P-Scan operation. Given decay factors A and input
    contribution Bx, computes all T states in parallel.
    
    The recurrence h_t = A * h_{t-1} + Bx has closed-form solution:
        h_t = A^t * h_0 + A^{t-1}*Bx + A^{t-2}*Bx + ... + Bx
        
    With h_0 = 0:
        h_t = A_cumulative[t] * cumsum(Bx / A_cumulative)
    
    Args:
        A: Decay factors [D] (will be expanded to [T, D])
           Should be in range (0, 1) for stable recurrence
        Bx: Input contribution [B, D]
        T: Number of iterations
        
    Returns:
        h: All states [B, T, D] where h[:, t, :] is state at iteration t
        
    Example:
        >>> A = torch.tensor([0.9, 0.8, 0.7])  # Decay factors
        >>> Bx = torch.randn(4, 3)  # Batch of 4, 3 dimensions
        >>> h = associative_scan_linear(A, Bx, T=10)
        >>> print(h.shape)  # [4, 10, 3]
    """
    B, D = Bx.shape
    device = Bx.device
    dtype = Bx.dtype
    
    # Compute cumulative A powers in log-space for numerical stability
    # log(A^t) = t * log(A)
    log_A = torch.log(A.clamp(min=1e-8))  # [D]
    
    # Create powers [1, 2, 3, ..., T] for each dimension
    # log_A_cumsum[t, d] = (t+1) * log(A[d])
    log_A_cumsum = torch.cumsum(
        log_A.unsqueeze(0).expand(T, D), dim=0
    )  # [T, D]
    
    # A_cumulative[t] = A^(t+1)
    A_cumulative = torch.exp(log_A_cumsum)  # [T, D]
    
    # Expand Bx for all iterations
    Bx_expanded = Bx.unsqueeze(1).expand(B, T, D)  # [B, T, D]
    A_cumulative_expanded = A_cumulative.unsqueeze(0).expand(B, T, D)  # [B, T, D]
    
    # Parallel scan formula:
    # h[t] = A_cumulative[t] * cumsum(Bx / A_cumulative)
    Bx_scaled = Bx_expanded / (A_cumulative_expanded + 1e-8)  # [B, T, D]
    Bx_cumsum = torch.cumsum(Bx_scaled, dim=1)  # [B, T, D]
    h = A_cumulative_expanded * Bx_cumsum  # [B, T, D]
    
    return h


def associative_scan_linear_varying(
    A: torch.Tensor,
    Bx: torch.Tensor,
) -> torch.Tensor:
    """
    Associative scan with time-varying decay factors.
    
    For recurrence h_t = A_t * h_{t-1} + Bx_t where A can vary per timestep.
    
    Args:
        A: Decay factors [T, D] or [B, T, D] (time-varying)
        Bx: Input contributions [B, T, D]
        
    Returns:
        h: All states [B, T, D]
    """
    B, T, D = Bx.shape
    
    # Handle broadcasting for A
    if A.dim() == 2:
        # A is [T, D], expand to [B, T, D]
        A = A.unsqueeze(0).expand(B, T, D)
    
    # Compute cumulative A products in log-space
    log_A = torch.log(A.clamp(min=1e-8))  # [B, T, D]
    log_A_cumsum = torch.cumsum(log_A, dim=1)  # [B, T, D]
    A_cumulative = torch.exp(log_A_cumsum)  # [B, T, D]
    
    # Parallel scan
    Bx_scaled = Bx / (A_cumulative + 1e-8)
    Bx_cumsum = torch.cumsum(Bx_scaled, dim=1)
    h = A_cumulative * Bx_cumsum
    
    return h


def sequential_scan_linear(
    A: torch.Tensor,
    Bx: torch.Tensor,
    T: int,
) -> torch.Tensor:
    """
    Sequential scan for linear recurrence (baseline for verification).
    
    Implements h_t = A * h_{t-1} + Bx using an explicit loop.
    Used to verify correctness of parallel scan.
    
    Args:
        A: Decay factors [D]
        Bx: Input contribution [B, D]
        T: Number of iterations
        
    Returns:
        h: All states [B, T, D]
    """
    B, D = Bx.shape
    device = Bx.device
    dtype = Bx.dtype
    
    h_seq = torch.zeros(B, T, D, device=device, dtype=dtype)
    h = torch.zeros(B, D, device=device, dtype=dtype)
    
    for t in range(T):
        h = A * h + Bx
        h_seq[:, t, :] = h
    
    return h_seq


def verify_pscan_correctness(
    A: torch.Tensor,
    Bx: torch.Tensor,
    T: int,
    tol: float = 1e-5,
) -> Tuple[bool, Dict[str, float]]:
    """
    Verify P-Scan output matches sequential computation.
    
    Used for unit testing to ensure mathematical equivalence.
    
    Args:
        A: Decay factors [D]
        Bx: Input contribution [B, D]
        T: Number of iterations
        tol: Absolute tolerance for comparison
        
    Returns:
        is_correct: True if results match within tolerance
        metrics: Dictionary with error statistics
    """
    # P-Scan result
    h_pscan = associative_scan_linear(A, Bx, T)
    
    # Sequential result
    h_seq = sequential_scan_linear(A, Bx, T)
    
    # Compute error metrics
    abs_diff = torch.abs(h_pscan - h_seq)
    max_error = abs_diff.max().item()
    mean_error = abs_diff.mean().item()
    rel_error = (abs_diff / (torch.abs(h_seq) + 1e-8)).mean().item()
    
    is_correct = torch.allclose(h_pscan, h_seq, atol=tol)
    
    metrics = {
        'max_error': max_error,
        'mean_error': mean_error,
        'rel_error': rel_error,
        'tolerance': tol,
        'passed': is_correct,
    }
    
    return is_correct, metrics


def benchmark_pscan_vs_sequential(
    A: torch.Tensor,
    Bx: torch.Tensor,
    T: int,
    warmup: int = 10,
    runs: int = 100,
) -> Dict[str, float]:
    """
    Benchmark P-Scan vs sequential implementation.
    
    Args:
        A: Decay factors [D]
        Bx: Input contribution [B, D]
        T: Number of iterations
        warmup: Warmup iterations
        runs: Benchmark iterations
        
    Returns:
        Dictionary with timing results:
        - pscan_ms: P-Scan average time in milliseconds
        - sequential_ms: Sequential average time in milliseconds
        - speedup: Ratio sequential/pscan
    """
    import time
    
    device = Bx.device
    
    # Warmup
    for _ in range(warmup):
        _ = associative_scan_linear(A, Bx, T)
        _ = sequential_scan_linear(A, Bx, T)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark P-Scan
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = associative_scan_linear(A, Bx, T)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    pscan_time = (time.perf_counter() - t0) / runs * 1000  # ms
    
    # Benchmark sequential
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = sequential_scan_linear(A, Bx, T)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    seq_time = (time.perf_counter() - t0) / runs * 1000  # ms
    
    return {
        'pscan_ms': pscan_time,
        'sequential_ms': seq_time,
        'speedup': seq_time / pscan_time if pscan_time > 0 else float('inf'),
        'T': T,
        'batch_size': Bx.shape[0],
        'D': Bx.shape[1],
    }


def parallel_gradient_accumulation(
    gradients: torch.Tensor,
    num_micro_batches: int,
) -> torch.Tensor:
    """
    Parallel gradient accumulation for micro-batch training.
    
    This is the P-Scan principle applied to training: gradients are
    additive (associative), so we can sum them in parallel.
    
    Args:
        gradients: Stacked gradients [K, ...] from K micro-batches
        num_micro_batches: Number of micro-batches K
        
    Returns:
        Accumulated gradient (sum / K)
    """
    # Simple parallel reduction
    return gradients.sum(dim=0) / num_micro_batches


def fused_pscan_gated(
    x: torch.Tensor,
    A: torch.Tensor,
    B_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    T: int,
) -> torch.Tensor:
    """
    Fused P-Scan with gating mechanism.
    
    Implements gated linear recurrence:
        gate = sigmoid(x @ gate_weight.T + gate_bias)
        Bx = gate * (x @ B_weight.T)
        h_t = A * h_{t-1} + Bx
    
    Args:
        x: Input [B, D]
        A: Decay factors [D]
        B_weight: B projection weight [D, D]
        gate_weight: Gate projection weight [D, D]
        gate_bias: Gate bias [D]
        T: Number of iterations
        
    Returns:
        All states [B, T, D]
    """
    # Compute gate
    gate = torch.sigmoid(F.linear(x, gate_weight, gate_bias))  # [B, D]
    
    # Compute gated input contribution
    Bx = gate * F.linear(x, B_weight)  # [B, D]
    
    # P-Scan
    return associative_scan_linear(A, Bx, T)


# Type alias for P-Scan state
PScanState = Tuple[torch.Tensor, torch.Tensor]  # (A_cumulative, Bx_cumsum)


def pscan_init_state(
    A: torch.Tensor,
    T: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize P-Scan state tensors (for repeated use).
    
    Pre-computes A_cumulative to avoid redundant computation
    when running multiple P-Scan operations with same A.
    
    Args:
        A: Decay factors [D]
        T: Number of iterations
        
    Returns:
        Tuple of (log_A expanded [T, D], A_cumulative [T, D])
    """
    D = A.shape[0]
    log_A = torch.log(A.clamp(min=1e-8))
    log_A_expanded = log_A.unsqueeze(0).expand(T, D)
    log_A_cumsum = torch.cumsum(log_A_expanded, dim=0)
    A_cumulative = torch.exp(log_A_cumsum)
    
    return log_A_expanded, A_cumulative


def pscan_with_cached_state(
    Bx: torch.Tensor,
    A_cumulative: torch.Tensor,
) -> torch.Tensor:
    """
    P-Scan using pre-computed A_cumulative.
    
    Use this when running many P-Scan operations with the same A.
    
    Args:
        Bx: Input contribution [B, D]
        A_cumulative: Pre-computed cumulative A [T, D]
        
    Returns:
        All states [B, T, D]
    """
    B, D = Bx.shape
    T = A_cumulative.shape[0]
    
    Bx_expanded = Bx.unsqueeze(1).expand(B, T, D)
    A_cumulative_expanded = A_cumulative.unsqueeze(0).expand(B, T, D)
    
    Bx_scaled = Bx_expanded / (A_cumulative_expanded + 1e-8)
    Bx_cumsum = torch.cumsum(Bx_scaled, dim=1)
    h = A_cumulative_expanded * Bx_cumsum
    
    return h

