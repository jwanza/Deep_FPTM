#!/usr/bin/env python3
"""
Comprehensive Triton Kernel Benchmarks.

This script benchmarks all Triton kernels implemented for the Tsetlin Machine
and reports speedups compared to PyTorch baselines.
"""
import torch
import time
from typing import Dict, List, Tuple
import sys


def robust_benchmark(fn, *args, warmup=10, iters=50, **kwargs):
    """Proper GPU benchmarking with warmup."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def benchmark_ste_ternary() -> Dict:
    """Benchmark STE ternary quantization."""
    from fptm_ste.kernels_fused import fused_ste_ternary, ste_ternary_reference
    
    device = 'cuda'
    results = {}
    
    for size in [(128, 256), (256, 512), (512, 1024)]:
        logits = torch.randn(*size, device=device)
        
        ref_time = robust_benchmark(ste_ternary_reference, logits, 0.3, 0.5)
        fused_time = robust_benchmark(fused_ste_ternary, logits, 0.3, 0.5)
        
        results[f"STE Ternary {size}"] = {
            'ref_ms': ref_time,
            'fused_ms': fused_time,
            'speedup': ref_time / fused_time
        }
    
    return results


def benchmark_gumbel_softmax() -> Dict:
    """Benchmark Gumbel-Softmax."""
    from fptm_ste.kernels_fused import fused_gumbel_softmax, gumbel_softmax_reference
    
    device = 'cuda'
    results = {}
    
    for size in [(64, 128, 3), (128, 256, 3), (256, 512, 3)]:
        logits = torch.randn(*size, device=device)
        
        ref_time = robust_benchmark(gumbel_softmax_reference, logits, 1.0, True)
        fused_time = robust_benchmark(fused_gumbel_softmax, logits, 1.0, True)
        
        results[f"Gumbel-Softmax {size}"] = {
            'ref_ms': ref_time,
            'fused_ms': fused_time,
            'speedup': ref_time / fused_time
        }
    
    return results


def benchmark_ternary_linear() -> Dict:
    """Benchmark ternary linear (tensor core matmul)."""
    try:
        from fptm_ste.kernels_bitplane16 import ternary_linear_tc, TRITON_AVAILABLE
        if not TRITON_AVAILABLE:
            return {'error': 'Triton not available'}
    except ImportError:
        return {'error': 'Import failed'}
    
    device = 'cuda'
    results = {}
    
    for M, N, K in [(128, 256, 512), (256, 512, 1024), (512, 1024, 2048)]:
        x = torch.randn(M, K, device=device)
        w = torch.randint(-1, 2, (N, K), device=device).float()
        
        ref_time = robust_benchmark(torch.nn.functional.linear, x, w)
        tc_time = robust_benchmark(ternary_linear_tc, x, w)
        
        results[f"Ternary Linear ({M}x{N}x{K})"] = {
            'ref_ms': ref_time,
            'fused_ms': tc_time,
            'speedup': ref_time / tc_time
        }
    
    return results


def benchmark_tnorm_operators() -> Dict:
    """Benchmark T-norm operators."""
    from fptm_ste.kernels_tnorm import FusedTNorm
    
    device = 'cuda'
    results = {}
    N = 1_000_000
    
    a = torch.rand(N, device=device)
    b = torch.rand(N, device=device)
    
    for op_name in ['lukasiewicz', 'godel', 'hamacher', 'product']:
        ref_fn = FusedTNorm._REFERENCE_FUNCS[op_name]
        
        ref_time = robust_benchmark(ref_fn, a, b)
        fused_time = robust_benchmark(FusedTNorm.apply, a, b, op_name)
        
        results[f"T-Norm {op_name}"] = {
            'ref_ms': ref_time,
            'fused_ms': fused_time,
            'speedup': ref_time / fused_time
        }
    
    return results


def benchmark_adaptive_mixer() -> Dict:
    """Benchmark adaptive operator mixer."""
    from fptm_ste.kernels_adaptive_mixer import fused_adaptive_mixer, adaptive_mixer_reference
    
    device = 'cuda'
    results = {}
    
    for N in [100_000, 500_000, 1_000_000]:
        a = torch.rand(N, device=device)
        b = torch.rand(N, device=device)
        weights = torch.softmax(torch.randn(4, device=device), dim=0)
        
        ref_time = robust_benchmark(adaptive_mixer_reference, (a, b), weights)
        fused_time = robust_benchmark(fused_adaptive_mixer, a, b, weights)
        
        results[f"Adaptive Mixer N={N}"] = {
            'ref_ms': ref_time,
            'fused_ms': fused_time,
            'speedup': ref_time / fused_time
        }
    
    return results


def benchmark_activation_packing() -> Dict:
    """Benchmark activation packing."""
    from fptm_ste.kernels_activation_pack import pack_clause_activations, pack_activations_reference
    
    device = 'cuda'
    results = {}
    
    for B, C in [(128, 256), (256, 512), (512, 1024)]:
        activations = torch.rand(B, C, device=device)
        
        ref_time = robust_benchmark(pack_activations_reference, activations)
        fused_time = robust_benchmark(pack_clause_activations, activations)
        
        results[f"Activation Pack ({B}x{C})"] = {
            'ref_ms': ref_time,
            'fused_ms': fused_time,
            'speedup': ref_time / fused_time
        }
    
    return results


def run_all_benchmarks():
    """Run all benchmarks and print report."""
    if not torch.cuda.is_available():
        print("CUDA not available. Cannot run benchmarks.")
        return
    
    print("=" * 80)
    print("TRITON KERNEL COMPREHENSIVE BENCHMARKS")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 80)
    
    all_results = {}
    
    # Run each benchmark category
    benchmarks = [
        ("STE Ternary", benchmark_ste_ternary),
        ("Gumbel-Softmax", benchmark_gumbel_softmax),
        ("Ternary Linear (Tensor Cores)", benchmark_ternary_linear),
        ("T-Norm Operators", benchmark_tnorm_operators),
        ("Adaptive Mixer", benchmark_adaptive_mixer),
        ("Activation Packing", benchmark_activation_packing),
    ]
    
    for name, bench_fn in benchmarks:
        print(f"\n{name}")
        print("-" * 60)
        try:
            results = bench_fn()
            for key, vals in results.items():
                if 'error' in vals:
                    print(f"  {key}: {vals['error']}")
                else:
                    print(f"  {key}: ref={vals['ref_ms']:.3f}ms, "
                          f"fused={vals['fused_ms']:.3f}ms, "
                          f"speedup={vals['speedup']:.2f}x")
                    all_results[key] = vals
        except Exception as e:
            print(f"  Error: {e}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if all_results:
        speedups = [v['speedup'] for v in all_results.values()]
        avg_speedup = sum(speedups) / len(speedups)
        max_speedup = max(speedups)
        min_speedup = min(speedups)
        
        print(f"Average speedup: {avg_speedup:.2f}x")
        print(f"Max speedup:     {max_speedup:.2f}x")
        print(f"Min speedup:     {min_speedup:.2f}x")
        
        # Categorize results
        fast = [k for k, v in all_results.items() if v['speedup'] >= 1.5]
        moderate = [k for k, v in all_results.items() if 1.0 <= v['speedup'] < 1.5]
        slow = [k for k, v in all_results.items() if v['speedup'] < 1.0]
        
        if fast:
            print(f"\nFast (>= 1.5x speedup): {len(fast)} kernels")
        if moderate:
            print(f"Moderate (1.0-1.5x speedup): {len(moderate)} kernels")
        if slow:
            print(f"Needs optimization (<1.0x): {len(slow)} kernels")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_all_benchmarks()



