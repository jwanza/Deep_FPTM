"""
Comprehensive Benchmark for all STCM Variants.

Measures:
1. Forward pass latency (inference speed)
2. Backward pass latency (training speed)
3. Memory usage
4. Accuracy on synthetic data
5. Throughput (samples/second)
"""

import torch
import torch.nn.functional as F
import time
import gc
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    name: str
    forward_ms: float
    backward_ms: float
    memory_mb: float
    accuracy: float
    throughput: float
    extra_info: Dict[str, Any]


def get_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def reset_memory():
    """Reset memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def benchmark_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    n_warmup: int = 20,
    n_iters: int = 100,
    name: str = "Model",
) -> BenchmarkResult:
    """Benchmark a single model."""
    device = x.device
    
    # Move model to device
    model = model.to(device)
    
    # Warmup
    model.eval()
    for _ in range(n_warmup):
        with torch.no_grad():
            out = model(x)
            if isinstance(out, tuple):
                _ = out[0]
    torch.cuda.synchronize()
    
    # Forward pass benchmark
    reset_memory()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            out = model(x)
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out
    torch.cuda.synchronize()
    forward_ms = (time.perf_counter() - t0) / n_iters * 1000
    forward_memory = get_memory_usage()
    
    # Accuracy
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        accuracy = (preds == y).float().mean().item()
    
    # Backward pass benchmark
    model.train()
    reset_memory()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        model.zero_grad()
        out = model(x)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
        loss = F.cross_entropy(logits, y)
        loss.backward()
    torch.cuda.synchronize()
    backward_ms = (time.perf_counter() - t0) / n_iters * 1000
    backward_memory = get_memory_usage()
    
    # Throughput (samples/second for inference)
    throughput = (x.shape[0] * 1000) / forward_ms
    
    return BenchmarkResult(
        name=name,
        forward_ms=forward_ms,
        backward_ms=backward_ms,
        memory_mb=max(forward_memory, backward_memory),
        accuracy=accuracy,
        throughput=throughput,
        extra_info={},
    )


def run_comprehensive_benchmark(
    batch_size: int = 256,
    n_features: int = 784,
    n_clauses: int = 512,
    n_classes: int = 10,
) -> List[BenchmarkResult]:
    """Run comprehensive benchmark on all STCM variants."""
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return []
    
    device = torch.device("cuda")
    torch.manual_seed(42)
    
    # Create synthetic data with structure
    centers = torch.randn(n_classes, n_features, device=device) * 3
    y = torch.randint(0, n_classes, (batch_size,), device=device)
    x = centers[y] + torch.randn(batch_size, n_features, device=device) * 0.5
    
    results = []
    
    # Define models to benchmark
    models_config = []
    
    # 1. Baseline STCM
    from fptm_ste.tm import FuzzyPatternTM_STCM
    models_config.append(("STCM (baseline)", lambda: FuzzyPatternTM_STCM(
        n_features=n_features, n_clauses=n_clauses, n_classes=n_classes
    )))
    
    # 2. OptimizedSTCM
    from fptm_ste.tm_optimized import OptimizedSTCM
    models_config.append(("OptimizedSTCM", lambda: OptimizedSTCM(
        n_features=n_features, n_clauses=n_clauses, n_classes=n_classes
    )))
    
    # 3. CompiledSTCM
    from fptm_ste.compiled_stcm import CompiledSTCM
    models_config.append(("CompiledSTCM", lambda: CompiledSTCM(
        n_features=n_features, n_clauses=n_clauses, n_classes=n_classes,
        compile_mode="reduce-overhead"
    )))
    
    # 4. SparseSTCM
    from fptm_ste.sparse_stcm import SparseSTCM
    models_config.append(("SparseSTCM (k=64)", lambda: SparseSTCM(
        n_features=n_features, n_clauses=n_clauses, n_classes=n_classes, k=64
    )))
    models_config.append(("SparseSTCM (k=128)", lambda: SparseSTCM(
        n_features=n_features, n_clauses=n_clauses, n_classes=n_classes, k=128
    )))
    
    # 5. HierarchicalSTCM
    from fptm_ste.hierarchical_stcm import HierarchicalSTCM
    models_config.append(("HierarchicalSTCM", lambda: HierarchicalSTCM(
        n_features=n_features, n_classes=n_classes, depth=3,
        base_clauses=32, branch_factor=4, confidence_threshold=0.8
    )))
    
    # 6. UltimateSTCM
    from fptm_ste.ultimate_stcm import UltimateSTCM
    models_config.append(("UltimateSTCM", lambda: UltimateSTCM(
        n_features=n_features, n_classes=n_classes, depth=3,
        base_clauses=32, k_factor=0.25, use_compile=False
    )))
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE STCM BENCHMARK")
    print(f"{'='*80}")
    print(f"Config: batch_size={batch_size}, n_features={n_features}, ")
    print(f"        n_clauses={n_clauses}, n_classes={n_classes}")
    print(f"{'='*80}\n")
    
    for name, model_fn in models_config:
        try:
            print(f"Benchmarking {name}...", end=" ", flush=True)
            model = model_fn()
            result = benchmark_model(model, x, y, name=name)
            results.append(result)
            print(f"Done. Forward: {result.forward_ms:.2f}ms, Backward: {result.backward_ms:.2f}ms")
            
            # Clear memory
            del model
            reset_memory()
            
        except Exception as e:
            print(f"FAILED: {e}")
    
    return results


def print_results_table(results: List[BenchmarkResult], baseline_name: str = "STCM (baseline)"):
    """Print results as a formatted table."""
    
    # Find baseline for speedup calculation
    baseline_forward = 1.0
    baseline_backward = 1.0
    for r in results:
        if r.name == baseline_name:
            baseline_forward = r.forward_ms
            baseline_backward = r.backward_ms
            break
    
    print(f"\n{'='*100}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*100}")
    print(f"{'Model':<25} {'Fwd (ms)':<12} {'Bwd (ms)':<12} {'Memory':<12} {'Accuracy':<10} {'Throughput':<15} {'Speedup':<10}")
    print(f"{'-'*100}")
    
    for r in results:
        fwd_speedup = baseline_forward / r.forward_ms
        bwd_speedup = baseline_backward / r.backward_ms
        print(f"{r.name:<25} {r.forward_ms:<12.2f} {r.backward_ms:<12.2f} {r.memory_mb:<12.1f} {r.accuracy:<10.4f} {r.throughput:<15.0f} {fwd_speedup:<10.2f}x")
    
    print(f"{'='*100}\n")
    
    # Print analysis
    print("ANALYSIS:")
    print("-" * 40)
    
    best_forward = min(results, key=lambda r: r.forward_ms)
    best_backward = min(results, key=lambda r: r.backward_ms)
    best_memory = min(results, key=lambda r: r.memory_mb)
    best_accuracy = max(results, key=lambda r: r.accuracy)
    
    print(f"Fastest forward pass: {best_forward.name} ({best_forward.forward_ms:.2f}ms)")
    print(f"Fastest backward pass: {best_backward.name} ({best_backward.backward_ms:.2f}ms)")
    print(f"Lowest memory: {best_memory.name} ({best_memory.memory_mb:.1f}MB)")
    print(f"Best accuracy: {best_accuracy.name} ({best_accuracy.accuracy:.4f})")


def run_evolutionary_benchmark():
    """Benchmark evolutionary training vs gradient descent."""
    from fptm_ste.evolutionary_stcm import EvolutionaryMaskOptimizer
    from fptm_ste.tm_optimized import OptimizedSTCM
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.device("cuda")
    torch.manual_seed(42)
    
    B, F_dim, C, K = 128, 256, 128, 10
    
    # Create data
    x = torch.rand(B, F_dim, device=device)
    y = torch.randint(0, K, (B,), device=device)
    
    # Gradient-based
    grad_model = OptimizedSTCM(n_features=F_dim, n_clauses=C, n_classes=K).to(device)
    grad_opt = torch.optim.AdamW(grad_model.parameters(), lr=1e-3)
    
    # ES-based
    es_model = OptimizedSTCM(n_features=F_dim, n_clauses=C, n_classes=K).to(device)
    es_opt = EvolutionaryMaskOptimizer(es_model, population_size=20, sigma=0.1)
    
    # Warmup
    for _ in range(10):
        grad_model.train()
        grad_opt.zero_grad()
        loss = F.cross_entropy(grad_model(x)[0], y)
        loss.backward()
        grad_opt.step()
        
        es_opt.step(x, y)
    torch.cuda.synchronize()
    
    n_iters = 50
    
    # Gradient timing
    reset_memory()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        grad_model.train()
        grad_opt.zero_grad()
        loss = F.cross_entropy(grad_model(x)[0], y)
        loss.backward()
        grad_opt.step()
    torch.cuda.synchronize()
    grad_time = (time.perf_counter() - t0) / n_iters * 1000
    grad_mem = get_memory_usage()
    
    # ES timing
    reset_memory()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        es_opt.step(x, y)
    torch.cuda.synchronize()
    es_time = (time.perf_counter() - t0) / n_iters * 1000
    es_mem = get_memory_usage()
    
    print(f"\n{'='*60}")
    print(f"EVOLUTIONARY VS GRADIENT TRAINING")
    print(f"{'='*60}")
    print(f"Gradient descent: {grad_time:.2f} ms/step, {grad_mem:.1f} MB")
    print(f"Evolution (ES):   {es_time:.2f} ms/step, {es_mem:.1f} MB")
    print(f"ES is {grad_time/es_time:.2f}x {'faster' if grad_time > es_time else 'slower'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Run main benchmark
    results = run_comprehensive_benchmark(
        batch_size=256,
        n_features=784,
        n_clauses=512,
        n_classes=10,
    )
    
    if results:
        print_results_table(results)
    
    # Run evolutionary benchmark
    run_evolutionary_benchmark()






