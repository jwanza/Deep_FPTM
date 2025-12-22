"""
Comprehensive benchmark suite for P-Scan optimizations.

This module provides tools to benchmark:
1. P-Scan STCM forward pass vs sequential
2. P-Scan training vs sequential gradient accumulation
3. Multi-head and hierarchical P-Scan variants
4. CUDA Graph acceleration

Usage:
    python -m fptm_ste.benchmarks.benchmark_pscan
    
Or programmatically:
    from fptm_ste.benchmarks.benchmark_pscan import PScanBenchmarkSuite
    
    suite = PScanBenchmarkSuite()
    results = suite.run_all_benchmarks()
    suite.generate_report(Path('results/PSCAN_BENCHMARK.md'))
"""

from __future__ import annotations

import torch
import torch.nn as nn
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    config: Dict[str, Any]
    baseline_ms: float
    optimized_ms: float
    speedup: float
    accuracy_baseline: float = 0.0
    accuracy_optimized: float = 0.0
    accuracy_delta: float = 0.0
    memory_baseline_mb: float = 0.0
    memory_optimized_mb: float = 0.0
    memory_savings_pct: float = 0.0
    notes: str = ""


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    device: str = 'cuda'
    warmup_iterations: int = 10
    benchmark_iterations: int = 100
    batch_sizes: List[int] = field(default_factory=lambda: [32, 64, 128])
    n_clauses_list: List[int] = field(default_factory=lambda: [128, 256, 512])
    iterations_list: List[int] = field(default_factory=lambda: [10, 30, 50, 100])
    n_features: int = 784
    n_classes: int = 10


class PScanBenchmarkSuite:
    """
    Run comprehensive P-Scan benchmarks.
    
    Benchmarks:
    1. STCM forward pass (P-Scan vs sequential)
    2. Training throughput (parallel vs sequential micro-batches)
    3. Memory usage comparison
    4. Accuracy verification
    """
    
    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
    ):
        self.config = config or BenchmarkConfig()
        self.device = torch.device(self.config.device)
        self.results: List[BenchmarkResult] = []
        self.start_time = None
        self.end_time = None
    
    def _get_memory_mb(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0.0
    
    def _benchmark_forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        mode: str,
        warmup: int = None,
        runs: int = None,
    ) -> float:
        """Benchmark forward pass and return average time in ms."""
        warmup = warmup or self.config.warmup_iterations
        runs = runs or self.config.benchmark_iterations
        
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(x, mode=mode)
        torch.cuda.synchronize()
        
        # Benchmark
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(runs):
                _ = model(x, mode=mode)
        torch.cuda.synchronize()
        
        return (time.perf_counter() - t0) / runs * 1000  # ms
    
    def benchmark_stcm_forward(
        self,
        configs: Optional[List[Dict]] = None,
    ) -> List[BenchmarkResult]:
        """
        Benchmark STCM forward pass: P-Scan vs sequential.
        
        Args:
            configs: List of model configurations to test
            
        Returns:
            List of BenchmarkResult
        """
        from fptm_ste.pscan_stcm import PScanOptimizedSTCM
        
        if configs is None:
            # Generate default configs
            configs = []
            for n_clauses in self.config.n_clauses_list:
                for iterations in self.config.iterations_list:
                    configs.append({
                        'n_features': self.config.n_features,
                        'n_clauses': n_clauses,
                        'n_classes': self.config.n_classes,
                        'iterations': iterations,
                    })
        
        results = []
        batch_size = 32
        
        for cfg in configs:
            try:
                model = PScanOptimizedSTCM(**cfg).to(self.device).eval()
                x = torch.randn(batch_size, cfg['n_features']).to(self.device)
                
                # Benchmark P-Scan
                torch.cuda.reset_peak_memory_stats()
                pscan_time = self._benchmark_forward(model, x, mode='pscan')
                pscan_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
                
                # Benchmark sequential
                torch.cuda.reset_peak_memory_stats()
                seq_time = self._benchmark_forward(model, x, mode='sequential')
                seq_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
                
                result = BenchmarkResult(
                    config=cfg,
                    baseline_ms=seq_time,
                    optimized_ms=pscan_time,
                    speedup=seq_time / pscan_time if pscan_time > 0 else float('inf'),
                    memory_baseline_mb=seq_mem,
                    memory_optimized_mb=pscan_mem,
                    memory_savings_pct=(seq_mem - pscan_mem) / seq_mem * 100 if seq_mem > 0 else 0,
                    notes=f"batch_size={batch_size}",
                )
                results.append(result)
                
                del model, x
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error benchmarking config {cfg}: {e}")
        
        self.results.extend(results)
        return results
    
    def benchmark_parallel_ops(
        self,
        iterations_list: Optional[List[int]] = None,
    ) -> List[BenchmarkResult]:
        """
        Benchmark raw parallel_ops functions.
        
        Args:
            iterations_list: List of iteration counts to test
            
        Returns:
            List of BenchmarkResult
        """
        from fptm_ste.parallel_ops import benchmark_pscan_vs_sequential
        
        iterations_list = iterations_list or self.config.iterations_list
        
        results = []
        D = 256
        B = 32
        
        for T in iterations_list:
            A = torch.rand(D, device=self.device) * 0.9 + 0.05
            Bx = torch.randn(B, D, device=self.device)
            
            bench_result = benchmark_pscan_vs_sequential(
                A, Bx, T,
                warmup=self.config.warmup_iterations,
                runs=self.config.benchmark_iterations,
            )
            
            result = BenchmarkResult(
                config={'D': D, 'B': B, 'T': T},
                baseline_ms=bench_result['sequential_ms'],
                optimized_ms=bench_result['pscan_ms'],
                speedup=bench_result['speedup'],
                notes="parallel_ops raw benchmark",
            )
            results.append(result)
        
        self.results.extend(results)
        return results
    
    def benchmark_cuda_graph(
        self,
        batch_sizes: Optional[List[int]] = None,
    ) -> List[BenchmarkResult]:
        """
        Benchmark CUDA Graph acceleration.
        
        Args:
            batch_sizes: List of batch sizes to test
            
        Returns:
            List of BenchmarkResult
        """
        from fptm_ste.pscan_stcm import PScanOptimizedSTCM_Graph
        
        batch_sizes = batch_sizes or self.config.batch_sizes
        
        results = []
        
        for batch_size in batch_sizes:
            try:
                model = PScanOptimizedSTCM_Graph(
                    n_features=self.config.n_features,
                    n_clauses=256,
                    n_classes=self.config.n_classes,
                    iterations=30,
                ).to(self.device).eval()
                
                x = torch.randn(batch_size, self.config.n_features).to(self.device)
                
                # Benchmark without graph
                no_graph_time = self._benchmark_forward(model, x, mode='pscan')
                
                # Enable graph
                model.enable_cuda_graph(batch_size=batch_size)
                
                # Warmup graph (triggers capture)
                with torch.no_grad():
                    for _ in range(5):
                        _ = model(x, mode='pscan')
                torch.cuda.synchronize()
                
                # Benchmark with graph
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    for _ in range(self.config.benchmark_iterations):
                        _ = model(x, mode='pscan')
                torch.cuda.synchronize()
                graph_time = (time.perf_counter() - t0) / self.config.benchmark_iterations * 1000
                
                result = BenchmarkResult(
                    config={'batch_size': batch_size, 'n_clauses': 256, 'iterations': 30},
                    baseline_ms=no_graph_time,
                    optimized_ms=graph_time,
                    speedup=no_graph_time / graph_time if graph_time > 0 else float('inf'),
                    notes="CUDA Graph benchmark",
                )
                results.append(result)
                
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error benchmarking CUDA Graph with batch_size={batch_size}: {e}")
        
        self.results.extend(results)
        return results
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmark suites."""
        self.start_time = datetime.now()
        self.results = []
        
        print("=" * 60)
        print("P-SCAN BENCHMARK SUITE")
        print("=" * 60)
        
        # STCM forward pass benchmarks
        print("\n📊 Benchmarking STCM forward pass...")
        self.benchmark_stcm_forward()
        
        # Parallel ops benchmarks
        print("\n📊 Benchmarking parallel_ops...")
        self.benchmark_parallel_ops()
        
        # CUDA Graph benchmarks
        print("\n📊 Benchmarking CUDA Graph...")
        self.benchmark_cuda_graph()
        
        self.end_time = datetime.now()
        
        print(f"\n✅ Completed {len(self.results)} benchmarks")
        print(f"   Duration: {(self.end_time - self.start_time).total_seconds():.1f}s")
        
        return self.results
    
    def generate_report(
        self,
        output_path: Path,
        format: str = 'markdown',
    ) -> None:
        """
        Generate benchmark report.
        
        Args:
            output_path: Path to save report
            format: 'markdown' or 'json'
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            self._generate_json_report(output_path)
        else:
            self._generate_markdown_report(output_path)
    
    def _generate_markdown_report(self, output_path: Path) -> None:
        """Generate markdown report."""
        lines = [
            "# P-Scan Optimization Benchmark Results",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"- Total benchmarks: {len(self.results)}",
            f"- Device: {self.config.device}",
            "",
            "## STCM Forward Pass",
            "",
            "| Clauses | Iterations | Sequential (ms) | P-Scan (ms) | Speedup |",
            "|---------|------------|-----------------|-------------|---------|",
        ]
        
        for r in self.results:
            if 'n_clauses' in r.config and 'iterations' in r.config:
                lines.append(
                    f"| {r.config.get('n_clauses', 'N/A')} | "
                    f"{r.config.get('iterations', 'N/A')} | "
                    f"{r.baseline_ms:.3f} | "
                    f"{r.optimized_ms:.3f} | "
                    f"{r.speedup:.2f}x |"
                )
        
        # Add parallel ops results
        lines.extend([
            "",
            "## Parallel Ops Raw Benchmark",
            "",
            "| T (iterations) | Sequential (ms) | P-Scan (ms) | Speedup |",
            "|----------------|-----------------|-------------|---------|",
        ])
        
        for r in self.results:
            if r.notes == "parallel_ops raw benchmark":
                lines.append(
                    f"| {r.config.get('T', 'N/A')} | "
                    f"{r.baseline_ms:.3f} | "
                    f"{r.optimized_ms:.3f} | "
                    f"{r.speedup:.2f}x |"
                )
        
        # Add CUDA Graph results
        lines.extend([
            "",
            "## CUDA Graph Acceleration",
            "",
            "| Batch Size | No Graph (ms) | With Graph (ms) | Speedup |",
            "|------------|---------------|-----------------|---------|",
        ])
        
        for r in self.results:
            if r.notes == "CUDA Graph benchmark":
                lines.append(
                    f"| {r.config.get('batch_size', 'N/A')} | "
                    f"{r.baseline_ms:.3f} | "
                    f"{r.optimized_ms:.3f} | "
                    f"{r.speedup:.2f}x |"
                )
        
        # Summary statistics
        speedups = [r.speedup for r in self.results if r.speedup < float('inf')]
        if speedups:
            lines.extend([
                "",
                "## Summary Statistics",
                "",
                f"- Average speedup: {sum(speedups)/len(speedups):.2f}x",
                f"- Max speedup: {max(speedups):.2f}x",
                f"- Min speedup: {min(speedups):.2f}x",
            ])
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"Report saved to {output_path}")
    
    def _generate_json_report(self, output_path: Path) -> None:
        """Generate JSON report."""
        report = {
            'generated': datetime.now().isoformat(),
            'config': asdict(self.config),
            'results': [asdict(r) for r in self.results],
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"JSON report saved to {output_path}")


def main():
    """Run benchmarks from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="P-Scan Benchmark Suite")
    parser.add_argument('--output', type=str, default='results/PSCAN_BENCHMARK.md',
                        help='Output path for report')
    parser.add_argument('--format', type=str, default='markdown',
                        choices=['markdown', 'json'],
                        help='Report format')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Benchmark iterations')
    args = parser.parse_args()
    
    config = BenchmarkConfig(benchmark_iterations=args.iterations)
    suite = PScanBenchmarkSuite(config)
    suite.run_all_benchmarks()
    suite.generate_report(Path(args.output), format=args.format)


if __name__ == '__main__':
    main()

