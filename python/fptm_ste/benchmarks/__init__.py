"""
Benchmark utilities for TM models.

This module provides standardized benchmarking infrastructure for comparing
TM variants across datasets, with JSON output for tracking progress.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


@dataclass
class BenchmarkResult:
    """Single benchmark run result."""
    
    model_name: str
    dataset: str
    test_accuracy: float
    train_accuracy: float
    parameters: int
    epochs: int
    batch_size: int
    training_time_seconds: float
    inference_throughput: float  # images/second
    peak_memory_mb: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkResult":
        return cls(**d)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    
    name: str
    description: str
    results: List[BenchmarkResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_result(self, result: BenchmarkResult) -> None:
        self.results.append(result)
    
    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "results": [r.to_dict() for r in self.results],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "BenchmarkSuite":
        with open(path, "r") as f:
            data = json.load(f)
        suite = cls(
            name=data["name"],
            description=data["description"],
            metadata=data.get("metadata", {}),
        )
        suite.results = [BenchmarkResult.from_dict(r) for r in data.get("results", [])]
        return suite
    
    def get_best(self, metric: str = "test_accuracy", model_filter: Optional[str] = None) -> Optional[BenchmarkResult]:
        """Get best result by metric, optionally filtered by model name."""
        filtered = self.results
        if model_filter:
            filtered = [r for r in filtered if model_filter in r.model_name]
        if not filtered:
            return None
        return max(filtered, key=lambda r: getattr(r, metric))
    
    def summary_table(self) -> str:
        """Generate markdown summary table."""
        if not self.results:
            return "No results recorded."
        
        lines = [
            "| Model | Dataset | Test Acc | Train Acc | Params | Throughput |",
            "|-------|---------|----------|-----------|--------|------------|",
        ]
        for r in sorted(self.results, key=lambda x: -x.test_accuracy):
            lines.append(
                f"| {r.model_name} | {r.dataset} | {r.test_accuracy:.4f} | "
                f"{r.train_accuracy:.4f} | {r.parameters:,} | {r.inference_throughput:.0f} img/s |"
            )
        return "\n".join(lines)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_throughput(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device,
    num_batches: int = 100,
    warmup_batches: int = 10,
) -> float:
    """Measure inference throughput in images/second."""
    model.eval()
    batch_size = input_shape[0]
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_batches):
            x = torch.randn(*input_shape, device=device)
            _ = model(x)
    
    # Synchronize if CUDA
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_batches):
            x = torch.randn(*input_shape, device=device)
            _ = model(x)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    total_images = batch_size * num_batches
    return total_images / elapsed


def get_peak_memory_mb() -> float:
    """Get peak GPU memory usage in MB, or 0 if not using CUDA."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


# Default baseline thresholds for regression testing
CIFAR10_BASELINES = {
    "tm": 0.45,           # Basic TM should achieve at least 45%
    "stcm": 0.50,         # STCM should do better
    "deep_tm": 0.55,      # Deep TM improves on basic
    "deep_stcm": 0.60,    # Deep STCM best among deep variants
    "deep_ctm": 0.65,     # Convolutional TM
    "deep_cstcm": 0.68,   # Convolutional STCM
    "transformer": 0.70,  # Transformer-based
}

MNIST_BASELINES = {
    "tm": 0.92,
    "stcm": 0.95,
    "deep_tm": 0.97,
    "deep_stcm": 0.98,
    "deep_ctm": 0.98,
    "deep_cstcm": 0.985,
    "transformer": 0.98,
}


def check_regression(
    model_name: str,
    dataset: str,
    accuracy: float,
    tolerance: float = 0.02,
) -> Tuple[bool, str]:
    """
    Check if accuracy meets baseline threshold.
    
    Returns:
        (passed, message)
    """
    if dataset == "cifar10":
        baselines = CIFAR10_BASELINES
    elif dataset == "mnist":
        baselines = MNIST_BASELINES
    else:
        return True, f"No baseline defined for {dataset}"
    
    if model_name not in baselines:
        return True, f"No baseline defined for {model_name}"
    
    threshold = baselines[model_name] - tolerance
    passed = accuracy >= threshold
    
    if passed:
        msg = f"{model_name} on {dataset}: {accuracy:.4f} >= {threshold:.4f} (baseline {baselines[model_name]:.4f}) ✓"
    else:
        msg = f"{model_name} on {dataset}: {accuracy:.4f} < {threshold:.4f} (baseline {baselines[model_name]:.4f}) ✗"
    
    return passed, msg

