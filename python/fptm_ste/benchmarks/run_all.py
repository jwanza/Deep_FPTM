#!/usr/bin/env python3
"""
Comprehensive benchmark suite for TM models.

This script runs standardized benchmarks across all TM variants on specified datasets
and produces JSON output for tracking progress over time.

Usage:
    python -m fptm_ste.benchmarks.run_all --dataset cifar10 --epochs 50
    python -m fptm_ste.benchmarks.run_all --dataset mnist --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Add project root to path
PROJECT_ROOT = Path(__file__).parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fptm_ste import FuzzyPatternTM_STE, FuzzyPatternTM_STCM
from fptm_ste.deep_tm import DeepTMNetwork
from fptm_ste.benchmarks import (
    BenchmarkResult,
    BenchmarkSuite,
    count_parameters,
    measure_inference_throughput,
    get_peak_memory_mb,
)


# Default configuration
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_CONFIGS = {
    "cifar10": {
        "n_features": 3072,  # 32x32x3
        "n_classes": 10,
        "image_shape": (3, 32, 32),
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
    "mnist": {
        "n_features": 784,  # 28x28
        "n_classes": 10,
        "image_shape": (1, 28, 28),
        "mean": (0.1307,),
        "std": (0.3081,),
    },
    "fashionmnist": {
        "n_features": 784,
        "n_classes": 10,
        "image_shape": (1, 28, 28),
        "mean": (0.1307,),
        "std": (0.3081,),
    },
}

MODEL_CONFIGS = {
    "tm": {
        "class": "FuzzyPatternTM_STE",
        "kwargs": {"n_clauses": 200, "tau": 0.5},
    },
    "stcm": {
        "class": "FuzzyPatternTM_STCM",
        "kwargs": {"n_clauses": 200, "operator": "capacity"},
    },
    "stcm_product": {
        "class": "FuzzyPatternTM_STCM",
        "kwargs": {"n_clauses": 200, "operator": "product"},
    },
    "deep_tm": {
        "class": "DeepTMNetwork",
        "kwargs": {
            "hidden_dims": [512, 256],
            "n_clauses": 100,
            "layer_cls": FuzzyPatternTM_STE,
        },
    },
    "deep_stcm": {
        "class": "DeepTMNetwork",
        "kwargs": {
            "hidden_dims": [512, 256],
            "n_clauses": 100,
            "layer_cls": FuzzyPatternTM_STCM,
            "layer_operator": "capacity",
        },
    },
}


def get_dataloader(
    dataset_name: str,
    train: bool,
    batch_size: int,
    subset_size: Optional[int] = None,
    data_root: str = "/tmp",
) -> DataLoader:
    """Create a dataloader for the specified dataset."""
    config = DATASET_CONFIGS[dataset_name]
    
    transform_list = [transforms.ToTensor()]
    if len(config["mean"]) == 3:
        transform_list.append(transforms.Normalize(config["mean"], config["std"]))
    else:
        transform_list.append(transforms.Normalize(config["mean"], config["std"]))
    transform = transforms.Compose(transform_list)
    
    if dataset_name == "cifar10":
        dataset = datasets.CIFAR10(
            root=os.path.join(data_root, "cifar10"),
            train=train,
            download=True,
            transform=transform,
        )
    elif dataset_name == "mnist":
        dataset = datasets.MNIST(
            root=os.path.join(data_root, "mnist"),
            train=train,
            download=True,
            transform=transform,
        )
    elif dataset_name == "fashionmnist":
        dataset = datasets.FashionMNIST(
            root=os.path.join(data_root, "fashionmnist"),
            train=train,
            download=True,
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if subset_size is not None and subset_size < len(dataset):
        torch.manual_seed(42)
        indices = torch.randperm(len(dataset))[:subset_size].tolist()
        dataset = Subset(dataset, indices)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4 if DEVICE.type == "cuda" else 0,
        pin_memory=DEVICE.type == "cuda",
    )


def create_model(model_name: str, dataset_name: str) -> nn.Module:
    """Create a model instance for the given dataset."""
    config = MODEL_CONFIGS[model_name]
    dataset_config = DATASET_CONFIGS[dataset_name]
    
    kwargs = dict(config["kwargs"])
    kwargs["n_features"] = dataset_config["n_features"]
    kwargs["n_classes"] = dataset_config["n_classes"]
    
    # Handle different model types
    if config["class"] == "FuzzyPatternTM_STE":
        model = FuzzyPatternTM_STE(**kwargs)
    elif config["class"] == "FuzzyPatternTM_STCM":
        model = FuzzyPatternTM_STCM(**kwargs)
    elif config["class"] == "DeepTMNetwork":
        # DeepTMNetwork uses input_dim instead of n_features
        kwargs["input_dim"] = kwargs.pop("n_features")
        model = DeepTMNetwork(**kwargs)
    else:
        raise ValueError(f"Unknown model class: {config['class']}")
    
    return model


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch, return (loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_flat = x.view(x.size(0), -1)
        
        optimizer.zero_grad()
        output = model(x_flat)
        logits = output[0] if isinstance(output, tuple) else output
        
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    return total_loss / total, correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate model, return accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_flat = x.view(x.size(0), -1)
            
            output = model(x_flat)
            logits = output[0] if isinstance(output, tuple) else output
            
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    return correct / total


def run_benchmark(
    model_name: str,
    dataset_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    train_subset: Optional[int] = None,
    test_subset: Optional[int] = None,
    data_root: str = "/tmp",
    verbose: bool = True,
) -> BenchmarkResult:
    """Run a full benchmark for a single model/dataset combination."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Benchmarking {model_name} on {dataset_name}")
        print(f"{'='*60}")
    
    # Create dataloaders
    train_loader = get_dataloader(
        dataset_name, train=True, batch_size=batch_size,
        subset_size=train_subset, data_root=data_root,
    )
    test_loader = get_dataloader(
        dataset_name, train=False, batch_size=batch_size,
        subset_size=test_subset, data_root=data_root,
    )
    
    # Create model
    model = create_model(model_name, dataset_name).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Count parameters
    n_params = count_parameters(model)
    if verbose:
        print(f"Parameters: {n_params:,}")
    
    # Reset peak memory
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    # Training loop
    start_time = time.perf_counter()
    train_acc = 0.0
    
    for epoch in range(epochs):
        loss, train_acc = train_epoch(model, train_loader, optimizer, DEVICE)
        
        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            test_acc = evaluate(model, test_loader, DEVICE)
            print(f"Epoch {epoch+1}/{epochs}: loss={loss:.4f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")
    
    training_time = time.perf_counter() - start_time
    
    # Final evaluation
    test_acc = evaluate(model, test_loader, DEVICE)
    
    # Measure throughput
    dataset_config = DATASET_CONFIGS[dataset_name]
    input_shape = (batch_size,) + dataset_config["image_shape"]
    throughput = measure_inference_throughput(
        model, (batch_size, dataset_config["n_features"]), DEVICE,
    )
    
    # Get peak memory
    peak_memory = get_peak_memory_mb()
    
    if verbose:
        print(f"\nFinal Results:")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Training Time: {training_time:.1f}s")
        print(f"  Throughput: {throughput:.0f} img/s")
        print(f"  Peak Memory: {peak_memory:.0f} MB")
    
    return BenchmarkResult(
        model_name=model_name,
        dataset=dataset_name,
        test_accuracy=test_acc,
        train_accuracy=train_acc,
        parameters=n_params,
        epochs=epochs,
        batch_size=batch_size,
        training_time_seconds=training_time,
        inference_throughput=throughput,
        peak_memory_mb=peak_memory,
        config={
            "lr": lr,
            "train_subset": train_subset,
            "test_subset": test_subset,
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Run TM benchmarks")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset to benchmark on")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated list of models to benchmark")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode with subsampling")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    parser.add_argument("--data-root", type=str, default="/tmp",
                        help="Root directory for dataset downloads")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")
    
    args = parser.parse_args()
    
    # Determine which models to run
    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
    else:
        model_names = list(MODEL_CONFIGS.keys())
    
    # Validate model names
    for name in model_names:
        if name not in MODEL_CONFIGS:
            print(f"Unknown model: {name}. Available: {list(MODEL_CONFIGS.keys())}")
            sys.exit(1)
    
    # Quick mode settings
    train_subset = 5000 if args.quick else None
    test_subset = 1000 if args.quick else None
    epochs = min(args.epochs, 10) if args.quick else args.epochs
    
    # Create benchmark suite
    suite = BenchmarkSuite(
        name=f"{args.dataset}_benchmark",
        description=f"TM model benchmarks on {args.dataset}",
        metadata={
            "device": str(DEVICE),
            "quick_mode": args.quick,
            "timestamp": datetime.now().isoformat(),
        },
    )
    
    # Run benchmarks
    print(f"\nRunning benchmarks on {args.dataset} ({len(model_names)} models)")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {epochs}, Batch size: {args.batch_size}")
    if args.quick:
        print("(Quick mode enabled)")
    
    for model_name in model_names:
        try:
            result = run_benchmark(
                model_name=model_name,
                dataset_name=args.dataset,
                epochs=epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                train_subset=train_subset,
                test_subset=test_subset,
                data_root=args.data_root,
                verbose=not args.quiet,
            )
            suite.add_result(result)
        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = DEFAULT_OUTPUT_DIR / f"{args.dataset}_baseline_{timestamp}.json"
    
    suite.save(output_path)
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(suite.summary_table())
    
    # Check against baselines
    print("\n" + "-"*60)
    print("Baseline Checks:")
    from fptm_ste.benchmarks import check_regression
    for result in suite.results:
        passed, msg = check_regression(result.model_name, result.dataset, result.test_accuracy)
        status = "✓" if passed else "✗"
        print(f"  {status} {msg}")


if __name__ == "__main__":
    main()




