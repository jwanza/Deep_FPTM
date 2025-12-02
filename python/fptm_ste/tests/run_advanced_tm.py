#!/usr/bin/env python3
"""
Advanced Tsetlin Machine Runner.

Provides CLI access to all advanced TM variants implemented in fptm_ste:
- Booleanization solutions (CRCM, Probabilistic, HD, IB, Hierarchical, NeuralSymbolic)
- Ultimate Hybrid TM
- Temporal TM (for sequences)
- Continual Learning (EWC, SI, MAS)
- Advanced optimizers (SAM)
- Data augmentation (Mixup, CutMix)

Usage:
    python run_advanced_tm.py --model crcm --dataset mnist --epochs 10
    python run_advanced_tm.py --model ultimate_hybrid --dataset cifar10 --epochs 30
    python run_advanced_tm.py --model temporal --dataset mnist_sequence --epochs 20
"""

import argparse
import os
import time
from typing import Any, Dict, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


# =============================================================================
# Available Models
# =============================================================================


AVAILABLE_MODELS = (
    # Base TM
    "stcm",
    
    # Booleanization Solutions
    "crcm",              # Continuous Residual Clause Machine
    "probabilistic",     # Probabilistic Literal TM
    "hyperdimensional",  # HD Computing TM
    "ib",                # Information Bottleneck TM
    "hierarchical",      # Hierarchical Multi-Resolution TM
    "neural_symbolic",   # Neural Symbolic Transformer
    
    # Hybrid Architectures
    "ultimate_hybrid",   # Ultimate Hybrid TM
    "light_hybrid",      # Lightweight Hybrid
    
    # Temporal
    "temporal",          # Temporal Clause Machine
    "bidirectional",     # Bidirectional Temporal TM
    
    # Sparse Routing
    "sparse_moe",        # Sparse Mixture of Experts TM
)

AVAILABLE_DATASETS = (
    "mnist",
    "fashionmnist",
    "cifar10",
    "cifar100",
    "synthetic",  # For quick testing
)

AVAILABLE_OPTIMIZERS = (
    "adam",
    "adamw",
    "sgd",
    "sam",
    "sam_adam",
)

AVAILABLE_CONTINUAL = (
    "none",
    "ewc",
    "si",
    "mas",
    "replay",
)

AVAILABLE_AUGMENTATIONS = (
    "none",
    "mixup",
    "cutmix",
    "mixup_cutmix",
)


# =============================================================================
# Dataset Loading
# =============================================================================


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """Get dataset configuration."""
    configs = {
        "mnist": {
            "n_features": 784,
            "n_classes": 10,
            "image_size": (28, 28),
            "channels": 1,
            "mean": (0.1307,),
            "std": (0.3081,),
        },
        "fashionmnist": {
            "n_features": 784,
            "n_classes": 10,
            "image_size": (28, 28),
            "channels": 1,
            "mean": (0.2860,),
            "std": (0.3530,),
        },
        "cifar10": {
            "n_features": 3072,
            "n_classes": 10,
            "image_size": (32, 32),
            "channels": 3,
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2470, 0.2435, 0.2616),
        },
        "cifar100": {
            "n_features": 3072,
            "n_classes": 100,
            "image_size": (32, 32),
            "channels": 3,
            "mean": (0.5071, 0.4867, 0.4408),
            "std": (0.2675, 0.2565, 0.2761),
        },
        "synthetic": {
            "n_features": 64,
            "n_classes": 5,
            "image_size": (8, 8),
            "channels": 1,
            "mean": (0.5,),
            "std": (0.5,),
        },
    }
    return configs.get(dataset_name, configs["synthetic"])


def load_dataset(
    dataset_name: str,
    data_root: str = "/tmp/data",
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """Load dataset and return train/test loaders with config."""
    config = get_dataset_config(dataset_name)
    
    if dataset_name == "synthetic":
        return create_synthetic_dataset(
            n_samples=2000,
            n_features=config["n_features"],
            n_classes=config["n_classes"],
            batch_size=batch_size,
        ), config
    
    # Determine transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config["mean"], config["std"]),
        transforms.Lambda(lambda x: x.view(-1)),  # Flatten
    ])
    
    # Load dataset
    dataset_classes = {
        "mnist": datasets.MNIST,
        "fashionmnist": datasets.FashionMNIST,
        "cifar10": datasets.CIFAR10,
        "cifar100": datasets.CIFAR100,
    }
    
    DatasetClass = dataset_classes.get(dataset_name)
    if DatasetClass is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_dataset = DatasetClass(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )
    
    test_dataset = DatasetClass(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, test_loader, config


def create_synthetic_dataset(
    n_samples: int,
    n_features: int,
    n_classes: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """Create synthetic dataset for testing."""
    torch.manual_seed(42)
    
    # Create data
    x = torch.rand(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))
    
    # Add class-dependent patterns
    for c in range(n_classes):
        mask = y == c
        x[mask, c * (n_features // n_classes):(c + 1) * (n_features // n_classes)] += 0.5
    x = x.clamp(0, 1)
    
    # Split
    split = int(0.8 * n_samples)
    train_x, train_y = x[:split], y[:split]
    test_x, test_y = x[split:], y[split:]
    
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
    )
    
    test_loader = DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=batch_size,
        shuffle=False,
    )
    
    config = {
        "n_features": n_features,
        "n_classes": n_classes,
    }
    
    return train_loader, test_loader, config


# =============================================================================
# Model Factory
# =============================================================================


def create_model(
    model_name: str,
    n_features: int,
    n_classes: int,
    n_clauses: int = 64,
    **kwargs,
) -> nn.Module:
    """Create a model by name."""
    
    if model_name == "stcm":
        from fptm_ste import FuzzyPatternTM_STCM
        return FuzzyPatternTM_STCM(
            n_features=n_features,
            n_clauses=n_clauses,
            n_classes=n_classes,
            **kwargs,
        )
    
    elif model_name == "crcm":
        from fptm_ste.booleanization import ContinuousResidualClauseMachine
        return ContinuousResidualClauseMachine(
            n_features=n_features,
            n_clauses=n_clauses,
            n_classes=n_classes,
            hidden_dim=kwargs.get("hidden_dim", 64),
        )
    
    elif model_name == "probabilistic":
        from fptm_ste.booleanization import ProbabilisticLiteralClauseMachine
        return ProbabilisticLiteralClauseMachine(
            n_features=n_features,
            n_clauses=n_clauses,
            n_classes=n_classes,
        )
    
    elif model_name == "hyperdimensional":
        from fptm_ste.booleanization import HyperdimensionalClauseMachine
        return HyperdimensionalClauseMachine(
            n_features=n_features,
            n_clauses=n_clauses,
            n_classes=n_classes,
            hd_dim=kwargs.get("hd_dim", 1000),
        )
    
    elif model_name == "ib":
        from fptm_ste.booleanization import InformationPreservingClauseMachine
        return InformationPreservingClauseMachine(
            n_features=n_features,
            n_clauses=n_clauses,
            n_classes=n_classes,
            n_binary=kwargs.get("n_binary", 16),
        )
    
    elif model_name == "hierarchical":
        from fptm_ste.booleanization import HierarchicalMultiResolutionTM
        return HierarchicalMultiResolutionTM(
            n_features=n_features,
            n_clauses_per_level=[n_clauses // 2] * kwargs.get("n_levels", 3),
            n_classes=n_classes,
            resolutions=[2, 4, 8][:kwargs.get("n_levels", 3)],
        )
    
    elif model_name == "neural_symbolic":
        from fptm_ste.booleanization import NeuralSymbolicTransformer
        return NeuralSymbolicTransformer(
            n_features=n_features,
            n_clauses=n_clauses,
            n_classes=n_classes,
            d_model=kwargs.get("d_model", 64),
        )
    
    elif model_name == "ultimate_hybrid":
        from fptm_ste.ultimate_hybrid import UltimateHybridTM
        return UltimateHybridTM(
            n_features=n_features,
            n_clauses=n_clauses,
            n_classes=n_classes,
            use_binary_stream=True,
            use_continuous_stream=True,
            use_hd_stream=True,
            use_ib_stream=False,
            use_probabilistic_stream=False,
        )
    
    elif model_name == "light_hybrid":
        from fptm_ste.ultimate_hybrid import create_light_hybrid
        return create_light_hybrid(
            n_features=n_features,
            n_clauses=n_clauses,
            n_classes=n_classes,
        )
    
    elif model_name == "temporal":
        from fptm_ste.temporal import TemporalClauseMachine
        return TemporalClauseMachine(
            n_features=n_features,
            n_clauses=n_clauses,
            n_classes=n_classes,
            state_dim=kwargs.get("state_dim", 64),
        )
    
    elif model_name == "bidirectional":
        from fptm_ste.temporal import BidirectionalTemporalClauseMachine
        return BidirectionalTemporalClauseMachine(
            n_features=n_features,
            n_clauses=n_clauses,
            n_classes=n_classes,
            state_dim=kwargs.get("state_dim", 64),
        )
    
    elif model_name == "sparse_moe":
        from fptm_ste.sparse_routing import SparseMoEClauseMachine
        return SparseMoEClauseMachine(
            n_features=n_features,
            n_clauses_per_expert=n_clauses // 4,
            n_classes=n_classes,
            n_experts=8,
            top_k=2,
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# Optimizer Factory
# =============================================================================


def create_optimizer(
    optimizer_name: str,
    model: nn.Module,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    **kwargs,
) -> torch.optim.Optimizer:
    """Create optimizer by name."""
    
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=kwargs.get("momentum", 0.9),
        )
    
    elif optimizer_name == "sam":
        from fptm_ste.sam_optimizer import SAM
        return SAM(
            model.parameters(),
            torch.optim.SGD,
            lr=lr,
            rho=kwargs.get("sam_rho", 0.05),
            momentum=kwargs.get("momentum", 0.9),
        )
    
    elif optimizer_name == "sam_adam":
        from fptm_ste.sam_optimizer import SAM
        return SAM(
            model.parameters(),
            torch.optim.Adam,
            lr=lr,
            rho=kwargs.get("sam_rho", 0.05),
        )
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


# =============================================================================
# Continual Learning Wrapper
# =============================================================================


def wrap_continual_learning(
    model: nn.Module,
    continual_type: str,
    **kwargs,
) -> nn.Module:
    """Wrap model with continual learning mechanism."""
    
    if continual_type == "none":
        return model
    
    elif continual_type == "ewc":
        from fptm_ste.continual import EWCWrapper
        return EWCWrapper(model, lambda_=kwargs.get("ewc_lambda", 1000.0))
    
    elif continual_type == "si":
        from fptm_ste.continual import SynapticIntelligence
        return SynapticIntelligence(model, c=kwargs.get("si_c", 1.0))
    
    elif continual_type == "mas":
        from fptm_ste.continual import MemoryAwareSynapses
        return MemoryAwareSynapses(model, lambda_=kwargs.get("mas_lambda", 1.0))
    
    else:
        raise ValueError(f"Unknown continual learning type: {continual_type}")


# =============================================================================
# Augmentation
# =============================================================================


def get_augmentation_fn(
    augmentation_type: str,
    alpha: float = 0.4,
) -> Optional[Callable]:
    """Get augmentation function."""
    
    if augmentation_type == "none":
        return None
    
    elif augmentation_type == "mixup":
        from fptm_ste.augmentation import mixup_data
        def augment(x, y):
            return mixup_data(x, y, alpha=alpha)
        return augment
    
    elif augmentation_type == "cutmix":
        from fptm_ste.augmentation import cutmix_data
        def augment(x, y):
            return cutmix_data(x, y, alpha=alpha)
        return augment
    
    elif augmentation_type == "mixup_cutmix":
        from fptm_ste.augmentation import mixup_data, cutmix_data
        import random
        def augment(x, y):
            if random.random() < 0.5:
                return mixup_data(x, y, alpha=alpha)
            else:
                return cutmix_data(x, y, alpha=alpha)
        return augment
    
    else:
        return None


# =============================================================================
# Training Loop
# =============================================================================


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    augment_fn: Optional[Callable] = None,
    is_sam: bool = False,
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # Apply augmentation
        lam = 1.0
        y_a, y_b = batch_y, batch_y
        if augment_fn is not None:
            batch_x, y_a, y_b, lam = augment_fn(batch_x, batch_y)
        
        if is_sam:
            # SAM requires two forward passes
            # First step
            output = model(batch_x)
            logits = output[0] if isinstance(output, tuple) else output
            loss = lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            
            # Second step
            output = model(batch_x)
            logits = output[0] if isinstance(output, tuple) else output
            loss = lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)
            loss.backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.zero_grad()
            
            output = model(batch_x)
            logits = output[0] if isinstance(output, tuple) else output
            
            loss = lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate model accuracy."""
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            output = model(batch_x)
            logits = output[0] if isinstance(output, tuple) else output
            
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    
    return correct / total


# =============================================================================
# Main Training Function
# =============================================================================


def train_model(args: argparse.Namespace) -> Dict[str, Any]:
    """Main training function."""
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    train_loader, test_loader, config = load_dataset(
        args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    n_features = config["n_features"]
    n_classes = config["n_classes"]
    print(f"Dataset: {n_features} features, {n_classes} classes")
    
    # Create model
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        n_features=n_features,
        n_classes=n_classes,
        n_clauses=args.n_clauses,
        hidden_dim=args.hidden_dim,
        hd_dim=args.hd_dim,
        state_dim=args.state_dim,
    )
    model = model.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable")
    
    # Wrap with continual learning if needed
    if args.continual != "none":
        print(f"Wrapping with continual learning: {args.continual}")
        model = wrap_continual_learning(model, args.continual)
    
    # Create optimizer
    is_sam = args.optimizer.startswith("sam")
    optimizer = create_optimizer(
        args.optimizer,
        model if not hasattr(model, 'model') else model.model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        sam_rho=args.sam_rho,
    )
    print(f"Optimizer: {args.optimizer}, lr={args.lr}")
    
    # Augmentation
    augment_fn = get_augmentation_fn(args.augmentation, alpha=args.mixup_alpha)
    if augment_fn:
        print(f"Augmentation: {args.augmentation}")
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 60)
    
    start_time = time.time()
    best_test_acc = 0.0
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch(
            model if not hasattr(model, 'model') else model.model,
            train_loader,
            optimizer,
            device,
            augment_fn=augment_fn,
            is_sam=is_sam,
        )
        
        test_acc = evaluate(
            model if not hasattr(model, 'model') else model.model,
            test_loader,
            device,
        )
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch + 1:3d}/{args.epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train: {train_acc:.4f} | "
              f"Test: {test_acc:.4f} | "
              f"Best: {best_test_acc:.4f} | "
              f"Time: {epoch_time:.1f}s")
    
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Training completed in {total_time:.1f}s")
    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"Best test accuracy: {best_test_acc:.4f}")
    
    return {
        "model": args.model,
        "dataset": args.dataset,
        "final_test_acc": test_acc,
        "best_test_acc": best_test_acc,
        "train_time": total_time,
        "n_params": n_params,
    }


# =============================================================================
# Argument Parser
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Advanced Tsetlin Machine Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="crcm",
        choices=AVAILABLE_MODELS,
        help="Model type to train",
    )
    
    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=AVAILABLE_DATASETS,
        help="Dataset to use",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/tmp/data",
        help="Root directory for datasets",
    )
    
    # Training
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    
    # Model architecture
    parser.add_argument("--n-clauses", type=int, default=64, help="Number of clauses")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--hd-dim", type=int, default=1000, help="HD vector dimension")
    parser.add_argument("--state-dim", type=int, default=64, help="Temporal state dimension")
    
    # Optimizer
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=AVAILABLE_OPTIMIZERS,
        help="Optimizer type",
    )
    parser.add_argument("--sam-rho", type=float, default=0.05, help="SAM perturbation radius")
    
    # Continual learning
    parser.add_argument(
        "--continual",
        type=str,
        default="none",
        choices=AVAILABLE_CONTINUAL,
        help="Continual learning method",
    )
    
    # Augmentation
    parser.add_argument(
        "--augmentation",
        type=str,
        default="none",
        choices=AVAILABLE_AUGMENTATIONS,
        help="Data augmentation method",
    )
    parser.add_argument("--mixup-alpha", type=float, default=0.4, help="Mixup/CutMix alpha")
    
    # Device
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")
    
    # Output
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    
    return parser


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()
    
    print("=" * 60)
    print("Advanced Tsetlin Machine Runner")
    print("=" * 60)
    
    results = train_model(args)
    
    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return results


if __name__ == "__main__":
    main()

