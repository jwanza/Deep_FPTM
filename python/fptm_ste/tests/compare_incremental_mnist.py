#!/usr/bin/env python3
"""
MNIST Comparison: Baseline STCM vs Enhanced STCM Training

This script compares different training approaches:
1. Baseline: Standard gradient training with AdamW
2. Stable: Enhanced training with EMA, confidence weighting, and regularization
3. Curriculum: Curriculum learning with annealed temperature

Key metrics:
- Final accuracy
- Training stability (variance across epochs)
- Convergence speed
"""

import argparse
import os
import sys
import time
import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fptm_ste import FuzzyPatternTM_STCM
from fptm_ste.tm_optimized import OptimizedSTCM


@dataclass
class ExperimentConfig:
    """Configuration for comparison experiment."""
    epochs: int = 20
    batch_size: int = 128
    n_clauses: int = 2000
    lr: float = 0.001
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    data_root: str = "/tmp/mnist"
    verbose: bool = True
    quick_test: bool = False


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_mnist(config: ExperimentConfig) -> Tuple[DataLoader, DataLoader]:
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST(
        root=config.data_root, train=True, download=True, transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=config.data_root, train=False, download=True, transform=transform,
    )
    
    if config.quick_test:
        train_dataset = torch.utils.data.Subset(train_dataset, range(5000))
        test_dataset = torch.utils.data.Subset(test_dataset, range(1000))
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True,
    )
    
    return train_loader, test_loader


class EMA:
    """Exponential Moving Average for model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow = {n: p.clone().detach() for n, p in model.named_parameters()}
    
    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.shadow:
                    self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)
    
    def apply(self, model: nn.Module) -> None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.shadow:
                    p.data.copy_(self.shadow[n])


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    """Evaluate model accuracy."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            logits, _ = model(data, use_ste=False)
            preds = logits.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    return correct / total


def train_baseline(train_loader, test_loader, config) -> Dict[str, List[float]]:
    """Baseline: Standard gradient training."""
    print("\n" + "="*60)
    print("BASELINE: Standard Gradient Training (AdamW)")
    print("="*60)
    
    set_seed(config.seed)
    
    model = OptimizedSTCM(
        n_features=784, n_clauses=config.n_clauses, n_classes=10,
        tau=0.5, operator="capacity", clause_dropout=0.1,
    ).to(config.device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'epoch_time': []}
    
    for epoch in range(config.epochs):
        start = time.time()
        model.train()
        total_loss = total_correct = total_samples = 0
        
        for data, target in train_loader:
            data = data.view(data.size(0), -1).to(config.device)
            target = target.to(config.device)
            
            optimizer.zero_grad()
            logits, _ = model(data, use_ste=True)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * target.size(0)
            total_correct += (logits.argmax(1) == target).sum().item()
            total_samples += target.size(0)
        
        scheduler.step()
        test_acc = evaluate(model, test_loader, config.device)
        elapsed = time.time() - start
        
        history['train_loss'].append(total_loss / total_samples)
        history['train_acc'].append(total_correct / total_samples)
        history['test_acc'].append(test_acc)
        history['epoch_time'].append(elapsed)
        
        if config.verbose:
            print(f"Epoch {epoch+1:3d}/{config.epochs}: "
                  f"loss={total_loss/total_samples:.4f}, "
                  f"train={total_correct/total_samples:.4f}, "
                  f"test={test_acc:.4f}, time={elapsed:.2f}s")
    
    return history


def train_with_ema(train_loader, test_loader, config) -> Dict[str, List[float]]:
    """Enhanced: Training with EMA and warmup."""
    print("\n" + "="*60)
    print("ENHANCED: Gradient Training + EMA (Exponential Moving Avg)")
    print("="*60)
    
    set_seed(config.seed)
    
    model = OptimizedSTCM(
        n_features=784, n_clauses=config.n_clauses, n_classes=10,
        tau=0.5, operator="capacity", clause_dropout=0.1,
    ).to(config.device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.epochs//2 + 1, T_mult=2)
    ema = EMA(model, decay=0.995)
    
    history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'epoch_time': []}
    
    for epoch in range(config.epochs):
        start = time.time()
        model.train()
        total_loss = total_correct = total_samples = 0
        
        for data, target in train_loader:
            data = data.view(data.size(0), -1).to(config.device)
            target = target.to(config.device)
            
            optimizer.zero_grad()
            logits, _ = model(data, use_ste=True)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)  # Update EMA after each step
            
            total_loss += loss.item() * target.size(0)
            total_correct += (logits.argmax(1) == target).sum().item()
            total_samples += target.size(0)
        
        scheduler.step()
        
        # Evaluate using EMA weights
        orig_state = {n: p.clone() for n, p in model.named_parameters()}
        ema.apply(model)
        test_acc = evaluate(model, test_loader, config.device)
        # Restore for next epoch training
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.data.copy_(orig_state[n])
        
        elapsed = time.time() - start
        
        history['train_loss'].append(total_loss / total_samples)
        history['train_acc'].append(total_correct / total_samples)
        history['test_acc'].append(test_acc)
        history['epoch_time'].append(elapsed)
        
        if config.verbose:
            print(f"Epoch {epoch+1:3d}/{config.epochs}: "
                  f"loss={total_loss/total_samples:.4f}, "
                  f"train={total_correct/total_samples:.4f}, "
                  f"test={test_acc:.4f} (EMA), time={elapsed:.2f}s")
    
    return history


def train_with_regularization(train_loader, test_loader, config) -> Dict[str, List[float]]:
    """Enhanced: Training with sparsity + diversity regularization."""
    print("\n" + "="*60)
    print("REGULARIZED: Gradient + Clause Sparsity + Diversity Loss")
    print("="*60)
    
    set_seed(config.seed)
    
    model = OptimizedSTCM(
        n_features=784, n_clauses=config.n_clauses, n_classes=10,
        tau=0.5, operator="capacity", clause_dropout=0.15,
    ).to(config.device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    # Regularization weights - mimic Julia's feedback behavior
    sparsity_weight = 0.001   # L1 to encourage sparsity like TM literal suppression
    diversity_weight = 0.0001  # Encourage clause diversity
    
    history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'epoch_time': []}
    
    for epoch in range(config.epochs):
        start = time.time()
        model.train()
        total_loss = total_correct = total_samples = 0
        
        for data, target in train_loader:
            data = data.view(data.size(0), -1).to(config.device)
            target = target.to(config.device)
            
            optimizer.zero_grad()
            logits, clause_outputs = model(data, use_ste=True)
            
            # Main classification loss
            ce_loss = F.cross_entropy(logits, target)
            
            # Sparsity regularization on clause weights (like Julia's suppress)
            pos_logits = model.pos_logits
            neg_logits = model.neg_logits
            sparsity_loss = sparsity_weight * (
                torch.sigmoid(pos_logits).mean() + 
                torch.sigmoid(neg_logits).mean()
            )
            
            # Diversity loss - encourage different clauses to activate for different samples
            # Compute correlation between clause outputs
            if clause_outputs.dim() == 2:
                clause_norm = F.normalize(clause_outputs.detach(), dim=0)
                correlation = (clause_norm.T @ clause_norm).abs()
                # Penalize high correlation (excluding diagonal)
                mask = 1 - torch.eye(correlation.size(0), device=correlation.device)
                diversity_loss = diversity_weight * (correlation * mask).mean()
            else:
                diversity_loss = torch.tensor(0.0, device=config.device)
            
            loss = ce_loss + sparsity_loss + diversity_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += ce_loss.item() * target.size(0)
            total_correct += (logits.argmax(1) == target).sum().item()
            total_samples += target.size(0)
        
        scheduler.step()
        test_acc = evaluate(model, test_loader, config.device)
        elapsed = time.time() - start
        
        history['train_loss'].append(total_loss / total_samples)
        history['train_acc'].append(total_correct / total_samples)
        history['test_acc'].append(test_acc)
        history['epoch_time'].append(elapsed)
        
        if config.verbose:
            print(f"Epoch {epoch+1:3d}/{config.epochs}: "
                  f"loss={total_loss/total_samples:.4f}, "
                  f"train={total_correct/total_samples:.4f}, "
                  f"test={test_acc:.4f}, time={elapsed:.2f}s")
    
    return history


def train_full_enhanced(train_loader, test_loader, config) -> Dict[str, List[float]]:
    """Full enhanced: EMA + Regularization + Label Smoothing."""
    print("\n" + "="*60)
    print("FULL ENHANCED: EMA + Regularization + Label Smoothing")
    print("="*60)
    
    set_seed(config.seed)
    
    model = OptimizedSTCM(
        n_features=784, n_clauses=config.n_clauses, n_classes=10,
        tau=0.5, operator="capacity", clause_dropout=0.15,
    ).to(config.device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(1, config.epochs//3), T_mult=2)
    ema = EMA(model, decay=0.995)
    
    # Regularization weights
    sparsity_weight = 0.0005
    label_smoothing = 0.1
    
    history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'epoch_time': []}
    
    for epoch in range(config.epochs):
        start = time.time()
        model.train()
        total_loss = total_correct = total_samples = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(data.size(0), -1).to(config.device)
            target = target.to(config.device)
            
            optimizer.zero_grad()
            logits, _ = model(data, use_ste=True)
            
            # Label smoothed cross entropy
            ce_loss = F.cross_entropy(logits, target, label_smoothing=label_smoothing)
            
            # Sparsity on clause selection
            pos_probs = torch.sigmoid(model.pos_logits)
            neg_probs = torch.sigmoid(model.neg_logits)
            sparsity_loss = sparsity_weight * (pos_probs.mean() + neg_probs.mean())
            
            loss = ce_loss + sparsity_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)
            
            total_loss += ce_loss.item() * target.size(0)
            total_correct += (logits.argmax(1) == target).sum().item()
            total_samples += target.size(0)
        
        scheduler.step()
        
        # Evaluate with EMA
        orig_state = {n: p.clone() for n, p in model.named_parameters()}
        ema.apply(model)
        test_acc = evaluate(model, test_loader, config.device)
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.data.copy_(orig_state[n])
        
        elapsed = time.time() - start
        
        history['train_loss'].append(total_loss / total_samples)
        history['train_acc'].append(total_correct / total_samples)
        history['test_acc'].append(test_acc)
        history['epoch_time'].append(elapsed)
        
        if config.verbose:
            print(f"Epoch {epoch+1:3d}/{config.epochs}: "
                  f"loss={total_loss/total_samples:.4f}, "
                  f"train={total_correct/total_samples:.4f}, "
                  f"test={test_acc:.4f}, time={elapsed:.2f}s")
    
    return history


def compute_metrics(history: Dict[str, List[float]]) -> Dict[str, float]:
    """Compute summary metrics from training history."""
    test_accs = history['test_acc']
    train_accs = history['train_acc']
    
    # Calculate stability: lower std = more stable
    mean_acc = sum(test_accs) / len(test_accs)
    std_acc = (sum((a - mean_acc)**2 for a in test_accs) / len(test_accs)) ** 0.5
    
    # Calculate late convergence (last 3 epochs)
    late_mean = sum(test_accs[-3:]) / min(3, len(test_accs))
    
    return {
        'best_test_acc': max(test_accs),
        'final_test_acc': test_accs[-1],
        'avg_test_acc': mean_acc,
        'test_acc_std': std_acc,
        'late_convergence': late_mean,
        'best_train_acc': max(train_accs),
        'generalization_gap': max(train_accs) - test_accs[train_accs.index(max(train_accs))],
        'total_time': sum(history['epoch_time']),
        'avg_epoch_time': sum(history['epoch_time']) / len(history['epoch_time']),
    }


def print_comparison(results: Dict[str, Dict]) -> None:
    """Print comparison table."""
    print("\n" + "="*90)
    print("COMPARISON RESULTS")
    print("="*90)
    
    methods = list(results.keys())
    
    # Table header
    header = f"{'Metric':<25}"
    for m in methods:
        header += f" {m[:12]:<12}"
    print(f"\n{header}")
    print("-"*90)
    
    metrics_info = [
        ('best_test_acc', 'Best Test Accuracy', True),
        ('final_test_acc', 'Final Test Accuracy', True),
        ('late_convergence', 'Late Convergence', True),
        ('test_acc_std', 'Stability (lower=better)', False),
        ('generalization_gap', 'Generalization Gap', False),
        ('avg_epoch_time', 'Avg Epoch Time (s)', False),
    ]
    
    for metric, label, higher_is_better in metrics_info:
        row = f"{label:<25}"
        values = [results[m].get(metric, 0) for m in methods]
        
        # Find best value
        if higher_is_better:
            best_val = max(values)
        else:
            best_val = min(values)
        
        for val in values:
            if 'time' in metric.lower():
                formatted = f"{val:.2f}"
            elif 'std' in metric.lower() or 'gap' in metric.lower():
                formatted = f"{val:.4f}"
            else:
                formatted = f"{val*100:.2f}%"
            
            if val == best_val:
                formatted = f"*{formatted}*"
            row += f" {formatted:<12}"
        
        print(row)
    
    # Find overall winner
    print("\n" + "-"*90)
    winner_scores = {m: 0 for m in methods}
    for metric, _, higher_is_better in metrics_info:
        values = [(m, results[m].get(metric, 0)) for m in methods]
        if higher_is_better:
            winner = max(values, key=lambda x: x[1])[0]
        else:
            winner = min(values, key=lambda x: x[1])[0]
        winner_scores[winner] += 1
    
    overall_winner = max(winner_scores.items(), key=lambda x: x[1])
    print(f"\n🏆 WINNER: {overall_winner[0]} (won {overall_winner[1]}/{len(metrics_info)} metrics)")
    
    # Summary
    print("\n📊 INSIGHTS:")
    baseline_acc = results.get('baseline', {}).get('best_test_acc', 0)
    for method in methods:
        if method != 'baseline':
            acc = results[method].get('best_test_acc', 0)
            diff = (acc - baseline_acc) * 100
            if diff > 0:
                print(f"   • {method}: +{diff:.2f}% vs baseline")
            else:
                print(f"   • {method}: {diff:.2f}% vs baseline")


def main():
    parser = argparse.ArgumentParser(description="Compare STCM training approaches on MNIST")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--n-clauses", type=int, default=2000, help="Number of clauses")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quick", action="store_true", help="Quick test with subset")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    config = ExperimentConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_clauses=args.n_clauses,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        quick_test=args.quick,
    )
    
    print("="*60)
    print("MNIST TRAINING METHODS COMPARISON")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Clauses: {config.n_clauses}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Device: {config.device}")
    print(f"  Quick test: {config.quick_test}")
    
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = load_mnist(config)
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    results = {}
    
    # Run all methods
    history = train_baseline(train_loader, test_loader, config)
    results['baseline'] = compute_metrics(history)
    
    history = train_with_ema(train_loader, test_loader, config)
    results['EMA'] = compute_metrics(history)
    
    history = train_with_regularization(train_loader, test_loader, config)
    results['regularized'] = compute_metrics(history)
    
    history = train_full_enhanced(train_loader, test_loader, config)
    results['full_enhanced'] = compute_metrics(history)
    
    # Print comparison
    print_comparison(results)
    
    print("\n" + "="*60)
    print("Experiment complete!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    main()
