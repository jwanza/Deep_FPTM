#!/usr/bin/env python3
"""
Deep STCM MNIST Benchmark with Enhanced Training

Compares baseline vs enhanced training for DeepTMNetwork.
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fptm_ste import FuzzyPatternTM_STCM
from fptm_ste.deep_tm import DeepTMNetwork
from fptm_ste.tm_optimized import OptimizedSTCM


class EMA:
    """Exponential Moving Average for model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.clone().detach()
    
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
    
    def store(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        return {n: p.clone() for n, p in model.named_parameters() if n in self.shadow}
    
    def restore(self, model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in state:
                    p.data.copy_(state[n])


def load_mnist(batch_size: int, data_root: str = "/tmp/mnist") -> Tuple[DataLoader, DataLoader]:
    """Load MNIST dataset."""
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_ds = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    """Evaluate model accuracy."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            logits = model(data, use_ste=False)
            if isinstance(logits, tuple):
                logits = logits[0]
            preds = logits.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    return correct / total


def train_baseline_deep_stcm(
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    n_clauses: int,
    device: str,
    verbose: bool = True,
) -> Dict[str, float]:
    """Train Deep STCM with baseline gradient descent."""
    print("\n" + "="*70)
    print("BASELINE: Deep STCM with Standard Gradient Training")
    print("="*70)
    
    torch.manual_seed(42)
    
    model = DeepTMNetwork(
        input_dim=784,
        hidden_dims=[256, 128],
        n_classes=10,
        n_clauses=n_clauses,
        dropout=0.1,
        tau=0.5,
        clause_dropout=0.1,
        layer_cls=OptimizedSTCM,
        layer_operator="capacity",
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    history = []
    
    for epoch in range(epochs):
        start = time.time()
        model.train()
        total_loss = total_correct = total_samples = 0
        
        for data, target in train_loader:
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            logits = model(data, use_ste=True)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = F.cross_entropy(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * target.size(0)
            total_correct += (logits.argmax(1) == target).sum().item()
            total_samples += target.size(0)
        
        scheduler.step()
        
        test_acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, test_acc)
        elapsed = time.time() - start
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': total_loss / total_samples,
            'train_acc': total_correct / total_samples,
            'test_acc': test_acc,
            'best_acc': best_acc,
            'time': elapsed,
        })
        
        if verbose:
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"loss={total_loss/total_samples:.4f}, "
                  f"train={total_correct/total_samples:.4f}, "
                  f"test={test_acc:.4f}, "
                  f"best={best_acc:.4f}, "
                  f"time={elapsed:.1f}s")
    
    return {
        'best_acc': best_acc,
        'final_acc': test_acc,
        'params': n_params,
        'history': history,
    }


def mixup_data(x, y, alpha=0.2):
    """Apply Mixup augmentation."""
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute Mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_enhanced_deep_stcm(
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    n_clauses: int,
    device: str,
    verbose: bool = True,
) -> Dict[str, float]:
    """Train Deep STCM with enhanced training (Mixup + Label Smoothing)."""
    print("\n" + "="*70)
    print("ENHANCED: Deep STCM with Mixup + Label Smoothing")
    print("="*70)
    
    torch.manual_seed(42)
    
    model = DeepTMNetwork(
        input_dim=784,
        hidden_dims=[256, 128],
        n_classes=10,
        n_clauses=n_clauses,
        dropout=0.1,
        tau=0.5,
        clause_dropout=0.1,
        layer_cls=OptimizedSTCM,
        layer_operator="capacity",
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training settings
    label_smoothing = 0.1
    mixup_alpha = 0.2
    
    best_acc = 0.0
    history = []
    
    for epoch in range(epochs):
        start = time.time()
        model.train()
        total_loss = total_correct = total_samples = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            
            # Apply Mixup
            mixed_data, target_a, target_b, lam = mixup_data(data, target, mixup_alpha)
            
            optimizer.zero_grad()
            logits = model(mixed_data, use_ste=True)
            if isinstance(logits, tuple):
                logits = logits[0]
            
            # Mixup loss with label smoothing
            criterion = lambda pred, t: F.cross_entropy(pred, t, label_smoothing=label_smoothing)
            loss = mixup_criterion(criterion, logits, target_a, target_b, lam)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * target.size(0)
            # For accuracy, use original targets
            with torch.no_grad():
                orig_logits = model(data, use_ste=False)
                if isinstance(orig_logits, tuple):
                    orig_logits = orig_logits[0]
                total_correct += (orig_logits.argmax(1) == target).sum().item()
            total_samples += target.size(0)
        
        scheduler.step()
        
        test_acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, test_acc)
        elapsed = time.time() - start
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': total_loss / total_samples,
            'train_acc': total_correct / total_samples,
            'test_acc': test_acc,
            'best_acc': best_acc,
            'time': elapsed,
        })
        
        if verbose:
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"loss={total_loss/total_samples:.4f}, "
                  f"train={total_correct/total_samples:.4f}, "
                  f"test={test_acc:.4f}, "
                  f"best={best_acc:.4f}, "
                  f"time={elapsed:.1f}s")
    
    return {
        'best_acc': best_acc,
        'final_acc': test_acc,
        'params': n_params,
        'history': history,
    }


def main():
    parser = argparse.ArgumentParser(description="Deep STCM MNIST Benchmark")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--n-clauses", type=int, default=500, help="Clauses per layer")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--baseline-only", action="store_true", help="Only run baseline")
    parser.add_argument("--enhanced-only", action="store_true", help="Only run enhanced")
    
    args = parser.parse_args()
    
    print("="*70)
    print("DEEP STCM MNIST BENCHMARK")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Clauses per layer: {args.n_clauses}")
    print(f"  Device: {args.device}")
    
    print("\nLoading MNIST...")
    train_loader, test_loader = load_mnist(args.batch_size)
    print(f"  Train: 60,000 samples")
    print(f"  Test: 10,000 samples")
    
    results = {}
    
    if not args.enhanced_only:
        results['baseline'] = train_baseline_deep_stcm(
            train_loader, test_loader, args.epochs, args.n_clauses, args.device
        )
    
    if not args.baseline_only:
        results['enhanced'] = train_enhanced_deep_stcm(
            train_loader, test_loader, args.epochs, args.n_clauses, args.device
        )
    
    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    for name, r in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Best Test Accuracy: {r['best_acc']*100:.2f}%")
        print(f"  Final Test Accuracy: {r['final_acc']*100:.2f}%")
        print(f"  Parameters: {r['params']:,}")
    
    if 'baseline' in results and 'enhanced' in results:
        diff = results['enhanced']['best_acc'] - results['baseline']['best_acc']
        print(f"\n📊 IMPROVEMENT: {diff*100:+.2f}% (Enhanced vs Baseline)")
        if diff > 0:
            print("✅ Enhanced training outperforms baseline!")
        elif diff < 0:
            print("⚠️  Baseline performed better (try tuning hyperparameters)")
        else:
            print("➡️  Performance is equivalent")
    
    print("\n" + "="*70)
    
    return results


if __name__ == "__main__":
    main()

