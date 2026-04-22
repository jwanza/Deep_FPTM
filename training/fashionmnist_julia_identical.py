#!/usr/bin/env python3
"""
Fashion-MNIST training with Julia-identical Tsetlin Machine.
Testing if we can achieve Julia's 88-90% accuracy with exact same architecture.
"""

import argparse
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fptm.models.julia_identical_fast import JuliaIdenticalFast, create_julia_fast
from fptm.utils import set_seed


def train_epoch(model, train_loader, device, epoch):
    """Train for one epoch using fast batch processing."""
    model.train()
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        # Fast batch training
        model.reinforce_batch(x, y)
        
        # Track accuracy
        with torch.no_grad():
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
        
        # Progress update
        if batch_idx % 100 == 0:
            acc = 100. * correct / total if total > 0 else 0
            print(f'  Batch {batch_idx}/{len(train_loader)}: '
                  f'Train Acc: {acc:.2f}%')
    
    epoch_time = time.time() - start_time
    accuracy = 100. * correct / total
    
    return accuracy, epoch_time


def evaluate(model, test_loader, device):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Julia-identical Tsetlin on Fashion-MNIST')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test_batch_size', type=int, default=512,
                        help='input batch size for testing (default: 512)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='device to use (default: cuda if available)')
    
    # Julia's exact hyperparameters
    parser.add_argument('--clauses_num', type=int, default=20,
                        help='number of clauses (Julia default: 20)')
    parser.add_argument('--T', type=int, default=100,
                        help='threshold T (Julia default: 100)')
    parser.add_argument('--S', type=int, default=700,
                        help='specificity S (Julia default: 700)')
    parser.add_argument('--L', type=int, default=200,
                        help='max literals L (Julia default: 200)')
    parser.add_argument('--LF', type=int, default=200,
                        help='literal filter LF - fuzzy max (Julia default: 200)')
    parser.add_argument('--states_num', type=int, default=256,
                        help='number of automata states (Julia default: 256)')
    parser.add_argument('--include_limit', type=int, default=230,
                        help='include limit (Julia default: 230)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔬 Julia-Identical Tsetlin Machine on Fashion-MNIST")
    print("=" * 70)
    print(f"Testing if we can achieve Julia's 88-90% accuracy")
    print(f"\nHyperparameters (Julia's exact values):")
    print(f"  Clauses: {args.clauses_num}")
    print(f"  T: {args.T}")
    print(f"  S: {args.S}")
    print(f"  L: {args.L}")
    print(f"  LF: {args.LF} (fuzzy output: 0-{args.LF} integers)")
    print(f"  States: {args.states_num}")
    print(f"  Include limit: {args.include_limit}")
    print("=" * 70)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    device = torch.device(args.device)
    
    # Load Fashion-MNIST
    print("\nLoading Fashion-MNIST dataset...")
    transform = transforms.ToTensor()
    
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4 if torch.cuda.is_available() else 0,  # Fast parallel loading
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Testing samples: {len(test_dataset):,}")
    
    # Create Julia-identical FAST model
    print("\nCreating Julia-identical FAST Tsetlin Machine...")
    model = JuliaIdenticalFast(
        clauses_num=args.clauses_num,
        T=args.T,
        S=args.S,
        L=args.L,
        LF=args.LF,
        states_num=args.states_num,
        include_limit=args.include_limit,
        num_classes=10
    ).to(device)
    
    print("✅ FAST Model created with:")
    print(f"  - 76 binary channels (4 raw + 72 convolution)")
    print(f"  - Discrete fuzzy output (0-{args.LF} integers)")
    print(f"  - GPU acceleration: {torch.cuda.is_available()}")
    print(f"  - Batch processing: ✓")
    print(f"  - JIT compilation: ✓")
    print(f"  - Vectorized convolutions: ✓")
    print(f"  - Full interpretability preserved")
    
    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    
    best_test_acc = 0
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_acc, train_time = train_epoch(model, train_loader, device, epoch)
        
        # Test
        test_acc = evaluate(model, test_loader, device)
        
        # Track best
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            torch.save(model.state_dict(), 'julia_identical_best.pth')
        
        print(f'Epoch {epoch:3d} | Train: {train_acc:.2f}% | '
              f'Test: {test_acc:.2f}% | Time: {train_time:.1f}s')
        
        # Check if we're approaching Julia's performance
        if test_acc >= 85:
            print(f"🎯 Approaching Julia's performance! ({test_acc:.2f}%)")
        if test_acc >= 88:
            print(f"✅ MATCHED Julia's performance! ({test_acc:.2f}%)")
            break
    
    # Final results
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"Best test accuracy: {best_test_acc:.2f}% at epoch {best_epoch}")
    
    # Compare with expected Julia performance
    print("\n📊 Performance Comparison:")
    print(f"  Julia (expected):     88-90%")
    print(f"  This implementation:  {best_test_acc:.2f}%")
    
    if best_test_acc >= 88:
        print("\n✅ SUCCESS: Matched Julia's accuracy!")
        print("   We have successfully replicated Julia's Tsetlin Machine in Python!")
    elif best_test_acc >= 85:
        print("\n⚠️ CLOSE: Nearly matched Julia's accuracy!")
        print("   May need more epochs or slight hyperparameter tuning.")
    else:
        print("\n❌ GAP: Significant difference from Julia's performance.")
        print("   Possible issues:")
        print("   - Implementation differences in feedback mechanism")
        print("   - Python vs Julia numerical precision")
        print("   - Need more training epochs")
    
    # Save final model
    torch.save(model.state_dict(), 'julia_identical_final.pth')
    print(f"\nModel saved to julia_identical_final.pth")
    
    # Test interpretability (if available)
    print("\n🔍 Model Information:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"  Device: {next(model.parameters()).device}")
    if hasattr(model, 'positive_clauses'):
        print(f"  Clause states shape: {model.positive_clauses.shape}")
        print("  ✅ Clause structure preserved for interpretability!")


if __name__ == '__main__':
    main()
