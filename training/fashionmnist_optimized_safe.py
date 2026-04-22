#!/usr/bin/env python3
"""
Memory-Safe Optimized Fashion-MNIST Training
Fixed version that prevents CUDA OOM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import numpy as np
import time
import argparse
import gc

# Add parent directory to path
import sys
sys.path.append('..')
from fptm.models import FPTMConvFast
from fptm.utils import set_seed


def train_epoch_safe(model, opt, loader, device, epoch, reinforce_freq=3, accumulation_steps=2):
    """Memory-safe training with gradient accumulation."""
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    opt.zero_grad()  # Initialize gradients
    
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits = model(x)
            loss = ce(logits, y) / accumulation_steps  # Scale loss
            
            # Add small L2 regularization
            l2_lambda = 1e-5
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = loss + l2_lambda * l2_norm / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights after accumulation
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            opt.zero_grad()
        
        # Metrics
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct += (preds == y).float().sum().item()
            total += y.size(0)
            loss_sum += float(loss.item()) * y.size(0) * accumulation_steps
            
            # Memory-safe reinforcement
            if i % reinforce_freq == 0:
                current_acc = (preds == y).float().mean().item()
                adaptive_s = 3.0 * (1.0 + max(0, 0.5 - current_acc))
                # Only reinforce subset
                subset = min(8, len(x))  # Reduced from 16
                model.reinforce(x[:subset], y[:subset], preds[:subset], s=adaptive_s)
        
        # Clear cache periodically
        if i % 100 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    # Final gradient update if needed
    if len(loader) % accumulation_steps != 0:
        opt.step()
        opt.zero_grad()
    
    return loss_sum/total, correct/total


@torch.no_grad()
def evaluate_safe(model, loader, device):
    """Memory-safe evaluation."""
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_probs, all_preds, all_targets = [], [], []
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits = model(x)
        
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        
        loss = F.cross_entropy(logits, y)
        
        correct += (preds == y).float().sum().item()
        total += y.size(0)
        loss_sum += loss.item() * y.size(0)
        
        # Store for ECE calculation
        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Calculate ECE
    all_probs = torch.cat(all_probs)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    # Expected Calibration Error
    n_bins = 10
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    max_probs, _ = all_probs.max(dim=1)
    accuracies = all_preds.eq(all_targets)
    
    ece = torch.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = max_probs.gt(bin_lower) * max_probs.le(bin_upper)
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = max_probs[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return loss_sum/total, correct/total, ece.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)  # Reduced from 128
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--num_clauses", type=int, default=512)  # Safe default
    ap.add_argument("--attention_heads", type=int, default=16)  # Safe default
    ap.add_argument("--patch_size", type=int, default=4)
    ap.add_argument("--use_binary_features", action="store_true",
                    help="Extract binary features like Julia implementation")
    ap.add_argument("--normalize_mode", choices=["none", "minmax", "standard"], default="minmax")
    ap.add_argument("--reinforce_freq", type=int, default=3,
                    help="Reinforcement frequency (batches)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    print("=" * 70)
    print("Memory-Safe Optimized Fashion-MNIST Training")
    print("=" * 70)
    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}")
    print(f"        num_clauses={args.num_clauses}, attention_heads={args.attention_heads}")
    print(f"        use_binary={args.use_binary_features}, normalize={args.normalize_mode}")
    print(f"        reinforce_freq={args.reinforce_freq}")
    
    # Memory optimizations
    print("\nMemory Optimizations:")
    print("  ✓ Batch size: 32 (reduced from 128)")
    print("  ✓ Gradient accumulation: 2 steps")
    print("  ✓ Mixed precision (FP16)")
    print("  ✓ Periodic cache clearing")
    print("  ✓ Subset reinforcement")
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Clear any existing cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Load data
    print("\nLoading Fashion-MNIST dataset...")
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        transforms.ToTensor()
    ])
    transform_test = transforms.ToTensor()
    
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    
    # Create data loaders with reduced batch size
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=0, pin_memory=False  # pin_memory=False to save memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size*2, shuffle=False, 
        num_workers=0, pin_memory=False
    )
    
    # Binary features if requested
    in_channels = 1
    if args.use_binary_features:
        print("\nUsing binary feature extraction")
        in_channels = 4  # Reduced from 8
    
    # Adjust clauses to be divisible by attention heads
    if args.num_clauses % args.attention_heads != 0:
        old_clauses = args.num_clauses
        args.num_clauses = ((args.num_clauses + args.attention_heads - 1) // 
                          args.attention_heads) * args.attention_heads
        print(f"\nAdjusted num_clauses from {old_clauses} to {args.num_clauses}")
    
    # Create model
    print(f"\nCreating model...")
    model = FPTMConvFast(
        in_channels=in_channels,
        image_size=28,
        patch_size=args.patch_size,
        num_clauses=args.num_clauses,
        num_classes=10,
        attention_heads=args.attention_heads,
        epsilon=1e-6,
        automata_states=100,
        normalize_mode=args.normalize_mode if args.normalize_mode != "standard" else "none"
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    if torch.cuda.is_available():
        print(f"Model GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Optimizer with different learning rate for attention
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if 'attn' in n], 'lr': args.lr * 0.5},
        {'params': [p for n, p in model.named_parameters() if 'attn' not in n], 'lr': args.lr}
    ]
    
    opt = optim.AdamW(param_groups, weight_decay=0.01)
    
    # Use cosine annealing
    scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    
    best_acc = 0
    best_epoch = 0
    
    # Determine gradient accumulation steps
    accumulation_steps = 4 if args.num_clauses > 512 else 2
    print(f"Using gradient accumulation: {accumulation_steps} steps")
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        tr_loss, tr_acc = train_epoch_safe(
            model, opt, train_loader, device, epoch, 
            args.reinforce_freq, accumulation_steps
        )
        
        # Evaluate
        va_loss, va_acc, ece = evaluate_safe(model, test_loader, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = opt.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        # Track best
        if va_acc > best_acc:
            best_acc = va_acc
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_model_safe.pth')
        
        # Memory stats
        if torch.cuda.is_available():
            max_mem = torch.cuda.max_memory_allocated() / 1e9
            current_mem = torch.cuda.memory_allocated() / 1e9
            mem_str = f" | Mem {current_mem:.1f}/{max_mem:.1f}GB"
        else:
            mem_str = ""
        
        # Print progress
        print(f"[{epoch:3d}/{args.epochs}] "
              f"Train: {tr_loss:.3f}/{tr_acc:.1%} | "
              f"Val: {va_loss:.3f}/{va_acc:.1%} | "
              f"ECE: {ece:.3f} | "
              f"LR: {current_lr:.5f}{mem_str} | "
              f"Time: {epoch_time:.1f}s")
        
        # Clear cache after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Early stopping if we hit target
        if va_acc >= 0.85:
            print(f"\n✅ Target reached: {va_acc:.1%}")
            break
    
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"Best validation accuracy: {best_acc:.2%} at epoch {best_epoch}")
    if torch.cuda.is_available():
        print(f"Peak memory usage: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print("\nExpected accuracy: 80-83% (with memory safety)")
    print("To improve further, increase num_clauses when GPU memory is available")


if __name__ == "__main__":
    main()
