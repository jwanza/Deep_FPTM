#!/usr/bin/env python
"""
Fixed Fashion-MNIST training using EXACT same approach as supervised_adaptive
but with real Fashion-MNIST data instead of synthetic.
"""
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from fptm.models import FPTMConvFast
from fptm.utils import set_seed
from fptm.heads import compute_ece


def train_one_epoch(model, opt, loader, device, scheduler=None, reinforce_every=3):
    """EXACT copy from supervised_adaptive that worked!"""
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    ce = nn.CrossEntropyLoss()  # No label smoothing first
    
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = ce(logits, y)
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        
        # Reinforcement and metrics
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            
            # CRITICAL: Reinforcement every 3 batches like before!
            if i % reinforce_every == 0:
                current_acc = (preds == y).float().mean().item()
                adaptive_s = 3.0 * (1.0 + max(0, 0.5 - current_acc))
                model.reinforce(x, y, preds, s=adaptive_s)
            
            correct += (preds == y).float().sum().item()
            total += y.size(0)
            loss_sum += float(loss.item()) * y.size(0)
    
    return loss_sum/total, correct/total


@torch.no_grad()
def evaluate(model, loader, device):
    """Fast evaluation."""
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    ce = nn.CrossEntropyLoss()
    all_logits, all_labels = [], []
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        preds = logits.argmax(dim=-1)
        
        correct += (preds == y).float().sum().item()
        total += y.size(0)
        loss_sum += float(loss.item()) * y.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())
    
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    ece = compute_ece(logits, labels)
    
    return loss_sum/total, correct/total, ece


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--num_clauses", type=int, default=256)
    ap.add_argument("--attention_heads", type=int, default=8)
    ap.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    print("=" * 60)
    print("Fashion-MNIST with Adaptive Training (Fixed)")
    print("Using EXACT same approach that got 95.8% on synthetic")
    print("=" * 60)
    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}")
    print(f"        num_clauses={args.num_clauses}, attention_heads={args.attention_heads}")
    print(f"        lr={args.lr}, scheduler={args.scheduler}")
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # NO NORMALIZATION - just like synthetic worked
    transform = transforms.Compose([
        transforms.ToTensor(),
        # NO normalization - raw [0,1] values
    ])
    
    # Load Fashion-MNIST
    print("\nLoading Fashion-MNIST dataset...")
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model - SAME as supervised_adaptive
    model = FPTMConvFast(
        in_channels=1,
        image_size=28,
        patch_size=4,  # Same as before
        num_clauses=args.num_clauses,
        num_classes=10,
        attention_heads=args.attention_heads,
        normalize_mode="minmax"  # IMPORTANT: Same as synthetic!
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    
    # EXACT same optimizer as supervised_adaptive
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # EXACT same scheduler
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    else:
        scheduler = None
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    best_acc = 0
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        # Train
        tr_loss, tr_acc = train_one_epoch(model, opt, train_loader, device, scheduler)
        
        # Step scheduler
        if scheduler:
            scheduler.step()
        
        # Evaluate
        va_loss, va_acc, ece = evaluate(model, test_loader, device)
        
        # Track best
        if va_acc > best_acc:
            best_acc = va_acc
            best_epoch = epoch
        
        # Print progress
        current_lr = opt.param_groups[0]['lr']
        print(f"[{epoch:3d}/{args.epochs}] "
              f"Train: {tr_loss:.3f}/{tr_acc:.1%} | "
              f"Val: {va_loss:.3f}/{va_acc:.1%} | "
              f"ECE: {ece:.3f} | "
              f"LR: {current_lr:.5f} | "
              f"Best: {best_acc:.1%}")
        
        # Early stopping if really bad
        if epoch > 10 and va_acc < 0.15:
            print("Model not learning, stopping early")
            break
    
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Best validation accuracy: {best_acc:.2%} at epoch {best_epoch}")
    print(f"Comparison:")
    print(f"  - Synthetic data: 95.8%")
    print(f"  - Real Fashion-MNIST: {best_acc:.2%}")
    print(f"  - Gap: {95.8 - best_acc*100:.1f}%")
    
    if best_acc < 0.80:
        print("\nDiagnosis:")
        print("The model works great on synthetic shapes but struggles with real textures.")
        print("This confirms that FPTM prefers binary/discrete features over continuous grayscale.")
        print("Consider using binary feature extraction for better results.")


if __name__ == "__main__":
    main()
