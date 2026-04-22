#!/usr/bin/env python3
"""
Real Fashion-MNIST training with Binary FPTM - combining best of both approaches.
Uses the stable training from fashionmnist_real.py with BinaryFPTM from step4_binary_features.py
Expected: ~80-85% accuracy
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
import numpy as np

from fptm.models import FPTMConvFast
from fptm.utils import set_seed
from fptm.heads import compute_ece


def extract_binary_features(x: torch.Tensor, num_thresholds: int = 8) -> torch.Tensor:
    """
    Convert continuous images to binary features using adaptive thresholding.
    Similar to the successful Julia implementation.
    """
    B, C, H, W = x.shape
    
    # Calculate quantiles for adaptive thresholding
    x_flat = x.view(B, -1)
    quantiles = torch.quantile(
        x_flat, 
        torch.linspace(0.1, 0.9, num_thresholds).to(x.device), 
        dim=1
    )
    
    # Create binary features for each threshold
    binary_features = []
    for i in range(num_thresholds):
        threshold = quantiles[i].view(B, 1, 1, 1)
        binary = (x > threshold).float()
        binary_features.append(binary)
    
    # Stack all binary features
    return torch.cat(binary_features, dim=1)  # (B, C*num_thresholds, H, W)


class BinaryFPTM(nn.Module):
    """
    FPTM that works on binary features for better discrete pattern learning.
    Combines binary feature extraction with the working FPTM configuration.
    """
    
    def __init__(self, num_clauses: int = 1024, num_classes: int = 10, 
                 attention_heads: int = 32, num_thresholds: int = 8,
                 patch_size: int = 4, use_channel_mixing: bool = True):
        super().__init__()
        
        self.num_thresholds = num_thresholds
        self.use_channel_mixing = use_channel_mixing
        
        # Determine input channels
        if use_channel_mixing:
            # Mix multiple binary channels down to fewer channels for efficiency
            self.channel_mixer = nn.Conv2d(num_thresholds, 1, kernel_size=1)
            in_channels = 1
        else:
            # Use all binary channels directly
            in_channels = num_thresholds
            self.channel_mixer = None
        
        # Main FPTM model - using defaults that work!
        self.fptm = FPTMConvFast(
            in_channels=in_channels,
            image_size=28,
            patch_size=patch_size,
            num_clauses=num_clauses,
            attention_heads=attention_heads,
            num_classes=num_classes,
            normalize_mode="none"  # Binary features don't need normalization
            # NOT setting automata_states or epsilon - use defaults!
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to binary features
        binary_x = extract_binary_features(x, self.num_thresholds)
        
        # Optionally mix channels
        if self.use_channel_mixing:
            binary_x = self.channel_mixer(binary_x)
        
        return self.fptm(binary_x)
    
    @torch.no_grad()
    def reinforce(self, x: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor, s: float = 3.0):
        """Reinforcement with binary features"""
        binary_x = extract_binary_features(x, self.num_thresholds)
        
        if self.use_channel_mixing:
            binary_x = self.channel_mixer(binary_x)
        
        self.fptm.reinforce(binary_x, y_true, y_pred, s=s)


def train_one_epoch(model, opt, loader, device, scheduler=None, reinforce_every=5):
    """Train for one epoch - same as fashionmnist_real.py"""
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    ce = nn.CrossEntropyLoss()  # No label smoothing - proven to work
    
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        opt.zero_grad(set_to_none=True)  # Efficient clearing
        logits = model(x)
        loss = ce(logits, y)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        
        # Update scheduler if using OneCycle
        if scheduler and hasattr(scheduler, 'step') and hasattr(scheduler, 'total_steps'):
            scheduler.step()
        
        # Metrics and reinforcement
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            
            # Reinforcement learning - FULL BATCH (proven to work)
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
    """Evaluate model with ECE calculation"""
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


def analyze_binary_performance(model, loader, device):
    """Analyze which classes benefit most from binary features"""
    model.eval()
    class_correct = torch.zeros(10)
    class_total = torch.zeros(10)
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            
            for c in range(10):
                mask = (y == c)
                class_correct[c] += (preds[mask] == y[mask]).float().sum().cpu()
                class_total[c] += mask.sum().cpu()
    
    class_acc = (class_correct / class_total) * 100
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    print("\n" + "=" * 60)
    print("Per-Class Performance with Binary Features:")
    print("=" * 60)
    for i, (name, acc) in enumerate(zip(class_names, class_acc)):
        print(f"{name:12s}: {acc:.1f}%")
    print("=" * 60)
    print(f"Average: {class_acc.mean():.1f}%")
    print(f"Best:    {class_names[class_acc.argmax()]} ({class_acc.max():.1f}%)")
    print(f"Worst:   {class_names[class_acc.argmin()]} ({class_acc.min():.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--patch_size", type=int, default=4)
    ap.add_argument("--num_clauses", type=int, default=1024)
    ap.add_argument("--attention_heads", type=int, default=32)
    ap.add_argument("--num_thresholds", type=int, default=8,
                    help="Number of binary thresholds (8 works well)")
    ap.add_argument("--no_channel_mixing", action="store_true",
                    help="Use all binary channels directly without mixing")
    ap.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--analyze", action="store_true",
                    help="Perform per-class analysis at the end")
    args = ap.parse_args()
    
    print("=" * 60)
    print("Binary FPTM Training on Real Fashion-MNIST")
    print("=" * 60)
    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}")
    print(f"        num_clauses={args.num_clauses}, attention_heads={args.attention_heads}")
    print(f"        lr={args.lr}, scheduler={args.scheduler}")
    print(f"        num_thresholds={args.num_thresholds}, mix_channels={not args.no_channel_mixing}")
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Adjust clauses to be divisible by attention heads
    if args.num_clauses % args.attention_heads != 0:
        old_clauses = args.num_clauses
        args.num_clauses = ((args.num_clauses + args.attention_heads - 1) // 
                           args.attention_heads) * args.attention_heads
        print(f"\nAdjusted num_clauses from {old_clauses} to {args.num_clauses}")
    
    # Load Fashion-MNIST
    print("\nLoading Fashion-MNIST dataset...")
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        transforms.ToTensor()
    ])
    test_transform = transforms.ToTensor()
    
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )
    
    # Create Binary FPTM model
    print("\nCreating Binary FPTM model...")
    model = BinaryFPTM(
        num_clauses=args.num_clauses,
        num_classes=10,
        attention_heads=args.attention_heads,
        num_thresholds=args.num_thresholds,
        patch_size=args.patch_size,
        use_channel_mixing=not args.no_channel_mixing
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    
    # Explain binary features
    print("\n" + "=" * 60)
    print("Binary Feature Extraction:")
    print("=" * 60)
    print(f"  • Converting grayscale (0-255) → {args.num_thresholds} binary channels")
    print(f"  • Using adaptive quantile thresholds (10%, 20%, ..., 90%)")
    print(f"  • {'Mixing channels with 1x1 conv' if not args.no_channel_mixing else 'Using all channels directly'}")
    print(f"  • Result: Discrete features that match FPTM's Tsetlin nature")
    print("=" * 60)
    
    # Setup optimizer and scheduler
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    else:
        scheduler = None
    
    # Training loop
    print("\nStarting Training")
    print("=" * 60)
    
    best_acc = 0
    best_epoch = 0
    total_start = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Train
        epoch_start = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, opt, train_loader, device, scheduler
        )
        
        # Step scheduler
        if scheduler and isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        
        # Evaluate
        va_loss, va_acc, ece = evaluate(model, test_loader, device)
        epoch_time = time.time() - epoch_start
        
        # Track best
        if va_acc > best_acc:
            best_acc = va_acc
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_binary_real_model.pth')
        
        # Print progress
        current_lr = opt.param_groups[0]['lr']
        print(f"[{epoch:3d}/{args.epochs}] "
              f"Train: {tr_loss:.3f}/{tr_acc:.1%} | "
              f"Val: {va_loss:.3f}/{va_acc:.1%} | "
              f"ECE: {ece:.3f} | "
              f"LR: {current_lr:.5f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Early stopping if we hit target
        if va_acc >= 0.85:
            print(f"\n✅ Target reached: {va_acc:.1%}")
            break
        
        # Early stopping on plateau
        if epoch > 20 and va_acc < best_acc - 0.05:
            print("Early stopping triggered")
            break
    
    # Final results
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Best validation accuracy: {best_acc:.2%} at epoch {best_epoch}")
    print(f"Total training time: {total_time:.1f}s")
    print(f"Average epoch time: {total_time/epoch:.1f}s")
    
    # Per-class analysis if requested
    if args.analyze:
        analyze_binary_performance(model, test_loader, device)
    
    # Compare with other approaches
    print("\n" + "=" * 60)
    print("Performance Comparison:")
    print("=" * 60)
    print(f"Binary FPTM (this):     {best_acc:.2%}")
    print(f"Regular FPTM (real):    78.8% (from fashionmnist_real.py)")
    print(f"Binary FPTM (step4):    76.9% (from step4_binary_features.py)")
    print(f"Expected with binary:   80-85%")
    print("\nKey advantages of binary features:")
    print("  • Matches FPTM's discrete Tsetlin automata")
    print("  • More robust to noise and variations")
    print("  • Better edge and pattern detection")
    print("  • Similar to Julia's successful approach (92-93%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
