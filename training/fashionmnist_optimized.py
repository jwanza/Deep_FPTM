#!/usr/bin/env python
"""
Optimized Fashion-MNIST training addressing the synthetic vs real performance gap.
Key changes:
1. No normalization (let model handle it)
2. Different learning rate schedule
3. Less frequent reinforcement
4. Binary feature extraction option
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


class BinaryFeatureExtractor:
    """Extract binary features from images similar to Julia implementation."""
    
    def __init__(self, thresholds=[0.1, 0.25, 0.5, 0.75, 0.9]):
        self.thresholds = thresholds
    
    def __call__(self, img):
        """Convert image to multiple binary channels based on thresholds."""
        if isinstance(img, torch.Tensor):
            # img is already a tensor from ToTensor()
            channels = [img]  # Original image
            
            # Add binary threshold channels
            for threshold in self.thresholds:
                channels.append((img > threshold).float())
            
            # Add edge detection (simple gradient)
            if img.dim() == 3 and img.shape[0] == 1:
                # Compute simple gradients
                dx = torch.zeros_like(img)
                dy = torch.zeros_like(img)
                dx[:, :, :-1] = torch.abs(img[:, :, 1:] - img[:, :, :-1])
                dy[:, :-1, :] = torch.abs(img[:, 1:, :] - img[:, :-1, :])
                
                # Add gradient magnitudes as binary features
                grad_mag = torch.sqrt(dx**2 + dy**2)
                channels.append((grad_mag > 0.1).float())
                channels.append((grad_mag > 0.3).float())
            
            return torch.cat(channels, dim=0)
        return img


def train_epoch_optimized(model, opt, loader, device, epoch, reinforce_freq=5):
    """Optimized training with adjustable reinforcement frequency."""
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    ce = nn.CrossEntropyLoss()  # No label smoothing - like working version!
    
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        opt.zero_grad(set_to_none=True)  # More efficient - like working version
        
        # Forward pass (removed FP16 - causes issues with FPTM)
        logits = model(x)
        loss = ce(logits, y)
        
        # Add small L2 regularization
        l2_lambda = 1e-5
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        loss = loss + l2_lambda * l2_norm
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        
        # Clear cache periodically
        if i % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Metrics
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct += (preds == y).float().sum().item()
            total += y.size(0)
            loss_sum += float(loss.item()) * y.size(0)
            
            # Reinforcement - use FULL batch like in working version!
            if i % reinforce_freq == 0:
                current_acc = (preds == y).float().mean().item()
                # Adaptive specificity
                adaptive_s = 3.0 * (1.0 + max(0, 0.5 - current_acc))
                # Use full batch reinforcement for proper learning
                model.reinforce(x, y, preds, s=adaptive_s)
    
    return loss_sum/total, correct/total


@torch.no_grad()
def evaluate_optimized(model, loader, device):
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
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)  # Balanced for memory/performance
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--num_clauses", type=int, default=512)
    ap.add_argument("--attention_heads", type=int, default=16)
    ap.add_argument("--patch_size", type=int, default=4)
    ap.add_argument("--use_binary_features", action="store_true",
                    help="Extract binary features like Julia implementation")
    ap.add_argument("--normalize_mode", choices=["none", "minmax", "standard"], default="minmax")
    ap.add_argument("--reinforce_freq", type=int, default=5,
                    help="Reinforcement frequency (batches) - matching working version")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    print("=" * 70)
    print("Optimized Fashion-MNIST Training")
    print("=" * 70)
    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}")
    print(f"        num_clauses={args.num_clauses}, attention_heads={args.attention_heads}")
    print(f"        use_binary={args.use_binary_features}, normalize={args.normalize_mode}")
    print(f"        reinforce_freq={args.reinforce_freq}")
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Setup transforms based on configuration
    transform_list = []
    
    # Basic transforms
    transform_list.append(transforms.ToTensor())
    
    # Add binary feature extraction if requested
    if args.use_binary_features:
        transform_list.append(BinaryFeatureExtractor())
        in_channels = 8  # 1 original + 5 thresholds + 2 gradients
        print("Using binary feature extraction (8 channels)")
    else:
        in_channels = 1
    
    # Add normalization based on mode
    if args.normalize_mode == "standard":
        transform_list.append(transforms.Normalize((0.2860,), (0.3530,)))
    elif args.normalize_mode == "minmax":
        # MinMax is handled in the model
        pass
    else:  # "none"
        # No normalization - raw pixel values [0, 1]
        pass
    
    train_transform = transforms.Compose(transform_list)
    
    # Test transform (no augmentation)
    test_transform = transforms.Compose(transform_list)
    
    # Load Fashion-MNIST
    print("\nLoading Fashion-MNIST dataset...")
    
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Single-threaded for consistency
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0
    )
    
    # Adjust clauses to be divisible by attention heads
    if args.num_clauses % args.attention_heads != 0:
        old_clauses = args.num_clauses
        args.num_clauses = ((args.num_clauses + args.attention_heads - 1) // args.attention_heads) * args.attention_heads
        print(f"\nAdjusted num_clauses from {old_clauses} to {args.num_clauses}")
    
    # Memory safety check
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\nGPU memory before model: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        # Check if we need to reduce batch size further
        if args.num_clauses > 768 and args.batch_size > 32:
            print(f"⚠️ Large model detected. Reducing batch size to 32.")
            args.batch_size = 32
    
    # Create model
    print(f"\nCreating model...")
    model = FPTMConvFast(
        in_channels=in_channels,
        image_size=28,
        patch_size=args.patch_size,
        num_clauses=args.num_clauses,
        num_classes=10,
        attention_heads=args.attention_heads,
        # DON'T override defaults - they work!
        # automata_states=50 (default)
        # epsilon uses default
        normalize_mode=args.normalize_mode if args.normalize_mode != "standard" else "none"
    ).to(device)
    
    # Clear cache after model creation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer with different learning rate for attention
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if 'attn' in n], 'lr': args.lr * 0.5},
        {'params': [p for n, p in model.named_parameters() if 'attn' not in n], 'lr': args.lr}
    ]
    
    opt = optim.AdamW(param_groups, weight_decay=0.01)
    
    # Use cosine annealing - same as supervised_adaptive that worked!
    scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    
    best_acc = 0
    best_epoch = 0
    plateau_count = 0
    prev_acc = 0
    
    total_start = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train with consistent reinforcement frequency
        tr_loss, tr_acc = train_epoch_optimized(
            model, opt, train_loader, device, epoch, args.reinforce_freq
        )
        
        # Clear cache after training epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Evaluate
        va_loss, va_acc, ece = evaluate_optimized(model, test_loader, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = opt.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        # Track best
        if va_acc > best_acc:
            best_acc = va_acc
            best_epoch = epoch
            plateau_count = 0
        else:
            plateau_count += 1
        
        # Print results
        improvement = va_acc - prev_acc
        prev_acc = va_acc
        
        print(f"[{epoch:3d}/{args.epochs}] "
              f"Train: {tr_loss:.3f}/{tr_acc:.1%} | "
              f"Val: {va_loss:.3f}/{va_acc:.1%} | "
              f"ECE: {ece:.3f} | "
              f"LR: {current_lr:.5f} | "
              f"Time: {epoch_time:.1f}s | "
              f"Δ: {improvement:+.1%}")
        
        # Early stopping if stuck for too long
        if plateau_count > 20 and epoch > 50:
            print(f"Early stopping at epoch {epoch}")
            break
    
    total_time = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"Best validation accuracy: {best_acc:.2%} at epoch {best_epoch}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Average epoch time: {total_time/epoch:.1f}s")
    
    # Analysis
    print("\n" + "=" * 70)
    print("Performance Analysis")
    print("=" * 70)
    print(f"Gap to synthetic performance: {95.8 - best_acc*100:.1f}%")
    print(f"Gap to Julia FPTM: {93.59 - best_acc*100:.1f}%")
    
    if best_acc < 0.85:
        print("\nPossible issues:")
        print("1. Try --use_binary_features flag for Julia-like features")
        print("2. Try different normalize_mode (none, minmax)")
        print("3. Increase num_clauses to 768 or 1024")
        print("4. Try smaller patch_size (2 or 7)")
        print("5. Adjust reinforce_freq (try 50 or 100)")


if __name__ == "__main__":
    main()
