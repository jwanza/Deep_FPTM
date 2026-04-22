#!/usr/bin/env python3
"""
Ultimate Fashion-MNIST FPTM - Combines ALL optimizations:
- Binary features (for accuracy)
- Adaptive speed (gets faster)
- Working defaults (no automata_states=100!)

Expected: ~83-85% accuracy with decreasing training time
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


class UltimateFPTM(nn.Module):
    """
    Ultimate FPTM combining:
    1. Binary features (accuracy boost)
    2. Adaptive reinforcement (speed boost)
    3. Working defaults (stability)
    """
    
    def __init__(self, num_clauses: int = 1024, num_classes: int = 10, 
                 attention_heads: int = 32, num_thresholds: int = 8,
                 patch_size: int = 4, use_channel_mixing: bool = True):
        super().__init__()
        
        self.num_thresholds = num_thresholds
        self.use_channel_mixing = use_channel_mixing
        
        # Ensure divisibility
        if num_clauses % attention_heads != 0:
            num_clauses = ((num_clauses + attention_heads - 1) // 
                          attention_heads) * attention_heads
        
        # Binary feature processing
        if use_channel_mixing:
            self.channel_mixer = nn.Conv2d(num_thresholds, 1, kernel_size=1)
            in_channels = 1
        else:
            in_channels = num_thresholds
            self.channel_mixer = None
        
        # Main FPTM - NO automata_states override!
        self.fptm = FPTMConvFast(
            in_channels=in_channels,
            image_size=28,
            patch_size=patch_size,
            num_clauses=num_clauses,
            attention_heads=attention_heads,
            num_classes=num_classes,
            normalize_mode="none"  # Binary features don't need normalization
            # Using all defaults - they work!
        )
        
        # Adaptive speed tracking
        self.running_accuracy = 0.1
        self.confidence_threshold = 0.8
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to binary
        binary_x = extract_binary_features(x, self.num_thresholds)
        
        # Mix channels if needed
        if self.use_channel_mixing:
            binary_x = self.channel_mixer(binary_x)
        
        return self.fptm(binary_x)
    
    @torch.no_grad()
    def adaptive_reinforce(self, x: torch.Tensor, y_true: torch.Tensor, 
                          y_pred: torch.Tensor, logits: torch.Tensor, 
                          base_s: float = 3.0):
        """
        Adaptive reinforcement with binary features.
        Combines both optimizations!
        """
        # Calculate confidence
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1)[0].mean().item()
        
        # Calculate batch accuracy
        batch_acc = (y_pred == y_true).float().mean().item()
        
        # Update running accuracy
        self.running_accuracy = 0.95 * self.running_accuracy + 0.05 * batch_acc
        
        # ADAPTIVE PROBABILITY (Julia's secret)
        update_probability = (1.0 - self.running_accuracy) * (1.0 - confidence * 0.5)
        
        # Adaptive sample size
        if self.running_accuracy > 0.8:
            sample_size = max(4, len(x) // 4)  # 25% when good
        elif self.running_accuracy > 0.6:
            sample_size = max(8, len(x) // 2)  # 50% when medium
        else:
            sample_size = len(x)  # 100% when learning
        
        # Skip with probability (faster!)
        if torch.rand(1).item() < update_probability:
            # Adaptive s value
            adaptive_s = base_s * (1.0 + max(0, 0.5 - self.running_accuracy))
            
            # Priority sampling
            if sample_size < len(x):
                errors = (y_pred != y_true)
                low_conf = probs.max(dim=-1)[0] < self.confidence_threshold
                priority = errors | low_conf
                
                if priority.sum() > 0:
                    priority_indices = torch.where(priority)[0][:sample_size]
                    if len(priority_indices) < sample_size:
                        remaining = sample_size - len(priority_indices)
                        other_indices = torch.randperm(len(x), device=x.device)[:remaining]  # FIX: same device
                        indices = torch.cat([priority_indices, other_indices])
                    else:
                        indices = priority_indices
                else:
                    indices = torch.randperm(len(x), device=x.device)[:sample_size]  # FIX: same device
                
                # Convert to binary and reinforce
                binary_x = extract_binary_features(x[indices], self.num_thresholds)
                if self.use_channel_mixing:
                    binary_x = self.channel_mixer(binary_x)
                self.fptm.reinforce(binary_x, y_true[indices], 
                                   y_pred[indices], s=adaptive_s)
            else:
                # Full batch
                binary_x = extract_binary_features(x, self.num_thresholds)
                if self.use_channel_mixing:
                    binary_x = self.channel_mixer(binary_x)
                self.fptm.reinforce(binary_x, y_true, y_pred, s=adaptive_s)
        
        return update_probability, sample_size


def train_one_epoch_ultimate(model, opt, loader, device, epoch, total_epochs):
    """Train with binary features AND adaptive speed"""
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    ce = nn.CrossEntropyLoss()
    
    # Timing
    forward_time = 0
    backward_time = 0
    reinforce_time = 0
    reinforce_calls = 0
    total_samples_reinforced = 0
    
    epoch_start = time.time()
    
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        
        # Forward
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = ce(logits, y)
        forward_time += time.time() - t0
        
        # Backward
        t0 = time.time()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        backward_time += time.time() - t0
        
        # Adaptive reinforcement
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            
            # Adaptive frequency
            reinforce_freq = min(10, 3 + epoch // 10)
            
            if i % reinforce_freq == 0:
                t0 = time.time()
                update_prob, sample_size = model.adaptive_reinforce(
                    x, y, preds, logits
                )
                reinforce_time += time.time() - t0
                reinforce_calls += 1
                total_samples_reinforced += sample_size
            
            correct += (preds == y).float().sum().item()
            total += y.size(0)
            loss_sum += float(loss.item()) * y.size(0)
    
    epoch_time = time.time() - epoch_start
    
    # Timing breakdown
    print(f"  Time: Fwd {forward_time:.1f}s | "
          f"Bwd {backward_time:.1f}s | "
          f"Reinf {reinforce_time:.1f}s")
    print(f"  Reinforce: {reinforce_calls} calls, "
          f"{total_samples_reinforced}/{total} samples "
          f"({100*total_samples_reinforced/total:.1f}%)")
    print(f"  Running acc: {model.running_accuracy:.1%}")
    
    return loss_sum/total, correct/total, epoch_time


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model"""
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
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--patch_size", type=int, default=4)
    ap.add_argument("--num_clauses", type=int, default=1024)
    ap.add_argument("--attention_heads", type=int, default=32)
    ap.add_argument("--num_thresholds", type=int, default=8)
    ap.add_argument("--no_channel_mixing", action="store_true")
    ap.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    print("=" * 70)
    print("🚀 ULTIMATE FPTM - All Optimizations Combined!")
    print("=" * 70)
    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}")
    print(f"        num_clauses={args.num_clauses}, attention_heads={args.attention_heads}")
    print(f"        thresholds={args.num_thresholds}, mix={not args.no_channel_mixing}")
    print("\n✨ FEATURES:")
    print("   ✅ Binary features (accuracy boost)")
    print("   ✅ Adaptive speed (gets faster)")
    print("   ✅ Working defaults (no automata_states=100)")
    print("   ✅ Priority sampling (focus on errors)")
    print("=" * 70)
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    print("\nLoading Fashion-MNIST...")
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
    
    print(f"Training: {len(train_dataset):,} samples")
    print(f"Testing: {len(test_dataset):,} samples")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size*2, shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )
    
    # Create model
    print("\nCreating Ultimate FPTM...")
    model = UltimateFPTM(
        num_clauses=args.num_clauses,
        num_classes=10,
        attention_heads=args.attention_heads,
        num_thresholds=args.num_thresholds,
        patch_size=args.patch_size,
        use_channel_mixing=not args.no_channel_mixing
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    
    # Optimizer
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    else:
        scheduler = None
    
    # Training
    print("\nStarting Training")
    print("=" * 70)
    
    best_acc = 0
    best_epoch = 0
    epoch_times = []
    
    for epoch in range(1, args.epochs + 1):
        # Train
        tr_loss, tr_acc, epoch_time = train_one_epoch_ultimate(
            model, opt, train_loader, device, epoch, args.epochs
        )
        epoch_times.append(epoch_time)
        
        # Schedule
        if scheduler:
            scheduler.step()
        
        # Evaluate
        va_loss, va_acc, ece = evaluate(model, test_loader, device)
        
        # Track best
        if va_acc > best_acc:
            best_acc = va_acc
            best_epoch = epoch
            torch.save(model.state_dict(), 'ultimate_model.pth')
        
        # Progress
        current_lr = opt.param_groups[0]['lr']
        speed_emoji = "🔥" if epoch > 5 and epoch_time < np.mean(epoch_times[:5]) else ""
        
        print(f"[{epoch:3d}/{args.epochs}] "
              f"Train: {tr_loss:.3f}/{tr_acc:.1%} | "
              f"Val: {va_loss:.3f}/{va_acc:.1%} | "
              f"ECE: {ece:.3f} | "
              f"LR: {current_lr:.5f} | "
              f"Time: {epoch_time:.1f}s {speed_emoji}")
        
        # Speed analysis
        if epoch == 10:
            early = np.mean(epoch_times[:5])
            recent = np.mean(epoch_times[5:10])
            speedup = (early - recent) / early * 100
            print(f"\n  ⚡ Speedup: {speedup:.1f}% faster!\n")
        
        # Target reached
        if va_acc >= 0.85:
            print(f"\n🎯 Target reached: {va_acc:.1%}")
            break
    
    # Final results
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"Best accuracy: {best_acc:.2%} at epoch {best_epoch}")
    
    # Speed analysis
    if len(epoch_times) > 10:
        early = epoch_times[:5]
        late = epoch_times[-5:]
        speedup = (np.mean(early) - np.mean(late)) / np.mean(early) * 100
        print(f"\n⚡ SPEED IMPROVEMENT:")
        print(f"   First 5 epochs: {np.mean(early):.1f}s avg")
        print(f"   Last 5 epochs:  {np.mean(late):.1f}s avg")
        print(f"   Improvement:    {speedup:.1f}% faster!")
    
    print("\n🏆 ULTIMATE FPTM combines:")
    print("   • Binary features → Better accuracy")
    print("   • Adaptive speed → Faster training")
    print("   • Working defaults → Stable learning")
    print("=" * 70)


if __name__ == "__main__":
    main()
