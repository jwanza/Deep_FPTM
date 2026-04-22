#!/usr/bin/env python3
"""
Fashion-MNIST 90% Target - Optimized approach building on proven 82.5% success
==============================================================================
Strategy: Take what worked (82.5% with 8 thresholds) and enhance it with:
1. More thresholds (16 instead of 8)
2. Simple edge detection (Sobel filters)
3. Longer training (150 epochs)
4. Better learning rate schedule

This is simpler than full Julia implementation but should reach 88-90%
"""
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
import numpy as np

from fptm.models import FPTMConvFast
from fptm.utils import set_seed
from fptm.heads import compute_ece


def extract_enhanced_binary_features(x: torch.Tensor, num_thresholds: int = 16) -> torch.Tensor:
    """
    Enhanced binary features with edge detection
    Simpler than full Julia but more powerful than basic thresholding
    """
    B, C, H, W = x.shape
    device = x.device
    
    # 1. Basic adaptive thresholding (proven to work)
    x_flat = x.view(B, -1)
    quantiles = torch.quantile(
        x_flat,
        torch.linspace(0.05, 0.95, num_thresholds).to(device),
        dim=1
    )
    
    binary_features = []
    for i in range(num_thresholds):
        threshold = quantiles[i].view(B, 1, 1, 1)
        binary = (x > threshold).float()
        binary_features.append(binary)
    
    # 2. Add simple edge detection (Sobel filters)
    # Define Sobel kernels
    sobel_x = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    
    sobel_y = torch.tensor([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    
    # Apply Sobel filters
    edges_x = F.conv2d(x, sobel_x, padding=1)
    edges_y = F.conv2d(x, sobel_y, padding=1)
    edges = torch.sqrt(edges_x**2 + edges_y**2)
    
    # Binarize edges with adaptive thresholds
    edges_flat = edges.view(B, -1)
    edge_quantiles = torch.quantile(
        edges_flat,
        torch.linspace(0.3, 0.9, 4).to(device),  # 4 edge thresholds
        dim=1
    )
    
    for i in range(4):
        threshold = edge_quantiles[i].view(B, 1, 1, 1)
        binary = (edges > threshold).float()
        binary_features.append(binary)
    
    # Stack all features
    return torch.cat(binary_features, dim=1)  # (B, num_thresholds+4, H, W)


class Enhanced90PercentFPTM(nn.Module):
    """
    Enhanced FPTM targeting 90% accuracy
    Builds on proven 82.5% approach with better features
    """
    
    def __init__(self, num_clauses: int = 1536, num_classes: int = 10,
                 attention_heads: int = 32, num_thresholds: int = 16,
                 patch_size: int = 4):
        super().__init__()
        
        self.num_thresholds = num_thresholds
        total_channels = num_thresholds + 4  # Basic + edge features
        
        # Ensure divisibility
        if num_clauses % attention_heads != 0:
            num_clauses = ((num_clauses + attention_heads - 1) // 
                          attention_heads) * attention_heads
        
        # Channel mixer - combine all binary features to 1 channel (proven to work!)
        self.channel_mixer = nn.Conv2d(total_channels, 1, kernel_size=1)
        
        # Main FPTM with increased capacity
        self.fptm = FPTMConvFast(
            in_channels=1,
            image_size=28,
            patch_size=patch_size,
            num_clauses=num_clauses,
            attention_heads=attention_heads,
            num_classes=num_classes,
            normalize_mode="none"  # Binary features
        )
        
        # Tracking for adaptive training
        self.running_accuracy = 0.1
        self.confidence_threshold = 0.85
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract enhanced binary features
        binary_x = extract_enhanced_binary_features(x, self.num_thresholds)
        
        # Mix channels
        mixed = self.channel_mixer(binary_x)
        
        return self.fptm(mixed)
    
    @torch.no_grad()
    def adaptive_reinforce(self, x: torch.Tensor, y_true: torch.Tensor,
                          y_pred: torch.Tensor, logits: torch.Tensor,
                          base_s: float = 3.5):
        """Proven adaptive reinforcement from 82.5% success"""
        # Calculate confidence
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1)[0].mean().item()
        
        # Update running accuracy
        batch_acc = (y_pred == y_true).float().mean().item()
        self.running_accuracy = 0.95 * self.running_accuracy + 0.05 * batch_acc
        
        # Julia-style adaptive probability
        update_probability = (1.0 - self.running_accuracy) * (1.0 - confidence * 0.4)
        
        # Adaptive sample size
        if self.running_accuracy > 0.85:
            sample_size = max(4, len(x) // 10)  # 10% when excellent
        elif self.running_accuracy > 0.75:
            sample_size = max(8, len(x) // 4)  # 25% when good
        elif self.running_accuracy > 0.6:
            sample_size = max(16, len(x) // 2)  # 50% when medium
        else:
            sample_size = len(x)  # 100% when learning
        
        # Skip with increasing probability
        if torch.rand(1).item() < update_probability:
            # Adaptive strength
            adaptive_s = base_s * (1.0 + max(0, 0.6 - self.running_accuracy))
            
            # Priority sampling on errors and low confidence
            if sample_size < len(x):
                errors = (y_pred != y_true)
                low_conf = probs.max(dim=-1)[0] < self.confidence_threshold
                priority = errors | low_conf
                
                if priority.sum() > 0:
                    priority_indices = torch.where(priority)[0][:sample_size]
                    if len(priority_indices) < sample_size:
                        remaining = sample_size - len(priority_indices)
                        other_indices = torch.randperm(len(x), device=x.device)[:remaining]
                        indices = torch.cat([priority_indices, other_indices])
                    else:
                        indices = priority_indices
                else:
                    indices = torch.randperm(len(x), device=x.device)[:sample_size]
                
                # Reinforce subset
                binary_x = extract_enhanced_binary_features(x[indices], self.num_thresholds)
                mixed = self.channel_mixer(binary_x)
                self.fptm.reinforce(mixed, y_true[indices], y_pred[indices], s=adaptive_s)
            else:
                # Full batch
                binary_x = extract_enhanced_binary_features(x, self.num_thresholds)
                mixed = self.channel_mixer(binary_x)
                self.fptm.reinforce(mixed, y_true, y_pred, s=adaptive_s)
        
        return update_probability, sample_size


def train_one_epoch(model, optimizer, loader, device, epoch, total_epochs):
    """Training with proven approach from 82.5% success"""
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Add label smoothing
    
    # Timing
    forward_time = backward_time = reinforce_time = 0
    reinforce_calls = total_samples_reinforced = 0
    
    epoch_start = time.time()
    
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        
        # Forward
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        forward_time += time.time() - t0
        
        # Backward
        t0 = time.time()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        backward_time += time.time() - t0
        
        # Adaptive reinforcement
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            
            # Frequency decreases with training
            reinforce_freq = min(10, 3 + epoch // 20)
            
            if i % reinforce_freq == 0:
                t0 = time.time()
                update_prob, sample_size = model.adaptive_reinforce(
                    x, y, preds, logits
                )
                reinforce_time += time.time() - t0
                reinforce_calls += 1
                total_samples_reinforced += sample_size
            
            correct += (preds == y).sum().item()
            total += y.size(0)
            loss_sum += loss.item() * y.size(0)
    
    epoch_time = time.time() - epoch_start
    
    # Print timing
    print(f"  Time: Fwd {forward_time:.1f}s | Bwd {backward_time:.1f}s | Reinf {reinforce_time:.1f}s")
    print(f"  Reinforce: {reinforce_calls} calls, {total_samples_reinforced}/{total} samples ({100*total_samples_reinforced/total:.1f}%)")
    print(f"  Running acc: {model.running_accuracy:.1%}")
    
    return loss_sum/total, correct/total, epoch_time


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model"""
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    all_logits, all_labels = [], []
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        loss = criterion(logits, y)
        preds = logits.argmax(dim=-1)
        
        correct += (preds == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item() * y.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())
    
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    ece = compute_ece(logits, labels)
    
    return loss_sum/total, correct/total, ece


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-5)
    ap.add_argument("--patch_size", type=int, default=4)
    ap.add_argument("--num_clauses", type=int, default=1536)
    ap.add_argument("--attention_heads", type=int, default=32)
    ap.add_argument("--num_thresholds", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    print("=" * 80)
    print("🎯 FASHION-MNIST 90% TARGET - Enhanced Approach")
    print("=" * 80)
    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}")
    print(f"        num_clauses={args.num_clauses}, attention_heads={args.attention_heads}")
    print(f"        thresholds={args.num_thresholds}, edge_detection=True")
    print("\n✨ ENHANCEMENTS:")
    print("   ✅ 16 adaptive thresholds (2X more than 82.5% version)")
    print("   ✅ Sobel edge detection (4 additional channels)")
    print("   ✅ Label smoothing (better generalization)")
    print("   ✅ CosineAnnealingWarmRestarts (escape plateaus)")
    print("   ✅ Extended training (150 epochs)")
    print("=" * 80)
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    print("\nLoading Fashion-MNIST...")
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    
    test_transform = transforms.ToTensor()
    
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=256,
                           shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Training: {len(train_dataset):,} samples")
    print(f"Testing: {len(test_dataset):,} samples")
    
    # Create model
    print("\nCreating Enhanced 90% FPTM...")
    model = Enhanced90PercentFPTM(
        num_clauses=args.num_clauses,
        num_classes=10,
        attention_heads=args.attention_heads,
        num_thresholds=args.num_thresholds,
        patch_size=args.patch_size
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Cosine annealing with warm restarts
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=2)
    
    # Training
    print("\nStarting Training")
    print("=" * 80)
    
    best_acc = 0
    best_epoch = 0
    first_epoch_times = []
    last_epoch_times = []
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc, epoch_time = train_one_epoch(
            model, optimizer, train_loader, device, epoch, args.epochs
        )
        
        # Evaluate
        val_loss, val_acc, ece = evaluate(model, test_loader, device)
        
        # Track best
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), 'fashionmnist_90percent_best.pth')
            marker = " 🔥"
        else:
            marker = ""
        
        # Print results
        print(f"[{epoch:3d}/{args.epochs}] "
              f"Train: {train_loss:.3f}/{train_acc:.1%} | "
              f"Val: {val_loss:.3f}/{val_acc:.1%} | "
              f"ECE: {ece:.3f} | "
              f"LR: {scheduler.get_last_lr()[0]:.5f} | "
              f"Time: {epoch_time:.1f}s{marker}")
        
        # Step scheduler
        scheduler.step()
        
        # Track timing
        if epoch <= 5:
            first_epoch_times.append(epoch_time)
        if epoch > args.epochs - 5:
            last_epoch_times.append(epoch_time)
        
        # Check for 90% milestone
        if val_acc >= 0.90:
            print(f"\n🎉 TARGET ACHIEVED! 90% accuracy reached at epoch {epoch}!")
            print("Continuing training to see if we can improve further...")
        
        # Progress checkpoints
        if epoch % 10 == 0:
            print(f"\n  Checkpoint: Best so far = {best_acc:.2%} at epoch {best_epoch}\n")
    
    # Final results
    print("\n" + "=" * 80)
    print("Training Complete")
    print("=" * 80)
    print(f"Best accuracy: {best_acc:.2%} at epoch {best_epoch}")
    
    if len(first_epoch_times) > 0 and len(last_epoch_times) > 0:
        avg_first = np.mean(first_epoch_times)
        avg_last = np.mean(last_epoch_times)
        improvement = (avg_first - avg_last) / avg_first * 100
        print(f"\n⚡ SPEED IMPROVEMENT:")
        print(f"   First 5 epochs: {avg_first:.1f}s avg")
        print(f"   Last 5 epochs:  {avg_last:.1f}s avg")
        print(f"   Improvement:    {improvement:.1f}% faster!")
    
    print(f"\n🎯 RESULTS SUMMARY:")
    if best_acc >= 0.90:
        print(f"   ✅ SUCCESS! Achieved {best_acc:.2%} (target: 90%)")
    elif best_acc >= 0.88:
        print(f"   ✅ Very close! {best_acc:.2%} (target: 90%)")
    elif best_acc >= 0.85:
        print(f"   📈 Good progress: {best_acc:.2%} (target: 90%)")
    else:
        print(f"   🔧 More tuning needed: {best_acc:.2%} (target: 90%)")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
