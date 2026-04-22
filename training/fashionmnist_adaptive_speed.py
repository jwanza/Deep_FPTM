#!/usr/bin/env python3
"""
Fashion-MNIST with Adaptive Speed - Mimics Julia's behavior
Gets FASTER as accuracy improves by reducing reinforcement probability
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
import math

from fptm.models import FPTMConvFast
from fptm.utils import set_seed
from fptm.heads import compute_ece


class AdaptiveFPTM(nn.Module):
    """
    FPTM with adaptive reinforcement that speeds up as accuracy improves.
    Mimics Julia's behavior where update probability decreases with better predictions.
    """
    
    def __init__(self, num_clauses: int = 1024, num_classes: int = 10, 
                 attention_heads: int = 32, patch_size: int = 4):
        super().__init__()
        
        # Ensure divisibility
        if num_clauses % attention_heads != 0:
            num_clauses = ((num_clauses + attention_heads - 1) // 
                          attention_heads) * attention_heads
        
        self.fptm = FPTMConvFast(
            in_channels=1,
            image_size=28,
            patch_size=patch_size,
            num_clauses=num_clauses,
            attention_heads=attention_heads,
            num_classes=num_classes,
            normalize_mode="minmax"
            # Using defaults - no automata_states override!
        )
        
        # Track performance for adaptive behavior
        self.running_accuracy = 0.1  # Start at 10% (random)
        self.confidence_threshold = 0.8  # High confidence predictions
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fptm(x)
    
    @torch.no_grad()
    def adaptive_reinforce(self, x: torch.Tensor, y_true: torch.Tensor, 
                          y_pred: torch.Tensor, logits: torch.Tensor, 
                          base_s: float = 3.0):
        """
        Adaptive reinforcement that reduces with accuracy and confidence.
        Mimics Julia's: if rand() < update_probability
        """
        # Calculate confidence of predictions
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1)[0].mean().item()
        
        # Calculate current batch accuracy
        batch_acc = (y_pred == y_true).float().mean().item()
        
        # Update running accuracy (exponential moving average)
        self.running_accuracy = 0.95 * self.running_accuracy + 0.05 * batch_acc
        
        # ADAPTIVE PROBABILITY - key insight from Julia!
        # As accuracy → 1.0, probability → 0
        # As confidence → 1.0, probability → 0
        update_probability = (1.0 - self.running_accuracy) * (1.0 - confidence * 0.5)
        
        # Additional speed optimization: reduce samples as accuracy improves
        if self.running_accuracy > 0.8:
            # High accuracy: reinforce only 25% of batch
            sample_size = max(4, len(x) // 4)
        elif self.running_accuracy > 0.6:
            # Medium accuracy: reinforce 50% of batch
            sample_size = max(8, len(x) // 2)
        else:
            # Low accuracy: reinforce full batch
            sample_size = len(x)
        
        # SKIP reinforcement with probability (Julia's key optimization!)
        if torch.rand(1).item() < update_probability:
            # Adaptive s value
            adaptive_s = base_s * (1.0 + max(0, 0.5 - self.running_accuracy))
            
            # Select samples to reinforce
            if sample_size < len(x):
                # Focus on errors and low-confidence correct predictions
                errors = (y_pred != y_true)
                low_conf = probs.max(dim=-1)[0] < self.confidence_threshold
                priority = errors | low_conf
                
                if priority.sum() > 0:
                    # Prioritize errors and uncertain predictions
                    priority_indices = torch.where(priority)[0][:sample_size]
                    if len(priority_indices) < sample_size:
                        # Add random samples if needed
                        remaining = sample_size - len(priority_indices)
                        other_indices = torch.randperm(len(x))[:remaining]
                        indices = torch.cat([priority_indices, other_indices])
                    else:
                        indices = priority_indices
                else:
                    # All high confidence correct - random sample
                    indices = torch.randperm(len(x))[:sample_size]
                
                # Reinforce selected samples
                self.fptm.reinforce(x[indices], y_true[indices], 
                                   y_pred[indices], s=adaptive_s)
            else:
                # Full batch reinforcement
                self.fptm.reinforce(x, y_true, y_pred, s=adaptive_s)
        
        # Return stats for monitoring
        return update_probability, sample_size


def train_one_epoch_adaptive(model, opt, loader, device, epoch, total_epochs):
    """Train with adaptive speed - gets faster as accuracy improves"""
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    ce = nn.CrossEntropyLoss()
    
    # Track timing
    forward_time = 0
    backward_time = 0
    reinforce_time = 0
    reinforce_calls = 0
    total_samples_reinforced = 0
    
    epoch_start = time.time()
    
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = ce(logits, y)
        forward_time += time.time() - t0
        
        # Backward pass
        t0 = time.time()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        backward_time += time.time() - t0
        
        # Adaptive reinforcement
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            
            # Reduce reinforcement frequency as training progresses
            # Early epochs: every 3 batches, Later: every 10 batches
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
    
    # Print timing breakdown
    print(f"  Timing: Forward {forward_time:.1f}s | "
          f"Backward {backward_time:.1f}s | "
          f"Reinforce {reinforce_time:.1f}s")
    print(f"  Reinforce: {reinforce_calls} calls, "
          f"{total_samples_reinforced}/{total} samples ({100*total_samples_reinforced/total:.1f}%)")
    print(f"  Running accuracy: {model.running_accuracy:.1%}")
    
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
    ap.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    print("=" * 70)
    print("Adaptive Speed FPTM - Mimics Julia's Behavior")
    print("=" * 70)
    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}")
    print(f"        num_clauses={args.num_clauses}, attention_heads={args.attention_heads}")
    print(f"        lr={args.lr}, scheduler={args.scheduler}")
    print("\n🚀 KEY FEATURE: Training speeds up as accuracy improves!")
    print("   - Update probability decreases with accuracy")
    print("   - Fewer samples reinforced as model improves")
    print("   - Reinforcement frequency decreases over time")
    print("=" * 70)
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Fashion-MNIST
    print("\nLoading Fashion-MNIST dataset...")
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
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
        test_dataset, batch_size=args.batch_size*2, shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )
    
    # Create Adaptive FPTM model
    print("\nCreating Adaptive FPTM model...")
    model = AdaptiveFPTM(
        num_clauses=args.num_clauses,
        num_classes=10,
        attention_heads=args.attention_heads,
        patch_size=args.patch_size
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    
    # Setup optimizer and scheduler
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    else:
        scheduler = None
    
    # Training loop
    print("\nStarting Training")
    print("=" * 70)
    
    best_acc = 0
    best_epoch = 0
    epoch_times = []
    
    for epoch in range(1, args.epochs + 1):
        # Train
        tr_loss, tr_acc, epoch_time = train_one_epoch_adaptive(
            model, opt, train_loader, device, epoch, args.epochs
        )
        epoch_times.append(epoch_time)
        
        # Step scheduler
        if scheduler:
            scheduler.step()
        
        # Evaluate
        va_loss, va_acc, ece = evaluate(model, test_loader, device)
        
        # Track best
        if va_acc > best_acc:
            best_acc = va_acc
            best_epoch = epoch
        
        # Print progress with speed indicator
        current_lr = opt.param_groups[0]['lr']
        speed_indicator = "🔥" if epoch > 5 and epoch_time < np.mean(epoch_times[:5]) else ""
        
        print(f"[{epoch:3d}/{args.epochs}] "
              f"Train: {tr_loss:.3f}/{tr_acc:.1%} | "
              f"Val: {va_loss:.3f}/{va_acc:.1%} | "
              f"ECE: {ece:.3f} | "
              f"LR: {current_lr:.5f} | "
              f"Time: {epoch_time:.1f}s {speed_indicator}")
        
        # Show speed improvement
        if epoch == 10:
            early_avg = np.mean(epoch_times[:5])
            recent_avg = np.mean(epoch_times[5:10])
            speedup = (early_avg - recent_avg) / early_avg * 100
            print(f"\n  ⚡ Speed improvement: {speedup:.1f}% faster than initial epochs\n")
        
        # Early stopping if target reached
        if va_acc >= 0.82:
            print(f"\n✅ Target reached: {va_acc:.1%}")
            break
    
    # Final results
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"Best validation accuracy: {best_acc:.2%} at epoch {best_epoch}")
    
    # Analyze speed improvement
    if len(epoch_times) > 10:
        early_times = epoch_times[:5]
        late_times = epoch_times[-5:]
        speedup = (np.mean(early_times) - np.mean(late_times)) / np.mean(early_times) * 100
        print(f"\n🚀 SPEED ANALYSIS:")
        print(f"   First 5 epochs avg: {np.mean(early_times):.1f}s")
        print(f"   Last 5 epochs avg:  {np.mean(late_times):.1f}s")
        print(f"   Speed improvement:  {speedup:.1f}% faster!")
        print("\nThis matches Julia's behavior where training gets faster as accuracy improves!")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
