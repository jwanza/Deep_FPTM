#!/usr/bin/env python3
"""
Ultimate Fashion-MNIST FPTM v2 - ALL optimizations combined:
- Binary features (accuracy boost)
- Adaptive speed (gets faster like Julia)
- FPTMConvDeepFast (fixes backward pass bottleneck)
- Working defaults (no automata_states=100!)

Expected: 83-87% accuracy with FAST training that gets even faster!
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
import gc

from fptm.models.fptm_conv_deep_fast import FPTMConvDeepFast, create_fptm_deep_fast_medium
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


class UltimateFPTMv2(nn.Module):
    """
    Ultimate FPTM v2 combining ALL optimizations:
    1. Binary features (accuracy)
    2. Adaptive reinforcement (speed)
    3. Deep architecture (faster backward)
    4. Working defaults (stability)
    """
    
    def __init__(self, num_thresholds: int = 8,
                 stages_clauses = [128, 256, 512],
                 stages_heads = [8, 16, 32],
                 stages_bottlenecks = [64, 128, 0],
                 use_checkpoint: bool = True):
        super().__init__()
        
        self.num_thresholds = num_thresholds
        
        # Binary feature processing
        self.channel_mixer = nn.Conv2d(num_thresholds, 1, kernel_size=1)
        
        # Main FPTM - Using DeepFast for better backward performance!
        self.fptm = FPTMConvDeepFast(
            in_channels=1,  # After channel mixing
            image_size=28,
            patch_size=4,
            stages_num_clauses=stages_clauses,
            stages_heads=stages_heads,
            stages_bottlenecks=stages_bottlenecks,
            num_classes=10,
            normalize_mode="none",  # Binary features don't need normalization
            use_checkpoint=use_checkpoint,  # Save memory!
            dropout_rate=0.1
            # Using all defaults - they work!
        )
        
        # Adaptive speed tracking (Julia-style)
        self.running_accuracy = 0.1
        self.confidence_threshold = 0.8
        self.update_probability = 1.0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to binary
        binary_x = extract_binary_features(x, self.num_thresholds)
        
        # Mix channels
        binary_x = self.channel_mixer(binary_x)
        
        return self.fptm(binary_x)
    
    @torch.no_grad()
    def adaptive_reinforce(self, x: torch.Tensor, y_true: torch.Tensor, 
                          y_pred: torch.Tensor, logits: torch.Tensor, 
                          epoch: int, base_s: float = 3.0):
        """
        Julia-style adaptive reinforcement:
        - Probability decreases as accuracy improves
        - Sample size reduces with confidence
        - Frequency adapts over epochs
        """
        # Calculate confidence
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1)[0].mean().item()
        
        # Calculate batch accuracy
        batch_acc = (y_pred == y_true).float().mean().item()
        
        # Update running accuracy
        self.running_accuracy = 0.95 * self.running_accuracy + 0.05 * batch_acc
        
        # JULIA-STYLE UPDATE PROBABILITY!
        # As accuracy approaches 1.0, probability approaches 0
        self.update_probability = (1.0 - self.running_accuracy) * (1.0 - confidence * 0.5)
        
        # Additional epoch-based reduction
        epoch_factor = max(0.2, 1.0 - epoch / 50)  # Reduce over epochs
        self.update_probability *= epoch_factor
        
        # Adaptive sample size based on performance
        if self.running_accuracy > 0.8:
            sample_size = max(2, len(x) // 8)  # 12.5% when excellent
        elif self.running_accuracy > 0.7:
            sample_size = max(4, len(x) // 4)  # 25% when good
        elif self.running_accuracy > 0.5:
            sample_size = max(8, len(x) // 2)  # 50% when medium
        else:
            sample_size = len(x)  # 100% when learning
        
        # SKIP with probability (Julia's key optimization!)
        if torch.rand(1).item() < self.update_probability:
            # Adaptive s value
            adaptive_s = base_s * (1.0 + max(0, 0.5 - self.running_accuracy))
            
            # Priority sampling - focus on errors
            if sample_size < len(x):
                errors = (y_pred != y_true)
                low_conf = probs.max(dim=-1)[0] < self.confidence_threshold
                priority = errors | low_conf
                
                if priority.sum() > 0:
                    priority_indices = torch.where(priority)[0][:sample_size]
                    indices = priority_indices
                else:
                    indices = torch.randperm(len(x))[:sample_size]
                
                # Process binary features and reinforce
                binary_x = extract_binary_features(x[indices], self.num_thresholds)
                binary_x = self.channel_mixer(binary_x)
                self.fptm.reinforce(binary_x, y_true[indices], 
                                   y_pred[indices], s=adaptive_s)
            else:
                # Full batch
                binary_x = extract_binary_features(x, self.num_thresholds)
                binary_x = self.channel_mixer(binary_x)
                self.fptm.reinforce(binary_x, y_true, y_pred, s=adaptive_s)
            
            return True, sample_size  # Did reinforce
        else:
            return False, 0  # Skipped reinforcement


def train_one_epoch_ultimate_v2(model, opt, loader, device, epoch, total_epochs):
    """Train with ALL optimizations"""
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    ce = nn.CrossEntropyLoss()
    
    # Timing
    forward_time = 0
    backward_time = 0
    reinforce_time = 0
    data_time = 0
    reinforce_calls = 0
    reinforce_skips = 0
    total_samples_reinforced = 0
    
    epoch_start = time.time()
    data_start = time.time()
    
    for i, (x, y) in enumerate(loader):
        data_time += time.time() - data_start
        x, y = x.to(device), y.to(device)
        
        # Forward (through binary features + deep architecture)
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = ce(logits, y)
        forward_time += time.time() - t0
        
        # Backward (optimized with FPTMConvDeepFast!)
        t0 = time.time()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        backward_time += time.time() - t0
        
        # Adaptive reinforcement (Julia-style)
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            
            # Dynamic frequency based on epoch
            reinforce_freq = min(20, 3 + epoch // 5)  # Increase interval over time
            
            if i % reinforce_freq == 0:
                t0 = time.time()
                did_reinforce, sample_size = model.adaptive_reinforce(
                    x, y, preds, logits, epoch
                )
                reinforce_time += time.time() - t0
                
                if did_reinforce:
                    reinforce_calls += 1
                    total_samples_reinforced += sample_size
                else:
                    reinforce_skips += 1
            
            correct += (preds == y).float().sum().item()
            total += y.size(0)
            loss_sum += float(loss.item()) * y.size(0)
        
        # Clear cache periodically
        if i % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        data_start = time.time()
    
    epoch_time = time.time() - epoch_start
    
    # Detailed timing breakdown
    print(f"  ⏱️ Time: Data {data_time:.1f}s | Fwd {forward_time:.1f}s | "
          f"Bwd {backward_time:.1f}s | Reinf {reinforce_time:.1f}s")
    print(f"  📊 Reinforce: {reinforce_calls} done, {reinforce_skips} skipped, "
          f"{total_samples_reinforced}/{total} samples ({100*total_samples_reinforced/total:.1f}%)")
    print(f"  📈 Running acc: {model.running_accuracy:.1%} | "
          f"Update prob: {model.update_probability:.1%}")
    
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
    ap.add_argument("--num_thresholds", type=int, default=8)
    ap.add_argument("--stages_clauses", type=str, default="128,256,512")
    ap.add_argument("--stages_heads", type=str, default="8,16,32")
    ap.add_argument("--stages_bottlenecks", type=str, default="64,128,0")
    ap.add_argument("--use_checkpoint", action="store_true", help="Memory-efficient training")
    ap.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    print("=" * 70)
    print("🚀 ULTIMATE FPTM v2 - Everything Optimized!")
    print("=" * 70)
    
    # Parse stage configuration
    stages_clauses = [int(x) for x in args.stages_clauses.split(',')]
    stages_heads = [int(x) for x in args.stages_heads.split(',')]
    stages_bottlenecks = [int(x) for x in args.stages_bottlenecks.split(',')]
    
    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}")
    print(f"        stages={len(stages_clauses)}, clauses={stages_clauses}")
    print(f"        thresholds={args.num_thresholds}")
    print("\n✨ ALL OPTIMIZATIONS:")
    print("   ✅ Binary features (accuracy boost)")
    print("   ✅ Adaptive speed (Julia-style, gets faster)")
    print("   ✅ FPTMConvDeepFast (faster backward pass)")
    print("   ✅ Working defaults (no automata_states=100)")
    print("   ✅ Priority sampling (focus on errors)")
    print("   ✅ Gradient checkpointing" if args.use_checkpoint else "   - No checkpointing")
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
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size*2, shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )
    
    # Create model
    print("\nCreating Ultimate FPTM v2...")
    model = UltimateFPTMv2(
        num_thresholds=args.num_thresholds,
        stages_clauses=stages_clauses,
        stages_heads=stages_heads,
        stages_bottlenecks=stages_bottlenecks,
        use_checkpoint=args.use_checkpoint
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    
    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
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
        tr_loss, tr_acc, epoch_time = train_one_epoch_ultimate_v2(
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
            torch.save(model.state_dict(), 'ultimate_v2_model.pth')
        
        # Progress with speed indicator
        current_lr = opt.param_groups[0]['lr']
        speed_emoji = "🔥" if epoch > 5 and epoch_time < np.mean(epoch_times[:5]) * 0.9 else ""
        
        print(f"[{epoch:3d}/{args.epochs}] "
              f"Train: {tr_loss:.3f}/{tr_acc:.1%} | "
              f"Val: {va_loss:.3f}/{va_acc:.1%} | "
              f"ECE: {ece:.3f} | "
              f"LR: {current_lr:.5f} | "
              f"Time: {epoch_time:.1f}s {speed_emoji}")
        
        # Speed analysis
        if epoch == 10 and len(epoch_times) >= 10:
            early = np.mean(epoch_times[:5])
            recent = np.mean(epoch_times[5:10])
            speedup = (early - recent) / early * 100
            print(f"\n  ⚡ Speedup: {speedup:.1f}% faster than initial epochs!\n")
        
        # Memory stats
        if torch.cuda.is_available() and epoch == 1:
            max_mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  💾 Peak GPU memory: {max_mem:.2f} GB")
        
        # Target reached
        if va_acc >= 0.87:
            print(f"\n🎯 Target reached: {va_acc:.1%}")
            break
        
        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
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
    
    print("\n🏆 ULTIMATE v2 combines:")
    print("   • Binary features → Better accuracy")
    print("   • FPTMConvDeepFast → Faster backward pass")
    print("   • Julia-style adaptive → Gets faster over time")
    print("   • Working defaults → Stable learning")
    print("=" * 70)


if __name__ == "__main__":
    main()
