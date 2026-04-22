#!/usr/bin/env python3
"""
Improved Fashion-MNIST Training - Addressing Key Issues
Target: 85%+ accuracy with practical improvements
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
import numpy as np
import time
import argparse

from fptm.models import FPTMConvFast
from fptm.utils import set_seed


class ImprovedFPTM(nn.Module):
    """
    Improved FPTM with better feature extraction for Fashion-MNIST
    """
    def __init__(self, 
                 in_channels: int = 1,
                 image_size: int = 28,
                 num_clauses: int = 1024,
                 num_classes: int = 10):
        super().__init__()
        
        # Feature extractor: Convert continuous grayscale to richer features
        self.feature_extractor = nn.Sequential(
            # Initial convolution to extract edge features
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Extract texture features
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Final feature map
            nn.Conv2d(64, 16, kernel_size=1),  # Reduce to 16 channels
            nn.BatchNorm2d(16),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        # FPTM on extracted features
        self.fptm = FPTMConvFast(
            in_channels=16,  # Process extracted features
            image_size=image_size,
            patch_size=7,  # Larger patches for better context
            num_clauses=num_clauses,
            attention_heads=32,  # More attention heads
            num_classes=num_classes,
            normalize_mode="none"  # Features already normalized
        )
        
        # Additional classifier head for ensemble
        self.auxiliary_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Fusion weight
        self.fusion_weight = nn.Parameter(torch.tensor(0.8))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        features = self.feature_extractor(x)
        
        # FPTM predictions
        fptm_logits = self.fptm(features)
        
        # Auxiliary predictions
        aux_logits = self.auxiliary_head(features)
        
        # Weighted fusion
        w = torch.sigmoid(self.fusion_weight)
        return w * fptm_logits + (1 - w) * aux_logits
    
    @torch.no_grad()
    def reinforce(self, x: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor, s: float = 3.0):
        features = self.feature_extractor(x)
        self.fptm.reinforce(features, y_true, y_pred, s=s)


def extract_binary_features(x: torch.Tensor, num_thresholds: int = 4) -> torch.Tensor:
    """
    Extract binary features using adaptive thresholding
    Similar to the Julia implementation
    """
    B, C, H, W = x.shape
    
    # Calculate quantiles for adaptive thresholding
    x_flat = x.view(B, -1)
    quantiles = torch.quantile(x_flat, torch.linspace(0.1, 0.9, num_thresholds).to(x.device), dim=1)
    
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
    FPTM that works on binary features (like Julia implementation)
    """
    def __init__(self, num_clauses: int = 2048, num_classes: int = 10):
        super().__init__()
        
        self.num_thresholds = 8  # More thresholds for richer features
        
        # Process binary features
        self.fptm = FPTMConvFast(
            in_channels=self.num_thresholds,
            image_size=28,
            patch_size=7,
            num_clauses=num_clauses,
            attention_heads=32,
            num_classes=num_classes,
            normalize_mode="none"  # Binary features don't need normalization
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to binary features
        binary_x = extract_binary_features(x, self.num_thresholds)
        return self.fptm(binary_x)
    
    @torch.no_grad()
    def reinforce(self, x: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor, s: float = 3.0):
        binary_x = extract_binary_features(x, self.num_thresholds)
        self.fptm.reinforce(binary_x, y_true, y_pred, s=s)


def create_optimized_dataloaders(batch_size: int = 128) -> tuple:
    """Create dataloaders with light augmentation"""
    
    # Light augmentation that preserves Fashion-MNIST characteristics
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        transforms.ToTensor(),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    return train_loader, test_loader


def train_epoch_improved(model: nn.Module, opt: optim.Optimizer, 
                         train_loader: DataLoader, device: torch.device,
                         epoch: int) -> tuple:
    """Improved training with adaptive reinforcement"""
    
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0
    
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        logits = model(x)
        loss = F.cross_entropy(logits, y, label_smoothing=0.1)  # Label smoothing
        
        # Backward pass
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        
        # Reinforcement learning
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct += (preds == y).float().sum().item()
            total += y.size(0)
            loss_sum += loss.item() * y.size(0)
            
            # Adaptive reinforcement
            if i % 3 == 0:
                batch_acc = (preds == y).float().mean().item()
                # Stronger reinforcement when accuracy is low
                adaptive_s = 5.0 * max(0.1, 1.0 - batch_acc)
                model.reinforce(x, y, preds, s=adaptive_s)
            
            # Progress indicator
            if i % 100 == 0:
                print(f"  Batch {i}/{len(train_loader)} - Acc: {correct/total:.1%}", end='\r')
    
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate_improved(model: nn.Module, test_loader: DataLoader, device: torch.device) -> tuple:
    """Evaluate model"""
    
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        
        preds = logits.argmax(dim=-1)
        correct += (preds == y).float().sum().item()
        total += y.size(0)
        loss_sum += loss.item() * y.size(0)
    
    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["improved", "binary"], default="binary",
                       help="Model type: improved (with CNN features) or binary (like Julia)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_clauses", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    print("=" * 70)
    print("Improved Fashion-MNIST Training")
    print("=" * 70)
    print(f"Model Type: {args.model_type}")
    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}")
    print(f"        num_clauses={args.num_clauses}, lr={args.lr}")
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Create model based on type
    if args.model_type == "binary":
        print("Using Binary Feature FPTM (Julia-like)")
        model = BinaryFPTM(num_clauses=args.num_clauses, num_classes=10).to(device)
    else:
        print("Using Improved FPTM with CNN features")
        model = ImprovedFPTM(num_clauses=args.num_clauses, num_classes=10).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Data
    train_loader, test_loader = create_optimized_dataloaders(args.batch_size)
    print(f"Training samples: {len(train_loader.dataset):,}")
    print(f"Test samples: {len(test_loader.dataset):,}")
    
    # Optimizer with warm restarts
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=args.lr * 0.01)
    
    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    
    best_acc = 0
    no_improve = 0
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train
        tr_loss, tr_acc = train_epoch_improved(model, opt, train_loader, device, epoch)
        
        # Evaluate
        va_loss, va_acc = evaluate_improved(model, test_loader, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = opt.param_groups[0]['lr']
        
        # Track best
        if va_acc > best_acc:
            best_acc = va_acc
            no_improve = 0
            torch.save(model.state_dict(), "best_improved_model.pth")
        else:
            no_improve += 1
        
        # Log
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:3d}: Train Loss={tr_loss:.3f} Acc={tr_acc:.1%} | "
              f"Val Loss={va_loss:.3f} Acc={va_acc:.1%} | "
              f"LR={current_lr:.5f} | Best={best_acc:.1%} | Time={elapsed:.1f}s")
        
        # Early stopping
        if no_improve >= 15:
            print("\nEarly stopping - no improvement for 15 epochs")
            break
        
        # Success check
        if va_acc >= 0.85:
            print(f"\n✅ Reached 85% accuracy target!")
            break
    
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"Best validation accuracy: {best_acc:.2%}")
    
    # Final inference speed test
    print("\n" + "=" * 70)
    print("Inference Speed Test")
    print("=" * 70)
    
    model.eval()
    with torch.no_grad():
        for bs in [32, 64, 128, 256]:
            dummy_input = torch.randn(bs, 1, 28, 28).to(device)
            
            # Warmup
            for _ in range(10):
                _ = model(dummy_input)
            
            # Time
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(100):
                _ = model(dummy_input)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            samples_per_sec = (bs * 100) / elapsed
            print(f"Batch size {bs:3d}: {samples_per_sec:,.0f} samples/sec")


if __name__ == "__main__":
    main()
