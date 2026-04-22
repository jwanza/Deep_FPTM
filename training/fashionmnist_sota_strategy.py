#!/usr/bin/env python3
"""
Fashion-MNIST SOTA Strategy - Hierarchical FPTM with Advanced Techniques
Target: 90%+ accuracy on Fashion-MNIST
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Optional
import time
import argparse

from fptm.models import FPTMConvFast
from fptm.utils import set_seed


class HierarchicalFPTM(nn.Module):
    """
    Hierarchical FPTM with multiple scales and feature fusion
    """
    def __init__(self, 
                 in_channels: int = 1,
                 num_classes: int = 10,
                 base_clauses: int = 256,
                 device: str = "cuda"):
        super().__init__()
        
        # Multi-scale feature extraction
        # Small patches for fine details (7x7 patches = 4x4 grid)
        self.fine_model = FPTMConvFast(
            in_channels=in_channels,
            image_size=28,
            patch_size=7,
            num_clauses=base_clauses,
            attention_heads=8,
            num_classes=num_classes,
            normalize_mode="minmax"
        )
        
        # Medium patches for mid-level features (14x14 patches = 2x2 grid)
        self.mid_model = FPTMConvFast(
            in_channels=in_channels,
            image_size=28,
            patch_size=14,
            num_clauses=base_clauses // 2,
            attention_heads=4,
            num_classes=num_classes,
            normalize_mode="minmax"
        )
        
        # Global features (full image)
        self.global_model = FPTMConvFast(
            in_channels=in_channels,
            image_size=28,
            patch_size=28,
            num_clauses=base_clauses // 4,
            attention_heads=1,
            num_classes=num_classes,
            normalize_mode="minmax"
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(num_classes * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get predictions at different scales
        fine_logits = self.fine_model(x)
        mid_logits = self.mid_model(x)
        global_logits = self.global_model(x)
        
        # Concatenate all predictions
        combined = torch.cat([fine_logits, mid_logits, global_logits], dim=1)
        
        # Fuse predictions
        output = self.fusion(combined)
        
        # Apply temperature scaling
        return output / self.temperature
    
    @torch.no_grad()
    def reinforce(self, x: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor, s: float = 3.0):
        # Reinforce all sub-models
        self.fine_model.reinforce(x, y_true, y_pred, s=s * 1.5)  # More emphasis on fine details
        self.mid_model.reinforce(x, y_true, y_pred, s=s)
        self.global_model.reinforce(x, y_true, y_pred, s=s * 0.5)  # Less emphasis on global


class AdvancedAugmentation:
    """Advanced augmentation techniques for Fashion-MNIST"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
        
    def cutmix(self, images: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """CutMix augmentation"""
        if np.random.random() > self.p:
            return images, labels, labels, 1.0
            
        batch_size = images.size(0)
        indices = torch.randperm(batch_size).to(images.device)
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(alpha, alpha)
        
        # Get random box
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
        
        # Apply CutMix
        images_mixed = images.clone()
        images_mixed[:, :, bbx1:bbx2, bby1:bby2] = images[indices, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda based on actual box area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
        
        return images_mixed, labels, labels[indices], lam
    
    def _rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2


def create_dataloaders(batch_size: int = 128, augment: bool = True) -> Tuple[DataLoader, DataLoader]:
    """Create dataloaders with advanced augmentation"""
    
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=0, pin_memory=True
    )
    
    return train_loader, test_loader


def train_epoch(model: nn.Module, 
                opt: optim.Optimizer,
                train_loader: DataLoader,
                device: torch.device,
                epoch: int,
                augmenter: Optional[AdvancedAugmentation] = None) -> Tuple[float, float]:
    """Train with CutMix augmentation"""
    
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0
    
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        # Apply CutMix augmentation
        if augmenter and epoch > 5:  # Start CutMix after warmup
            x_mixed, y_a, y_b, lam = augmenter.cutmix(x, y)
            
            # Forward pass with mixed samples
            logits = model(x_mixed)
            loss = lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)
            
            # Use original samples for reinforcement
            with torch.no_grad():
                orig_logits = model(x)
                preds = orig_logits.argmax(dim=-1)
                if i % 5 == 0:  # Reinforce every 5 batches
                    model.reinforce(x, y, preds, s=3.0)
        else:
            # Standard training
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                if i % 5 == 0:
                    model.reinforce(x, y, preds, s=3.0)
        
        # Optimization step
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        
        # Track metrics
        with torch.no_grad():
            if augmenter and epoch > 5:
                # For CutMix, evaluate on original samples
                orig_logits = model(x)
                preds = orig_logits.argmax(dim=-1)
                correct += (preds == y).float().sum().item()
            else:
                correct += (preds == y).float().sum().item()
            
            total += y.size(0)
            loss_sum += loss.item() * y.size(0)
    
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Evaluate with Test-Time Augmentation (TTA)"""
    
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    
    # TTA transforms
    tta_transforms = [
        lambda x: x,  # Original
        lambda x: torch.flip(x, dims=[3]),  # Horizontal flip
        lambda x: F.pad(x[:, :, 2:, 2:], (2, 2, 2, 2), mode='reflect'),  # Slight shift
    ]
    
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        
        # Apply TTA
        logits_sum = torch.zeros(x.size(0), 10).to(device)
        for transform in tta_transforms:
            x_aug = transform(x)
            logits = model(x_aug)
            logits_sum += F.softmax(logits, dim=1)
        
        logits_avg = logits_sum / len(tta_transforms)
        loss = F.cross_entropy(logits_avg, y)
        
        preds = logits_avg.argmax(dim=-1)
        correct += (preds == y).float().sum().item()
        total += y.size(0)
        loss_sum += loss.item() * y.size(0)
    
    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--base_clauses", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--use_augmentation", action="store_true")
    parser.add_argument("--use_hierarchical", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    print("=" * 70)
    print("Fashion-MNIST SOTA Strategy")
    print("=" * 70)
    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}")
    print(f"        base_clauses={args.base_clauses}, lr={args.lr}")
    print(f"        hierarchical={args.use_hierarchical}, augmentation={args.use_augmentation}")
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create model
    if args.use_hierarchical:
        print("\nUsing Hierarchical Multi-Scale FPTM")
        model = HierarchicalFPTM(
            in_channels=1,
            num_classes=10,
            base_clauses=args.base_clauses,
            device=device
        ).to(device)
    else:
        print("\nUsing Standard FPTMConvFast")
        model = FPTMConvFast(
            in_channels=1,
            image_size=28,
            patch_size=7,  # Larger patches
            num_clauses=args.base_clauses,
            attention_heads=16,
            num_classes=10,
            normalize_mode="minmax"
        ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Data loaders
    train_loader, test_loader = create_dataloaders(args.batch_size, args.use_augmentation)
    
    # Optimizer with OneCycleLR
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = OneCycleLR(
        opt, 
        max_lr=args.lr * 10, 
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Augmentation
    augmenter = AdvancedAugmentation(p=0.5) if args.use_augmentation else None
    
    # Training
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train
        tr_loss, tr_acc = train_epoch(model, opt, train_loader, device, epoch, augmenter)
        
        # Step scheduler after each batch
        for _ in range(len(train_loader)):
            scheduler.step()
        
        # Evaluate
        va_loss, va_acc = evaluate(model, test_loader, device)
        
        # Track best
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), "best_fashionmnist_model.pth")
        
        # Log
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:3d}: Train Loss={tr_loss:.3f} Acc={tr_acc:.1%} | "
              f"Val Loss={va_loss:.3f} Acc={va_acc:.1%} | "
              f"Best={best_acc:.1%} | Time={elapsed:.1f}s")
        
        # Early stopping if we reach target
        if va_acc >= 0.90:
            print(f"\n🎯 Reached 90% accuracy target!")
            break
    
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"Best validation accuracy: {best_acc:.2%}")


if __name__ == "__main__":
    main()
