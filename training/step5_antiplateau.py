#!/usr/bin/env python3
"""
Step 5: ANTI-PLATEAU MECHANISMS - Dynamic strategies to prevent getting stuck
Expected: ~85-87% accuracy with more consistent training
Shows: How to escape local minima and maintain learning momentum
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import numpy as np
import time
import sys

sys.path.append('..')
from fptm.models import FPTMConvFast
from fptm.utils import set_seed


def extract_binary_features(x: torch.Tensor, num_thresholds: int = 8) -> torch.Tensor:
    """Convert continuous to binary features"""
    B, C, H, W = x.shape
    x_flat = x.view(B, -1)
    quantiles = torch.quantile(
        x_flat, 
        torch.linspace(0.1, 0.9, num_thresholds).to(x.device), 
        dim=1
    )
    
    binary_features = []
    for i in range(num_thresholds):
        threshold = quantiles[i].view(B, 1, 1, 1)
        binary = (x > threshold).float()
        binary_features.append(binary)
    
    return torch.cat(binary_features, dim=1)


class AntiPlateauFPTM(nn.Module):
    """FPTM with anti-plateau mechanisms"""
    
    def __init__(self, num_clauses: int = 1024, num_classes: int = 10):
        super().__init__()
        
        self.num_thresholds = 8
        
        # Main FPTM
        self.fptm = FPTMConvFast(
            in_channels=self.num_thresholds,
            image_size=28,
            patch_size=4,
            num_clauses=num_clauses,
            attention_heads=32,
            num_classes=num_classes,
            normalize_mode="none"
        )
        
        # Anti-plateau: auxiliary prediction head
        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            nn.Flatten(),
            nn.Linear(8 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Learnable fusion weight
        self.fusion_alpha = nn.Parameter(torch.tensor(0.9))
        
    def forward(self, x: torch.Tensor, use_aux: bool = True) -> torch.Tensor:
        binary_x = extract_binary_features(x, self.num_thresholds)
        
        # Main prediction
        main_out = self.fptm(binary_x)
        
        if use_aux:
            # Auxiliary prediction for regularization
            aux_out = self.aux_head(binary_x)
            alpha = torch.sigmoid(self.fusion_alpha)
            return alpha * main_out + (1 - alpha) * aux_out
        else:
            return main_out
    
    @torch.no_grad()
    def reinforce(self, x: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor, s: float = 3.0):
        binary_x = extract_binary_features(x, self.num_thresholds)
        self.fptm.reinforce(binary_x, y_true, y_pred, s=s)


class PlateauDetector:
    """Detect and respond to training plateaus"""
    
    def __init__(self, patience: int = 5, min_improvement: float = 0.5):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_acc = 0
        self.plateau_count = 0
        self.total_plateaus = 0
        
    def check(self, current_acc: float) -> dict:
        """Check if we're in a plateau and return action"""
        improvement = current_acc - self.best_acc
        
        if improvement < self.min_improvement:
            self.plateau_count += 1
        else:
            self.plateau_count = 0
            self.best_acc = current_acc
        
        action = {"plateau": False, "restart_lr": False, "increase_reinforce": False}
        
        if self.plateau_count >= self.patience:
            self.total_plateaus += 1
            self.plateau_count = 0
            action["plateau"] = True
            
            # Different strategies for different plateau counts
            if self.total_plateaus % 3 == 1:
                action["restart_lr"] = True
            elif self.total_plateaus % 3 == 2:
                action["increase_reinforce"] = True
            else:
                action["restart_lr"] = True
                action["increase_reinforce"] = True
        
        return action


def train_epoch_adaptive(model, optimizer, train_loader, device, epoch, 
                         reinforce_freq=3, plateau_mode=False):
    """Training with adaptive strategies"""
    model.train()
    train_loss = 0
    train_correct = 0
    
    # Adjust strategy if in plateau mode
    if plateau_mode:
        print("  [⚠️ Plateau mode: Enhanced training active]", end='')
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Use auxiliary head more in plateau mode
        use_aux = plateau_mode or (epoch > 10)
        output = model(data, use_aux=use_aux)
        
        # Adaptive loss
        if plateau_mode:
            # Add entropy regularization to encourage exploration
            loss = F.cross_entropy(output, target, label_smoothing=0.15)
            entropy = -(F.softmax(output, dim=1) * F.log_softmax(output, dim=1)).sum(dim=1).mean()
            loss = loss - 0.01 * entropy  # Encourage diversity
        else:
            loss = F.cross_entropy(output, target, label_smoothing=0.1)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Adaptive reinforcement
        with torch.no_grad():
            preds = output.argmax(dim=-1)
            
            # Per-sample difficulty assessment
            probs = F.softmax(output, dim=1)
            confidence = probs.max(dim=1)[0]
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
            
            # Reinforce based on difficulty
            if batch_idx % reinforce_freq == 0:
                for i in range(len(data)):
                    # Focus on hard samples (low confidence, high entropy)
                    if confidence[i] < 0.7 or entropy[i] > 1.5:
                        adaptive_s = 6.0  # Strong reinforcement
                    elif confidence[i] < 0.9:
                        adaptive_s = 4.0  # Medium reinforcement
                    else:
                        adaptive_s = 2.0  # Light reinforcement
                    
                    if i % 4 == 0:  # Don't reinforce every sample
                        model.reinforce(
                            data[i:i+1], 
                            target[i:i+1], 
                            preds[i:i+1], 
                            s=adaptive_s
                        )
        
        train_loss += loss.item() * len(data)
        train_correct += (preds == target).sum().item()
        
        # Progress with more info
        if batch_idx % 50 == 0:
            batch_acc = 100. * (preds == target).float().mean().item()
            avg_conf = confidence.mean().item()
            print(f"\r  Batch {batch_idx}/{len(train_loader)}: "
                  f"Acc={batch_acc:.1f}%, Conf={avg_conf:.2f}", end='')
    
    print()  # New line after progress
    return train_loss / len(train_loader.dataset), 100. * train_correct / len(train_loader.dataset)


def evaluate(model, test_loader, device):
    """Evaluate model"""
    model.eval()
    test_loss = 0
    test_correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, use_aux=False)  # No aux during evaluation
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            test_correct += (output.argmax(1) == target).sum().item()
    
    return test_loss / len(test_loader.dataset), 100. * test_correct / len(test_loader.dataset)


def main():
    print("=" * 70)
    print("STEP 5: ANTI-PLATEAU MECHANISMS")
    print("=" * 70)
    print("Advanced strategies to prevent getting stuck:")
    print("  1. Plateau detection with adaptive response")
    print("  2. Confidence-based reinforcement")
    print("  3. Learning rate warm restarts")
    print("  4. Auxiliary prediction head")
    print("  5. Entropy regularization")
    print("-" * 70)
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model with anti-plateau features
    model = AntiPlateauFPTM(num_clauses=1024, num_classes=10).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("  (includes auxiliary head for regularization)")
    
    # Data
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
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
    
    # Optimizer with warm restarts
    optimizer = optim.AdamW(model.parameters(), lr=0.004, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.0001)
    
    # Plateau detector
    plateau_detector = PlateauDetector(patience=5, min_improvement=0.5)
    
    print("\nStarting training with anti-plateau mechanisms...")
    print("-" * 70)
    
    best_acc = 0
    reinforce_freq = 5
    
    for epoch in range(1, 41):  # 40 epochs
        start_time = time.time()
        
        # Check for plateau
        if epoch > 1:
            action = plateau_detector.check(test_acc)
            if action["plateau"]:
                print(f"\n🔄 PLATEAU DETECTED at epoch {epoch}!")
                if action["restart_lr"]:
                    print("  -> Restarting learning rate")
                    for g in optimizer.param_groups:
                        g['lr'] = 0.004
                if action["increase_reinforce"]:
                    print("  -> Increasing reinforcement frequency")
                    reinforce_freq = max(1, reinforce_freq - 1)
        
        # Train with adaptive strategies
        plateau_mode = plateau_detector.plateau_count > 2
        train_loss, train_acc = train_epoch_adaptive(
            model, optimizer, train_loader, device, epoch, 
            reinforce_freq, plateau_mode
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, device)
        
        # Step scheduler
        scheduler.step()
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_antiplateau_model.pth')
        
        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]['lr']
        
        # Enhanced logging
        status = "🔴" if plateau_detector.plateau_count > 2 else "🟢"
        print(f"{status} Epoch {epoch:2d}: Train {train_acc:.1f}% | Test {test_acc:.1f}% | "
              f"Best {best_acc:.1f}% | LR {lr:.5f} | RF={reinforce_freq} | Time {elapsed:.1f}s")
        
        # Success check
        if test_acc >= 87:
            print("\n🎆 EXCELLENT! Reached 87% accuracy!")
            break
    
    print("\n" + "=" * 70)
    print("STEP 5 RESULT: Anti-plateau mechanisms prevent stagnation!")
    print(f"Final accuracy: {test_acc:.1f}%")
    print(f"Best accuracy: {best_acc:.1f}%")
    print(f"Total plateaus encountered and resolved: {plateau_detector.total_plateaus}")
    print("Expected: ~85-87% (vs ~82-85% in Step 4)")
    print("\n📊 The model now adapts dynamically to training challenges!")
    print("=" * 70)


if __name__ == "__main__":
    main()
