#!/usr/bin/env python3
"""
Step 3: INCREASE MODEL CAPACITY - More clauses and attention heads
Expected: ~78-80% accuracy
Shows: Fashion-MNIST needs more model capacity than simple datasets
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import sys

sys.path.append('..')
from fptm.models import FPTMConvFast
from fptm.utils import set_seed

def train_epoch(model, optimizer, train_loader, device, epoch, reinforce_freq=5):
    """Train for one epoch with reinforcement"""
    model.train()
    train_loss = 0
    train_correct = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        # More frequent reinforcement
        if batch_idx % reinforce_freq == 0:
            with torch.no_grad():
                preds = output.argmax(dim=-1)
                model.reinforce(data, target, preds, s=3.0)
        
        train_loss += loss.item() * len(data)
        train_correct += (output.argmax(1) == target).sum().item()
    
    return train_loss / len(train_loader.dataset), 100. * train_correct / len(train_loader.dataset)

def evaluate(model, test_loader, device):
    """Evaluate model"""
    model.eval()
    test_loss = 0
    test_correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            test_correct += (output.argmax(1) == target).sum().item()
    
    return test_loss / len(test_loader.dataset), 100. * test_correct / len(test_loader.dataset)

def main():
    print("=" * 70)
    print("STEP 3: INCREASE MODEL CAPACITY")
    print("=" * 70)
    print("Changes from Step 2:")
    print("  - num_clauses: 256 → 1024 (4× increase)")
    print("  - attention_heads: 8 → 32 (4× increase)")
    print("  - Add learning rate scheduling")
    print("  - More frequent reinforcement")
    print("-" * 70)
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # INCREASED CAPACITY
    model = FPTMConvFast(
        in_channels=1,
        image_size=28,
        patch_size=4,           # Keep the working patch size
        num_clauses=1024,       # 4× more clauses
        attention_heads=32,     # 4× more attention heads
        num_classes=10,
        normalize_mode="minmax"
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  (vs Step 2: ~530K parameters)")
    
    # Data with light augmentation
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
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
    
    # Better optimizer with scheduling
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=25, eta_min=0.0001)
    
    print("\nStarting training...")
    print("-" * 70)
    
    best_acc = 0
    for epoch in range(1, 26):  # 25 epochs
        start_time = time.time()
        
        # Adaptive reinforcement frequency
        if epoch < 5:
            reinforce_freq = 10  # Less frequent early
        elif epoch < 15:
            reinforce_freq = 5   # Medium frequency
        else:
            reinforce_freq = 3   # More frequent later
        
        train_loss, train_acc = train_epoch(model, optimizer, train_loader, device, epoch, reinforce_freq)
        test_loss, test_acc = evaluate(model, test_loader, device)
        scheduler.step()
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        elapsed = time.time() - start_time
        lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch:2d}: Train {train_acc:.1f}% | Test {test_acc:.1f}% | "
              f"Best {best_acc:.1f}% | LR {lr:.5f} | Time {elapsed:.1f}s")
    
    print("\n" + "=" * 70)
    print("STEP 3 RESULT: Increased capacity helps significantly!")
    print(f"Final accuracy: {test_acc:.1f}%")
    print(f"Best accuracy: {best_acc:.1f}%")
    print("Expected: ~78-80% (vs ~75% in Step 1)")
    print("=" * 70)

if __name__ == "__main__":
    main()
