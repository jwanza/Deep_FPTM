#!/usr/bin/env python3
"""
Step 1: BASELINE - Current configuration showing the problem
Expected: ~75% accuracy, may plateau
This establishes our starting point
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import sys

# Add parent directory to path for imports
sys.path.append('..')
from fptm.models import FPTMConvFast
from fptm.utils import set_seed

def main():
    print("=" * 70)
    print("STEP 1: BASELINE - Current Default Configuration")
    print("=" * 70)
    print("Problem: Low capacity, suboptimal hyperparameters")
    print("Expected: ~75% accuracy with plateaus")
    print("-" * 70)
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # BASELINE CONFIGURATION - Original settings
    model = FPTMConvFast(
        in_channels=1,
        image_size=28,
        patch_size=4,         # Default patch size
        num_clauses=256,      # Low capacity
        attention_heads=8,    # Few attention heads
        num_classes=10,
        normalize_mode="none" # No normalization
    ).to(device)
    
    print(f"Configuration:")
    print(f"  patch_size: 4 (7×7=49 patches)")
    print(f"  num_clauses: 256 (LOW)")
    print(f"  attention_heads: 8 (LOW)")
    print(f"  normalize_mode: none")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Standard data loading
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
    
    # Simple optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    print("\nStarting training...")
    print("-" * 70)
    
    for epoch in range(1, 16):  # Just 15 epochs to show plateau
        # Train
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
            
            # Occasional reinforcement (infrequent)
            if batch_idx % 20 == 0:  # Very infrequent
                with torch.no_grad():
                    preds = output.argmax(dim=-1)
                    model.reinforce(data, target, preds, s=3.0)
            
            train_loss += loss.item() * len(data)
            train_correct += (output.argmax(1) == target).sum().item()
        
        # Evaluate
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_correct += (output.argmax(1) == target).sum().item()
        
        train_acc = 100. * train_correct / len(train_loader.dataset)
        test_acc = 100. * test_correct / len(test_loader.dataset)
        
        print(f"Epoch {epoch:2d}: Train {train_acc:.1f}% | Test {test_acc:.1f}%")
        
        if epoch > 10 and test_acc < 76:
            print("\n⚠️ PLATEAU DETECTED - Model stuck around 75%")
    
    print("\n" + "=" * 70)
    print("BASELINE RESULT: Limited by low capacity and poor hyperparameters")
    print(f"Final accuracy: {test_acc:.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
