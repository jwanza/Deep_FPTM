"""
FAST Fashion-MNIST training using preprocessed binary features.
No more wasting time on feature extraction every batch!
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import time
import argparse
import os
import sys
sys.path.append('.')

from fptm.models.fptm_conv_julia import FPTMConvJulia
from fptm.utils import set_seed


class PreprocessedDataset(torch.utils.data.Dataset):
    """Dataset that loads preprocessed binary features from disk."""
    def __init__(self, features_path):
        print(f"Loading preprocessed data from {features_path}...")
        data = torch.load(features_path, map_location='cpu')
        self.features = data['features']
        self.labels = data['labels']
        self.num_channels = data['num_channels']
        print(f"✅ Loaded {len(self.labels)} samples with {self.num_channels} channels")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class FastFPTM(nn.Module):
    """
    Fast FPTM that uses preprocessed features directly.
    No feature extraction needed!
    """
    def __init__(self, num_channels, num_clauses=512, attention_heads=0):
        super().__init__()
        
        # Channel mixer to reduce to 1 channel for FPTM
        self.channel_mixer = nn.Conv2d(num_channels, 1, kernel_size=1)
        
        # FPTM model
        self.fptm = FPTMConvJulia(
            in_channels=1,
            image_size=28,
            patch_size=4,
            num_clauses=num_clauses,
            num_classes=10,
            attention_heads=attention_heads,
            epsilon=1e-6,
            automata_states=50,
            normalize_mode="none"  # Already normalized during preprocessing
        )
    
    def forward(self, x):
        # x is already binary features: (B, num_channels, 28, 28)
        mixed = self.channel_mixer(x)
        return self.fptm(mixed)
    
    def reinforce(self, x, y_true, y_pred, s=3.0):
        mixed = self.channel_mixer(x)
        self.fptm.reinforce(mixed, y_true, y_pred, s)


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Fast training epoch without feature extraction overhead."""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        # Forward pass (NO FEATURE EXTRACTION!)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Reinforce every 3rd batch
        if batch_idx % 3 == 0:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                model.reinforce(x, y, preds, s=3.0)
        
        # Stats
        running_loss += loss.item()
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += y.size(0)
        
        # Progress
        if batch_idx % 50 == 0:
            acc = 100 * correct / total
            print(f"  Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.3f}, Acc={acc:.1f}%")
    
    epoch_time = time.time() - start_time
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    
    return train_loss, train_acc, epoch_time


def evaluate(model, test_loader, device):
    """Fast evaluation without feature extraction."""
    model.eval()
    
    correct = 0
    total = 0
    running_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = criterion(logits, y)
            
            running_loss += loss.item()
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += y.size(0)
    
    val_loss = running_loss / len(test_loader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_thresholds', type=int, default=16,
                       help='Number of thresholds used in preprocessing')
    parser.add_argument('--num_clauses', type=int, default=512)
    parser.add_argument('--attention_heads', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("⚡ FAST FASHION-MNIST TRAINING WITH PREPROCESSED FEATURES")
    print("="*70)
    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}")
    print(f"        num_clauses={args.num_clauses}, attention_heads={args.attention_heads}")
    print(f"        num_thresholds={args.num_thresholds} (preprocessed)")
    print("="*70)
    
    # Load preprocessed datasets
    train_path = f'./preprocessed_data/fashionmnist_train_{args.num_thresholds}thresh.pt'
    test_path = f'./preprocessed_data/fashionmnist_test_{args.num_thresholds}thresh.pt'
    
    if not os.path.exists(train_path):
        print(f"❌ Preprocessed data not found at {train_path}")
        print(f"   Please run: python preprocess_and_save.py --num_thresholds {args.num_thresholds}")
        return
    
    train_dataset = PreprocessedDataset(train_path)
    test_dataset = PreprocessedDataset(test_path)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    print(f"\nCreating Fast FPTM...")
    model = FastFPTM(
        num_channels=train_dataset.num_channels,
        num_clauses=args.num_clauses,
        attention_heads=args.attention_heads
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Device: {device}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    print("\n" + "="*70)
    print("Starting FAST Training (No Feature Extraction Overhead!)")
    print("="*70)
    
    best_val_acc = 0
    first_epoch_time = None
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc, epoch_time = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Evaluate
        val_loss, val_acc = evaluate(model, test_loader, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Track speedup
        if epoch == 1:
            first_epoch_time = epoch_time
        
        # Print results
        emoji = "🔥" if val_acc > best_val_acc else ""
        print(f"[{epoch:3}/{args.epochs}] "
              f"Train: {train_loss:.3f}/{train_acc:.1f}% | "
              f"Val: {val_loss:.3f}/{val_acc:.1f}% | "
              f"LR: {current_lr:.5f} | "
              f"Time: {epoch_time:.1f}s {emoji}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # Show speedup every 5 epochs
        if epoch % 5 == 0 and first_epoch_time:
            speedup = first_epoch_time / epoch_time
            print(f"\n  ⚡ Speedup: {speedup:.2f}x (Epoch 1: {first_epoch_time:.1f}s → Epoch {epoch}: {epoch_time:.1f}s)\n")
    
    print("\n" + "="*70)
    print("📊 FINAL RESULTS")
    print("-"*70)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    if first_epoch_time:
        final_speedup = first_epoch_time / epoch_time
        print(f"⚡ TRAINING SPEEDUP: {final_speedup:.2f}x")
        print(f"  First epoch: {first_epoch_time:.1f}s")
        print(f"  Last epoch: {epoch_time:.1f}s")
    
    print("="*70)


if __name__ == '__main__':
    main()