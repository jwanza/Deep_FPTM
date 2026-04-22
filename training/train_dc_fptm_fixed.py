#!/usr/bin/env python3
"""
Training script for FIXED DC-FPTM with working binarization
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms

import sys
sys.path.append(str(Path(__file__).parent.parent))

from fptm.models import create_fixed_dc_fptm


def get_data_loaders(dataset_name: str, batch_size: int, num_workers: int = 4):
    """Get data loaders for specified dataset"""
    
    # Common transform
    if dataset_name in ['mnist', 'fashionmnist']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        transform_test = transform
    else:  # CIFAR
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    # Load datasets
    if dataset_name == 'mnist':
        train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('data', train=False, transform=transform_test)
    elif dataset_name == 'fashionmnist':
        train_dataset = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST('data', train=False, transform=transform_test)
    elif dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('data', train=False, transform=transform_test)
    elif dataset_name == 'cifar100':
        train_dataset = datasets.CIFAR100('data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100('data', train=False, transform=transform_test)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    mixed_precision: bool = False
) -> Tuple[float, float]:
    """Train for one epoch"""
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        if mixed_precision and scaler is not None:
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            acc = 100. * correct / total
            print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} '
                  f'({100.*batch_idx/len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}\tAcc: {acc:.2f}%')
            
            # Check binary statistics on first batch
            if batch_idx == 0 and hasattr(model, 'get_binary_stats'):
                with torch.no_grad():
                    stats = model.get_binary_stats(data[:4])
                    print(f"  Binary sparsity: {stats['binary_0']['sparsity']:.1%}")
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def test_epoch(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Test the model"""
    
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train Fixed DC-FPTM')
    parser.add_argument('--dataset', type=str, default='fashionmnist',
                        choices=['mnist', 'fashionmnist', 'cifar10', 'cifar100'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--simplified', action='store_true', default=False,
                        help='Use simplified model configuration')
    parser.add_argument('--mixed_precision', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--anneal_interval', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='checkpoints_fixed')
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print(f"Creating Fixed DC-FPTM for {args.dataset}...")
    model = create_fixed_dc_fptm(args.dataset, simplified=args.simplified).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Data loaders
    print(f"Loading {args.dataset} dataset...")
    train_loader, test_loader = get_data_loaders(
        args.dataset, 
        args.batch_size,
        args.num_workers
    )
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    scaler = GradScaler() if args.mixed_precision else None
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'lr': []
    }
    
    best_acc = 0
    
    print("\nStarting training...")
    print("="*60)
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, 
            epoch, device, scaler, args.mixed_precision
        )
        
        # Test
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'args': args
            }, os.path.join(args.save_dir, f'{args.dataset}_best.pth'))
            print(f"Saved best model with accuracy: {test_acc:.2f}%")
        
        # Anneal binarization
        if epoch % args.anneal_interval == 0:
            model.anneal_binarization(factor=0.9)
            print("Annealed binarization temperature")
        
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch}/{args.epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print(f"  Best Test Acc: {best_acc:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Epoch Time: {epoch_time:.2f}s")
        print("="*60)
    
    # Save training history
    with open(os.path.join(args.save_dir, f'{args.dataset}_history.json'), 'w') as f:
        json.dump(history, f)
    
    print("\nTraining completed!")
    print(f"Best test accuracy: {best_acc:.2f}%")
    
    # Check prediction diversity
    print("\nChecking prediction diversity...")
    model.eval()
    all_preds = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            if len(all_preds) >= 100:
                break
    
    unique_preds = len(set(all_preds[:100]))
    num_classes = 10 if args.dataset != 'cifar100' else 100
    print(f"Unique predictions in first 100 samples: {unique_preds}/{num_classes}")
    
    # Final verdict
    print("\n" + "="*60)
    if best_acc > 80 and args.dataset in ['mnist']:
        print(f"🎉 EXCELLENT! Best accuracy: {best_acc:.2f}%")
    elif best_acc > 70 and args.dataset in ['fashionmnist']:
        print(f"🎉 GREAT! Best accuracy: {best_acc:.2f}%")
    elif best_acc > 50:
        print(f"✅ GOOD! Best accuracy: {best_acc:.2f}%")
    elif best_acc > 20:
        print(f"⚠️ LEARNING BUT NEEDS TUNING. Best accuracy: {best_acc:.2f}%")
    else:
        print(f"❌ POOR PERFORMANCE. Best accuracy: {best_acc:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()
