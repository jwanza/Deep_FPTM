#!/usr/bin/env python3
"""
Training script for Deep Convolutional Fuzzy Pattern Tsetlin Machine (DC-FPTM).
This revolutionary architecture combines CNN feature learning with Tsetlin interpretability.
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import json
import numpy as np
from typing import Tuple, Dict, Optional

# Import our DC-FPTM model and preprocessing
import sys
sys.path.append(str(Path(__file__).parent.parent))
from fptm.models.dc_fptm import DeepConvTsetlin, create_dc_fptm
from smart_preprocessor import SmartPreprocessor
from torch.utils.data import Dataset


class CPUCachedDataset(Dataset):
    """Dataset that keeps preprocessed data in CPU memory and streams to GPU."""
    
    def __init__(self, data, labels=None, pin_memory=True):
        """
        Args:
            data: preprocessed features (keep on CPU)
            labels: target labels (keep on CPU) 
            pin_memory: whether to use pinned memory for faster GPU transfer
        """
        # Keep data on CPU with optional pinned memory
        if torch.is_tensor(data):
            self.features = data.pin_memory() if pin_memory else data
        else:
            self.features = torch.tensor(data)
            if pin_memory:
                self.features = self.features.pin_memory()
        
        if labels is not None:
            if torch.is_tensor(labels):
                self.labels = labels.pin_memory() if pin_memory else labels
            else:
                self.labels = torch.tensor(labels)
                if pin_memory:
                    self.labels = self.labels.pin_memory()
        else:
            self.labels = None
            
        print(f"📦 CPUCachedDataset created:")
        print(f"   Features: {self.features.shape}, dtype: {self.features.dtype}")
        if self.labels is not None:
            print(f"   Labels: {self.labels.shape}, dtype: {self.labels.dtype}")
        print(f"   Pinned memory: {pin_memory}")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        """Return data on CPU - GPU transfer handled by DataLoader."""
        if self.labels is not None:
            return (
                self.features[idx],
                self.labels[idx]
            )
        else:
            return self.features[idx]


def get_dataloaders(args) -> Tuple[DataLoader, DataLoader]:
    """Get train and test dataloaders with appropriate augmentation or cached preprocessing."""
    
    # Dataset-specific configurations
    dataset_configs = {
        'mnist': {
            'dataset_class': datasets.MNIST,
            'mean': (0.1307,),
            'std': (0.3081,),
            'input_size': 28,
            'num_classes': 10
        },
        'fashionmnist': {
            'dataset_class': datasets.FashionMNIST,
            'mean': (0.2860,),
            'std': (0.3530,),
            'input_size': 28,
            'num_classes': 10
        },
        'cifar10': {
            'dataset_class': datasets.CIFAR10,
            'mean': (0.4914, 0.4822, 0.4465),
            'std': (0.2470, 0.2435, 0.2616),
            'input_size': 32,
            'num_classes': 10
        },
        'cifar100': {
            'dataset_class': datasets.CIFAR100,
            'mean': (0.5071, 0.4867, 0.4408),
            'std': (0.2675, 0.2565, 0.2761),
            'input_size': 32,
            'num_classes': 100
        },
        'svhn': {
            'dataset_class': datasets.SVHN,
            'mean': (0.4377, 0.4438, 0.4728),
            'std': (0.1980, 0.2010, 0.1970),
            'input_size': 32,
            'num_classes': 10
        }
    }
    
    config = dataset_configs[args.dataset]
    
    # Check if using cached preprocessed data
    if args.use_cached:
        print(f"🚀 Using cached preprocessed data for {args.dataset}")
        print(f"   Cache thresholds: {args.cache_thresholds}")
        print(f"   Include inverted: {args.cache_include_inverted}")
        print(f"   Include edges: {args.cache_include_edges}")
        
        # Initialize smart preprocessor
        preprocessor = SmartPreprocessor(args.dataset)
        
        # Get or create preprocessed training data
        print("📊 Loading training data...")
        train_data = preprocessor.get_or_create_preprocessed(
            'train',
            num_thresholds=args.cache_thresholds,
            include_edges=args.cache_include_edges,
            include_inverted=args.cache_include_inverted,
            force_recreate=args.force_recreate_cache
        )
        
        # Get or create preprocessed test data
        print("📊 Loading test data...")
        test_data = preprocessor.get_or_create_preprocessed(
            'test',
            num_thresholds=args.cache_thresholds,
            include_edges=args.cache_include_edges,
            include_inverted=args.cache_include_inverted,
            force_recreate=args.force_recreate_cache
        )
        
        # Extract features and labels
        if isinstance(train_data, dict):
            train_features = train_data['features']
            train_labels = train_data['labels']
            test_features = test_data['features']
            test_labels = test_data['labels']
        else:
            # Fallback: assume it's just features, get labels from original dataset
            train_features = train_data
            test_features = test_data
            
            # Get labels from original dataset
            data_dir = Path(args.data_dir) / args.dataset
            data_dir.mkdir(parents=True, exist_ok=True)
            
            if args.dataset == 'svhn':
                orig_train = config['dataset_class'](root=data_dir, split='train', download=True)
                orig_test = config['dataset_class'](root=data_dir, split='test', download=True)
            else:
                orig_train = config['dataset_class'](root=data_dir, train=True, download=True)
                orig_test = config['dataset_class'](root=data_dir, train=False, download=True)
            
            train_labels = torch.tensor([orig_train[i][1] for i in range(len(orig_train))])
            test_labels = torch.tensor([orig_test[i][1] for i in range(len(orig_test))])
        
        # Create CPU-cached datasets
        train_dataset = CPUCachedDataset(train_features, train_labels, pin_memory=args.pin_memory)
        test_dataset = CPUCachedDataset(test_features, test_labels, pin_memory=args.pin_memory)
        
        print(f"✅ Cached datasets created successfully!")
        
    else:
        # Use standard PyTorch datasets with transforms
        print(f"📊 Using standard datasets with on-the-fly transforms for {args.dataset}")
        
        # Build transforms
        normalize = transforms.Normalize(mean=config['mean'], std=config['std'])
        
        # Training transforms with augmentation
        if args.augmentation and args.dataset in ['cifar10', 'cifar100', 'svhn']:
            train_transform = transforms.Compose([
                transforms.RandomCrop(config['input_size'], padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15) if args.strong_augmentation else transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2) if args.strong_augmentation else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomRotation(5) if args.augmentation else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                normalize,
            ])
        
        # Test transforms (no augmentation)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        
        # Load datasets
        data_dir = Path(args.data_dir) / args.dataset
        data_dir.mkdir(parents=True, exist_ok=True)
        
        if args.dataset == 'svhn':
            train_dataset = config['dataset_class'](
                root=data_dir, split='train', download=True, transform=train_transform
            )
            test_dataset = config['dataset_class'](
                root=data_dir, split='test', download=True, transform=test_transform
            )
        else:
            train_dataset = config['dataset_class'](
                root=data_dir, train=True, download=True, transform=train_transform
            )
            test_dataset = config['dataset_class'](
                root=data_dir, train=False, download=True, transform=test_transform
            )
    
    # Create dataloaders
    # For cached data, reduce num_workers to avoid CUDA multiprocessing issues
    num_workers = 0 if args.use_cached else args.num_workers
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Always use pin_memory for fast transfer
        drop_last=True  # For consistent batch sizes
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True  # Always use pin_memory for fast transfer
    )
    
    return train_loader, test_loader, config['num_classes']


def train_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch, args):
    """Train for one epoch with mixed precision and gradient accumulation."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Annealing schedule for binarization
    if epoch % args.anneal_interval == 0:
        model.anneal_binarization(factor=args.anneal_factor)
        print(f"Annealed binarization temperature at epoch {epoch}")
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        # Mixed precision forward pass
        with autocast(enabled=args.mixed_precision):
            output = model(data)
            loss = criterion(output, target)
            
            # Scale loss for gradient accumulation
            loss = loss / args.gradient_accumulation
        
        # Backward pass
        if args.mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation step
        if (batch_idx + 1) % args.gradient_accumulation == 0:
            if args.mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            
            # Tsetlin reinforcement (if enabled)
            if args.use_reinforcement and batch_idx % args.reinforce_interval == 0:
                with torch.no_grad():
                    predictions = output.argmax(dim=1)
                    model.reinforce(data, target, predictions)
        
        # Statistics
        running_loss += loss.item() * args.gradient_accumulation
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Progress reporting
        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} '
                  f'({100.*batch_idx/len(train_loader):.0f}%)]\t'
                  f'Loss: {running_loss/(batch_idx+1):.6f}\t'
                  f'Acc: {100.*correct/total:.2f}%')
            
        # Memory cleanup
        if args.cleanup_interval > 0 and batch_idx % args.cleanup_interval == 0:
            torch.cuda.empty_cache()
    
    return running_loss / len(train_loader), 100. * correct / total


def test_epoch(model, test_loader, criterion, device, args, return_explanation=False):
    """Evaluate model on test set."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    explanations = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            if return_explanation and len(explanations) < args.num_explanations:
                # Get explanations for first few batches
                output, explanation = model(data, return_explanation=True)
                explanations.append(model.get_interpretable_summary(data[:1]))  # Just first sample
            else:
                output = model(data)
            
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    
    if return_explanation:
        return test_loss, accuracy, explanations
    return test_loss, accuracy


def save_checkpoint(model, optimizer, epoch, best_acc, args, is_best=False):
    """Save model checkpoint."""
    checkpoint_dir = Path(args.checkpoint_dir) / args.dataset
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'args': vars(args)
    }
    
    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = checkpoint_dir / 'best_model.pt'
        torch.save(checkpoint, best_path)
        print(f"Saved best model with accuracy: {best_acc:.2f}%")
    
    # Keep only last N checkpoints
    checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
    if len(checkpoints) > args.keep_checkpoints:
        for old_checkpoint in checkpoints[:-args.keep_checkpoints]:
            old_checkpoint.unlink()


def main():
    parser = argparse.ArgumentParser(description='Train DC-FPTM Model')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['mnist', 'fashionmnist', 'cifar10', 'cifar100', 'svhn'],
                        help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory to store datasets')
    
    # Model arguments
    parser.add_argument('--cnn_channels', type=int, nargs='+', default=None,
                        help='CNN channel progression (e.g., 64 128 256)')
    parser.add_argument('--num_thresholds', type=int, nargs='+', default=None,
                        help='Number of thresholds per scale (e.g., 8 16 32)')
    parser.add_argument('--tsetlin_clauses', type=int, nargs='+', default=None,
                        help='Number of clauses per scale (e.g., 256 512 1024)')
    parser.add_argument('--automata_states', type=int, default=50,
                        help='Number of automata states')
    parser.add_argument('--attention_heads', type=int, default=8,
                        help='Number of attention heads for cross-scale reasoning')
    parser.add_argument('--use_cross_scale', action='store_true',
                        help='Use cross-scale attention')
    
    # Julia parameters
    parser.add_argument('--T', type=int, default=100, help='Decision threshold')
    parser.add_argument('--s', type=float, default=3.0, help='Reinforcement strength')
    parser.add_argument('--L', type=int, default=16, help='Learning sensitivity')
    parser.add_argument('--lf', type=int, default=200, help='Leakage factor')
    parser.add_argument('--include_limit', type=int, default=128, help='Include limit')
    parser.add_argument('--use_julia_eval', action='store_true',
                        help='Use Julia-style evaluation')
    parser.add_argument('--use_discrete', action='store_true',
                        help='Use discrete (hard) binarization mode')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=100,
                        help='Test batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--gradient_accumulation', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Augmentation arguments
    parser.add_argument('--augmentation', action='store_true',
                        help='Use data augmentation')
    parser.add_argument('--strong_augmentation', action='store_true',
                        help='Use strong data augmentation')
    
    # Cached preprocessing arguments
    parser.add_argument('--use_cached', action='store_true',
                        help='Use preprocessed cached data for 7.5x faster loading')
    parser.add_argument('--cache_thresholds', type=int, default=16,
                        help='Number of thresholds for cached binary data')
    parser.add_argument('--cache_include_inverted', action='store_true', default=True,
                        help='Include inverted features in cached data')
    parser.add_argument('--cache_include_edges', action='store_true',
                        help='Include edge features in cached data')
    parser.add_argument('--force_recreate_cache', action='store_true',
                        help='Force recreation of cached data even if it exists')
    parser.add_argument('--pin_memory', action='store_true', default=True,
                        help='Use pinned memory for faster CPU-GPU transfer')
    
    # Annealing arguments
    parser.add_argument('--anneal_interval', type=int, default=5,
                        help='Epochs between temperature annealing')
    parser.add_argument('--anneal_factor', type=float, default=0.9,
                        help='Temperature annealing factor')
    
    # Reinforcement arguments
    parser.add_argument('--use_reinforcement', action='store_true',
                        help='Use Tsetlin reinforcement learning')
    parser.add_argument('--reinforce_interval', type=int, default=10,
                        help='Batches between reinforcement')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--cleanup_interval', type=int, default=50,
                        help='Batches between GPU cache cleanup')
    
    # Logging arguments
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Batches between logging')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--keep_checkpoints', type=int, default=5,
                        help='Number of checkpoints to keep')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Epochs between checkpoint saves')
    
    # Explanation arguments
    parser.add_argument('--explain_interval', type=int, default=20,
                        help='Epochs between generating explanations')
    parser.add_argument('--num_explanations', type=int, default=5,
                        help='Number of sample explanations to generate')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading {args.dataset} dataset...")
    train_loader, test_loader, num_classes = get_dataloaders(args)
    
    # Create model
    print("Creating DC-FPTM model...")
    model_kwargs = {
        'num_classes': num_classes,
        'automata_states': args.automata_states,
        'T': args.T,
        's': args.s,
        'L': args.L,
        'lf': args.lf,
        'include_limit': args.include_limit,
        'use_julia_eval': args.use_julia_eval,
        'use_discrete': args.use_discrete,
        'use_cross_scale': args.use_cross_scale,
        'attention_heads': args.attention_heads,
        'dropout': args.dropout
    }
    
    # Adjust input channels for cached data
    if args.use_cached:
        # Cached data has cache_thresholds * (2 if inverted else 1) channels
        input_channels = args.cache_thresholds
        if args.cache_include_inverted:
            input_channels *= 2
        if args.cache_include_edges:
            input_channels += 2  # Sobel x and y
        model_kwargs['input_channels'] = input_channels
        print(f"   Cached data input channels: {input_channels}")
    
    # Add custom architecture if specified
    if args.cnn_channels:
        model_kwargs['cnn_channels'] = args.cnn_channels
    if args.num_thresholds:
        model_kwargs['num_thresholds'] = args.num_thresholds
    if args.tsetlin_clauses:
        model_kwargs['tsetlin_clauses'] = args.tsetlin_clauses
    
    model = create_dc_fptm(args.dataset, **model_kwargs).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer and loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision scaler
    scaler = GradScaler() if args.mixed_precision else None
    
    # Training loop
    print("\nStarting training...")
    print("="*60)
    
    best_acc = 0.0
    training_history = []
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch, args
        )
        
        # Test
        if epoch % args.explain_interval == 0:
            test_loss, test_acc, explanations = test_epoch(
                model, test_loader, criterion, device, args, return_explanation=True
            )
            # Print sample explanations
            print("\nSample Explanations:")
            for i, exp in enumerate(explanations[:2]):
                print(f"Sample {i+1}:")
                print(exp['decision_process'][0])
                print("-"*40)
        else:
            test_loss, test_acc = test_epoch(
                model, test_loader, criterion, device, args
            )
        
        # Update learning rate
        scheduler.step()
        
        # Track history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'lr': scheduler.get_last_lr()[0]
        })
        
        # Save checkpoint
        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
        
        if epoch % args.save_interval == 0 or is_best:
            save_checkpoint(model, optimizer, epoch, best_acc, args, is_best)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch}/{args.epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print(f"  Best Test Acc: {best_acc:.2f}%")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print(f"  Epoch Time: {epoch_time:.2f}s")
        print("="*60)
    
    # Save training history
    history_path = Path(args.checkpoint_dir) / args.dataset / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Training history saved to: {history_path}")


if __name__ == '__main__':
    main()
