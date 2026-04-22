#!/usr/bin/env python3
"""
Memory-Optimized Fashion-MNIST Training
Leverages CPU memory (251GB) and implements proper GPU memory management
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import argparse
import time
import gc
from contextlib import nullcontext

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fptm.models import FPTMConvJulia
from smart_preprocessor import SmartPreprocessor


class CPUCachedDataset(Dataset):
    """Dataset that keeps data in CPU memory and only sends batches to GPU."""
    
    def __init__(self, data, labels=None, pin_memory=True):
        """
        Args:
            data: preprocessed features (keep on CPU)
            labels: target labels (keep on CPU)
            pin_memory: whether to use pinned memory for faster GPU transfer
        """
        # CRITICAL: Keep on CPU, not GPU!
        if torch.is_tensor(data) and data.is_cuda:
            data = data.cpu()
        if labels is not None and torch.is_tensor(labels) and labels.is_cuda:
            labels = labels.cpu()
            
        # Use pinned memory for faster GPU transfer
        if pin_memory and torch.cuda.is_available():
            self.data = data.pin_memory() if torch.is_tensor(data) else torch.tensor(data).pin_memory()
            self.labels = labels.pin_memory() if labels is None else (
                labels.pin_memory() if torch.is_tensor(labels) else torch.tensor(labels).pin_memory()
            )
        else:
            self.data = data if torch.is_tensor(data) else torch.tensor(data)
            self.labels = labels if labels is None else (
                labels if torch.is_tensor(labels) else torch.tensor(labels)
            )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return CPU tensors - DataLoader will handle GPU transfer
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx], idx


class MemoryOptimizedFPTM(nn.Module):
    """Memory-optimized wrapper around FPTMConvJulia."""
    
    def __init__(self, *args, gradient_checkpointing=False, **kwargs):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.fptm = FPTMConvJulia(*args, **kwargs)
        
        # Pre-allocate buffers to avoid repeated allocation
        self.register_buffer('_forward_cache', None, persistent=False)
    
    def forward(self, x):
        if self.gradient_checkpointing and self.training:
            # Use gradient checkpointing to trade compute for memory
            import torch.utils.checkpoint as cp
            return cp.checkpoint(self.fptm.forward, x, use_reentrant=False)
        else:
            return self.fptm(x)
    
    def reinforce(self, *args, **kwargs):
        return self.fptm.reinforce(*args, **kwargs)


def train_epoch_memory_optimized(
    model, train_loader, optimizer, criterion, device, epoch,
    gradient_accumulation_steps=1, mixed_precision=False, 
    memory_cleanup_interval=10, max_gpu_memory_mb=None
):
    """
    Memory-optimized training epoch.
    
    Args:
        gradient_accumulation_steps: Accumulate gradients over multiple batches
        mixed_precision: Use AMP for lower memory usage
        memory_cleanup_interval: Clean GPU memory every N batches
        max_gpu_memory_mb: Maximum GPU memory to use (in MB)
    """
    model.train()
    
    # Mixed precision setup
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    autocast_ctx = torch.cuda.amp.autocast if mixed_precision else nullcontext
    
    # Set memory limit if specified
    if max_gpu_memory_mb and torch.cuda.is_available():
        max_memory_bytes = max_gpu_memory_mb * 1024 * 1024
        torch.cuda.set_per_process_memory_fraction(
            max_memory_bytes / torch.cuda.get_device_properties(0).total_memory
        )
    
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()
    
    start_time = time.time()
    
    for batch_idx, (x, y) in enumerate(train_loader):
        # Move to GPU only when needed
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        # Forward pass with optional mixed precision
        with autocast_ctx():
            logits = model(x)
            loss = criterion(logits, y) / gradient_accumulation_steps
        
        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights only after accumulating gradients
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # Reinforcement learning (less frequent to save memory)
        if batch_idx % (gradient_accumulation_steps * 3) == 0:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                model.reinforce(x, y, preds)
        
        # Statistics
        with torch.no_grad():
            running_loss += loss.item() * gradient_accumulation_steps
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += y.size(0)
        
        # Memory cleanup
        if batch_idx % memory_cleanup_interval == 0:
            # Delete intermediate tensors
            del x, y, logits, loss
            
            # Force garbage collection
            if batch_idx % (memory_cleanup_interval * 5) == 0:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        # Progress
        if batch_idx % 50 == 0 and batch_idx > 0:
            # Get memory stats
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated(device) / 1024**2
                mem_reserved = torch.cuda.memory_reserved(device) / 1024**2
                print(f"  Batch {batch_idx}/{len(train_loader)}: "
                      f"Loss={running_loss/(batch_idx+1):.3f}, "
                      f"Acc={100*correct/total:.1f}%, "
                      f"Mem={mem_allocated:.0f}/{mem_reserved:.0f}MB")
    
    # Final cleanup
    torch.cuda.empty_cache()
    
    epoch_time = time.time() - start_time
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    
    return train_loss, train_acc, epoch_time


def evaluate_memory_optimized(model, test_loader, device, mixed_precision=False):
    """Memory-optimized evaluation."""
    model.eval()
    
    autocast_ctx = torch.cuda.amp.autocast if mixed_precision else nullcontext
    
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            with autocast_ctx():
                logits = model(x)
                loss = criterion(logits, y)
            
            running_loss += loss.item()
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += y.size(0)
            
            # Cleanup every 10 batches
            if batch_idx % 10 == 0:
                del x, y, logits, loss
    
    torch.cuda.empty_cache()
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100 * correct / total
    
    return test_loss, test_acc


def main():
    parser = argparse.ArgumentParser(description='Memory-Optimized Fashion-MNIST Training')
    
    # Model parameters
    parser.add_argument('--num_clauses', type=int, default=512, help='Number of clauses')
    parser.add_argument('--automata_states', type=int, default=8, help='Automata states (8 works well!)')
    parser.add_argument('--attention_heads', type=int, default=0, help='Number of attention heads')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    
    # Memory optimization
    parser.add_argument('--gradient_accumulation', type=int, default=2, 
                       help='Gradient accumulation steps')
    parser.add_argument('--mixed_precision', action='store_true', 
                       help='Use mixed precision training')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                       help='Use gradient checkpointing')
    parser.add_argument('--max_gpu_memory_mb', type=int, default=None,
                       help='Maximum GPU memory to use (MB)')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='DataLoader workers (use more with 72 CPUs!)')
    parser.add_argument('--pin_memory', action='store_true', default=True,
                       help='Pin memory for faster GPU transfer')
    
    # Preprocessing
    parser.add_argument('--num_thresholds', type=int, default=16, help='Number of thresholds')
    parser.add_argument('--continuous', action='store_true', help='Use continuous mode')
    
    # Julia parameters
    parser.add_argument('--use_julia_eval', action='store_true', help='Use Julia-style evaluation')
    parser.add_argument('--T', type=int, default=100, help='Decision threshold')
    parser.add_argument('--s', type=float, default=3.5, help='Reinforcement strength')
    parser.add_argument('--L', type=int, default=20, help='Learning sensitivity')
    parser.add_argument('--lf', type=int, default=200, help='Leakage factor')
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("🚀 MEMORY-OPTIMIZED FASHION-MNIST TRAINING")
    print("="*70)
    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}")
    print(f"        num_clauses={args.num_clauses}, automata_states={args.automata_states}")
    print(f"        gradient_accumulation={args.gradient_accumulation}")
    print(f"        mixed_precision={args.mixed_precision}")
    print(f"        num_workers={args.num_workers} (using those 72 CPUs!)")
    print("="*70)
    
    # Load preprocessed data (keep on CPU!)
    print("\n📊 Loading data to CPU memory (251GB available!)...")
    
    if args.continuous:
        # Use raw Fashion-MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST('./data', train=False, transform=transform)
        num_channels = 1
    else:
        # Use preprocessed features
        preprocessor = SmartPreprocessor('fashionmnist')
        
        train_features = preprocessor.get_or_create_preprocessed(
            'train',
            num_thresholds=args.num_thresholds,
            include_inverted=True
        )
        test_features = preprocessor.get_or_create_preprocessed(
            'test',
            num_thresholds=args.num_thresholds,
            include_inverted=True
        )
        
        # Get labels
        train_labels = datasets.FashionMNIST('./data', train=True, download=True).targets
        test_labels = datasets.FashionMNIST('./data', train=False).targets
        
        # Create CPU-cached datasets
        train_dataset = CPUCachedDataset(
            train_features, train_labels, 
            pin_memory=args.pin_memory
        )
        test_dataset = CPUCachedDataset(
            test_features, test_labels,
            pin_memory=args.pin_memory
        )
        num_channels = args.num_thresholds * 2  # *2 for inverted
    
    print(f"  Train samples: {len(train_dataset):,} (on CPU)")
    print(f"  Test samples: {len(test_dataset):,} (on CPU)")
    
    # Create data loaders with proper settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,  # Use multiple workers
        pin_memory=args.pin_memory,     # Pin memory for faster transfer
        persistent_workers=True,         # Keep workers alive
        prefetch_factor=2                # Prefetch batches
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,  # Larger batch for eval
        shuffle=False,
        num_workers=args.num_workers // 2,
        pin_memory=args.pin_memory
    )
    
    # Create memory-optimized model
    print(f"\nCreating Memory-Optimized FPTM...")
    model = MemoryOptimizedFPTM(
        in_channels=num_channels,
        image_size=28,
        patch_size=4,
        num_clauses=args.num_clauses,
        num_classes=10,
        attention_heads=args.attention_heads,
        automata_states=args.automata_states,
        gradient_checkpointing=args.gradient_checkpointing,
        T=args.T,
        s=args.s,
        L=args.L,
        lf=args.lf,
        use_julia_eval=args.use_julia_eval
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Device: {device}")
    
    # Memory info
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {total_mem:.1f}GB total")
        if args.max_gpu_memory_mb:
            print(f"Limited to: {args.max_gpu_memory_mb}MB")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    print("\n" + "="*70)
    print("Starting Memory-Optimized Training")
    print("="*70)
    
    best_val_acc = 0
    first_epoch_time = None
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc, epoch_time = train_epoch_memory_optimized(
            model, train_loader, optimizer, criterion, device, epoch,
            gradient_accumulation_steps=args.gradient_accumulation,
            mixed_precision=args.mixed_precision,
            memory_cleanup_interval=10,
            max_gpu_memory_mb=args.max_gpu_memory_mb
        )
        
        # Evaluate
        val_loss, val_acc = evaluate_memory_optimized(
            model, test_loader, device, 
            mixed_precision=args.mixed_precision
        )
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Track speedup
        if epoch == 1:
            first_epoch_time = epoch_time
        
        speedup = first_epoch_time / epoch_time if first_epoch_time else 1.0
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model_memory_opt.pt')
            marker = "🔥"
        else:
            marker = ""
        
        # Print progress
        print(f"[{epoch:3d}/{args.epochs}] "
              f"Train: {train_loss:.3f}/{train_acc:.1f}% | "
              f"Val: {val_loss:.3f}/{val_acc:.1f}% | "
              f"LR: {current_lr:.5f} | "
              f"Time: {epoch_time:.1f}s | "
              f"Speed: {speedup:.2f}x {marker}")
        
        # Memory stats
        if torch.cuda.is_available() and epoch % 5 == 0:
            mem_gb = torch.cuda.max_memory_allocated(device) / 1024**3
            print(f"  Peak GPU memory: {mem_gb:.2f}GB")
            torch.cuda.reset_peak_memory_stats(device)
    
    print("\n" + "="*70)
    print("📊 FINAL RESULTS")
    print("-"*70)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Training speedup: {speedup:.2f}x")
    print("="*70)


if __name__ == "__main__":
    main()
