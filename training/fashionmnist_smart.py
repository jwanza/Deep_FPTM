"""
Smart Fashion-MNIST training that automatically uses cached preprocessed features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import time
import argparse
import sys
sys.path.append('.')

from fptm.models.fptm_conv_julia import FPTMConvJulia
from fptm.utils import set_seed
from smart_preprocessor import SmartPreprocessor, CachedDataset


class SmartFPTM(nn.Module):
    """
    Smart FPTM that supports both binary (preprocessed) and continuous modes.
    """
    def __init__(self, num_channels, num_clauses=512, attention_heads=0, 
                 automata_states=50, continuous_mode=False, normalize_mode="minmax",
                 T=100, s=3.0, L=16, lf=200, include_limit=128,
                 use_julia_eval=False, use_julia_kernels=False):
        super().__init__()
        
        self.continuous_mode = continuous_mode
        self.num_channels = num_channels
        
        if continuous_mode:
            # Continuous mode: raw images, no channel mixer needed
            self.channel_mixer = None
            input_channels = num_channels  # Usually 1 for grayscale
        else:
            # Binary mode: preprocessed features, need channel mixer
            self.channel_mixer = nn.Conv2d(num_channels, 1, kernel_size=1)
            input_channels = 1
        
        # FPTM model with ALL configurable parameters
        self.fptm = FPTMConvJulia(
            in_channels=input_channels,
            image_size=28,
            patch_size=4,
            num_clauses=num_clauses,
            num_classes=10,
            attention_heads=attention_heads,
            epsilon=1e-6,
            automata_states=automata_states,  # Now configurable!
            normalize_mode=normalize_mode if continuous_mode else "none",  # Use normalization only in continuous mode
            T=T,  # Decision threshold for voting
            s=s,  # Reinforcement strength
            L=L,  # Learning sensitivity
            lf=lf,  # Leakage factor for discrete fuzzy
            include_limit=include_limit,  # Literal inclusion threshold
            use_julia_eval=use_julia_eval,  # Discrete vs continuous fuzzy
            use_julia_kernels=use_julia_kernels  # Vision-specific kernels
        )
    
    def forward(self, x):
        if self.continuous_mode:
            # Direct pass for continuous images
            return self.fptm(x)
        else:
            # Mix channels for binary features
            mixed = self.channel_mixer(x)
            return self.fptm(mixed)
    
    def reinforce(self, x, y_true, y_pred, s=3.0):
        if self.continuous_mode:
            self.fptm.reinforce(x, y_true, y_pred, s)
        else:
            mixed = self.channel_mixer(x)
            self.fptm.reinforce(mixed, y_true, y_pred, s)


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, verbose=False):
    """Training epoch with optional verbose output."""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Reinforcement learning step (every 3rd batch for efficiency)
        if batch_idx % 3 == 0:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                model.reinforce(x, y, preds, s=3.0)
        
        # Statistics
        running_loss += loss.item()
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += y.size(0)
        
        # Progress (less verbose than before)
        if verbose and batch_idx % 100 == 0 and batch_idx > 0:
            acc = 100 * correct / total
            print(f"    Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.3f}, Acc={acc:.1f}%")
    
    epoch_time = time.time() - start_time
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    
    return train_loss, train_acc, epoch_time


def evaluate(model, test_loader, device):
    """Evaluation without gradients."""
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
    # Mode selection
    parser.add_argument('--continuous', action='store_true',
                       help='Use continuous (raw) images instead of binary features')
    parser.add_argument('--normalize_mode', type=str, default='minmax',
                       choices=['none', 'minmax', 'global'],
                       help='Normalization mode for continuous images')
    
    # Preprocessing parameters (only for binary mode)
    parser.add_argument('--num_thresholds', type=int, default=8,
                       help='Number of thresholds for binarization (binary mode only)')
    parser.add_argument('--threshold_min', type=float, default=0.1,
                       help='Minimum threshold value')
    parser.add_argument('--threshold_max', type=float, default=0.9,
                       help='Maximum threshold value')
    parser.add_argument('--include_edges', action='store_true',
                       help='Include edge features')
    parser.add_argument('--no_inverted', action='store_true',
                       help='Do not include inverted features')
    
    # Model parameters
    parser.add_argument('--num_clauses', type=int, default=512)
    parser.add_argument('--attention_heads', type=int, default=0)
    parser.add_argument('--automata_states', type=int, default=50,
                       help='Number of automata states (50, 100, 256, etc.)')
    
    # Advanced Tsetlin parameters
    parser.add_argument('--T', type=int, default=100,
                       help='Decision threshold for clause voting')
    parser.add_argument('--s', type=float, default=3.0,
                       help='Base reinforcement strength')
    parser.add_argument('--L', type=int, default=16,
                       help='Learning sensitivity factor')
    parser.add_argument('--lf', type=int, default=200,
                       help='Leakage factor for discrete fuzzy (Julia-style)')
    parser.add_argument('--include_limit', type=int, default=128,
                       help='State threshold for literal inclusion')
    parser.add_argument('--use_julia_eval', action='store_true',
                       help='Use discrete Julia-style evaluation (counting)')
    parser.add_argument('--use_julia_kernels', action='store_true',
                       help='Use Julia vision kernels for feature extraction')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')
    
    # Smart preprocessing
    parser.add_argument('--force_recreate', action='store_true',
                       help='Force recreate preprocessed data')
    parser.add_argument('--list_cached', action='store_true',
                       help='List cached datasets and exit')
    
    args = parser.parse_args()
    
    # Initialize smart preprocessor
    preprocessor = SmartPreprocessor()
    
    if args.list_cached:
        preprocessor.list_cached()
        return
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("⚡ SMART FASHION-MNIST TRAINING")
    print("="*70)
    print(f"Mode: {'CONTINUOUS' if args.continuous else 'BINARY'}")
    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}")
    print(f"        num_clauses={args.num_clauses}, attention_heads={args.attention_heads}")
    print(f"        automata_states={args.automata_states}, T={args.T}, s={args.s}, L={args.L}")
    if args.use_julia_eval:
        print(f"        Using Julia discrete evaluation (LF={args.lf}, include_limit={args.include_limit})")
    if args.use_julia_kernels:
        print(f"        Using Julia vision kernels (8x feature expansion)")
    
    if args.continuous:
        print(f"        normalize_mode={args.normalize_mode}")
    else:
        print(f"        thresholds={args.num_thresholds} ({args.threshold_min:.2f} to {args.threshold_max:.2f})")
        print(f"        edges={args.include_edges}, inverted={not args.no_inverted}")
    print("="*70)
    
    if args.continuous:
        # Continuous mode: Also use caching for raw images
        from torchvision import datasets, transforms
        import os
        
        print("\n📊 CONTINUOUS mode - checking for cached raw data...")
        
        # Create cache filenames for continuous mode
        cache_dir = preprocessor.base_dir
        train_cache_file = os.path.join(cache_dir, f"fashionmnist_train_continuous_{args.normalize_mode}.pt")
        test_cache_file = os.path.join(cache_dir, f"fashionmnist_test_continuous_{args.normalize_mode}.pt")
        
        # Check if continuous data is already cached
        if os.path.exists(train_cache_file) and os.path.exists(test_cache_file) and not args.force_recreate:
            print(f"✅ Found cached continuous data!")
            print(f"   Loading: {os.path.basename(train_cache_file)}")
            train_data = torch.load(train_cache_file, map_location='cpu')
            print(f"   Loading: {os.path.basename(test_cache_file)}")
            test_data = torch.load(test_cache_file, map_location='cpu')
            
            # Create simple datasets
            class ContinuousDataset(torch.utils.data.Dataset):
                def __init__(self, data_dict):
                    self.images = data_dict['images']
                    self.labels = data_dict['labels']
                
                def __len__(self):
                    return len(self.labels)
                
                def __getitem__(self, idx):
                    return self.images[idx], self.labels[idx]
            
            train_dataset = ContinuousDataset(train_data)
            test_dataset = ContinuousDataset(test_data)
            
        else:
            # Load and cache raw Fashion-MNIST
            print(f"📥 Loading and caching raw Fashion-MNIST...")
            transform = transforms.Compose([transforms.ToTensor()])
            
            train_raw = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
            test_raw = datasets.FashionMNIST('./data', train=False, transform=transform)
            
            # Convert to tensors for caching
            print("   Converting to tensors...")
            train_images = torch.stack([img for img, _ in train_raw])
            train_labels = torch.tensor([label for _, label in train_raw])
            test_images = torch.stack([img for img, _ in test_raw])
            test_labels = torch.tensor([label for _, label in test_raw])
            
            # Save to cache
            print(f"💾 Saving continuous mode cache...")
            torch.save({
                'images': train_images,
                'labels': train_labels,
                'normalize_mode': args.normalize_mode,
                'mode': 'continuous'
            }, train_cache_file)
            
            torch.save({
                'images': test_images,
                'labels': test_labels,
                'normalize_mode': args.normalize_mode,
                'mode': 'continuous'
            }, test_cache_file)
            
            print(f"   Saved: {os.path.basename(train_cache_file)} ({os.path.getsize(train_cache_file)/(1024**2):.1f}MB)")
            print(f"   Saved: {os.path.basename(test_cache_file)} ({os.path.getsize(test_cache_file)/(1024**2):.1f}MB)")
            
            # Create datasets from cached data
            class ContinuousDataset(torch.utils.data.Dataset):
                def __init__(self, images, labels):
                    self.images = images
                    self.labels = labels
                
                def __len__(self):
                    return len(self.labels)
                
                def __getitem__(self, idx):
                    return self.images[idx], self.labels[idx]
            
            train_dataset = ContinuousDataset(train_images, train_labels)
            test_dataset = ContinuousDataset(test_images, test_labels)
        
        num_channels = 1  # Grayscale images
        print(f"   Train: {len(train_dataset)} samples")
        print(f"   Test: {len(test_dataset)} samples")
        
    else:
        # Binary mode: Use preprocessed features
        print("\n📊 Loading preprocessed features for BINARY mode...")
        
        # Get or create preprocessed data
        train_data = preprocessor.get_or_create_preprocessed(
            'train',
            num_thresholds=args.num_thresholds,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            include_edges=args.include_edges,
            include_inverted=not args.no_inverted,
            force_recreate=args.force_recreate
        )
        
        test_data = preprocessor.get_or_create_preprocessed(
            'test',
            num_thresholds=args.num_thresholds,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            include_edges=args.include_edges,
            include_inverted=not args.no_inverted,
            force_recreate=args.force_recreate
        )
        
        # Create datasets
        train_dataset = CachedDataset(train_data)
        test_dataset = CachedDataset(test_data)
        num_channels = train_dataset.num_channels
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Create model
    print(f"\nCreating Smart FPTM...")
    model = SmartFPTM(
        num_channels=num_channels,
        num_clauses=args.num_clauses,
        attention_heads=args.attention_heads,
        automata_states=args.automata_states,
        continuous_mode=args.continuous,
        normalize_mode=args.normalize_mode,
        T=args.T,
        s=args.s,
        L=args.L,
        lf=args.lf,
        include_limit=args.include_limit,
        use_julia_eval=args.use_julia_eval,
        use_julia_kernels=args.use_julia_kernels
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Device: {device}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    print("\n" + "="*70)
    print("Starting Training (Using Cached Features = FAST!)")
    print("="*70)
    
    best_val_acc = 0
    first_epoch_time = None
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc, epoch_time = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args.verbose
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
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'args': args
            }, 'best_model.pt')
        
        # Show speedup every 10 epochs
        if epoch % 10 == 0 and first_epoch_time:
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
    
    print(f"\n💾 Best model saved to: best_model.pt")
    print("="*70)


if __name__ == '__main__':
    main()
