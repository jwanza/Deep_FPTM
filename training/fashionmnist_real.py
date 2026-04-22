"""
Real Fashion-MNIST training with FPTM - inspired by Julia implementation.
Uses genuine Fashion-MNIST dataset with optional convolutional preprocessing.
"""
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import numpy as np

from fptm.models import FPTMConvFast
from fptm.utils import set_seed
from fptm.heads import compute_ece


class FashionMNISTWithConv:
    """Real Fashion-MNIST dataset with optional convolutional preprocessing."""
    
    def __init__(self, use_conv_features=False):
        self.use_conv_features = use_conv_features
        
        # Define transforms based on whether we use conv features
        if use_conv_features:
            # Similar to Julia: apply edge detection convolutions
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                # Don't normalize - we'll do custom processing
            ])
        else:
            # Standard normalization for Fashion-MNIST
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST stats
            ])
        
        # Download and load Fashion-MNIST
        self.train_dataset = torchvision.datasets.FashionMNIST(
            root='./data', 
            train=True, 
            download=True, 
            transform=self.transform
        )
        
        self.test_dataset = torchvision.datasets.FashionMNIST(
            root='./data', 
            train=False, 
            download=True, 
            transform=self.transform
        )
        
        # Define convolution kernels (similar to Julia implementation)
        if use_conv_features:
            self.setup_conv_kernels()
    
    def setup_conv_kernels(self):
        """Setup convolution kernels for edge detection."""
        # Sobel-like kernels for edge detection (as in Julia code)
        self.kernel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Diagonal edge detectors
        self.kernel_diag1 = torch.tensor([
            [ 0,  1,  2],
            [-1,  0,  1],
            [-2, -1,  0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel_diag2 = torch.tensor([
            [ 2,  1,  0],
            [ 1,  0, -1],
            [ 0, -1, -2]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    def apply_conv_features(self, x):
        """Apply convolutional preprocessing to extract edge features."""
        device = x.device
        
        # Move kernels to same device
        kx = self.kernel_x.to(device)
        ky = self.kernel_y.to(device)
        kd1 = self.kernel_diag1.to(device)
        kd2 = self.kernel_diag2.to(device)
        
        # Apply convolutions
        edge_x = torch.nn.functional.conv2d(x, kx, padding=1)
        edge_y = torch.nn.functional.conv2d(x, ky, padding=1)
        edge_d1 = torch.nn.functional.conv2d(x, kd1, padding=1)
        edge_d2 = torch.nn.functional.conv2d(x, kd2, padding=1)
        
        # Combine features (stack channel-wise)
        features = torch.cat([
            x,  # Original image
            torch.abs(edge_x),  # Horizontal edges
            torch.abs(edge_y),  # Vertical edges  
            torch.abs(edge_d1),  # Diagonal edges 1
            torch.abs(edge_d2),  # Diagonal edges 2
            torch.sqrt(edge_x**2 + edge_y**2),  # Edge magnitude
        ], dim=1)
        
        return features
    
    def get_loaders(self, batch_size=128, num_workers=4):
        """Get data loaders."""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        return train_loader, test_loader


def train_one_epoch(model, opt, loader, device, scheduler=None, reinforce_every=5, use_conv=False, conv_processor=None):
    """Train for one epoch."""
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    ce = nn.CrossEntropyLoss()
    
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        
        # Apply convolutional features if enabled
        if use_conv and conv_processor:
            x = conv_processor(x)
        
        # Forward pass
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = ce(logits, y)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        
        # Update scheduler if using OneCycle
        if scheduler and hasattr(scheduler, 'step') and hasattr(scheduler, 'total_steps'):
            scheduler.step()
        
        # Metrics and reinforcement
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            
            # Reinforcement learning
            if i % reinforce_every == 0:
                current_acc = (preds == y).float().mean().item()
                adaptive_s = 3.0 * (1.0 + max(0, 0.5 - current_acc))
                model.reinforce(x, y, preds, s=adaptive_s)
            
            correct += (preds == y).float().sum().item()
            total += y.size(0)
            loss_sum += float(loss.item()) * y.size(0)
    
    return loss_sum/total, correct/total


@torch.no_grad()
def evaluate(model, loader, device, use_conv=False, conv_processor=None):
    """Evaluate model."""
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    ce = nn.CrossEntropyLoss()
    all_logits, all_labels = [], []
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        # Apply convolutional features if enabled
        if use_conv and conv_processor:
            x = conv_processor(x)
        
        logits = model(x)
        loss = ce(logits, y)
        preds = logits.argmax(dim=-1)
        
        correct += (preds == y).float().sum().item()
        total += y.size(0)
        loss_sum += float(loss.item()) * y.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())
    
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    ece = compute_ece(logits, labels)
    
    return loss_sum/total, correct/total, ece


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--patch_size", type=int, default=4)
    ap.add_argument("--num_clauses", type=int, default=256)
    ap.add_argument("--attention_heads", type=int, default=4)
    ap.add_argument("--use_conv_features", action="store_true", 
                    help="Use convolutional edge detection features like Julia implementation")
    ap.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    print("=" * 60)
    print("FPTM Training on Real Fashion-MNIST")
    print("=" * 60)
    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}")
    print(f"        num_clauses={args.num_clauses}, attention_heads={args.attention_heads}")
    print(f"        lr={args.lr}, scheduler={args.scheduler}")
    print(f"        use_conv_features={args.use_conv_features}")
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Fashion-MNIST
    print("\nLoading Fashion-MNIST dataset...")
    data = FashionMNISTWithConv(use_conv_features=args.use_conv_features)
    train_loader, test_loader = data.get_loaders(batch_size=args.batch_size)
    print(f"Training samples: {len(train_loader.dataset):,}")
    print(f"Test samples: {len(test_loader.dataset):,}")
    
    # Determine input channels
    if args.use_conv_features:
        in_channels = 6  # Original + 5 edge features
        conv_processor = data.apply_conv_features
    else:
        in_channels = 1  # Just grayscale
        conv_processor = None
    
    # Create model
    model = FPTMConvFast(
        in_channels=in_channels,
        image_size=28,
        patch_size=args.patch_size,
        num_clauses=args.num_clauses,
        num_classes=10,
        attention_heads=args.attention_heads,
        normalize_mode="none" if args.use_conv_features else "minmax"
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    
    # Setup optimizer and scheduler
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    else:
        scheduler = None
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    best_acc = 0
    best_epoch = 0
    total_start = time.time()
    
    # Fashion-MNIST class names for analysis
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    for epoch in range(1, args.epochs + 1):
        # Train
        epoch_start = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, opt, train_loader, device, scheduler, 
            use_conv=args.use_conv_features, conv_processor=conv_processor
        )
        
        # Step scheduler
        if scheduler and isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        
        # Evaluate
        va_loss, va_acc, ece = evaluate(
            model, test_loader, device,
            use_conv=args.use_conv_features, conv_processor=conv_processor
        )
        epoch_time = time.time() - epoch_start
        
        # Track best
        if va_acc > best_acc:
            best_acc = va_acc
            best_epoch = epoch
            # Could save checkpoint here
        
        # Print progress
        current_lr = opt.param_groups[0]['lr']
        print(f"[{epoch:3d}/{args.epochs}] "
              f"Train: {tr_loss:.3f}/{tr_acc:.1%} | "
              f"Val: {va_loss:.3f}/{va_acc:.1%} | "
              f"ECE: {ece:.3f} | "
              f"LR: {current_lr:.5f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Early stopping
        if epoch > 20 and va_acc < best_acc - 0.05:
            print("Early stopping triggered")
            break
    
    # Final results
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Best validation accuracy: {best_acc:.2%} at epoch {best_epoch}")
    print(f"Total training time: {total_time:.1f}s")
    print(f"Average epoch time: {total_time/epoch:.1f}s")
    
    # Compare with Julia results
    print("\n" + "=" * 60)
    print("Comparison with Julia FPTM on Fashion-MNIST:")
    print("=" * 60)
    print(f"Python FPTM: {best_acc:.2%} with {args.num_clauses} clauses")
    print("Julia FPTM:  92.53% with 2 clauses (using 68 binary features)")
    print("Julia FPTM:  93.59% with 20 clauses (using 68 binary features)")
    print("Note: Julia uses extensive feature engineering with multiple conv kernels")


if __name__ == "__main__":
    main()
