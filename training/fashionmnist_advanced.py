"""
Advanced Fashion-MNIST training that closely mimics the Julia implementation.
Uses multiple convolution kernels and quantile-based binarization for rich feature extraction.
"""
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import numpy as np

from fptm.models import FPTMConvFast
from fptm.utils import set_seed
from fptm.heads import compute_ece


class AdvancedFeatureExtractor:
    """
    Mimics the Julia implementation's feature extraction:
    - Multiple kernel sizes (3x3, 5x5, 7x7, 9x9)
    - Quantile-based binarization
    - 68+ binary features per pixel
    """
    
    def __init__(self):
        # Define convolution kernels matching Julia implementation
        
        # 3x3 Sobel-like kernels
        self.Kx3 = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32)
        
        # 5x5 extended edge detector
        self.Kx5 = torch.tensor([
            [0, 1, 2, 3, 4],
            [-1, 0, 2, 3, 3],
            [-2, -2, 0, 2, 2],
            [-3, -3, -2, 0, 1],
            [-4, -3, -2, -1, 0]
        ], dtype=torch.float32)
        
        # 7x7 large-scale edge detector
        self.Kx7 = torch.tensor([
            [-3, -2, -1, 0, 1, 2, 3],
            [-4, -3, -2, 0, 2, 3, 4],
            [-5, -4, -3, 0, 3, 4, 5],
            [-6, -5, -4, 0, 4, 5, 6],
            [-5, -4, -3, 0, 3, 4, 5],
            [-4, -3, -2, 0, 2, 3, 4],
            [-3, -2, -1, 0, 1, 2, 3]
        ], dtype=torch.float32)
        
        # 3x3 horizontal edge detector
        self.Kx9 = torch.tensor([
            [-1, -1, -1],
            [ 2,  2,  2],
            [-1, -1, -1]
        ], dtype=torch.float32)
        
        # Y-direction kernels (90-degree rotation)
        self.Ky3 = torch.rot90(self.Kx3, k=1, dims=[0, 1])
        self.Ky5 = torch.rot90(self.Kx5, k=1, dims=[0, 1])
        self.Ky7 = torch.rot90(self.Kx7, k=1, dims=[0, 1])
        self.Ky9 = torch.rot90(self.Kx9, k=1, dims=[0, 1])
        
        # Convert to proper conv2d format (out_channels, in_channels, H, W)
        self.kernels = {
            'x3': self.Kx3.unsqueeze(0).unsqueeze(0),
            'y3': self.Ky3.unsqueeze(0).unsqueeze(0),
            'x5': self.Kx5.unsqueeze(0).unsqueeze(0),
            'y5': self.Ky5.unsqueeze(0).unsqueeze(0),
            'x7': self.Kx7.unsqueeze(0).unsqueeze(0),
            'y7': self.Ky7.unsqueeze(0).unsqueeze(0),
            'x9': self.Kx9.unsqueeze(0).unsqueeze(0),
            'y9': self.Ky9.unsqueeze(0).unsqueeze(0),
        }
    
    def compute_quantiles(self, tensor, quantiles=[0.25, 0.34, 0.50, 0.75]):
        """Compute quantiles for a tensor."""
        flat = tensor.flatten()
        flat_sorted = torch.sort(flat)[0]
        n = len(flat_sorted)
        
        results = []
        for q in quantiles:
            idx = min(int(n * q), n - 1)
            results.append(flat_sorted[idx])
        
        return results
    
    def extract_features(self, x):
        """
        Extract 68+ binary features per pixel as in Julia implementation.
        
        Args:
            x: (B, 1, 28, 28) tensor
        Returns:
            (B, 68, 28, 28) binary feature tensor
        """
        device = x.device
        B = x.shape[0]
        features = []
        
        # Move kernels to device
        kernels_device = {}
        for name, kernel in self.kernels.items():
            kernels_device[name] = kernel.to(device)
        
        # Apply convolutions with different kernel sizes
        conv_x3 = F.conv2d(x, kernels_device['x3'], padding=1)
        conv_y3 = F.conv2d(x, kernels_device['y3'], padding=1)
        conv_x5 = F.conv2d(x, kernels_device['x5'], padding=2)
        conv_y5 = F.conv2d(x, kernels_device['y5'], padding=2)
        conv_x7 = F.conv2d(x, kernels_device['x7'], padding=3)
        conv_y7 = F.conv2d(x, kernels_device['y7'], padding=3)
        conv_x9 = F.conv2d(x, kernels_device['x9'], padding=1)
        conv_y9 = F.conv2d(x, kernels_device['y9'], padding=1)
        
        # Process each sample in batch
        batch_features = []
        
        for b in range(B):
            sample_features = []
            
            # Raw pixel features (4 features)
            raw = x[b, 0]
            raw_quantiles = self.compute_quantiles(raw[raw > 0])
            sample_features.append((raw > 0).float())
            for q in raw_quantiles[:3]:  # 25%, 34%, 50%
                sample_features.append((raw > q).float())
            
            # Process each convolution result (8 features each)
            for conv_result in [conv_x3[b, 0], conv_y3[b, 0], 
                               conv_x5[b, 0], conv_y5[b, 0],
                               conv_x7[b, 0], conv_y7[b, 0],
                               conv_x9[b, 0], conv_y9[b, 0]]:
                
                # Positive values
                pos_vals = conv_result[conv_result > 0]
                if len(pos_vals) > 0:
                    pos_quantiles = self.compute_quantiles(pos_vals)
                    sample_features.append((conv_result > 0).float())
                    for q in pos_quantiles:
                        sample_features.append((conv_result > q).float())
                else:
                    # If no positive values, add zeros
                    for _ in range(5):
                        sample_features.append(torch.zeros_like(conv_result))
                
                # Negative values  
                neg_vals = conv_result[conv_result < 0]
                if len(neg_vals) > 0:
                    neg_quantiles = self.compute_quantiles(-neg_vals)  # Flip for quantiles
                    for q in neg_quantiles[:3]:  # Use 3 negative quantiles
                        sample_features.append((conv_result < -q).float())
                else:
                    # If no negative values, add zeros
                    for _ in range(3):
                        sample_features.append(torch.zeros_like(conv_result))
            
            # Stack features for this sample
            batch_features.append(torch.stack(sample_features, dim=0))
        
        # Stack all batch samples
        return torch.stack(batch_features, dim=0)


class FashionMNISTAdvanced(Dataset):
    """Fashion-MNIST with advanced feature extraction."""
    
    def __init__(self, train=True, use_advanced_features=True):
        self.use_advanced_features = use_advanced_features
        
        # Load Fashion-MNIST
        self.dataset = torchvision.datasets.FashionMNIST(
            root='./data',
            train=train,
            download=True,
            transform=transforms.ToTensor()
        )
        
        if use_advanced_features:
            self.feature_extractor = AdvancedFeatureExtractor()
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        if self.use_advanced_features:
            # Extract advanced features
            features = self.feature_extractor.extract_features(img.unsqueeze(0))
            return features.squeeze(0), label
        else:
            return img, label


def train_one_epoch(model, opt, loader, device, scheduler=None, reinforce_every=3):
    """Train for one epoch with advanced features."""
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    ce = nn.CrossEntropyLoss()
    
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = ce(logits, y)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        
        # Reinforcement and metrics
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            
            if i % reinforce_every == 0:
                current_acc = (preds == y).float().mean().item()
                # Higher specificity when accuracy is low
                adaptive_s = 5.0 * (1.0 + max(0, 0.6 - current_acc))
                model.reinforce(x, y, preds, s=adaptive_s)
            
            correct += (preds == y).float().sum().item()
            total += y.size(0)
            loss_sum += float(loss.item()) * y.size(0)
    
    return loss_sum/total, correct/total


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model."""
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    ce = nn.CrossEntropyLoss()
    all_logits, all_labels = [], []
    
    # Per-class accuracy tracking
    class_correct = [0] * 10
    class_total = [0] * 10
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        loss = ce(logits, y)
        preds = logits.argmax(dim=-1)
        
        # Overall metrics
        correct += (preds == y).float().sum().item()
        total += y.size(0)
        loss_sum += float(loss.item()) * y.size(0)
        
        # Per-class metrics
        for i in range(y.size(0)):
            label = y[i].item()
            class_total[label] += 1
            if preds[i] == y[i]:
                class_correct[label] += 1
        
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())
    
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    ece = compute_ece(logits, labels)
    
    # Compute per-class accuracy
    class_acc = []
    for i in range(10):
        if class_total[i] > 0:
            class_acc.append(class_correct[i] / class_total[i])
        else:
            class_acc.append(0.0)
    
    return loss_sum/total, correct/total, ece, class_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patch_size", type=int, default=4)
    ap.add_argument("--num_clauses", type=int, default=512)
    ap.add_argument("--attention_heads", type=int, default=8)
    ap.add_argument("--use_advanced_features", action="store_true", default=True,
                    help="Use advanced multi-kernel feature extraction like Julia")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    print("=" * 70)
    print("FPTM Advanced Training on Fashion-MNIST")
    print("Mimicking Julia's Multi-Kernel Feature Extraction")
    print("=" * 70)
    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}")
    print(f"        num_clauses={args.num_clauses}, attention_heads={args.attention_heads}")
    print(f"        lr={args.lr}, use_advanced_features={args.use_advanced_features}")
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create datasets
    print("\nPreparing Fashion-MNIST with advanced features...")
    print("This mimics Julia's 68-feature extraction process...")
    
    train_dataset = FashionMNISTAdvanced(train=True, use_advanced_features=args.use_advanced_features)
    test_dataset = FashionMNISTAdvanced(train=False, use_advanced_features=args.use_advanced_features)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    
    # Determine input channels
    if args.use_advanced_features:
        in_channels = 68  # 68 binary features as in Julia
        print(f"Using 68 binary features per pixel (like Julia implementation)")
    else:
        in_channels = 1
    
    # Create model
    model = FPTMConvFast(
        in_channels=in_channels,
        image_size=28,
        patch_size=args.patch_size,
        num_clauses=args.num_clauses,
        num_classes=10,
        attention_heads=args.attention_heads,
        normalize_mode="none"  # Features are already binary/normalized
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    
    # Optimizer and scheduler
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.001)
    
    # Class names for Fashion-MNIST
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Training
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    
    best_acc = 0
    best_epoch = 0
    total_start = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Train
        epoch_start = time.time()
        tr_loss, tr_acc = train_one_epoch(model, opt, train_loader, device, scheduler)
        
        # Step scheduler
        scheduler.step()
        
        # Evaluate
        va_loss, va_acc, ece, class_acc = evaluate(model, test_loader, device)
        epoch_time = time.time() - epoch_start
        
        # Track best
        if va_acc > best_acc:
            best_acc = va_acc
            best_epoch = epoch
        
        # Print progress
        current_lr = opt.param_groups[0]['lr']
        print(f"[{epoch:3d}/{args.epochs}] "
              f"Train: {tr_loss:.3f}/{tr_acc:.1%} | "
              f"Val: {va_loss:.3f}/{va_acc:.1%} | "
              f"ECE: {ece:.3f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Print per-class accuracy every 10 epochs
        if epoch % 10 == 0:
            print("\nPer-class accuracy:")
            for i, acc in enumerate(class_acc):
                print(f"  {class_names[i]:12s}: {acc:.1%}")
            print()
        
        # Early stopping
        if epoch > 30 and va_acc < best_acc - 0.08:
            print("Early stopping triggered")
            break
    
    # Final results
    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"Best validation accuracy: {best_acc:.2%} at epoch {best_epoch}")
    print(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Average epoch time: {total_time/epoch:.1f}s")
    
    # Final evaluation for per-class results
    _, final_acc, final_ece, final_class_acc = evaluate(model, test_loader, device)
    
    print("\nFinal Per-Class Accuracy:")
    for i, acc in enumerate(final_class_acc):
        print(f"  {class_names[i]:12s}: {acc:.1%}")
    
    # Compare with Julia
    print("\n" + "=" * 70)
    print("Comparison with Julia FPTM:")
    print("=" * 70)
    print(f"Python FPTM (this run): {best_acc:.2%} with {args.num_clauses} clauses")
    print(f"                        Using {'68 binary features' if args.use_advanced_features else 'raw pixels'}")
    print()
    print("Julia FPTM results on Fashion-MNIST:")
    print("  - 92.53% accuracy with just 2 clauses")
    print("  - 93.59% accuracy with 20 clauses") 
    print("  - Uses 68 binary features from multi-scale convolutions")
    print()
    print("Note: The Julia implementation uses a different clause evaluation")
    print("      mechanism (fuzzy literals) which may be more efficient")


if __name__ == "__main__":
    main()
