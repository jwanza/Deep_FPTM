"""
Adaptive supervised training with learning rate scheduling and better hyperparameters.
"""
import argparse, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from fptm.models import FPTMConvFast
from fptm.utils import set_seed
from fptm.heads import compute_ece
from data.synth_fmnist import get_loaders


def train_one_epoch(model, opt, loader, device, scheduler=None, reinforce_every=3):
    """Train with adaptive learning rate and optimized reinforcement."""
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    ce = nn.CrossEntropyLoss()
    
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        opt.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        logits = model(x)
        loss = ce(logits, y)
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        
        if scheduler and hasattr(scheduler, 'step'):
            if isinstance(scheduler, OneCycleLR):
                scheduler.step()
        
        # Reinforcement and metrics
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            
            # Adaptive reinforcement frequency
            if i % reinforce_every == 0:
                # Adaptive specificity based on accuracy
                current_acc = (preds == y).float().mean().item()
                adaptive_s = 3.0 * (1.0 + max(0, 0.5 - current_acc))  # Higher s when accuracy is low
                model.reinforce(x, y, preds, s=adaptive_s)
            
            correct += (preds == y).float().sum().item()
            total += y.size(0)
            loss_sum += float(loss.item()) * y.size(0)
    
    return loss_sum/total, correct/total


@torch.no_grad()
def evaluate(model, loader, device):
    """Fast evaluation."""
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    ce = nn.CrossEntropyLoss()
    all_logits, all_labels = [], []
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
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
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--patch_size", type=int, default=4)
    ap.add_argument("--num_clauses", type=int, default=256)
    ap.add_argument("--attention_heads", type=int, default=4)
    ap.add_argument("--scheduler", choices=["none", "cosine", "onecycle"], default="cosine")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    print("=== FPTM Adaptive Supervised Training ===")
    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}, num_clauses={args.num_clauses}")
    print(f"        attention_heads={args.attention_heads}, lr={args.lr}, scheduler={args.scheduler}")
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = FPTMConvFast(
        in_channels=1, 
        image_size=28, 
        patch_size=args.patch_size,
        num_clauses=args.num_clauses, 
        num_classes=10,
        attention_heads=args.attention_heads
    ).to(device)
    
    # Optimizer with weight decay
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Learning rate scheduler
    train_loader, test_loader = get_loaders(batch_size=args.batch_size)
    total_steps = len(train_loader) * args.epochs
    
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    elif args.scheduler == "onecycle":
        scheduler = OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps)
    else:
        scheduler = None
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(train_loader) * args.batch_size:,}")
    print(f"Device: {device}")
    
    # Training loop
    best_acc = 0
    total_start = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Train
        epoch_start = time.time()
        tr_loss, tr_acc = train_one_epoch(model, opt, train_loader, device, scheduler)
        
        # Step scheduler (for cosine)
        if scheduler and isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        
        # Evaluate
        va_loss, va_acc, ece = evaluate(model, test_loader, device)
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
              f"LR: {current_lr:.5f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Early stopping check
        if epoch > 20 and va_acc < best_acc - 0.05:
            print(f"Early stopping: validation accuracy dropped below best")
            break
    
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Training completed in {total_time:.1f}s")
    print(f"Best validation accuracy: {best_acc:.1%} at epoch {best_epoch}")
    print(f"Throughput: {len(train_loader) * args.batch_size * epoch / total_time:.1f} samples/sec")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
