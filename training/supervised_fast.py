"""
High-performance supervised training using the optimized FPTMConvFast model.
"""
import argparse, time
import torch
import torch.nn as nn
import torch.optim as optim
from fptm.models import FPTMConvFast
from fptm.utils import set_seed, accuracy_from_logits
from fptm.heads import compute_ece
from data.synth_fmnist import get_loaders

def train_one_epoch(model, opt, loader, device, reinforce_every=5):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    ce = nn.CrossEntropyLoss()
    print(f"  Training on {len(loader)} batches...")

    epoch_start = time.time()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()
        opt.step()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            if (i % reinforce_every) == 0:
                if len(loader) > 50 and (i % (len(loader)//10)) == 0:  # Progress every 10%
                    elapsed = time.time() - epoch_start
                    print(f"    Batch {i+1}/{len(loader)} ({elapsed:.1f}s)")
                model.reinforce(x, y, preds)
            acc = (preds == y).float().sum().item()
            correct += acc
            total += y.size(0)
            loss_sum += float(loss.item()) * y.size(0)

    epoch_time = time.time() - epoch_start
    print(f"  Epoch completed in {epoch_time:.1f}s")
    return loss_sum/total, correct/total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    ce = nn.CrossEntropyLoss()
    all_logits, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().sum().item()
        correct += acc
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
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patch_size", type=int, default=4)
    ap.add_argument("--num_clauses", type=int, default=64)
    ap.add_argument("--attention_heads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=== FPTM Fast Supervised Training ===")
    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}, num_clauses={args.num_clauses}")
    print(f"        patch_size={args.patch_size}, attention_heads={args.attention_heads}, lr={args.lr}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Creating optimized model and data loaders...")
    model = FPTMConvFast(in_channels=1, image_size=28, patch_size=args.patch_size,
                         num_clauses=args.num_clauses, num_classes=10,
                         attention_heads=args.attention_heads).to(device)

    # Print performance stats
    stats = model.get_performance_stats()
    print(f"Model stats: {stats}")

    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    train_loader, test_loader = get_loaders(batch_size=args.batch_size)
    print(f"Data loaded: {len(train_loader)} train batches, {len(test_loader)} test batches")

    total_start = time.time()
    for epoch in range(1, args.epochs+1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        tr_loss, tr_acc = train_one_epoch(model, opt, train_loader, device)
        va_loss, va_acc, ece = evaluate(model, test_loader, device)
        print(f"Results: train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} | val_loss={va_loss:.4f} val_acc={va_acc:.3f} ECE={ece:.4f}")

    total_time = time.time() - total_start
    print(f"\n🎯 Training completed in {total_time:.1f}s total!")

    # Show throughput
    total_samples = len(train_loader) * args.batch_size * args.epochs
    throughput = total_samples / total_time
    print(f"Overall throughput: {throughput:.1f} samples/sec")

if __name__ == "__main__":
    main()