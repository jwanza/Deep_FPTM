import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data.synth_fmnist import get_loaders
from fptm.models import FPTMConv
from fptm.utils import set_seed

class TinyCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*7*7, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def kd_loss(student_logits, teacher_logits, T=2.0):
    # KL Divergence between softened distributions
    p_t = F.softmax(teacher_logits / T, dim=-1)
    log_p_s = F.log_softmax(student_logits / T, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T*T)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=0.5)  # CE weight
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = TinyCNN().to(device)
    student = FPTMConv(in_channels=1, image_size=28, patch_size=4,
                       num_clauses=64, num_classes=10, attention_heads=0).to(device)

    opt_t = optim.AdamW(teacher.parameters(), lr=args.lr)
    opt_s = optim.AdamW(student.parameters(), lr=args.lr)

    train_loader, test_loader = get_loaders(batch_size=args.batch_size)
    ce = nn.CrossEntropyLoss()

    # Train teacher briefly
    teacher.train()
    for epoch in range(1, max(2, args.epochs//2)+1):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt_t.zero_grad()
            logits = teacher(x)
            loss = ce(logits, y)
            loss.backward()
            opt_t.step()

    # Distill to student
    for epoch in range(1, args.epochs+1):
        student.train(); teacher.eval()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = student(x)
            loss = args.alpha * ce(s_logits, y) + (1-args.alpha) * kd_loss(s_logits, t_logits, T=args.temperature)

            opt_s.zero_grad()
            loss.backward()
            opt_s.step()

            with torch.no_grad():
                preds = s_logits.argmax(dim=-1)
                student.reinforce(x, y, preds)

        # quick eval
        with torch.no_grad():
            total, correct = 0, 0
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                s_logits = student(x)
                preds = s_logits.argmax(dim=-1)
                correct += (preds == y).float().sum().item()
                total += y.size(0)
            acc = correct/total
        print(f"[Epoch {epoch}] Student val acc {acc:.3f}")

if __name__ == "__main__":
    main()
