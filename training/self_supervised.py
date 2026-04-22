import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from fptm.models import FPTMConv
from fptm.utils import set_seed
from data.synth_fmnist import get_loaders

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, proj_dim)
        )
    def forward(self, x):
        return self.net(x)

def mask_augment(x: torch.Tensor, p: float = 0.3):
    # Randomly zero-out patches by setting pixels to 0
    m = (torch.rand_like(x) > p).float()
    return x * m

def info_nce(z1, z2, temperature: float = 0.2):
    # normalize
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    N = z1.size(0)
    reps = torch.cat([z1, z2], dim=0)  # (2N, D)
    sim = reps @ reps.t()               # (2N, 2N)
    sim = sim / temperature
    # mask out self-similarity
    mask = torch.eye(2*N, device=z1.device, dtype=torch.bool)
    sim_no_diag = sim[~mask].view(2*N, 2*N - 1)  # each row has 2N-1 negatives/one positive position

    # Build targets: for rows [0..N-1], positive is at original col i+N; after diag removal its index is (i+N-1).
    # For rows [N..2N-1], i=row-N; positive is at original col i; after diag removal its index is i.
    targets = torch.arange(2*N, device=z1.device)
    pos_idx = targets.clone()
    pos_idx[:N] = pos_idx[:N] + (N - 1)  # i -> i+N-1
    pos_idx[N:] = pos_idx[N:] - N        # i+N -> i

    loss = F.cross_entropy(sim_no_diag, pos_idx)
    return loss

def extract_clause_features(model: FPTMConv, x: torch.Tensor):
    # forward but return pooled clause features right before head
    with torch.no_grad():
        # replicate forward steps
        if x.max() > 1.0 or x.min() < 0.0:
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        patches = model.patchify(x)
        B, N, D = patches.shape
        feats = model.bank(patches.view(B*N, D)).view(B, N, -1)
        if model.attn is not None:
            feats = model.attn(feats)
        pooled = feats.mean(dim=1)  # (B, C)
    return pooled

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--proj_dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_clauses = 64
    model = FPTMConv(in_channels=1, image_size=28, patch_size=4,
                     num_clauses=num_clauses, num_classes=10, attention_heads=0).to(device)
    proj = ProjectionHead(in_dim=num_clauses, proj_dim=args.proj_dim).to(device)  # Use actual num_clauses
    opt = optim.AdamW(list(model.parameters()) + list(proj.parameters()), lr=args.lr)

    train_loader, _ = get_loaders(batch_size=args.batch_size)
    for epoch in range(1, args.epochs+1):
        model.train(); proj.train()
        loss_sum, total = 0.0, 0
        for x, _ in train_loader:
            x = x.to(device)
            x1 = mask_augment(x, p=0.3)
            x2 = mask_augment(x, p=0.3)

            f1 = extract_clause_features(model, x1)
            f2 = extract_clause_features(model, x2)
            z1 = proj(f1)
            z2 = proj(f2)
            loss = info_nce(z1, z2, temperature=0.2)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += float(loss.item()) * x.size(0)
            total += x.size(0)
        print(f"[Epoch {epoch}] InfoNCE loss {loss_sum/total:.4f}")

if __name__ == "__main__":
    main()
