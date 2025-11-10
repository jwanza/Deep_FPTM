import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from fptm_ste import SwinFeatureExtractor, MultiScaleTMEnsemble
from fptm_ste.trainers import train_step


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CIFAR-10 -> 224x224 for Swin
    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    testset = datasets.CIFAR10(root="/tmp/cifar", train=False, download=True, transform=tfm)
    trainset = datasets.CIFAR10(root="/tmp/cifar", train=True, download=True, transform=tfm)

    # Small subset for quick run
    train_idx = list(range(0, 1000))
    test_idx = list(range(0, 500))
    train_loader = DataLoader(Subset(trainset, train_idx), batch_size=16, shuffle=True, num_workers=2)
    test_loader = DataLoader(Subset(testset, test_idx), batch_size=16, shuffle=False, num_workers=2)

    # Backbone + ensemble
    backbone = SwinFeatureExtractor(pretrained=True, freeze=True).to(device)

    # Peek dims
    with torch.no_grad():
        sample = next(iter(train_loader))[0].to(device)
        feats = backbone(sample)
        feature_dims = [f.shape[1] for f in feats]

    head_configs = [
        {"stages": [0], "n_clauses": 200, "thresholds": 16, "binarizer": "dual"},
        {"stages": [1], "n_clauses": 150, "thresholds": 16, "binarizer": "dual"},
        {"stages": [2], "n_clauses": 120, "thresholds": 16, "binarizer": "dual"},
        {"stages": [3], "n_clauses": 100, "thresholds": 16, "binarizer": "dual"},
        # Extra heads reusing stages to demonstrate N > S
        {"stages": [1, 2], "n_clauses": 180, "thresholds": 8, "binarizer": "dual"},
        {"stages": [2, 3], "n_clauses": 160, "thresholds": 8, "binarizer": "dual"},
    ]
    model = MultiScaleTMEnsemble(feature_dims, n_classes=10, head_configs=head_configs, backbone_type="swin", init_temperature=1.0).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    def evaluate():
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                feats = backbone(x)
                logits, _, _ = model(feats, use_ste=True)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / total

    acc_before = evaluate()

    # Brief training
    model.train()
    for epoch in range(2):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            feats = backbone(x)
            optim.zero_grad()
            logits, _, _ = model(feats, use_ste=True)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optim.step()

    acc_after = evaluate()
    print(f"CIFAR10 Swin+TM ensemble accuracy: before={acc_before:.3f}, after={acc_after:.3f}")


if __name__ == "__main__":
    main()


