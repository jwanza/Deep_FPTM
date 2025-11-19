
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "python"))

from fptm_ste.swin_tm import SwinTM, build_swin_stage_configs, SwinTMStageConfig
from fptm_ste.tm import FuzzyPatternTM_STCM, FuzzyPatternTM_STE

def train_one_epoch(model, loader, optimizer, device, name):
    model.train()
    correct = 0
    total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits, _, _, _ = model(data)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        optimizer.step()
        
        preds = logits.argmax(dim=1)
        correct += (preds == target).sum().item()
        total += target.size(0)
    return correct / total

def debug_swin():
    print("=== Debugging Swin TM on MNIST ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Standard MNIST normalization might help? 
        # TMs usually like [0,1], but Swin backbone is deep learning.
        # Let's try standard normalization.
    ])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    # Subset for speed
    subset_indices = list(range(2000)) 
    subset = torch.utils.data.Subset(dataset, subset_indices)
    loader = DataLoader(subset, batch_size=64, shuffle=True)
    
    # Config: Patch=2 -> 14x14 grid. Window=7 -> 2x2 windows. Perfect fit.
    # We manually build configs to ensure parameters are sane for MNIST
    stage_configs = []
    # 2 stages
    # Stage 1: 14x14 -> 14x14 (no merge yet). 
    stage_configs.append(SwinTMStageConfig(
        embed_dim=96, depth=2, num_heads=3, window_size=7, 
        tm_hidden_dim=128, tm_clauses=64, head_clauses=64
    ))
    # Stage 2: PatchMerge -> 7x7. Window=7 -> 1x1 window.
    stage_configs.append(SwinTMStageConfig(
        embed_dim=192, depth=2, num_heads=6, window_size=7, 
        tm_hidden_dim=128, tm_clauses=64, head_clauses=64
    ))

    # 1. Baseline: STE
    print("\n--- Model 1: Swin + STE (Baseline) ---")
    model_ste = SwinTM(
        stage_configs=stage_configs,
        num_classes=10,
        in_channels=1,
        image_size=(28, 28),
        tm_cls=FuzzyPatternTM_STE
    ).to(device)
    opt_ste = optim.AdamW(model_ste.parameters(), lr=1e-3)
    
    acc_ste = train_one_epoch(model_ste, loader, opt_ste, device, "STE")
    print(f"STE Epoch 1 Acc: {acc_ste:.4f}")

    # 2. STCM
    print("\n--- Model 2: Swin + STCM (Capacity, Ternary Vote) ---")
    tm_kwargs = {"ternary_voting": True, "operator": "capacity", "ternary_band": 0.05}
    model_stcm = SwinTM(
        stage_configs=stage_configs,
        num_classes=10,
        in_channels=1,
        image_size=(28, 28),
        tm_cls=FuzzyPatternTM_STCM,
        tm_kwargs=tm_kwargs
    ).to(device)
    opt_stcm = optim.AdamW(model_stcm.parameters(), lr=1e-3)
    
    # Check gradients on first batch
    data, target = next(iter(loader))
    data, target = data.to(device), target.to(device)
    logits, _, _, _ = model_stcm(data)
    loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()
    
    # Check if STCM params have grad
    has_grad = False
    for name, param in model_stcm.named_parameters():
        if "pos_logits" in name and param.grad is not None:
            print(f"Gradient detected on {name}: Mean {param.grad.abs().mean().item():.6f}")
            has_grad = True
            break
    if not has_grad:
        print("WARNING: No gradient on STCM logits!")
        
    acc_stcm = train_one_epoch(model_stcm, loader, opt_stcm, device, "STCM")
    print(f"STCM Epoch 1 Acc: {acc_stcm:.4f}")

if __name__ == "__main__":
    debug_swin()

