
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import sys
import os

# Ensure python path
sys.path.append(os.path.join(os.getcwd(), "python"))

from fptm_ste.swin_tm import SwinTM
from fptm_ste.tm import FuzzyPatternTM_STCM, FuzzyPatternTM_STE
from fptm_ste.deep_tm import DeepTMNetwork

class DeepSTCM_Wrapper(DeepTMNetwork):
    """
    Wrapper to make DeepTMNetwork compatible with SwinTM's instantiation signature.
    """
    def __init__(self, n_features, n_clauses, n_classes, tau, **kwargs):
        operator = kwargs.get('operator', 'capacity')
        ternary_voting = kwargs.get('ternary_voting', True)
        ternary_band = kwargs.get('ternary_band', 0.1)
        deep_kwargs = {k: v for k, v in kwargs.items() if k not in ['operator', 'ternary_voting', 'ternary_band']}
        
        super().__init__(
            input_dim=n_features,
            hidden_dims=[n_clauses], 
            n_classes=n_classes,
            n_clauses=n_clauses,
            tau=tau,
            layer_cls=FuzzyPatternTM_STCM,
            layer_operator=operator,
            layer_ternary_voting=ternary_voting,
            layer_extra_kwargs={"ternary_band": ternary_band},
            **deep_kwargs
        )

def train(model, device, train_loader, optimizer, epoch, log_interval=200):
    model.train()
    correct = 0
    total = 0
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits, _, _, _ = model(data)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        optimizer.step()
        
        preds = logits.argmax(dim=1)
        correct += (preds == target).sum().item()
        total += target.size(0)
        
        if batch_idx % log_interval == 0 and batch_idx > 0:
            print(f"  Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f} Acc: {correct/total:.4f}")
            
    acc = correct / total
    duration = time.time() - start_time
    print(f"Epoch {epoch} Train Acc: {acc:.4f} Time: {duration:.1f}s")
    return acc

def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits, _, _, _ = model(data)
            preds = logits.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    
    acc = correct / total
    print(f"Test Acc: {acc:.4f}")
    return acc

def run_experiment():
    print("=== Setting up SOTA Experiments (MNIST) ===")
    
    BATCH_SIZE = 128
    EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    # Standard MNIST transform
    # Using [0,1] range is standard for TMs, but Swin usually expects normalized inputs.
    # Let's use [0,1] (ToTensor) because FuzzyPatternTM expects inputs in [0,1].
    # SwinTM applies LayerNorm internally.
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # --- Baseline: Swin + STE ---
    # Patch size 2 -> 14x14. Window size 7.
    print("\n\n>>> Baseline: Swin-Tiny + STE (Patch=2) <<<")
    model_ste = SwinTM(
        preset="tiny",
        num_classes=10,
        in_channels=1, 
        image_size=(28, 28),
        patch_size=2, # 14x14 tokens
        tm_cls=FuzzyPatternTM_STE,
    ).to(DEVICE)
    optimizer_ste = optim.AdamW(model_ste.parameters(), lr=1e-3)
    
    ste_results = []
    # Train for 3 epochs just to check convergence
    for epoch in range(1, 4):
        train(model_ste, DEVICE, train_loader, optimizer_ste, epoch)
        acc = test(model_ste, DEVICE, test_loader)
        ste_results.append(acc)
        
    # --- Experiment 1: Swin + STCM (Shallow Head) ---
    print("\n\n>>> Experiment 1: Swin-Tiny + STCM Head (Patch=2, Capacity, Ternary Vote) <<<")
    
    tm_kwargs = {
        "ternary_voting": True,
        "operator": "capacity",
        "ternary_band": 0.1
    }
    
    model_stcm = SwinTM(
        preset="tiny",
        num_classes=10,
        in_channels=1, 
        image_size=(28, 28),
        patch_size=2,
        tm_cls=FuzzyPatternTM_STCM,
        tm_kwargs=tm_kwargs
    ).to(DEVICE)
    
    optimizer_stcm = optim.AdamW(model_stcm.parameters(), lr=1e-3)
    
    stcm_results = []
    for epoch in range(1, EPOCHS + 1):
        train(model_stcm, DEVICE, train_loader, optimizer_stcm, epoch)
        acc = test(model_stcm, DEVICE, test_loader)
        stcm_results.append(acc)

    # --- Experiment 2: Swin + Deep STCM (Deep Head) ---
    print("\n\n>>> Experiment 2: Swin-Tiny + Deep STCM Head (Patch=2, Stacked) <<<")
    
    model_deep = SwinTM(
        preset="tiny",
        num_classes=10,
        in_channels=1,
        image_size=(28, 28),
        patch_size=2,
        tm_cls=DeepSTCM_Wrapper, 
        tm_kwargs=tm_kwargs
    ).to(DEVICE)
    
    optimizer_deep = optim.AdamW(model_deep.parameters(), lr=1e-3)
    
    deep_results = []
    for epoch in range(1, EPOCHS + 1):
        train(model_deep, DEVICE, train_loader, optimizer_deep, epoch)
        acc = test(model_deep, DEVICE, test_loader)
        deep_results.append(acc)

    # --- Comparison ---
    print("\n\n=== Final Comparison (Test Accuracy) ===")
    print(f"Epoch | Swin-STE (Base) | Swin-STCM | Swin-DeepSTCM")
    for i in range(EPOCHS):
        ste_val = ste_results[i] if i < len(ste_results) else "-"
        stcm_val = stcm_results[i] if i < len(stcm_results) else "-"
        deep_val = deep_results[i] if i < len(deep_results) else "-"
        
        if isinstance(ste_val, float): ste_val = f"{ste_val:.4f}"
        if isinstance(stcm_val, float): stcm_val = f"{stcm_val:.4f}"
        if isinstance(deep_val, float): deep_val = f"{deep_val:.4f}"
        
        print(f"{i+1:5d} | {ste_val}          | {stcm_val}    | {deep_val}")

if __name__ == "__main__":
    run_experiment()
