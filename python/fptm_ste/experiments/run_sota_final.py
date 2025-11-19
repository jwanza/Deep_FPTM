
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import sys
import os
import gc

# Ensure python path
sys.path.append(os.path.join(os.getcwd(), "python"))

from fptm_ste.swin_tm import SwinTM
from fptm_ste.resnet_tm import resnet_tm18
from fptm_ste.tm import FuzzyPatternTM_STCM, FuzzyPatternTM_STE
from fptm_ste.deep_tm import DeepTMNetwork

class DeepSTCM_Wrapper(DeepTMNetwork):
    """
    Wrapper to make DeepTMNetwork compatible with SwinTM's instantiation signature.
    """
    def __init__(self, n_features, n_clauses, n_classes, tau, **kwargs):
        # Extract DeepTMNetwork specific args
        operator = kwargs.pop('operator', 'capacity')
        ternary_voting = kwargs.pop('ternary_voting', True)
        ternary_band = kwargs.pop('ternary_band', 0.1)
        
        # Remove 'lf' if present, as DeepTMNetwork doesn't accept it in __init__ directly
        # It's passed to layer_extra_kwargs instead
        lf = kwargs.pop('lf', 4)
        
        # Clean kwargs for DeepTMNetwork
        # We need to be careful not to pass arguments DeepTMNetwork doesn't expect
        valid_keys = [
            'input_dim', 'hidden_dims', 'n_classes', 'n_clauses', 'dropout', 
            'tau', 'noise_std', 'input_shape', 'auto_expand_grayscale', 
            'allow_channel_reduce', 'clause_dropout', 'literal_dropout', 
            'clause_bias_init', 'layer_cls', 'layer_operator', 
            'layer_ternary_voting', 'layer_extra_kwargs'
        ]
        
        deep_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        
        super().__init__(
            input_dim=n_features,
            hidden_dims=[n_clauses], 
            n_classes=n_classes,
            n_clauses=n_clauses,
            tau=tau,
            layer_cls=FuzzyPatternTM_STCM,
            layer_operator=operator,
            layer_ternary_voting=ternary_voting,
            layer_extra_kwargs={"ternary_band": ternary_band, "lf": lf},
            **deep_kwargs
        )

def train(model, device, train_loader, optimizer, epoch, log_interval=100):
    model.train()
    correct = 0
    total = 0
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # Different models have different return signatures
        out = model(data)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
            
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
            out = model(data)
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out
            preds = logits.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    
    acc = correct / total
    print(f"Test Acc: {acc:.4f}")
    return acc

def run_experiment():
    print("=== Running SOTA Experiments (MNIST) - Comparison ===")
    
    BATCH_SIZE = 128
    EPOCHS = 5 # Shortened for demo
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    # Normalization helps convolution convergence
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # --- Model 1: ResNet18 + STCM (SOTA Candidate) ---
    print("\n\n>>> Model 1: ResNet18 + STCM (Deep Convolutional SOTA) <<<")
    
    tm_kwargs = {
        "ternary_voting": True,
        "operator": "capacity",
        "ternary_band": 0.05,
        "lf": 10 
    }
    
    model_resnet = resnet_tm18(
        num_classes=10,
        in_channels=1,
        tm_cls=FuzzyPatternTM_STCM,
        tm_kwargs=tm_kwargs,
        tm_hidden=256,
        tm_clauses=128
    ).to(DEVICE)
    
    optimizer_resnet = optim.AdamW(model_resnet.parameters(), lr=0.001)
    
    resnet_results = []
    for epoch in range(1, EPOCHS + 1):
        train(model_resnet, DEVICE, train_loader, optimizer_resnet, epoch)
        acc = test(model_resnet, DEVICE, test_loader)
        resnet_results.append(acc)
        
    del model_resnet, optimizer_resnet
    torch.cuda.empty_cache()
    gc.collect()

    # --- Model 2: Swin + Deep STCM (Complex Architecture) ---
    print("\n\n>>> Model 2: Swin-Tiny + Deep STCM Head (Patch=2) <<<")
    
    model_swin = SwinTM(
        preset="tiny",
        num_classes=10,
        in_channels=1,
        image_size=(28, 28),
        patch_size=2,
        tm_cls=DeepSTCM_Wrapper, 
        tm_kwargs=tm_kwargs
    ).to(DEVICE)
    
    optimizer_swin = optim.AdamW(model_swin.parameters(), lr=0.0005) # Lower LR for Swin
    
    swin_results = []
    for epoch in range(1, EPOCHS + 1):
        train(model_swin, DEVICE, train_loader, optimizer_swin, epoch)
        acc = test(model_swin, DEVICE, test_loader)
        swin_results.append(acc)

    # --- Comparison ---
    print("\n\n=== Final Comparison (Test Accuracy) ===")
    print(f"Epoch | ResNet18-STCM | Swin-DeepSTCM")
    for i in range(EPOCHS):
        resnet_val = f"{resnet_results[i]:.4f}"
        swin_val = f"{swin_results[i]:.4f}"
        print(f"{i+1:5d} | {resnet_val}       | {swin_val}")

if __name__ == "__main__":
    run_experiment()

