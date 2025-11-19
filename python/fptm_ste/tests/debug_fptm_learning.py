import torch
import torch.nn as nn
import torch.nn.functional as F
from fptm_ste.tm import FuzzyPatternTMFPTM

def test_fptm_learning_dynamics():
    print("--- Testing FPTM Equiv Learning Dynamics ---")
    
    # Hyperparams
    n_features = 100
    n_clauses_per_class = 10
    n_classes = 2
    # total clauses passed to constructor = n_clauses_per_class * n_classes = 20
    total_clauses = n_clauses_per_class * n_classes
    batch_size = 8
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Instantiate Model
    model = FuzzyPatternTMFPTM(
        n_features=n_features,
        n_clauses=total_clauses,
        n_classes=n_classes,
        tau=0.5,
        team_per_class=True,
        T=1.0
    ).to(device)
    
    # Dummy Data
    x = torch.rand(batch_size, n_features).to(device) # Float inputs [0,1]
    y = torch.randint(0, n_classes, (batch_size,)).to(device)
    
    # Forward Pass
    print("\n1. Forward Pass Check")
    logits, _ = model(x, use_ste=True)
    print(f"Logits mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}")
    print(f"Logits: {logits[0].detach().cpu().numpy()}")
    
    # Loss Calculation (Margin Loss Logic)
    print("\n2. Loss Calculation Check")
    T = getattr(model, "T", 1.0)
    correct_scores = logits.gather(1, y.view(-1, 1)).squeeze()
    loss_correct = F.relu(T - correct_scores)
    loss_incorrect = F.relu(T + logits)
    
    # Mask out correct class
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, y.view(-1, 1), False)
    loss_incorrect = loss_incorrect[mask].view(batch_size, n_classes - 1)
    
    loss = (loss_correct + loss_incorrect.sum(dim=1)).mean()
    print(f"Loss: {loss.item():.4f}")
    
    # Backward Pass
    print("\n3. Gradient Check")
    loss.backward()
    
    # Check gradients on literals
    ta_pos_grad = model.ta_pos.grad
    if ta_pos_grad is None:
        print("ERROR: model.ta_pos.grad is None!")
    else:
        grad_norm = ta_pos_grad.norm().item()
        print(f"model.ta_pos.grad norm: {grad_norm:.6f}")
        print(f"model.ta_pos.grad mean: {ta_pos_grad.abs().mean().item():.6f}")
        
        if grad_norm == 0.0:
            print("WARNING: Gradients are all zero!")
        else:
            print("SUCCESS: Gradients are flowing.")

if __name__ == "__main__":
    test_fptm_learning_dynamics()

