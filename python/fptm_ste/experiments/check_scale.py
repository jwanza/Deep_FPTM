
import torch
from fptm_ste.tm import FuzzyPatternTM_STCM

def check_logits_scale():
    print("Checking Logits Scale for STCM Capacity Operator...")
    n_features = 96
    n_clauses = 128
    n_classes = 10
    
    # STCM with Capacity + Ternary Voting
    tm = FuzzyPatternTM_STCM(
        n_features=n_features,
        n_clauses=n_clauses,
        n_classes=n_classes,
        operator="capacity",
        ternary_voting=True,
        ternary_band=0.1
    )
    
    # Fake input (already prepared/flattened) in [0, 1]
    x = torch.rand(16, n_features)
    
    logits, clauses = tm(x)
    
    print(f"Clause Outputs Mean: {clauses.mean().item():.4f}")
    print(f"Clause Outputs Max:  {clauses.max().item():.4f}")
    print(f"Logits Mean: {logits.mean().item():.4f}")
    print(f"Logits Std:  {logits.std().item():.4f}")
    print(f"Logits Max:  {logits.max().item():.4f}")
    print(f"Logits Min:  {logits.min().item():.4f}")

if __name__ == "__main__":
    check_logits_scale()

