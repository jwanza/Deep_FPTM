
import torch
import torch.nn.functional as F
from fptm_ste.tm import FuzzyPatternTM_STCM

def check_logits_detailed():
    print("Checking Logits DETAILED for STCM Capacity Operator...")
    n_features = 96
    n_clauses = 128
    n_classes = 10
    
    tm = FuzzyPatternTM_STCM(
        n_features=n_features,
        n_clauses=n_clauses,
        n_classes=n_classes,
        operator="capacity",
        ternary_voting=True,
        ternary_band=0.1,
        ste_temperature=1.0
    )
    
    # Inspect masks
    with torch.no_grad():
        pos, neg, pos_inv, neg_inv = tm.get_masks(use_ste=True)
        print(f"Pos Mask Mean: {pos.mean().item():.4f} Sum: {pos.sum().item()}")
        print(f"Pos Inv Mask Mean: {pos_inv.mean().item():.4f} Sum: {pos_inv.sum().item()}")
        
        # Calculate capacity
        capacity = tm._clause_capacity(pos, pos_inv)
        print(f"Capacity Mean: {capacity.mean().item():.4f}")
        
        # Fake input
        x = torch.rand(16, n_features)
        
        # Calculate mismatch
        mismatch = F.linear(1.0 - x, pos) + F.linear(x, pos_inv)
        print(f"Mismatch Mean: {mismatch.mean().item():.4f}")
        
        raw = capacity - mismatch
        print(f"Raw Strength Mean: {raw.mean().item():.4f}")
        
        out = tm._straight_relu(raw)
        print(f"ReLU Output Mean: {out.mean().item():.4f}")
        print(f"ReLU Output Max: {out.max().item():.4f}")

if __name__ == "__main__":
    check_logits_detailed()

