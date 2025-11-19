
import torch
import torch.nn as nn
from fptm_ste.tm import FuzzyPatternTM_STE, FuzzyPatternTM_STCM

def run_repro():
    # The error:
    # RuntimeError: The size of tensor a (320) must match the size of tensor b (321) at non-singleton dimension 1
    # In tm.py line 271: logits = (clause_outputs + self.clause_bias.view(1, -1)) @ self.voting
    #
    # clause_outputs is [B, n_clauses]
    # clause_bias is [n_clauses]
    # voting is [n_clauses, n_classes]
    #
    # If n_clauses=320 but something expected 321 or vice versa.
    # 
    # The log says "Auto-tuned clause counts: 321".
    # This likely comes from auto_tune_transformer_clauses in run_mnist_equiv.py.
    # It returns a tuple of clause counts per layer.
    # 
    # If the model is re-initialized or parameters are updated with mismatching shapes.
    #
    # In TMFeedForward, we initialize core with n_clauses.
    # If n_clauses comes from the auto-tuner, it might be 321.
    # But where does 320 come from?
    # Maybe default was 320? 
    # Or maybe clause_bias was initialized with different size?
    #
    # In FuzzyPatternTM_STE.__init__:
    # self.clause_bias = nn.Parameter(torch.full((n_clauses,), clause_bias_init, dtype=torch.float32))
    #
    # So clause_bias should match n_clauses.
    #
    # However, if n_clauses is odd (321), and we have:
    # half = n_clauses // 2
    # n_clauses for STE/STCM usually implies 2 * half.
    # In STE:
    # half = n_clauses // 2
    # ta_pos = [half, features]
    # ...
    # clause_outputs = torch.cat([pos_prod, neg_prod], dim=1) -> [B, 2*half]
    #
    # If n_clauses is 321:
    # half = 160
    # clause_outputs size = 320.
    # But self.clause_bias size = 321.
    # 
    # Mismatch: 320 vs 321.
    #
    # Fix: n_clauses must be even.
    
    print("Simulating odd n_clauses issue...")
    try:
        tm = FuzzyPatternTM_STE(n_features=10, n_clauses=321, n_classes=10)
        x = torch.randn(1, 10)
        _ = tm(x)
    except RuntimeError as e:
        print(f"Caught expected error: {e}")

if __name__ == "__main__":
    run_repro()

