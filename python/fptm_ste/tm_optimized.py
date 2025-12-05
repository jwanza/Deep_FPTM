"""
Optimized implementation of Set Tsetlin Convolutional Machine (STCM).

This implementation leverages the mathematical equivalence between the Tsetlin Machine
clause matching operation and a constrained sparse/ternary linear layer to achieve
significant memory and computational efficiency gains (approx 2x).
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tm import FuzzyPatternTM_STCM, prepare_tm_input, _ste_ternary
from .operators import build_ternary_operator

class OptimizedSTCM(FuzzyPatternTM_STCM):
    """
    Optimized Setun-Ternary Clause Machine.
    
    This class inherits from FuzzyPatternTM_STCM but overrides the internal
    strength calculation mechanism to use a more efficient linear projection
    instead of the concatenation-based approach.
    
    Optimization Analysis:
    ----------------------
    Original:
        x_neg = 1 - x
        X_combined = [x_neg, x]  (Size: 2F)
        W_total = [mask_pos, mask_inv] (Size: C x 2F)
        mismatch = X_combined @ W_total.T
        
    Optimized:
        W_eff = mask_pos - mask_inv (Size: C x F)
        Bias_mismatch = sum(mask_pos)
        mismatch = Bias_mismatch - x @ W_eff.T
        
    This reduces input expansion (saving memory) and halves the matrix multiplication size.
    """
    
    def _strength(self, x: torch.Tensor, mask_pos: torch.Tensor, mask_inv: torch.Tensor) -> torch.Tensor:
        # Check for custom fuzzy operator (fallback to original logic for complex operators)
        if hasattr(self, 'operator_impl') and self.operator_impl is not None:
             return super()._strength(x, mask_pos, mask_inv)
            
        # Common calculations
        # W_eff is effectively the ternary weight (-1, 0, 1)
        # However, due to STE/Softmax, mask_pos and mask_inv are in [0, 1]
        # mask_pos means "require 1", mask_inv means "require 0"
        W_eff = mask_pos - mask_inv  # [half, F]
        
        # Mismatch = sum(mask_pos) - x @ W_eff.T
        # We can compute x @ W_eff.T efficiently
        projection = F.linear(x, W_eff) # [B, half]
        mismatch_bias = mask_pos.sum(dim=1).unsqueeze(0) # [1, half]
        mismatch = mismatch_bias - projection
        
        if self.operator == "capacity":
            # Capacity = sum(mask_pos + mask_inv) (potentially clamped)
            capacity = self._clause_capacity(mask_pos, mask_inv) # [1, half]
            
            # Strength = ReLU(capacity - mismatch)
            #          = ReLU(capacity - (sum(mask_pos) - projection))
            #          = ReLU(capacity - sum(mask_pos) + projection)
            
            # Note: We use _straight_relu from parent
            raw = capacity - mismatch
            return self._straight_relu(raw)
            
        else: # product
            # Strength = exp(-scale * mismatch)
            #          = exp(-scale * (sum(mask_pos) - projection))
            #          = exp(scale * (projection - sum(mask_pos)))
            
            scaled = torch.clamp((mismatch) * self.product_scale, min=0.0, max=10.0)
            return torch.exp(-scaled)

    def _clause_outputs(self, x: torch.Tensor, use_ste: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Copied and adapted from FuzzyPatternTM_STCM to use optimized _strength
        # Most logic is same, but we avoid creating X_combined
        
        # 1. Generate Masks (same as original)
        all_logits = torch.cat([self.pos_logits, self.neg_logits], dim=0)
        temp = max(self.ste_temperature, 1e-6)
        
        if use_ste:
             mask_all = _ste_ternary(all_logits, self.ternary_band, self.ste_temperature)
        else:
             mask_all = torch.tanh(all_logits / self.ste_temperature)
             
        half = self.n_clauses // 2
        
        # Split Masks Logic (same as original)
        soft_all = torch.sigmoid(all_logits / temp)
        soft_pos_all = soft_all
        soft_inv_all = 1.0 - soft_all
        
        hard_pos_all = torch.clamp(mask_all, min=0.0)
        hard_inv_all = torch.clamp(-mask_all, min=0.0)
        
        pos_all = hard_pos_all
        inv_all = hard_inv_all
        
        pos_pos = pos_all[:half]
        pos_inv = inv_all[:half]
        neg_pos = pos_all[half:]
        neg_inv = inv_all[half:]
        
        # Enforce budget (same as original)
        pos_pos, pos_inv = self._enforce_literal_budget(pos_pos, pos_inv)
        neg_pos, neg_inv = self._enforce_literal_budget(neg_pos, neg_inv)
        
        # 2. Calculate Strengths (Optimized)
        # We call _strength which now uses the optimized projection
        pos_strength = self._strength(x, pos_pos, pos_inv)
        neg_strength = self._strength(x, neg_pos, neg_inv)
        
        # 3. Combine and Vote
        clause_votes = torch.cat([pos_strength, -neg_strength], dim=1)

        if self.vote_clamp is not None:
            clause_votes = clause_votes.clamp(-self.vote_clamp, self.vote_clamp)
        if self.training and self.clause_dropout > 0.0:
            clause_votes = F.dropout(clause_votes, p=self.clause_dropout, training=True)
            
        return pos_strength, neg_strength, clause_votes


