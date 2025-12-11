"""
Optimized implementation of Set Tsetlin Convolutional Machine (STCM).

This implementation leverages the mathematical equivalence between the Tsetlin Machine
clause matching operation and a constrained sparse/ternary linear layer to achieve
significant memory and computational efficiency gains (approx 2x).

Optimizations:
1. Uses W_eff = mask_pos - mask_inv (ternary weights {-1, 0, 1})
2. Avoids input concatenation [x_neg, x] -> just uses x
3. Matrix size halved: [C, 2F] -> [C, F]
4. Optional: Packed ternary weights for 16x memory reduction
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tm import FuzzyPatternTM_STCM, prepare_tm_input, _ste_ternary
from .operators import build_ternary_operator
from .config import get_triton_status  # Respect global toggle

# Try to import triton kernels
try:
    # Import legacy V1 kernels
    from .kernels import pack_ternary_pytorch, ternary_linear_cached as ternary_linear_v1_cached
    
    # Import new V2 kernels
    from .kernels_optimized import pack_ternary_int32, ternary_linear_v2, TRITON_AVAILABLE
    
    # Import fused clause pipeline kernels (Massive Speedup)
    from .kernels_clause_pipeline import fused_ste_strength, fused_clause_outputs
    
    TRITON_KERNELS_AVAILABLE = TRITON_AVAILABLE
except ImportError:
    TRITON_KERNELS_AVAILABLE = False
    fused_ste_strength = None

# Try to import bitplane16 tensor core kernels (highest performance)
try:
    from .kernels_bitplane16 import (
        pack_ternary_int16,
        ternary_linear_tc,
        bitplane16_tc_matmul,
        TRITON_AVAILABLE as BITPLANE16_AVAILABLE,
    )
except ImportError:
    BITPLANE16_AVAILABLE = False
    pack_ternary_int16 = None
    ternary_linear_tc = None
    bitplane16_tc_matmul = None

# Helper for caching packed int32 weights
class TernaryWeightCacheV2:
    def get_or_pack(self, w: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        w_ternary = torch.round(w).clamp(-1, 1)
        return pack_ternary_int32(w_ternary)

_weight_cache_v2 = TernaryWeightCacheV2()

def ternary_linear_v2_cached(x, w):
    w_packed, shape = _weight_cache_v2.get_or_pack(w)
    return ternary_linear_v2(x, w_packed, shape)


class OptimizedSTCM(FuzzyPatternTM_STCM):
    """
    Optimized Setun-Ternary Clause Machine.
    
    This class inherits from FuzzyPatternTM_STCM but overrides the internal
    strength calculation mechanism to use a more efficient linear projection
    instead of the concatenation-based approach.
    """
    
    def _strength(self, x: torch.Tensor, mask_pos: torch.Tensor, mask_inv: torch.Tensor) -> torch.Tensor:
        # Check for custom fuzzy operator (fallback to original logic for complex operators)
        if hasattr(self, 'operator_impl') and self.operator_impl is not None:
             return super()._strength(x, mask_pos, mask_inv)
            
        # Common calculations
        W_eff = mask_pos - mask_inv  # [half, F]
        
        # Mismatch = sum(mask_pos) - x @ W_eff.T
        projection = F.linear(x, W_eff) # [B, half]
        mismatch_bias = mask_pos.sum(dim=1).unsqueeze(0) # [1, half]
        mismatch = mismatch_bias - projection
        
        if self.operator == "capacity":
            capacity = self._clause_capacity(mask_pos, mask_inv)
            raw = capacity - mismatch
            return self._straight_relu(raw)
            
        else: # product
            scaled = torch.clamp((mismatch) * self.product_scale, min=0.0, max=10.0)
            return torch.exp(-scaled)

    def _clause_outputs(self, x: torch.Tensor, use_ste: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Check for Massive Speedup opportunity (Fused Triton Kernel)
        # Criteria: Triton enabled, STE used, product/capacity operator, no complex constraints
        triton_status = get_triton_status()
        use_fused_kernel = (
            triton_status['triton_enabled'] and 
            TRITON_KERNELS_AVAILABLE and 
            fused_clause_outputs is not None and
            use_ste and
            x.is_cuda and
            self.operator in ('product', 'capacity') and  # Extended to include capacity
            getattr(self, 'literal_dropout', 0.0) == 0.0 and
            getattr(self, 'max_literals', None) is None
        )

        if use_fused_kernel:
            # Fused Path: Compute both clause banks in one launch
            clause_outputs = fused_clause_outputs(
                x, self.pos_logits, self.neg_logits,
                self.ternary_band, self.ste_temperature, self.product_scale,
                operator=self.operator,  # Pass operator for capacity support
            )
            half = self.n_clauses // 2
            pos_strength = clause_outputs[:, :half]
            neg_strength = clause_outputs[:, half:]
            clause_votes = torch.cat([pos_strength, -neg_strength], dim=1)

            if self.vote_clamp is not None:
                clause_votes = clause_votes.clamp(-self.vote_clamp, self.vote_clamp)
            if self.training and self.clause_dropout > 0.0:
                clause_votes = F.dropout(clause_votes, p=self.clause_dropout, training=True)
                
            return pos_strength, neg_strength, clause_votes

        # Standard Path (Optimized PyTorch or Fallback)
        # 1. Generate Masks
        all_logits = torch.cat([self.pos_logits, self.neg_logits], dim=0)
        temp = max(self.ste_temperature, 1e-6)
        
        if use_ste:
             mask_all = _ste_ternary(all_logits, self.ternary_band, self.ste_temperature, self.ste_gradient_mode)
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
        pos_strength = self._strength(x, pos_pos, pos_inv)
        neg_strength = self._strength(x, neg_pos, neg_inv)
        
        # 3. Combine and Vote
        clause_votes = torch.cat([pos_strength, -neg_strength], dim=1)

        if self.vote_clamp is not None:
            clause_votes = clause_votes.clamp(-self.vote_clamp, self.vote_clamp)
        if self.training and self.clause_dropout > 0.0:
            clause_votes = F.dropout(clause_votes, p=self.clause_dropout, training=True)
            
        return pos_strength, neg_strength, clause_votes


class TritonSTCM(OptimizedSTCM):
    """
    STCM with optional Triton kernel acceleration and packed ternary weights.
    
    This class extends OptimizedSTCM with:
    1. Packed ternary weight storage (16x memory reduction)
    2. Optional Triton kernel for accelerated matmul (when available)
    3. Bitplane16 tensor core acceleration for maximum performance
    
    The packed weights are stored as int16 tensors with 8 weights per int16.
    This significantly reduces memory bandwidth requirements for inference.
    
    To enable packed weights for inference:
        model.eval()
        model.freeze()  # Pre-computes and packs weights
    """
    
    def __init__(
        self, 
        *args, 
        use_packed_weights: bool = True, 
        use_v2_kernel: bool = True, 
        use_bitplane16: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.use_packed_weights = use_packed_weights and TRITON_KERNELS_AVAILABLE
        self.use_v2_kernel = use_v2_kernel
        self.use_bitplane16 = use_bitplane16 and BITPLANE16_AVAILABLE
        self.is_frozen = False
        
        # Buffers for packed weights (int32 format for V2 kernel)
        self.register_buffer('packed_W_pos', None)
        self.register_buffer('packed_W_neg', None)
        self.register_buffer('bias_pos', None)
        self.register_buffer('bias_neg', None)
        self.packed_shape = None
        
        # Buffers for int16 packed weights (bitplane16 format)
        self.register_buffer('packed16_W_pos', None)
        self.register_buffer('packed16_W_neg', None)
        self.packed16_shape = None
        
        if self.use_packed_weights and not TRITON_KERNELS_AVAILABLE:
            import warnings
            warnings.warn("Triton kernels not available, falling back to standard implementation")
            
    def freeze(self):
        """Pre-compute and pack weights for inference."""
        if not self.use_packed_weights and not self.use_bitplane16:
            return
            
        # Get current masks (using inference logic)
        with torch.no_grad():
            _, _, _ = self._clause_outputs(torch.zeros(1, self.n_features, device=self.pos_logits.device), use_ste=True)
            
            # Generate masks
            all_logits = torch.cat([self.pos_logits, self.neg_logits], dim=0)
            mask_all = _ste_ternary(all_logits, self.ternary_band, self.ste_temperature)
            
            half = self.n_clauses // 2
            pos_all = torch.clamp(mask_all, min=0.0)
            inv_all = torch.clamp(-mask_all, min=0.0)
            
            pos_pos = pos_all[:half]
            pos_inv = inv_all[:half]
            neg_pos = pos_all[half:]
            neg_inv = inv_all[half:]
            
            # Calculate W_eff and bias
            W_eff_pos = pos_pos - pos_inv
            W_eff_neg = neg_pos - neg_inv
            
            self.bias_pos = pos_pos.sum(dim=1).unsqueeze(0)
            self.bias_neg = neg_pos.sum(dim=1).unsqueeze(0)
            
            # Pack weights in int32 format (V2 kernel)
            if self.use_packed_weights and TRITON_KERNELS_AVAILABLE:
                self.packed_W_pos, self.packed_shape = pack_ternary_int32(W_eff_pos)
                self.packed_W_neg, _ = pack_ternary_int32(W_eff_neg)
            
            # Pack weights in int16 format (bitplane16 tensor core kernel)
            if self.use_bitplane16 and BITPLANE16_AVAILABLE and pack_ternary_int16 is not None:
                self.packed16_W_pos, self.packed16_shape = pack_ternary_int16(W_eff_pos)
                self.packed16_W_neg, _ = pack_ternary_int16(W_eff_neg)
            
            self.is_frozen = True
            
    def unfreeze(self):
        self.is_frozen = False
        self.packed_W_pos = None
        self.packed_W_neg = None
    
    def _strength(self, x: torch.Tensor, mask_pos: torch.Tensor, mask_inv: torch.Tensor) -> torch.Tensor:
        # Check for custom fuzzy operator (fallback to original logic)
        if hasattr(self, 'operator_impl') and self.operator_impl is not None:
            return FuzzyPatternTM_STCM._strength(self, x, mask_pos, mask_inv)
        
        # Check if we use frozen packed weights
        use_frozen = self.is_frozen and not self.training and self.use_packed_weights
        
        if use_frozen:
            # Identify if we are computing pos or neg strength
            # We can guess by shape match or pass argument?
            # _strength is called twice: once for pos, once for neg.
            # We assume first call is pos, second is neg? Risky.
            
            # Better check: compare mask_pos sum with stored bias?
            # Or just check equality of tensors? Slow.
            
            # Let's assume standard flow: pos then neg.
            # But simpler: _strength is generic.
            # If we are frozen, we shouldn't even be using mask_pos/mask_inv arguments!
            # But the signature requires them.
            
            # Hack: Compare mask_pos to self.bias_pos source? 
            # Let's just use dynamic packing if not frozen, and handle frozen logic in _clause_outputs?
            # No, _clause_outputs calls _strength.
            
            # Let's override _clause_outputs to handle frozen case entirely.
            pass
            
        # Fallback to dynamic (training/unfrozen)
        W_eff = mask_pos - mask_inv
        
        use_packed = (
            self.use_packed_weights 
            and TRITON_KERNELS_AVAILABLE 
            and self.enable_packed_inference # Legacy flag, keep for compat
            and not self.training
        )
        
        if use_packed:
            if self.use_v2_kernel:
                projection = ternary_linear_v2_cached(x, W_eff)
            else:
                projection = ternary_linear_v1_cached(x, W_eff)
        else:
            projection = F.linear(x, W_eff)
        
        mismatch_bias = mask_pos.sum(dim=1).unsqueeze(0)
        mismatch = mismatch_bias - projection
        
        if self.operator == "capacity":
            capacity = self._clause_capacity(mask_pos, mask_inv)
            raw = capacity - mismatch
            return self._straight_relu(raw)
        else:  # product
            scaled = torch.clamp(mismatch * self.product_scale, min=0.0, max=10.0)
            return torch.exp(-scaled)

    def _clause_outputs(self, x: torch.Tensor, use_ste: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.is_frozen and not self.training:
            # Fast path for frozen inference using packed weights
            
            # Choose kernel based on availability
            if self.use_bitplane16 and self.packed16_W_pos is not None and bitplane16_tc_matmul is not None:
                # Use highest performance bitplane16 tensor core kernel
                proj_pos = bitplane16_tc_matmul(x, self.packed16_W_pos, self.packed16_shape)
                proj_neg = bitplane16_tc_matmul(x, self.packed16_W_neg, self.packed16_shape)
            elif self.use_packed_weights and self.packed_W_pos is not None:
                # Use V2 packed kernel
                proj_pos = ternary_linear_v2(x, self.packed_W_pos, self.packed_shape)
                proj_neg = ternary_linear_v2(x, self.packed_W_neg, self.packed_shape)
            else:
                # Fallback to standard path
                return super()._clause_outputs(x, use_ste)
                
            mismatch_pos = self.bias_pos - proj_pos
            mismatch_neg = self.bias_neg - proj_neg
            
            if self.operator == "capacity":
                # For capacity operator, need to compute capacity too
                # Precompute during freeze? For now, fall back to full computation
                return super()._clause_outputs(x, use_ste)
            else:  # product
                # Fast path for product operator
                scaled_pos = torch.clamp(mismatch_pos * self.product_scale, min=0.0, max=10.0)
                scaled_neg = torch.clamp(mismatch_neg * self.product_scale, min=0.0, max=10.0)
                pos_strength = torch.exp(-scaled_pos)
                neg_strength = torch.exp(-scaled_neg)
                
                clause_votes = torch.cat([pos_strength, -neg_strength], dim=1)
                
                if self.vote_clamp is not None:
                    clause_votes = clause_votes.clamp(-self.vote_clamp, self.vote_clamp)
                
                return pos_strength, neg_strength, clause_votes

        return super()._clause_outputs(x, use_ste)
