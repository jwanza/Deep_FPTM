"""
Compiled STCM using PyTorch 2.0 torch.compile for automatic kernel fusion.

This module provides CompiledSTCM which wraps the forward pass with torch.compile
to achieve 2-3x speedup through:
1. Automatic kernel fusion (reducing memory bandwidth)
2. Operator fusion (fewer kernel launches)
3. Memory planning (better cache utilization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .tm_optimized import OptimizedSTCM
from .tm import FuzzyPatternTM_STCM, _ste_ternary


class CompiledSTCM(OptimizedSTCM):
    """
    STCM with torch.compile optimization for automatic kernel fusion.
    
    This class applies PyTorch 2.0's torch.compile to the forward pass,
    enabling automatic optimization including:
    - Kernel fusion (combine multiple operations)
    - Memory planning (reduce allocations)
    - Operator scheduling (better GPU utilization)
    
    Args:
        compile_mode: Compilation mode for torch.compile
            - "default": Good balance of compile time and speedup
            - "reduce-overhead": Minimize kernel launch overhead (best for small batches)
            - "max-autotune": Maximum optimization (longer compile, best speedup)
        compile_backend: Backend for compilation ("inductor" is default and recommended)
        compile_fullgraph: Whether to compile the full graph (True recommended)
        *args, **kwargs: Passed to OptimizedSTCM
    
    Example:
        >>> model = CompiledSTCM(n_features=784, n_clauses=512, n_classes=10)
        >>> # First forward pass triggers compilation (slower)
        >>> out = model(x)
        >>> # Subsequent passes are 2-3x faster
        >>> out = model(x)
    """
    
    def __init__(
        self,
        *args,
        compile_mode: str = "reduce-overhead",
        compile_backend: str = "inductor",
        compile_fullgraph: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.compile_mode = compile_mode
        self.compile_backend = compile_backend
        self.compile_fullgraph = compile_fullgraph
        self._compiled = False
        self._compiled_clause_outputs = None
        
    def _ensure_compiled(self):
        """Lazily compile the clause outputs function."""
        if not self._compiled:
            try:
                self._compiled_clause_outputs = torch.compile(
                    self._clause_outputs_impl,
                    mode=self.compile_mode,
                    backend=self.compile_backend,
                    fullgraph=self.compile_fullgraph,
                )
                self._compiled = True
            except Exception as e:
                import warnings
                warnings.warn(f"torch.compile failed: {e}. Falling back to eager mode.")
                self._compiled_clause_outputs = self._clause_outputs_impl
                self._compiled = True
    
    def _clause_outputs_impl(self, x: torch.Tensor, use_ste: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Clause outputs implementation that will be compiled.
        
        This is the hot path that benefits most from compilation.
        """
        # 1. Generate Masks
        all_logits = torch.cat([self.pos_logits, self.neg_logits], dim=0)
        temp = max(self.ste_temperature, 1e-6)
        
        if use_ste:
            mask_all = _ste_ternary(all_logits, self.ternary_band, self.ste_temperature)
        else:
            mask_all = torch.tanh(all_logits / self.ste_temperature)
             
        half = self.n_clauses // 2
        
        # Split Masks
        hard_pos_all = torch.clamp(mask_all, min=0.0)
        hard_inv_all = torch.clamp(-mask_all, min=0.0)
        
        pos_pos = hard_pos_all[:half]
        pos_inv = hard_inv_all[:half]
        neg_pos = hard_pos_all[half:]
        neg_inv = hard_inv_all[half:]
        
        # Enforce budget
        pos_pos, pos_inv = self._enforce_literal_budget(pos_pos, pos_inv)
        neg_pos, neg_inv = self._enforce_literal_budget(neg_pos, neg_inv)
        
        # 2. Calculate Strengths (Optimized path)
        pos_strength = self._strength_compiled(x, pos_pos, pos_inv)
        neg_strength = self._strength_compiled(x, neg_pos, neg_inv)
        
        # 3. Combine and Vote
        clause_votes = torch.cat([pos_strength, -neg_strength], dim=1)

        if self.vote_clamp is not None:
            clause_votes = clause_votes.clamp(-self.vote_clamp, self.vote_clamp)
            
        return pos_strength, neg_strength, clause_votes
    
    def _strength_compiled(self, x: torch.Tensor, mask_pos: torch.Tensor, mask_inv: torch.Tensor) -> torch.Tensor:
        """
        Optimized strength calculation for compilation.
        
        This is a streamlined version that avoids conditionals for better compilation.
        """
        # W_eff projection
        W_eff = mask_pos - mask_inv
        projection = torch.mm(x, W_eff.t())
        
        # Mismatch calculation
        mismatch_bias = mask_pos.sum(dim=1).unsqueeze(0)
        mismatch = mismatch_bias - projection
        
        # Capacity operator (most common)
        if self.operator == "capacity":
            capacity = self._clause_capacity(mask_pos, mask_inv)
            raw = capacity - mismatch
            # Straight-through ReLU
            clamped = F.leaky_relu(raw, negative_slope=0.01)
            return raw + (clamped - raw).detach()
        else:
            scaled = torch.clamp(mismatch * self.product_scale, min=0.0, max=10.0)
            return torch.exp(-scaled)
    
    def _clause_outputs(self, x: torch.Tensor, use_ste: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Override to use compiled version."""
        self._ensure_compiled()
        
        # Apply dropout in training mode (not compiled for dynamic behavior)
        pos_strength, neg_strength, clause_votes = self._compiled_clause_outputs(x, use_ste)
        
        if self.training and self.clause_dropout > 0.0:
            clause_votes = F.dropout(clause_votes, p=self.clause_dropout, training=True)
            
        return pos_strength, neg_strength, clause_votes


class DeepCompiledSTCM(nn.Module):
    """
    Deep STCM network with compiled layers for maximum performance.
    
    Each layer uses CompiledSTCM for automatic kernel fusion.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        n_classes: int,
        n_clauses: int = 256,
        dropout: float = 0.1,
        tau: float = 0.5,
        compile_mode: str = "reduce-overhead",
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        prev_dim = input_dim
        for h in hidden_dims:
            self.layers.append(CompiledSTCM(
                n_features=prev_dim,
                n_clauses=n_clauses,
                n_classes=h,
                tau=tau,
                compile_mode=compile_mode,
            ))
            self.norms.append(nn.LayerNorm(h))
            prev_dim = h
        
        # Final classifier
        self.head = CompiledSTCM(
            n_features=prev_dim,
            n_clauses=n_clauses,
            n_classes=n_classes,
            tau=tau,
            compile_mode=compile_mode,
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Flatten if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        for layer, norm in zip(self.layers, self.norms):
            out = layer(x)[0]  # Get logits
            out = norm(out)
            out = F.relu(out)
            out = self.dropout(out)
            x = out
        
        logits, clause_outputs = self.head(x)[:2]
        return logits, clause_outputs

