"""
Hierarchical Multi-Resolution Clause Machine.

Processes features at multiple resolutions, with coarse levels
capturing global patterns and fine levels capturing details.

Key Innovation:
Different abstraction levels require different binarization granularity.
High-level concepts need coarse patterns (fewer bits), while
fine-grained details need precise encoding.

Architecture:
1. Multi-Resolution Binarization: Different threshold counts per level
2. Level-Specific Clauses: Each level has its own clause bank
3. Cross-Level Attention: Levels exchange information
4. Hierarchical Fusion: Combine levels with learned gates

Benefits:
- Captures patterns at multiple scales
- Efficient: coarse levels are cheap
- Interpretable: clear hierarchy of concepts
- Progressive refinement possible
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..tm import FuzzyPatternTM_STCM, prepare_tm_input


# =============================================================================
# Multi-Resolution Binarizer
# =============================================================================


class MultiResolutionBinarizer(nn.Module):
    """
    Binarizes features at multiple resolutions.
    
    Higher resolutions use more thresholds (finer granularity),
    lower resolutions use fewer thresholds (coarser granularity).
    
    Args:
        n_features: Number of input features
        resolutions: List of number of thresholds per level
        temperature: Binarization temperature
    """
    
    def __init__(
        self,
        n_features: int,
        resolutions: List[int] = [2, 4, 8],
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_features = n_features
        self.resolutions = resolutions
        self.n_levels = len(resolutions)
        self.temperature = temperature
        
        # Thresholds for each level
        self.thresholds = nn.ParameterList([
            nn.Parameter(torch.linspace(0.1, 0.9, res).unsqueeze(1).expand(-1, n_features))
            for res in resolutions
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        use_ste: bool = True,
    ) -> List[torch.Tensor]:
        """
        Binarize at multiple resolutions.
        
        Args:
            x: Input features [batch, n_features]
            use_ste: Use straight-through estimator
            
        Returns:
            List of binary tensors, one per resolution level
        """
        outputs = []
        
        for level_idx, thresholds in enumerate(self.thresholds):
            # thresholds: [n_thresholds, n_features]
            # x: [batch, n_features]
            
            # Compare with all thresholds
            x_exp = x.unsqueeze(1)  # [batch, 1, n_features]
            thresh_exp = thresholds.unsqueeze(0)  # [1, n_thresholds, n_features]
            
            # Soft thresholding
            soft = torch.sigmoid((x_exp - thresh_exp) / self.temperature)
            
            if use_ste:
                hard = (x_exp > thresh_exp).float()
                soft = hard + (soft - soft.detach())
            
            # Flatten: [batch, n_thresholds * n_features]
            binary = soft.view(x.shape[0], -1)
            outputs.append(binary)
        
        return outputs


# =============================================================================
# Cross-Level Attention
# =============================================================================


class CrossLevelAttention(nn.Module):
    """
    Attention mechanism for information flow between hierarchy levels.
    
    Allows coarse levels to guide fine levels and vice versa.
    
    Args:
        dims: List of dimensions for each level
        n_heads: Number of attention heads
    """
    
    def __init__(
        self,
        dims: List[int],
        n_heads: int = 4,
    ):
        super().__init__()
        self.dims = dims
        self.n_levels = len(dims)
        
        # Project all levels to common dimension
        common_dim = max(dims)
        self.projections = nn.ModuleList([
            nn.Linear(d, common_dim) for d in dims
        ])
        
        # Cross-attention between adjacent levels
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(common_dim, n_heads, batch_first=True)
            for _ in range(self.n_levels - 1)
        ])
        
        # Project back to original dimensions
        self.output_projections = nn.ModuleList([
            nn.Linear(common_dim, d) for d in dims
        ])
    
    def forward(
        self,
        level_features: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Apply cross-level attention.
        
        Args:
            level_features: List of [batch, level_dim] tensors
            
        Returns:
            Enhanced features for each level
        """
        batch_size = level_features[0].shape[0]
        
        # Project to common dimension
        projected = [
            proj(feat).unsqueeze(1)  # [batch, 1, common_dim]
            for proj, feat in zip(self.projections, level_features)
        ]
        
        # Apply attention between adjacent levels
        enhanced = [projected[0]]
        
        for i, attn in enumerate(self.attentions):
            # Current level attends to previous level
            query = projected[i + 1]
            key = projected[i]
            value = projected[i]
            
            attended, _ = attn(query, key, value)
            enhanced.append(attended + projected[i + 1])  # Residual
        
        # Project back and squeeze
        outputs = [
            proj(feat.squeeze(1))
            for proj, feat in zip(self.output_projections, enhanced)
        ]
        
        return outputs


# =============================================================================
# Hierarchical Level
# =============================================================================


class HierarchicalLevel(nn.Module):
    """
    Single level in the hierarchy.
    
    Contains its own clause bank operating at its resolution.
    
    Args:
        n_binary_features: Number of binary features at this level
        n_clauses: Number of clauses
        n_classes: Number of output classes
        operator: Clause operator
    """
    
    def __init__(
        self,
        n_binary_features: int,
        n_clauses: int,
        n_classes: int,
        operator: str = "capacity",
    ):
        super().__init__()
        self.n_binary_features = n_binary_features
        self.n_clauses = n_clauses
        
        # TM for this level
        self.tm = FuzzyPatternTM_STCM(
            n_features=n_binary_features,
            n_clauses=n_clauses,
            n_classes=n_classes,
            operator=operator,
        )
        
        # Level-specific gate
        self.gate = nn.Sequential(
            nn.Linear(n_clauses, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        x_binary: torch.Tensor,
        use_ste: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process at this level.
        
        Args:
            x_binary: Binary features for this level
            use_ste: Use STE
            
        Returns:
            (logits, clause_outputs, gate_value)
        """
        logits, clauses = self.tm(x_binary, use_ste=use_ste, skip_norm=True)
        gate = self.gate(clauses)
        return logits, clauses, gate


# =============================================================================
# Hierarchical Multi-Resolution TM
# =============================================================================


class HierarchicalMultiResolutionTM(nn.Module):
    """
    Hierarchical Multi-Resolution Tsetlin Machine.
    
    Processes inputs at multiple resolutions with level-specific
    clause banks and cross-level attention for information flow.
    
    Args:
        n_features: Number of input features
        n_clauses_per_level: Clauses per hierarchy level
        n_classes: Number of output classes
        resolutions: List of threshold counts per level
        use_cross_attention: Enable cross-level attention
        operator: Clause operator
    """
    
    def __init__(
        self,
        n_features: int,
        n_clauses_per_level: List[int],
        n_classes: int,
        resolutions: List[int] = [2, 4, 8],
        use_cross_attention: bool = True,
        operator: str = "capacity",
    ):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_levels = len(resolutions)
        self.resolutions = resolutions
        
        assert len(n_clauses_per_level) == len(resolutions), \
            "Must have same number of clause counts as resolutions"
        
        # Multi-resolution binarizer
        self.binarizer = MultiResolutionBinarizer(
            n_features=n_features,
            resolutions=resolutions,
        )
        
        # Binary feature dimensions per level
        level_dims = [res * n_features for res in resolutions]
        
        # Hierarchical levels
        self.levels = nn.ModuleList([
            HierarchicalLevel(
                n_binary_features=level_dims[i],
                n_clauses=n_clauses_per_level[i],
                n_classes=n_classes,
                operator=operator,
            )
            for i in range(self.n_levels)
        ])
        
        # Cross-level attention
        if use_cross_attention:
            self.cross_attention = CrossLevelAttention(
                dims=n_clauses_per_level,
                n_heads=4,
            )
        else:
            self.cross_attention = None
        
        # Final fusion
        total_clauses = sum(n_clauses_per_level)
        self.fusion = nn.Sequential(
            nn.Linear(total_clauses, total_clauses // 2),
            nn.GELU(),
            nn.Linear(total_clauses // 2, n_classes),
        )
        
        # Level importance weights (learnable)
        self.level_importance = nn.Parameter(torch.ones(self.n_levels))
    
    def forward(
        self,
        x: torch.Tensor,
        use_ste: bool = True,
        return_level_outputs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hierarchical forward pass.
        
        Args:
            x: Input features [batch, n_features]
            use_ste: Use STE
            return_level_outputs: Return per-level details
            
        Returns:
            (logits, combined_clauses)
        """
        # Prepare input
        x_flat = prepare_tm_input(x, n_features=self.n_features)
        
        # Multi-resolution binarization
        binary_levels = self.binarizer(x_flat, use_ste=use_ste)
        
        # Process each level
        level_logits = []
        level_clauses = []
        level_gates = []
        
        for level_idx, level in enumerate(self.levels):
            logits, clauses, gate = level(binary_levels[level_idx], use_ste=use_ste)
            level_logits.append(logits)
            level_clauses.append(clauses)
            level_gates.append(gate)
        
        # Cross-level attention if enabled
        if self.cross_attention is not None:
            level_clauses = self.cross_attention(level_clauses)
        
        # Combine clauses from all levels
        combined_clauses = torch.cat(level_clauses, dim=-1)
        
        # Weighted sum of level logits
        importance = F.softmax(self.level_importance, dim=0)
        weighted_logits = sum(
            imp * logits for imp, logits in zip(importance, level_logits)
        )
        
        # Fusion for final prediction
        fusion_logits = self.fusion(combined_clauses)
        
        # Combine weighted sum and fusion
        final_logits = 0.5 * weighted_logits + 0.5 * fusion_logits
        
        if return_level_outputs:
            return {
                "logits": final_logits,
                "clause_outputs": combined_clauses,
                "level_logits": level_logits,
                "level_clauses": level_clauses,
                "level_gates": level_gates,
                "level_importance": importance,
            }
        
        return final_logits, combined_clauses
    
    def get_level_contributions(self) -> torch.Tensor:
        """
        Get normalized importance of each level.
        
        Returns:
            Softmax-normalized level importance weights
        """
        return F.softmax(self.level_importance, dim=0)



