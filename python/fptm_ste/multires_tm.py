"""
Multi-Resolution STCM Architecture.

This module implements multi-resolution clause processing where inputs are
processed at multiple binarization thresholds simultaneously. Different
thresholds capture different levels of feature activation, and the results
are fused via learned attention.

Key innovations:
1. Parallel STCM branches with different tau thresholds
2. Different ternary_band values for varying strictness
3. Optional different operators per branch
4. Learned attention fusion of multi-resolution outputs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tm import FuzzyPatternTM_STCM


class MultiResolutionBranch(nn.Module):
    """
    Single branch of the multi-resolution architecture.
    
    Each branch processes input with a specific tau threshold
    and configuration.
    """
    
    def __init__(
        self,
        n_features: int,
        n_clauses: int,
        n_classes: int,
        tau: float,
        ternary_band: float = 0.0,
        operator: str = "capacity",
        lf: int = 4,
        clause_dropout: float = 0.0,
    ):
        super().__init__()
        self.tau = tau
        
        self.stcm = FuzzyPatternTM_STCM(
            n_features=n_features,
            n_clauses=n_clauses,
            n_classes=n_classes,
            tau=tau,
            ternary_band=ternary_band,
            operator=operator,
            lf=lf,
            clause_dropout=clause_dropout,
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for this resolution branch.
        
        Returns:
            (logits, clause_outputs)
        """
        return self.stcm(x)


class ResolutionFusion(nn.Module):
    """
    Fusion module for combining multi-resolution outputs.
    
    Supports multiple fusion strategies:
    - 'attention': Learned attention weights per resolution
    - 'concat': Concatenation followed by projection
    - 'avg': Simple averaging
    - 'max': Element-wise maximum
    - 'gated': Gated fusion with input-dependent gates
    """
    
    def __init__(
        self,
        n_resolutions: int,
        n_classes: int,
        n_clauses: int,
        fusion_type: str = "attention",
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.n_resolutions = n_resolutions
        self.fusion_type = fusion_type
        
        if fusion_type == "attention":
            # Learn attention weights for each resolution
            self.resolution_weights = nn.Parameter(torch.zeros(n_resolutions))
            self.temperature = nn.Parameter(torch.ones(1))
        
        elif fusion_type == "concat":
            hidden_dim = hidden_dim or n_classes * 2
            self.fusion_net = nn.Sequential(
                nn.Linear(n_classes * n_resolutions, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_classes),
            )
        
        elif fusion_type == "gated":
            # Input-dependent gating
            self.gate_net = nn.Sequential(
                nn.Linear(n_clauses, n_resolutions),
                nn.Softmax(dim=-1),
            )
        
        elif fusion_type in ("avg", "max"):
            pass  # No learnable parameters needed
        
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(
        self,
        logits_list: List[torch.Tensor],
        clause_outputs_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse outputs from multiple resolutions.
        
        Args:
            logits_list: List of [batch, n_classes] tensors
            clause_outputs_list: List of [batch, n_clauses] tensors
            
        Returns:
            (fused_logits, fused_clauses)
        """
        # Stack logits: [batch, n_res, n_classes]
        stacked_logits = torch.stack(logits_list, dim=1)
        stacked_clauses = torch.stack(clause_outputs_list, dim=1)
        
        if self.fusion_type == "attention":
            # Compute attention weights
            weights = F.softmax(self.resolution_weights / self.temperature, dim=0)
            weights = weights.view(1, -1, 1)  # [1, n_res, 1]
            
            fused_logits = (stacked_logits * weights).sum(dim=1)
            fused_clauses = (stacked_clauses * weights).sum(dim=1)
        
        elif self.fusion_type == "concat":
            batch_size = stacked_logits.shape[0]
            flat_logits = stacked_logits.view(batch_size, -1)
            fused_logits = self.fusion_net(flat_logits)
            fused_clauses = stacked_clauses.mean(dim=1)
        
        elif self.fusion_type == "gated":
            # Use clause outputs to compute gates
            gate_input = stacked_clauses.mean(dim=1)  # [batch, n_clauses]
            gates = self.gate_net(gate_input)  # [batch, n_res]
            gates = gates.unsqueeze(-1)  # [batch, n_res, 1]
            
            fused_logits = (stacked_logits * gates).sum(dim=1)
            fused_clauses = (stacked_clauses * gates).sum(dim=1)
        
        elif self.fusion_type == "avg":
            fused_logits = stacked_logits.mean(dim=1)
            fused_clauses = stacked_clauses.mean(dim=1)
        
        elif self.fusion_type == "max":
            fused_logits = stacked_logits.max(dim=1).values
            fused_clauses = stacked_clauses.max(dim=1).values
        
        return fused_logits, fused_clauses


class MultiResolutionSTCM(nn.Module):
    """
    Multi-Resolution STCM Architecture.
    
    Processes input at multiple binarization thresholds simultaneously
    and fuses the results using learned attention. This allows capturing
    features at different activation levels.
    
    Args:
        n_features: Number of input features
        n_clauses: Number of clauses per branch
        n_classes: Number of output classes
        tau_values: List of tau thresholds for each branch
        ternary_bands: List of ternary band values (or single value)
        operators: List of operators per branch (or single value)
        fusion_type: How to fuse multi-resolution outputs
        shared_voting: Whether to share voting weights across branches
        clause_dropout: Clause dropout rate
        lf: Literal limit factor
    """
    
    def __init__(
        self,
        n_features: int,
        n_clauses: int,
        n_classes: int,
        tau_values: Optional[List[float]] = None,
        ternary_bands: Optional[Union[float, List[float]]] = None,
        operators: Optional[Union[str, List[str]]] = None,
        fusion_type: str = "attention",
        shared_voting: bool = False,
        clause_dropout: float = 0.0,
        lf: int = 4,
    ):
        super().__init__()
        
        # Default tau values span the typical range
        if tau_values is None:
            tau_values = [0.3, 0.5, 0.7]
        
        self.n_resolutions = len(tau_values)
        self.tau_values = tau_values
        self.n_features = n_features
        self.n_clauses = n_clauses
        self.n_classes = n_classes
        
        # Handle ternary_bands
        if ternary_bands is None:
            ternary_bands = [0.0] * self.n_resolutions
        elif isinstance(ternary_bands, (int, float)):
            ternary_bands = [float(ternary_bands)] * self.n_resolutions
        
        # Handle operators
        if operators is None:
            operators = ["capacity"] * self.n_resolutions
        elif isinstance(operators, str):
            operators = [operators] * self.n_resolutions
        
        # Create branches
        self.branches = nn.ModuleList()
        for i, tau in enumerate(tau_values):
            branch = MultiResolutionBranch(
                n_features=n_features,
                n_clauses=n_clauses,
                n_classes=n_classes,
                tau=tau,
                ternary_band=ternary_bands[i],
                operator=operators[i],
                lf=lf,
                clause_dropout=clause_dropout,
            )
            self.branches.append(branch)
        
        # Fusion module
        self.fusion = ResolutionFusion(
            n_resolutions=self.n_resolutions,
            n_classes=n_classes,
            n_clauses=n_clauses,
            fusion_type=fusion_type,
        )
        
        # Optional shared voting weights
        if shared_voting:
            self.shared_voting = nn.Parameter(
                torch.randn(n_clauses, n_classes) * 0.1
            )
        else:
            self.shared_voting = None
    
    def forward(
        self,
        x: torch.Tensor,
        return_branch_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict]:
        """
        Forward pass through all resolution branches.
        
        Args:
            x: Input tensor [batch, n_features]
            return_branch_outputs: If True, return dict with per-branch outputs
            
        Returns:
            (fused_logits, fused_clauses) or dict with all outputs
        """
        logits_list = []
        clauses_list = []
        
        for branch in self.branches:
            logits, clauses = branch(x)
            logits_list.append(logits)
            clauses_list.append(clauses)
        
        # Fuse outputs
        fused_logits, fused_clauses = self.fusion(logits_list, clauses_list)
        
        if return_branch_outputs:
            return {
                "logits": fused_logits,
                "clauses": fused_clauses,
                "branch_logits": logits_list,
                "branch_clauses": clauses_list,
                "tau_values": self.tau_values,
            }
        
        return fused_logits, fused_clauses
    
    def get_resolution_weights(self) -> Optional[torch.Tensor]:
        """Get learned resolution weights if using attention fusion."""
        if hasattr(self.fusion, 'resolution_weights'):
            with torch.no_grad():
                weights = F.softmax(
                    self.fusion.resolution_weights / self.fusion.temperature,
                    dim=0
                )
            return weights
        return None


class AdaptiveThresholdSTCM(nn.Module):
    """
    STCM with learnable, input-dependent thresholds.
    
    Instead of fixed tau values, this model learns to predict
    optimal thresholds per sample based on input statistics.
    """
    
    def __init__(
        self,
        n_features: int,
        n_clauses: int,
        n_classes: int,
        base_tau: float = 0.5,
        tau_range: float = 0.3,
        operator: str = "capacity",
    ):
        super().__init__()
        self.base_tau = base_tau
        self.tau_range = tau_range
        
        # Threshold predictor
        self.tau_predictor = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )
        
        # Base STCM (tau will be overridden per-sample)
        self.stcm = FuzzyPatternTM_STCM(
            n_features=n_features,
            n_clauses=n_clauses,
            n_classes=n_classes,
            tau=base_tau,
            operator=operator,
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with adaptive thresholds.
        
        Returns:
            (logits, clauses, predicted_tau)
        """
        # Predict threshold adjustment
        tau_adj = self.tau_predictor(x)  # [batch, 1]
        predicted_tau = self.base_tau + (tau_adj - 0.5) * 2 * self.tau_range
        
        # For now, use batch mean tau (individual tau per sample is complex)
        batch_tau = predicted_tau.mean()
        
        # Temporarily update tau
        original_tau = self.stcm.tau
        self.stcm.tau = float(batch_tau)
        
        logits, clauses = self.stcm(x)
        
        # Restore tau
        self.stcm.tau = original_tau
        
        return logits, clauses, predicted_tau.squeeze(-1)


class CascadeResolutionSTCM(nn.Module):
    """
    Cascaded multi-resolution STCM.
    
    Processes resolutions sequentially, with each stage refining
    the previous stage's output. Early resolutions provide coarse
    patterns, later resolutions add fine details.
    """
    
    def __init__(
        self,
        n_features: int,
        n_clauses: int,
        n_classes: int,
        n_stages: int = 3,
        tau_start: float = 0.3,
        tau_end: float = 0.7,
        residual: bool = True,
    ):
        super().__init__()
        self.n_stages = n_stages
        self.residual = residual
        
        # Linearly spaced tau values
        tau_values = torch.linspace(tau_start, tau_end, n_stages).tolist()
        
        # Create stages
        self.stages = nn.ModuleList()
        for tau in tau_values:
            stage = FuzzyPatternTM_STCM(
                n_features=n_features,
                n_clauses=n_clauses,
                n_classes=n_classes,
                tau=tau,
            )
            self.stages.append(stage)
        
        # Stage combination weights
        self.stage_weights = nn.Parameter(torch.ones(n_stages) / n_stages)
        
        # Optional refinement layers between stages
        self.refiners = nn.ModuleList([
            nn.Linear(n_classes, n_classes, bias=False)
            for _ in range(n_stages - 1)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cascaded forward pass.
        
        Returns:
            (final_logits, concatenated_clauses)
        """
        weights = F.softmax(self.stage_weights, dim=0)
        
        accumulated_logits = None
        all_clauses = []
        
        for i, stage in enumerate(self.stages):
            logits, clauses = stage(x)
            all_clauses.append(clauses)
            
            # Refine with previous accumulated output
            if accumulated_logits is not None and i > 0:
                refined_prev = self.refiners[i - 1](accumulated_logits)
                if self.residual:
                    logits = logits + 0.1 * refined_prev
            
            # Weighted accumulation
            if accumulated_logits is None:
                accumulated_logits = weights[i] * logits
            else:
                accumulated_logits = accumulated_logits + weights[i] * logits
        
        # Concatenate all clause outputs
        concat_clauses = torch.cat(all_clauses, dim=1)
        
        return accumulated_logits, concat_clauses


class HierarchicalResolutionSTCM(nn.Module):
    """
    Hierarchical multi-resolution STCM with feature grouping.
    
    Different feature groups are processed at different resolutions,
    allowing the model to learn optimal thresholds per feature type.
    """
    
    def __init__(
        self,
        n_features: int,
        n_clauses: int,
        n_classes: int,
        n_groups: int = 4,
        base_tau: float = 0.5,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_groups = n_groups
        
        # Features per group
        self.features_per_group = n_features // n_groups
        assert n_features % n_groups == 0, "n_features must be divisible by n_groups"
        
        # Learnable tau per group
        self.group_tau_logits = nn.Parameter(torch.zeros(n_groups))
        self.base_tau = base_tau
        
        # Per-group STCM (with subset of features)
        self.group_stcms = nn.ModuleList()
        for _ in range(n_groups):
            stcm = FuzzyPatternTM_STCM(
                n_features=self.features_per_group,
                n_clauses=n_clauses // n_groups,
                n_classes=n_classes,
                tau=base_tau,
            )
            self.group_stcms.append(stcm)
        
        # Combine group outputs
        self.combiner = nn.Linear(n_classes * n_groups, n_classes)
    
    def get_group_taus(self) -> torch.Tensor:
        """Get current tau values for each group."""
        return torch.sigmoid(self.group_tau_logits)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with per-group processing.
        
        Returns:
            (logits, concatenated_clauses)
        """
        batch_size = x.shape[0]
        group_taus = self.get_group_taus()
        
        group_logits = []
        group_clauses = []
        
        for i, stcm in enumerate(self.group_stcms):
            # Extract feature group
            start_idx = i * self.features_per_group
            end_idx = start_idx + self.features_per_group
            x_group = x[:, start_idx:end_idx]
            
            # Update tau for this group
            stcm.tau = float(group_taus[i])
            
            logits, clauses = stcm(x_group)
            group_logits.append(logits)
            group_clauses.append(clauses)
        
        # Combine group outputs
        combined_logits = torch.cat(group_logits, dim=1)
        final_logits = self.combiner(combined_logits)
        
        concat_clauses = torch.cat(group_clauses, dim=1)
        
        return final_logits, concat_clauses




@dataclass
class SpatialTMScaleConfig:
    in_channels: int
    image_size: int
    patch_size: int
    n_clauses: int
    tau: float = 0.5
    operator: str = "capacity"
    tm_kwargs: Optional[Dict[str, Any]] = None


class PatchwiseTMBlock(nn.Module):
    """TM block that preserves spatial structure via patchifying inputs."""

    def __init__(
        self,
        config: SpatialTMScaleConfig,
    ) -> None:
        super().__init__()
        if config.image_size % config.patch_size != 0:
            raise ValueError(
                f"image_size {config.image_size} must be divisible by patch_size {config.patch_size}."
            )
        self.config = config
        self.patch_dim = config.in_channels * config.patch_size * config.patch_size
        self.h = config.image_size // config.patch_size
        self.w = config.image_size // config.patch_size
        tm_kwargs = dict(config.tm_kwargs or {})
        self.tm = FuzzyPatternTM_STCM(
            n_features=self.patch_dim,
            n_clauses=config.n_clauses,
            n_classes=config.n_clauses,
            tau=config.tau,
            operator=config.operator,
            **tm_kwargs,
        )
        self.unfold = nn.Unfold(kernel_size=config.patch_size, stride=config.patch_size)

    def forward(self, x: torch.Tensor, use_ste: bool = True) -> torch.Tensor:
        patches = self.unfold(x).transpose(1, 2)  # [B, N, patch_dim]
        B, N, _ = patches.shape
        flat = patches.reshape(B * N, self.patch_dim)
        _, clause_outputs = self.tm(flat, use_ste=use_ste)
        clause_outputs = clause_outputs.view(B, N, self.config.n_clauses)
        clause_outputs = clause_outputs.permute(0, 2, 1).contiguous()
        clause_maps = clause_outputs.view(B, self.config.n_clauses, self.h, self.w)
        return clause_maps


class SpatialCrossScaleFusion(nn.Module):
    """Aligns and fuses clause maps from multiple scales via attention."""

    def __init__(self, clause_dims: Sequence[int], num_heads: int = 4, proj_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_scales = len(clause_dims)
        self.proj_dim = proj_dim
        self.projections = nn.ModuleList([nn.Conv2d(dim, proj_dim, kernel_size=1) for dim in clause_dims])
        self.attn = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(proj_dim)
        self.output = nn.ModuleList([nn.Conv2d(proj_dim, dim, kernel_size=1) for dim in clause_dims])

    def forward(self, features: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if not features:
            return []
        bsz = features[0].shape[0]
        min_h = min(f.shape[2] for f in features)
        min_w = min(f.shape[3] for f in features)

        aligned = []
        shapes = []
        for feat, proj in zip(features, self.projections):
            shapes.append(feat.shape)
            if feat.shape[2:] != (min_h, min_w):
                feat = F.adaptive_avg_pool2d(feat, (min_h, min_w))
            feat = proj(feat)
            aligned.append(feat.flatten(2).transpose(1, 2))

        combined = torch.cat(aligned, dim=1)
        attended, _ = self.attn(combined, combined, combined)
        attended = self.norm(attended + combined)

        splits = torch.split(attended, [a.shape[1] for a in aligned], dim=1)
        outputs: List[torch.Tensor] = []
        for split, out_proj, (b, c, h, w) in zip(splits, self.output, shapes):
            split = split.transpose(1, 2).reshape(bsz, self.proj_dim, min_h, min_w)
            split = out_proj(split)
            if (h, w) != (min_h, min_w):
                split = F.interpolate(split, size=(h, w), mode="bilinear", align_corners=False)
            outputs.append(split)
        return outputs


class ResidualClauseDecision(nn.Module):
    """Aggregates per-scale clause maps into final logits with residual MLPs."""

    def __init__(self, clause_dims: Sequence[int], n_classes: int, hidden: Optional[int] = None) -> None:
        super().__init__()
        self.n_classes = n_classes
        total_dim = sum(clause_dims)
        hidden = hidden or max(64, total_dim // 2)
        self.proj = nn.Sequential(
            nn.Linear(total_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_classes),
        )
        self.scale_heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, n_classes)) for dim in clause_dims]
        )

    def forward(self, clause_maps: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        pooled = [cm.mean(dim=(2, 3)) for cm in clause_maps]
        concat = torch.cat(pooled, dim=1)
        logits = self.proj(concat)
        aux = [head(p) for head, p in zip(self.scale_heads, pooled)]
        return logits, aux


class SpatialTMEnsemble(nn.Module):
    """End-to-end spatial TM pipeline with cross-scale fusion and residual decisions."""

    def __init__(
        self,
        scale_configs: Sequence[SpatialTMScaleConfig],
        n_classes: int,
        use_cross_scale_attention: bool = True,
        attention_heads: int = 4,
        attention_dim: int = 256,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([PatchwiseTMBlock(cfg) for cfg in scale_configs])
        clause_dims = [cfg.n_clauses for cfg in scale_configs]
        self.cross_scale = SpatialCrossScaleFusion(clause_dims, num_heads=attention_heads, proj_dim=attention_dim) if use_cross_scale_attention else None
        self.decision = ResidualClauseDecision(clause_dims, n_classes)

    def forward(self, features: Sequence[torch.Tensor], use_ste: bool = True):
        if len(features) != len(self.blocks):
            raise ValueError(f"Expected {len(self.blocks)} feature maps, received {len(features)}")
        clause_maps = [block(feat, use_ste=use_ste) for block, feat in zip(self.blocks, features)]
        fused_maps = self.cross_scale(clause_maps) if self.cross_scale is not None else clause_maps
        logits, aux = self.decision(fused_maps)
        return logits, aux, fused_maps

