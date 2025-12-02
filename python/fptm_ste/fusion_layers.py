"""
TM-Attention Fusion Layers.

This module implements hybrid layers that combine attention mechanisms with
TM clause processing, enabling the best of both neural attention and
interpretable TM patterns.

Key innovations:
1. Interleaved attention-TM blocks
2. Gated residual connections
3. Cross-modal attention between features and clauses
4. Adaptive fusion weights
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tm import FuzzyPatternTM_STCM


class SelfAttentionBlock(nn.Module):
    """
    Standard self-attention block for feature processing.
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, dim]
        Returns:
            [batch, seq_len, dim]
        """
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class TMBlock(nn.Module):
    """
    TM processing block that takes feature vectors and returns
    clause-based representations.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_clauses: int = 64,
        operator: str = "capacity",
        tau: float = 0.5,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, input_dim)
        
        self.stcm = FuzzyPatternTM_STCM(
            n_features=input_dim,
            n_clauses=n_clauses,
            n_classes=output_dim,
            operator=operator,
            tau=tau,
        )
        
        self.output_proj = nn.Linear(output_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, input_dim] or [batch, seq_len, input_dim]
        Returns:
            (output, clause_outputs)
        """
        if x.dim() == 3:
            # Process each position independently
            B, N, C = x.shape
            x_flat = x.reshape(B * N, C)
            x_proj = self.input_proj(x_flat)
            logits, clauses = self.stcm(x_proj)
            logits = logits.reshape(B, N, -1)
            clauses = clauses.reshape(B, N, -1)
            output = self.output_proj(logits)
            return output, clauses
        else:
            x_proj = self.input_proj(x)
            logits, clauses = self.stcm(x_proj)
            output = self.output_proj(logits)
            return output, clauses


class GatedResidual(nn.Module):
    """
    Gated residual connection for combining attention and TM outputs.
    """
    
    def __init__(self, dim: int, gate_type: str = "sigmoid"):
        super().__init__()
        self.gate_type = gate_type
        
        if gate_type == "sigmoid":
            self.gate = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.Sigmoid(),
            )
        elif gate_type == "tanh":
            self.gate = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.Tanh(),
            )
        elif gate_type == "softmax":
            # Soft selection between two streams
            self.gate = nn.Linear(dim * 2, 2)
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")
    
    def forward(
        self,
        residual: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gated residual: residual + gate * update
        """
        combined = torch.cat([residual, update], dim=-1)
        
        if self.gate_type == "softmax":
            weights = F.softmax(self.gate(combined), dim=-1)
            return weights[..., 0:1] * residual + weights[..., 1:2] * update
        else:
            gate = self.gate(combined)
            return residual + gate * update


class TMAttentionFusion(nn.Module):
    """
    Fusion layer that combines self-attention with TM processing.
    
    Architecture:
        x -> SelfAttention -> Norm -> TM -> Gated Residual -> Output
    
    The attention captures global dependencies while TM provides
    interpretable pattern matching.
    
    Args:
        dim: Feature dimension
        n_heads: Attention heads
        n_clauses: Number of TM clauses
        dropout: Dropout rate
        fusion_type: 'parallel', 'sequential', or 'interleaved'
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        n_clauses: int = 64,
        dropout: float = 0.1,
        fusion_type: str = "sequential",
        operator: str = "capacity",
    ):
        super().__init__()
        self.fusion_type = fusion_type
        
        # Attention branch
        self.attn = SelfAttentionBlock(dim, n_heads, dropout)
        self.attn_norm = nn.LayerNorm(dim)
        
        # TM branch
        self.tm = TMBlock(dim, dim, n_clauses, operator)
        self.tm_norm = nn.LayerNorm(dim)
        
        # Fusion
        self.gate = GatedResidual(dim)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(dim)
    
    def forward(
        self,
        x: torch.Tensor,
        return_clauses: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: [batch, seq_len, dim]
            return_clauses: Whether to return clause outputs
            
        Returns:
            output or (output, clauses)
        """
        if self.fusion_type == "sequential":
            # Attention first, then TM
            attn_out = self.attn(x)
            x = self.attn_norm(x + attn_out)
            
            tm_out, clauses = self.tm(x)
            x = self.gate(x, tm_out)
            x = self.tm_norm(x)
        
        elif self.fusion_type == "parallel":
            # Attention and TM in parallel
            attn_out = self.attn(x)
            tm_out, clauses = self.tm(x)
            
            # Combine both
            combined = self.attn_norm(x + attn_out)
            combined = self.gate(combined, tm_out)
            x = self.tm_norm(combined)
        
        elif self.fusion_type == "interleaved":
            # Alternate attention and TM
            attn_out = self.attn(x)
            x = self.attn_norm(x + attn_out)
            
            tm_out, clauses = self.tm(x)
            x = self.tm_norm(x + 0.5 * tm_out)
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.ff_norm(x + ff_out)
        
        if return_clauses:
            return x, clauses
        return x


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention between features and clause representations.
    
    Features attend to clauses to gather pattern information,
    and clauses attend to features for context.
    """
    
    def __init__(
        self,
        feature_dim: int,
        clause_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        
        # Feature -> Clause attention
        self.f2c_q = nn.Linear(feature_dim, feature_dim)
        self.f2c_k = nn.Linear(clause_dim, feature_dim)
        self.f2c_v = nn.Linear(clause_dim, feature_dim)
        self.f2c_proj = nn.Linear(feature_dim, feature_dim)
        
        # Clause -> Feature attention
        self.c2f_q = nn.Linear(clause_dim, clause_dim)
        self.c2f_k = nn.Linear(feature_dim, clause_dim)
        self.c2f_v = nn.Linear(feature_dim, clause_dim)
        self.c2f_proj = nn.Linear(clause_dim, clause_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        features: torch.Tensor,
        clauses: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-modal attention.
        
        Args:
            features: [batch, n_features, feature_dim]
            clauses: [batch, n_clauses, clause_dim]
            
        Returns:
            (updated_features, updated_clauses)
        """
        # Features attend to clauses
        q = self.f2c_q(features)
        k = self.f2c_k(clauses)
        v = self.f2c_v(clauses)
        
        attn = torch.bmm(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        f_update = torch.bmm(attn, v)
        f_update = self.f2c_proj(f_update)
        
        # Clauses attend to features
        q = self.c2f_q(clauses)
        k = self.c2f_k(features)
        v = self.c2f_v(features)
        
        attn = torch.bmm(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        c_update = torch.bmm(attn, v)
        c_update = self.c2f_proj(c_update)
        
        return features + f_update, clauses + c_update


class AdaptiveFusionBlock(nn.Module):
    """
    Adaptively learns how to combine attention and TM outputs.
    
    Uses a learned fusion network to determine optimal combination
    based on input characteristics.
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        n_clauses: int = 64,
        hidden_dim: int = None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        
        # Attention and TM branches
        self.attn = SelfAttentionBlock(dim, n_heads)
        self.tm = TMBlock(dim, dim, n_clauses)
        
        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(dim * 3, hidden_dim),  # input, attn_out, tm_out
            nn.ReLU(),
            nn.Linear(hidden_dim, dim * 2),
        )
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adaptive fusion.
        
        Args:
            x: [batch, seq_len, dim]
            
        Returns:
            [batch, seq_len, dim]
        """
        # Run both branches
        attn_out = self.attn(x)
        tm_out, _ = self.tm(x)
        
        # Compute fusion weights
        combined = torch.cat([x, attn_out, tm_out], dim=-1)
        fusion = self.fusion_net(combined)
        gate, value = fusion.chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        
        # Fused output
        fused = gate * attn_out + (1 - gate) * tm_out
        
        return self.norm(x + fused)


class DeepTMAttentionNetwork(nn.Module):
    """
    Deep network stacking TM-Attention fusion layers.
    
    Creates a full architecture by stacking multiple fusion blocks
    with optional skip connections.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_classes: int,
        n_layers: int = 4,
        n_heads: int = 4,
        n_clauses: int = 64,
        dropout: float = 0.1,
        fusion_type: str = "sequential",
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Fusion layers
        self.layers = nn.ModuleList([
            TMAttentionFusion(
                dim=hidden_dim,
                n_heads=n_heads,
                n_clauses=n_clauses,
                dropout=dropout,
                fusion_type=fusion_type,
            )
            for _ in range(n_layers)
        ])
        
        # Global pooling options
        self.pool_type = "mean"  # 'mean', 'max', 'attention'
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: [batch, seq_len, input_dim] or [batch, input_dim]
            return_features: Whether to return intermediate features
            
        Returns:
            logits or (logits, features)
        """
        # Handle 2D input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, input_dim]
        
        # Project to hidden dim
        x = self.input_proj(x)
        
        # Apply fusion layers
        for layer in self.layers:
            x = layer(x)
        
        # Global pooling
        if self.pool_type == "mean":
            features = x.mean(dim=1)
        elif self.pool_type == "max":
            features = x.max(dim=1).values
        else:
            features = x.mean(dim=1)
        
        # Classification
        logits = self.head(features)
        
        if return_features:
            return logits, features
        return logits


class HybridVisionTM(nn.Module):
    """
    Hybrid Vision-TM architecture for image classification.
    
    Uses a CNN or ViT backbone for feature extraction, then
    applies TM-Attention fusion for final classification.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        n_classes: int,
        n_clauses: int = 128,
        n_fusion_layers: int = 2,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        
        # Feature projection
        self.proj = nn.Linear(feature_dim, feature_dim)
        
        # TM-Attention fusion
        self.fusion = nn.Sequential(*[
            TMAttentionFusion(
                dim=feature_dim,
                n_clauses=n_clauses,
            )
            for _ in range(n_fusion_layers)
        ])
        
        # Head
        self.head = nn.Linear(feature_dim, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [batch, C, H, W]
            
        Returns:
            Class logits [batch, n_classes]
        """
        # Extract features
        features = self.backbone(x)
        
        # Handle different feature shapes
        if features.dim() == 4:
            # CNN features [batch, C, H, W] -> [batch, H*W, C]
            B, C, H, W = features.shape
            features = features.flatten(2).transpose(1, 2)
        elif features.dim() == 2:
            # Already pooled [batch, dim]
            features = features.unsqueeze(1)
        
        # Project
        features = self.proj(features)
        
        # Fusion
        for layer in self.fusion:
            features = layer(features)
        
        # Pool and classify
        pooled = features.mean(dim=1)
        logits = self.head(pooled)
        
        return logits

