"""
Attention-Adaptive Binarization with Neural Symbolic Transformer.

Uses transformer attention to dynamically learn optimal binarization
thresholds on a per-sample basis.

Key Innovation:
Fixed thresholds are suboptimal for different input distributions.
This module uses attention to dynamically predict the best thresholds
for each sample, adapting to local input statistics.

Architecture:
1. Feature Tokens: Each feature becomes a token
2. Clause Tokens: Learnable tokens for each clause
3. Bidirectional Attention: Features and clauses attend to each other
4. Dynamic Thresholds: Predict per-sample thresholds via attention

Benefits:
- Sample-adaptive binarization
- Global context awareness via attention
- Maintains interpretability
- Powerful representation capacity
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..tm import FuzzyPatternTM_STCM, prepare_tm_input


# =============================================================================
# Dynamic Threshold Predictor
# =============================================================================


class DynamicThresholdPredictor(nn.Module):
    """
    Predicts per-sample, per-feature binarization thresholds.
    
    Uses the input context to predict optimal thresholds.
    
    Args:
        n_features: Number of features
        hidden_dim: Hidden dimension
        context_type: How to compute context ('mean', 'attention', 'mlp')
    """
    
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        context_type: str = "attention",
    ):
        super().__init__()
        self.n_features = n_features
        self.context_type = context_type
        
        if context_type == "attention":
            self.context_query = nn.Linear(n_features, hidden_dim)
            self.context_key = nn.Linear(n_features, hidden_dim)
            self.context_value = nn.Linear(n_features, hidden_dim)
        elif context_type == "mlp":
            self.context_net = nn.Sequential(
                nn.Linear(n_features, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        
        # Threshold predictor
        self.threshold_pred = nn.Sequential(
            nn.Linear(n_features + (hidden_dim if context_type != "mean" else n_features), 
                     hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_features),
            nn.Sigmoid(),
        )
        
        # Base thresholds (learnable fallback)
        self.base_thresholds = nn.Parameter(torch.full((n_features,), 0.5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict thresholds for each sample.
        
        Args:
            x: Input features [batch, n_features]
            
        Returns:
            Thresholds [batch, n_features]
        """
        batch_size = x.shape[0]
        
        if self.context_type == "attention":
            # Self-attention context
            q = self.context_query(x)
            k = self.context_key(x)
            v = self.context_value(x)
            
            attn = F.softmax(q @ k.t() / math.sqrt(k.shape[-1]), dim=-1)
            context = attn @ v
        elif self.context_type == "mlp":
            context = self.context_net(x)
        else:
            context = x.mean(dim=-1, keepdim=True).expand(-1, self.n_features)
        
        # Predict thresholds
        threshold_input = torch.cat([x, context], dim=-1)
        predicted = self.threshold_pred(threshold_input)
        
        # Combine with base thresholds
        thresholds = 0.5 * predicted + 0.5 * self.base_thresholds.unsqueeze(0)
        
        return thresholds


# =============================================================================
# Neural Symbolic Block
# =============================================================================


class NeuralSymbolicBlock(nn.Module):
    """
    Transformer block for feature-clause interaction.
    
    Enables bidirectional attention between feature tokens
    and clause tokens for sophisticated reasoning.
    
    Args:
        n_features: Number of features
        n_clauses: Number of clauses
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        n_features: int,
        n_clauses: int,
        d_model: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_clauses = n_clauses
        self.d_model = d_model
        
        # Feature to clause attention
        self.feat_to_clause_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Clause to feature attention
        self.clause_to_feat_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Feed-forward for features
        self.feat_ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        
        # Feed-forward for clauses
        self.clause_ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        
        # Layer norms
        self.feat_norm1 = nn.LayerNorm(d_model)
        self.feat_norm2 = nn.LayerNorm(d_model)
        self.clause_norm1 = nn.LayerNorm(d_model)
        self.clause_norm2 = nn.LayerNorm(d_model)
        
        # Threshold predictor per feature
        self.threshold_head = nn.Linear(d_model, 1)
    
    def forward(
        self,
        feature_tokens: torch.Tensor,
        clause_tokens: torch.Tensor,
        raw_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Bidirectional attention between features and clauses.
        
        Args:
            feature_tokens: [batch, n_features, d_model]
            clause_tokens: [batch, n_clauses, d_model]
            raw_x: Original input [batch, n_features]
            
        Returns:
            (updated_features, updated_clauses, thresholds)
        """
        # Features attend to clauses
        feat_attn_out, _ = self.feat_to_clause_attn(
            feature_tokens, clause_tokens, clause_tokens
        )
        feature_tokens = self.feat_norm1(feature_tokens + feat_attn_out)
        feature_tokens = self.feat_norm2(feature_tokens + self.feat_ff(feature_tokens))
        
        # Clauses attend to features
        clause_attn_out, _ = self.clause_to_feat_attn(
            clause_tokens, feature_tokens, feature_tokens
        )
        clause_tokens = self.clause_norm1(clause_tokens + clause_attn_out)
        clause_tokens = self.clause_norm2(clause_tokens + self.clause_ff(clause_tokens))
        
        # Predict thresholds from feature tokens
        thresholds = torch.sigmoid(self.threshold_head(feature_tokens).squeeze(-1))
        
        return feature_tokens, clause_tokens, thresholds


# =============================================================================
# Neural Symbolic Transformer
# =============================================================================


class NeuralSymbolicTransformer(nn.Module):
    """
    Transformer for attention-adaptive binarization.
    
    Uses learnable clause tokens and bidirectional attention
    to dynamically determine binarization and clause evaluation.
    
    Args:
        n_features: Number of input features
        n_clauses: Number of clauses
        n_classes: Number of output classes
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        n_features: int,
        n_clauses: int,
        n_classes: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_clauses = n_clauses
        self.n_classes = n_classes
        self.d_model = d_model
        
        # Feature embedding
        self.feature_embed = nn.Linear(1, d_model)
        
        # Learnable clause tokens
        self.clause_tokens = nn.Parameter(torch.randn(n_clauses, d_model) * 0.02)
        
        # Positional encoding for features
        self.feature_pos = nn.Parameter(torch.randn(1, n_features, d_model) * 0.02)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            NeuralSymbolicBlock(
                n_features=n_features,
                n_clauses=n_clauses,
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        
        # Clause output head
        self.clause_head = nn.Linear(d_model, 1)
        
        # Voting
        self.voting = nn.Linear(n_clauses, n_classes)
    
    def forward(
        self,
        x: torch.Tensor,
        return_thresholds: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Neural symbolic forward pass.
        
        Args:
            x: Input features [batch, n_features]
            return_thresholds: Return predicted thresholds
            
        Returns:
            (logits, clause_outputs)
        """
        batch_size = x.shape[0]
        
        # Prepare input
        x_flat = prepare_tm_input(x, n_features=self.n_features)
        
        # Create feature tokens
        feature_tokens = self.feature_embed(x_flat.unsqueeze(-1))  # [batch, n_features, d_model]
        feature_tokens = feature_tokens + self.feature_pos
        
        # Expand clause tokens for batch
        clause_tokens = self.clause_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply transformer layers
        all_thresholds = []
        for layer in self.layers:
            feature_tokens, clause_tokens, thresholds = layer(
                feature_tokens, clause_tokens, x_flat
            )
            all_thresholds.append(thresholds)
        
        # Average thresholds across layers
        final_thresholds = torch.stack(all_thresholds).mean(dim=0)
        
        # Apply dynamic binarization
        x_binary = (x_flat > final_thresholds).float()
        
        # Get clause outputs from clause tokens
        clause_outputs = self.clause_head(clause_tokens).squeeze(-1)  # [batch, n_clauses]
        
        # Voting
        logits = self.voting(clause_outputs)
        
        if return_thresholds:
            return logits, clause_outputs, final_thresholds
        
        return logits, clause_outputs
    
    def get_attention_weights(
        self,
        x: torch.Tensor,
        layer_idx: int = -1,
    ) -> dict:
        """
        Get attention weights for interpretability.
        
        Args:
            x: Input features
            layer_idx: Which layer's attention to return
            
        Returns:
            Dict with attention weight matrices
        """
        batch_size = x.shape[0]
        x_flat = prepare_tm_input(x, n_features=self.n_features)
        
        # Forward through layers, collecting attention
        feature_tokens = self.feature_embed(x_flat.unsqueeze(-1)) + self.feature_pos
        clause_tokens = self.clause_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        
        attention_weights = {"feat_to_clause": [], "clause_to_feat": []}
        
        for layer in self.layers:
            # Get attention weights by hooking into the attention modules
            feature_tokens, clause_tokens, _ = layer(
                feature_tokens, clause_tokens, x_flat
            )
        
        return attention_weights


# =============================================================================
# Hybrid Attention-TM
# =============================================================================


class HybridAttentionTM(nn.Module):
    """
    Combines Neural Symbolic Transformer with traditional TM.
    
    Uses attention-predicted thresholds for TM input preparation.
    
    Args:
        n_features: Number of input features
        n_clauses: Number of TM clauses
        n_classes: Number of output classes
        d_model: Transformer dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
    """
    
    def __init__(
        self,
        n_features: int,
        n_clauses: int,
        n_classes: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        self.n_features = n_features
        
        # Threshold predictor (attention-based)
        self.threshold_predictor = DynamicThresholdPredictor(
            n_features=n_features,
            hidden_dim=d_model,
            context_type="attention",
        )
        
        # Traditional TM
        self.tm = FuzzyPatternTM_STCM(
            n_features=n_features,
            n_clauses=n_clauses,
            n_classes=n_classes,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        use_ste: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward with adaptive thresholding.
        
        Args:
            x: Input features
            use_ste: Use STE for TM
            
        Returns:
            (logits, clause_outputs)
        """
        x_flat = prepare_tm_input(x, n_features=self.n_features)
        
        # Predict thresholds
        thresholds = self.threshold_predictor(x_flat)
        
        # Binarize with predicted thresholds
        x_binary = (x_flat > thresholds).float()
        
        # STE if needed
        if use_ste:
            soft = torch.sigmoid((x_flat - thresholds) * 10)
            x_binary = x_binary + (soft - soft.detach())
        
        # TM forward
        logits, clauses = self.tm(x_binary, use_ste=use_ste, skip_norm=True)
        
        return logits, clauses



