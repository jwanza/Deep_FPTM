"""
Hierarchical Clause Attention (HCA) module.

This module enables clauses to attend to each other before voting, creating
"clause reasoning chains" that can capture complex relationships between
pattern matches.

Key innovations:
1. Intra-polarity attention: Clauses within positive/negative banks interact
2. Cross-polarity attention: Positive and negative clauses influence each other
3. Global clause consensus: All clauses participate in final refinement
4. Clause gating: Learned gates control information flow
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClauseAttentionHead(nn.Module):
    """
    Single attention head for clause interactions.
    
    Computes attention between clause outputs to enable inter-clause reasoning.
    Uses scaled dot-product attention with optional relative position bias.
    """
    
    def __init__(
        self,
        clause_dim: int,
        head_dim: int = 32,
        dropout: float = 0.1,
        use_bias: bool = True,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        self.q_proj = nn.Linear(clause_dim, head_dim, bias=use_bias)
        self.k_proj = nn.Linear(clause_dim, head_dim, bias=use_bias)
        self.v_proj = nn.Linear(clause_dim, head_dim, bias=use_bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, n_query, clause_dim]
            key: [batch, n_key, clause_dim]
            value: [batch, n_value, clause_dim]
            mask: Optional attention mask
            
        Returns:
            Attended values [batch, n_query, head_dim]
        """
        q = self.q_proj(query) * self.scale  # [batch, n_query, head_dim]
        k = self.k_proj(key)                  # [batch, n_key, head_dim]
        v = self.v_proj(value)                # [batch, n_value, head_dim]
        
        # Compute attention scores
        attn = torch.bmm(q, k.transpose(-2, -1))  # [batch, n_query, n_key]
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.bmm(attn, v)  # [batch, n_query, head_dim]
        return out


class MultiHeadClauseAttention(nn.Module):
    """
    Multi-head attention over clause outputs.
    
    Enables clauses to exchange information through multiple attention heads,
    each potentially capturing different types of relationships.
    """
    
    def __init__(
        self,
        clause_dim: int,
        n_heads: int = 4,
        head_dim: int = 32,
        dropout: float = 0.1,
        use_bias: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.embed_dim = n_heads * head_dim
        
        self.q_proj = nn.Linear(clause_dim, self.embed_dim, bias=use_bias)
        self.k_proj = nn.Linear(clause_dim, self.embed_dim, bias=use_bias)
        self.v_proj = nn.Linear(clause_dim, self.embed_dim, bias=use_bias)
        self.out_proj = nn.Linear(self.embed_dim, clause_dim, bias=use_bias)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = head_dim ** -0.5
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, n_clauses, clause_dim]
            key: Optional, defaults to query (self-attention)
            value: Optional, defaults to key
            mask: Optional attention mask
            
        Returns:
            Attended output [batch, n_clauses, clause_dim]
        """
        if key is None:
            key = query
        if value is None:
            value = key
        
        batch_size, n_query, _ = query.shape
        n_key = key.shape[1]
        
        # Project and reshape for multi-head attention
        q = self.q_proj(query).view(batch_size, n_query, self.n_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, n_key, self.n_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, n_key, self.n_heads, self.head_dim)
        
        # Transpose for attention: [batch, n_heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [batch, n_heads, n_query, head_dim]
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, n_query, self.embed_dim)
        out = self.out_proj(out)
        
        return out


class ClauseGate(nn.Module):
    """
    Gating mechanism for clause attention outputs.
    
    Controls how much the attention-refined clause representations
    should influence the final output.
    """
    
    def __init__(self, clause_dim: int, gate_type: str = "sigmoid"):
        super().__init__()
        self.gate_type = gate_type
        
        if gate_type == "sigmoid":
            self.gate = nn.Sequential(
                nn.Linear(clause_dim * 2, clause_dim),
                nn.Sigmoid(),
            )
        elif gate_type == "tanh":
            self.gate = nn.Sequential(
                nn.Linear(clause_dim * 2, clause_dim),
                nn.Tanh(),
            )
        elif gate_type == "glu":
            self.gate = nn.Linear(clause_dim * 2, clause_dim * 2)
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")
    
    def forward(self, original: torch.Tensor, refined: torch.Tensor) -> torch.Tensor:
        """
        Args:
            original: Original clause outputs [batch, n_clauses, clause_dim]
            refined: Attention-refined outputs [batch, n_clauses, clause_dim]
            
        Returns:
            Gated output [batch, n_clauses, clause_dim]
        """
        combined = torch.cat([original, refined], dim=-1)
        
        if self.gate_type == "glu":
            x = self.gate(combined)
            gate, value = x.chunk(2, dim=-1)
            return original + torch.sigmoid(gate) * value
        else:
            gate = self.gate(combined)
            return original + gate * (refined - original)


class IntraPolarityAttention(nn.Module):
    """
    Attention within same-polarity clause groups.
    
    Positive clauses attend to other positive clauses, and negative clauses
    attend to other negative clauses. This allows similar patterns to
    reinforce each other.
    """
    
    def __init__(
        self,
        clause_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = MultiHeadClauseAttention(
            clause_dim=clause_dim,
            n_heads=n_heads,
            head_dim=clause_dim // n_heads,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(clause_dim)
    
    def forward(
        self,
        pos_clauses: torch.Tensor,
        neg_clauses: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pos_clauses: [batch, n_pos, clause_dim]
            neg_clauses: [batch, n_neg, clause_dim]
            
        Returns:
            Refined (pos_clauses, neg_clauses)
        """
        # Self-attention within positive clauses
        pos_refined = self.attn(pos_clauses)
        pos_out = self.norm(pos_clauses + pos_refined)
        
        # Self-attention within negative clauses
        neg_refined = self.attn(neg_clauses)
        neg_out = self.norm(neg_clauses + neg_refined)
        
        return pos_out, neg_out


class CrossPolarityAttention(nn.Module):
    """
    Attention across clause polarities.
    
    Positive clauses can attend to negative clauses and vice versa,
    enabling competition and cooperation between pattern types.
    """
    
    def __init__(
        self,
        clause_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        
        self.pos_to_neg_attn = MultiHeadClauseAttention(
            clause_dim=clause_dim,
            n_heads=n_heads,
            head_dim=clause_dim // n_heads,
            dropout=dropout,
        )
        
        if bidirectional:
            self.neg_to_pos_attn = MultiHeadClauseAttention(
                clause_dim=clause_dim,
                n_heads=n_heads,
                head_dim=clause_dim // n_heads,
                dropout=dropout,
            )
        
        self.pos_norm = nn.LayerNorm(clause_dim)
        self.neg_norm = nn.LayerNorm(clause_dim)
    
    def forward(
        self,
        pos_clauses: torch.Tensor,
        neg_clauses: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pos_clauses: [batch, n_pos, clause_dim]
            neg_clauses: [batch, n_neg, clause_dim]
            
        Returns:
            Refined (pos_clauses, neg_clauses)
        """
        # Positive clauses attend to negative clauses
        pos_refined = self.pos_to_neg_attn(pos_clauses, neg_clauses, neg_clauses)
        pos_out = self.pos_norm(pos_clauses + pos_refined)
        
        if self.bidirectional:
            # Negative clauses attend to positive clauses
            neg_refined = self.neg_to_pos_attn(neg_clauses, pos_clauses, pos_clauses)
            neg_out = self.neg_norm(neg_clauses + neg_refined)
        else:
            neg_out = neg_clauses
        
        return pos_out, neg_out


class GlobalClauseConsensus(nn.Module):
    """
    Global attention over all clauses for final refinement.
    
    All clauses (positive and negative) participate in a global attention
    step to reach consensus on the final representation.
    """
    
    def __init__(
        self,
        clause_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.use_cls_token = use_cls_token
        
        self.attn = MultiHeadClauseAttention(
            clause_dim=clause_dim,
            n_heads=n_heads,
            head_dim=clause_dim // n_heads,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(clause_dim)
        
        if use_cls_token:
            # Learnable [CLS] token for global pooling
            self.cls_token = nn.Parameter(torch.randn(1, 1, clause_dim) * 0.02)
    
    def forward(
        self,
        pos_clauses: torch.Tensor,
        neg_clauses: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            pos_clauses: [batch, n_pos, clause_dim]
            neg_clauses: [batch, n_neg, clause_dim]
            
        Returns:
            Refined (pos_clauses, neg_clauses, cls_output)
        """
        batch_size = pos_clauses.shape[0]
        
        # Concatenate all clauses
        all_clauses = torch.cat([pos_clauses, neg_clauses], dim=1)
        
        if self.use_cls_token:
            cls_expanded = self.cls_token.expand(batch_size, -1, -1)
            all_clauses = torch.cat([cls_expanded, all_clauses], dim=1)
        
        # Global self-attention
        refined = self.attn(all_clauses)
        all_out = self.norm(all_clauses + refined)
        
        # Split back
        if self.use_cls_token:
            cls_out = all_out[:, 0]  # [batch, clause_dim]
            all_out = all_out[:, 1:]
        else:
            cls_out = None
        
        n_pos = pos_clauses.shape[1]
        pos_out = all_out[:, :n_pos]
        neg_out = all_out[:, n_pos:]
        
        return pos_out, neg_out, cls_out


class HierarchicalClauseAttention(nn.Module):
    """
    Full hierarchical clause attention module.
    
    Implements a three-stage attention hierarchy:
    1. Intra-polarity: Clauses within same polarity interact
    2. Cross-polarity: Positive and negative clauses exchange information
    3. Global consensus: All clauses participate in final refinement
    
    Each stage is optional and can be configured independently.
    
    Args:
        clause_dim: Dimension of clause representations
        n_heads: Number of attention heads
        dropout: Dropout rate
        stages: Which stages to use ('intra', 'cross', 'global')
        use_gates: Whether to use gating mechanisms
        use_cls_token: Whether to use [CLS] token in global stage
    """
    
    def __init__(
        self,
        clause_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        stages: Tuple[str, ...] = ("intra", "cross", "global"),
        use_gates: bool = True,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.stages = stages
        self.use_gates = use_gates
        
        if "intra" in stages:
            self.intra_attn = IntraPolarityAttention(
                clause_dim=clause_dim,
                n_heads=n_heads,
                dropout=dropout,
            )
            if use_gates:
                self.intra_gate = ClauseGate(clause_dim)
        
        if "cross" in stages:
            self.cross_attn = CrossPolarityAttention(
                clause_dim=clause_dim,
                n_heads=n_heads,
                dropout=dropout,
            )
            if use_gates:
                self.cross_gate = ClauseGate(clause_dim)
        
        if "global" in stages:
            self.global_attn = GlobalClauseConsensus(
                clause_dim=clause_dim,
                n_heads=n_heads,
                dropout=dropout,
                use_cls_token=use_cls_token,
            )
            if use_gates:
                self.global_gate = ClauseGate(clause_dim)
    
    def forward(
        self,
        pos_clauses: torch.Tensor,
        neg_clauses: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply hierarchical attention to clause outputs.
        
        Args:
            pos_clauses: [batch, n_pos, clause_dim] or [batch, n_pos]
            neg_clauses: [batch, n_neg, clause_dim] or [batch, n_neg]
            
        Returns:
            (refined_pos, refined_neg, cls_output)
        """
        # Ensure 3D tensors
        if pos_clauses.dim() == 2:
            pos_clauses = pos_clauses.unsqueeze(-1)
            neg_clauses = neg_clauses.unsqueeze(-1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        cls_out = None
        
        # Stage 1: Intra-polarity attention
        if "intra" in self.stages:
            pos_orig, neg_orig = pos_clauses, neg_clauses
            pos_clauses, neg_clauses = self.intra_attn(pos_clauses, neg_clauses)
            
            if self.use_gates:
                pos_clauses = self.intra_gate(pos_orig, pos_clauses)
                neg_clauses = self.intra_gate(neg_orig, neg_clauses)
        
        # Stage 2: Cross-polarity attention
        if "cross" in self.stages:
            pos_orig, neg_orig = pos_clauses, neg_clauses
            pos_clauses, neg_clauses = self.cross_attn(pos_clauses, neg_clauses)
            
            if self.use_gates:
                pos_clauses = self.cross_gate(pos_orig, pos_clauses)
                neg_clauses = self.cross_gate(neg_orig, neg_clauses)
        
        # Stage 3: Global consensus
        if "global" in self.stages:
            pos_orig, neg_orig = pos_clauses, neg_clauses
            pos_clauses, neg_clauses, cls_out = self.global_attn(pos_clauses, neg_clauses)
            
            if self.use_gates:
                pos_clauses = self.global_gate(pos_orig, pos_clauses)
                neg_clauses = self.global_gate(neg_orig, neg_clauses)
        
        if squeeze_output:
            pos_clauses = pos_clauses.squeeze(-1)
            neg_clauses = neg_clauses.squeeze(-1)
        
        return pos_clauses, neg_clauses, cls_out


class ClauseTransformerBlock(nn.Module):
    """
    Transformer block for clause processing.
    
    Combines multi-head attention with feed-forward network
    for more expressive clause transformations.
    """
    
    def __init__(
        self,
        clause_dim: int,
        n_heads: int = 4,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        
        if ff_dim is None:
            ff_dim = clause_dim * 4
        
        self.attn = MultiHeadClauseAttention(
            clause_dim=clause_dim,
            n_heads=n_heads,
            head_dim=clause_dim // n_heads,
            dropout=dropout,
        )
        self.attn_norm = nn.LayerNorm(clause_dim)
        
        activation_fn = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ff = nn.Sequential(
            nn.Linear(clause_dim, ff_dim),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(ff_dim, clause_dim),
            nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(clause_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, n_clauses, clause_dim]
            
        Returns:
            Transformed clauses [batch, n_clauses, clause_dim]
        """
        # Self-attention with residual
        attn_out = self.attn(x)
        x = self.attn_norm(x + attn_out)
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.ff_norm(x + ff_out)
        
        return x


class ClauseReasoningNetwork(nn.Module):
    """
    Deep clause reasoning network using stacked transformer blocks.
    
    Enables deep multi-step reasoning over clause patterns through
    multiple transformer layers.
    """
    
    def __init__(
        self,
        clause_dim: int,
        n_layers: int = 2,
        n_heads: int = 4,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_polarity_embedding: bool = True,
    ):
        super().__init__()
        self.use_polarity_embedding = use_polarity_embedding
        
        if use_polarity_embedding:
            self.polarity_embedding = nn.Embedding(2, clause_dim)
        
        self.blocks = nn.ModuleList([
            ClauseTransformerBlock(
                clause_dim=clause_dim,
                n_heads=n_heads,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.out_proj = nn.Linear(clause_dim, clause_dim)
    
    def forward(
        self,
        pos_clauses: torch.Tensor,
        neg_clauses: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply deep reasoning over clauses.
        
        Args:
            pos_clauses: [batch, n_pos, clause_dim]
            neg_clauses: [batch, n_neg, clause_dim]
            
        Returns:
            (refined_pos, refined_neg)
        """
        batch_size = pos_clauses.shape[0]
        n_pos = pos_clauses.shape[1]
        
        # Concatenate clauses
        x = torch.cat([pos_clauses, neg_clauses], dim=1)
        
        # Add polarity embeddings
        if self.use_polarity_embedding:
            polarity_ids = torch.cat([
                torch.zeros(n_pos, dtype=torch.long, device=x.device),
                torch.ones(neg_clauses.shape[1], dtype=torch.long, device=x.device),
            ])
            polarity_ids = polarity_ids.unsqueeze(0).expand(batch_size, -1)
            polarity_emb = self.polarity_embedding(polarity_ids)
            x = x + polarity_emb
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Output projection
        x = self.out_proj(x)
        
        # Split back
        pos_out = x[:, :n_pos]
        neg_out = x[:, n_pos:]
        
        return pos_out, neg_out


# =============================================================================
# Enhanced Positional Encodings
# =============================================================================


class ClausePositionalEncoding(nn.Module):
    """
    Learnable positional encoding for clause sequences.
    
    Provides unique position embeddings for each clause position,
    allowing the attention mechanism to be position-aware.
    
    Supports multiple encoding types:
    - 'learned': Fully learnable position embeddings
    - 'sinusoidal': Fixed sinusoidal encoding (non-learnable)
    - 'rotary': Rotary position embeddings (RoPE)
    
    Args:
        max_clauses: Maximum number of clauses
        embed_dim: Embedding dimension
        encoding_type: Type of positional encoding
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        max_clauses: int,
        embed_dim: int,
        encoding_type: str = "learned",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_clauses = max_clauses
        self.embed_dim = embed_dim
        self.encoding_type = encoding_type
        
        if encoding_type == "learned":
            self.position_embedding = nn.Embedding(max_clauses, embed_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0, std=0.02)
        elif encoding_type == "sinusoidal":
            # Create fixed sinusoidal embeddings
            position = torch.arange(max_clauses).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
            )
            pe = torch.zeros(max_clauses, embed_dim)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("position_embedding", pe)
        elif encoding_type == "rotary":
            # Rotary embedding parameters
            inv_freq = 1.0 / (10000 ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
            self.register_buffer("inv_freq", inv_freq)
        else:
            raise ValueError(f"Unknown encoding_type: {encoding_type}")
        
        self.dropout = nn.Dropout(dropout)
    
    def _get_sinusoidal(self, seq_len: int) -> torch.Tensor:
        """Get sinusoidal embeddings for sequence length."""
        return self.position_embedding[:seq_len]
    
    def _get_learned(self, seq_len: int) -> torch.Tensor:
        """Get learned embeddings for sequence length."""
        positions = torch.arange(seq_len, device=self.position_embedding.weight.device)
        return self.position_embedding(positions)
    
    def _get_rotary(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get rotary embedding components (cos, sin)."""
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()
    
    def forward(
        self,
        x: torch.Tensor,
        return_rotary: bool = False,
    ):
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            return_rotary: For rotary, return (cos, sin) instead of adding
            
        Returns:
            Position-encoded tensor or rotary components
        """
        seq_len = x.shape[1]
        
        if self.encoding_type == "rotary":
            cos, sin = self._get_rotary(seq_len, x.device)
            if return_rotary:
                return cos, sin
            # Apply rotary by rotation
            return self._apply_rotary(x, cos, sin)
        elif self.encoding_type == "learned":
            pos_emb = self._get_learned(seq_len)
        else:
            pos_emb = self._get_sinusoidal(seq_len)
        
        # Add position embedding and apply dropout
        return self.dropout(x + pos_emb.unsqueeze(0))
    
    @staticmethod
    def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding to input."""
        # Split into pairs
        x1, x2 = x[..., ::2], x[..., 1::2]
        # Rotate
        rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
        # Apply rotation
        return x * cos + rotated * sin
    
    def extra_repr(self) -> str:
        return f"max_clauses={self.max_clauses}, embed_dim={self.embed_dim}, type='{self.encoding_type}'"


class RelativePositionBias(nn.Module):
    """
    Relative position bias for attention.
    
    Learns a bias term for each relative position difference,
    which is added to attention scores.
    
    Args:
        n_heads: Number of attention heads
        max_distance: Maximum relative distance to encode
        bidirectional: Whether to use bidirectional bias
    """
    
    def __init__(
        self,
        n_heads: int,
        max_distance: int = 128,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        
        # Number of relative positions
        if bidirectional:
            n_positions = 2 * max_distance + 1
        else:
            n_positions = max_distance + 1
        
        self.relative_bias = nn.Embedding(n_positions, n_heads)
        nn.init.normal_(self.relative_bias.weight, std=0.02)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Compute relative position bias matrix.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Bias matrix [1, n_heads, seq_len, seq_len]
        """
        device = self.relative_bias.weight.device
        
        # Create relative position indices
        positions = torch.arange(seq_len, device=device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # Clamp and shift to valid range
        relative_positions = relative_positions.clamp(-self.max_distance, self.max_distance)
        if self.bidirectional:
            relative_positions = relative_positions + self.max_distance
        else:
            relative_positions = relative_positions.abs()
        
        # Look up biases
        bias = self.relative_bias(relative_positions)  # [seq_len, seq_len, n_heads]
        bias = bias.permute(2, 0, 1).unsqueeze(0)  # [1, n_heads, seq_len, seq_len]
        
        return bias


class EnhancedMultiHeadClauseAttention(nn.Module):
    """
    Enhanced multi-head attention with positional encodings.
    
    Combines multi-head attention with various positional encoding
    schemes for better clause relationship modeling.
    
    Features:
    - Multiple positional encoding options
    - Relative position bias
    - Flash attention support (when available)
    - Gradient checkpointing option
    
    Args:
        clause_dim: Clause embedding dimension
        n_heads: Number of attention heads
        head_dim: Dimension per head
        dropout: Dropout rate
        positional_encoding: Type of positional encoding
        use_relative_bias: Whether to use relative position bias
        max_clauses: Maximum number of clauses
    """
    
    def __init__(
        self,
        clause_dim: int,
        n_heads: int = 4,
        head_dim: Optional[int] = None,
        dropout: float = 0.1,
        positional_encoding: Optional[str] = "learned",
        use_relative_bias: bool = True,
        max_clauses: int = 256,
    ):
        super().__init__()
        
        if head_dim is None:
            head_dim = clause_dim // n_heads
        
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.embed_dim = n_heads * head_dim
        self.scale = head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(clause_dim, self.embed_dim)
        self.k_proj = nn.Linear(clause_dim, self.embed_dim)
        self.v_proj = nn.Linear(clause_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, clause_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Positional encoding
        if positional_encoding is not None:
            self.pos_encoder = ClausePositionalEncoding(
                max_clauses=max_clauses,
                embed_dim=clause_dim,
                encoding_type=positional_encoding,
                dropout=dropout,
            )
        else:
            self.pos_encoder = None
        
        # Relative position bias
        if use_relative_bias:
            self.relative_bias = RelativePositionBias(
                n_heads=n_heads,
                max_distance=max_clauses,
            )
        else:
            self.relative_bias = None
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Enhanced attention forward pass.
        
        Args:
            query: [batch, n_query, clause_dim]
            key: Optional, defaults to query
            value: Optional, defaults to key
            mask: Optional attention mask
            
        Returns:
            Attended output [batch, n_query, clause_dim]
        """
        if key is None:
            key = query
        if value is None:
            value = key
        
        batch_size, n_query, _ = query.shape
        n_key = key.shape[1]
        
        # Apply positional encoding
        if self.pos_encoder is not None:
            query = self.pos_encoder(query)
            if key is not query:
                key = self.pos_encoder(key)
        
        # Project and reshape
        q = self.q_proj(query).view(batch_size, n_query, self.n_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, n_key, self.n_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, n_key, self.n_heads, self.head_dim)
        
        # Transpose for attention: [batch, n_heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Add relative position bias
        if self.relative_bias is not None:
            bias = self.relative_bias(n_key)
            attn = attn + bias
        
        # Apply mask
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply to values
        out = torch.matmul(attn, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, n_query, self.embed_dim)
        out = self.out_proj(out)
        
        return out


class CrossClauseReasoning(nn.Module):
    """
    Advanced cross-clause reasoning module.
    
    Enables sophisticated reasoning between clause groups through
    bidirectional cross-attention with gating mechanisms.
    
    Features:
    - Bidirectional cross-attention
    - Gated residual connections
    - Multi-step iterative refinement
    - Contrastive clause pairing
    
    Args:
        clause_dim: Clause embedding dimension
        n_heads: Number of attention heads
        n_iterations: Number of reasoning iterations
        dropout: Dropout rate
        use_contrastive: Use contrastive clause pairing
    """
    
    def __init__(
        self,
        clause_dim: int,
        n_heads: int = 4,
        n_iterations: int = 2,
        dropout: float = 0.1,
        use_contrastive: bool = False,
    ):
        super().__init__()
        self.n_iterations = n_iterations
        self.use_contrastive = use_contrastive
        
        # Cross-attention layers
        self.pos_to_neg_attn = nn.ModuleList([
            EnhancedMultiHeadClauseAttention(
                clause_dim=clause_dim,
                n_heads=n_heads,
                dropout=dropout,
                positional_encoding=None,  # No position encoding for cross-attention
                use_relative_bias=False,
            )
            for _ in range(n_iterations)
        ])
        
        self.neg_to_pos_attn = nn.ModuleList([
            EnhancedMultiHeadClauseAttention(
                clause_dim=clause_dim,
                n_heads=n_heads,
                dropout=dropout,
                positional_encoding=None,
                use_relative_bias=False,
            )
            for _ in range(n_iterations)
        ])
        
        # Gating
        self.pos_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(clause_dim * 2, clause_dim),
                nn.Sigmoid(),
            )
            for _ in range(n_iterations)
        ])
        
        self.neg_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(clause_dim * 2, clause_dim),
                nn.Sigmoid(),
            )
            for _ in range(n_iterations)
        ])
        
        # Layer norms
        self.pos_norms = nn.ModuleList([nn.LayerNorm(clause_dim) for _ in range(n_iterations)])
        self.neg_norms = nn.ModuleList([nn.LayerNorm(clause_dim) for _ in range(n_iterations)])
        
        # Contrastive projections
        if use_contrastive:
            self.contrast_proj = nn.Linear(clause_dim, clause_dim)
    
    def forward(
        self,
        pos_clauses: torch.Tensor,
        neg_clauses: torch.Tensor,
        return_contrast_loss: bool = False,
    ):
        """
        Apply cross-clause reasoning.
        
        Args:
            pos_clauses: [batch, n_pos, clause_dim]
            neg_clauses: [batch, n_neg, clause_dim]
            return_contrast_loss: Return contrastive loss
            
        Returns:
            (refined_pos, refined_neg) or (refined_pos, refined_neg, contrast_loss)
        """
        for i in range(self.n_iterations):
            # Cross-attention: pos attends to neg
            pos_from_neg = self.pos_to_neg_attn[i](pos_clauses, neg_clauses, neg_clauses)
            
            # Cross-attention: neg attends to pos
            neg_from_pos = self.neg_to_pos_attn[i](neg_clauses, pos_clauses, pos_clauses)
            
            # Gated residual for positive clauses
            pos_gate = self.pos_gates[i](torch.cat([pos_clauses, pos_from_neg], dim=-1))
            pos_clauses = self.pos_norms[i](pos_clauses + pos_gate * pos_from_neg)
            
            # Gated residual for negative clauses
            neg_gate = self.neg_gates[i](torch.cat([neg_clauses, neg_from_pos], dim=-1))
            neg_clauses = self.neg_norms[i](neg_clauses + neg_gate * neg_from_pos)
        
        if return_contrast_loss and self.use_contrastive:
            contrast_loss = self._contrastive_loss(pos_clauses, neg_clauses)
            return pos_clauses, neg_clauses, contrast_loss
        
        return pos_clauses, neg_clauses
    
    def _contrastive_loss(
        self,
        pos_clauses: torch.Tensor,
        neg_clauses: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute contrastive loss between clause polarities.
        
        Encourages positive clauses to be dissimilar from negative clauses.
        """
        # Project to contrastive space
        pos_proj = F.normalize(self.contrast_proj(pos_clauses.mean(dim=1)), dim=-1)
        neg_proj = F.normalize(self.contrast_proj(neg_clauses.mean(dim=1)), dim=-1)
        
        # Similarity matrix
        sim = torch.matmul(pos_proj, neg_proj.t()) / temperature
        
        # Negative pairs should have low similarity
        return sim.mean()

