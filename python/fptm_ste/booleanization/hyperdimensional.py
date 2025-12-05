"""
Hyperdimensional Clause Machine.

Encodes continuous values into high-dimensional binary vectors that
preserve similarity relationships.

Key Innovation:
Standard binarization loses similarity: 0.49 and 0.51 become completely
different (0 vs 1). Hyperdimensional computing encodes values into
high-dimensional vectors where similar values have high cosine similarity.

Architecture:
1. Level Encodings: Create HD vectors for different quantization levels
2. Thermometer Encoding: Blend levels based on continuous value
3. HD Clause: Compare encoded input with learned clause prototypes
4. Similarity Voting: Class prediction via similarity matching

Benefits:
- Similar inputs have similar HD representations
- Binary operations are efficient
- Maintains approximate distances
- Robust to noise
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..tm import prepare_tm_input


# =============================================================================
# Level Hypervector Encoder
# =============================================================================


class LevelHVEncoder(nn.Module):
    """
    Creates level hypervectors for HD encoding.
    
    Each quantization level has a corresponding high-dimensional
    binary vector, with nearby levels having similar vectors.
    
    Args:
        n_levels: Number of quantization levels
        hd_dim: Hyperdimensional vector dimension
        similarity_decay: How quickly similarity drops between levels
    """
    
    def __init__(
        self,
        n_levels: int = 100,
        hd_dim: int = 1000,
        similarity_decay: float = 0.05,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.hd_dim = hd_dim
        self.similarity_decay = similarity_decay
        
        # Create level encodings with gradual similarity
        level_encodings = self._create_level_encodings()
        self.register_buffer("level_encodings", level_encodings)
    
    def _create_level_encodings(self) -> torch.Tensor:
        """
        Create level encodings with gradual similarity.
        
        Uses a flip-based approach: start with random vector,
        flip a small fraction of bits for each subsequent level.
        """
        # Start with random bipolar vector
        base = (torch.rand(self.hd_dim) > 0.5).float() * 2 - 1
        
        encodings = [base.clone()]
        current = base.clone()
        
        # Flip rate based on similarity decay
        flip_rate = self.similarity_decay
        
        for i in range(1, self.n_levels):
            # Flip some bits
            flip_mask = torch.rand(self.hd_dim) < flip_rate
            current = current.clone()
            current[flip_mask] = -current[flip_mask]
            encodings.append(current.clone())
        
        return torch.stack(encodings)  # [n_levels, hd_dim]
    
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """
        Encode continuous values [0, 1] into HD vectors.
        
        Uses interpolation between level encodings for smooth encoding.
        
        Args:
            values: Continuous values in [0, 1] [batch, n_features]
            
        Returns:
            HD encodings [batch, n_features, hd_dim]
        """
        batch_size, n_features = values.shape
        
        # Map [0, 1] to level indices
        level_float = values * (self.n_levels - 1)
        level_low = level_float.floor().long().clamp(0, self.n_levels - 2)
        level_high = (level_low + 1).clamp(max=self.n_levels - 1)
        
        # Interpolation weight
        alpha = (level_float - level_low.float()).unsqueeze(-1)
        
        # Get level encodings
        # level_encodings: [n_levels, hd_dim]
        enc_low = self.level_encodings[level_low]  # [batch, n_features, hd_dim]
        enc_high = self.level_encodings[level_high]
        
        # Interpolate (in bipolar space, average then sign)
        interpolated = (1 - alpha) * enc_low + alpha * enc_high
        
        return interpolated


# =============================================================================
# Feature HD Encoder
# =============================================================================


class HDEncoder(nn.Module):
    """
    Full HD encoder for input features.
    
    Combines level encoding with feature position encoding.
    
    Args:
        n_features: Number of input features
        hd_dim: Hyperdimensional vector dimension
        n_levels: Number of quantization levels
    """
    
    def __init__(
        self,
        n_features: int,
        hd_dim: int = 1000,
        n_levels: int = 100,
    ):
        super().__init__()
        self.n_features = n_features
        self.hd_dim = hd_dim
        
        # Level encoder
        self.level_encoder = LevelHVEncoder(
            n_levels=n_levels,
            hd_dim=hd_dim,
        )
        
        # Position (feature) encodings
        # Each feature has a unique random vector for binding
        position_encodings = (torch.rand(n_features, hd_dim) > 0.5).float() * 2 - 1
        self.register_buffer("position_encodings", position_encodings)
    
    def forward(
        self,
        x: torch.Tensor,
        aggregate: bool = True,
    ) -> torch.Tensor:
        """
        Encode input features into HD space.
        
        Args:
            x: Input features [batch, n_features] in [0, 1]
            aggregate: Whether to aggregate features into single vector
            
        Returns:
            HD encoding [batch, hd_dim] if aggregate else [batch, n_features, hd_dim]
        """
        # Get level encodings
        level_enc = self.level_encoder(x)  # [batch, n_features, hd_dim]
        
        # Bind with position encodings (element-wise multiply in bipolar)
        # position_encodings: [n_features, hd_dim]
        bound = level_enc * self.position_encodings.unsqueeze(0)
        
        if aggregate:
            # Bundle: sum and sign
            bundled = bound.sum(dim=1)  # [batch, hd_dim]
            return torch.sign(bundled)
        
        return bound


# =============================================================================
# Hyperdimensional Clause Machine
# =============================================================================


class HyperdimensionalClauseMachine(nn.Module):
    """
    Clause Machine using Hyperdimensional Computing.
    
    Encodes inputs into HD space and uses cosine similarity
    with learned clause prototypes for classification.
    
    Args:
        n_features: Number of input features
        n_clauses: Number of clause prototypes
        n_classes: Number of output classes
        hd_dim: Hyperdimensional vector dimension
        n_levels: Number of quantization levels
        temperature: Temperature for similarity softmax
    """
    
    def __init__(
        self,
        n_features: int,
        n_clauses: int,
        n_classes: int,
        hd_dim: int = 1000,
        n_levels: int = 100,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_clauses = n_clauses
        self.n_classes = n_classes
        self.hd_dim = hd_dim
        self.temperature = temperature
        
        # HD encoder
        self.encoder = HDEncoder(
            n_features=n_features,
            hd_dim=hd_dim,
            n_levels=n_levels,
        )
        
        # Learnable clause prototypes in HD space
        # Initialize as random bipolar vectors
        half = n_clauses // 2
        self.pos_prototypes = nn.Parameter(
            (torch.rand(half, hd_dim) > 0.5).float() * 2 - 1
        )
        self.neg_prototypes = nn.Parameter(
            (torch.rand(half, hd_dim) > 0.5).float() * 2 - 1
        )
        
        # Voting weights
        self.voting = nn.Parameter(torch.randn(n_clauses, n_classes) * 0.1)
    
    def _clause_similarity(
        self,
        hd_input: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity between input and clause prototypes.
        
        Args:
            hd_input: Encoded input [batch, hd_dim]
            prototypes: Clause prototypes [half, hd_dim]
            
        Returns:
            Similarity scores [batch, half] in [-1, 1]
        """
        # Normalize
        input_norm = F.normalize(hd_input, dim=-1)
        proto_norm = F.normalize(prototypes, dim=-1)
        
        # Cosine similarity
        similarity = input_norm @ proto_norm.t()  # [batch, half]
        
        return similarity
    
    def forward(
        self,
        x: torch.Tensor,
        return_hd: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        HD clause machine forward pass.
        
        Args:
            x: Input features [batch, n_features]
            return_hd: Also return HD encoding
            
        Returns:
            (logits, clause_activations)
        """
        # Prepare input
        x_flat = prepare_tm_input(x, n_features=self.n_features)
        
        # Encode to HD
        hd_input = self.encoder(x_flat, aggregate=True)  # [batch, hd_dim]
        
        # Compute similarities with clause prototypes
        pos_sim = self._clause_similarity(hd_input, self.pos_prototypes)
        neg_sim = self._clause_similarity(hd_input, self.neg_prototypes)
        
        # Combine (positive clauses contribute positively, etc.)
        clause_activations = torch.cat([pos_sim, -neg_sim], dim=1)
        
        # Voting
        logits = clause_activations @ self.voting
        
        if return_hd:
            return logits, clause_activations, hd_input
        
        return logits, clause_activations
    
    def similarity_preserving_loss(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        margin: float = 0.1,
    ) -> torch.Tensor:
        """
        Auxiliary loss to ensure similar inputs have similar HD representations.
        
        Args:
            x1: First batch of inputs
            x2: Second batch of inputs (augmented version of x1)
            margin: Minimum similarity for positive pairs
            
        Returns:
            Contrastive loss
        """
        # Encode both
        hd1 = self.encoder(x1, aggregate=True)
        hd2 = self.encoder(x2, aggregate=True)
        
        # Normalize
        hd1_norm = F.normalize(hd1, dim=-1)
        hd2_norm = F.normalize(hd2, dim=-1)
        
        # Positive pairs: same index should be similar
        pos_sim = (hd1_norm * hd2_norm).sum(dim=-1)
        
        # Loss: encourage high similarity
        loss = F.relu(margin - pos_sim).mean()
        
        return loss



