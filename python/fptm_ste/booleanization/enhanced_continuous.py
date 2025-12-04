"""
Enhanced Continuous Feature Handling for Tsetlin Machines.

This module implements advanced techniques to preserve continuous information
while maintaining TM's interpretability.

Key Innovations:
1. Multi-Scale Thermometer Encoding: Dense threshold encoding at multiple scales
2. Gaussian Basis Expansion: Project features onto Gaussian basis functions
3. Learnable Feature Binning: End-to-end learned discretization
4. Residual Continuous Path: Bypass with continuous features added back
5. Positional Value Encoding: Encode position in [0,1] range like positional embeddings
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Multi-Scale Thermometer Encoding
# =============================================================================


class MultiScaleThermometer(nn.Module):
    """
    Dense multi-scale thermometer encoding that preserves ordering.
    
    Instead of single threshold (x > 0.5 → 1), use many thresholds:
    [x > 0.1, x > 0.2, ..., x > 0.9] = thermometer code
    
    Multi-scale: use coarse (0.25, 0.5, 0.75) and fine (0.1, 0.2, ..., 0.9)
    scales simultaneously.
    
    This preserves:
    - Ordering: higher values → more 1s
    - Similarity: close values → similar codes
    - Differentiability via soft thresholding
    
    Args:
        n_features: Number of input features
        scales: List of (n_thresholds, scale_weight) for each scale
        temperature: Soft threshold temperature
        learnable: Whether thresholds are learnable
    """
    
    def __init__(
        self,
        n_features: int,
        scales: List[Tuple[int, float]] = [(4, 0.5), (8, 1.0), (16, 0.5)],
        temperature: float = 0.1,
        learnable: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.scales = scales
        self.temperature = temperature
        
        # Total output dimension
        total_thresholds = sum(n_thresh for n_thresh, _ in scales)
        self.output_dim = n_features * total_thresholds
        
        # Initialize thresholds for each scale
        self.threshold_lists = nn.ParameterList() if learnable else []
        self.scale_weights = []
        
        for n_thresholds, weight in scales:
            # Evenly spaced thresholds in (0, 1)
            thresh = torch.linspace(1/(n_thresholds+1), n_thresholds/(n_thresholds+1), n_thresholds)
            thresh = thresh.unsqueeze(1).expand(-1, n_features)  # [n_thresh, n_features]
            
            if learnable:
                self.threshold_lists.append(nn.Parameter(thresh))
            else:
                self.register_buffer(f"thresh_{len(self.threshold_lists)}", thresh)
                self.threshold_lists.append(getattr(self, f"thresh_{len(self.threshold_lists)}"))
            
            self.scale_weights.append(weight)
    
    def forward(
        self,
        x: torch.Tensor,
        use_ste: bool = True,
    ) -> torch.Tensor:
        """
        Encode continuous features to multi-scale thermometer codes.
        
        Args:
            x: Input [batch, n_features] in [0, 1]
            use_ste: Use straight-through estimator
            
        Returns:
            Thermometer codes [batch, total_thresholds * n_features]
        """
        batch_size = x.shape[0]
        outputs = []
        
        for thresh, weight in zip(self.threshold_lists, self.scale_weights):
            # thresh: [n_thresh, n_features]
            # x: [batch, n_features]
            
            x_exp = x.unsqueeze(1)  # [batch, 1, n_features]
            thresh_exp = thresh.unsqueeze(0)  # [1, n_thresh, n_features]
            
            # Soft thresholding
            soft = torch.sigmoid((x_exp - thresh_exp) / self.temperature)
            
            if use_ste:
                hard = (x_exp > thresh_exp).float()
                soft = hard + (soft - soft.detach())
            
            # Apply scale weight and flatten
            weighted = soft * weight  # [batch, n_thresh, n_features]
            outputs.append(weighted.reshape(batch_size, -1))
        
        return torch.cat(outputs, dim=1)
    
    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.output_dim


# =============================================================================
# Gaussian Basis Function Expansion
# =============================================================================


class GaussianBasisExpansion(nn.Module):
    """
    Expand features using Gaussian basis functions (RBFs).
    
    For each feature, compute responses to multiple Gaussians centered
    at different locations. This creates a smooth, differentiable
    encoding that preserves similarity.
    
    y_i = exp(-||x - μ_i||² / (2σ²))
    
    Args:
        n_features: Number of input features
        n_bases: Number of Gaussian bases per feature
        sigma: Gaussian width (learnable if None)
        learnable_centers: Whether centers are learnable
    """
    
    def __init__(
        self,
        n_features: int,
        n_bases: int = 8,
        sigma: Optional[float] = None,
        learnable_centers: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_bases = n_bases
        self.output_dim = n_features * n_bases
        
        # Initialize centers evenly in [0, 1]
        centers = torch.linspace(0, 1, n_bases).unsqueeze(1).expand(-1, n_features)
        
        if learnable_centers:
            self.centers = nn.Parameter(centers)
        else:
            self.register_buffer("centers", centers)
        
        # Sigma (bandwidth)
        if sigma is None:
            # Learnable sigma, initialized based on spacing
            self.log_sigma = nn.Parameter(torch.tensor(-1.5))  # ~0.22
        else:
            self.register_buffer("log_sigma", torch.tensor(math.log(sigma)))
    
    @property
    def sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expand features using Gaussian bases.
        
        Args:
            x: Input [batch, n_features] in [0, 1]
            
        Returns:
            Gaussian responses [batch, n_bases * n_features]
        """
        batch_size = x.shape[0]
        
        # x: [batch, n_features]
        # centers: [n_bases, n_features]
        
        x_exp = x.unsqueeze(1)  # [batch, 1, n_features]
        centers_exp = self.centers.unsqueeze(0)  # [1, n_bases, n_features]
        
        # Gaussian responses
        dist_sq = (x_exp - centers_exp) ** 2  # [batch, n_bases, n_features]
        responses = torch.exp(-dist_sq / (2 * self.sigma ** 2))
        
        return responses.reshape(batch_size, -1)
    
    def get_output_dim(self) -> int:
        return self.output_dim


# =============================================================================
# Learned Feature Binning
# =============================================================================


class LearnedFeatureBins(nn.Module):
    """
    Learn optimal feature discretization end-to-end.
    
    Instead of fixed thresholds, learn a soft assignment to bins
    using attention-like mechanism:
    
    p(bin_k | x) = softmax(similarity(x, bin_center_k))
    output = sum_k p(bin_k | x) * bin_embedding_k
    
    Args:
        n_features: Number of input features
        n_bins: Number of bins per feature
        embed_dim: Dimension of bin embeddings
        temperature: Softmax temperature
    """
    
    def __init__(
        self,
        n_features: int,
        n_bins: int = 8,
        embed_dim: int = 4,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_bins = n_bins
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.output_dim = n_features * embed_dim
        
        # Bin centers: learnable locations in [0, 1]
        self.bin_centers = nn.Parameter(
            torch.linspace(0, 1, n_bins).unsqueeze(1).expand(-1, n_features)
        )
        
        # Bin embeddings: what each bin represents
        self.bin_embeddings = nn.Parameter(
            torch.randn(n_bins, n_features, embed_dim) * 0.1
        )
    
    def forward(
        self,
        x: torch.Tensor,
        use_hard: bool = False,
    ) -> torch.Tensor:
        """
        Assign features to learned bins.
        
        Args:
            x: Input [batch, n_features] in [0, 1]
            use_hard: Use hard assignment (argmax)
            
        Returns:
            Binned features [batch, n_features * embed_dim]
        """
        batch_size = x.shape[0]
        
        # x: [batch, n_features]
        # bin_centers: [n_bins, n_features]
        
        x_exp = x.unsqueeze(1)  # [batch, 1, n_features]
        centers_exp = self.bin_centers.unsqueeze(0)  # [1, n_bins, n_features]
        
        # Distance to each bin center
        distances = -torch.abs(x_exp - centers_exp)  # [batch, n_bins, n_features]
        
        # Soft assignment
        weights = F.softmax(distances / self.temperature, dim=1)  # [batch, n_bins, n_features]
        
        if use_hard:
            # Hard assignment with STE
            hard = F.one_hot(weights.argmax(dim=1), self.n_bins).permute(0, 2, 1).float()
            weights = hard + (weights - weights.detach())
        
        # Weighted sum of embeddings
        # weights: [batch, n_bins, n_features]
        # embeddings: [n_bins, n_features, embed_dim]
        
        output = torch.einsum('bnf,nfe->bfe', weights, self.bin_embeddings)
        return output.reshape(batch_size, -1)
    
    def get_output_dim(self) -> int:
        return self.output_dim


# =============================================================================
# Positional Value Encoding (like positional embeddings for values)
# =============================================================================


class PositionalValueEncoding(nn.Module):
    """
    Encode continuous values using sinusoidal positional encodings.
    
    Inspired by Transformer positional encodings, this encodes
    the position of a value in [0, 1] using sine/cosine functions
    at multiple frequencies:
    
    PE(v, 2i) = sin(v * 2π * freq_i)
    PE(v, 2i+1) = cos(v * 2π * freq_i)
    
    This creates a unique, continuous embedding for each value
    that preserves local similarity.
    
    Args:
        n_features: Number of input features
        n_frequencies: Number of frequency bands
        max_freq: Maximum frequency
        learnable_freqs: Whether frequencies are learnable
    """
    
    def __init__(
        self,
        n_features: int,
        n_frequencies: int = 8,
        max_freq: float = 64.0,
        learnable_freqs: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_frequencies = n_frequencies
        self.output_dim = n_features * n_frequencies * 2  # sin + cos
        
        # Frequencies: geometric series
        freqs = torch.exp(
            torch.linspace(0, math.log(max_freq), n_frequencies)
        )
        
        if learnable_freqs:
            self.log_freqs = nn.Parameter(torch.log(freqs))
        else:
            self.register_buffer("log_freqs", torch.log(freqs))
    
    @property
    def freqs(self) -> torch.Tensor:
        return torch.exp(self.log_freqs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode values using positional encoding.
        
        Args:
            x: Input [batch, n_features] in [0, 1]
            
        Returns:
            Positional encodings [batch, n_features * n_frequencies * 2]
        """
        batch_size = x.shape[0]
        
        # x: [batch, n_features]
        # freqs: [n_frequencies]
        
        x_exp = x.unsqueeze(-1)  # [batch, n_features, 1]
        freqs_exp = self.freqs.unsqueeze(0).unsqueeze(0)  # [1, 1, n_frequencies]
        
        # Compute angles
        angles = x_exp * freqs_exp * 2 * math.pi  # [batch, n_features, n_frequencies]
        
        # Sin and cos
        sin_enc = torch.sin(angles)
        cos_enc = torch.cos(angles)
        
        # Interleave sin and cos
        encoding = torch.stack([sin_enc, cos_enc], dim=-1)  # [batch, n_features, n_freq, 2]
        
        return encoding.reshape(batch_size, -1)
    
    def get_output_dim(self) -> int:
        return self.output_dim


# =============================================================================
# Combined Enhanced Encoder
# =============================================================================


class EnhancedContinuousEncoder(nn.Module):
    """
    Combines multiple encoding strategies for rich continuous representation.
    
    Outputs:
    1. Multi-scale thermometer codes (preserves ordering)
    2. Gaussian basis expansion (smooth similarity)
    3. Positional value encodings (frequency-based)
    4. Original continuous values (residual)
    
    All combined with learned fusion.
    
    Args:
        n_features: Number of input features
        thermometer_scales: Scales for thermometer encoding
        n_gaussian_bases: Number of Gaussian bases
        n_frequencies: Number of positional frequencies
        output_dim: Final output dimension (None = auto)
        dropout: Dropout rate
        lite_mode: Use memory-efficient encoding for large inputs
    """
    
    def __init__(
        self,
        n_features: int,
        thermometer_scales: List[Tuple[int, float]] = [(4, 0.5), (8, 1.0)],
        n_gaussian_bases: int = 8,
        n_frequencies: int = 8,
        output_dim: Optional[int] = None,
        dropout: float = 0.1,
        lite_mode: bool = False,
    ):
        super().__init__()
        self.n_features = n_features
        self.lite_mode = lite_mode
        
        # For large inputs, use lite mode to avoid OOM
        if n_features > 1000 or lite_mode:
            # Lite mode: project down first, then encode
            self.input_proj = nn.Sequential(
                nn.Linear(n_features, 256),
                nn.LayerNorm(256),
                nn.GELU(),
            )
            effective_features = 256
            thermometer_scales = [(2, 1.0), (4, 0.5)]
            n_gaussian_bases = 4
            n_frequencies = 4
        else:
            self.input_proj = None
            effective_features = n_features
        
        # Individual encoders on effective features
        self.thermometer = MultiScaleThermometer(
            effective_features, thermometer_scales
        )
        self.gaussian = GaussianBasisExpansion(
            effective_features, n_gaussian_bases
        )
        self.positional = PositionalValueEncoding(
            effective_features, n_frequencies
        )
        
        # Total dimension from all encoders + original
        concat_dim = (
            self.thermometer.get_output_dim() +
            self.gaussian.get_output_dim() +
            self.positional.get_output_dim() +
            effective_features  # Original/projected values
        )
        
        self.concat_dim = concat_dim
        self._output_dim = output_dim or min(concat_dim, 512)  # Cap output dim
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, self._output_dim),
            nn.LayerNorm(self._output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Residual projection
        self.residual = nn.Linear(effective_features, self._output_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        use_ste: bool = True,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        Encode continuous features with multiple strategies.
        
        Args:
            x: Input [batch, n_features] in [0, 1]
            use_ste: Use STE for thermometer
            return_components: Return individual components
            
        Returns:
            Encoded features [batch, output_dim]
        """
        # Project down for large inputs
        if self.input_proj is not None:
            x_enc = self.input_proj(x)
        else:
            x_enc = x
        
        # Individual encodings
        therm = self.thermometer(x_enc, use_ste=use_ste)
        gauss = self.gaussian(x_enc)
        pos = self.positional(x_enc)
        
        # Concatenate all + projected
        combined = torch.cat([therm, gauss, pos, x_enc], dim=1)
        
        # Fusion with residual
        fused = self.fusion(combined) + self.residual(x_enc)
        
        if return_components:
            return {
                "output": fused,
                "thermometer": therm,
                "gaussian": gauss,
                "positional": pos,
                "projected": x_enc,
            }
        
        return fused
    
    def get_output_dim(self) -> int:
        return self._output_dim


# =============================================================================
# Enhanced Clause Machine with Rich Continuous Encoding
# =============================================================================


class EnhancedContinuousTM(nn.Module):
    """
    Tsetlin Machine with enhanced continuous feature encoding.
    
    Uses multi-strategy encoding to preserve continuous information
    before clause evaluation.
    
    Args:
        n_features: Number of original input features
        n_clauses: Number of TM clauses
        n_classes: Number of output classes
        encoding_dim: Dimension of encoded features
        operator: TM clause operator
    """
    
    def __init__(
        self,
        n_features: int,
        n_clauses: int,
        n_classes: int,
        encoding_dim: Optional[int] = None,
        operator: str = "capacity",
    ):
        super().__init__()
        self.n_features = n_features
        self.n_clauses = n_clauses
        self.n_classes = n_classes
        
        # Enhanced encoder
        self.encoder = EnhancedContinuousEncoder(
            n_features=n_features,
            output_dim=encoding_dim,
        )
        
        # TM on encoded features
        from ..tm import FuzzyPatternTM_STCM
        
        self.tm = FuzzyPatternTM_STCM(
            n_features=self.encoder.get_output_dim(),
            n_clauses=n_clauses,
            n_classes=n_classes,
            operator=operator,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        use_ste: bool = True,
        return_encoding: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with enhanced encoding.
        
        Args:
            x: Input [batch, n_features] in [0, 1]
            use_ste: Use STE
            return_encoding: Also return encoding
            
        Returns:
            (logits, clause_outputs) or (logits, clause_outputs, encoding)
        """
        # Normalize to [0, 1] if needed
        if x.min() < 0 or x.max() > 1:
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        
        # Encode
        encoded = self.encoder(x, use_ste=use_ste)
        
        # Normalize encoded features
        encoded = torch.sigmoid(encoded)
        
        # TM forward
        logits, clauses = self.tm(encoded, use_ste=use_ste, skip_norm=True)
        
        if return_encoding:
            return logits, clauses, encoded
        return logits, clauses

