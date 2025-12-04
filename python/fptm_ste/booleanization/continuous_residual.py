"""
Continuous Residual Clause Machine (CRCM).

This module implements a dual-stream architecture that processes both
binary and continuous representations in parallel, with learned fusion.

Key Innovation:
The fundamental insight is that booleanization loses information about
feature magnitudes. CRCM maintains a parallel "residual" stream that
preserves continuous values and fuses them back into the output.

Architecture:
1. Binary Stream: Standard TM clause evaluation on thresholded features
2. Continuous Stream: MLP processing on original continuous features
3. Fusion Gate: Learned combination of both streams
4. Reconstruction: Auxiliary loss to ensure continuous stream is informative

Benefits:
- Retains full continuous information
- Binary stream provides interpretability
- Fusion learns optimal combination per sample
- Compatible with existing TM infrastructure
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..tm import FuzzyPatternTM_STCM, prepare_tm_input


# =============================================================================
# Soft Thresholding
# =============================================================================


class SoftThresholdBinarizer(nn.Module):
    """
    Differentiable soft-threshold binarization.
    
    Converts continuous features to soft binary values using
    learned thresholds with temperature-controlled sigmoid.
    
    This is more flexible than fixed 0.5 threshold and allows
    the model to learn optimal binarization boundaries.
    
    Args:
        n_features: Number of input features
        n_thresholds: Number of threshold levels per feature
        temperature: Sigmoid temperature (lower = sharper)
        learnable_thresholds: Whether thresholds are learnable
    """
    
    def __init__(
        self,
        n_features: int,
        n_thresholds: int = 1,
        temperature: float = 1.0,
        learnable_thresholds: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_thresholds = n_thresholds
        self.temperature = temperature
        
        # Initialize thresholds at 0.5 (middle of [0, 1])
        if learnable_thresholds:
            self.thresholds = nn.Parameter(
                torch.full((n_thresholds, n_features), 0.5)
            )
        else:
            self.register_buffer(
                "thresholds",
                torch.full((n_thresholds, n_features), 0.5)
            )
    
    def forward(
        self,
        x: torch.Tensor,
        use_ste: bool = True,
    ) -> torch.Tensor:
        """
        Apply soft thresholding.
        
        Args:
            x: Input tensor [batch, n_features] in [0, 1]
            use_ste: Use straight-through estimator
            
        Returns:
            Soft binary tensor [batch, n_thresholds, n_features] or
            [batch, n_features] if n_thresholds=1
        """
        # x: [batch, n_features]
        # thresholds: [n_thresholds, n_features]
        
        # Broadcast: x becomes [batch, 1, n_features]
        x_exp = x.unsqueeze(1)
        thresholds = self.thresholds.unsqueeze(0)  # [1, n_thresholds, n_features]
        
        # Soft thresholding
        soft = torch.sigmoid((x_exp - thresholds) / self.temperature)
        
        if use_ste:
            # Hard threshold with soft gradients
            hard = (x_exp > thresholds).float()
            soft = hard + (soft - soft.detach())
        
        if self.n_thresholds == 1:
            return soft.squeeze(1)
        return soft
    
    def get_thresholds(self) -> torch.Tensor:
        """Get current threshold values."""
        return self.thresholds.detach()


# =============================================================================
# Continuous Stream Encoder
# =============================================================================


class ContinuousStreamEncoder(nn.Module):
    """
    Encodes continuous features for the residual stream.
    
    Uses a small MLP to extract features from the original
    continuous values that complement the binary stream.
    
    Args:
        n_features: Number of input features
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        n_layers: Number of hidden layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        output_dim: Optional[int] = None,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.output_dim = output_dim or n_features
        
        layers = []
        in_dim = n_features
        
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else self.output_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Reconstruction decoder (for auxiliary loss)
        self.decoder = nn.Linear(self.output_dim, n_features)
    
    def forward(
        self,
        x: torch.Tensor,
        return_reconstruction: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode continuous features.
        
        Args:
            x: Continuous input [batch, n_features]
            return_reconstruction: Also return reconstructed input
            
        Returns:
            Encoded features, optionally with reconstruction
        """
        encoded = self.encoder(x)
        
        if return_reconstruction:
            reconstruction = torch.sigmoid(self.decoder(encoded))
            return encoded, reconstruction
        
        return encoded


# =============================================================================
# Fusion Gate
# =============================================================================


class StreamFusionGate(nn.Module):
    """
    Learnable gate for combining binary and continuous streams.
    
    Computes a per-sample, per-feature gate that determines
    how much to rely on each stream.
    
    Args:
        binary_dim: Dimension of binary stream features
        continuous_dim: Dimension of continuous stream features
        output_dim: Output dimension after fusion
        gate_type: Type of gating ('sigmoid', 'softmax', 'attention')
    """
    
    def __init__(
        self,
        binary_dim: int,
        continuous_dim: int,
        output_dim: int,
        gate_type: str = "sigmoid",
    ):
        super().__init__()
        self.gate_type = gate_type
        
        combined_dim = binary_dim + continuous_dim
        
        if gate_type == "sigmoid":
            self.gate = nn.Sequential(
                nn.Linear(combined_dim, output_dim),
                nn.Sigmoid(),
            )
        elif gate_type == "softmax":
            # Produces two weights that sum to 1
            self.gate = nn.Linear(combined_dim, 2)
        elif gate_type == "attention":
            self.query = nn.Linear(binary_dim, output_dim)
            self.key_binary = nn.Linear(binary_dim, output_dim)
            self.key_continuous = nn.Linear(continuous_dim, output_dim)
        
        # Projection for each stream
        self.binary_proj = nn.Linear(binary_dim, output_dim)
        self.continuous_proj = nn.Linear(continuous_dim, output_dim)
    
    def forward(
        self,
        binary_features: torch.Tensor,
        continuous_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse binary and continuous streams.
        
        Args:
            binary_features: From TM clause evaluation [batch, binary_dim]
            continuous_features: From continuous encoder [batch, continuous_dim]
            
        Returns:
            (fused_features, gate_values)
        """
        # Project to common dimension
        binary_proj = self.binary_proj(binary_features)
        continuous_proj = self.continuous_proj(continuous_features)
        
        if self.gate_type == "sigmoid":
            combined = torch.cat([binary_features, continuous_features], dim=-1)
            gate = self.gate(combined)
            fused = gate * binary_proj + (1 - gate) * continuous_proj
        
        elif self.gate_type == "softmax":
            combined = torch.cat([binary_features, continuous_features], dim=-1)
            weights = F.softmax(self.gate(combined), dim=-1)
            fused = weights[:, 0:1] * binary_proj + weights[:, 1:2] * continuous_proj
            gate = weights[:, 0:1]
        
        elif self.gate_type == "attention":
            # Use binary features as query
            q = self.query(binary_features)
            k_bin = self.key_binary(binary_features)
            k_cont = self.key_continuous(continuous_features)
            
            # Attention scores
            score_bin = (q * k_bin).sum(dim=-1, keepdim=True)
            score_cont = (q * k_cont).sum(dim=-1, keepdim=True)
            
            weights = F.softmax(torch.cat([score_bin, score_cont], dim=-1), dim=-1)
            fused = weights[:, 0:1] * binary_proj + weights[:, 1:2] * continuous_proj
            gate = weights[:, 0:1]
        
        return fused, gate


# =============================================================================
# Continuous Residual Clause Machine
# =============================================================================


class ContinuousResidualClauseMachine(nn.Module):
    """
    Continuous Residual Clause Machine (CRCM).
    
    Dual-stream architecture that processes both binary (via TM) and
    continuous (via MLP) representations, with learned fusion.
    
    The key insight is that booleanization loses magnitude information,
    which the continuous stream preserves and contributes via fusion.
    
    Args:
        n_features: Number of input features
        n_clauses: Number of TM clauses
        n_classes: Number of output classes
        hidden_dim: Hidden dimension for continuous stream
        fusion_type: Type of stream fusion ('sigmoid', 'softmax', 'attention')
        reconstruction_weight: Weight for reconstruction loss
        temperature: Soft threshold temperature
        operator: TM clause operator
    """
    
    def __init__(
        self,
        n_features: int,
        n_clauses: int,
        n_classes: int,
        hidden_dim: int = 64,
        fusion_type: str = "sigmoid",
        reconstruction_weight: float = 0.1,
        temperature: float = 1.0,
        operator: str = "capacity",
    ):
        super().__init__()
        self.n_features = n_features
        self.n_clauses = n_clauses
        self.n_classes = n_classes
        self.reconstruction_weight = reconstruction_weight
        
        # Soft threshold binarizer
        self.binarizer = SoftThresholdBinarizer(
            n_features=n_features,
            temperature=temperature,
            learnable_thresholds=True,
        )
        
        # Binary stream: TM
        self.binary_tm = FuzzyPatternTM_STCM(
            n_features=n_features,
            n_clauses=n_clauses,
            n_classes=n_classes,
            operator=operator,
        )
        
        # Continuous stream: MLP encoder
        self.continuous_encoder = ContinuousStreamEncoder(
            n_features=n_features,
            hidden_dim=hidden_dim,
            output_dim=n_clauses,  # Match clause dimension
        )
        
        # Fusion gate
        self.fusion = StreamFusionGate(
            binary_dim=n_clauses,
            continuous_dim=n_clauses,
            output_dim=n_clauses,
            gate_type=fusion_type,
        )
        
        # Output voting (shared with both streams)
        self.voting = nn.Linear(n_clauses, n_classes)
    
    def forward(
        self,
        x: torch.Tensor,
        use_ste: bool = True,
        return_details: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], dict]:
        """
        Dual-stream forward pass.
        
        Args:
            x: Continuous input [batch, n_features] in [0, 1]
            use_ste: Use STE for binarization
            return_details: Return detailed outputs
            
        Returns:
            (logits, clause_outputs) or dict with details
        """
        # Prepare input
        x_flat = prepare_tm_input(x, n_features=self.n_features)
        
        # Binary stream
        x_binary = self.binarizer(x_flat, use_ste=use_ste)
        _, binary_clauses = self.binary_tm(x_binary, use_ste=use_ste, skip_norm=True)
        
        # Continuous stream
        continuous_encoded, reconstruction = self.continuous_encoder(
            x_flat, return_reconstruction=True
        )
        
        # Fuse streams
        fused, gate = self.fusion(binary_clauses, continuous_encoded)
        
        # Output
        logits = self.voting(fused)
        
        if return_details:
            return {
                "logits": logits,
                "clause_outputs": fused,
                "binary_clauses": binary_clauses,
                "continuous_encoded": continuous_encoded,
                "reconstruction": reconstruction,
                "gate": gate,
            }
        
        return logits, fused
    
    def information_preservation_loss(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss for information preservation.
        
        Args:
            x: Original input
            reconstruction: Reconstructed input
            
        Returns:
            Reconstruction MSE loss
        """
        return F.mse_loss(reconstruction, x)
    
    def get_total_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        reconstruction: torch.Tensor,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total loss with all components.
        
        Args:
            logits: Model predictions
            targets: Ground truth labels
            reconstruction: Reconstructed input
            x: Original input
            
        Returns:
            (total_loss, loss_components)
        """
        # Classification loss
        cls_loss = F.cross_entropy(logits, targets)
        
        # Reconstruction loss
        recon_loss = self.information_preservation_loss(x, reconstruction)
        
        # Total
        total = cls_loss + self.reconstruction_weight * recon_loss
        
        return total, {
            "classification": cls_loss.item(),
            "reconstruction": recon_loss.item(),
            "total": total.item(),
        }


# =============================================================================
# Dual Stream TM (Simplified Version)
# =============================================================================


class DualStreamTM(nn.Module):
    """
    Simplified dual-stream TM with direct addition.
    
    A simpler variant that directly adds continuous features
    to binary clause outputs without complex fusion.
    
    Args:
        n_features: Number of input features
        n_clauses: Number of TM clauses
        n_classes: Number of output classes
        continuous_weight: Weight for continuous stream contribution
        operator: TM clause operator
    """
    
    def __init__(
        self,
        n_features: int,
        n_clauses: int,
        n_classes: int,
        continuous_weight: float = 0.5,
        operator: str = "capacity",
    ):
        super().__init__()
        self.n_features = n_features
        self.n_clauses = n_clauses
        self.n_classes = n_classes
        self.continuous_weight = continuous_weight
        
        # Binary stream
        self.tm = FuzzyPatternTM_STCM(
            n_features=n_features,
            n_clauses=n_clauses,
            n_classes=n_classes,
            operator=operator,
        )
        
        # Continuous stream (simple projection)
        self.continuous_proj = nn.Sequential(
            nn.Linear(n_features, n_clauses),
            nn.Tanh(),
        )
        
        # Learnable combination weight
        self.alpha = nn.Parameter(torch.tensor(continuous_weight))
    
    def forward(
        self,
        x: torch.Tensor,
        use_ste: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dual-stream forward.
        
        Args:
            x: Input tensor [batch, n_features]
            use_ste: Use STE for TM
            
        Returns:
            (logits, combined_clauses)
        """
        # Prepare input
        x_flat = prepare_tm_input(x, n_features=self.n_features)
        
        # Binary stream
        logits_binary, clauses_binary = self.tm(x_flat, use_ste=use_ste)
        
        # Continuous stream
        clauses_continuous = self.continuous_proj(x_flat)
        
        # Combine
        alpha = torch.sigmoid(self.alpha)
        combined = (1 - alpha) * clauses_binary + alpha * clauses_continuous
        
        # Re-vote
        if hasattr(self.tm, '_voting_matrix'):
            voting = self.tm._voting_matrix(use_ste)
        else:
            voting = self.tm.voting
        
        logits = combined @ voting
        
        return logits, combined


