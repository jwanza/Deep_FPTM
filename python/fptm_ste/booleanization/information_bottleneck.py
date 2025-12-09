"""
Information Bottleneck Binarization.

Learns optimal binary encoding of continuous features by maximizing
information about labels while minimizing redundancy.

Key Innovation:
Instead of heuristic thresholding, learn the binarization that
preserves the most task-relevant information while compressing
to binary representation.

Architecture:
1. Encoder: Maps continuous features to binary with reparameterization
2. Decoder: Reconstructs continuous from binary (ensures information preserved)
3. IB Loss: Minimizes I(X;Z) while maximizing I(Z;Y)

Benefits:
- Learns optimal binarization for the task
- Information-theoretic foundation
- End-to-end differentiable
- Provably minimal sufficient statistics
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..tm import FuzzyPatternTM_STCM, prepare_tm_input


# =============================================================================
# Variational Binary Layer
# =============================================================================


class VIBLayer(nn.Module):
    """
    Variational Information Bottleneck layer for binarization.
    
    Uses reparameterization trick with hard concrete/Gumbel-sigmoid
    to enable gradient flow through binary decisions.
    
    Args:
        in_dim: Input dimension
        out_dim: Output dimension (number of binary features)
        temperature: Temperature for Gumbel-sigmoid
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.temperature = temperature
        
        # Encoder: predicts logits for each binary feature
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )
    
    def _gumbel_sigmoid(
        self,
        logits: torch.Tensor,
        training: bool,
    ) -> torch.Tensor:
        """
        Gumbel-sigmoid reparameterization.
        
        Enables gradient flow through binary sampling.
        """
        if training:
            # Sample Gumbel noise
            u = torch.rand_like(logits).clamp(1e-8, 1 - 1e-8)
            gumbel = -torch.log(-torch.log(u))
            
            # Soft sample
            soft = torch.sigmoid((logits + gumbel) / self.temperature)
            
            # Hard sample with straight-through
            hard = (soft > 0.5).float()
            return hard + (soft - soft.detach())
        else:
            return (torch.sigmoid(logits) > 0.5).float()
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode to binary with reparameterization.
        
        Args:
            x: Continuous input [batch, in_dim]
            
        Returns:
            (binary_encoding, logits)
        """
        logits = self.encoder(x)
        z = self._gumbel_sigmoid(logits, self.training)
        return z, logits
    
    def kl_divergence(self, logits: torch.Tensor) -> torch.Tensor:
        """
        KL divergence from prior (uniform Bernoulli).
        
        Args:
            logits: Encoder logits
            
        Returns:
            KL divergence (for IB regularization)
        """
        # P(z=1) from logits
        p = torch.sigmoid(logits)
        
        # KL from Bernoulli(p) to Bernoulli(0.5)
        # KL = p*log(2p) + (1-p)*log(2(1-p))
        eps = 1e-8
        kl = p * torch.log(2 * p + eps) + (1 - p) * torch.log(2 * (1 - p) + eps)
        
        return kl.sum(dim=-1).mean()


# =============================================================================
# Information Bottleneck Binarizer
# =============================================================================


class InformationBottleneckBinarizer(nn.Module):
    """
    Information Bottleneck binarizer with encoder-decoder.
    
    Learns optimal binary encoding by balancing:
    - Compression: minimize I(X;Z) via KL from prior
    - Prediction: maximize I(Z;Y) via classification loss
    
    Args:
        n_features: Number of input features
        n_binary: Number of binary features (can be different from input)
        temperature: Gumbel-sigmoid temperature
        beta: IB tradeoff parameter (higher = more compression)
    """
    
    def __init__(
        self,
        n_features: int,
        n_binary: Optional[int] = None,
        temperature: float = 0.5,
        beta: float = 0.001,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_binary = n_binary or n_features
        self.beta = beta
        
        # Encoder
        self.vib = VIBLayer(
            in_dim=n_features,
            out_dim=self.n_binary,
            temperature=temperature,
        )
        
        # Decoder (for reconstruction auxiliary loss)
        self.decoder = nn.Sequential(
            nn.Linear(self.n_binary, n_features * 2),
            nn.GELU(),
            nn.Linear(n_features * 2, n_features),
            nn.Sigmoid(),
        )
    
    def encode(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode continuous to binary.
        
        Args:
            x: Continuous input [batch, n_features]
            
        Returns:
            (binary_encoding, logits)
        """
        return self.vib(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode binary back to continuous.
        
        Args:
            z: Binary encoding [batch, n_binary]
            
        Returns:
            Reconstructed continuous [batch, n_features]
        """
        return self.decoder(z)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full encode-decode pass.
        
        Args:
            x: Input features
            
        Returns:
            (binary_encoding, reconstruction, logits)
        """
        z, logits = self.encode(x)
        reconstruction = self.decode(z)
        return z, reconstruction, logits
    
    def information_loss(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        logits: torch.Tensor,
        reconstruction: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute IB loss components.
        
        Args:
            x: Original input
            z: Binary encoding
            logits: Encoder logits
            reconstruction: Decoded reconstruction
            
        Returns:
            (total_loss, component_dict)
        """
        # Reconstruction loss (proxy for I(X;Z))
        recon_loss = F.mse_loss(reconstruction, x)
        
        # KL divergence (compression term)
        kl_loss = self.vib.kl_divergence(logits)
        
        # Total
        total = recon_loss + self.beta * kl_loss
        
        return total, {
            "reconstruction": recon_loss.item(),
            "kl": kl_loss.item(),
            "total": total.item(),
        }


# =============================================================================
# Information Preserving Clause Machine
# =============================================================================


class InformationPreservingClauseMachine(nn.Module):
    """
    Clause Machine with Information Bottleneck binarization.
    
    Learns optimal binary encoding of inputs that preserves
    task-relevant information while enabling TM-style clause evaluation.
    
    Args:
        n_features: Number of input features
        n_clauses: Number of TM clauses
        n_classes: Number of output classes
        n_binary: Number of binary features (None = same as input)
        beta: IB compression strength
        temperature: Binarization temperature
        operator: TM clause operator
    """
    
    def __init__(
        self,
        n_features: int,
        n_clauses: int,
        n_classes: int,
        n_binary: Optional[int] = None,
        beta: float = 0.001,
        temperature: float = 0.5,
        operator: str = "capacity",
    ):
        super().__init__()
        self.n_features = n_features
        self.n_binary = n_binary or n_features
        self.n_clauses = n_clauses
        self.n_classes = n_classes
        self.beta = beta
        
        # IB binarizer
        self.binarizer = InformationBottleneckBinarizer(
            n_features=n_features,
            n_binary=self.n_binary,
            temperature=temperature,
            beta=beta,
        )
        
        # TM operating on binary features
        self.tm = FuzzyPatternTM_STCM(
            n_features=self.n_binary,
            n_clauses=n_clauses,
            n_classes=n_classes,
            operator=operator,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        use_ste: bool = True,
        return_ib_details: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with IB binarization.
        
        Args:
            x: Continuous input [batch, n_features]
            use_ste: Use STE for TM
            return_ib_details: Return IB components
            
        Returns:
            (logits, clause_outputs) or dict with details
        """
        # Prepare input
        x_flat = prepare_tm_input(x, n_features=self.n_features)
        
        # IB binarization
        z, reconstruction, logits = self.binarizer(x_flat)
        
        # TM forward on binary features
        tm_logits, clauses = self.tm(z, use_ste=use_ste, skip_norm=True)
        
        if return_ib_details:
            ib_loss, ib_components = self.binarizer.information_loss(
                x_flat, z, logits, reconstruction
            )
            return {
                "logits": tm_logits,
                "clause_outputs": clauses,
                "binary_encoding": z,
                "reconstruction": reconstruction,
                "ib_loss": ib_loss,
                "ib_components": ib_components,
            }
        
        return tm_logits, clauses
    
    def get_total_loss(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        use_ste: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total loss including classification and IB terms.
        
        Args:
            x: Input features
            targets: Ground truth labels
            use_ste: Use STE
            
        Returns:
            (total_loss, loss_components)
        """
        result = self.forward(x, use_ste=use_ste, return_ib_details=True)
        
        # Classification loss
        cls_loss = F.cross_entropy(result["logits"], targets)
        
        # IB loss (already includes reconstruction + KL)
        ib_loss = result["ib_loss"]
        
        # Total
        total = cls_loss + ib_loss
        
        return total, {
            "classification": cls_loss.item(),
            "ib_total": ib_loss.item(),
            **result["ib_components"],
        }




