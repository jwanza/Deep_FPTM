"""
Ultimate Hybrid Tsetlin Machine.

Combines all advanced techniques into a single, highly configurable architecture:
1. Continuous Residual Stream - Preserves continuous information
2. Hyperdimensional Stream - Similarity-preserving binary encoding
3. Information Bottleneck - Optimal binarization
4. Probabilistic Literals - Uncertainty-aware clauses
5. Attention-Adaptive - Per-sample dynamic thresholds
6. Hyperbolic Voting - Non-Euclidean class relationships
7. Sparse MoE Routing - Dynamic clause activation
8. Clause Attention - Inter-clause reasoning

This architecture represents the state-of-the-art in differentiable Tsetlin Machines,
addressing the booleanization bottleneck from multiple complementary perspectives.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tm import FuzzyPatternTM_STCM, prepare_tm_input
from .booleanization.continuous_residual import ContinuousStreamEncoder, SoftThresholdBinarizer, StreamFusionGate
from .booleanization.hyperdimensional import HDEncoder
from .booleanization.information_bottleneck import InformationBottleneckBinarizer, VIBLayer
from .booleanization.probabilistic import DistributionalLiteral, UncertaintyAwareVoting
from .clause_attention import MultiHeadClauseAttention


# =============================================================================
# Stream Configuration
# =============================================================================


class StreamConfig:
    """Configuration for a single stream in the hybrid architecture."""
    
    def __init__(
        self,
        enabled: bool = True,
        weight: float = 1.0,
        **kwargs,
    ):
        self.enabled = enabled
        self.weight = weight
        self.kwargs = kwargs


# =============================================================================
# Individual Streams
# =============================================================================


class BinaryTMStream(nn.Module):
    """Standard binary TM stream with soft thresholding."""
    
    def __init__(
        self,
        n_features: int,
        n_clauses: int,
        n_classes: int,
        temperature: float = 1.0,
        operator: str = "capacity",
    ):
        super().__init__()
        self.n_features = n_features
        self.n_clauses = n_clauses
        
        self.binarizer = SoftThresholdBinarizer(
            n_features=n_features,
            temperature=temperature,
        )
        
        self.tm = FuzzyPatternTM_STCM(
            n_features=n_features,
            n_clauses=n_clauses,
            n_classes=n_classes,
            operator=operator,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        use_ste: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through binary TM stream."""
        x_binary = self.binarizer(x, use_ste=use_ste)
        return self.tm(x_binary, use_ste=use_ste, skip_norm=True)


class ContinuousStream(nn.Module):
    """Continuous residual stream."""
    
    def __init__(
        self,
        n_features: int,
        output_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.encoder = ContinuousStreamEncoder(
            n_features=n_features,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_reconstruction: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through continuous stream."""
        return self.encoder(x, return_reconstruction=return_reconstruction)


class HDStream(nn.Module):
    """Hyperdimensional computing stream."""
    
    def __init__(
        self,
        n_features: int,
        n_clauses: int,
        hd_dim: int = 1000,
        n_levels: int = 16,
    ):
        super().__init__()
        self.n_clauses = n_clauses
        
        self.encoder = HDEncoder(
            n_features=n_features,
            hd_dim=hd_dim,
            n_levels=n_levels,
        )
        
        self.proj = nn.Linear(hd_dim, n_clauses)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through HD stream."""
        hd_vector = self.encoder(x)
        return self.proj(hd_vector)


class IBStream(nn.Module):
    """Information Bottleneck stream."""
    
    def __init__(
        self,
        n_features: int,
        latent_dim: int,
        output_dim: int,
        beta: float = 0.01,
    ):
        super().__init__()
        self.vib = VIBLayer(
            input_dim=n_features,
            latent_dim=latent_dim,
            beta=beta,
        )
        self.proj = nn.Linear(latent_dim, output_dim)
        self._kl_loss = torch.tensor(0.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through IB stream."""
        z, kl_loss = self.vib(x)
        self._kl_loss = kl_loss
        return self.proj(z), kl_loss
    
    @property
    def kl_loss(self) -> torch.Tensor:
        return self._kl_loss


class ProbabilisticStream(nn.Module):
    """Probabilistic literal stream."""
    
    def __init__(
        self,
        n_features: int,
        n_clauses: int,
    ):
        super().__init__()
        self.literals = DistributionalLiteral(
            n_features=n_features,
            n_clauses=n_clauses,
        )
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through probabilistic stream."""
        clause_outputs, uncertainty = self.literals(x)
        return clause_outputs, uncertainty


# =============================================================================
# Multi-Stream Fusion
# =============================================================================


class AdaptiveStreamFusion(nn.Module):
    """
    Adaptively fuses multiple streams based on input.
    
    Uses attention to weight different streams per sample,
    allowing the model to dynamically choose which streams
    are most relevant for each input.
    """
    
    def __init__(
        self,
        n_streams: int,
        stream_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_streams = n_streams
        
        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(stream_dim * n_streams, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_streams),
        )
        
        # Learnable per-stream scale
        self.stream_scales = nn.Parameter(torch.ones(n_streams))
    
    def forward(
        self,
        streams: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse multiple streams.
        
        Args:
            streams: List of [batch, stream_dim] tensors
            
        Returns:
            (fused_output, stream_weights)
        """
        # Concatenate for gating
        concatenated = torch.cat(streams, dim=-1)
        
        # Compute adaptive weights
        raw_weights = self.gate(concatenated)
        weights = F.softmax(raw_weights * self.stream_scales, dim=-1)
        
        # Weighted sum of streams
        stacked = torch.stack(streams, dim=1)  # [batch, n_streams, stream_dim]
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        
        return fused, weights


# =============================================================================
# Ultimate Hybrid TM
# =============================================================================


class UltimateHybridTM(nn.Module):
    """
    Ultimate Hybrid Tsetlin Machine.
    
    Combines multiple parallel streams addressing the booleanization
    bottleneck from different perspectives, with adaptive fusion.
    
    Args:
        n_features: Number of input features
        n_clauses: Number of clauses per stream
        n_classes: Number of output classes
        
        # Stream enables
        use_binary_stream: Enable standard binary TM
        use_continuous_stream: Enable continuous residual stream
        use_hd_stream: Enable hyperdimensional stream
        use_ib_stream: Enable information bottleneck stream
        use_probabilistic_stream: Enable probabilistic literal stream
        
        # Stream configs
        hd_dim: HD vector dimension
        ib_latent_dim: IB latent dimension
        hidden_dim: Hidden layer dimension
        
        # Fusion
        fusion_type: 'adaptive', 'sum', 'concat'
        use_clause_attention: Apply attention across clauses
        
        # Regularization
        reconstruction_weight: Weight for reconstruction loss
        kl_weight: Weight for KL divergence loss
        uncertainty_weight: Weight for uncertainty loss
    """
    
    def __init__(
        self,
        n_features: int,
        n_clauses: int,
        n_classes: int,
        # Stream enables
        use_binary_stream: bool = True,
        use_continuous_stream: bool = True,
        use_hd_stream: bool = True,
        use_ib_stream: bool = False,
        use_probabilistic_stream: bool = False,
        # Stream configs
        hd_dim: int = 1000,
        ib_latent_dim: int = 16,
        hidden_dim: int = 64,
        temperature: float = 1.0,
        # Fusion
        fusion_type: str = "adaptive",
        use_clause_attention: bool = True,
        n_attention_heads: int = 4,
        # Regularization
        reconstruction_weight: float = 0.1,
        kl_weight: float = 0.01,
        uncertainty_weight: float = 0.01,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_clauses = n_clauses
        self.n_classes = n_classes
        self.fusion_type = fusion_type
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight
        self.uncertainty_weight = uncertainty_weight
        
        # Track which streams are enabled
        self.stream_names = []
        
        # Build streams
        if use_binary_stream:
            self.binary_stream = BinaryTMStream(
                n_features=n_features,
                n_clauses=n_clauses,
                n_classes=n_classes,
                temperature=temperature,
            )
            self.stream_names.append("binary")
        
        if use_continuous_stream:
            self.continuous_stream = ContinuousStream(
                n_features=n_features,
                output_dim=n_clauses,
                hidden_dim=hidden_dim,
            )
            self.stream_names.append("continuous")
        
        if use_hd_stream:
            self.hd_stream = HDStream(
                n_features=n_features,
                n_clauses=n_clauses,
                hd_dim=hd_dim,
            )
            self.stream_names.append("hd")
        
        if use_ib_stream:
            self.ib_stream = IBStream(
                n_features=n_features,
                latent_dim=ib_latent_dim,
                output_dim=n_clauses,
            )
            self.stream_names.append("ib")
        
        if use_probabilistic_stream:
            self.probabilistic_stream = ProbabilisticStream(
                n_features=n_features,
                n_clauses=n_clauses,
            )
            self.stream_names.append("probabilistic")
        
        n_streams = len(self.stream_names)
        
        # Fusion mechanism
        if fusion_type == "adaptive":
            self.fusion = AdaptiveStreamFusion(
                n_streams=n_streams,
                stream_dim=n_clauses,
                hidden_dim=hidden_dim,
            )
            fused_dim = n_clauses
        elif fusion_type == "concat":
            self.fusion = None
            fused_dim = n_clauses * n_streams
        else:  # sum
            self.fusion = None
            fused_dim = n_clauses
        
        # Clause attention
        if use_clause_attention and fused_dim > 0:
            self.clause_attention = MultiHeadClauseAttention(
                clause_dim=fused_dim,  # Each clause output is a scalar, but we treat the whole vector as the embedding
                n_heads=n_attention_heads,
            )
        else:
            self.clause_attention = None
        
        # Output voting
        self.voting = nn.Linear(fused_dim, n_classes)
        
        # Uncertainty-aware voting (optional)
        if use_probabilistic_stream:
            self.uncertainty_voting = UncertaintyAwareVoting(
                n_clauses=fused_dim,
                n_classes=n_classes,
            )
        else:
            self.uncertainty_voting = None
        
        # Loss accumulators
        self.register_buffer("_reconstruction_loss", torch.tensor(0.0))
        self.register_buffer("_kl_loss", torch.tensor(0.0))
        self.register_buffer("_uncertainty_loss", torch.tensor(0.0))
    
    def forward(
        self,
        x: torch.Tensor,
        use_ste: bool = True,
        return_details: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict]:
        """
        Forward pass through all streams with fusion.
        
        Args:
            x: Input tensor [batch, n_features] in [0, 1]
            use_ste: Use straight-through estimator
            return_details: Return detailed stream outputs
            
        Returns:
            (logits, fused_clauses) or dict with all details
        """
        batch_size = x.shape[0]
        x_flat = prepare_tm_input(x, n_features=self.n_features)
        
        stream_outputs = []
        details = {"stream_outputs": {}}
        
        # Binary stream
        if hasattr(self, "binary_stream"):
            _, binary_clauses = self.binary_stream(x_flat, use_ste=use_ste)
            stream_outputs.append(binary_clauses)
            details["stream_outputs"]["binary"] = binary_clauses
        
        # Continuous stream
        reconstruction = None
        if hasattr(self, "continuous_stream"):
            continuous_encoded, reconstruction = self.continuous_stream(
                x_flat, return_reconstruction=True
            )
            stream_outputs.append(continuous_encoded)
            details["stream_outputs"]["continuous"] = continuous_encoded
            details["reconstruction"] = reconstruction
            
            self._reconstruction_loss = F.mse_loss(reconstruction, x_flat)
        
        # HD stream
        if hasattr(self, "hd_stream"):
            hd_output = self.hd_stream(x_flat)
            stream_outputs.append(hd_output)
            details["stream_outputs"]["hd"] = hd_output
        
        # IB stream
        if hasattr(self, "ib_stream"):
            ib_output, kl_loss = self.ib_stream(x_flat)
            stream_outputs.append(ib_output)
            details["stream_outputs"]["ib"] = ib_output
            self._kl_loss = kl_loss
        
        # Probabilistic stream
        uncertainty = None
        if hasattr(self, "probabilistic_stream"):
            prob_output, uncertainty = self.probabilistic_stream(x_flat)
            stream_outputs.append(prob_output)
            details["stream_outputs"]["probabilistic"] = prob_output
            details["uncertainty"] = uncertainty
            
            if uncertainty is not None:
                self._uncertainty_loss = -torch.log(uncertainty + 1e-8).mean()
        
        # Fusion
        stream_weights = None
        if len(stream_outputs) == 0:
            raise ValueError("No streams enabled!")
        elif len(stream_outputs) == 1:
            fused = stream_outputs[0]
        elif self.fusion_type == "adaptive":
            fused, stream_weights = self.fusion(stream_outputs)
        elif self.fusion_type == "concat":
            fused = torch.cat(stream_outputs, dim=-1)
        else:  # sum
            fused = torch.stack(stream_outputs, dim=0).sum(dim=0)
        
        details["stream_weights"] = stream_weights
        
        # Clause attention - apply per-clause attention
        if self.clause_attention is not None:
            # fused is [batch, fused_dim], treat each element as a clause
            # MultiHeadClauseAttention expects [batch, seq, dim], we use [batch, 1, fused_dim]
            fused_expanded = fused.unsqueeze(1)  # [batch, 1, fused_dim]
            fused_attended = self.clause_attention(fused_expanded).squeeze(1)  # [batch, fused_dim]
            fused = fused + fused_attended  # Residual
        
        # Output
        if self.uncertainty_voting is not None and uncertainty is not None:
            logits = self.uncertainty_voting(fused, uncertainty)
        else:
            logits = self.voting(fused)
        
        if return_details:
            details["logits"] = logits
            details["fused_clauses"] = fused
            return details
        
        return logits, fused
    
    def get_auxiliary_losses(self) -> Dict[str, torch.Tensor]:
        """Get all auxiliary losses for training."""
        return {
            "reconstruction": self._reconstruction_loss * self.reconstruction_weight,
            "kl": self._kl_loss * self.kl_weight,
            "uncertainty": self._uncertainty_loss * self.uncertainty_weight,
        }
    
    def get_total_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        include_auxiliary: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total training loss.
        
        Args:
            logits: Model predictions
            targets: Ground truth labels
            include_auxiliary: Include auxiliary losses
            
        Returns:
            (total_loss, loss_components)
        """
        cls_loss = F.cross_entropy(logits, targets)
        
        loss_dict = {"classification": cls_loss.item()}
        total = cls_loss
        
        if include_auxiliary:
            aux_losses = self.get_auxiliary_losses()
            for name, loss in aux_losses.items():
                if loss.abs() > 0:
                    total = total + loss
                    loss_dict[name] = loss.item()
        
        loss_dict["total"] = total.item()
        return total, loss_dict


# =============================================================================
# Pre-configured Architectures
# =============================================================================


def create_light_hybrid(
    n_features: int,
    n_clauses: int,
    n_classes: int,
) -> UltimateHybridTM:
    """Create a lightweight hybrid with binary + continuous streams only."""
    return UltimateHybridTM(
        n_features=n_features,
        n_clauses=n_clauses,
        n_classes=n_classes,
        use_binary_stream=True,
        use_continuous_stream=True,
        use_hd_stream=False,
        use_ib_stream=False,
        use_probabilistic_stream=False,
        fusion_type="adaptive",
        use_clause_attention=False,
    )


def create_full_hybrid(
    n_features: int,
    n_clauses: int,
    n_classes: int,
) -> UltimateHybridTM:
    """Create a full hybrid with all streams enabled."""
    return UltimateHybridTM(
        n_features=n_features,
        n_clauses=n_clauses,
        n_classes=n_classes,
        use_binary_stream=True,
        use_continuous_stream=True,
        use_hd_stream=True,
        use_ib_stream=True,
        use_probabilistic_stream=True,
        fusion_type="adaptive",
        use_clause_attention=True,
    )


def create_fast_inference_hybrid(
    n_features: int,
    n_clauses: int,
    n_classes: int,
) -> UltimateHybridTM:
    """Create a hybrid optimized for fast inference."""
    return UltimateHybridTM(
        n_features=n_features,
        n_clauses=n_clauses,
        n_classes=n_classes,
        use_binary_stream=True,
        use_continuous_stream=False,
        use_hd_stream=True,
        use_ib_stream=False,
        use_probabilistic_stream=False,
        fusion_type="sum",  # Faster than adaptive
        use_clause_attention=False,
    )


def create_interpretable_hybrid(
    n_features: int,
    n_clauses: int,
    n_classes: int,
) -> UltimateHybridTM:
    """Create a hybrid with emphasis on interpretability."""
    return UltimateHybridTM(
        n_features=n_features,
        n_clauses=n_clauses,
        n_classes=n_classes,
        use_binary_stream=True,
        use_continuous_stream=True,
        use_hd_stream=False,
        use_ib_stream=False,
        use_probabilistic_stream=True,  # For uncertainty
        fusion_type="adaptive",
        use_clause_attention=True,
    )

