"""
Hyperbolic Geometry for Clause Embeddings and Voting.

This module implements hyperbolic space operations for Tsetlin Machines,
enabling hierarchical representation learning where semantic relationships
naturally correspond to geometric distances.

Key innovations:
1. Poincare Ball Projection: Maps clause outputs to hyperbolic space
2. Hyperbolic Distance: Geodesic distances capture hierarchical similarity
3. Mobius Operations: Proper vector arithmetic in hyperbolic space
4. Hyperbolic Voting: Class predictions via hyperbolic distances to prototypes

Mathematical Background:
- Poincare ball model: Unit ball with metric ds^2 = (2/(1-||x||^2))^2 * ||dx||^2
- Points near boundary represent more specialized/leaf concepts
- Points near origin represent more general/root concepts
- Geodesic distances grow logarithmically near boundary

References:
- Nickel & Kiela (2017): Poincare Embeddings for Learning Hierarchical Representations
- Ganea et al. (2018): Hyperbolic Neural Networks
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Mathematical Constants and Utilities
# =============================================================================

EPS = 1e-6  # Numerical stability constant
MAX_NORM = 1.0 - 1e-5  # Maximum norm to stay inside Poincare ball


def _clamp_norm(x: torch.Tensor, max_norm: float = MAX_NORM) -> torch.Tensor:
    """
    Clamp tensor norms to stay strictly inside the Poincare ball.
    
    Args:
        x: Input tensor of shape [..., d]
        max_norm: Maximum allowed norm (< 1.0)
        
    Returns:
        Tensor with norms clamped below max_norm
    """
    norms = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    scale = torch.clamp(max_norm / norms, max=1.0)
    return x * scale


def _mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Mobius addition in the Poincare ball.
    
    Implements the formula:
    x ⊕ y = ((1 + 2c<x,y> + c||y||^2)x + (1 - c||x||^2)y) / 
            (1 + 2c<x,y> + c^2||x||^2||y||^2)
    
    Args:
        x: First operand [..., d]
        y: Second operand [..., d]
        c: Curvature parameter (default: 1.0 for unit curvature)
        
    Returns:
        Mobius sum x ⊕ y
    """
    x_sq = (x * x).sum(dim=-1, keepdim=True).clamp(max=1.0 - EPS)
    y_sq = (y * y).sum(dim=-1, keepdim=True).clamp(max=1.0 - EPS)
    xy = (x * y).sum(dim=-1, keepdim=True)
    
    num = (1 + 2 * c * xy + c * y_sq) * x + (1 - c * x_sq) * y
    denom = 1 + 2 * c * xy + c * c * x_sq * y_sq
    
    return _clamp_norm(num / denom.clamp(min=EPS))


def _mobius_scalar_mul(r: torch.Tensor, x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Mobius scalar multiplication in the Poincare ball.
    
    Implements: r ⊗ x = tanh(r * arctanh(sqrt(c)||x||)) * x / (sqrt(c)||x||)
    
    Args:
        r: Scalar multiplier [...] or scalar
        x: Point in Poincare ball [..., d]
        c: Curvature parameter
        
    Returns:
        Scaled point r ⊗ x
    """
    sqrt_c = math.sqrt(c)
    x_norm = x.norm(dim=-1, keepdim=True).clamp(min=EPS, max=1.0 - EPS)
    
    # arctanh(sqrt(c) * ||x||)
    scaled_norm = (sqrt_c * x_norm).clamp(max=1.0 - EPS)
    arctanh_norm = 0.5 * torch.log((1 + scaled_norm) / (1 - scaled_norm + EPS))
    
    # tanh(r * arctanh(...)) / (sqrt(c) * ||x||)
    if isinstance(r, (int, float)):
        r = torch.tensor(r, device=x.device, dtype=x.dtype)
    if r.dim() < x.dim():
        r = r.unsqueeze(-1)
    
    new_norm = torch.tanh(r * arctanh_norm)
    scale = new_norm / (sqrt_c * x_norm + EPS)
    
    return _clamp_norm(scale * x)


def _poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Geodesic distance in the Poincare ball.
    
    Implements: d(x, y) = (2/sqrt(c)) * arctanh(sqrt(c) * ||−x ⊕ y||)
    
    Args:
        x: First point [..., d]
        y: Second point [..., d]
        c: Curvature parameter
        
    Returns:
        Poincare distances [...] (scalar per point pair)
    """
    sqrt_c = math.sqrt(c)
    
    # Compute -x ⊕ y (Mobius addition of negated x and y)
    neg_x = -x
    diff = _mobius_add(neg_x, y, c)
    
    # Distance formula
    diff_norm = (diff.norm(dim=-1) * sqrt_c).clamp(max=1.0 - EPS)
    dist = (2.0 / sqrt_c) * torch.atanh(diff_norm)
    
    return dist


def _exp_map(v: torch.Tensor, x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Exponential map from tangent space at x to Poincare ball.
    
    Maps a tangent vector v at point x to a point on the manifold.
    
    Args:
        v: Tangent vector at x [..., d]
        x: Base point [..., d]
        c: Curvature parameter
        
    Returns:
        Point on manifold corresponding to v
    """
    sqrt_c = math.sqrt(c)
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=EPS)
    
    # Conformal factor at x
    x_sq = (x * x).sum(dim=-1, keepdim=True)
    lambda_x = 2.0 / (1 - c * x_sq + EPS)
    
    # Exponential map formula
    second_term = torch.tanh(sqrt_c * lambda_x * v_norm / 2) * v / (sqrt_c * v_norm)
    
    return _mobius_add(x, second_term, c)


def _log_map(y: torch.Tensor, x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Logarithmic map from Poincare ball to tangent space at x.
    
    Inverse of exponential map: maps point y to tangent vector at x.
    
    Args:
        y: Point on manifold [..., d]
        x: Base point [..., d]
        c: Curvature parameter
        
    Returns:
        Tangent vector at x pointing toward y
    """
    sqrt_c = math.sqrt(c)
    
    # Compute -x ⊕ y
    diff = _mobius_add(-x, y, c)
    diff_norm = diff.norm(dim=-1, keepdim=True).clamp(min=EPS)
    
    # Conformal factor at x
    x_sq = (x * x).sum(dim=-1, keepdim=True)
    lambda_x = 2.0 / (1 - c * x_sq + EPS)
    
    # Log map formula
    scaled_norm = (sqrt_c * diff_norm).clamp(max=1.0 - EPS)
    arctanh_term = torch.atanh(scaled_norm)
    
    return (2.0 / (sqrt_c * lambda_x)) * arctanh_term * diff / diff_norm


# =============================================================================
# Neural Network Modules
# =============================================================================


class PoincareBallProjection(nn.Module):
    """
    Projects Euclidean vectors into the Poincare ball.
    
    Uses exponential map from origin or learned projection to map
    arbitrary vectors into the hyperbolic space while maintaining
    gradient flow.
    
    Args:
        in_dim: Input dimension
        out_dim: Output dimension (hyperbolic embedding size)
        curvature: Curvature parameter (default: 1.0)
        method: Projection method ('exp' for exponential map, 'normalize' for simple normalization)
        learnable_curvature: Whether curvature is learnable
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        curvature: float = 1.0,
        method: str = "exp",
        learnable_curvature: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.method = method
        
        # Linear projection to embedding dimension
        self.proj = nn.Linear(in_dim, out_dim)
        
        # Curvature parameter
        if learnable_curvature:
            self.log_curvature = nn.Parameter(torch.tensor(math.log(curvature)))
        else:
            self.register_buffer("log_curvature", torch.tensor(math.log(curvature)))
    
    @property
    def curvature(self) -> float:
        """Current curvature value."""
        return torch.exp(self.log_curvature).item()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input to Poincare ball.
        
        Args:
            x: Input tensor [..., in_dim]
            
        Returns:
            Hyperbolic embeddings [..., out_dim] with norm < 1
        """
        # Linear projection
        h = self.proj(x)
        
        c = torch.exp(self.log_curvature)
        
        if self.method == "exp":
            # Use exponential map from origin
            # At origin, exp_0(v) = tanh(sqrt(c)||v||/2) * v / (sqrt(c)||v||)
            sqrt_c = torch.sqrt(c)
            h_norm = h.norm(dim=-1, keepdim=True).clamp(min=EPS)
            scale = torch.tanh(sqrt_c * h_norm / 2) / (sqrt_c * h_norm + EPS)
            out = h * scale
        else:
            # Simple normalization with scaling
            out = h / (1 + h.norm(dim=-1, keepdim=True))
        
        return _clamp_norm(out)
    
    def extra_repr(self) -> str:
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}, method='{self.method}'"


class HyperbolicDistance(nn.Module):
    """
    Computes pairwise geodesic distances in Poincare ball.
    
    Provides batched, differentiable distance computation with
    optional temperature scaling for attention-like mechanisms.
    
    Args:
        curvature: Curvature parameter
        temperature: Temperature for distance scaling
        learnable_curvature: Whether curvature is learnable
    """
    
    def __init__(
        self,
        curvature: float = 1.0,
        temperature: float = 1.0,
        learnable_curvature: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        
        if learnable_curvature:
            self.log_curvature = nn.Parameter(torch.tensor(math.log(curvature)))
        else:
            self.register_buffer("log_curvature", torch.tensor(math.log(curvature)))
    
    @property
    def curvature(self) -> float:
        return torch.exp(self.log_curvature).item()
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        pairwise: bool = True,
    ) -> torch.Tensor:
        """
        Compute hyperbolic distances.
        
        Args:
            x: First set of points [batch, n, d] or [batch, d]
            y: Second set of points [batch, m, d] or [batch, d]
            pairwise: If True, compute all pairwise distances [batch, n, m]
                     If False, compute element-wise distances [batch, n]
                     
        Returns:
            Distance tensor, scaled by temperature
        """
        c = torch.exp(self.log_curvature).item()
        
        if pairwise:
            # Ensure 3D tensors
            if x.dim() == 2:
                x = x.unsqueeze(1)
            if y.dim() == 2:
                y = y.unsqueeze(1)
            
            # Expand for pairwise computation
            # x: [batch, n, 1, d], y: [batch, 1, m, d]
            x_exp = x.unsqueeze(2)
            y_exp = y.unsqueeze(1)
            
            # Compute pairwise distances
            dist = _poincare_distance(x_exp, y_exp, c)  # [batch, n, m]
        else:
            dist = _poincare_distance(x, y, c)
        
        return dist / self.temperature
    
    def extra_repr(self) -> str:
        return f"temperature={self.temperature}"


class MobiusAddition(nn.Module):
    """
    Learnable Mobius addition layer.
    
    Combines two hyperbolic embeddings using Mobius addition,
    optionally with learnable weights.
    
    Args:
        dim: Embedding dimension
        curvature: Curvature parameter
        learnable_weights: Whether to learn combination weights
    """
    
    def __init__(
        self,
        dim: int,
        curvature: float = 1.0,
        learnable_weights: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.curvature = curvature
        
        if learnable_weights:
            self.alpha = nn.Parameter(torch.tensor(0.5))
            self.beta = nn.Parameter(torch.tensor(0.5))
        else:
            self.register_buffer("alpha", torch.tensor(0.5))
            self.register_buffer("beta", torch.tensor(0.5))
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted Mobius sum.
        
        Args:
            x: First hyperbolic embedding [..., d]
            y: Second hyperbolic embedding [..., d]
            
        Returns:
            Combined embedding (alpha ⊗ x) ⊕ (beta ⊗ y)
        """
        # Scale each input
        x_scaled = _mobius_scalar_mul(self.alpha, x, self.curvature)
        y_scaled = _mobius_scalar_mul(self.beta, y, self.curvature)
        
        # Combine via Mobius addition
        return _mobius_add(x_scaled, y_scaled, self.curvature)


class HyperbolicLinear(nn.Module):
    """
    Linear layer operating in hyperbolic space.
    
    Implements the Mobius matrix-vector multiplication for
    transforming hyperbolic embeddings.
    
    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        curvature: Curvature parameter
        bias: Whether to include bias
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        curvature: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.curvature = curvature
        
        # Weight matrix (operates in tangent space)
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.01)
        
        if bias:
            # Bias is a point in hyperbolic space
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply hyperbolic linear transformation.
        
        Args:
            x: Hyperbolic input [..., in_dim]
            
        Returns:
            Transformed hyperbolic output [..., out_dim]
        """
        # Map to tangent space at origin
        v = _log_map(x, torch.zeros_like(x), self.curvature)
        
        # Apply linear transformation in tangent space
        v_transformed = F.linear(v, self.weight)
        
        # Map back to hyperbolic space
        out = _exp_map(v_transformed, torch.zeros(v_transformed.shape[-1], device=x.device), self.curvature)
        
        # Add bias via Mobius addition if present
        if self.bias is not None:
            bias_point = _clamp_norm(self.bias.unsqueeze(0))
            out = _mobius_add(out, bias_point, self.curvature)
        
        return out
    
    def extra_repr(self) -> str:
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}, bias={self.bias is not None}"


class HyperbolicClauseVoting(nn.Module):
    """
    Hyperbolic voting mechanism for Tsetlin Machine classification.
    
    Replaces traditional linear voting with hyperbolic distance-based
    classification. Each class is represented by a learnable prototype
    in hyperbolic space, and predictions are based on distances to
    these prototypes.
    
    This is particularly effective for hierarchical classification where
    semantic relationships between classes should be preserved.
    
    Args:
        n_clauses: Number of clauses
        n_classes: Number of output classes
        embed_dim: Hyperbolic embedding dimension
        curvature: Curvature parameter
        temperature: Temperature for softmax over distances
        use_projection: Whether to project clause outputs before voting
        margin: Margin for hierarchical separation
    """
    
    def __init__(
        self,
        n_clauses: int,
        n_classes: int,
        embed_dim: int = 64,
        curvature: float = 1.0,
        temperature: float = 1.0,
        use_projection: bool = True,
        margin: float = 0.1,
    ):
        super().__init__()
        self.n_clauses = n_clauses
        self.n_classes = n_classes
        self.embed_dim = embed_dim
        self.curvature = curvature
        self.temperature = temperature
        self.margin = margin
        
        # Clause projection to hyperbolic space
        if use_projection:
            self.clause_proj = PoincareBallProjection(
                in_dim=n_clauses,
                out_dim=embed_dim,
                curvature=curvature,
            )
        else:
            self.clause_proj = None
            assert n_clauses == embed_dim, "Without projection, n_clauses must equal embed_dim"
        
        # Class prototypes in hyperbolic space
        # Initialize near origin for general concepts
        self.class_prototypes = nn.Parameter(
            torch.randn(n_classes, embed_dim) * 0.1
        )
        
        # Optional learnable temperature
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))
        
        # Distance module
        self.dist_fn = HyperbolicDistance(curvature=curvature)
    
    @property
    def effective_temperature(self) -> float:
        return torch.exp(self.log_temperature).item()
    
    def forward(
        self,
        clause_outputs: torch.Tensor,
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute class logits via hyperbolic distances.
        
        Args:
            clause_outputs: Clause activation tensor [batch, n_clauses]
            return_embeddings: Whether to also return hyperbolic embeddings
            
        Returns:
            Class logits [batch, n_classes] (negative distances)
            Optionally also returns embeddings [batch, embed_dim]
        """
        # Project to hyperbolic space
        if self.clause_proj is not None:
            embeddings = self.clause_proj(clause_outputs)
        else:
            embeddings = _clamp_norm(clause_outputs)
        
        # Normalize class prototypes
        prototypes = _clamp_norm(self.class_prototypes)
        
        # Compute distances to all class prototypes
        # embeddings: [batch, embed_dim] -> [batch, 1, embed_dim]
        # prototypes: [n_classes, embed_dim] -> [1, n_classes, embed_dim]
        emb_exp = embeddings.unsqueeze(1)
        proto_exp = prototypes.unsqueeze(0)
        
        distances = _poincare_distance(emb_exp, proto_exp, self.curvature)
        distances = distances.squeeze(1)  # [batch, n_classes]
        
        # Convert distances to logits (negative distance = higher similarity)
        temp = torch.exp(self.log_temperature)
        logits = -distances / temp
        
        if return_embeddings:
            return logits, embeddings
        return logits
    
    def hierarchical_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        parent_map: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Compute hierarchical regularization loss.
        
        Encourages parent classes to be closer to origin (more general)
        and child classes to be further out (more specific).
        
        Args:
            embeddings: Sample embeddings [batch, embed_dim]
            labels: Class labels [batch]
            parent_map: Dict mapping child class idx to parent class idx
            
        Returns:
            Hierarchical regularization loss
        """
        if parent_map is None:
            return torch.tensor(0.0, device=embeddings.device)
        
        prototypes = _clamp_norm(self.class_prototypes)
        loss = torch.tensor(0.0, device=embeddings.device)
        
        for child_idx, parent_idx in parent_map.items():
            child_proto = prototypes[child_idx]
            parent_proto = prototypes[parent_idx]
            
            # Child should have larger norm than parent
            child_norm = child_proto.norm()
            parent_norm = parent_proto.norm()
            
            # Margin-based loss
            loss = loss + F.relu(parent_norm - child_norm + self.margin)
        
        return loss
    
    def extra_repr(self) -> str:
        return (
            f"n_clauses={self.n_clauses}, n_classes={self.n_classes}, "
            f"embed_dim={self.embed_dim}, curvature={self.curvature}"
        )


class HyperbolicClauseAggregator(nn.Module):
    """
    Aggregates clause outputs in hyperbolic space.
    
    Uses hyperbolic centroid or attention-weighted combination
    to aggregate multiple clause activations.
    
    Args:
        n_clauses: Number of clauses
        embed_dim: Hyperbolic embedding dimension
        curvature: Curvature parameter
        aggregation: Aggregation method ('centroid', 'attention', 'weighted')
    """
    
    def __init__(
        self,
        n_clauses: int,
        embed_dim: int = 64,
        curvature: float = 1.0,
        aggregation: str = "attention",
    ):
        super().__init__()
        self.n_clauses = n_clauses
        self.embed_dim = embed_dim
        self.curvature = curvature
        self.aggregation = aggregation
        
        # Project each clause to its own embedding
        self.clause_embeddings = nn.Parameter(
            torch.randn(n_clauses, embed_dim) * 0.1
        )
        
        if aggregation == "attention":
            self.attention = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.Tanh(),
                nn.Linear(embed_dim, 1),
            )
    
    def _hyperbolic_centroid(
        self,
        points: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute weighted centroid in hyperbolic space.
        
        Uses iterative algorithm to find the Frechet mean.
        
        Args:
            points: Points in Poincare ball [batch, n, d]
            weights: Optional weights [batch, n]
            
        Returns:
            Hyperbolic centroid [batch, d]
        """
        batch_size, n_points, dim = points.shape
        
        if weights is None:
            weights = torch.ones(batch_size, n_points, device=points.device) / n_points
        else:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        
        # Initialize at Euclidean weighted mean
        centroid = (points * weights.unsqueeze(-1)).sum(dim=1)
        centroid = _clamp_norm(centroid)
        
        # Iterative refinement (3-5 iterations usually sufficient)
        for _ in range(5):
            # Log map: project points to tangent space at centroid
            tangent_vectors = _log_map(points, centroid.unsqueeze(1), self.curvature)
            
            # Weighted average in tangent space
            weighted_avg = (tangent_vectors * weights.unsqueeze(-1)).sum(dim=1)
            
            # Exp map: project back to manifold
            centroid = _exp_map(weighted_avg, centroid, self.curvature)
            centroid = _clamp_norm(centroid)
        
        return centroid
    
    def forward(self, clause_outputs: torch.Tensor) -> torch.Tensor:
        """
        Aggregate clause outputs in hyperbolic space.
        
        Args:
            clause_outputs: Clause activations [batch, n_clauses]
            
        Returns:
            Aggregated hyperbolic embedding [batch, embed_dim]
        """
        batch_size = clause_outputs.shape[0]
        
        # Get clause embeddings in hyperbolic space
        embeddings = _clamp_norm(self.clause_embeddings)  # [n_clauses, embed_dim]
        embeddings = embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Use clause outputs as weights
        if self.aggregation == "weighted":
            # Direct weighting by clause strength
            weights = F.softmax(clause_outputs, dim=-1)
        elif self.aggregation == "attention":
            # Attention-based weighting
            attn_scores = self.attention(embeddings).squeeze(-1)  # [batch, n_clauses]
            attn_scores = attn_scores * clause_outputs  # Modulate by clause strength
            weights = F.softmax(attn_scores, dim=-1)
        else:
            # Uniform weighting (centroid)
            weights = None
        
        # Compute hyperbolic centroid
        return self._hyperbolic_centroid(embeddings, weights)


class HyperbolicSTCM(nn.Module):
    """
    STCM with Hyperbolic Voting Layer.
    
    Wraps a base STCM model and replaces its voting mechanism
    with hyperbolic distance-based classification.
    
    Args:
        base_tm: Base Tsetlin Machine module (e.g., FuzzyPatternTM_STCM)
        embed_dim: Hyperbolic embedding dimension
        curvature: Curvature parameter
        temperature: Temperature for distance-based softmax
    """
    
    def __init__(
        self,
        base_tm: nn.Module,
        embed_dim: int = 64,
        curvature: float = 1.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.base_tm = base_tm
        
        # Extract TM parameters
        n_clauses = base_tm.n_clauses
        n_classes = base_tm.n_classes
        
        # Replace voting with hyperbolic layer
        self.hyperbolic_voting = HyperbolicClauseVoting(
            n_clauses=n_clauses,
            n_classes=n_classes,
            embed_dim=embed_dim,
            curvature=curvature,
            temperature=temperature,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        use_ste: bool = True,
        skip_norm: bool = False,
        return_embeddings: bool = False,
    ):
        """
        Forward pass with hyperbolic voting.
        
        Args:
            x: Input tensor
            use_ste: Whether to use STE for base TM
            skip_norm: Whether to skip input normalization
            return_embeddings: Whether to return hyperbolic embeddings
            
        Returns:
            logits, clause_outputs, (optional) embeddings
        """
        # Get clause outputs from base TM
        _, clause_outputs = self.base_tm(x, use_ste=use_ste, skip_norm=skip_norm)
        
        # Apply hyperbolic voting
        if return_embeddings:
            logits, embeddings = self.hyperbolic_voting(clause_outputs, return_embeddings=True)
            return logits, clause_outputs, embeddings
        else:
            logits = self.hyperbolic_voting(clause_outputs)
            return logits, clause_outputs


# =============================================================================
# Utility Functions
# =============================================================================


def hyperbolic_distance_matrix(
    x: torch.Tensor,
    curvature: float = 1.0,
) -> torch.Tensor:
    """
    Compute pairwise hyperbolic distance matrix.
    
    Args:
        x: Points in Poincare ball [n, d]
        curvature: Curvature parameter
        
    Returns:
        Distance matrix [n, n]
    """
    n = x.shape[0]
    x1 = x.unsqueeze(1).expand(n, n, -1)
    x2 = x.unsqueeze(0).expand(n, n, -1)
    return _poincare_distance(x1, x2, curvature)


def initialize_hierarchical_prototypes(
    n_classes: int,
    embed_dim: int,
    hierarchy: Optional[dict] = None,
    curvature: float = 1.0,
) -> torch.Tensor:
    """
    Initialize class prototypes respecting hierarchy.
    
    Places parent classes near origin and children progressively
    further out along geodesics from their parents.
    
    Args:
        n_classes: Number of classes
        embed_dim: Embedding dimension
        hierarchy: Dict mapping child -> parent indices
        curvature: Curvature parameter
        
    Returns:
        Initialized prototypes [n_classes, embed_dim]
    """
    if hierarchy is None:
        # Uniform initialization
        prototypes = torch.randn(n_classes, embed_dim) * 0.1
    else:
        prototypes = torch.zeros(n_classes, embed_dim)
        
        # Find root classes (no parent)
        children = set(hierarchy.keys())
        parents = set(hierarchy.values())
        roots = parents - children
        
        # Initialize roots near origin
        for root_idx in roots:
            prototypes[root_idx] = torch.randn(embed_dim) * 0.05
        
        # BFS to initialize children
        visited = set(roots)
        queue = list(roots)
        
        while queue:
            parent_idx = queue.pop(0)
            parent_proto = prototypes[parent_idx]
            
            # Find children of this parent
            for child_idx, p_idx in hierarchy.items():
                if p_idx == parent_idx and child_idx not in visited:
                    # Place child along random direction from parent
                    direction = torch.randn(embed_dim)
                    direction = direction / direction.norm()
                    
                    # Move along geodesic from parent
                    step = direction * 0.2  # Step size
                    child_proto = _exp_map(step, parent_proto, curvature)
                    prototypes[child_idx] = child_proto
                    
                    visited.add(child_idx)
                    queue.append(child_idx)
    
    return _clamp_norm(prototypes)



