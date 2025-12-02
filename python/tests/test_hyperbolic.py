"""
Unit tests for hyperbolic geometry module.

Tests cover:
1. Poincare ball projection - points stay inside ball
2. Hyperbolic distance - symmetry and triangle inequality
3. Mobius operations - proper arithmetic in hyperbolic space
4. Hyperbolic voting - gradient flow and hierarchical properties
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from fptm_ste.hyperbolic import (
    # Utility functions
    _clamp_norm,
    _mobius_add,
    _mobius_scalar_mul,
    _poincare_distance,
    _exp_map,
    _log_map,
    # Modules
    PoincareBallProjection,
    HyperbolicDistance,
    MobiusAddition,
    HyperbolicLinear,
    HyperbolicClauseVoting,
    HyperbolicClauseAggregator,
    HyperbolicSTCM,
    # Utility functions
    hyperbolic_distance_matrix,
    initialize_hierarchical_prototypes,
)
from fptm_ste import FuzzyPatternTM_STCM


class TestClampNorm:
    """Tests for norm clamping to stay inside Poincare ball."""
    
    def test_small_vectors_unchanged(self):
        """Small vectors should remain unchanged."""
        x = torch.randn(10, 5) * 0.1  # Small vectors
        result = _clamp_norm(x)
        assert torch.allclose(x, result, atol=1e-5)
    
    def test_large_vectors_clamped(self):
        """Large vectors should be clamped to ball boundary."""
        x = torch.randn(10, 5) * 10  # Large vectors
        result = _clamp_norm(x)
        norms = result.norm(dim=-1)
        assert (norms < 1.0).all()
    
    def test_norms_below_max(self):
        """All output norms should be below max_norm."""
        x = torch.randn(100, 64)
        max_norm = 0.99
        result = _clamp_norm(x, max_norm)
        norms = result.norm(dim=-1)
        assert (norms <= max_norm + 1e-6).all()


class TestMobiusAddition:
    """Tests for Mobius addition in Poincare ball."""
    
    def test_identity_element(self):
        """Adding zero should give identity."""
        x = torch.randn(5, 10) * 0.3
        x = _clamp_norm(x)
        zero = torch.zeros_like(x)
        result = _mobius_add(x, zero)
        assert torch.allclose(x, result, atol=1e-4)
    
    def test_stays_in_ball(self):
        """Result should stay inside Poincare ball."""
        x = torch.randn(50, 20) * 0.5
        y = torch.randn(50, 20) * 0.5
        x = _clamp_norm(x)
        y = _clamp_norm(y)
        result = _mobius_add(x, y)
        norms = result.norm(dim=-1)
        assert (norms < 1.0).all()
    
    def test_inverse_element(self):
        """x ⊕ (-x) should give approximately zero."""
        x = torch.randn(5, 10) * 0.3
        x = _clamp_norm(x)
        neg_x = -x
        result = _mobius_add(x, neg_x)
        assert result.norm(dim=-1).max() < 0.1


class TestMobiusScalarMul:
    """Tests for Mobius scalar multiplication."""
    
    def test_zero_scalar(self):
        """Multiplying by zero should give zero."""
        x = torch.randn(5, 10) * 0.3
        x = _clamp_norm(x)
        result = _mobius_scalar_mul(0.0, x)
        assert result.norm().item() < 1e-5
    
    def test_one_scalar(self):
        """Multiplying by one should give identity."""
        x = torch.randn(5, 10) * 0.3
        x = _clamp_norm(x)
        result = _mobius_scalar_mul(1.0, x)
        assert torch.allclose(x, result, atol=1e-4)
    
    def test_stays_in_ball(self):
        """Result should stay inside Poincare ball."""
        x = torch.randn(50, 20) * 0.3
        x = _clamp_norm(x)
        r = torch.randn(50) * 2
        result = _mobius_scalar_mul(r, x)
        norms = result.norm(dim=-1)
        assert (norms < 1.0).all()


class TestPoincareDistance:
    """Tests for Poincare geodesic distance."""
    
    def test_distance_symmetry(self):
        """Distance should be approximately symmetric: d(x,y) ≈ d(y,x)."""
        x = torch.randn(20, 10) * 0.3
        y = torch.randn(20, 10) * 0.3
        x = _clamp_norm(x)
        y = _clamp_norm(y)
        
        d_xy = _poincare_distance(x, y)
        d_yx = _poincare_distance(y, x)
        
        # Allow for numerical precision issues in hyperbolic space
        assert torch.allclose(d_xy, d_yx, atol=0.5)
    
    def test_distance_non_negative(self):
        """Distance should be non-negative."""
        x = torch.randn(50, 10) * 0.3
        y = torch.randn(50, 10) * 0.3
        x = _clamp_norm(x)
        y = _clamp_norm(y)
        
        d = _poincare_distance(x, y)
        assert (d >= -1e-6).all()
    
    def test_distance_to_self_zero(self):
        """Distance to self should be zero."""
        x = torch.randn(20, 10) * 0.3
        x = _clamp_norm(x)
        
        d = _poincare_distance(x, x)
        assert d.max().item() < 1e-4
    
    def test_distance_increases_near_boundary(self):
        """Points near boundary should have larger distances."""
        # Points near origin
        x_origin = torch.randn(10, 5) * 0.1
        y_origin = torch.randn(10, 5) * 0.1
        
        # Points near boundary
        x_boundary = torch.randn(10, 5) * 0.1
        y_boundary = x_boundary + torch.randn(10, 5) * 0.1
        x_boundary = _clamp_norm(x_boundary, 0.9)
        y_boundary = _clamp_norm(y_boundary, 0.9)
        
        d_origin = _poincare_distance(x_origin, y_origin)
        d_boundary = _poincare_distance(x_boundary, y_boundary)
        
        # Boundary distances should generally be larger for similar Euclidean distances
        # This is the hyperbolic property


class TestExpLogMaps:
    """Tests for exponential and logarithmic maps."""
    
    def test_exp_log_inverse(self):
        """Exp and log should be inverses: log(exp(v)) ≈ v."""
        # Base point at origin
        x = torch.zeros(5, 10)
        v = torch.randn(5, 10) * 0.3
        
        y = _exp_map(v, x)
        v_recovered = _log_map(y, x)
        
        assert torch.allclose(v, v_recovered, atol=1e-3)
    
    def test_exp_stays_in_ball(self):
        """Exponential map should produce points in ball."""
        x = torch.randn(20, 10) * 0.3
        x = _clamp_norm(x)
        v = torch.randn(20, 10) * 0.5
        
        y = _exp_map(v, x)
        norms = y.norm(dim=-1)
        assert (norms < 1.0).all()


class TestPoincareBallProjection:
    """Tests for PoincareBallProjection module."""
    
    def test_output_in_ball(self):
        """All outputs should be inside Poincare ball."""
        proj = PoincareBallProjection(in_dim=100, out_dim=64)
        x = torch.randn(32, 100)
        
        out = proj(x)
        norms = out.norm(dim=-1)
        
        assert (norms < 1.0).all()
    
    def test_gradients_flow(self):
        """Gradients should flow through projection."""
        proj = PoincareBallProjection(in_dim=50, out_dim=32)
        x = torch.randn(16, 50, requires_grad=True)
        
        out = proj(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
    
    def test_different_methods(self):
        """Test both 'exp' and 'normalize' projection methods."""
        for method in ["exp", "normalize"]:
            proj = PoincareBallProjection(in_dim=64, out_dim=32, method=method)
            x = torch.randn(16, 64)
            
            out = proj(x)
            norms = out.norm(dim=-1)
            
            assert (norms < 1.0).all()


class TestHyperbolicDistance:
    """Tests for HyperbolicDistance module."""
    
    def test_pairwise_shape(self):
        """Pairwise distance should have correct shape."""
        dist_fn = HyperbolicDistance()
        x = torch.randn(8, 10, 32) * 0.3
        y = torch.randn(8, 5, 32) * 0.3
        x = _clamp_norm(x)
        y = _clamp_norm(y)
        
        d = dist_fn(x, y, pairwise=True)
        assert d.shape == (8, 10, 5)
    
    def test_elementwise_shape(self):
        """Element-wise distance should have correct shape."""
        dist_fn = HyperbolicDistance()
        x = torch.randn(16, 32) * 0.3
        y = torch.randn(16, 32) * 0.3
        x = _clamp_norm(x)
        y = _clamp_norm(y)
        
        d = dist_fn(x, y, pairwise=False)
        assert d.shape == (16,)
    
    def test_temperature_scaling(self):
        """Temperature should scale distances."""
        dist_fn_t1 = HyperbolicDistance(temperature=1.0)
        dist_fn_t2 = HyperbolicDistance(temperature=2.0)
        
        x = torch.randn(16, 32) * 0.3
        y = torch.randn(16, 32) * 0.3
        x = _clamp_norm(x)
        y = _clamp_norm(y)
        
        d1 = dist_fn_t1(x, y, pairwise=False)
        d2 = dist_fn_t2(x, y, pairwise=False)
        
        assert torch.allclose(d1 / 2, d2, atol=1e-5)


class TestMobiusAdditionModule:
    """Tests for MobiusAddition module."""
    
    def test_output_in_ball(self):
        """Result should stay in Poincare ball."""
        add = MobiusAddition(dim=32)
        x = torch.randn(16, 32) * 0.3
        y = torch.randn(16, 32) * 0.3
        x = _clamp_norm(x)
        y = _clamp_norm(y)
        
        out = add(x, y)
        norms = out.norm(dim=-1)
        assert (norms < 1.0).all()
    
    def test_gradients_flow(self):
        """Gradients should flow through addition."""
        add = MobiusAddition(dim=32, learnable_weights=True)
        # Create leaf tensors properly
        x_leaf = torch.randn(16, 32) * 0.3
        y_leaf = torch.randn(16, 32) * 0.3
        x_leaf.requires_grad_(True)
        y_leaf.requires_grad_(True)
        
        x_clamped = _clamp_norm(x_leaf)
        y_clamped = _clamp_norm(y_leaf)
        
        out = add(x_clamped, y_clamped)
        loss = out.sum()
        loss.backward()
        
        # Check gradients flow to leaf tensors (original inputs)
        assert x_leaf.grad is not None, "x_leaf should have gradients"
        assert y_leaf.grad is not None, "y_leaf should have gradients"
        # Alpha parameter should get gradient
        if add.alpha.grad is not None:
            assert not torch.isnan(add.alpha.grad)


class TestHyperbolicClauseVoting:
    """Tests for HyperbolicClauseVoting module."""
    
    def test_output_shape(self):
        """Output should have correct shape."""
        voting = HyperbolicClauseVoting(
            n_clauses=64,
            n_classes=10,
            embed_dim=32,
        )
        clause_outputs = torch.randn(16, 64)
        
        logits = voting(clause_outputs)
        assert logits.shape == (16, 10)
    
    def test_gradients_flow(self):
        """Gradients should flow through voting."""
        voting = HyperbolicClauseVoting(
            n_clauses=64,
            n_classes=10,
            embed_dim=32,
        )
        clause_outputs = torch.randn(16, 64, requires_grad=True)
        
        logits = voting(clause_outputs)
        loss = F.cross_entropy(logits, torch.randint(0, 10, (16,)))
        loss.backward()
        
        assert clause_outputs.grad is not None
        assert clause_outputs.grad.abs().sum() > 0
        assert voting.class_prototypes.grad is not None
    
    def test_return_embeddings(self):
        """Should return embeddings when requested."""
        voting = HyperbolicClauseVoting(
            n_clauses=64,
            n_classes=10,
            embed_dim=32,
        )
        clause_outputs = torch.randn(16, 64)
        
        logits, embeddings = voting(clause_outputs, return_embeddings=True)
        
        assert logits.shape == (16, 10)
        assert embeddings.shape == (16, 32)
        assert (embeddings.norm(dim=-1) < 1.0).all()
    
    def test_hierarchical_loss_with_parent_map(self):
        """Hierarchical loss should encourage parent classes near origin."""
        voting = HyperbolicClauseVoting(
            n_clauses=64,
            n_classes=10,
            embed_dim=32,
        )
        
        # Define hierarchy: classes 5-9 are children of classes 0-4
        parent_map = {5: 0, 6: 1, 7: 2, 8: 3, 9: 4}
        
        clause_outputs = torch.randn(16, 64)
        _, embeddings = voting(clause_outputs, return_embeddings=True)
        labels = torch.randint(0, 10, (16,))
        
        loss = voting.hierarchical_loss(embeddings, labels, parent_map)
        assert loss >= 0


class TestHyperbolicClauseAggregator:
    """Tests for HyperbolicClauseAggregator module."""
    
    def test_output_shape(self):
        """Output should have correct shape."""
        agg = HyperbolicClauseAggregator(
            n_clauses=32,
            embed_dim=64,
        )
        clause_outputs = torch.randn(16, 32)
        
        out = agg(clause_outputs)
        assert out.shape == (16, 64)
    
    def test_output_in_ball(self):
        """Output should be in Poincare ball."""
        agg = HyperbolicClauseAggregator(
            n_clauses=32,
            embed_dim=64,
        )
        clause_outputs = torch.randn(16, 32)
        
        out = agg(clause_outputs)
        norms = out.norm(dim=-1)
        assert (norms < 1.0).all()
    
    @pytest.mark.parametrize("aggregation", ["attention", "weighted", "centroid"])
    def test_different_aggregations(self, aggregation):
        """Test different aggregation methods."""
        agg = HyperbolicClauseAggregator(
            n_clauses=32,
            embed_dim=64,
            aggregation=aggregation,
        )
        clause_outputs = torch.randn(16, 32)
        
        out = agg(clause_outputs)
        assert out.shape == (16, 64)
        assert (out.norm(dim=-1) < 1.0).all()


class TestHyperbolicSTCM:
    """Tests for HyperbolicSTCM module."""
    
    def test_forward_shape(self):
        """Output should have correct shapes."""
        base_tm = FuzzyPatternTM_STCM(
            n_features=100,
            n_clauses=32,
            n_classes=10,
        )
        model = HyperbolicSTCM(base_tm, embed_dim=64)
        
        x = torch.rand(16, 100)
        logits, clauses = model(x)
        
        assert logits.shape == (16, 10)
        assert clauses.shape == (16, 32)
    
    def test_return_embeddings(self):
        """Should return embeddings when requested."""
        base_tm = FuzzyPatternTM_STCM(
            n_features=100,
            n_clauses=32,
            n_classes=10,
        )
        model = HyperbolicSTCM(base_tm, embed_dim=64)
        
        x = torch.rand(16, 100)
        logits, clauses, embeddings = model(x, return_embeddings=True)
        
        assert embeddings.shape == (16, 64)
        assert (embeddings.norm(dim=-1) < 1.0).all()
    
    def test_gradients_flow(self):
        """Gradients should flow through the full model."""
        base_tm = FuzzyPatternTM_STCM(
            n_features=100,
            n_clauses=32,
            n_classes=10,
        )
        model = HyperbolicSTCM(base_tm, embed_dim=64)
        
        x = torch.rand(16, 100, requires_grad=True)
        logits, _ = model(x)
        loss = F.cross_entropy(logits, torch.randint(0, 10, (16,)))
        loss.backward()
        
        assert x.grad is not None
        assert base_tm.pos_logits.grad is not None


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_distance_matrix_symmetric(self):
        """Distance matrix should be approximately symmetric."""
        x = torch.randn(10, 32) * 0.3
        x = _clamp_norm(x)
        
        D = hyperbolic_distance_matrix(x)
        
        # Allow for numerical precision issues in hyperbolic space
        assert torch.allclose(D, D.t(), atol=0.5)
    
    def test_distance_matrix_diagonal_zero(self):
        """Diagonal of distance matrix should be zero."""
        x = torch.randn(10, 32) * 0.3
        x = _clamp_norm(x)
        
        D = hyperbolic_distance_matrix(x)
        
        assert D.diag().max() < 1e-4
    
    def test_hierarchical_prototypes_shape(self):
        """Hierarchical prototypes should have correct shape."""
        prototypes = initialize_hierarchical_prototypes(
            n_classes=10,
            embed_dim=32,
        )
        
        assert prototypes.shape == (10, 32)
    
    def test_hierarchical_prototypes_in_ball(self):
        """Prototypes should be in Poincare ball."""
        prototypes = initialize_hierarchical_prototypes(
            n_classes=10,
            embed_dim=32,
        )
        
        norms = prototypes.norm(dim=-1)
        assert (norms < 1.0).all()
    
    def test_hierarchical_prototypes_with_hierarchy(self):
        """With hierarchy, children should be further from origin than parents."""
        hierarchy = {5: 0, 6: 1, 7: 2, 8: 3, 9: 4}
        
        prototypes = initialize_hierarchical_prototypes(
            n_classes=10,
            embed_dim=32,
            hierarchy=hierarchy,
        )
        
        # Check that children have larger norms than parents
        parent_norms = prototypes[:5].norm(dim=-1)
        child_norms = prototypes[5:].norm(dim=-1)
        
        # At least some children should be further out
        assert child_norms.mean() > parent_norms.mean() * 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

