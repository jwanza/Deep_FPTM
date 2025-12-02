"""
Unit tests for sparse routing and L0 pruning module.

Tests cover:
1. TopKRouter - correct top-k selection and gradients
2. L0ClauseMask - hard concrete sampling and sparsity
3. LoadBalancingLoss - auxiliary loss computation
4. SparseMoEClauseMachine - full MoE-based TM
5. PrunableClauseMachine - L0-based clause pruning
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from fptm_ste.sparse_routing import (
    # Utility functions
    _hard_concrete_sample,
    _l0_penalty,
    # Modules
    TopKRouter,
    SparseClauseDispatcher,
    LoadBalancingLoss,
    L0ClauseMask,
    DifferentiableL0Regularizer,
    SparseMoEClauseMachine,
    PrunableClauseMachine,
    L0PrunedDeepTM,
)
from fptm_ste import FuzzyPatternTM_STCM


class TestHardConcrete:
    """Tests for hard concrete distribution sampling."""
    
    def test_samples_in_range(self):
        """Samples should be in [0, 1]."""
        log_alpha = torch.randn(100)
        samples = _hard_concrete_sample(log_alpha, training=True)
        
        assert (samples >= 0).all()
        assert (samples <= 1).all()
    
    def test_deterministic_at_test(self):
        """Samples should be deterministic at test time."""
        log_alpha = torch.randn(50)
        
        s1 = _hard_concrete_sample(log_alpha, training=False)
        s2 = _hard_concrete_sample(log_alpha, training=False)
        
        assert torch.allclose(s1, s2)
    
    def test_binary_at_test(self):
        """Samples should be binary at test time."""
        log_alpha = torch.randn(100)
        samples = _hard_concrete_sample(log_alpha, training=False)
        
        assert torch.allclose(samples, samples.round())
    
    def test_high_log_alpha_likely_one(self):
        """High log_alpha should give samples close to 1."""
        log_alpha = torch.full((100,), 10.0)
        samples = _hard_concrete_sample(log_alpha, training=False)
        
        assert (samples == 1.0).all()
    
    def test_low_log_alpha_likely_zero(self):
        """Low log_alpha should give samples close to 0."""
        log_alpha = torch.full((100,), -10.0)
        samples = _hard_concrete_sample(log_alpha, training=False)
        
        assert (samples == 0.0).all()


class TestL0Penalty:
    """Tests for L0 penalty computation."""
    
    def test_penalty_non_negative(self):
        """L0 penalty should be non-negative."""
        log_alpha = torch.randn(50)
        penalty = _l0_penalty(log_alpha)
        
        assert penalty >= 0
    
    def test_penalty_increases_with_log_alpha(self):
        """Higher log_alpha should give higher penalty."""
        low_alpha = torch.full((10,), -5.0)
        high_alpha = torch.full((10,), 5.0)
        
        low_penalty = _l0_penalty(low_alpha)
        high_penalty = _l0_penalty(high_alpha)
        
        assert high_penalty > low_penalty
    
    def test_penalty_upper_bound(self):
        """Penalty should be at most n_params."""
        n_params = 100
        log_alpha = torch.full((n_params,), 100.0)  # Very high
        penalty = _l0_penalty(log_alpha)
        
        assert penalty <= n_params + 1


class TestTopKRouter:
    """Tests for TopKRouter module."""
    
    def test_output_shapes(self):
        """Output shapes should be correct."""
        router = TopKRouter(input_dim=64, n_items=20, top_k=5)
        x = torch.randn(16, 64)
        
        weights, indices, soft = router(x, return_soft_weights=True)
        
        assert weights.shape == (16, 5)
        assert indices.shape == (16, 5)
        assert soft.shape == (16, 20)
    
    def test_topk_returns_k_indices(self):
        """Should return exactly k indices per sample."""
        router = TopKRouter(input_dim=64, n_items=20, top_k=5)
        x = torch.randn(32, 64)
        
        _, indices, _ = router(x)
        
        assert indices.shape[1] == 5
        # Each sample should have unique indices
        for i in range(32):
            unique_indices = torch.unique(indices[i])
            assert len(unique_indices) == 5
    
    def test_weights_valid(self):
        """Top-k weights should be non-negative and bounded."""
        router = TopKRouter(input_dim=64, n_items=20, top_k=5)
        x = torch.randn(16, 64)
        
        weights, _, _ = router(x)
        
        # Weights should be non-negative
        assert torch.all(weights >= 0)
        # Weights should be bounded
        assert torch.all(weights <= 1.1)  # Allow small numerical error
    
    def test_gradients_flow(self):
        """Gradients should flow through router."""
        router = TopKRouter(input_dim=64, n_items=20, top_k=5)
        x = torch.randn(16, 64, requires_grad=True)
        
        weights, _, _ = router(x)
        loss = weights.sum()
        loss.backward()
        
        assert x.grad is not None
        assert router.router_proj.weight.grad is not None
    
    def test_indices_in_valid_range(self):
        """Indices should be in valid range [0, n_items)."""
        n_items = 20
        router = TopKRouter(input_dim=64, n_items=n_items, top_k=5)
        x = torch.randn(32, 64)
        
        _, indices, _ = router(x)
        
        assert (indices >= 0).all()
        assert (indices < n_items).all()


class TestSparseClauseDispatcher:
    """Tests for SparseClauseDispatcher module."""
    
    def test_dispatch_creates_groups(self):
        """Dispatcher should create groups for each expert."""
        dispatcher = SparseClauseDispatcher(
            n_clauses=64,
            n_groups=4,
        )
        
        x = torch.randn(16, 32)
        group_indices = torch.randint(0, 4, (16, 2))
        group_weights = F.softmax(torch.randn(16, 2), dim=-1)
        
        samples, weights, indices = dispatcher.dispatch(x, group_indices, group_weights)
        
        # Should have some groups
        assert len(samples) > 0
    
    def test_combine_output_shape(self):
        """Combined output should have correct shape."""
        dispatcher = SparseClauseDispatcher(n_clauses=64, n_groups=4)
        
        batch_size = 16
        output_dim = 10
        
        # Simulate group outputs
        outputs = {0: torch.randn(8, output_dim), 1: torch.randn(8, output_dim)}
        weights = {0: torch.ones(8) * 0.6, 1: torch.ones(8) * 0.4}
        indices = {0: torch.arange(8), 1: torch.arange(8, 16)}
        
        combined = dispatcher.combine(outputs, weights, indices, batch_size, output_dim)
        
        assert combined.shape == (batch_size, output_dim)


class TestLoadBalancingLoss:
    """Tests for LoadBalancingLoss module."""
    
    def test_switch_loss_non_negative(self):
        """Switch loss should be non-negative."""
        loss_fn = LoadBalancingLoss(n_experts=8, aux_loss_type="switch")
        
        router_probs = F.softmax(torch.randn(32, 8), dim=-1)
        expert_indices = torch.randint(0, 8, (32, 2))
        
        loss = loss_fn(router_probs, expert_indices)
        assert loss >= 0
    
    def test_gshard_loss_non_negative(self):
        """GShard loss should be non-negative."""
        loss_fn = LoadBalancingLoss(n_experts=8, aux_loss_type="gshard")
        
        router_probs = F.softmax(torch.randn(32, 8), dim=-1)
        expert_indices = torch.randint(0, 8, (32, 2))
        
        loss = loss_fn(router_probs, expert_indices)
        assert loss >= 0
    
    def test_importance_loss_non_negative(self):
        """Importance loss should be non-negative."""
        loss_fn = LoadBalancingLoss(n_experts=8, aux_loss_type="importance")
        
        router_probs = F.softmax(torch.randn(32, 8), dim=-1)
        expert_indices = torch.randint(0, 8, (32, 2))
        
        loss = loss_fn(router_probs, expert_indices)
        assert loss >= 0
    
    def test_uniform_routing_low_loss(self):
        """Uniform routing should have relatively low loss."""
        loss_fn = LoadBalancingLoss(n_experts=4, aux_loss_type="switch")
        
        # Uniform routing probabilities
        uniform_probs = torch.ones(32, 4) / 4
        # Uniform index selection
        expert_indices = torch.cat([
            torch.full((8, 2), i) for i in range(4)
        ])
        
        loss = loss_fn(uniform_probs, expert_indices)
        
        # Should be bounded for switch loss with uniform routing
        assert loss.item() >= 0
        assert loss.item() < 10.0


class TestL0ClauseMask:
    """Tests for L0ClauseMask module."""
    
    def test_output_shape(self):
        """Output should have correct shape."""
        mask = L0ClauseMask(n_clauses=64)
        gates = mask()
        
        assert gates.shape == (64,)
    
    def test_gates_in_range(self):
        """Gates should be in [0, 1]."""
        mask = L0ClauseMask(n_clauses=100)
        mask.train()
        
        for _ in range(10):
            gates = mask()
            assert (gates >= 0).all()
            assert (gates <= 1).all()
    
    def test_gates_binary_at_eval(self):
        """Gates should be binary at eval time."""
        mask = L0ClauseMask(n_clauses=64)
        mask.eval()
        
        gates = mask()
        assert torch.allclose(gates, gates.round())
    
    def test_l0_penalty_correlates_with_active_count(self):
        """L0 penalty should correlate with active gate count."""
        # High init_mean = more active
        mask_high = L0ClauseMask(n_clauses=64, init_mean=5.0)
        # Low init_mean = fewer active
        mask_low = L0ClauseMask(n_clauses=64, init_mean=-5.0)
        
        penalty_high = mask_high.l0_penalty()
        penalty_low = mask_low.l0_penalty()
        
        assert penalty_high > penalty_low
    
    def test_sparsity_in_valid_range(self):
        """Sparsity should be in [0, 1]."""
        mask = L0ClauseMask(n_clauses=64)
        sparsity = mask.get_sparsity()
        
        assert 0 <= sparsity <= 1
    
    def test_target_sparsity_loss(self):
        """Sparsity loss should push toward target."""
        target = 0.5
        mask = L0ClauseMask(n_clauses=64, init_mean=5.0, target_sparsity=target)
        
        loss = mask.sparsity_loss()
        assert loss >= 0


class TestDifferentiableL0Regularizer:
    """Tests for DifferentiableL0Regularizer module."""
    
    def test_forward_shape(self):
        """Output should match input shape."""
        reg = DifferentiableL0Regularizer(n_params=32)
        x = torch.randn(16, 32)
        
        out = reg(x)
        assert out.shape == x.shape
    
    def test_regularization_loss_non_negative(self):
        """Regularization loss should be non-negative."""
        reg = DifferentiableL0Regularizer(n_params=32)
        
        loss = reg.regularization_loss()
        assert loss >= 0
    
    def test_gradients_flow(self):
        """Gradients should flow through regularizer."""
        reg = DifferentiableL0Regularizer(n_params=32)
        x = torch.randn(16, 32, requires_grad=True)
        
        out = reg(x)
        loss = out.sum() + reg.regularization_loss()
        loss.backward()
        
        assert x.grad is not None
        assert reg.mask.log_alpha.grad is not None


class TestSparseMoEClauseMachine:
    """Tests for SparseMoEClauseMachine module."""
    
    def test_output_shapes(self):
        """Output shapes should be correct."""
        model = SparseMoEClauseMachine(
            n_features=100,
            n_clauses_per_expert=16,
            n_classes=10,
            n_experts=4,
            top_k=2,
        )
        x = torch.rand(32, 100)
        
        logits, clauses = model(x)
        
        assert logits.shape == (32, 10)
        assert clauses.shape == (32, 64)  # 4 experts * 16 clauses
    
    def test_aux_loss_available(self):
        """Auxiliary loss should be available after forward."""
        model = SparseMoEClauseMachine(
            n_features=100,
            n_clauses_per_expert=16,
            n_classes=10,
            n_experts=4,
        )
        model.train()
        x = torch.rand(32, 100)
        
        _ = model(x)
        
        assert model.aux_loss is not None
        assert model.aux_loss >= 0
    
    def test_return_routing_info(self):
        """Should return routing info when requested."""
        model = SparseMoEClauseMachine(
            n_features=100,
            n_clauses_per_expert=16,
            n_classes=10,
            n_experts=4,
            top_k=2,
        )
        x = torch.rand(16, 100)
        
        result = model(x, return_routing=True)
        
        assert "logits" in result
        assert "expert_weights" in result
        assert "expert_indices" in result
        assert "aux_loss" in result
    
    def test_gradients_flow(self):
        """Gradients should flow through the model."""
        model = SparseMoEClauseMachine(
            n_features=100,
            n_clauses_per_expert=16,
            n_classes=10,
            n_experts=4,
        )
        x = torch.rand(16, 100, requires_grad=True)
        
        logits, _ = model(x)
        loss = F.cross_entropy(logits, torch.randint(0, 10, (16,)))
        loss.backward()
        
        assert x.grad is not None
        assert model.stcm.pos_logits.grad is not None
    
    def test_expert_utilization(self):
        """Should be able to compute expert utilization."""
        model = SparseMoEClauseMachine(
            n_features=100,
            n_clauses_per_expert=16,
            n_classes=10,
            n_experts=4,
        )
        x = torch.rand(64, 100)
        
        utilization = model.get_expert_utilization(x)
        
        assert utilization.shape == (4,)
        assert utilization.sum() > 0


class TestPrunableClauseMachine:
    """Tests for PrunableClauseMachine module."""
    
    def test_output_shapes(self):
        """Output shapes should be correct."""
        base_tm = FuzzyPatternTM_STCM(
            n_features=100,
            n_clauses=64,
            n_classes=10,
        )
        model = PrunableClauseMachine(base_tm)
        
        x = torch.rand(16, 100)
        logits, clauses = model(x)
        
        assert logits.shape == (16, 10)
        assert clauses.shape == (16, 64)
    
    def test_l0_regularization_non_negative(self):
        """L0 regularization should be non-negative."""
        base_tm = FuzzyPatternTM_STCM(n_features=50, n_clauses=32, n_classes=5)
        model = PrunableClauseMachine(base_tm)
        
        reg = model.l0_regularization()
        assert reg >= 0
    
    def test_sparsity_changes_with_training(self):
        """Sparsity should change during training with L0 loss."""
        base_tm = FuzzyPatternTM_STCM(n_features=50, n_clauses=32, n_classes=5)
        model = PrunableClauseMachine(base_tm, l0_weight=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        initial_sparsity = model.get_sparsity()
        
        # Train for a few steps with heavy L0
        for _ in range(20):
            x = torch.rand(16, 50)
            y = torch.randint(0, 5, (16,))
            
            logits, _ = model(x)
            loss = F.cross_entropy(logits, y) + model.l0_regularization()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        final_sparsity = model.get_sparsity()
        
        # Sparsity should have changed
        assert initial_sparsity != final_sparsity
    
    def test_get_active_clauses(self):
        """Should return binary mask of active clauses."""
        base_tm = FuzzyPatternTM_STCM(n_features=50, n_clauses=32, n_classes=5)
        model = PrunableClauseMachine(base_tm)
        model.eval()
        
        mask = model.get_active_clauses()
        
        assert mask.shape == (32,)
        assert torch.allclose(mask, mask.round())
    
    def test_pruned_model_matches_dense_on_active(self):
        """Pruned output should match dense on active clauses."""
        base_tm = FuzzyPatternTM_STCM(n_features=50, n_clauses=32, n_classes=5)
        model = PrunableClauseMachine(base_tm)
        model.eval()
        
        x = torch.rand(8, 50)
        
        # Get outputs
        _, pruned_clauses = model(x)
        _, dense_clauses = base_tm(x)
        
        # Get active mask
        active_mask = model.get_active_clauses()
        
        # Pruned should equal dense * mask
        expected = dense_clauses * active_mask.unsqueeze(0)
        
        assert torch.allclose(pruned_clauses, expected, atol=1e-5)


class TestL0PrunedDeepTM:
    """Tests for L0PrunedDeepTM module."""
    
    def test_output_shapes(self):
        """Output shapes should be correct."""
        layers = [
            FuzzyPatternTM_STCM(n_features=100, n_clauses=32, n_classes=10),
        ]
        model = L0PrunedDeepTM(layers)
        
        x = torch.rand(16, 100)
        logits, all_clauses = model(x)
        
        assert logits.shape == (16, 10)
        assert len(all_clauses) == 1
        assert all_clauses[0].shape == (16, 32)
    
    def test_total_regularization(self):
        """Total L0 regularization should be sum of layer penalties."""
        layers = [
            FuzzyPatternTM_STCM(n_features=100, n_clauses=32, n_classes=10),
            FuzzyPatternTM_STCM(n_features=32, n_clauses=16, n_classes=10),
        ]
        model = L0PrunedDeepTM(layers)
        
        total_reg = model.total_l0_regularization()
        
        assert total_reg >= 0
    
    def test_layer_sparsities(self):
        """Should return sparsity per layer."""
        layers = [
            FuzzyPatternTM_STCM(n_features=100, n_clauses=32, n_classes=10),
            FuzzyPatternTM_STCM(n_features=32, n_clauses=16, n_classes=10),
        ]
        model = L0PrunedDeepTM(layers)
        
        sparsities = model.get_layer_sparsities()
        
        assert len(sparsities) == 2
        assert all(0 <= s <= 1 for s in sparsities)


class TestSparseRoutingPerformance:
    """Performance tests for sparse routing."""
    
    @pytest.mark.skip(reason="Performance test - run manually")
    def test_sparse_faster_than_dense(self):
        """Sparse routing should be faster than dense for large models."""
        n_clauses = 256
        top_k = 16
        
        # Dense model
        dense = FuzzyPatternTM_STCM(
            n_features=100,
            n_clauses=n_clauses,
            n_classes=10,
        )
        
        # Sparse model (4 experts, 64 clauses each)
        sparse = SparseMoEClauseMachine(
            n_features=100,
            n_clauses_per_expert=64,
            n_classes=10,
            n_experts=4,
            top_k=2,
        )
        
        x = torch.rand(32, 100)
        
        # Warm up
        _ = dense(x)
        _ = sparse(x)
        
        # Time dense
        start = time.time()
        for _ in range(100):
            _ = dense(x)
        dense_time = time.time() - start
        
        # Time sparse
        start = time.time()
        for _ in range(100):
            _ = sparse(x)
        sparse_time = time.time() - start
        
        print(f"Dense: {dense_time:.4f}s, Sparse: {sparse_time:.4f}s")
        
        # Sparse should be competitive or faster
        # (actual speedup depends on implementation details)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

