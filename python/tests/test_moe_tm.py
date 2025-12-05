"""
Unit tests for Sparse Mixture-of-Experts TM.

Tests cover routing, expert selection, load balancing, and gradient flow.
"""

import pytest
import torch
import torch.nn as nn

from fptm_ste.moe_tm import (
    MoEClauseRouter,
    ClauseExpert,
    SparseMoETM,
    BatchedSparseMoETM,
    HierarchicalMoETM,
    SwitchMoETM,
)


@pytest.fixture
def input_tensor():
    torch.manual_seed(42)
    return torch.rand(8, 64)


class TestMoEClauseRouter:
    """Test the routing mechanism."""
    
    def test_output_shapes(self, input_tensor):
        router = MoEClauseRouter(input_dim=64, n_experts=8, top_k=2)
        weights, indices, gates, aux = router(input_tensor)
        
        assert weights.shape == (8, 2)
        assert indices.shape == (8, 2)
        assert gates.shape == (8, 8)
        assert aux is not None
    
    def test_weights_normalized(self, input_tensor):
        router = MoEClauseRouter(input_dim=64, n_experts=8, top_k=3)
        weights, _, _, _ = router(input_tensor)
        
        # Top-k weights should sum to ~1
        assert torch.allclose(weights.sum(dim=1), torch.ones(8), atol=1e-5)
    
    def test_gates_sum_to_one(self, input_tensor):
        router = MoEClauseRouter(input_dim=64, n_experts=8, top_k=2)
        _, _, gates, _ = router(input_tensor)
        
        assert torch.allclose(gates.sum(dim=1), torch.ones(8), atol=1e-5)


class TestClauseExpert:
    """Test individual experts."""
    
    def test_output_shapes(self, input_tensor):
        expert = ClauseExpert(n_features=64, n_clauses=32, n_classes=10)
        logits, clauses = expert(input_tensor)
        
        assert logits.shape == (8, 10)
        assert clauses.shape == (8, 32)


class TestSparseMoETM:
    """Test main MoE TM."""
    
    def test_output_shapes(self, input_tensor):
        model = SparseMoETM(
            n_features=64,
            n_clauses_per_expert=16,
            n_classes=10,
            n_experts=4,
            top_k=2,
        )
        
        logits, clauses = model(input_tensor)
        assert logits.shape == (8, 10)
        assert clauses.shape[0] == 8
    
    def test_routing_info(self, input_tensor):
        model = SparseMoETM(
            n_features=64,
            n_clauses_per_expert=16,
            n_classes=10,
            n_experts=4,
        )
        
        output = model(input_tensor, return_routing=True)
        assert "logits" in output
        assert "expert_weights" in output
        assert "expert_indices" in output
        assert "aux_loss" in output
    
    def test_gradient_flow(self, input_tensor):
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        model = SparseMoETM(
            n_features=64,
            n_clauses_per_expert=16,
            n_classes=10,
            n_experts=4,
        )
        
        logits, _ = model(input_tensor)
        loss = logits.sum() + model.aux_loss
        loss.backward()
        
        assert input_tensor.grad is not None
    
    def test_training_loop(self, input_tensor):
        model = SparseMoETM(
            n_features=64,
            n_clauses_per_expert=16,
            n_classes=10,
            n_experts=4,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        labels = torch.randint(0, 10, (8,))
        
        for _ in range(3):
            logits, _ = model(input_tensor)
            loss = nn.functional.cross_entropy(logits, labels) + model.aux_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


class TestBatchedSparseMoETM:
    """Test batched MoE variant."""
    
    def test_output_shapes(self, input_tensor):
        model = BatchedSparseMoETM(
            n_features=64,
            n_clauses_per_expert=16,
            n_classes=10,
            n_experts=4,
        )
        
        logits, clauses = model(input_tensor)
        assert logits.shape == (8, 10)


class TestHierarchicalMoETM:
    """Test hierarchical MoE variant."""
    
    def test_output_shapes(self, input_tensor):
        model = HierarchicalMoETM(
            n_features=64,
            n_clauses_per_expert=8,
            n_classes=10,
            n_families=2,
            experts_per_family=2,
        )
        
        logits, clauses = model(input_tensor)
        assert logits.shape == (8, 10)


class TestSwitchMoETM:
    """Test Switch-style MoE."""
    
    def test_output_shapes(self, input_tensor):
        model = SwitchMoETM(
            n_features=64,
            n_clauses_per_expert=16,
            n_classes=10,
            n_experts=4,
        )
        
        logits, clauses = model(input_tensor)
        assert logits.shape == (8, 10)
        assert clauses.shape == (8, 16)
    
    def test_gradient_flow(self, input_tensor):
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        model = SwitchMoETM(
            n_features=64,
            n_clauses_per_expert=16,
            n_classes=10,
            n_experts=4,
        )
        
        logits, _ = model(input_tensor)
        loss = logits.sum() + model.aux_loss
        loss.backward()
        
        assert input_tensor.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])



