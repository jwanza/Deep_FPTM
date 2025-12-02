"""
Unit tests for Multi-Resolution STCM architectures.

Tests cover shape correctness, gradient flow, fusion types, and configurations.
"""

import pytest
import torch
import torch.nn as nn

from fptm_ste.multires_tm import (
    MultiResolutionBranch,
    ResolutionFusion,
    MultiResolutionSTCM,
    AdaptiveThresholdSTCM,
    CascadeResolutionSTCM,
    HierarchicalResolutionSTCM,
)


@pytest.fixture
def input_tensor():
    """Standard input tensor."""
    torch.manual_seed(42)
    return torch.rand(8, 64)


class TestMultiResolutionBranch:
    """Test individual resolution branch."""
    
    def test_shape(self, input_tensor):
        """Output shapes are correct."""
        branch = MultiResolutionBranch(
            n_features=64,
            n_clauses=32,
            n_classes=10,
            tau=0.5,
        )
        
        logits, clauses = branch(input_tensor)
        assert logits.shape == (8, 10)
        assert clauses.shape == (8, 32)
    
    def test_different_tau(self, input_tensor):
        """Different tau values produce different outputs."""
        branch_low = MultiResolutionBranch(
            n_features=64, n_clauses=32, n_classes=10, tau=0.3
        )
        branch_high = MultiResolutionBranch(
            n_features=64, n_clauses=32, n_classes=10, tau=0.7
        )
        
        out_low, _ = branch_low(input_tensor)
        out_high, _ = branch_high(input_tensor)
        
        # Outputs should differ (different thresholds)
        assert not torch.allclose(out_low, out_high)


class TestResolutionFusion:
    """Test fusion mechanisms."""
    
    @pytest.fixture
    def logits_list(self):
        return [torch.randn(8, 10) for _ in range(3)]
    
    @pytest.fixture
    def clauses_list(self):
        return [torch.rand(8, 32) for _ in range(3)]
    
    @pytest.mark.parametrize("fusion_type", ["attention", "concat", "gated", "avg", "max"])
    def test_fusion_types(self, logits_list, clauses_list, fusion_type):
        """All fusion types work correctly."""
        fusion = ResolutionFusion(
            n_resolutions=3,
            n_classes=10,
            n_clauses=32,
            fusion_type=fusion_type,
        )
        
        fused_logits, fused_clauses = fusion(logits_list, clauses_list)
        assert fused_logits.shape == (8, 10)
        assert fused_clauses.shape == (8, 32)
    
    def test_attention_weights_learnable(self, logits_list, clauses_list):
        """Attention fusion has learnable weights."""
        fusion = ResolutionFusion(
            n_resolutions=3,
            n_classes=10,
            n_clauses=32,
            fusion_type="attention",
        )
        
        assert fusion.resolution_weights.requires_grad


class TestMultiResolutionSTCM:
    """Test main MultiResolutionSTCM class."""
    
    def test_default_config(self, input_tensor):
        """Default configuration works."""
        model = MultiResolutionSTCM(
            n_features=64,
            n_clauses=32,
            n_classes=10,
        )
        
        logits, clauses = model(input_tensor)
        assert logits.shape == (8, 10)
        assert clauses.shape == (8, 32)
    
    def test_custom_tau_values(self, input_tensor):
        """Custom tau values work."""
        model = MultiResolutionSTCM(
            n_features=64,
            n_clauses=32,
            n_classes=10,
            tau_values=[0.2, 0.4, 0.6, 0.8],
        )
        
        assert model.n_resolutions == 4
        logits, clauses = model(input_tensor)
        assert logits.shape == (8, 10)
    
    def test_per_branch_config(self, input_tensor):
        """Per-branch configuration works."""
        model = MultiResolutionSTCM(
            n_features=64,
            n_clauses=32,
            n_classes=10,
            tau_values=[0.3, 0.5, 0.7],
            ternary_bands=[0.0, 0.1, 0.2],
            operators=["capacity", "product", "capacity"],
        )
        
        logits, clauses = model(input_tensor)
        assert logits.shape == (8, 10)
    
    def test_gradient_flow(self, input_tensor):
        """Gradients flow through all branches."""
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        model = MultiResolutionSTCM(
            n_features=64,
            n_clauses=32,
            n_classes=10,
        )
        
        logits, _ = model(input_tensor)
        loss = logits.sum()
        loss.backward()
        
        assert input_tensor.grad is not None
        assert not torch.isnan(input_tensor.grad).any()
    
    def test_branch_outputs(self, input_tensor):
        """Can get per-branch outputs."""
        model = MultiResolutionSTCM(
            n_features=64,
            n_clauses=32,
            n_classes=10,
            tau_values=[0.3, 0.5, 0.7],
        )
        
        outputs = model(input_tensor, return_branch_outputs=True)
        
        assert "logits" in outputs
        assert "branch_logits" in outputs
        assert len(outputs["branch_logits"]) == 3
    
    def test_resolution_weights(self, input_tensor):
        """Can get learned resolution weights."""
        model = MultiResolutionSTCM(
            n_features=64,
            n_clauses=32,
            n_classes=10,
            fusion_type="attention",
        )
        
        weights = model.get_resolution_weights()
        assert weights is not None
        assert weights.shape == (3,)
        assert torch.allclose(weights.sum(), torch.tensor(1.0))


class TestAdaptiveThresholdSTCM:
    """Test adaptive threshold model."""
    
    def test_shape(self, input_tensor):
        """Output shapes are correct."""
        model = AdaptiveThresholdSTCM(
            n_features=64,
            n_clauses=32,
            n_classes=10,
        )
        
        logits, clauses, tau = model(input_tensor)
        assert logits.shape == (8, 10)
        assert clauses.shape == (8, 32)
        assert tau.shape == (8,)
    
    def test_tau_in_range(self, input_tensor):
        """Predicted tau is in valid range."""
        model = AdaptiveThresholdSTCM(
            n_features=64,
            n_clauses=32,
            n_classes=10,
            base_tau=0.5,
            tau_range=0.3,
        )
        
        _, _, tau = model(input_tensor)
        assert torch.all(tau >= 0.2)  # 0.5 - 0.3
        assert torch.all(tau <= 0.8)  # 0.5 + 0.3
    
    def test_gradient_flow(self, input_tensor):
        """Gradients flow through tau predictor."""
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        model = AdaptiveThresholdSTCM(
            n_features=64,
            n_clauses=32,
            n_classes=10,
        )
        
        logits, _, _ = model(input_tensor)
        loss = logits.sum()
        loss.backward()
        
        assert input_tensor.grad is not None


class TestCascadeResolutionSTCM:
    """Test cascaded resolution model."""
    
    def test_shape(self, input_tensor):
        """Output shapes are correct."""
        model = CascadeResolutionSTCM(
            n_features=64,
            n_clauses=32,
            n_classes=10,
            n_stages=3,
        )
        
        logits, clauses = model(input_tensor)
        assert logits.shape == (8, 10)
        assert clauses.shape == (8, 32 * 3)  # Concatenated from all stages
    
    def test_residual_connection(self, input_tensor):
        """Residual connection affects output."""
        model_residual = CascadeResolutionSTCM(
            n_features=64,
            n_clauses=32,
            n_classes=10,
            residual=True,
        )
        model_no_residual = CascadeResolutionSTCM(
            n_features=64,
            n_clauses=32,
            n_classes=10,
            residual=False,
        )
        
        out_r, _ = model_residual(input_tensor)
        out_nr, _ = model_no_residual(input_tensor)
        
        # Outputs should differ
        assert not torch.allclose(out_r, out_nr)
    
    def test_gradient_flow(self, input_tensor):
        """Gradients flow through all stages."""
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        model = CascadeResolutionSTCM(
            n_features=64,
            n_clauses=32,
            n_classes=10,
            n_stages=4,
        )
        
        logits, _ = model(input_tensor)
        loss = logits.sum()
        loss.backward()
        
        assert input_tensor.grad is not None


class TestHierarchicalResolutionSTCM:
    """Test hierarchical resolution model."""
    
    def test_shape(self, input_tensor):
        """Output shapes are correct."""
        model = HierarchicalResolutionSTCM(
            n_features=64,
            n_clauses=32,
            n_classes=10,
            n_groups=4,
        )
        
        logits, clauses = model(input_tensor)
        assert logits.shape == (8, 10)
        assert clauses.shape == (8, 32)  # 32/4 * 4 groups
    
    def test_group_taus(self, input_tensor):
        """Can get per-group tau values."""
        model = HierarchicalResolutionSTCM(
            n_features=64,
            n_clauses=32,
            n_classes=10,
            n_groups=4,
        )
        
        taus = model.get_group_taus()
        assert taus.shape == (4,)
        assert torch.all(taus >= 0) and torch.all(taus <= 1)
    
    def test_gradient_flow(self, input_tensor):
        """Gradients flow through all groups."""
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        model = HierarchicalResolutionSTCM(
            n_features=64,
            n_clauses=32,
            n_classes=10,
            n_groups=4,
        )
        
        logits, _ = model(input_tensor)
        loss = logits.sum()
        loss.backward()
        
        assert input_tensor.grad is not None


class TestTraining:
    """Test training behavior."""
    
    def test_training_loop(self, input_tensor):
        """Full training loop works."""
        model = MultiResolutionSTCM(
            n_features=64,
            n_clauses=32,
            n_classes=10,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        labels = torch.randint(0, 10, (8,))
        
        for _ in range(3):
            logits, _ = model(input_tensor)
            loss = nn.functional.cross_entropy(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def test_weights_update(self, input_tensor):
        """Resolution weights update during training."""
        model = MultiResolutionSTCM(
            n_features=64,
            n_clauses=32,
            n_classes=10,
            fusion_type="attention",
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        initial_weights = model.fusion.resolution_weights.clone()
        labels = torch.randint(0, 10, (8,))
        
        for _ in range(10):
            logits, _ = model(input_tensor)
            loss = nn.functional.cross_entropy(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        assert not torch.allclose(model.fusion.resolution_weights, initial_weights)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

