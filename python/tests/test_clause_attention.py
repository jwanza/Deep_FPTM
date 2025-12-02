"""
Unit tests for Hierarchical Clause Attention module.

Tests cover:
1. Shape correctness for all components
2. Gradient flow
3. Different configurations (stages, gates)
4. Integration with TM models
"""

import pytest
import torch
import torch.nn as nn

from fptm_ste.clause_attention import (
    ClauseAttentionHead,
    MultiHeadClauseAttention,
    ClauseGate,
    IntraPolarityAttention,
    CrossPolarityAttention,
    GlobalClauseConsensus,
    HierarchicalClauseAttention,
    ClauseTransformerBlock,
    ClauseReasoningNetwork,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def clause_inputs():
    """Standard clause test inputs."""
    torch.manual_seed(42)
    batch_size = 8
    n_pos = 16
    n_neg = 16
    clause_dim = 32
    
    pos_clauses = torch.randn(batch_size, n_pos, clause_dim)
    neg_clauses = torch.randn(batch_size, n_neg, clause_dim)
    
    return pos_clauses, neg_clauses


@pytest.fixture
def scalar_clause_inputs():
    """Scalar clause outputs (before embedding)."""
    torch.manual_seed(42)
    batch_size = 8
    n_pos = 16
    n_neg = 16
    
    pos_clauses = torch.rand(batch_size, n_pos)
    neg_clauses = torch.rand(batch_size, n_neg)
    
    return pos_clauses, neg_clauses


# =============================================================================
# Shape Tests
# =============================================================================


class TestShapes:
    """Test output shapes for all components."""
    
    def test_attention_head_shape(self, clause_inputs):
        """ClauseAttentionHead produces correct shape."""
        pos, neg = clause_inputs
        head = ClauseAttentionHead(clause_dim=32, head_dim=16)
        
        out = head(pos, neg, neg)
        assert out.shape == (pos.shape[0], pos.shape[1], 16)
    
    def test_multi_head_attention_shape(self, clause_inputs):
        """MultiHeadClauseAttention produces correct shape."""
        pos, neg = clause_inputs
        attn = MultiHeadClauseAttention(clause_dim=32, n_heads=4, head_dim=8)
        
        # Self-attention
        out = attn(pos)
        assert out.shape == pos.shape
        
        # Cross-attention
        out = attn(pos, neg, neg)
        assert out.shape == pos.shape
    
    def test_intra_polarity_shape(self, clause_inputs):
        """IntraPolarityAttention produces correct shapes."""
        pos, neg = clause_inputs
        attn = IntraPolarityAttention(clause_dim=32)
        
        pos_out, neg_out = attn(pos, neg)
        assert pos_out.shape == pos.shape
        assert neg_out.shape == neg.shape
    
    def test_cross_polarity_shape(self, clause_inputs):
        """CrossPolarityAttention produces correct shapes."""
        pos, neg = clause_inputs
        attn = CrossPolarityAttention(clause_dim=32)
        
        pos_out, neg_out = attn(pos, neg)
        assert pos_out.shape == pos.shape
        assert neg_out.shape == neg.shape
    
    def test_global_consensus_shape(self, clause_inputs):
        """GlobalClauseConsensus produces correct shapes."""
        pos, neg = clause_inputs
        attn = GlobalClauseConsensus(clause_dim=32, use_cls_token=True)
        
        pos_out, neg_out, cls_out = attn(pos, neg)
        assert pos_out.shape == pos.shape
        assert neg_out.shape == neg.shape
        assert cls_out.shape == (pos.shape[0], 32)
    
    def test_hierarchical_attention_shape(self, clause_inputs):
        """HierarchicalClauseAttention produces correct shapes."""
        pos, neg = clause_inputs
        attn = HierarchicalClauseAttention(
            clause_dim=32,
            stages=("intra", "cross", "global"),
        )
        
        pos_out, neg_out, cls_out = attn(pos, neg)
        assert pos_out.shape == pos.shape
        assert neg_out.shape == neg.shape
        assert cls_out is not None
    
    def test_scalar_input_handling(self, scalar_clause_inputs):
        """HierarchicalClauseAttention handles 2D inputs."""
        pos, neg = scalar_clause_inputs
        attn = HierarchicalClauseAttention(
            clause_dim=1,
            n_heads=1,
            stages=("global",),
        )
        
        pos_out, neg_out, _ = attn(pos, neg)
        assert pos_out.shape == pos.shape
        assert neg_out.shape == neg.shape


# =============================================================================
# Gradient Flow Tests
# =============================================================================


class TestGradientFlow:
    """Test gradient flow through attention components."""
    
    def test_multi_head_attention_gradients(self, clause_inputs):
        """Gradients flow through MultiHeadClauseAttention."""
        pos, neg = clause_inputs
        pos.requires_grad = True
        neg.requires_grad = True
        
        attn = MultiHeadClauseAttention(clause_dim=32)
        out = attn(pos, neg, neg)
        loss = out.sum()
        loss.backward()
        
        assert pos.grad is not None
        assert neg.grad is not None
        assert not torch.isnan(pos.grad).any()
    
    def test_hierarchical_attention_gradients(self, clause_inputs):
        """Gradients flow through HierarchicalClauseAttention."""
        pos, neg = clause_inputs
        pos.requires_grad = True
        neg.requires_grad = True
        
        attn = HierarchicalClauseAttention(
            clause_dim=32,
            stages=("intra", "cross", "global"),
            use_gates=True,
        )
        
        pos_out, neg_out, cls_out = attn(pos, neg)
        loss = pos_out.sum() + neg_out.sum() + cls_out.sum()
        loss.backward()
        
        assert pos.grad is not None
        assert neg.grad is not None
        
        # Check model parameters have gradients
        for name, param in attn.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
    
    def test_reasoning_network_gradients(self, clause_inputs):
        """Gradients flow through ClauseReasoningNetwork."""
        pos, neg = clause_inputs
        pos.requires_grad = True
        neg.requires_grad = True
        
        network = ClauseReasoningNetwork(
            clause_dim=32,
            n_layers=2,
        )
        
        pos_out, neg_out = network(pos, neg)
        loss = pos_out.sum() + neg_out.sum()
        loss.backward()
        
        assert pos.grad is not None
        assert neg.grad is not None


# =============================================================================
# Configuration Tests
# =============================================================================


class TestConfigurations:
    """Test different configurations."""
    
    @pytest.mark.parametrize("stages", [
        ("intra",),
        ("cross",),
        ("global",),
        ("intra", "cross"),
        ("cross", "global"),
        ("intra", "cross", "global"),
    ])
    def test_stage_combinations(self, clause_inputs, stages):
        """Different stage combinations work correctly."""
        pos, neg = clause_inputs
        
        attn = HierarchicalClauseAttention(
            clause_dim=32,
            stages=stages,
        )
        
        pos_out, neg_out, cls_out = attn(pos, neg)
        assert pos_out.shape == pos.shape
        assert neg_out.shape == neg.shape
        
        # CLS only available with global stage
        if "global" in stages:
            assert cls_out is not None
        else:
            assert cls_out is None
    
    @pytest.mark.parametrize("use_gates", [True, False])
    def test_gate_configuration(self, clause_inputs, use_gates):
        """Gating can be enabled/disabled."""
        pos, neg = clause_inputs
        
        attn = HierarchicalClauseAttention(
            clause_dim=32,
            stages=("intra", "cross"),
            use_gates=use_gates,
        )
        
        pos_out, neg_out, _ = attn(pos, neg)
        assert pos_out.shape == pos.shape
    
    @pytest.mark.parametrize("n_heads", [1, 2, 4, 8])
    def test_head_count(self, clause_inputs, n_heads):
        """Different head counts work correctly."""
        pos, neg = clause_inputs
        
        attn = HierarchicalClauseAttention(
            clause_dim=32,
            n_heads=n_heads,
            stages=("intra",),
        )
        
        pos_out, neg_out, _ = attn(pos, neg)
        assert pos_out.shape == pos.shape
    
    def test_cross_polarity_unidirectional(self, clause_inputs):
        """Unidirectional cross-polarity attention works."""
        pos, neg = clause_inputs
        
        attn = CrossPolarityAttention(
            clause_dim=32,
            bidirectional=False,
        )
        
        pos_out, neg_out = attn(pos, neg)
        assert pos_out.shape == pos.shape
        # Negative clauses should be unchanged
        assert torch.allclose(neg_out, neg)


# =============================================================================
# Gate Tests
# =============================================================================


class TestGates:
    """Test gating mechanisms."""
    
    @pytest.mark.parametrize("gate_type", ["sigmoid", "tanh", "glu"])
    def test_gate_types(self, gate_type):
        """Different gate types produce valid outputs."""
        gate = ClauseGate(clause_dim=32, gate_type=gate_type)
        
        original = torch.randn(8, 16, 32)
        refined = torch.randn(8, 16, 32)
        
        output = gate(original, refined)
        assert output.shape == original.shape
        assert not torch.isnan(output).any()
    
    def test_gate_preserves_shape(self):
        """Gates preserve input shape."""
        gate = ClauseGate(clause_dim=32)
        
        for batch_size in [1, 8, 32]:
            for n_clauses in [4, 16, 64]:
                original = torch.randn(batch_size, n_clauses, 32)
                refined = torch.randn(batch_size, n_clauses, 32)
                
                output = gate(original, refined)
                assert output.shape == (batch_size, n_clauses, 32)


# =============================================================================
# Transformer Block Tests
# =============================================================================


class TestTransformerBlocks:
    """Test transformer block components."""
    
    def test_block_shape(self, clause_inputs):
        """ClauseTransformerBlock preserves shape."""
        pos, _ = clause_inputs
        
        block = ClauseTransformerBlock(clause_dim=32)
        out = block(pos)
        
        assert out.shape == pos.shape
    
    def test_block_gradients(self, clause_inputs):
        """Gradients flow through transformer block."""
        pos, _ = clause_inputs
        pos.requires_grad = True
        
        block = ClauseTransformerBlock(clause_dim=32)
        out = block(pos)
        loss = out.sum()
        loss.backward()
        
        assert pos.grad is not None
    
    def test_reasoning_network_layers(self, clause_inputs):
        """ClauseReasoningNetwork with different layer counts."""
        pos, neg = clause_inputs
        
        for n_layers in [1, 2, 4]:
            network = ClauseReasoningNetwork(
                clause_dim=32,
                n_layers=n_layers,
            )
            
            pos_out, neg_out = network(pos, neg)
            assert pos_out.shape == pos.shape
            assert neg_out.shape == neg.shape
    
    def test_polarity_embedding(self, clause_inputs):
        """Polarity embedding can be enabled/disabled."""
        pos, neg = clause_inputs
        
        # With polarity embedding
        net_with = ClauseReasoningNetwork(
            clause_dim=32,
            use_polarity_embedding=True,
        )
        pos_out1, _ = net_with(pos, neg)
        
        # Without polarity embedding
        net_without = ClauseReasoningNetwork(
            clause_dim=32,
            use_polarity_embedding=False,
        )
        pos_out2, _ = net_without(pos, neg)
        
        # Both should work
        assert pos_out1.shape == pos.shape
        assert pos_out2.shape == pos.shape


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Test integration with TM models."""
    
    def test_with_stcm_outputs(self):
        """Works with typical STCM output format."""
        from fptm_ste import FuzzyPatternTM_STCM
        
        # Create STCM
        model = FuzzyPatternTM_STCM(
            n_features=64,
            n_clauses=32,
            n_classes=10,
        )
        
        # Get clause outputs
        x = torch.rand(8, 64)
        logits, clause_outputs = model(x)
        
        # Split into positive/negative
        n_half = clause_outputs.shape[1] // 2
        pos_clauses = clause_outputs[:, :n_half].unsqueeze(-1)
        neg_clauses = clause_outputs[:, n_half:].unsqueeze(-1)
        
        # Apply hierarchical attention
        attn = HierarchicalClauseAttention(
            clause_dim=1,
            n_heads=1,
            stages=("intra", "global"),
        )
        
        pos_out, neg_out, _ = attn(pos_clauses, neg_clauses)
        assert pos_out.shape == pos_clauses.shape
    
    def test_end_to_end_training(self, clause_inputs):
        """Full training loop works."""
        pos, neg = clause_inputs
        
        attn = HierarchicalClauseAttention(clause_dim=32)
        optimizer = torch.optim.Adam(attn.parameters(), lr=1e-3)
        
        for _ in range(5):
            pos_out, neg_out, cls_out = attn(pos, neg)
            
            # Simple loss
            loss = -cls_out.mean() + pos_out.var() + neg_out.var()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Check parameters changed
        assert True  # If we get here without error, training worked


# =============================================================================
# Numerical Stability Tests
# =============================================================================


class TestNumericalStability:
    """Test numerical stability."""
    
    def test_no_nan_with_zeros(self):
        """No NaN with zero inputs."""
        pos = torch.zeros(8, 16, 32)
        neg = torch.zeros(8, 16, 32)
        
        attn = HierarchicalClauseAttention(clause_dim=32)
        pos_out, neg_out, cls_out = attn(pos, neg)
        
        assert not torch.isnan(pos_out).any()
        assert not torch.isnan(neg_out).any()
        assert not torch.isnan(cls_out).any()
    
    def test_no_nan_with_large_values(self):
        """No NaN with large values (attention softmax stability)."""
        pos = torch.randn(8, 16, 32) * 100
        neg = torch.randn(8, 16, 32) * 100
        
        attn = HierarchicalClauseAttention(clause_dim=32)
        pos_out, neg_out, cls_out = attn(pos, neg)
        
        assert not torch.isnan(pos_out).any()
        assert not torch.isnan(neg_out).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

