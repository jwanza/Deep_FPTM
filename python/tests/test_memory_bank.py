"""
Unit tests for ClauseMemoryBank.

Tests cover:
1. Shape correctness
2. EMA update behavior
3. Memory read/write operations
4. Gradient flow
5. Integration with TM models
"""

import pytest
import torch
import torch.nn as nn

from fptm_ste.tm import ClauseMemoryBank, ClauseMemoryAttention


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def clause_inputs():
    """Standard clause test inputs."""
    torch.manual_seed(42)
    batch_size = 8
    n_clauses = 32
    clause_dim = 64
    
    return torch.randn(batch_size, n_clauses, clause_dim)


@pytest.fixture
def scalar_clause_inputs():
    """Scalar clause inputs (2D)."""
    torch.manual_seed(42)
    return torch.rand(8, 32)


# =============================================================================
# Shape Tests
# =============================================================================


class TestShapes:
    """Test output shapes for memory bank."""
    
    def test_basic_read_shape(self, clause_inputs):
        """Read operation preserves shape."""
        memory = ClauseMemoryBank(
            n_slots=16,
            clause_dim=64,
        )
        
        output = memory.read(clause_inputs)
        assert output.shape == clause_inputs.shape
    
    def test_forward_shape(self, clause_inputs):
        """Forward pass preserves shape."""
        memory = ClauseMemoryBank(
            n_slots=16,
            clause_dim=64,
        )
        
        output = memory(clause_inputs)
        assert output.shape == clause_inputs.shape
    
    def test_2d_input_handling(self, scalar_clause_inputs):
        """Memory bank handles 2D inputs."""
        memory = ClauseMemoryBank(
            n_slots=16,
            clause_dim=1,
        )
        
        output = memory.read(scalar_clause_inputs)
        assert output.shape == scalar_clause_inputs.shape
    
    @pytest.mark.parametrize("n_slots", [8, 16, 64, 128])
    def test_various_slot_counts(self, clause_inputs, n_slots):
        """Different slot counts work correctly."""
        memory = ClauseMemoryBank(
            n_slots=n_slots,
            clause_dim=64,
        )
        
        output = memory(clause_inputs)
        assert output.shape == clause_inputs.shape
    
    @pytest.mark.parametrize("batch_size", [1, 8, 32])
    def test_various_batch_sizes(self, batch_size):
        """Different batch sizes work correctly."""
        x = torch.randn(batch_size, 32, 64)
        
        memory = ClauseMemoryBank(
            n_slots=16,
            clause_dim=64,
        )
        
        output = memory(x)
        assert output.shape == x.shape


# =============================================================================
# EMA Update Tests
# =============================================================================


class TestEMAUpdates:
    """Test EMA memory update behavior."""
    
    def test_memory_updates_during_training(self, clause_inputs):
        """Memory updates during training mode."""
        memory = ClauseMemoryBank(
            n_slots=16,
            clause_dim=64,
            ema_decay=0.9,
        )
        memory.train()
        
        initial_memory = memory.memory.clone()
        
        # Forward pass with update
        _ = memory(clause_inputs, update_memory=True)
        
        # Memory should have changed
        assert not torch.allclose(memory.memory, initial_memory)
    
    def test_memory_stable_during_eval(self, clause_inputs):
        """Memory doesn't update during eval mode."""
        memory = ClauseMemoryBank(
            n_slots=16,
            clause_dim=64,
        )
        memory.eval()
        
        initial_memory = memory.memory.clone()
        
        # Forward pass
        _ = memory(clause_inputs, update_memory=True)
        
        # Memory should be unchanged (eval mode)
        assert torch.allclose(memory.memory, initial_memory)
    
    def test_ema_decay_effect(self, clause_inputs):
        """Higher decay means slower updates."""
        memory_fast = ClauseMemoryBank(n_slots=16, clause_dim=64, ema_decay=0.5)
        memory_slow = ClauseMemoryBank(n_slots=16, clause_dim=64, ema_decay=0.99)
        
        # Initialize with same memory
        initial = torch.randn(16, 64)
        memory_fast.memory.copy_(initial)
        memory_slow.memory.copy_(initial)
        
        memory_fast.train()
        memory_slow.train()
        
        # Same input
        _ = memory_fast(clause_inputs)
        _ = memory_slow(clause_inputs)
        
        # Fast should have changed more
        fast_change = (memory_fast.memory - initial).abs().mean()
        slow_change = (memory_slow.memory - initial).abs().mean()
        
        assert fast_change > slow_change
    
    def test_explicit_write(self, clause_inputs):
        """Explicit write updates memory."""
        memory = ClauseMemoryBank(n_slots=16, clause_dim=64)
        memory.train()
        
        initial = memory.memory.clone()
        memory.write(clause_inputs)
        
        assert not torch.allclose(memory.memory, initial)


# =============================================================================
# Gradient Flow Tests
# =============================================================================


class TestGradientFlow:
    """Test gradient flow through memory bank."""
    
    def test_read_gradients(self, clause_inputs):
        """Gradients flow through read operation."""
        clause_inputs = clause_inputs.clone().requires_grad_(True)
        
        memory = ClauseMemoryBank(n_slots=16, clause_dim=64)
        output = memory.read(clause_inputs)
        loss = output.sum()
        loss.backward()
        
        assert clause_inputs.grad is not None
        assert not torch.isnan(clause_inputs.grad).any()
    
    def test_forward_gradients(self, clause_inputs):
        """Gradients flow through forward pass (memory update disabled)."""
        clause_inputs = clause_inputs.clone().requires_grad_(True)
        
        memory = ClauseMemoryBank(n_slots=16, clause_dim=64)
        memory.train()
        
        # Disable memory updates during gradient computation
        output = memory(clause_inputs, update_memory=False)
        loss = output.sum()
        loss.backward()
        
        # Input gradients should flow
        assert clause_inputs.grad is not None
        assert not torch.isnan(clause_inputs.grad).any()
        
        # At least the gate parameters should have gradients
        assert memory.output_gate[0].weight.grad is not None
    
    def test_training_updates_parameters(self, clause_inputs):
        """Parameters update during training."""
        memory = ClauseMemoryBank(n_slots=16, clause_dim=64)
        optimizer = torch.optim.Adam(memory.parameters(), lr=0.01)
        
        memory.train()
        
        # Get initial parameters (only the ones that will be used)
        initial_gate = memory.output_gate[0].weight.clone()
        
        for _ in range(5):
            # Forward without memory update for gradient computation
            output = memory(clause_inputs, update_memory=False)
            loss = (output - 0.5).pow(2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update memory after backward pass
            with torch.no_grad():
                memory.write(clause_inputs)
        
        # Check that gate parameters changed (these are the primary learnable params)
        assert not torch.allclose(memory.output_gate[0].weight, initial_gate), \
            "Gate parameters did not change"


# =============================================================================
# Memory Attention Tests
# =============================================================================


class TestClauseMemoryAttention:
    """Test ClauseMemoryAttention module."""
    
    def test_shape(self, clause_inputs):
        """Output shape is correct."""
        attn = ClauseMemoryAttention(
            clause_dim=64,
            n_slots=32,
            n_heads=4,
        )
        
        output = attn(clause_inputs)
        assert output.shape == clause_inputs.shape
    
    def test_gradient_flow(self, clause_inputs):
        """Gradients flow through module (memory update disabled)."""
        clause_inputs = clause_inputs.clone().requires_grad_(True)
        
        attn = ClauseMemoryAttention(clause_dim=64, n_slots=16)
        # Disable memory updates during gradient computation
        output = attn(clause_inputs, update_memory=False)
        loss = output.sum()
        loss.backward()
        
        assert clause_inputs.grad is not None
    
    def test_training_loop(self, clause_inputs):
        """Full training loop works."""
        attn = ClauseMemoryAttention(clause_dim=64, n_slots=16)
        optimizer = torch.optim.Adam(attn.parameters(), lr=0.001)
        
        attn.train()
        
        for _ in range(5):
            # Forward without memory update
            output = attn(clause_inputs, update_memory=False)
            loss = output.var()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # If we get here without error, training works
        assert True


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Test integration with TM models."""
    
    def test_with_stcm(self):
        """Works with STCM outputs."""
        from fptm_ste import FuzzyPatternTM_STCM
        
        model = FuzzyPatternTM_STCM(
            n_features=64,
            n_clauses=32,
            n_classes=10,
        )
        
        memory = ClauseMemoryBank(
            n_slots=16,
            clause_dim=1,
        )
        
        x = torch.rand(8, 64)
        _, clause_outputs = model(x)
        
        # Apply memory bank
        enhanced = memory(clause_outputs.unsqueeze(-1))
        assert enhanced.shape == clause_outputs.unsqueeze(-1).shape
    
    def test_end_to_end_training(self):
        """End-to-end training with STCM and memory."""
        from fptm_ste import FuzzyPatternTM_STCM
        
        class MemoryEnhancedSTCM(nn.Module):
            def __init__(self):
                super().__init__()
                self.stcm = FuzzyPatternTM_STCM(
                    n_features=64,
                    n_clauses=32,
                    n_classes=10,
                )
                # Use clause_dim=1 for scalar clause outputs
                self.memory = ClauseMemoryBank(
                    n_slots=16,
                    clause_dim=1,
                    learnable_keys=False,  # Avoid projection for dim=1
                )
                self.head = nn.Linear(32, 10)
            
            def forward(self, x):
                logits, clauses = self.stcm(x)
                # clauses: [batch, n_clauses], need [batch, n_clauses, 1]
                clauses_3d = clauses.unsqueeze(-1)
                # Disable memory update during forward for gradient computation
                enhanced = self.memory(clauses_3d, update_memory=False).squeeze(-1)
                return logits + 0.1 * self.head(enhanced)
        
        model = MemoryEnhancedSTCM()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        x = torch.rand(8, 64)
        y = torch.randint(0, 10, (8,))
        
        for _ in range(3):
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# =============================================================================
# Numerical Stability Tests
# =============================================================================


class TestNumericalStability:
    """Test numerical stability."""
    
    def test_no_nan_with_zeros(self):
        """No NaN with zero inputs."""
        x = torch.zeros(8, 32, 64)
        
        memory = ClauseMemoryBank(n_slots=16, clause_dim=64)
        output = memory(x)
        
        assert not torch.isnan(output).any()
    
    def test_no_nan_with_large_values(self):
        """No NaN with large values."""
        x = torch.randn(8, 32, 64) * 100
        
        memory = ClauseMemoryBank(n_slots=16, clause_dim=64)
        output = memory(x)
        
        assert not torch.isnan(output).any()
    
    def test_memory_stays_bounded(self):
        """Memory values stay bounded after many updates."""
        memory = ClauseMemoryBank(n_slots=16, clause_dim=64, ema_decay=0.9)
        memory.train()
        
        for _ in range(100):
            x = torch.randn(8, 32, 64)
            _ = memory(x)
        
        # Memory should not explode
        assert memory.memory.abs().max() < 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

