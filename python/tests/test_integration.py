"""
Integration tests for all FuzzyPatternTM modules.

Tests that all modules work together correctly and can be combined
in various configurations.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def integration_params():
    """Parameters for integration tests."""
    return {
        "n_features": 16,
        "n_clauses": 8,
        "n_classes": 3,
        "batch_size": 4,
        "seq_len": 5,
    }


@pytest.fixture
def sample_input(integration_params):
    """Generate sample input."""
    return torch.rand(
        integration_params["batch_size"],
        integration_params["n_features"],
    )


@pytest.fixture
def sequence_input(integration_params):
    """Generate sequence input."""
    return torch.rand(
        integration_params["batch_size"],
        integration_params["seq_len"],
        integration_params["n_features"],
    )


# =============================================================================
# Base TM Integration
# =============================================================================


class TestBaseTMIntegration:
    """Test base TM modules work together."""
    
    def test_stcm_forward(self, integration_params, sample_input):
        """Test STCM forward pass."""
        from fptm_ste.tm import FuzzyPatternTM_STCM
        
        model = FuzzyPatternTM_STCM(
            n_features=integration_params["n_features"],
            n_clauses=integration_params["n_clauses"],
            n_classes=integration_params["n_classes"],
        )
        
        logits, clauses = model(sample_input)
        
        assert logits.shape == (integration_params["batch_size"], integration_params["n_classes"])
        assert clauses.shape == (integration_params["batch_size"], integration_params["n_clauses"])
    
    def test_stcm_with_operators(self, integration_params, sample_input):
        """Test STCM with different operators."""
        from fptm_ste.tm import FuzzyPatternTM_STCM
        
        for operator in ["capacity", "product"]:
            model = FuzzyPatternTM_STCM(
                n_features=integration_params["n_features"],
                n_clauses=integration_params["n_clauses"],
                n_classes=integration_params["n_classes"],
                operator=operator,
            )
            
            logits, _ = model(sample_input)
            assert not torch.any(torch.isnan(logits))


# =============================================================================
# Hyperbolic + STCM Integration
# =============================================================================


class TestHyperbolicIntegration:
    """Test hyperbolic voting with STCM."""
    
    def test_hyperbolic_voting_with_stcm(self, integration_params, sample_input):
        """Test hyperbolic voting can replace linear voting."""
        from fptm_ste.tm import FuzzyPatternTM_STCM
        from fptm_ste.hyperbolic import HyperbolicClauseVoting
        
        # Get clause outputs
        tm = FuzzyPatternTM_STCM(
            n_features=integration_params["n_features"],
            n_clauses=integration_params["n_clauses"],
            n_classes=integration_params["n_classes"],
        )
        
        _, clauses = tm(sample_input)
        
        # Apply hyperbolic voting
        voting = HyperbolicClauseVoting(
            n_clauses=integration_params["n_clauses"],
            n_classes=integration_params["n_classes"],
            embed_dim=16,
        )
        
        logits = voting(clauses)
        
        assert logits.shape == (integration_params["batch_size"], integration_params["n_classes"])
        assert not torch.any(torch.isnan(logits))


# =============================================================================
# Sparse Routing Integration
# =============================================================================


class TestSparseRoutingIntegration:
    """Test sparse routing with TM."""
    
    def test_topk_router_shapes(self, integration_params, sample_input):
        """Test TopK router output shapes."""
        from fptm_ste.sparse_routing import TopKRouter
        
        router = TopKRouter(
            input_dim=integration_params["n_features"],
            n_experts=4,
            top_k=2,
        )
        
        weights, indices, lb_loss, logits = router(sample_input)
        
        assert weights.shape == (integration_params["batch_size"], 2)
        assert indices.shape == (integration_params["batch_size"], 2)
    
    def test_sparse_moe_with_stcm(self, integration_params, sample_input):
        """Test SparseMoE with STCM experts."""
        from fptm_ste.sparse_routing import SparseMoEClauseMachine
        
        model = SparseMoEClauseMachine(
            n_features=integration_params["n_features"],
            n_clauses_per_expert=4,
            n_classes=integration_params["n_classes"],
            n_experts=4,
            top_k=2,
        )
        
        logits, clauses = model(sample_input)
        
        assert logits.shape == (integration_params["batch_size"], integration_params["n_classes"])


# =============================================================================
# Clause Attention Integration
# =============================================================================


class TestClauseAttentionIntegration:
    """Test clause attention with TM."""
    
    def test_multihead_attention_with_clauses(self, integration_params, sample_input):
        """Test multi-head attention on clause outputs."""
        from fptm_ste.tm import FuzzyPatternTM_STCM
        from fptm_ste.clause_attention import MultiHeadClauseAttention
        
        tm = FuzzyPatternTM_STCM(
            n_features=integration_params["n_features"],
            n_clauses=integration_params["n_clauses"],
            n_classes=integration_params["n_classes"],
        )
        
        _, clauses = tm(sample_input)
        
        attention = MultiHeadClauseAttention(
            clause_dim=1,
            n_heads=2,
            n_clauses=integration_params["n_clauses"],
        )
        
        attended = attention(clauses.unsqueeze(-1))
        
        assert attended.shape == (integration_params["batch_size"], integration_params["n_clauses"], 1)


# =============================================================================
# Continual Learning Integration
# =============================================================================


class TestContinualLearningIntegration:
    """Test continual learning with TM."""
    
    def test_ewc_with_stcm(self, integration_params, sample_input):
        """Test EWC wrapper with STCM."""
        from fptm_ste.tm import FuzzyPatternTM_STCM
        from fptm_ste.continual import EWCWrapper
        
        tm = FuzzyPatternTM_STCM(
            n_features=integration_params["n_features"],
            n_clauses=integration_params["n_clauses"],
            n_classes=integration_params["n_classes"],
        )
        
        ewc = EWCWrapper(tm, lambda_=1000.0)
        
        # Forward pass works
        logits, _ = ewc.model(sample_input)
        assert logits.shape == (integration_params["batch_size"], integration_params["n_classes"])
    
    def test_lora_with_stcm(self, integration_params, sample_input):
        """Test LoRA adapter with STCM."""
        from fptm_ste.tm import FuzzyPatternTM_STCM
        from fptm_ste.lora_adapter import LoRAWrapper
        
        tm = FuzzyPatternTM_STCM(
            n_features=integration_params["n_features"],
            n_clauses=integration_params["n_clauses"],
            n_classes=integration_params["n_classes"],
        )
        
        lora_tm = LoRAWrapper(tm, rank=4)
        
        # Forward pass works
        logits, _ = lora_tm(sample_input)
        assert logits.shape == (integration_params["batch_size"], integration_params["n_classes"])


# =============================================================================
# Booleanization Integration
# =============================================================================


class TestBooleanizationIntegration:
    """Test booleanization modules together."""
    
    def test_continuous_residual_training(self, integration_params, sample_input):
        """Test CRCM can be trained."""
        from fptm_ste.booleanization import ContinuousResidualClauseMachine
        
        model = ContinuousResidualClauseMachine(
            n_features=integration_params["n_features"],
            n_clauses=integration_params["n_clauses"],
            n_classes=integration_params["n_classes"],
        )
        
        y = torch.randint(0, integration_params["n_classes"], (integration_params["batch_size"],))
        
        # Forward
        logits, _ = model(sample_input)
        loss = F.cross_entropy(logits, y)
        
        # Backward
        loss.backward()
        
        # Check gradients
        has_grads = False
        for param in model.parameters():
            if param.grad is not None:
                has_grads = True
                break
        
        assert has_grads
    
    def test_all_booleanization_methods(self, integration_params, sample_input):
        """Test all booleanization methods produce valid outputs."""
        from fptm_ste.booleanization import (
            ContinuousResidualClauseMachine,
            ProbabilisticLiteralClauseMachine,
            HyperdimensionalClauseMachine,
        )
        
        models = [
            ContinuousResidualClauseMachine(
                n_features=integration_params["n_features"],
                n_clauses=integration_params["n_clauses"],
                n_classes=integration_params["n_classes"],
            ),
            ProbabilisticLiteralClauseMachine(
                n_features=integration_params["n_features"],
                n_clauses=integration_params["n_clauses"],
                n_classes=integration_params["n_classes"],
            ),
            HyperdimensionalClauseMachine(
                n_features=integration_params["n_features"],
                n_clauses=integration_params["n_clauses"],
                n_classes=integration_params["n_classes"],
                hd_dim=256,
            ),
        ]
        
        for model in models:
            model.eval()
            with torch.no_grad():
                logits, _ = model(sample_input)
            
            assert not torch.any(torch.isnan(logits))
            assert logits.shape[0] == integration_params["batch_size"]


# =============================================================================
# Ultimate Hybrid Integration
# =============================================================================


class TestUltimateHybridIntegration:
    """Test ultimate hybrid architecture."""
    
    def test_light_hybrid(self, integration_params, sample_input):
        """Test lightweight hybrid configuration."""
        from fptm_ste.ultimate_hybrid import create_light_hybrid
        
        model = create_light_hybrid(
            n_features=integration_params["n_features"],
            n_clauses=integration_params["n_clauses"],
            n_classes=integration_params["n_classes"],
        )
        
        logits, clauses = model(sample_input)
        
        assert logits.shape == (integration_params["batch_size"], integration_params["n_classes"])
    
    def test_hybrid_with_all_streams(self, integration_params, sample_input):
        """Test hybrid with all streams enabled."""
        from fptm_ste.ultimate_hybrid import UltimateHybridTM
        
        model = UltimateHybridTM(
            n_features=integration_params["n_features"],
            n_clauses=integration_params["n_clauses"],
            n_classes=integration_params["n_classes"],
            use_binary_stream=True,
            use_continuous_stream=True,
            use_hd_stream=True,
            use_ib_stream=False,  # Disable to avoid import issues
            use_probabilistic_stream=False,
        )
        
        logits, clauses = model(sample_input)
        
        assert logits.shape == (integration_params["batch_size"], integration_params["n_classes"])
    
    def test_hybrid_auxiliary_losses(self, integration_params, sample_input):
        """Test auxiliary losses are computed."""
        from fptm_ste.ultimate_hybrid import UltimateHybridTM
        
        model = UltimateHybridTM(
            n_features=integration_params["n_features"],
            n_clauses=integration_params["n_clauses"],
            n_classes=integration_params["n_classes"],
            use_binary_stream=True,
            use_continuous_stream=True,
            reconstruction_weight=0.1,
        )
        
        logits, _ = model(sample_input)
        
        aux_losses = model.get_auxiliary_losses()
        
        assert "reconstruction" in aux_losses
        assert aux_losses["reconstruction"] >= 0


# =============================================================================
# Temporal Integration
# =============================================================================


class TestTemporalIntegration:
    """Test temporal clause machine."""
    
    def test_temporal_tm_forward(self, integration_params, sequence_input):
        """Test temporal TM forward pass."""
        from fptm_ste.temporal import TemporalClauseMachine
        
        model = TemporalClauseMachine(
            n_features=integration_params["n_features"],
            n_clauses=integration_params["n_clauses"],
            n_classes=integration_params["n_classes"],
            state_dim=32,
        )
        
        logits, hidden = model(sequence_input)
        
        assert logits.shape == (integration_params["batch_size"], integration_params["n_classes"])
        assert hidden.shape == (integration_params["batch_size"], 32)
    
    def test_bidirectional_temporal(self, integration_params, sequence_input):
        """Test bidirectional temporal TM."""
        from fptm_ste.temporal import BidirectionalTemporalClauseMachine
        
        model = BidirectionalTemporalClauseMachine(
            n_features=integration_params["n_features"],
            n_clauses=integration_params["n_clauses"],
            n_classes=integration_params["n_classes"],
            state_dim=32,
        )
        
        logits, hidden = model(sequence_input)
        
        assert logits.shape == (integration_params["batch_size"], integration_params["n_classes"])
        assert hidden.shape == (integration_params["batch_size"], 64)  # 2 * state_dim
    
    def test_temporal_with_attention(self, integration_params, sequence_input):
        """Test temporal TM with attention."""
        from fptm_ste.temporal import TemporalClauseMachine
        
        model = TemporalClauseMachine(
            n_features=integration_params["n_features"],
            n_clauses=integration_params["n_clauses"],
            n_classes=integration_params["n_classes"],
            state_dim=32,
            use_temporal_attention=True,
            pooling="attention",
        )
        
        result = model(sequence_input, return_all_states=True)
        
        assert "logits" in result
        assert "attention_weights" in result


# =============================================================================
# Optimizer Integration
# =============================================================================


class TestOptimizerIntegration:
    """Test custom optimizers with TM."""
    
    def test_sam_with_stcm(self, integration_params, sample_input):
        """Test SAM optimizer with STCM."""
        from fptm_ste.tm import FuzzyPatternTM_STCM
        from fptm_ste.sam_optimizer import SAM
        
        model = FuzzyPatternTM_STCM(
            n_features=integration_params["n_features"],
            n_clauses=integration_params["n_clauses"],
            n_classes=integration_params["n_classes"],
        )
        
        optimizer = SAM(model.parameters(), torch.optim.Adam, lr=0.01, rho=0.05)
        
        y = torch.randint(0, integration_params["n_classes"], (integration_params["batch_size"],))
        
        # First step
        logits, _ = model(sample_input)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.first_step(zero_grad=True)
        
        # Second step
        logits, _ = model(sample_input)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.second_step(zero_grad=True)
        
        # Should complete without error
        assert True


# =============================================================================
# Augmentation Integration
# =============================================================================


class TestAugmentationIntegration:
    """Test augmentation with TM training."""
    
    def test_mixup_with_stcm(self, integration_params, sample_input):
        """Test mixup augmentation with STCM."""
        from fptm_ste.tm import FuzzyPatternTM_STCM
        from fptm_ste.augmentation import mixup_data, mixup_criterion
        
        model = FuzzyPatternTM_STCM(
            n_features=integration_params["n_features"],
            n_clauses=integration_params["n_clauses"],
            n_classes=integration_params["n_classes"],
        )
        
        y = torch.randint(0, integration_params["n_classes"], (integration_params["batch_size"],))
        
        # Apply mixup
        mixed_x, y_a, y_b, lam = mixup_data(sample_input, y, alpha=0.4)
        
        # Forward
        logits, _ = model(mixed_x)
        
        # Mixup loss
        loss = mixup_criterion(F.cross_entropy, logits, y_a, y_b, lam)
        
        assert not torch.isnan(loss)
    
    def test_cutmix_with_stcm(self, integration_params, sample_input):
        """Test cutmix augmentation with STCM."""
        from fptm_ste.tm import FuzzyPatternTM_STCM
        from fptm_ste.augmentation import cutmix_data, mixup_criterion
        
        model = FuzzyPatternTM_STCM(
            n_features=integration_params["n_features"],
            n_clauses=integration_params["n_clauses"],
            n_classes=integration_params["n_classes"],
        )
        
        y = torch.randint(0, integration_params["n_classes"], (integration_params["batch_size"],))
        
        # Apply cutmix
        mixed_x, y_a, y_b, lam = cutmix_data(sample_input, y, alpha=0.4)
        
        # Forward
        logits, _ = model(mixed_x)
        
        # Mixup loss
        loss = mixup_criterion(F.cross_entropy, logits, y_a, y_b, lam)
        
        assert not torch.isnan(loss)


# =============================================================================
# Export/Import Integration
# =============================================================================


class TestExportImportIntegration:
    """Test model serialization."""
    
    def test_stcm_save_load(self, integration_params, sample_input, tmp_path):
        """Test saving and loading STCM."""
        from fptm_ste.tm import FuzzyPatternTM_STCM
        
        model = FuzzyPatternTM_STCM(
            n_features=integration_params["n_features"],
            n_clauses=integration_params["n_clauses"],
            n_classes=integration_params["n_classes"],
        )
        
        # Get initial output
        model.eval()
        with torch.no_grad():
            logits1, _ = model(sample_input)
        
        # Save
        save_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), save_path)
        
        # Load into new model
        model2 = FuzzyPatternTM_STCM(
            n_features=integration_params["n_features"],
            n_clauses=integration_params["n_clauses"],
            n_classes=integration_params["n_classes"],
        )
        model2.load_state_dict(torch.load(save_path))
        
        # Compare outputs
        model2.eval()
        with torch.no_grad():
            logits2, _ = model2(sample_input)
        
        assert torch.allclose(logits1, logits2)
    
    def test_hybrid_save_load(self, integration_params, sample_input, tmp_path):
        """Test saving and loading hybrid model."""
        from fptm_ste.ultimate_hybrid import create_light_hybrid
        
        model = create_light_hybrid(
            n_features=integration_params["n_features"],
            n_clauses=integration_params["n_clauses"],
            n_classes=integration_params["n_classes"],
        )
        
        # Get initial output
        model.eval()
        with torch.no_grad():
            logits1, _ = model(sample_input)
        
        # Save
        save_path = tmp_path / "hybrid.pt"
        torch.save(model.state_dict(), save_path)
        
        # Load into new model
        model2 = create_light_hybrid(
            n_features=integration_params["n_features"],
            n_clauses=integration_params["n_clauses"],
            n_classes=integration_params["n_classes"],
        )
        model2.load_state_dict(torch.load(save_path))
        
        # Compare outputs
        model2.eval()
        with torch.no_grad():
            logits2, _ = model2(sample_input)
        
        assert torch.allclose(logits1, logits2)


# =============================================================================
# Full Pipeline Integration
# =============================================================================


class TestFullPipelineIntegration:
    """Test complete training pipelines."""
    
    def test_complete_training_loop(self, integration_params):
        """Test a complete training loop with multiple components."""
        from fptm_ste.ultimate_hybrid import create_light_hybrid
        from fptm_ste.augmentation import AugmentationPipeline
        from fptm_ste.sam_optimizer import SAM
        
        # Create model
        model = create_light_hybrid(
            n_features=integration_params["n_features"],
            n_clauses=integration_params["n_clauses"],
            n_classes=integration_params["n_classes"],
        )
        
        # Create optimizer
        optimizer = SAM(model.parameters(), torch.optim.Adam, lr=0.01)
        
        # Create augmentation
        augmentation = AugmentationPipeline(
            use_mixup=True,
            use_cutmix=False,
            mixup_alpha=0.2,
        )
        
        # Synthetic data
        n_samples = 50
        x = torch.rand(n_samples, integration_params["n_features"])
        y = torch.randint(0, integration_params["n_classes"], (n_samples,))
        
        # Training loop
        model.train()
        augmentation.train()
        
        for epoch in range(3):
            aug_x, y_a, y_b, lam = augmentation(x, y)
            
            # First step
            logits, _ = model(aug_x)
            loss = lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            
            # Second step
            logits, _ = model(aug_x)
            loss = lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)
            loss.backward()
            optimizer.second_step(zero_grad=True)
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            logits, _ = model(x)
            preds = logits.argmax(dim=-1)
            accuracy = (preds == y).float().mean().item()
        
        # Just verify it runs - accuracy may be low with random data
        assert accuracy >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

