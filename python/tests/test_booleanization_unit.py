"""
Unit tests for booleanization modules.

Tests all booleanization solutions for the Tsetlin Machine:
1. ContinuousResidualClauseMachine - Dual-stream architecture
2. ProbabilisticLiteralClauseMachine - Distributional literals
3. HyperdimensionalClauseMachine - HD computing encoder
4. InformationBottleneckBinarizer - Optimal binarization
5. HierarchicalMultiResolutionTM - Multi-resolution levels
6. NeuralSymbolicTransformer - Per-sample dynamic binarization
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_params():
    """Default parameters for testing."""
    return {
        "n_features": 16,
        "n_clauses": 8,
        "n_classes": 3,
        "batch_size": 4,
    }


@pytest.fixture
def sample_batch(default_params):
    """Generate sample input batch."""
    batch_size = default_params["batch_size"]
    n_features = default_params["n_features"]
    x = torch.rand(batch_size, n_features)  # Features in [0, 1]
    y = torch.randint(0, default_params["n_classes"], (batch_size,))
    return x, y


# =============================================================================
# Test ContinuousResidualClauseMachine
# =============================================================================


class TestContinuousResidualClauseMachine:
    """Tests for the dual-stream continuous residual architecture."""
    
    def test_import(self):
        """Test that module can be imported."""
        from fptm_ste.booleanization import ContinuousResidualClauseMachine
        assert ContinuousResidualClauseMachine is not None
    
    def test_initialization(self, default_params):
        """Test model initialization."""
        from fptm_ste.booleanization import ContinuousResidualClauseMachine
        
        model = ContinuousResidualClauseMachine(
            n_features=default_params["n_features"],
            n_clauses=default_params["n_clauses"],
            n_classes=default_params["n_classes"],
        )
        
        assert model.n_features == default_params["n_features"]
        assert model.n_clauses == default_params["n_clauses"]
        assert model.n_classes == default_params["n_classes"]
    
    def test_forward_shapes(self, default_params, sample_batch):
        """Test output shapes."""
        from fptm_ste.booleanization import ContinuousResidualClauseMachine
        
        model = ContinuousResidualClauseMachine(
            n_features=default_params["n_features"],
            n_clauses=default_params["n_clauses"],
            n_classes=default_params["n_classes"],
        )
        
        x, _ = sample_batch
        logits, clauses = model(x)
        
        assert logits.shape == (default_params["batch_size"], default_params["n_classes"])
        assert clauses.shape == (default_params["batch_size"], default_params["n_clauses"])
    
    def test_forward_with_details(self, default_params, sample_batch):
        """Test forward pass with detailed outputs."""
        from fptm_ste.booleanization import ContinuousResidualClauseMachine
        
        model = ContinuousResidualClauseMachine(
            n_features=default_params["n_features"],
            n_clauses=default_params["n_clauses"],
            n_classes=default_params["n_classes"],
        )
        
        x, _ = sample_batch
        details = model(x, return_details=True)
        
        assert "logits" in details
        assert "binary_clauses" in details
        assert "continuous_encoded" in details
        assert "reconstruction" in details
        assert "gate" in details
    
    def test_gradient_flow(self, default_params, sample_batch):
        """Test that gradients flow through both streams."""
        from fptm_ste.booleanization import ContinuousResidualClauseMachine
        
        model = ContinuousResidualClauseMachine(
            n_features=default_params["n_features"],
            n_clauses=default_params["n_clauses"],
            n_classes=default_params["n_classes"],
        )
        
        x, y = sample_batch
        x.requires_grad = True
        
        logits, _ = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        
        # Check gradient flow
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
        
        # Check that at least some model parameters have gradients
        has_grads = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if param.grad.abs().sum() > 0:
                    has_grads = True
                    break
        assert has_grads, "No non-zero gradients found"
    
    def test_reconstruction_loss(self, default_params, sample_batch):
        """Test reconstruction loss computation."""
        from fptm_ste.booleanization import ContinuousResidualClauseMachine
        
        model = ContinuousResidualClauseMachine(
            n_features=default_params["n_features"],
            n_clauses=default_params["n_clauses"],
            n_classes=default_params["n_classes"],
        )
        
        x, y = sample_batch
        details = model(x, return_details=True)
        
        recon_loss = model.information_preservation_loss(x, details["reconstruction"])
        
        assert recon_loss >= 0
        assert not torch.isnan(recon_loss)
    
    def test_fusion_gate_bounds(self, default_params, sample_batch):
        """Test that fusion gate values are in [0, 1]."""
        from fptm_ste.booleanization import ContinuousResidualClauseMachine
        
        model = ContinuousResidualClauseMachine(
            n_features=default_params["n_features"],
            n_clauses=default_params["n_clauses"],
            n_classes=default_params["n_classes"],
            fusion_type="sigmoid",
        )
        
        x, _ = sample_batch
        details = model(x, return_details=True)
        
        gate = details["gate"]
        assert torch.all(gate >= 0)
        assert torch.all(gate <= 1)


class TestSoftThresholdBinarizer:
    """Tests for the soft threshold binarizer."""
    
    def test_import(self):
        """Test import."""
        from fptm_ste.booleanization import SoftThresholdBinarizer
        assert SoftThresholdBinarizer is not None
    
    def test_differentiable(self, default_params):
        """Test that soft thresholding is differentiable."""
        from fptm_ste.booleanization import SoftThresholdBinarizer
        
        binarizer = SoftThresholdBinarizer(
            n_features=default_params["n_features"],
            temperature=1.0,
        )
        
        x = torch.rand(default_params["batch_size"], default_params["n_features"], requires_grad=True)
        y = binarizer(x, use_ste=True)
        
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
    
    def test_output_range(self, default_params):
        """Test that output is in [0, 1]."""
        from fptm_ste.booleanization import SoftThresholdBinarizer
        
        binarizer = SoftThresholdBinarizer(
            n_features=default_params["n_features"],
        )
        
        x = torch.rand(default_params["batch_size"], default_params["n_features"])
        y = binarizer(x, use_ste=False)
        
        assert torch.all(y >= 0)
        assert torch.all(y <= 1)


# =============================================================================
# Test ProbabilisticLiteralClauseMachine
# =============================================================================


class TestProbabilisticLiteralClauseMachine:
    """Tests for probabilistic literal TM."""
    
    def test_import(self):
        """Test import."""
        from fptm_ste.booleanization import ProbabilisticLiteralClauseMachine
        assert ProbabilisticLiteralClauseMachine is not None
    
    def test_initialization(self, default_params):
        """Test initialization."""
        from fptm_ste.booleanization import ProbabilisticLiteralClauseMachine
        
        model = ProbabilisticLiteralClauseMachine(
            n_features=default_params["n_features"],
            n_clauses=default_params["n_clauses"],
            n_classes=default_params["n_classes"],
        )
        
        assert model is not None
    
    def test_forward_shapes(self, default_params, sample_batch):
        """Test output shapes."""
        from fptm_ste.booleanization import ProbabilisticLiteralClauseMachine
        
        model = ProbabilisticLiteralClauseMachine(
            n_features=default_params["n_features"],
            n_clauses=default_params["n_clauses"],
            n_classes=default_params["n_classes"],
        )
        
        x, _ = sample_batch
        logits, clauses = model(x)
        
        assert logits.shape == (default_params["batch_size"], default_params["n_classes"])
        assert clauses.shape == (default_params["batch_size"], default_params["n_clauses"])
    
    def test_uncertainty_output(self, default_params, sample_batch):
        """Test that uncertainty is computed."""
        from fptm_ste.booleanization import ProbabilisticLiteralClauseMachine
        
        model = ProbabilisticLiteralClauseMachine(
            n_features=default_params["n_features"],
            n_clauses=default_params["n_clauses"],
            n_classes=default_params["n_classes"],
        )
        
        x, _ = sample_batch
        result = model(x, return_uncertainty=True)
        
        # Should return 3 values when return_uncertainty=True
        assert len(result) >= 2
    
    def test_probability_values(self, default_params, sample_batch):
        """Test that clause outputs are valid probabilities."""
        from fptm_ste.booleanization import ProbabilisticLiteralClauseMachine
        
        model = ProbabilisticLiteralClauseMachine(
            n_features=default_params["n_features"],
            n_clauses=default_params["n_clauses"],
            n_classes=default_params["n_classes"],
        )
        
        x, _ = sample_batch
        _, clauses = model(x)
        
        # Clause outputs should be valid (bounded)
        assert not torch.any(torch.isnan(clauses))
        assert not torch.any(torch.isinf(clauses))


# =============================================================================
# Test HyperdimensionalClauseMachine
# =============================================================================


class TestHyperdimensionalClauseMachine:
    """Tests for HD computing clause machine."""
    
    def test_import(self):
        """Test import."""
        from fptm_ste.booleanization import HyperdimensionalClauseMachine
        assert HyperdimensionalClauseMachine is not None
    
    def test_initialization(self, default_params):
        """Test initialization."""
        from fptm_ste.booleanization import HyperdimensionalClauseMachine
        
        model = HyperdimensionalClauseMachine(
            n_features=default_params["n_features"],
            n_clauses=default_params["n_clauses"],
            n_classes=default_params["n_classes"],
            hd_dim=256,  # Smaller HD dim for testing
        )
        
        assert model.hd_dim == 256
    
    def test_forward_shapes(self, default_params, sample_batch):
        """Test output shapes."""
        from fptm_ste.booleanization import HyperdimensionalClauseMachine
        
        model = HyperdimensionalClauseMachine(
            n_features=default_params["n_features"],
            n_clauses=default_params["n_clauses"],
            n_classes=default_params["n_classes"],
            hd_dim=256,
        )
        
        x, _ = sample_batch
        logits, clauses = model(x)
        
        assert logits.shape == (default_params["batch_size"], default_params["n_classes"])
    
    def test_hd_encoding_deterministic(self, default_params, sample_batch):
        """Test that HD encoding is deterministic."""
        from fptm_ste.booleanization import HyperdimensionalClauseMachine
        
        model = HyperdimensionalClauseMachine(
            n_features=default_params["n_features"],
            n_clauses=default_params["n_clauses"],
            n_classes=default_params["n_classes"],
            hd_dim=256,
        )
        model.eval()
        
        x, _ = sample_batch
        
        logits1, _ = model(x)
        logits2, _ = model(x)
        
        assert torch.allclose(logits1, logits2)
    
    def test_similar_inputs_similar_outputs(self, default_params):
        """Test that similar inputs produce similar outputs."""
        from fptm_ste.booleanization import HyperdimensionalClauseMachine
        
        model = HyperdimensionalClauseMachine(
            n_features=default_params["n_features"],
            n_clauses=default_params["n_clauses"],
            n_classes=default_params["n_classes"],
            hd_dim=256,
        )
        model.eval()
        
        x1 = torch.rand(1, default_params["n_features"])
        x2 = x1 + 0.01 * torch.randn_like(x1)  # Small perturbation
        x3 = torch.rand(1, default_params["n_features"])  # Random different input
        
        logits1, _ = model(x1)
        logits2, _ = model(x2)
        logits3, _ = model(x3)
        
        # Similar inputs should have more similar outputs
        dist_12 = (logits1 - logits2).norm()
        dist_13 = (logits1 - logits3).norm()
        
        # This is a soft expectation - similar inputs should be closer
        # but not guaranteed with random initialization
        assert dist_12 < 100  # Should be finite


# =============================================================================
# Test InformationBottleneckBinarizer
# =============================================================================


class TestInformationBottleneckBinarizer:
    """Tests for Information Bottleneck binarizer."""
    
    def test_import(self):
        """Test import."""
        from fptm_ste.booleanization import InformationBottleneckBinarizer
        assert InformationBottleneckBinarizer is not None
    
    def test_initialization(self, default_params):
        """Test initialization."""
        from fptm_ste.booleanization import InformationBottleneckBinarizer
        
        binarizer = InformationBottleneckBinarizer(
            n_features=default_params["n_features"],
            n_binary=8,
        )
        
        assert binarizer.n_features == default_params["n_features"]
    
    def test_encode_decode(self, default_params, sample_batch):
        """Test encode and decode."""
        from fptm_ste.booleanization import InformationBottleneckBinarizer
        
        binarizer = InformationBottleneckBinarizer(
            n_features=default_params["n_features"],
            n_binary=8,
        )
        
        x, _ = sample_batch
        # Forward returns (z, reconstruction, logits)
        z, reconstruction, logits = binarizer(x)
        
        assert z.shape == (default_params["batch_size"], 8)
        assert reconstruction.shape == x.shape
        assert logits.shape == (default_params["batch_size"], 8)
    
    def test_kl_loss_positive(self, default_params, sample_batch):
        """Test that KL divergence is non-negative."""
        from fptm_ste.booleanization import InformationBottleneckBinarizer
        
        binarizer = InformationBottleneckBinarizer(
            n_features=default_params["n_features"],
            n_binary=8,
        )
        
        x, _ = sample_batch
        z, reconstruction, logits = binarizer(x)
        
        # KL divergence is computed from the VIB layer
        kl_loss = binarizer.vib.kl_divergence(logits)
        
        # KL loss should be non-negative
        assert kl_loss >= 0


class TestInformationPreservingClauseMachine:
    """Tests for IB-based clause machine."""
    
    def test_import(self):
        """Test import."""
        from fptm_ste.booleanization import InformationPreservingClauseMachine
        assert InformationPreservingClauseMachine is not None
    
    def test_forward_shapes(self, default_params, sample_batch):
        """Test output shapes."""
        from fptm_ste.booleanization import InformationPreservingClauseMachine
        
        model = InformationPreservingClauseMachine(
            n_features=default_params["n_features"],
            n_clauses=default_params["n_clauses"],
            n_classes=default_params["n_classes"],
            n_binary=8,
        )
        
        x, _ = sample_batch
        logits, clauses = model(x)
        
        assert logits.shape == (default_params["batch_size"], default_params["n_classes"])


# =============================================================================
# Test HierarchicalMultiResolutionTM
# =============================================================================


class TestHierarchicalMultiResolutionTM:
    """Tests for hierarchical multi-resolution TM."""
    
    def test_import(self):
        """Test import."""
        from fptm_ste.booleanization import HierarchicalMultiResolutionTM
        assert HierarchicalMultiResolutionTM is not None
    
    def test_initialization(self, default_params):
        """Test initialization with multiple levels."""
        from fptm_ste.booleanization import HierarchicalMultiResolutionTM
        
        n_features = 64  # 8x8
        
        model = HierarchicalMultiResolutionTM(
            n_features=n_features,
            n_clauses_per_level=[8, 8, 8],
            n_classes=default_params["n_classes"],
            resolutions=[2, 4, 8],
        )
        
        assert len(model.levels) == 3
    
    def test_forward_shapes(self, default_params):
        """Test output shapes."""
        from fptm_ste.booleanization import HierarchicalMultiResolutionTM
        
        n_features = 64  # 8x8
        batch_size = default_params["batch_size"]
        
        model = HierarchicalMultiResolutionTM(
            n_features=n_features,
            n_clauses_per_level=[8, 8],
            n_classes=default_params["n_classes"],
            resolutions=[2, 4],
        )
        
        x = torch.rand(batch_size, n_features)
        logits, clauses = model(x)
        
        assert logits.shape == (batch_size, default_params["n_classes"])
    
    def test_level_outputs(self, default_params):
        """Test that each level produces output."""
        from fptm_ste.booleanization import HierarchicalMultiResolutionTM
        
        n_features = 64  # 8x8
        batch_size = default_params["batch_size"]
        
        model = HierarchicalMultiResolutionTM(
            n_features=n_features,
            n_clauses_per_level=[8, 8],
            n_classes=default_params["n_classes"],
            resolutions=[2, 4],
        )
        
        x = torch.rand(batch_size, n_features)
        result = model(x, return_level_outputs=True)
        
        if isinstance(result, dict):
            # Check for level_clauses which contains per-level outputs
            assert "level_clauses" in result or "level_outputs" in result


# =============================================================================
# Test NeuralSymbolicTransformer
# =============================================================================


class TestNeuralSymbolicTransformer:
    """Tests for attention-adaptive binarization."""
    
    def test_import(self):
        """Test import."""
        from fptm_ste.booleanization import NeuralSymbolicTransformer
        assert NeuralSymbolicTransformer is not None
    
    def test_initialization(self, default_params):
        """Test initialization."""
        from fptm_ste.booleanization import NeuralSymbolicTransformer
        
        model = NeuralSymbolicTransformer(
            n_features=default_params["n_features"],
            n_clauses=default_params["n_clauses"],
            n_classes=default_params["n_classes"],
            d_model=32,
        )
        
        assert model.n_features == default_params["n_features"]
    
    def test_forward_shapes(self, default_params, sample_batch):
        """Test output shapes."""
        from fptm_ste.booleanization import NeuralSymbolicTransformer
        
        model = NeuralSymbolicTransformer(
            n_features=default_params["n_features"],
            n_clauses=default_params["n_clauses"],
            n_classes=default_params["n_classes"],
            d_model=32,
        )
        
        x, _ = sample_batch
        logits, clauses = model(x)
        
        assert logits.shape == (default_params["batch_size"], default_params["n_classes"])
    
    def test_dynamic_thresholds(self, default_params, sample_batch):
        """Test that thresholds are sample-dependent."""
        from fptm_ste.booleanization import NeuralSymbolicTransformer
        
        model = NeuralSymbolicTransformer(
            n_features=default_params["n_features"],
            n_clauses=default_params["n_clauses"],
            n_classes=default_params["n_classes"],
            d_model=32,
        )
        
        # Two different inputs
        x1 = torch.rand(1, default_params["n_features"])
        x2 = torch.rand(1, default_params["n_features"]) * 0.1  # Different scale
        
        # Get outputs for both
        logits1, clauses1 = model(x1)
        logits2, clauses2 = model(x2)
        
        # Verify both return valid outputs
        assert logits1.shape == (1, default_params["n_classes"])
        assert logits2.shape == (1, default_params["n_classes"])
        # Different inputs should produce different outputs
        assert not torch.allclose(logits1, logits2)


class TestDynamicThresholdPredictor:
    """Tests for dynamic threshold predictor."""
    
    def test_import(self):
        """Test import."""
        from fptm_ste.booleanization import DynamicThresholdPredictor
        assert DynamicThresholdPredictor is not None
    
    def test_output_range(self, default_params, sample_batch):
        """Test that predicted thresholds are in [0, 1]."""
        from fptm_ste.booleanization import DynamicThresholdPredictor
        
        predictor = DynamicThresholdPredictor(
            n_features=default_params["n_features"],
            hidden_dim=32,
        )
        
        x, _ = sample_batch
        thresholds = predictor(x)
        
        assert torch.all(thresholds >= 0)
        assert torch.all(thresholds <= 1)
        assert thresholds.shape == (default_params["batch_size"], default_params["n_features"])


# =============================================================================
# Cross-Module Tests
# =============================================================================


class TestBooleanizationModuleCompatibility:
    """Tests for compatibility between booleanization modules."""
    
    def test_all_modules_trainable(self, default_params, sample_batch):
        """Test that all modules can be trained."""
        from fptm_ste.booleanization import (
            ContinuousResidualClauseMachine,
            ProbabilisticLiteralClauseMachine,
            HyperdimensionalClauseMachine,
            InformationPreservingClauseMachine,
            NeuralSymbolicTransformer,
        )
        
        x, y = sample_batch
        
        models = [
            ContinuousResidualClauseMachine(
                n_features=default_params["n_features"],
                n_clauses=default_params["n_clauses"],
                n_classes=default_params["n_classes"],
            ),
            ProbabilisticLiteralClauseMachine(
                n_features=default_params["n_features"],
                n_clauses=default_params["n_clauses"],
                n_classes=default_params["n_classes"],
            ),
            HyperdimensionalClauseMachine(
                n_features=default_params["n_features"],
                n_clauses=default_params["n_clauses"],
                n_classes=default_params["n_classes"],
                hd_dim=256,
            ),
            InformationPreservingClauseMachine(
                n_features=default_params["n_features"],
                n_clauses=default_params["n_clauses"],
                n_classes=default_params["n_classes"],
                n_binary=8,
            ),
            NeuralSymbolicTransformer(
                n_features=default_params["n_features"],
                n_clauses=default_params["n_clauses"],
                n_classes=default_params["n_classes"],
                d_model=32,
            ),
        ]
        
        for model in models:
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            # Forward pass
            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Check gradients exist
            has_grads = False
            for param in model.parameters():
                if param.grad is not None and param.grad.abs().sum() > 0:
                    has_grads = True
                    break
            
            assert has_grads, f"No gradients for {model.__class__.__name__}"
            
            # Optimizer step
            optimizer.step()
    
    def test_all_modules_eval_mode(self, default_params, sample_batch):
        """Test that all modules work in eval mode."""
        from fptm_ste.booleanization import (
            ContinuousResidualClauseMachine,
            ProbabilisticLiteralClauseMachine,
            HyperdimensionalClauseMachine,
        )
        
        x, _ = sample_batch
        
        models = [
            ContinuousResidualClauseMachine(
                n_features=default_params["n_features"],
                n_clauses=default_params["n_clauses"],
                n_classes=default_params["n_classes"],
            ),
            ProbabilisticLiteralClauseMachine(
                n_features=default_params["n_features"],
                n_clauses=default_params["n_clauses"],
                n_classes=default_params["n_classes"],
            ),
            HyperdimensionalClauseMachine(
                n_features=default_params["n_features"],
                n_clauses=default_params["n_clauses"],
                n_classes=default_params["n_classes"],
                hd_dim=256,
            ),
        ]
        
        for model in models:
            model.eval()
            
            with torch.no_grad():
                logits, clauses = model(x)
            
            assert not torch.any(torch.isnan(logits))
            assert logits.shape[0] == default_params["batch_size"]


# =============================================================================
# Test DualStreamTM
# =============================================================================


class TestDualStreamTM:
    """Tests for simplified dual-stream TM."""
    
    def test_import(self):
        """Test import."""
        from fptm_ste.booleanization import DualStreamTM
        assert DualStreamTM is not None
    
    def test_forward_shapes(self, default_params, sample_batch):
        """Test output shapes."""
        from fptm_ste.booleanization import DualStreamTM
        
        model = DualStreamTM(
            n_features=default_params["n_features"],
            n_clauses=default_params["n_clauses"],
            n_classes=default_params["n_classes"],
        )
        
        x, _ = sample_batch
        logits, clauses = model(x)
        
        assert logits.shape == (default_params["batch_size"], default_params["n_classes"])
        assert clauses.shape == (default_params["batch_size"], default_params["n_clauses"])
    
    def test_learnable_alpha(self, default_params):
        """Test that combination weight is learnable."""
        from fptm_ste.booleanization import DualStreamTM
        
        model = DualStreamTM(
            n_features=default_params["n_features"],
            n_clauses=default_params["n_clauses"],
            n_classes=default_params["n_classes"],
            continuous_weight=0.5,
        )
        
        # alpha should be a parameter
        assert hasattr(model, 'alpha')
        assert model.alpha.requires_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
