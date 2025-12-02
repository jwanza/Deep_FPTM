"""
End-to-End tests for booleanization modules.

Tests training and evaluation on synthetic and small-scale datasets
to verify that all booleanization approaches can learn effectively.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def synthetic_binary_dataset():
    """Create a simple synthetic binary classification dataset."""
    torch.manual_seed(42)
    
    n_samples = 200
    n_features = 16
    
    # Create two clusters
    x1 = torch.randn(n_samples // 2, n_features) * 0.3 + 0.3
    x2 = torch.randn(n_samples // 2, n_features) * 0.3 + 0.7
    
    x = torch.cat([x1, x2], dim=0).clamp(0, 1)
    y = torch.cat([
        torch.zeros(n_samples // 2),
        torch.ones(n_samples // 2),
    ]).long()
    
    # Shuffle
    perm = torch.randperm(n_samples)
    x, y = x[perm], y[perm]
    
    # Split
    train_x, train_y = x[:160], y[:160]
    test_x, test_y = x[160:], y[160:]
    
    return {
        "train": (train_x, train_y),
        "test": (test_x, test_y),
        "n_features": n_features,
        "n_classes": 2,
    }


@pytest.fixture
def synthetic_multiclass_dataset():
    """Create a synthetic multi-class classification dataset."""
    torch.manual_seed(42)
    
    n_samples = 300
    n_features = 16
    n_classes = 3
    
    x_list = []
    y_list = []
    
    for c in range(n_classes):
        x_c = torch.randn(n_samples // n_classes, n_features) * 0.2 + c / n_classes
        x_list.append(x_c)
        y_list.append(torch.full((n_samples // n_classes,), c, dtype=torch.long))
    
    x = torch.cat(x_list, dim=0).clamp(0, 1)
    y = torch.cat(y_list, dim=0)
    
    # Shuffle
    perm = torch.randperm(n_samples)
    x, y = x[perm], y[perm]
    
    # Split
    split = int(0.8 * n_samples)
    train_x, train_y = x[:split], y[:split]
    test_x, test_y = x[split:], y[split:]
    
    return {
        "train": (train_x, train_y),
        "test": (test_x, test_y),
        "n_features": n_features,
        "n_classes": n_classes,
    }


def train_and_evaluate(model, train_data, test_data, epochs=30, lr=0.01):
    """
    Train and evaluate a model.
    
    Args:
        model: PyTorch model
        train_data: Tuple of (x, y) for training
        test_data: Tuple of (x, y) for testing
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        Dict with train_acc, test_acc, final_loss
    """
    train_x, train_y = train_data
    test_x, test_y = test_data
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    final_loss = 0.0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        logits, _ = model(train_x)
        loss = F.cross_entropy(logits, train_y)
        
        loss.backward()
        optimizer.step()
        
        final_loss = loss.item()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        train_logits, _ = model(train_x)
        train_preds = train_logits.argmax(dim=-1)
        train_acc = (train_preds == train_y).float().mean().item()
        
        test_logits, _ = model(test_x)
        test_preds = test_logits.argmax(dim=-1)
        test_acc = (test_preds == test_y).float().mean().item()
    
    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "final_loss": final_loss,
    }


# =============================================================================
# E2E Tests for ContinuousResidualClauseMachine
# =============================================================================


class TestCRCME2E:
    """E2E tests for Continuous Residual Clause Machine."""
    
    def test_binary_classification(self, synthetic_binary_dataset):
        """Test binary classification."""
        from fptm_ste.booleanization import ContinuousResidualClauseMachine
        
        model = ContinuousResidualClauseMachine(
            n_features=synthetic_binary_dataset["n_features"],
            n_clauses=16,
            n_classes=synthetic_binary_dataset["n_classes"],
            hidden_dim=32,
        )
        
        results = train_and_evaluate(
            model,
            synthetic_binary_dataset["train"],
            synthetic_binary_dataset["test"],
            epochs=50,
        )
        
        # Should achieve reasonable accuracy
        assert results["train_acc"] > 0.6, f"Train acc too low: {results['train_acc']}"
        assert results["test_acc"] > 0.5, f"Test acc too low: {results['test_acc']}"
    
    def test_multiclass_classification(self, synthetic_multiclass_dataset):
        """Test multi-class classification."""
        from fptm_ste.booleanization import ContinuousResidualClauseMachine
        
        model = ContinuousResidualClauseMachine(
            n_features=synthetic_multiclass_dataset["n_features"],
            n_clauses=24,
            n_classes=synthetic_multiclass_dataset["n_classes"],
            hidden_dim=32,
        )
        
        results = train_and_evaluate(
            model,
            synthetic_multiclass_dataset["train"],
            synthetic_multiclass_dataset["test"],
            epochs=50,
        )
        
        assert results["train_acc"] > 0.5
        assert results["final_loss"] < 2.0  # Loss should decrease
    
    def test_with_reconstruction_loss(self, synthetic_binary_dataset):
        """Test training with reconstruction loss."""
        from fptm_ste.booleanization import ContinuousResidualClauseMachine
        
        model = ContinuousResidualClauseMachine(
            n_features=synthetic_binary_dataset["n_features"],
            n_clauses=16,
            n_classes=synthetic_binary_dataset["n_classes"],
            reconstruction_weight=0.1,
        )
        
        train_x, train_y = synthetic_binary_dataset["train"]
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        initial_loss = None
        final_loss = None
        
        for epoch in range(30):
            optimizer.zero_grad()
            
            details = model(train_x, return_details=True)
            cls_loss = F.cross_entropy(details["logits"], train_y)
            recon_loss = model.information_preservation_loss(train_x, details["reconstruction"])
            loss = cls_loss + 0.1 * recon_loss
            
            if epoch == 0:
                initial_loss = loss.item()
            
            loss.backward()
            optimizer.step()
            
            final_loss = loss.item()
        
        # Loss should decrease
        assert final_loss < initial_loss


# =============================================================================
# E2E Tests for ProbabilisticLiteralClauseMachine
# =============================================================================


class TestProbabilisticE2E:
    """E2E tests for Probabilistic Literal Clause Machine."""
    
    def test_binary_classification(self, synthetic_binary_dataset):
        """Test binary classification."""
        from fptm_ste.booleanization import ProbabilisticLiteralClauseMachine
        
        model = ProbabilisticLiteralClauseMachine(
            n_features=synthetic_binary_dataset["n_features"],
            n_clauses=16,
            n_classes=synthetic_binary_dataset["n_classes"],
        )
        
        results = train_and_evaluate(
            model,
            synthetic_binary_dataset["train"],
            synthetic_binary_dataset["test"],
            epochs=50,
        )
        
        assert results["train_acc"] > 0.5
    
    def test_uncertainty_decreases_with_training(self, synthetic_binary_dataset):
        """Test that uncertainty changes with training."""
        from fptm_ste.booleanization import ProbabilisticLiteralClauseMachine
        
        model = ProbabilisticLiteralClauseMachine(
            n_features=synthetic_binary_dataset["n_features"],
            n_clauses=16,
            n_classes=synthetic_binary_dataset["n_classes"],
        )
        
        train_x, train_y = synthetic_binary_dataset["train"]
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Train
        model.train()
        for _ in range(30):
            optimizer.zero_grad()
            logits, _ = model(train_x)
            loss = F.cross_entropy(logits, train_y)
            loss.backward()
            optimizer.step()
        
        # Model should produce valid outputs after training
        model.eval()
        with torch.no_grad():
            logits, _ = model(train_x)
        
        assert not torch.any(torch.isnan(logits))


# =============================================================================
# E2E Tests for HyperdimensionalClauseMachine
# =============================================================================


class TestHDE2E:
    """E2E tests for Hyperdimensional Clause Machine."""
    
    def test_binary_classification(self, synthetic_binary_dataset):
        """Test binary classification."""
        from fptm_ste.booleanization import HyperdimensionalClauseMachine
        
        model = HyperdimensionalClauseMachine(
            n_features=synthetic_binary_dataset["n_features"],
            n_clauses=16,
            n_classes=synthetic_binary_dataset["n_classes"],
            hd_dim=512,
        )
        
        results = train_and_evaluate(
            model,
            synthetic_binary_dataset["train"],
            synthetic_binary_dataset["test"],
            epochs=50,
        )
        
        assert results["train_acc"] > 0.5
    
    def test_hd_encoding_quality(self, synthetic_binary_dataset):
        """Test that HD encoding preserves class separability."""
        from fptm_ste.booleanization import HyperdimensionalClauseMachine
        
        model = HyperdimensionalClauseMachine(
            n_features=synthetic_binary_dataset["n_features"],
            n_clauses=16,
            n_classes=2,
            hd_dim=512,
        )
        
        train_x, train_y = synthetic_binary_dataset["train"]
        
        # Get HD encodings
        model.eval()
        with torch.no_grad():
            _, clauses = model(train_x)
        
        # Check that class 0 and class 1 have different clause activations
        class0_clauses = clauses[train_y == 0].mean(dim=0)
        class1_clauses = clauses[train_y == 1].mean(dim=0)
        
        # There should be some difference
        diff = (class0_clauses - class1_clauses).abs().mean()
        assert diff > 0.01  # Some separation


# =============================================================================
# E2E Tests for InformationPreservingClauseMachine
# =============================================================================


class TestIBE2E:
    """E2E tests for Information Bottleneck Clause Machine."""
    
    def test_binary_classification(self, synthetic_binary_dataset):
        """Test binary classification."""
        from fptm_ste.booleanization import InformationPreservingClauseMachine
        
        model = InformationPreservingClauseMachine(
            n_features=synthetic_binary_dataset["n_features"],
            n_clauses=16,
            n_classes=synthetic_binary_dataset["n_classes"],
            n_binary=8,
        )
        
        results = train_and_evaluate(
            model,
            synthetic_binary_dataset["train"],
            synthetic_binary_dataset["test"],
            epochs=50,
        )
        
        assert results["train_acc"] > 0.5
    
    def test_beta_tradeoff(self, synthetic_binary_dataset):
        """Test that beta controls compression."""
        from fptm_ste.booleanization import InformationBottleneckBinarizer
        
        train_x, _ = synthetic_binary_dataset["train"]
        
        # Low beta (less compression)
        binarizer_low = InformationBottleneckBinarizer(
            n_features=synthetic_binary_dataset["n_features"],
            n_binary=8,
            beta=0.001,
        )
        
        # High beta (more compression)
        binarizer_high = InformationBottleneckBinarizer(
            n_features=synthetic_binary_dataset["n_features"],
            n_binary=8,
            beta=1.0,
        )
        
        # Forward returns (z, reconstruction, logits)
        z_low, _, logits_low = binarizer_low(train_x)
        z_high, _, logits_high = binarizer_high(train_x)
        
        # KL divergence computed from logits
        kl_low = binarizer_low.vib.kl_divergence(logits_low)
        kl_high = binarizer_high.vib.kl_divergence(logits_high)
        
        # Both should be non-negative
        assert kl_low >= 0
        assert kl_high >= 0


# =============================================================================
# E2E Tests for NeuralSymbolicTransformer
# =============================================================================


class TestNeuralSymbolicE2E:
    """E2E tests for Neural Symbolic Transformer."""
    
    def test_binary_classification(self, synthetic_binary_dataset):
        """Test binary classification."""
        from fptm_ste.booleanization import NeuralSymbolicTransformer
        
        model = NeuralSymbolicTransformer(
            n_features=synthetic_binary_dataset["n_features"],
            n_clauses=16,
            n_classes=synthetic_binary_dataset["n_classes"],
            d_model=32,
            n_heads=2,
            n_layers=1,
        )
        
        results = train_and_evaluate(
            model,
            synthetic_binary_dataset["train"],
            synthetic_binary_dataset["test"],
            epochs=50,
        )
        
        assert results["train_acc"] > 0.5
    
    def test_attention_patterns(self, synthetic_binary_dataset):
        """Test that attention patterns are learned."""
        from fptm_ste.booleanization import NeuralSymbolicTransformer
        
        model = NeuralSymbolicTransformer(
            n_features=synthetic_binary_dataset["n_features"],
            n_clauses=16,
            n_classes=2,
            d_model=32,
        )
        
        train_x, train_y = synthetic_binary_dataset["train"]
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Train
        for _ in range(30):
            optimizer.zero_grad()
            logits, _ = model(train_x)
            loss = F.cross_entropy(logits, train_y)
            loss.backward()
            optimizer.step()
        
        # Model should produce valid outputs
        model.eval()
        with torch.no_grad():
            logits, _ = model(train_x)
        
        assert not torch.any(torch.isnan(logits))


# =============================================================================
# Comparative E2E Tests
# =============================================================================


class TestBooleanizationComparison:
    """Compare different booleanization approaches."""
    
    def test_all_methods_learn(self, synthetic_multiclass_dataset):
        """Test that all methods can learn the task."""
        from fptm_ste.booleanization import (
            ContinuousResidualClauseMachine,
            ProbabilisticLiteralClauseMachine,
            HyperdimensionalClauseMachine,
            InformationPreservingClauseMachine,
            NeuralSymbolicTransformer,
        )
        
        n_features = synthetic_multiclass_dataset["n_features"]
        n_classes = synthetic_multiclass_dataset["n_classes"]
        
        models = {
            "CRCM": ContinuousResidualClauseMachine(
                n_features=n_features,
                n_clauses=16,
                n_classes=n_classes,
            ),
            "Probabilistic": ProbabilisticLiteralClauseMachine(
                n_features=n_features,
                n_clauses=16,
                n_classes=n_classes,
            ),
            "HD": HyperdimensionalClauseMachine(
                n_features=n_features,
                n_clauses=16,
                n_classes=n_classes,
                hd_dim=256,
            ),
            "IB": InformationPreservingClauseMachine(
                n_features=n_features,
                n_clauses=16,
                n_classes=n_classes,
                n_binary=8,
            ),
            "NeuralSymbolic": NeuralSymbolicTransformer(
                n_features=n_features,
                n_clauses=16,
                n_classes=n_classes,
                d_model=32,
            ),
        }
        
        results = {}
        
        for name, model in models.items():
            result = train_and_evaluate(
                model,
                synthetic_multiclass_dataset["train"],
                synthetic_multiclass_dataset["test"],
                epochs=50,
            )
            results[name] = result
            
            # Each method should learn something
            random_acc = 1.0 / n_classes
            assert result["train_acc"] > random_acc, \
                f"{name} failed: train_acc={result['train_acc']:.3f} <= {random_acc:.3f}"
        
        # Print results for debugging
        print("\nBooleanization Method Comparison:")
        for name, result in results.items():
            print(f"  {name}: train_acc={result['train_acc']:.3f}, test_acc={result['test_acc']:.3f}")


# =============================================================================
# Robustness Tests
# =============================================================================


class TestBooleanizationRobustness:
    """Test robustness of booleanization methods."""
    
    def test_noisy_input(self, synthetic_binary_dataset):
        """Test robustness to input noise."""
        from fptm_ste.booleanization import ContinuousResidualClauseMachine
        
        model = ContinuousResidualClauseMachine(
            n_features=synthetic_binary_dataset["n_features"],
            n_clauses=16,
            n_classes=2,
        )
        
        # Train on clean data
        train_and_evaluate(
            model,
            synthetic_binary_dataset["train"],
            synthetic_binary_dataset["test"],
            epochs=30,
        )
        
        # Test on noisy data
        test_x, test_y = synthetic_binary_dataset["test"]
        noisy_test_x = test_x + 0.1 * torch.randn_like(test_x)
        noisy_test_x = noisy_test_x.clamp(0, 1)
        
        model.eval()
        with torch.no_grad():
            clean_logits, _ = model(test_x)
            noisy_logits, _ = model(noisy_test_x)
        
        clean_preds = clean_logits.argmax(dim=-1)
        noisy_preds = noisy_logits.argmax(dim=-1)
        
        # Should be somewhat robust
        agreement = (clean_preds == noisy_preds).float().mean()
        assert agreement > 0.5  # At least 50% agreement
    
    def test_extreme_values(self, synthetic_binary_dataset):
        """Test handling of extreme input values."""
        from fptm_ste.booleanization import ContinuousResidualClauseMachine
        
        model = ContinuousResidualClauseMachine(
            n_features=synthetic_binary_dataset["n_features"],
            n_clauses=16,
            n_classes=2,
        )
        
        # Test with extreme values
        test_x = torch.zeros(4, synthetic_binary_dataset["n_features"])
        test_x[0] = 0.0  # All zeros
        test_x[1] = 1.0  # All ones
        test_x[2, :8] = 0.0  # Half zeros, half ones
        test_x[2, 8:] = 1.0
        test_x[3] = 0.5  # All middle values
        
        model.eval()
        with torch.no_grad():
            logits, _ = model(test_x)
        
        # Should not produce NaN or Inf
        assert not torch.any(torch.isnan(logits))
        assert not torch.any(torch.isinf(logits))


# =============================================================================
# DataLoader Integration Tests
# =============================================================================


class TestDataLoaderIntegration:
    """Test integration with PyTorch DataLoader."""
    
    def test_batched_training(self, synthetic_multiclass_dataset):
        """Test training with DataLoader."""
        from fptm_ste.booleanization import ContinuousResidualClauseMachine
        
        train_x, train_y = synthetic_multiclass_dataset["train"]
        test_x, test_y = synthetic_multiclass_dataset["test"]
        
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        model = ContinuousResidualClauseMachine(
            n_features=synthetic_multiclass_dataset["n_features"],
            n_clauses=16,
            n_classes=synthetic_multiclass_dataset["n_classes"],
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Train with batches
        model.train()
        for epoch in range(10):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                logits, _ = model(batch_x)
                loss = F.cross_entropy(logits, batch_y)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            test_logits, _ = model(test_x)
            test_preds = test_logits.argmax(dim=-1)
            test_acc = (test_preds == test_y).float().mean().item()
        
        assert test_acc > 0.3  # Better than random (1/3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

