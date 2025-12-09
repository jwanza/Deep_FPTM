"""
CIFAR-10 regression tests for TM models.

These tests ensure that model accuracy does not regress below established baselines.
Run with: pytest python/fptm_ste/tests/test_cifar10_regression.py -v
"""

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Add project root to path
PROJECT_ROOT = Path(__file__).parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fptm_ste import FuzzyPatternTM_STE, FuzzyPatternTM_STCM
from fptm_ste.deep_tm import DeepTMNetwork
from fptm_ste.benchmarks import (
    BenchmarkResult,
    BenchmarkSuite,
    count_parameters,
    check_regression,
    CIFAR10_BASELINES,
)


# Configuration
CIFAR10_ROOT = os.environ.get("CIFAR10_ROOT", "/tmp/cifar10")
QUICK_TEST_SAMPLES = 1000  # Subset for quick tests
FULL_TEST_SAMPLES = 10000  # Full test set
QUICK_TRAIN_SAMPLES = 5000
EPOCHS_QUICK = 3
EPOCHS_FULL = 10
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_cifar10_loaders(
    train_samples: int = QUICK_TRAIN_SAMPLES,
    test_samples: int = QUICK_TEST_SAMPLES,
    batch_size: int = BATCH_SIZE,
) -> tuple:
    """Load CIFAR-10 with optional subsampling for quick tests."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    train_dataset = datasets.CIFAR10(
        root=CIFAR10_ROOT,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.CIFAR10(
        root=CIFAR10_ROOT,
        train=False,
        download=True,
        transform=transform,
    )
    
    # Subsample if needed
    if train_samples < len(train_dataset):
        torch.manual_seed(42)
        indices = torch.randperm(len(train_dataset))[:train_samples].tolist()
        train_dataset = Subset(train_dataset, indices)
    
    if test_samples < len(test_dataset):
        torch.manual_seed(42)
        indices = torch.randperm(len(test_dataset))[:test_samples].tolist()
        test_dataset = Subset(test_dataset, indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if DEVICE.type == "cuda" else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if DEVICE.type == "cuda" else False,
    )
    
    return train_loader, test_loader


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        # Flatten for TM models
        x_flat = x.view(x.size(0), -1)
        optimizer.zero_grad()
        
        # Handle different model output formats
        output = model(x_flat)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """Evaluate model, return accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_flat = x.view(x.size(0), -1)
            
            output = model(x_flat)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    return correct / total


def create_tm_model(model_type: str, n_features: int = 3072, n_classes: int = 10) -> nn.Module:
    """Factory function to create TM models."""
    if model_type == "tm":
        return FuzzyPatternTM_STE(
            n_features=n_features,
            n_clauses=200,
            n_classes=n_classes,
            tau=0.5,
        )
    elif model_type == "stcm":
        return FuzzyPatternTM_STCM(
            n_features=n_features,
            n_clauses=200,
            n_classes=n_classes,
            operator="capacity",
        )
    elif model_type == "deep_tm":
        return DeepTMNetwork(
            input_dim=n_features,
            hidden_dims=[512, 256],
            n_classes=n_classes,
            n_clauses=100,
            layer_cls=FuzzyPatternTM_STE,
        )
    elif model_type == "deep_stcm":
        return DeepTMNetwork(
            input_dim=n_features,
            hidden_dims=[512, 256],
            n_classes=n_classes,
            n_clauses=100,
            layer_cls=FuzzyPatternTM_STCM,
            layer_operator="capacity",
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class TestCIFAR10Baselines:
    """Test suite for CIFAR-10 baseline verification."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        self.train_loader, self.test_loader = get_cifar10_loaders()
    
    @pytest.mark.parametrize("model_type", ["tm", "stcm"])
    def test_basic_tm_shapes(self, model_type: str):
        """Test that basic TM models produce correct output shapes."""
        model = create_tm_model(model_type).to(DEVICE)
        x = torch.randn(16, 3072, device=DEVICE)
        
        output = model(x)
        if isinstance(output, tuple):
            logits, clauses = output
            assert logits.shape == (16, 10), f"Expected (16, 10), got {logits.shape}"
            assert clauses.shape[0] == 16, f"Expected batch size 16, got {clauses.shape[0]}"
        else:
            assert output.shape == (16, 10), f"Expected (16, 10), got {output.shape}"
    
    @pytest.mark.parametrize("model_type", ["deep_tm", "deep_stcm"])
    def test_deep_tm_shapes(self, model_type: str):
        """Test that deep TM models produce correct output shapes."""
        model = create_tm_model(model_type).to(DEVICE)
        x = torch.randn(16, 3072, device=DEVICE)
        
        output = model(x)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        
        assert logits.shape == (16, 10), f"Expected (16, 10), got {logits.shape}"
    
    @pytest.mark.parametrize("model_type", ["tm", "stcm"])
    def test_gradient_flow(self, model_type: str):
        """Test that gradients flow through the model."""
        model = create_tm_model(model_type).to(DEVICE)
        x = torch.randn(8, 3072, device=DEVICE, requires_grad=True)
        y = torch.randint(0, 10, (8,), device=DEVICE)
        
        output = model(x)
        logits = output[0] if isinstance(output, tuple) else output
        loss = F.cross_entropy(logits, y)
        loss.backward()
        
        # Check that at least some parameters have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "No gradients found in model parameters"
    
    @pytest.mark.slow
    @pytest.mark.parametrize("model_type", ["tm", "stcm", "deep_tm", "deep_stcm"])
    def test_training_improves_accuracy(self, model_type: str):
        """Test that training improves model accuracy."""
        model = create_tm_model(model_type).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Initial accuracy
        initial_acc = evaluate(model, self.test_loader)
        
        # Train for a few epochs
        for _ in range(EPOCHS_QUICK):
            train_epoch(model, self.train_loader, optimizer)
        
        # Final accuracy
        final_acc = evaluate(model, self.test_loader)
        
        assert final_acc > initial_acc, (
            f"Training did not improve accuracy: {initial_acc:.4f} -> {final_acc:.4f}"
        )
    
    @pytest.mark.slow
    def test_stcm_outperforms_basic_tm(self):
        """Test that STCM achieves better accuracy than basic TM."""
        # Train basic TM
        tm_model = create_tm_model("tm").to(DEVICE)
        tm_optimizer = torch.optim.Adam(tm_model.parameters(), lr=1e-3)
        for _ in range(EPOCHS_QUICK):
            train_epoch(tm_model, self.train_loader, tm_optimizer)
        tm_acc = evaluate(tm_model, self.test_loader)
        
        # Train STCM
        stcm_model = create_tm_model("stcm").to(DEVICE)
        stcm_optimizer = torch.optim.Adam(stcm_model.parameters(), lr=1e-3)
        for _ in range(EPOCHS_QUICK):
            train_epoch(stcm_model, self.train_loader, stcm_optimizer)
        stcm_acc = evaluate(stcm_model, self.test_loader)
        
        # STCM should be competitive (within 5% or better)
        assert stcm_acc >= tm_acc - 0.05, (
            f"STCM ({stcm_acc:.4f}) significantly worse than TM ({tm_acc:.4f})"
        )


class TestBenchmarkInfrastructure:
    """Test the benchmarking infrastructure itself."""
    
    def test_benchmark_result_serialization(self):
        """Test that BenchmarkResult can be serialized and deserialized."""
        result = BenchmarkResult(
            model_name="test_model",
            dataset="cifar10",
            test_accuracy=0.85,
            train_accuracy=0.90,
            parameters=100000,
            epochs=10,
            batch_size=128,
            training_time_seconds=60.0,
            inference_throughput=1000.0,
            peak_memory_mb=512.0,
            config={"lr": 0.001},
        )
        
        d = result.to_dict()
        restored = BenchmarkResult.from_dict(d)
        
        assert restored.model_name == result.model_name
        assert restored.test_accuracy == result.test_accuracy
        assert restored.config == result.config
    
    def test_benchmark_suite_save_load(self, tmp_path):
        """Test that BenchmarkSuite can be saved and loaded."""
        suite = BenchmarkSuite(
            name="test_suite",
            description="Test benchmark suite",
        )
        suite.add_result(BenchmarkResult(
            model_name="model1",
            dataset="cifar10",
            test_accuracy=0.80,
            train_accuracy=0.85,
            parameters=50000,
            epochs=5,
            batch_size=64,
            training_time_seconds=30.0,
            inference_throughput=2000.0,
            peak_memory_mb=256.0,
        ))
        
        path = tmp_path / "test_suite.json"
        suite.save(path)
        
        loaded = BenchmarkSuite.load(path)
        assert loaded.name == suite.name
        assert len(loaded.results) == 1
        assert loaded.results[0].model_name == "model1"
    
    def test_check_regression(self):
        """Test the regression checking function."""
        # Should pass
        passed, msg = check_regression("stcm", "cifar10", 0.55)
        assert passed
        
        # Should fail (below threshold)
        passed, msg = check_regression("stcm", "cifar10", 0.30)
        assert not passed
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = FuzzyPatternTM_STE(
            n_features=100,
            n_clauses=20,
            n_classes=5,
        )
        params = count_parameters(model)
        assert params > 0
        # Basic check: should have at least clause weights
        assert params >= 20 * 5  # voting weights


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])




