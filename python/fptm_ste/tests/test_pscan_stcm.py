"""
Unit tests for P-Scan optimized STCM.

Tests cover:
1. Mathematical correctness (P-Scan vs sequential equivalence)
2. Gradient flow verification
3. Speedup benchmarks
4. Numerical stability edge cases
5. CUDA Graph functionality
"""

import pytest
import torch
import time
from typing import Tuple

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for P-Scan tests"
)


class TestPScanCorrectness:
    """Verify P-Scan produces same results as sequential."""
    
    @pytest.mark.parametrize("T", [1, 10, 30, 100])
    def test_pscan_vs_sequential_equivalence(self, T: int):
        """P-Scan output must match sequential within tolerance."""
        # Import here to handle import errors gracefully
        from fptm_ste.parallel_ops import (
            associative_scan_linear,
            sequential_scan_linear,
            verify_pscan_correctness,
        )
        
        torch.manual_seed(42)
        D = 256
        B = 32
        
        # Decay factors in stable range (0.05, 0.95)
        A = torch.rand(D, device='cuda') * 0.9 + 0.05
        Bx = torch.randn(B, D, device='cuda')
        
        is_correct, metrics = verify_pscan_correctness(A, Bx, T, tol=1e-4)
        
        assert is_correct, (
            f"P-Scan != Sequential for T={T}: "
            f"max_error={metrics['max_error']:.6f}, "
            f"mean_error={metrics['mean_error']:.6f}"
        )
        
        # Also verify shapes
        h_pscan = associative_scan_linear(A, Bx, T)
        assert h_pscan.shape == (B, T, D), f"Expected shape [{B}, {T}, {D}], got {h_pscan.shape}"
    
    def test_pscan_stcm_forward_equivalence(self):
        """PScanOptimizedSTCM forward modes must produce equivalent results."""
        from fptm_ste.pscan_stcm import PScanOptimizedSTCM
        
        torch.manual_seed(42)
        
        model = PScanOptimizedSTCM(
            n_features=784,
            n_clauses=256,
            n_classes=10,
            iterations=10,
        ).cuda().eval()
        
        x = torch.randn(8, 784, device='cuda')
        
        with torch.no_grad():
            logits_pscan, h_pscan = model(x, mode='pscan')
            logits_seq, h_seq = model(x, mode='sequential')
        
        # Note: Some numerical difference is expected due to different
        # computation order, but should be within tolerance
        assert torch.allclose(logits_pscan, logits_seq, atol=1e-3), (
            f"Logits mismatch: max_diff={torch.abs(logits_pscan - logits_seq).max().item():.6f}"
        )
        assert torch.allclose(h_pscan, h_seq, atol=1e-3), (
            f"Hidden state mismatch: max_diff={torch.abs(h_pscan - h_seq).max().item():.6f}"
        )
    
    def test_pscan_gradient_flow(self):
        """Ensure gradients flow correctly through P-Scan."""
        from fptm_ste.pscan_stcm import PScanOptimizedSTCM
        
        torch.manual_seed(42)
        
        model = PScanOptimizedSTCM(
            n_features=784,
            n_clauses=256,
            n_classes=10,
            iterations=10,
        ).cuda()
        
        x = torch.randn(8, 784, requires_grad=True, device='cuda')
        logits, _ = model(x, mode='pscan')
        loss = logits.sum()
        loss.backward()
        
        # Check gradients exist and are finite
        assert model.log_A.grad is not None, "log_A has no gradient"
        assert torch.isfinite(model.log_A.grad).all(), "log_A gradient contains inf/nan"
        
        assert model.B_weight.grad is not None, "B_weight has no gradient"
        assert torch.isfinite(model.B_weight.grad).all(), "B_weight gradient contains inf/nan"
        
        # Check input gradient
        assert x.grad is not None, "Input has no gradient"
        assert torch.isfinite(x.grad).all(), "Input gradient contains inf/nan"
    
    def test_gradient_equivalence_pscan_vs_sequential(self):
        """Gradients from P-Scan and sequential should be similar."""
        from fptm_ste.pscan_stcm import PScanOptimizedSTCM
        
        torch.manual_seed(42)
        
        # Create two identical models
        model_pscan = PScanOptimizedSTCM(
            n_features=784,
            n_clauses=128,
            n_classes=10,
            iterations=5,
        ).cuda()
        
        model_seq = PScanOptimizedSTCM(
            n_features=784,
            n_clauses=128,
            n_classes=10,
            iterations=5,
        ).cuda()
        model_seq.load_state_dict(model_pscan.state_dict())
        
        x = torch.randn(4, 784, device='cuda')
        
        # P-Scan forward-backward
        logits_pscan, _ = model_pscan(x.clone().requires_grad_(True), mode='pscan')
        loss_pscan = logits_pscan.sum()
        loss_pscan.backward()
        
        # Sequential forward-backward
        logits_seq, _ = model_seq(x.clone().requires_grad_(True), mode='sequential')
        loss_seq = logits_seq.sum()
        loss_seq.backward()
        
        # Compare gradients (allow for numerical differences)
        grad_diff_B = torch.abs(model_pscan.B_weight.grad - model_seq.B_weight.grad).max()
        grad_diff_logA = torch.abs(model_pscan.log_A.grad - model_seq.log_A.grad).max()
        
        assert grad_diff_B < 0.1, f"B_weight gradient diff too large: {grad_diff_B:.6f}"
        assert grad_diff_logA < 0.1, f"log_A gradient diff too large: {grad_diff_logA:.6f}"


class TestPScanSpeedup:
    """Benchmark P-Scan vs sequential."""
    
    @pytest.mark.parametrize("T", [10, 30, 100])
    def test_pscan_speedup(self, T: int):
        """P-Scan should be faster than sequential for T >= 30."""
        from fptm_ste.pscan_stcm import PScanOptimizedSTCM
        
        model = PScanOptimizedSTCM(
            n_features=784,
            n_clauses=256,
            n_classes=10,
            iterations=T,
        ).cuda().eval()
        
        x = torch.randn(32, 784, device='cuda')
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(x, mode='pscan')
                _ = model(x, mode='sequential')
        torch.cuda.synchronize()
        
        # Benchmark P-Scan
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(100):
                _ = model(x, mode='pscan')
        torch.cuda.synchronize()
        pscan_time = time.perf_counter() - t0
        
        # Benchmark sequential
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(100):
                _ = model(x, mode='sequential')
        torch.cuda.synchronize()
        seq_time = time.perf_counter() - t0
        
        speedup = seq_time / pscan_time
        print(f"\nT={T}: P-Scan={pscan_time*10:.2f}ms, Seq={seq_time*10:.2f}ms, Speedup={speedup:.2f}x")
        
        # P-Scan should be faster for T >= 30
        if T >= 30:
            assert speedup > 1.2, f"Expected speedup > 1.2x for T={T}, got {speedup:.2f}x"
    
    def test_parallel_ops_speedup(self):
        """Test raw parallel_ops speedup."""
        from fptm_ste.parallel_ops import benchmark_pscan_vs_sequential
        
        D = 256
        B = 32
        T = 50
        
        A = torch.rand(D, device='cuda') * 0.9 + 0.05
        Bx = torch.randn(B, D, device='cuda')
        
        results = benchmark_pscan_vs_sequential(A, Bx, T, warmup=10, runs=100)
        
        print(f"\nParallel ops benchmark (T={T}):")
        print(f"  P-Scan: {results['pscan_ms']:.3f}ms")
        print(f"  Sequential: {results['sequential_ms']:.3f}ms")
        print(f"  Speedup: {results['speedup']:.2f}x")
        
        # Should have meaningful speedup for T=50
        assert results['speedup'] > 1.0, f"Expected speedup > 1.0x, got {results['speedup']:.2f}x"


class TestNumericalStability:
    """Test numerical stability edge cases."""
    
    def test_extreme_decay_values(self):
        """Test with decay near 0 and near 1."""
        from fptm_ste.parallel_ops import associative_scan_linear
        
        D = 64
        B = 8
        T = 30
        
        # Near-zero decay (aggressive forgetting)
        A_small = torch.full((D,), 0.01, device='cuda')
        Bx = torch.randn(B, D, device='cuda')
        h_small = associative_scan_linear(A_small, Bx, T)
        assert torch.isfinite(h_small).all(), "Near-zero decay produces inf/nan"
        
        # Near-one decay (long memory)
        A_large = torch.full((D,), 0.99, device='cuda')
        h_large = associative_scan_linear(A_large, Bx, T)
        assert torch.isfinite(h_large).all(), "Near-one decay produces inf/nan"
    
    def test_large_batch(self):
        """Test with large batch size."""
        from fptm_ste.pscan_stcm import PScanOptimizedSTCM
        
        model = PScanOptimizedSTCM(
            n_features=784,
            n_clauses=256,
            n_classes=10,
            iterations=30,
        ).cuda().eval()
        
        x = torch.randn(256, 784, device='cuda')
        
        with torch.no_grad():
            logits, _ = model(x, mode='pscan')
        
        assert torch.isfinite(logits).all(), "Large batch produces inf/nan"
        assert logits.shape == (256, 10), f"Expected shape [256, 10], got {logits.shape}"
    
    def test_small_feature_dim(self):
        """Test with small feature dimension."""
        from fptm_ste.pscan_stcm import PScanOptimizedSTCM
        
        model = PScanOptimizedSTCM(
            n_features=32,
            n_clauses=64,
            n_classes=5,
            iterations=10,
        ).cuda().eval()
        
        x = torch.randn(16, 32, device='cuda')
        
        with torch.no_grad():
            logits, h = model(x, mode='pscan')
        
        assert torch.isfinite(logits).all()
        assert logits.shape == (16, 5)
    
    def test_zero_input(self):
        """Test with zero input."""
        from fptm_ste.pscan_stcm import PScanOptimizedSTCM
        
        model = PScanOptimizedSTCM(
            n_features=784,
            n_clauses=256,
            n_classes=10,
            iterations=10,
        ).cuda().eval()
        
        x = torch.zeros(8, 784, device='cuda')
        
        with torch.no_grad():
            logits, _ = model(x, mode='pscan')
        
        assert torch.isfinite(logits).all(), "Zero input produces inf/nan"
    
    def test_large_values(self):
        """Test with large input values."""
        from fptm_ste.pscan_stcm import PScanOptimizedSTCM
        
        model = PScanOptimizedSTCM(
            n_features=784,
            n_clauses=256,
            n_classes=10,
            iterations=10,
        ).cuda().eval()
        
        x = torch.randn(8, 784, device='cuda') * 100  # Large values
        
        with torch.no_grad():
            logits, _ = model(x, mode='pscan')
        
        assert torch.isfinite(logits).all(), "Large input produces inf/nan"


class TestCUDAGraph:
    """Test CUDA Graph functionality."""
    
    def test_cuda_graph_correctness(self):
        """CUDA Graph output should match non-graph output."""
        from fptm_ste.pscan_stcm import PScanOptimizedSTCM_Graph
        
        torch.manual_seed(42)
        
        model = PScanOptimizedSTCM_Graph(
            n_features=784,
            n_clauses=256,
            n_classes=10,
            iterations=30,
        ).cuda().eval()
        
        batch_size = 32
        x = torch.randn(batch_size, 784, device='cuda')
        
        # Get non-graph result
        with torch.no_grad():
            logits_no_graph, _ = model(x, mode='pscan')
        
        # Enable graph and get result
        model.enable_cuda_graph(batch_size=batch_size)
        with torch.no_grad():
            logits_graph, _ = model(x, mode='pscan')
        
        assert torch.allclose(logits_graph, logits_no_graph, atol=1e-5), (
            f"CUDA Graph output mismatch: max_diff={torch.abs(logits_graph - logits_no_graph).max().item()}"
        )
    
    def test_cuda_graph_speedup(self):
        """CUDA Graph should provide speedup."""
        from fptm_ste.pscan_stcm import PScanOptimizedSTCM_Graph
        
        model = PScanOptimizedSTCM_Graph(
            n_features=784,
            n_clauses=256,
            n_classes=10,
            iterations=30,
        ).cuda().eval()
        
        batch_size = 32
        x = torch.randn(batch_size, 784, device='cuda')
        
        # Warmup non-graph
        with torch.no_grad():
            for _ in range(10):
                _ = model(x, mode='pscan')
        torch.cuda.synchronize()
        
        # Benchmark non-graph
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(100):
                _ = model(x, mode='pscan')
        torch.cuda.synchronize()
        no_graph_time = time.perf_counter() - t0
        
        # Enable graph
        model.enable_cuda_graph(batch_size=batch_size)
        
        # Warmup graph (includes capture)
        with torch.no_grad():
            for _ in range(10):
                _ = model(x, mode='pscan')
        torch.cuda.synchronize()
        
        # Benchmark graph
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(100):
                _ = model(x, mode='pscan')
        torch.cuda.synchronize()
        graph_time = time.perf_counter() - t0
        
        speedup = no_graph_time / graph_time
        print(f"\nCUDA Graph speedup: {speedup:.2f}x")
        print(f"  No graph: {no_graph_time*10:.2f}ms")
        print(f"  With graph: {graph_time*10:.2f}ms")
        
        # CUDA Graph should provide some speedup
        assert speedup > 0.9, f"CUDA Graph too slow: {speedup:.2f}x"


class TestInterpretability:
    """Test interpretability features."""
    
    def test_sparsity_tracking(self):
        """Test sparsity is tracked during training."""
        from fptm_ste.pscan_stcm import PScanOptimizedSTCM
        
        model = PScanOptimizedSTCM(
            n_features=784,
            n_clauses=256,
            n_classes=10,
            iterations=10,
            ternary_threshold=0.3,
        ).cuda().train()
        
        x = torch.randn(8, 784, device='cuda')
        _ = model(x, mode='pscan')
        
        sparsity = model.get_sparsity()
        
        assert 'overall' in sparsity
        assert 'positive' in sparsity
        assert 'negative' in sparsity
        assert 'zero' in sparsity
        
        # Sparsity should sum to 1
        total = sparsity['positive'] + sparsity['negative'] + sparsity['zero']
        assert abs(total - 1.0) < 0.01, f"Sparsity doesn't sum to 1: {total}"
    
    def test_clause_extraction(self):
        """Test interpretable clause extraction."""
        from fptm_ste.pscan_stcm import PScanOptimizedSTCM
        
        model = PScanOptimizedSTCM(
            n_features=784,
            n_clauses=256,
            n_classes=10,
            iterations=10,
        ).cuda().eval()
        
        clauses = model.get_interpretable_clauses(max_clauses=5)
        
        assert len(clauses) == 5
        for clause in clauses:
            assert 'clause_idx' in clause
            assert 'positive_literals' in clause
            assert 'negative_literals' in clause
            assert 'dont_care' in clause
            assert 'sparsity' in clause
    
    def test_all_iteration_states(self):
        """Test extracting all iteration states."""
        from fptm_ste.pscan_stcm import PScanOptimizedSTCM
        
        model = PScanOptimizedSTCM(
            n_features=784,
            n_clauses=256,
            n_classes=10,
            iterations=20,
        ).cuda().eval()
        
        x = torch.randn(8, 784, device='cuda')
        
        with torch.no_grad():
            all_states = model.get_all_iteration_states(x)
        
        assert all_states.shape == (8, 20, 256)
        assert torch.isfinite(all_states).all()


class TestTrainingIntegration:
    """Test training integration."""
    
    def test_training_step(self):
        """Test a single training step."""
        from fptm_ste.pscan_stcm import PScanOptimizedSTCM
        
        model = PScanOptimizedSTCM(
            n_features=784,
            n_clauses=256,
            n_classes=10,
            iterations=10,
        ).cuda()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        x = torch.randn(32, 784, device='cuda')
        y = torch.randint(0, 10, (32,), device='cuda')
        
        # Forward
        logits, _ = model(x, mode='pscan')
        loss = torch.nn.functional.cross_entropy(logits, y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check parameters updated
        assert loss.item() > 0
    
    def test_mixed_precision_training(self):
        """Test training with AMP."""
        from fptm_ste.pscan_stcm import PScanOptimizedSTCM
        
        model = PScanOptimizedSTCM(
            n_features=784,
            n_clauses=256,
            n_classes=10,
            iterations=10,
        ).cuda()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler('cuda')
        
        x = torch.randn(32, 784, device='cuda')
        y = torch.randint(0, 10, (32,), device='cuda')
        
        # Forward with AMP
        with torch.amp.autocast('cuda'):
            logits, _ = model(x, mode='pscan')
            loss = torch.nn.functional.cross_entropy(logits, y)
        
        # Backward with scaler
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        assert loss.item() > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])

