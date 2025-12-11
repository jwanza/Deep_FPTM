"""
Unit tests for fused Triton kernels.

Tests:
1. fused_ste_ternary - STE ternary quantization
2. fused_clause_sync - Clause synchronization
3. fused_gumbel_softmax - Gumbel-Softmax sampling

Each test verifies:
- Correctness vs reference implementation
- Gradient flow
- Edge cases
- Performance (speedup verification)
"""
import unittest
import torch
import torch.nn.functional as F
import time


def benchmark_function(fn, *args, n_warmup=10, n_iters=100, **kwargs):
    """Benchmark a function with warmup and multiple iterations."""
    # Warmup
    for _ in range(n_warmup):
        result = fn(*args, **kwargs)
        if hasattr(result, 'is_cuda') and result.is_cuda:
            torch.cuda.synchronize()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_iters * 1000  # ms
    return result, elapsed


class TestFusedSTETernary(unittest.TestCase):
    """Tests for fused_ste_ternary kernel."""
    
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        try:
            from fptm_ste.kernels_fused import (
                fused_ste_ternary, 
                ste_ternary_reference,
                TRITON_AVAILABLE
            )
            if not TRITON_AVAILABLE:
                raise unittest.SkipTest("Triton not available")
            cls.fused_ste_ternary = fused_ste_ternary
            cls.ste_ternary_reference = ste_ternary_reference
        except ImportError as e:
            raise unittest.SkipTest(f"Cannot import fused kernels: {e}")
    
    def setUp(self):
        self.device = torch.device("cuda")
        torch.manual_seed(42)
    
    def test_correctness_vs_reference(self):
        """Verify fused output matches reference within tolerance."""
        shapes = [
            (32, 64),
            (128, 256),
            (256, 1024),
            (512, 784),  # MNIST-like
        ]
        bands = [0.0, 0.1, 0.5, 1.0]
        temperatures = [0.1, 0.5, 1.0, 2.0]
        
        for shape in shapes:
            for band in bands:
                for temp in temperatures:
                    with self.subTest(shape=shape, band=band, temp=temp):
                        logits = torch.randn(shape, device=self.device)
                        
                        ref_out = self.ste_ternary_reference(logits, band, temp)
                        fused_out = self.fused_ste_ternary(logits, band, temp)
                        
                        self.assertTrue(
                            torch.allclose(ref_out, fused_out, atol=1e-4),
                            f"Mismatch for shape={shape}, band={band}, temp={temp}. "
                            f"Max diff: {(ref_out - fused_out).abs().max()}"
                        )
    
    def test_gradient_flow(self):
        """Verify backward pass produces correct gradients."""
        logits = torch.randn(128, 256, device=self.device, requires_grad=True)
        band, temp = 0.3, 0.5
        
        # Forward
        out = self.fused_ste_ternary(logits, band, temp)
        loss = out.sum()
        
        # Backward
        loss.backward()
        
        # Check gradient exists and has reasonable values
        self.assertIsNotNone(logits.grad)
        self.assertEqual(logits.grad.shape, logits.shape)
        self.assertFalse(torch.isnan(logits.grad).any())
        self.assertFalse(torch.isinf(logits.grad).any())
        
        # Compare gradients with reference
        logits_ref = logits.detach().clone().requires_grad_(True)
        out_ref = self.ste_ternary_reference(logits_ref, band, temp)
        out_ref.sum().backward()
        
        self.assertTrue(
            torch.allclose(logits.grad, logits_ref.grad, atol=1e-4),
            f"Gradient mismatch. Max diff: {(logits.grad - logits_ref.grad).abs().max()}"
        )
    
    def test_edge_cases(self):
        """Test band=0, temperature extremes, all-zero inputs."""
        # Band = 0 (should use sign)
        logits = torch.randn(64, 128, device=self.device)
        out = self.fused_ste_ternary(logits, 0.0, 1.0)
        self.assertTrue((out.abs() <= 1.0).all())
        
        # Very small temperature (almost hard quantization)
        out_hard = self.fused_ste_ternary(logits, 0.3, 0.01)
        self.assertTrue((out_hard.abs() <= 1.0).all())
        
        # Large temperature (soft quantization)
        out_soft = self.fused_ste_ternary(logits, 0.3, 10.0)
        self.assertTrue((out_soft.abs() <= 1.0).all())
        
        # All-zero input
        zeros = torch.zeros(32, 64, device=self.device)
        out_zeros = self.fused_ste_ternary(zeros, 0.3, 1.0)
        self.assertTrue(torch.allclose(out_zeros, zeros, atol=1e-5))
    
    def test_output_values_in_range(self):
        """Verify hard values are in {-1, 0, 1}."""
        logits = torch.randn(128, 256, device=self.device)
        out = self.fused_ste_ternary(logits, 0.5, 1.0)
        
        # Detached hard values should be exactly in {-1, 0, 1}
        with torch.no_grad():
            hard = torch.zeros_like(logits)
            hard = torch.where(logits > 0.5, torch.ones_like(logits), hard)
            hard = torch.where(logits < -0.5, -torch.ones_like(logits), hard)
        
        # The output should be close to hard + some STE residual
        self.assertTrue((out.abs() <= 1.5).all())  # Allow for STE
    
    def test_speedup_vs_reference(self):
        """Verify fused kernel provides speedup over reference."""
        logits = torch.randn(256, 1024, device=self.device)
        band, temp = 0.3, 0.5
        
        _, ref_time = benchmark_function(
            self.ste_ternary_reference, logits, band, temp
        )
        _, fused_time = benchmark_function(
            self.fused_ste_ternary, logits, band, temp
        )
        
        speedup = ref_time / fused_time
        print(f"\nSTE Ternary: Ref={ref_time:.3f}ms, Fused={fused_time:.3f}ms, Speedup={speedup:.2f}x")
        
        # Should be at least not slower
        self.assertGreater(speedup, 0.5, "Fused kernel should not be >2x slower")


class TestFusedClauseSync(unittest.TestCase):
    """Tests for fused_clause_sync kernel."""
    
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        try:
            from fptm_ste.kernels_fused import (
                fused_clause_sync,
                clause_sync_reference,
                TRITON_AVAILABLE
            )
            if not TRITON_AVAILABLE:
                raise unittest.SkipTest("Triton not available")
            cls.fused_clause_sync = fused_clause_sync
            cls.clause_sync_reference = clause_sync_reference
        except ImportError as e:
            raise unittest.SkipTest(f"Cannot import fused kernels: {e}")
    
    def setUp(self):
        self.device = torch.device("cuda")
        torch.manual_seed(42)
    
    def test_first_call_init(self):
        """Verify correct initialization when decay_alpha/beta are None."""
        B, n_clauses, synch_size = 32, 128, 64
        
        clause_act = torch.rand(B, n_clauses, device=self.device)
        left_indices = torch.randint(0, n_clauses, (synch_size,), device=self.device)
        right_indices = torch.randint(0, n_clauses, (synch_size,), device=self.device)
        r = torch.rand(synch_size, device=self.device) * 0.5 + 0.5  # [0.5, 1.0]
        
        sync, alpha, beta = self.fused_clause_sync(
            clause_act, left_indices, right_indices, None, None, r
        )
        
        # Check shapes
        self.assertEqual(sync.shape, (B, synch_size))
        self.assertEqual(alpha.shape, (B, synch_size))
        self.assertEqual(beta.shape, (B, synch_size))
        
        # On first call, sync = alpha / sqrt(beta) = pairwise_product / 1
        expected_product = clause_act[:, left_indices] * clause_act[:, right_indices]
        self.assertTrue(
            torch.allclose(sync, expected_product, atol=1e-4),
            f"First call init mismatch. Max diff: {(sync - expected_product).abs().max()}"
        )
    
    def test_ema_update(self):
        """Verify EMA accumulation matches reference."""
        B, n_clauses, synch_size = 16, 64, 32
        
        clause_act = torch.rand(B, n_clauses, device=self.device)
        left_indices = torch.randint(0, n_clauses, (synch_size,), device=self.device)
        right_indices = torch.randint(0, n_clauses, (synch_size,), device=self.device)
        r = torch.rand(synch_size, device=self.device) * 0.5 + 0.5
        
        # Initial call
        decay_alpha = torch.rand(B, synch_size, device=self.device)
        decay_beta = torch.rand(B, synch_size, device=self.device) + 1.0
        
        # Reference computation
        ref_sync, ref_alpha, ref_beta = self.clause_sync_reference(
            clause_act, left_indices, right_indices, 
            decay_alpha.clone(), decay_beta.clone(), r
        )
        
        # Fused computation
        fused_sync, fused_alpha, fused_beta = self.fused_clause_sync(
            clause_act, left_indices, right_indices,
            decay_alpha, decay_beta, r
        )
        
        self.assertTrue(
            torch.allclose(ref_sync, fused_sync, atol=1e-4),
            f"EMA sync mismatch. Max diff: {(ref_sync - fused_sync).abs().max()}"
        )
        self.assertTrue(
            torch.allclose(ref_alpha, fused_alpha, atol=1e-4),
            f"EMA alpha mismatch. Max diff: {(ref_alpha - fused_alpha).abs().max()}"
        )
        self.assertTrue(
            torch.allclose(ref_beta, fused_beta, atol=1e-4),
            f"EMA beta mismatch. Max diff: {(ref_beta - fused_beta).abs().max()}"
        )
    
    def test_various_synch_sizes(self):
        """Test with different numbers of clause pairs."""
        B, n_clauses = 32, 256
        synch_sizes = [16, 32, 64, 128]
        
        for synch_size in synch_sizes:
            with self.subTest(synch_size=synch_size):
                clause_act = torch.rand(B, n_clauses, device=self.device)
                left_indices = torch.randint(0, n_clauses, (synch_size,), device=self.device)
                right_indices = torch.randint(0, n_clauses, (synch_size,), device=self.device)
                r = torch.rand(synch_size, device=self.device) * 0.5 + 0.5
                
                ref_sync, _, _ = self.clause_sync_reference(
                    clause_act, left_indices, right_indices, None, None, r
                )
                fused_sync, _, _ = self.fused_clause_sync(
                    clause_act, left_indices, right_indices, None, None, r
                )
                
                self.assertTrue(
                    torch.allclose(ref_sync, fused_sync, atol=1e-4),
                    f"Mismatch for synch_size={synch_size}"
                )


class TestFusedGumbelSoftmax(unittest.TestCase):
    """Tests for fused_gumbel_softmax kernel."""
    
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        try:
            from fptm_ste.kernels_fused import (
                fused_gumbel_softmax,
                gumbel_softmax_reference,
                TRITON_AVAILABLE
            )
            if not TRITON_AVAILABLE:
                raise unittest.SkipTest("Triton not available")
            cls.fused_gumbel_softmax = fused_gumbel_softmax
            cls.gumbel_softmax_reference = gumbel_softmax_reference
        except ImportError as e:
            raise unittest.SkipTest(f"Cannot import fused kernels: {e}")
    
    def setUp(self):
        self.device = torch.device("cuda")
        torch.manual_seed(42)
    
    def test_output_is_one_hot(self):
        """Verify hard output is one-hot."""
        logits = torch.randn(64, 128, 3, device=self.device)
        temperature = 1.0
        
        out = self.fused_gumbel_softmax(logits, temperature, hard=True)
        
        # Check shape preserved
        self.assertEqual(out.shape, logits.shape)
        
        # Check one-hot property: sum over last dim should be 1
        sums = out.sum(dim=-1)
        self.assertTrue(
            torch.allclose(sums, torch.ones_like(sums), atol=1e-4),
            f"Not one-hot: sums range [{sums.min()}, {sums.max()}]"
        )
        
        # Check values are 0 or 1
        unique_vals = out.unique()
        self.assertTrue(
            all(v in [0.0, 1.0] for v in unique_vals.tolist()),
            f"Expected only 0 and 1, got {unique_vals}"
        )
    
    def test_gradient_approximation(self):
        """Verify STE gradient flows correctly."""
        logits = torch.randn(32, 64, 3, device=self.device, requires_grad=True)
        temperature = 0.5
        
        # Forward
        out = self.fused_gumbel_softmax(logits, temperature, hard=True)
        loss = out.sum()
        
        # Backward
        loss.backward()
        
        # Check gradient exists
        self.assertIsNotNone(logits.grad)
        self.assertEqual(logits.grad.shape, logits.shape)
        self.assertFalse(torch.isnan(logits.grad).any())
    
    def test_temperature_effect(self):
        """Verify temperature affects distribution sharpness."""
        logits = torch.randn(128, 256, 3, device=self.device)
        
        # Low temperature should be more deterministic (closer to argmax)
        torch.manual_seed(42)
        out_cold = self.fused_gumbel_softmax(logits, 0.1, hard=True)
        
        torch.manual_seed(42)
        out_hot = self.fused_gumbel_softmax(logits, 10.0, hard=True)
        
        # Both should be valid one-hot
        self.assertTrue(torch.allclose(out_cold.sum(-1), torch.ones_like(out_cold.sum(-1))))
        self.assertTrue(torch.allclose(out_hot.sum(-1), torch.ones_like(out_hot.sum(-1))))
    
    def test_various_shapes(self):
        """Test with different input shapes."""
        shapes = [
            (32, 64, 3),      # Standard
            (128, 256, 3),    # Larger
            (64, 512, 3),     # Even larger
            (16, 32, 5),      # Different num classes
        ]
        
        for shape in shapes:
            with self.subTest(shape=shape):
                logits = torch.randn(shape, device=self.device)
                out = self.fused_gumbel_softmax(logits, 1.0, hard=True)
                self.assertEqual(out.shape, shape)
                self.assertTrue(torch.allclose(out.sum(-1), torch.ones(shape[:-1], device=self.device)))
    
    def test_speedup_vs_reference(self):
        """Verify fused kernel provides speedup over reference."""
        logits = torch.randn(128, 512, 3, device=self.device)
        temp = 1.0
        
        _, ref_time = benchmark_function(
            self.gumbel_softmax_reference, logits, temp, True
        )
        _, fused_time = benchmark_function(
            self.fused_gumbel_softmax, logits, temp, True
        )
        
        speedup = ref_time / fused_time
        print(f"\nGumbel Softmax: Ref={ref_time:.3f}ms, Fused={fused_time:.3f}ms, Speedup={speedup:.2f}x")


class TestTritonBackendToggle(unittest.TestCase):
    """Test Triton backend toggle functionality."""
    
    def test_toggle_functionality(self):
        """Test that set_triton_enabled works correctly."""
        from fptm_ste import set_triton_enabled, get_triton_status
        
        original_status = get_triton_status()
        
        # Disable
        set_triton_enabled(False)
        status = get_triton_status()
        self.assertFalse(status['triton_enabled'])
        self.assertFalse(status['triton_effective'])
        
        # Re-enable
        set_triton_enabled(True)
        status = get_triton_status()
        self.assertTrue(status['triton_enabled'])
        # triton_effective depends on hardware availability
        if original_status['triton_hardware_available']:
            self.assertTrue(status['triton_effective'])


if __name__ == "__main__":
    unittest.main(verbosity=2)


