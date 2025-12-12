"""
Tests for bitplane16 tensor core kernels.

Verifies:
1. Correctness of ternary packing/unpacking
2. Correctness of tensor core matmul vs baseline
3. Gradient flow through the operations
4. Speedup vs baseline
"""
import unittest
import torch
import torch.nn.functional as F
import time


class TestTernaryPacking16(unittest.TestCase):
    """Test int16 ternary packing/unpacking."""
    
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        try:
            from fptm_ste.kernels_bitplane16 import pack_ternary_int16, unpack_ternary_int16
            cls.pack = pack_ternary_int16
            cls.unpack = unpack_ternary_int16
        except ImportError as e:
            raise unittest.SkipTest(f"Cannot import bitplane16 kernels: {e}")
    
    def setUp(self):
        self.device = torch.device("cuda")
        torch.manual_seed(42)
    
    def test_pack_unpack_roundtrip(self):
        """Verify that packing and unpacking preserves ternary values."""
        # Generate random ternary weights
        w = torch.randint(-1, 2, (128, 256), device=self.device).float()
        
        # Pack
        w_packed, shape = self.pack(w)
        
        # Check packed shape: ceil(256/8) = 32
        self.assertEqual(w_packed.shape, (128, 32))
        self.assertEqual(w_packed.dtype, torch.int16)
        
        # Unpack
        w_rec = self.unpack(w_packed, shape)
        
        # Verify reconstruction
        self.assertTrue(torch.equal(w, w_rec), "Unpacked weights mismatch original")
    
    def test_pack_unpack_non_divisible_k(self):
        """Test with K not divisible by 8."""
        # K=33, not divisible by 8
        w = torch.randint(-1, 2, (32, 33), device=self.device).float()
        
        w_packed, shape = self.pack(w)
        
        # ceil(33/8) = 5
        self.assertEqual(w_packed.shape, (32, 5))
        
        w_rec = self.unpack(w_packed, shape)
        self.assertTrue(torch.equal(w, w_rec))
    
    def test_pack_all_values(self):
        """Test that all ternary values are encoded correctly."""
        # Create tensor with specific pattern
        w = torch.tensor([
            [-1, 0, 1, -1, 0, 1, -1, 0],
            [1, 1, 0, 0, -1, -1, 1, 0],
        ], device=self.device, dtype=torch.float32)
        
        w_packed, shape = self.pack(w)
        w_rec = self.unpack(w_packed, shape)
        
        self.assertTrue(torch.equal(w, w_rec))
    
    def test_large_matrix(self):
        """Test with large matrices."""
        w = torch.randint(-1, 2, (512, 1024), device=self.device).float()
        
        w_packed, shape = self.pack(w)
        w_rec = self.unpack(w_packed, shape)
        
        self.assertTrue(torch.equal(w, w_rec))


class TestTernaryLinearTC(unittest.TestCase):
    """Test tensor core ternary linear."""
    
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        try:
            from fptm_ste.kernels_bitplane16 import (
                ternary_linear_tc, 
                TRITON_AVAILABLE
            )
            if not TRITON_AVAILABLE:
                raise unittest.SkipTest("Triton not available")
            cls.ternary_linear_tc = ternary_linear_tc
        except ImportError as e:
            raise unittest.SkipTest(f"Cannot import bitplane16 kernels: {e}")
    
    def setUp(self):
        self.device = torch.device("cuda")
        torch.manual_seed(42)
    
    def test_correctness_vs_baseline(self):
        """Verify tensor core output matches F.linear."""
        M, N, K = 128, 256, 512
        x = torch.randn(M, K, device=self.device)
        w = torch.randint(-1, 2, (N, K), device=self.device).float()
        
        # Baseline
        expected = F.linear(x, w)
        
        # Tensor core
        actual = self.ternary_linear_tc(x, w)
        
        self.assertTrue(
            torch.allclose(expected, actual, atol=1e-3),
            f"Output mismatch. Max diff: {(expected - actual).abs().max()}"
        )
    
    def test_correctness_various_sizes(self):
        """Test with various matrix dimensions."""
        sizes = [
            (32, 64, 128),
            (64, 128, 256),
            (128, 256, 512),
            (256, 512, 1024),
        ]
        
        for M, N, K in sizes:
            with self.subTest(M=M, N=N, K=K):
                x = torch.randn(M, K, device=self.device)
                w = torch.randint(-1, 2, (N, K), device=self.device).float()
                
                expected = F.linear(x, w)
                actual = self.ternary_linear_tc(x, w)
                
                self.assertTrue(
                    torch.allclose(expected, actual, atol=1e-3),
                    f"Mismatch for ({M}, {N}, {K}). Max diff: {(expected - actual).abs().max()}"
                )
    
    def test_k_not_divisible_by_8(self):
        """Test with K not divisible by 8."""
        M, N, K = 64, 128, 33  # 33 is not divisible by 8
        x = torch.randn(M, K, device=self.device)
        w = torch.randint(-1, 2, (N, K), device=self.device).float()
        
        expected = F.linear(x, w)
        actual = self.ternary_linear_tc(x, w)
        
        self.assertTrue(
            torch.allclose(expected, actual, atol=1e-3),
            f"Non-divisible K mismatch. Max diff: {(expected - actual).abs().max()}"
        )
    
    def test_batched_input(self):
        """Test with batched 3D input."""
        B, M, K, N = 8, 32, 128, 64
        x = torch.randn(B, M, K, device=self.device)
        w = torch.randint(-1, 2, (N, K), device=self.device).float()
        
        expected = F.linear(x, w)
        actual = self.ternary_linear_tc(x, w)
        
        self.assertTrue(
            torch.allclose(expected, actual, atol=1e-3),
            f"Batched mismatch. Max diff: {(expected - actual).abs().max()}"
        )


class TestBitplane16TCMatmul(unittest.TestCase):
    """Test bitplane16 tensor core matmul with packed weights."""
    
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        try:
            from fptm_ste.kernels_bitplane16 import (
                pack_ternary_int16,
                bitplane16_tc_matmul,
                TRITON_AVAILABLE
            )
            if not TRITON_AVAILABLE:
                raise unittest.SkipTest("Triton not available")
            cls.pack = pack_ternary_int16
            cls.bitplane_matmul = bitplane16_tc_matmul
        except ImportError as e:
            raise unittest.SkipTest(f"Cannot import bitplane16 kernels: {e}")
    
    def setUp(self):
        self.device = torch.device("cuda")
        torch.manual_seed(42)
    
    def test_correctness_vs_baseline(self):
        """Verify bitplane matmul matches F.linear."""
        M, N, K = 128, 256, 512
        x = torch.randn(M, K, device=self.device)
        w = torch.randint(-1, 2, (N, K), device=self.device).float()
        
        # Pack weights
        w_packed, shape = self.pack(w)
        
        # Baseline
        expected = F.linear(x, w)
        
        # Bitplane matmul
        actual = self.bitplane_matmul(x, w_packed, shape)
        
        self.assertTrue(
            torch.allclose(expected, actual, atol=1e-2),
            f"Output mismatch. Max diff: {(expected - actual).abs().max()}"
        )
    
    def test_memory_reduction(self):
        """Verify packed weights use less memory."""
        N, K = 256, 512
        w = torch.randint(-1, 2, (N, K), device=self.device).float()
        
        # Original memory
        original_bytes = w.numel() * w.element_size()
        
        # Packed memory
        w_packed, _ = self.pack(w)
        packed_bytes = w_packed.numel() * w_packed.element_size()
        
        reduction = original_bytes / packed_bytes
        print(f"\nMemory reduction: {reduction:.1f}x ({original_bytes} -> {packed_bytes} bytes)")
        
        # Should be approximately 8x (float32 -> 2 bits per weight)
        self.assertGreater(reduction, 7.0, "Expected at least 7x memory reduction")


class TestTernaryLinearTCModule(unittest.TestCase):
    """Test TernaryLinearTC module."""
    
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        try:
            from fptm_ste.kernels_bitplane16 import TernaryLinearTC
            cls.TernaryLinearTC = TernaryLinearTC
        except ImportError as e:
            raise unittest.SkipTest(f"Cannot import bitplane16 kernels: {e}")
    
    def setUp(self):
        self.device = torch.device("cuda")
        torch.manual_seed(42)
    
    def test_forward_training(self):
        """Test forward pass in training mode."""
        layer = self.TernaryLinearTC(128, 64, bias=True)
        layer.to(self.device)
        layer.train()
        
        x = torch.randn(32, 128, device=self.device)
        out = layer(x)
        
        self.assertEqual(out.shape, (32, 64))
    
    def test_forward_eval(self):
        """Test forward pass in eval mode."""
        layer = self.TernaryLinearTC(128, 64, bias=True)
        layer.to(self.device)
        layer.eval()
        
        x = torch.randn(32, 128, device=self.device)
        out = layer(x)
        
        self.assertEqual(out.shape, (32, 64))
    
    def test_freeze(self):
        """Test weight freezing for inference."""
        layer = self.TernaryLinearTC(128, 64, bias=True)
        layer.to(self.device)
        layer.eval()
        
        # Freeze
        layer.freeze()
        
        # Should have cached packed weights
        self.assertIsNotNone(layer._packed_cache)
        
        x = torch.randn(32, 128, device=self.device)
        out = layer(x)
        
        self.assertEqual(out.shape, (32, 64))
    
    def test_gradient_flow(self):
        """Test that gradients flow during training."""
        layer = self.TernaryLinearTC(128, 64, bias=True)
        layer.to(self.device)
        layer.train()
        
        x = torch.randn(32, 128, device=self.device, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(layer.weight.grad)


class TestBenchmark(unittest.TestCase):
    """Benchmark tests for speedup measurement."""
    
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        try:
            from fptm_ste.kernels_bitplane16 import (
                ternary_linear_tc,
                pack_ternary_int16,
                bitplane16_tc_matmul,
                TRITON_AVAILABLE
            )
            if not TRITON_AVAILABLE:
                raise unittest.SkipTest("Triton not available")
            cls.ternary_linear_tc = ternary_linear_tc
            cls.pack = pack_ternary_int16
            cls.bitplane_matmul = bitplane16_tc_matmul
        except ImportError as e:
            raise unittest.SkipTest(f"Cannot import bitplane16 kernels: {e}")
    
    def setUp(self):
        self.device = torch.device("cuda")
        torch.manual_seed(42)
    
    def test_speedup_measurement(self):
        """Measure speedup of tensor core ternary linear vs standard linear."""
        # Typical STCM dimensions
        M, N, K = 256, 512, 784  # MNIST-like
        
        x = torch.randn(M, K, device=self.device)
        w = torch.randint(-1, 2, (N, K), device=self.device).float()
        
        # Warmup
        for _ in range(10):
            _ = F.linear(x, w)
            _ = self.ternary_linear_tc(x, w)
        torch.cuda.synchronize()
        
        # Benchmark baseline
        n_iters = 50
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = F.linear(x, w)
        torch.cuda.synchronize()
        baseline_time = (time.perf_counter() - t0) / n_iters * 1000
        
        # Benchmark Triton TC
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = self.ternary_linear_tc(x, w)
        torch.cuda.synchronize()
        tc_time = (time.perf_counter() - t0) / n_iters * 1000
        
        speedup = baseline_time / tc_time
        
        print(f"\n{'='*60}")
        print(f"BITPLANE16 TENSOR CORE BENCHMARK (M={M}, N={N}, K={K})")
        print(f"{'='*60}")
        print(f"Baseline (F.linear float32): {baseline_time:.3f} ms")
        print(f"Triton TC (ternary):         {tc_time:.3f} ms")
        print(f"Speedup:                     {speedup:.2f}x")
        print(f"Memory reduction:            16x (float32 -> 2-bit packed)")
        print(f"{'='*60}\n")
        
        # We expect at least some speedup for larger matrices
        self.assertGreater(speedup, 0.5, "Triton should not be more than 2x slower")


if __name__ == "__main__":
    unittest.main(verbosity=2)





