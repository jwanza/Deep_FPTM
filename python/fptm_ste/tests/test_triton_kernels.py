"""
Unit tests for Triton kernels for STCM speedup.
"""
import unittest
import torch
import torch.nn.functional as F


class TestTernaryPacking(unittest.TestCase):
    """Test pack/unpack correctness."""
    
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")
        torch.manual_seed(42)

    def test_pack_unpack_roundtrip(self):
        """Verify that packing and unpacking preserves ternary values."""
        from fptm_ste.kernels import pack_ternary_pytorch, unpack_ternary_pytorch
        
        # Generate random ternary weights
        w = torch.randint(-1, 2, (128, 256), device=self.device).float()
        
        # Pack
        w_packed, original_shape = pack_ternary_pytorch(w)
        
        # Check packed shape: ceil(256/4) = 64
        self.assertEqual(w_packed.shape, (128, 64))
        self.assertEqual(w_packed.dtype, torch.int8)
        
        # Unpack
        w_rec = unpack_ternary_pytorch(w_packed, original_shape)
        
        # Verify reconstruction
        self.assertTrue(torch.equal(w, w_rec), "Unpacked weights mismatch original")

    def test_pack_unpack_non_divisible_k(self):
        """Test with K not divisible by 4."""
        from fptm_ste.kernels import pack_ternary_pytorch, unpack_ternary_pytorch
        
        # K=33, not divisible by 4
        w = torch.randint(-1, 2, (32, 33), device=self.device).float()
        
        w_packed, original_shape = pack_ternary_pytorch(w)
        
        # ceil(33/4) = 9
        self.assertEqual(w_packed.shape, (32, 9))
        
        w_rec = unpack_ternary_pytorch(w_packed, original_shape)
        self.assertTrue(torch.equal(w, w_rec))

    def test_pack_all_values(self):
        """Test that all ternary values are encoded correctly."""
        from fptm_ste.kernels import pack_ternary_pytorch, unpack_ternary_pytorch
        
        # Create tensor with specific pattern
        w = torch.tensor([[-1, 0, 1, -1], [1, 1, 0, 0]], device=self.device, dtype=torch.float32)
        
        w_packed, original_shape = pack_ternary_pytorch(w)
        w_rec = unpack_ternary_pytorch(w_packed, original_shape)
        
        self.assertTrue(torch.equal(w, w_rec))


class TestTernaryMatmul(unittest.TestCase):
    """Test ternary matmul correctness."""
    
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")
        torch.manual_seed(42)

    def test_ternary_matmul_correctness(self):
        """Verify triton matmul against torch.matmul."""
        from fptm_ste.kernels import pack_ternary_pytorch, ternary_linear
        
        M, N, K = 32, 64, 128
        x = torch.randn(M, K, device=self.device)
        w = torch.randint(-1, 2, (N, K), device=self.device).float()
        
        # Torch baseline
        expected = F.linear(x, w)
        
        # Triton
        w_packed, original_shape = pack_ternary_pytorch(w)
        output = ternary_linear(x, w_packed, original_shape)
        
        # Check
        self.assertTrue(
            torch.allclose(output, expected, atol=1e-4),
            f"Triton matmul output mismatch. Max diff: {(output - expected).abs().max()}"
        )

    def test_ternary_matmul_large(self):
        """Test with larger matrices."""
        from fptm_ste.kernels import pack_ternary_pytorch, ternary_linear
        
        M, N, K = 256, 512, 1024
        x = torch.randn(M, K, device=self.device)
        w = torch.randint(-1, 2, (N, K), device=self.device).float()
        
        expected = F.linear(x, w)
        
        w_packed, original_shape = pack_ternary_pytorch(w)
        output = ternary_linear(x, w_packed, original_shape)
        
        self.assertTrue(
            torch.allclose(output, expected, atol=1e-3),
            f"Large matmul mismatch. Max diff: {(output - expected).abs().max()}"
        )

    def test_ternary_matmul_non_divisible_k(self):
        """Test with K not divisible by 4."""
        from fptm_ste.kernels import pack_ternary_pytorch, ternary_linear
        
        M, N, K = 16, 32, 33  # 33 is not divisible by 4
        x = torch.randn(M, K, device=self.device)
        w = torch.randint(-1, 2, (N, K), device=self.device).float()
        
        expected = F.linear(x, w)
        
        w_packed, original_shape = pack_ternary_pytorch(w)
        output = ternary_linear(x, w_packed, original_shape)
        
        self.assertTrue(
            torch.allclose(output, expected, atol=1e-4),
            f"Non-divisible K mismatch. Max diff: {(output - expected).abs().max()}"
        )

    def test_backward_pass(self):
        """Test backward pass works correctly."""
        from fptm_ste.kernels import pack_ternary_pytorch, ternary_linear
        
        M, N, K = 32, 64, 128
        x = torch.randn(M, K, device=self.device, requires_grad=True)
        w = torch.randint(-1, 2, (N, K), device=self.device).float()
        
        w_packed, original_shape = pack_ternary_pytorch(w)
        
        # Forward
        output = ternary_linear(x, w_packed, original_shape)
        loss = output.sum()
        
        # Backward
        loss.backward()
        
        # Check gradient exists and has correct shape
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)


class TestBenchmark(unittest.TestCase):
    """Benchmark tests for speedup measurement."""
    
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")
        torch.manual_seed(42)

    def test_speedup_measurement(self):
        """Measure speedup of ternary linear vs standard linear."""
        import time
        from fptm_ste.kernels import pack_ternary_pytorch, ternary_linear
        
        # Typical STCM dimensions
        M, N, K = 256, 512, 784  # MNIST-like
        
        x = torch.randn(M, K, device=self.device)
        w = torch.randint(-1, 2, (N, K), device=self.device).float()
        w_packed, original_shape = pack_ternary_pytorch(w)
        
        # Warmup
        for _ in range(10):
            _ = F.linear(x, w)
            _ = ternary_linear(x, w_packed, original_shape)
        torch.cuda.synchronize()
        
        # Benchmark baseline
        n_iters = 100
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = F.linear(x, w)
        torch.cuda.synchronize()
        baseline_time = (time.perf_counter() - t0) / n_iters * 1000
        
        # Benchmark Triton
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = ternary_linear(x, w_packed, original_shape)
        torch.cuda.synchronize()
        triton_time = (time.perf_counter() - t0) / n_iters * 1000
        
        speedup = baseline_time / triton_time
        
        print(f"\n{'='*60}")
        print(f"TERNARY LINEAR BENCHMARK (M={M}, N={N}, K={K})")
        print(f"{'='*60}")
        print(f"Baseline (F.linear float32): {baseline_time:.3f} ms")
        print(f"Triton (packed ternary):     {triton_time:.3f} ms")
        print(f"Speedup:                     {speedup:.2f}x")
        print(f"Memory reduction:            16x (float32 -> 2-bit packed)")
        print(f"{'='*60}\n")
        
        # We expect at least some speedup, but may vary by GPU
        # The main benefit is memory reduction
        self.assertGreater(speedup, 0.5, "Triton should not be more than 2x slower")


if __name__ == "__main__":
    unittest.main(verbosity=2)
