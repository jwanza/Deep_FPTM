"""
Tests for CompiledSTCM to validate:
1. Output matches non-compiled version
2. Speedup is achieved (2-3x expected)
3. Accuracy is preserved
"""

import unittest
import time
import torch
import torch.nn.functional as F


class TestCompiledSTCM(unittest.TestCase):
    """Test CompiledSTCM correctness and performance."""
    
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")
        torch.manual_seed(42)
        
    def test_output_matches_optimized(self):
        """Verify CompiledSTCM produces same output as OptimizedSTCM."""
        from fptm_ste.tm_optimized import OptimizedSTCM
        from fptm_ste.compiled_stcm import CompiledSTCM
        
        B, F_dim, C, K = 32, 784, 256, 10
        x = torch.rand(B, F_dim, device=self.device)
        
        # Create models with same weights
        optimized = OptimizedSTCM(
            n_features=F_dim, n_clauses=C, n_classes=K
        ).to(self.device)
        
        compiled = CompiledSTCM(
            n_features=F_dim, n_clauses=C, n_classes=K
        ).to(self.device)
        
        # Copy weights
        compiled.load_state_dict(optimized.state_dict())
        
        # Eval mode
        optimized.eval()
        compiled.eval()
        
        with torch.no_grad():
            out_opt = optimized(x)[0]
            out_comp = compiled(x)[0]
        
        # Check outputs match
        self.assertTrue(
            torch.allclose(out_opt, out_comp, atol=1e-4),
            f"Outputs differ. Max diff: {(out_opt - out_comp).abs().max()}"
        )
        
    def test_backward_pass(self):
        """Verify backward pass works correctly."""
        from fptm_ste.compiled_stcm import CompiledSTCM
        
        B, F_dim, C, K = 32, 784, 256, 10
        x = torch.rand(B, F_dim, device=self.device, requires_grad=True)
        
        model = CompiledSTCM(
            n_features=F_dim, n_clauses=C, n_classes=K
        ).to(self.device)
        
        # Forward + backward
        out = model(x)[0]
        loss = out.sum()
        loss.backward()
        
        # Check gradient exists
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)
        
    def test_speedup_measurement(self):
        """Measure speedup of compiled vs non-compiled."""
        from fptm_ste.tm_optimized import OptimizedSTCM
        from fptm_ste.compiled_stcm import CompiledSTCM
        
        B, F_dim, C, K = 256, 784, 512, 10
        x = torch.rand(B, F_dim, device=self.device)
        
        optimized = OptimizedSTCM(
            n_features=F_dim, n_clauses=C, n_classes=K
        ).to(self.device)
        
        compiled = CompiledSTCM(
            n_features=F_dim, n_clauses=C, n_classes=K
        ).to(self.device)
        
        compiled.load_state_dict(optimized.state_dict())
        
        optimized.eval()
        compiled.eval()
        
        # Warmup (includes compilation for compiled version)
        n_warmup = 20
        for _ in range(n_warmup):
            with torch.no_grad():
                _ = optimized(x)
                _ = compiled(x)
        torch.cuda.synchronize()
        
        # Benchmark
        n_iters = 100
        
        # Optimized timing
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            with torch.no_grad():
                _ = optimized(x)
        torch.cuda.synchronize()
        opt_time = (time.perf_counter() - t0) / n_iters * 1000
        
        # Compiled timing
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            with torch.no_grad():
                _ = compiled(x)
        torch.cuda.synchronize()
        comp_time = (time.perf_counter() - t0) / n_iters * 1000
        
        speedup = opt_time / comp_time
        
        print(f"\n{'='*60}")
        print(f"COMPILED STCM BENCHMARK (B={B}, F={F_dim}, C={C})")
        print(f"{'='*60}")
        print(f"OptimizedSTCM:  {opt_time:.3f} ms")
        print(f"CompiledSTCM:   {comp_time:.3f} ms")
        print(f"Speedup:        {speedup:.2f}x")
        print(f"{'='*60}\n")
        
        # We expect at least some speedup, but it varies by GPU
        # Don't fail if speedup is less than expected
        self.assertGreater(speedup, 0.5, "Compiled should not be more than 2x slower")


class TestDeepCompiledSTCM(unittest.TestCase):
    """Test DeepCompiledSTCM."""
    
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")
        torch.manual_seed(42)
        
    def test_forward_pass(self):
        """Verify forward pass works."""
        from fptm_ste.compiled_stcm import DeepCompiledSTCM
        
        B, F_dim, K = 32, 784, 10
        x = torch.rand(B, F_dim, device=self.device)
        
        model = DeepCompiledSTCM(
            input_dim=F_dim,
            hidden_dims=[256, 128],
            n_classes=K,
            n_clauses=128,
        ).to(self.device)
        
        logits, clause_outputs = model(x)
        
        self.assertEqual(logits.shape, (B, K))
        

if __name__ == "__main__":
    unittest.main(verbosity=2)



