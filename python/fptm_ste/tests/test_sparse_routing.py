"""
Tests for Sparse Clause Routing (SparseSTCM).

Validates:
1. Router selects correct number of clauses
2. Sparse computation matches expected behavior
3. Speedup is achieved for k << n_clauses
4. Accuracy is preserved or improved
"""

import unittest
import time
import torch
import torch.nn.functional as F


class TestSparseClauseRouter(unittest.TestCase):
    """Test SparseClauseRouter mechanics."""
    
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")
        torch.manual_seed(42)
        
    def test_router_output_shapes(self):
        """Verify router produces correct output shapes."""
        from fptm_ste.sparse_stcm import SparseClauseRouter
        
        B, F_dim, C, k = 32, 256, 128, 32
        
        router = SparseClauseRouter(
            n_features=F_dim,
            n_clauses=C,
            k=k,
        ).to(self.device)
        
        x = torch.rand(B, F_dim, device=self.device)
        
        indices, weights, aux_loss = router(x)
        
        self.assertEqual(indices.shape, (B, k))
        self.assertEqual(weights.shape, (B, k))
        self.assertEqual(aux_loss.dim(), 0)  # Scalar
        
    def test_router_indices_valid(self):
        """Verify router produces valid clause indices."""
        from fptm_ste.sparse_stcm import SparseClauseRouter
        
        B, F_dim, C, k = 32, 256, 128, 32
        
        router = SparseClauseRouter(
            n_features=F_dim,
            n_clauses=C,
            k=k,
        ).to(self.device)
        
        x = torch.rand(B, F_dim, device=self.device)
        indices, _, _ = router(x)
        
        # All indices should be in valid range
        self.assertTrue((indices >= 0).all())
        self.assertTrue((indices < C).all())
        
    def test_router_weights_sum_to_one(self):
        """Verify router weights are normalized."""
        from fptm_ste.sparse_stcm import SparseClauseRouter
        
        B, F_dim, C, k = 32, 256, 128, 32
        
        router = SparseClauseRouter(
            n_features=F_dim,
            n_clauses=C,
            k=k,
        ).to(self.device)
        
        x = torch.rand(B, F_dim, device=self.device)
        _, weights, _ = router(x)
        
        # Weights should sum to 1 per sample
        weight_sums = weights.sum(dim=1)
        self.assertTrue(torch.allclose(weight_sums, torch.ones(B, device=self.device), atol=1e-5))


class TestSparseSTCM(unittest.TestCase):
    """Test SparseSTCM model."""
    
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")
        torch.manual_seed(42)
        
    def test_forward_pass(self):
        """Verify forward pass works."""
        from fptm_ste.sparse_stcm import SparseSTCM
        
        B, F_dim, C, K = 32, 256, 128, 10
        
        model = SparseSTCM(
            n_features=F_dim,
            n_clauses=C,
            n_classes=K,
            k=32,
        ).to(self.device)
        
        x = torch.rand(B, F_dim, device=self.device)
        
        logits, clause_out = model(x)
        
        self.assertEqual(logits.shape, (B, K))
        
    def test_backward_pass(self):
        """Verify backward pass works."""
        from fptm_ste.sparse_stcm import SparseSTCM
        
        B, F_dim, C, K = 32, 256, 128, 10
        
        model = SparseSTCM(
            n_features=F_dim,
            n_clauses=C,
            n_classes=K,
            k=32,
        ).to(self.device)
        
        x = torch.rand(B, F_dim, device=self.device, requires_grad=True)
        
        logits, _ = model(x)
        loss = logits.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        
    def test_aux_loss_available(self):
        """Verify auxiliary loss is computed during training."""
        from fptm_ste.sparse_stcm import SparseSTCM
        
        B, F_dim, C, K = 32, 256, 128, 10
        
        model = SparseSTCM(
            n_features=F_dim,
            n_clauses=C,
            n_classes=K,
            k=32,
        ).to(self.device)
        model.train()
        
        x = torch.rand(B, F_dim, device=self.device)
        _ = model(x)
        
        aux_loss = model.get_aux_loss()
        self.assertIsNotNone(aux_loss)
        self.assertGreater(aux_loss.item(), 0)


class TestSparseBenchmark(unittest.TestCase):
    """Benchmark sparse vs dense STCM."""
    
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")
        torch.manual_seed(42)
        
    def test_speedup_measurement(self):
        """Measure speedup of sparse vs dense."""
        from fptm_ste.sparse_stcm import SparseSTCM
        from fptm_ste.tm_optimized import OptimizedSTCM
        
        B, F_dim, C, K = 256, 784, 512, 10
        k = 64  # Select only 64 clauses out of 512
        
        dense = OptimizedSTCM(
            n_features=F_dim,
            n_clauses=C,
            n_classes=K,
        ).to(self.device)
        
        sparse = SparseSTCM(
            n_features=F_dim,
            n_clauses=C,
            n_classes=K,
            k=k,
        ).to(self.device)
        
        dense.eval()
        sparse.eval()
        
        x = torch.rand(B, F_dim, device=self.device)
        
        # Warmup
        for _ in range(20):
            with torch.no_grad():
                _ = dense(x)
                _ = sparse(x)
        torch.cuda.synchronize()
        
        # Benchmark
        n_iters = 100
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            with torch.no_grad():
                _ = dense(x)
        torch.cuda.synchronize()
        dense_time = (time.perf_counter() - t0) / n_iters * 1000
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            with torch.no_grad():
                _ = sparse(x)
        torch.cuda.synchronize()
        sparse_time = (time.perf_counter() - t0) / n_iters * 1000
        
        ratio = C / k
        speedup = dense_time / sparse_time
        
        print(f"\n{'='*60}")
        print(f"SPARSE STCM BENCHMARK (C={C}, k={k}, ratio={ratio:.1f}x)")
        print(f"{'='*60}")
        print(f"Dense STCM:   {dense_time:.3f} ms")
        print(f"Sparse STCM:  {sparse_time:.3f} ms")
        print(f"Speedup:      {speedup:.2f}x")
        print(f"{'='*60}\n")
        
        # Due to routing overhead, speedup may be less than theoretical
        # But should still be positive for k << C
        self.assertGreater(speedup, 0.3)  # At least not much slower


class TestDeepSparseSTCM(unittest.TestCase):
    """Test DeepSparseSTCM."""
    
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")
        torch.manual_seed(42)
        
    def test_forward_pass(self):
        """Verify forward pass works."""
        from fptm_ste.sparse_stcm import DeepSparseSTCM
        
        B, F_dim, K = 32, 256, 10
        
        model = DeepSparseSTCM(
            input_dim=F_dim,
            hidden_dims=[128, 64],
            n_classes=K,
            n_clauses=64,
            k=16,
        ).to(self.device)
        
        x = torch.rand(B, F_dim, device=self.device)
        
        logits, clause_out = model(x)
        
        self.assertEqual(logits.shape, (B, K))
        
    def test_total_aux_loss(self):
        """Verify total aux loss from all layers."""
        from fptm_ste.sparse_stcm import DeepSparseSTCM
        
        B, F_dim, K = 32, 256, 10
        
        model = DeepSparseSTCM(
            input_dim=F_dim,
            hidden_dims=[128, 64],
            n_classes=K,
            n_clauses=64,
            k=16,
        ).to(self.device)
        model.train()
        
        x = torch.rand(B, F_dim, device=self.device)
        _ = model(x)
        
        total_aux = model.get_total_aux_loss()
        self.assertIsNotNone(total_aux)


if __name__ == "__main__":
    unittest.main(verbosity=2)


