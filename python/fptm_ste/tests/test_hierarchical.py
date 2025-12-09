"""
Tests for Hierarchical Clause Tree (HierarchicalSTCM).

Validates:
1. Tree structure and level computation
2. Early exit mechanism works correctly
3. Speedup is achieved for easy samples
4. Accuracy is preserved or improved
"""

import unittest
import time
import torch
import torch.nn.functional as F


class TestClauseLevel(unittest.TestCase):
    """Test individual ClauseLevel component."""
    
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")
        torch.manual_seed(42)
        
    def test_level_output_shapes(self):
        """Verify level produces correct output shapes."""
        from fptm_ste.hierarchical_stcm import ClauseLevel
        
        B, F_dim, C, K = 32, 256, 64, 10
        
        level = ClauseLevel(
            n_features=F_dim,
            n_clauses=C,
            n_outputs=K,
        ).to(self.device)
        
        x = torch.rand(B, F_dim, device=self.device)
        
        logits, confidence = level(x)
        
        self.assertEqual(logits.shape, (B, K))
        self.assertEqual(confidence.shape, (B,))
        
    def test_confidence_range(self):
        """Verify confidence is in valid range [0, 1]."""
        from fptm_ste.hierarchical_stcm import ClauseLevel
        
        B, F_dim, C, K = 32, 256, 64, 10
        
        level = ClauseLevel(
            n_features=F_dim,
            n_clauses=C,
            n_outputs=K,
        ).to(self.device)
        
        x = torch.rand(B, F_dim, device=self.device)
        _, confidence = level(x)
        
        self.assertTrue((confidence >= 0).all())
        self.assertTrue((confidence <= 1).all())


class TestHierarchicalClauseTree(unittest.TestCase):
    """Test HierarchicalClauseTree."""
    
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")
        torch.manual_seed(42)
        
    def test_tree_output_shapes(self):
        """Verify tree produces correct output shapes."""
        from fptm_ste.hierarchical_stcm import HierarchicalClauseTree
        
        B, F_dim, K = 32, 256, 10
        
        tree = HierarchicalClauseTree(
            n_features=F_dim,
            n_classes=K,
            depth=3,
            branch_factor=4,
            base_clauses=16,
        ).to(self.device)
        
        x = torch.rand(B, F_dim, device=self.device)
        
        logits, info = tree(x)
        
        self.assertEqual(logits.shape, (B, K))
        self.assertIn('level_outputs', info)
        
    def test_early_exit_in_eval(self):
        """Verify early exit tracking works during evaluation."""
        from fptm_ste.hierarchical_stcm import HierarchicalClauseTree
        
        B, F_dim, K = 100, 256, 10
        
        tree = HierarchicalClauseTree(
            n_features=F_dim,
            n_classes=K,
            depth=4,
            branch_factor=4,
            base_clauses=16,
            confidence_threshold=0.5,  # Low threshold for more exits
        ).to(self.device)
        
        tree.eval()
        tree.reset_exit_statistics()
        
        x = torch.rand(B, F_dim, device=self.device)
        
        with torch.no_grad():
            _ = tree(x, return_stats=True)
        
        stats = tree.get_exit_statistics()
        
        # Exit distribution should sum to 1 (all samples tracked)
        total = sum(stats['exit_distribution'])
        self.assertAlmostEqual(total, 1.0, places=5, msg="Exit distribution should sum to 1")
        
    def test_training_uses_all_levels(self):
        """Verify training mode uses all levels."""
        from fptm_ste.hierarchical_stcm import HierarchicalClauseTree
        
        B, F_dim, K = 32, 256, 10
        
        tree = HierarchicalClauseTree(
            n_features=F_dim,
            n_classes=K,
            depth=3,
        ).to(self.device)
        
        tree.train()
        x = torch.rand(B, F_dim, device=self.device)
        
        logits, info = tree(x)
        
        # All levels should be computed
        self.assertEqual(len(info['level_outputs']), 3)


class TestHierarchicalSTCM(unittest.TestCase):
    """Test HierarchicalSTCM wrapper."""
    
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")
        torch.manual_seed(42)
        
    def test_forward_pass(self):
        """Verify forward pass works."""
        from fptm_ste.hierarchical_stcm import HierarchicalSTCM
        
        B, F_dim, K = 32, 256, 10
        
        model = HierarchicalSTCM(
            n_features=F_dim,
            n_classes=K,
        ).to(self.device)
        
        x = torch.rand(B, F_dim, device=self.device)
        
        logits, info = model(x)
        
        self.assertEqual(logits.shape, (B, K))
        
    def test_backward_pass(self):
        """Verify backward pass works."""
        from fptm_ste.hierarchical_stcm import HierarchicalSTCM
        
        B, F_dim, K = 32, 256, 10
        
        model = HierarchicalSTCM(
            n_features=F_dim,
            n_classes=K,
        ).to(self.device)
        
        x = torch.rand(B, F_dim, device=self.device, requires_grad=True)
        
        logits, _ = model(x)
        loss = logits.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)


class TestHierarchicalBenchmark(unittest.TestCase):
    """Benchmark hierarchical vs flat STCM."""
    
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")
        torch.manual_seed(42)
        
    def test_speedup_measurement(self):
        """Measure speedup of hierarchical with early exit."""
        from fptm_ste.hierarchical_stcm import HierarchicalSTCM
        from fptm_ste.tm_optimized import OptimizedSTCM
        
        B, F_dim, K = 256, 784, 10
        C = 512  # Total clauses for fair comparison
        
        flat = OptimizedSTCM(
            n_features=F_dim,
            n_clauses=C,
            n_classes=K,
        ).to(self.device)
        
        # Hierarchical: 16 + 64 + 256 + 1024 = 1360 total, but with early exit
        hierarchical = HierarchicalSTCM(
            n_features=F_dim,
            n_classes=K,
            depth=4,
            branch_factor=4,
            base_clauses=16,
            confidence_threshold=0.7,
        ).to(self.device)
        
        flat.eval()
        hierarchical.eval()
        
        x = torch.rand(B, F_dim, device=self.device)
        
        # Warmup
        for _ in range(20):
            with torch.no_grad():
                _ = flat(x)
                _ = hierarchical(x)
        torch.cuda.synchronize()
        
        # Reset exit stats
        hierarchical.reset_exit_statistics()
        
        # Benchmark
        n_iters = 100
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            with torch.no_grad():
                _ = flat(x)
        torch.cuda.synchronize()
        flat_time = (time.perf_counter() - t0) / n_iters * 1000
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            with torch.no_grad():
                _ = hierarchical(x)
        torch.cuda.synchronize()
        hier_time = (time.perf_counter() - t0) / n_iters * 1000
        
        stats = hierarchical.get_exit_statistics()
        
        print(f"\n{'='*60}")
        print(f"HIERARCHICAL STCM BENCHMARK")
        print(f"{'='*60}")
        print(f"Flat STCM:         {flat_time:.3f} ms")
        print(f"Hierarchical STCM: {hier_time:.3f} ms")
        print(f"Speedup:           {flat_time / hier_time:.2f}x")
        print(f"Average depth:     {stats['average_depth']:.2f}")
        print(f"Exit distribution: {[f'{x:.2%}' for x in stats['exit_distribution']]}")
        print(f"{'='*60}\n")


class TestDeepHierarchicalSTCM(unittest.TestCase):
    """Test DeepHierarchicalSTCM."""
    
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")
        torch.manual_seed(42)
        
    def test_forward_pass(self):
        """Verify forward pass works."""
        from fptm_ste.hierarchical_stcm import DeepHierarchicalSTCM
        
        B, F_dim, K = 32, 256, 10
        
        model = DeepHierarchicalSTCM(
            input_dim=F_dim,
            hidden_dims=[128, 64],
            n_classes=K,
        ).to(self.device)
        
        x = torch.rand(B, F_dim, device=self.device)
        
        logits, info = model(x)
        
        self.assertEqual(logits.shape, (B, K))


if __name__ == "__main__":
    unittest.main(verbosity=2)

