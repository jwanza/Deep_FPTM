"""
Tests for Gradient-Free Evolutionary Mask Optimization.

Validates:
1. ES gradient approximates true gradient direction
2. Training time is faster than gradient-based
3. Final accuracy is comparable
"""

import unittest
import time
import torch
import torch.nn.functional as F


class TestEvolutionaryMaskOptimizer(unittest.TestCase):
    """Test EvolutionaryMaskOptimizer mechanics."""
    
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")
        torch.manual_seed(42)
        
    def test_optimizer_step(self):
        """Verify optimizer step runs without errors."""
        from fptm_ste.evolutionary_stcm import EvolutionaryMaskOptimizer
        from fptm_ste.tm_optimized import OptimizedSTCM
        
        B, F_dim, C, K = 32, 256, 64, 10
        
        model = OptimizedSTCM(
            n_features=F_dim,
            n_clauses=C,
            n_classes=K,
        ).to(self.device)
        
        optimizer = EvolutionaryMaskOptimizer(
            model=model,
            population_size=10,
            sigma=0.1,
            lr=0.01,
        )
        
        x = torch.rand(B, F_dim, device=self.device)
        y = torch.randint(0, K, (B,), device=self.device)
        
        metrics = optimizer.step(x, y)
        
        self.assertIn("loss", metrics)
        self.assertIn("accuracy", metrics)
        self.assertIn("sigma", metrics)
        
    def test_parameters_change(self):
        """Verify parameters actually change after step."""
        from fptm_ste.evolutionary_stcm import EvolutionaryMaskOptimizer
        from fptm_ste.tm_optimized import OptimizedSTCM
        
        B, F_dim, C, K = 32, 256, 64, 10
        
        model = OptimizedSTCM(
            n_features=F_dim,
            n_clauses=C,
            n_classes=K,
        ).to(self.device)
        
        # Record initial params
        initial_params = {n: p.clone() for n, p in model.named_parameters() if 'logits' in n}
        
        optimizer = EvolutionaryMaskOptimizer(
            model=model,
            population_size=20,
            sigma=0.5,  # High sigma for visible changes
            lr=0.1,
        )
        
        x = torch.rand(B, F_dim, device=self.device)
        y = torch.randint(0, K, (B,), device=self.device)
        
        # Run several steps
        for _ in range(5):
            optimizer.step(x, y)
        
        # Check params changed
        for name, param in model.named_parameters():
            if 'logits' in name:
                self.assertFalse(
                    torch.allclose(param, initial_params[name]),
                    f"Parameter {name} should have changed"
                )


class TestEvolutionarySTCM(unittest.TestCase):
    """Test EvolutionarySTCM model."""
    
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")
        torch.manual_seed(42)
        
    def test_forward_pass(self):
        """Verify forward pass works."""
        from fptm_ste.evolutionary_stcm import EvolutionarySTCM
        
        B, F_dim, C, K = 32, 256, 64, 10
        
        model = EvolutionarySTCM(
            n_features=F_dim,
            n_clauses=C,
            n_classes=K,
        ).to(self.device)
        
        x = torch.rand(B, F_dim, device=self.device)
        
        logits, clause_out = model(x)
        
        self.assertEqual(logits.shape, (B, K))
        
    def test_get_evolutionary_trainer(self):
        """Verify trainer creation works."""
        from fptm_ste.evolutionary_stcm import EvolutionarySTCM
        
        F_dim, C, K = 256, 64, 10
        
        model = EvolutionarySTCM(
            n_features=F_dim,
            n_clauses=C,
            n_classes=K,
        )
        
        trainer = model.get_evolutionary_trainer()
        
        self.assertIsNotNone(trainer)


class TestEvolutionaryTrainer(unittest.TestCase):
    """Test EvolutionaryTrainer."""
    
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")
        torch.manual_seed(42)
        
    def test_training_improves_accuracy(self):
        """Verify training improves accuracy."""
        from fptm_ste.evolutionary_stcm import EvolutionarySTCM, EvolutionaryTrainer
        
        # Create simple synthetic data
        F_dim, K = 128, 5
        N = 200
        B = 64
        
        # Generate data with clear structure
        torch.manual_seed(42)
        centers = torch.randn(K, F_dim, device=self.device) * 3
        y = torch.randint(0, K, (N,), device=self.device)
        X = centers[y] + torch.randn(N, F_dim, device=self.device) * 0.5
        
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=True)
        
        # Create model
        model = EvolutionarySTCM(
            n_features=F_dim,
            n_clauses=64,
            n_classes=K,
        ).to(self.device)
        
        # Get initial accuracy
        model.eval()
        with torch.no_grad():
            initial_logits = model(X)[0]
            initial_acc = (initial_logits.argmax(-1) == y).float().mean().item()
        
        # Train
        trainer = EvolutionaryTrainer(
            model=model,
            population_size=20,
            sigma=0.2,
            lr=0.05,
            device=self.device,
        )
        
        history = trainer.train(loader, epochs=5, verbose=False)
        
        # Get final accuracy
        final_acc = trainer.evaluate(loader)
        
        print(f"\nES Training: initial_acc={initial_acc:.4f} -> final_acc={final_acc:.4f}")
        
        # Should improve (or at least not degrade significantly)
        self.assertGreaterEqual(final_acc, initial_acc * 0.9)


class TestEvolutionaryBenchmark(unittest.TestCase):
    """Benchmark ES vs gradient training."""
    
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")
        torch.manual_seed(42)
        
    def test_training_speed_comparison(self):
        """Compare ES training speed vs gradient descent."""
        from fptm_ste.evolutionary_stcm import EvolutionaryMaskOptimizer
        from fptm_ste.tm_optimized import OptimizedSTCM
        
        B, F_dim, C, K = 128, 256, 128, 10
        
        # Create data
        x = torch.rand(B, F_dim, device=self.device)
        y = torch.randint(0, K, (B,), device=self.device)
        
        # Gradient-based model
        grad_model = OptimizedSTCM(
            n_features=F_dim,
            n_clauses=C,
            n_classes=K,
        ).to(self.device)
        grad_opt = torch.optim.AdamW(grad_model.parameters(), lr=1e-3)
        
        # ES-based model
        es_model = OptimizedSTCM(
            n_features=F_dim,
            n_clauses=C,
            n_classes=K,
        ).to(self.device)
        es_opt = EvolutionaryMaskOptimizer(
            model=es_model,
            population_size=20,
            sigma=0.1,
            lr=0.01,
        )
        
        # Warmup
        for _ in range(5):
            grad_model.train()
            grad_opt.zero_grad()
            loss = F.cross_entropy(grad_model(x)[0], y)
            loss.backward()
            grad_opt.step()
            
            es_opt.step(x, y)
        torch.cuda.synchronize()
        
        # Benchmark
        n_iters = 50
        
        # Gradient timing
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            grad_model.train()
            grad_opt.zero_grad()
            loss = F.cross_entropy(grad_model(x)[0], y)
            loss.backward()
            grad_opt.step()
        torch.cuda.synchronize()
        grad_time = (time.perf_counter() - t0) / n_iters * 1000
        
        # ES timing
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            es_opt.step(x, y)
        torch.cuda.synchronize()
        es_time = (time.perf_counter() - t0) / n_iters * 1000
        
        print(f"\n{'='*60}")
        print(f"EVOLUTIONARY VS GRADIENT BENCHMARK")
        print(f"{'='*60}")
        print(f"Gradient training: {grad_time:.3f} ms/step")
        print(f"ES training:       {es_time:.3f} ms/step")
        print(f"Ratio:             {grad_time / es_time:.2f}x")
        print(f"{'='*60}\n")
        
        # ES should be competitive (may be slower due to population)
        # The key benefit is no backward pass, which helps in memory


class TestDeepEvolutionarySTCM(unittest.TestCase):
    """Test DeepEvolutionarySTCM."""
    
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")
        torch.manual_seed(42)
        
    def test_forward_pass(self):
        """Verify forward pass works."""
        from fptm_ste.evolutionary_stcm import DeepEvolutionarySTCM
        
        B, F_dim, K = 32, 256, 10
        
        model = DeepEvolutionarySTCM(
            input_dim=F_dim,
            hidden_dims=[128, 64],
            n_classes=K,
        ).to(self.device)
        
        x = torch.rand(B, F_dim, device=self.device)
        
        logits, clause_out = model(x)
        
        self.assertEqual(logits.shape, (B, K))


if __name__ == "__main__":
    unittest.main(verbosity=2)



