import unittest
import torch
import torch.nn as nn
import time
from fptm_ste.incremental_tm import IncrementalSTCM, IncrementalConfig

class ComparisonBenchmark(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 128
        self.n_features = 784
        self.n_clauses = 1000
        self.n_classes = 10
        
        # Setup model
        self.config = IncrementalConfig(states_num=256)
        self.model = IncrementalSTCM(
            n_features=self.n_features,
            n_clauses=self.n_clauses,
            n_classes=self.n_classes,
            config=self.config,
        ).to(self.device)
        
        # Setup dummy data
        self.x = (torch.rand(self.batch_size, self.n_features, device=self.device) > 0.5).float()
        self.y = torch.randint(0, self.n_classes, (self.batch_size,), device=self.device)
        
        # Pre-compute forward pass for feedback benchmarking
        with torch.no_grad():
            self.logits, self.clause_outputs = self.model(self.x)

    def benchmark_vectorized(self, n_iters=100):
        """Benchmark the current vectorized implementation."""
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start = time.time()
        
        for _ in range(n_iters):
            self.model.incremental_feedback(self.x, self.y, self.clause_outputs, self.logits)
            
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        end = time.time()
        return (end - start) / n_iters

    def benchmark_loop_based_simulation(self, n_iters=5):
        """
        Simulate the old loop-based implementation behavior for benchmarking.
        The old implementation iterated:
        for b in batch:
            for c in clauses:
                check match
                if match: update state
        """
        # We simulate the computational cost of the loop
        # We don't need to be exact, just demonstrate the order of magnitude difference
        
        # Extract data to CPU to simulate "Python loop overhead" which was the killer
        x_cpu = self.x.cpu()
        clause_outputs_cpu = self.clause_outputs.cpu()
        half = self.n_clauses // 2
        
        start = time.time()
        
        for _ in range(n_iters):
            for b in range(self.batch_size):
                # Simulate the logic steps
                # 1. Vote calc
                pos_sum = clause_outputs_cpu[b, :half].sum().item()
                neg_sum = clause_outputs_cpu[b, half:].abs().sum().item()
                vote = pos_sum - neg_sum
                
                # 2. Loop clauses
                for j in range(self.n_clauses):
                    # Simulate check
                    if clause_outputs_cpu[b, j] > 0:
                        # Simulate update (random small tensor op)
                        pass
        
        end = time.time()
        return (end - start) / n_iters

    def test_speedup(self):
        print(f"\nBenchmarking Incremental Feedback ({self.device})...")
        
        vec_time = self.benchmark_vectorized()
        print(f"Vectorized (Current): {vec_time*1000:.3f} ms/batch")
        
        # Only run loop simulation for a few iters because it's slow
        loop_time = self.benchmark_loop_based_simulation()
        print(f"Loop-based (Baseline): {loop_time*1000:.3f} ms/batch")
        
        speedup = loop_time / vec_time
        print(f"Speedup: {speedup:.1f}x")
        
        # Assert significant speedup (at least 10x, usually >100x)
        self.assertTrue(speedup > 10, f"Speedup {speedup:.1f}x is less than expected 10x")

if __name__ == '__main__':
    unittest.main()

