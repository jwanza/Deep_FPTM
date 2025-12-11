"""
Training Benchmark for STCM Variants.

Tests actual training performance and accuracy on MNIST-like data.
"""

import torch
import torch.nn.functional as F
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass
import sys


@dataclass
class TrainingResult:
    name: str
    final_accuracy: float
    train_time_per_epoch: float
    inference_time: float
    memory_mb: float
    epochs_trained: int


def create_mnist_like_data(n_train=5000, n_test=1000, n_features=784, n_classes=10, device='cuda'):
    """Create MNIST-like structured data."""
    torch.manual_seed(42)
    
    # Create class centers
    centers = torch.randn(n_classes, n_features, device=device) * 2
    
    # Generate train data
    train_y = torch.randint(0, n_classes, (n_train,), device=device)
    train_x = centers[train_y] + torch.randn(n_train, n_features, device=device) * 0.3
    train_x = torch.sigmoid(train_x)  # Normalize to [0, 1]
    
    # Generate test data
    test_y = torch.randint(0, n_classes, (n_test,), device=device)
    test_x = centers[test_y] + torch.randn(n_test, n_features, device=device) * 0.3
    test_x = torch.sigmoid(test_x)
    
    return train_x, train_y, test_x, test_y


def train_model(model, train_x, train_y, test_x, test_y, epochs=5, batch_size=64, lr=1e-3):
    """Train a model and return results."""
    device = train_x.device
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    n_batches = (len(train_x) + batch_size - 1) // batch_size
    
    # Training
    train_times = []
    for epoch in range(epochs):
        model.train()
        epoch_start = time.perf_counter()
        
        # Shuffle
        perm = torch.randperm(len(train_x), device=device)
        train_x_shuffled = train_x[perm]
        train_y_shuffled = train_y[perm]
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(train_x))
            
            batch_x = train_x_shuffled[start_idx:end_idx]
            batch_y = train_y_shuffled[start_idx:end_idx]
            
            optimizer.zero_grad()
            
            out = model(batch_x)
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out
            
            loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()
        
        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start
        train_times.append(epoch_time)
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        out = model(test_x)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
        
        torch.cuda.synchronize()
        inference_time = (time.perf_counter() - t0) * 1000
        
        preds = logits.argmax(dim=-1)
        accuracy = (preds == test_y).float().mean().item()
    
    # Memory
    torch.cuda.synchronize()
    memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    return TrainingResult(
        name=type(model).__name__,
        final_accuracy=accuracy,
        train_time_per_epoch=sum(train_times) / len(train_times),
        inference_time=inference_time,
        memory_mb=memory,
        epochs_trained=epochs,
    )


def run_training_benchmark():
    """Run comprehensive training benchmark."""
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.device("cuda")
    
    # Create data
    print("Creating synthetic MNIST-like data...")
    train_x, train_y, test_x, test_y = create_mnist_like_data(
        n_train=10000, n_test=2000, device=device
    )
    
    n_features = 784
    n_clauses = 256
    n_classes = 10
    epochs = 10
    
    print(f"\nConfig: train_size=10000, test_size=2000, features={n_features}")
    print(f"        clauses={n_clauses}, classes={n_classes}, epochs={epochs}")
    
    results = []
    
    # Models to benchmark
    models_config = [
        ("STCM", lambda: __import__('fptm_ste.tm', fromlist=['FuzzyPatternTM_STCM']).FuzzyPatternTM_STCM(
            n_features=n_features, n_clauses=n_clauses, n_classes=n_classes
        )),
        ("OptimizedSTCM", lambda: __import__('fptm_ste.tm_optimized', fromlist=['OptimizedSTCM']).OptimizedSTCM(
            n_features=n_features, n_clauses=n_clauses, n_classes=n_classes
        )),
        ("CompiledSTCM", lambda: __import__('fptm_ste.compiled_stcm', fromlist=['CompiledSTCM']).CompiledSTCM(
            n_features=n_features, n_clauses=n_clauses, n_classes=n_classes, compile_mode="reduce-overhead"
        )),
        ("SparseSTCM", lambda: __import__('fptm_ste.sparse_stcm', fromlist=['SparseSTCM']).SparseSTCM(
            n_features=n_features, n_clauses=n_clauses, n_classes=n_classes, k=64
        )),
        ("HierarchicalSTCM", lambda: __import__('fptm_ste.hierarchical_stcm', fromlist=['HierarchicalSTCM']).HierarchicalSTCM(
            n_features=n_features, n_classes=n_classes, depth=3, base_clauses=32
        )),
        ("UltimateSTCM", lambda: __import__('fptm_ste.ultimate_stcm', fromlist=['UltimateSTCM']).UltimateSTCM(
            n_features=n_features, n_classes=n_classes, depth=2, base_clauses=32, use_compile=False
        )),
    ]
    
    print("\n" + "="*90)
    print("TRAINING BENCHMARK")
    print("="*90 + "\n")
    
    for name, model_fn in models_config:
        print(f"Training {name}...", end=" ", flush=True)
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            model = model_fn()
            result = train_model(model, train_x, train_y, test_x, test_y, epochs=epochs)
            result.name = name
            results.append(result)
            
            print(f"Done. Accuracy: {result.final_accuracy:.4f}, Time/epoch: {result.train_time_per_epoch:.2f}s")
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    # Print results table
    print("\n" + "="*100)
    print("TRAINING RESULTS SUMMARY")
    print("="*100)
    print(f"{'Model':<20} {'Accuracy':<12} {'Time/Epoch':<15} {'Inference':<15} {'Memory':<12} {'Epochs':<8}")
    print("-"*100)
    
    baseline_time = results[0].train_time_per_epoch if results else 1.0
    
    for r in results:
        speedup = baseline_time / r.train_time_per_epoch
        print(f"{r.name:<20} {r.final_accuracy:<12.4f} {r.train_time_per_epoch:<15.2f}s {r.inference_time:<15.2f}ms {r.memory_mb:<12.1f}MB {r.epochs_trained:<8}")
    
    print("="*100)
    
    # Analysis
    print("\nKEY FINDINGS:")
    print("-" * 50)
    
    best_acc = max(results, key=lambda r: r.final_accuracy)
    fastest_train = min(results, key=lambda r: r.train_time_per_epoch)
    fastest_infer = min(results, key=lambda r: r.inference_time)
    lowest_mem = min(results, key=lambda r: r.memory_mb)
    
    print(f"✓ Best accuracy: {best_acc.name} ({best_acc.final_accuracy:.4f})")
    print(f"✓ Fastest training: {fastest_train.name} ({fastest_train.train_time_per_epoch:.2f}s/epoch)")
    print(f"✓ Fastest inference: {fastest_infer.name} ({fastest_infer.inference_time:.2f}ms)")
    print(f"✓ Lowest memory: {lowest_mem.name} ({lowest_mem.memory_mb:.1f}MB)")
    
    return results


if __name__ == "__main__":
    results = run_training_benchmark()



