
import time
import torch
import torch.nn as nn
from fptm_ste.conv_tm import ConvSTE2d, ConvSTCM2d, ConvTM2d

def benchmark_layer(name, layer_cls, input_shape, device='cuda', iterations=100):
    if not torch.cuda.is_available() and device == 'cuda':
        print("CUDA not available, skipping benchmark")
        return

    B, C, H, W = input_shape
    x = torch.rand(B, C, H, W, device=device)
    
    # Initialize layer
    try:
        model = layer_cls(
            in_channels=C, 
            out_channels=32, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            n_clauses=128
        ).to(device)
    except Exception as e:
        print(f"Failed to init {name}: {e}")
        return
    
    # Warmup
    for _ in range(10):
        _ = model(x)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    start_mem = torch.cuda.memory_allocated()
    max_mem = 0
    
    for _ in range(iterations):
        _ = model(x)
        mem = torch.cuda.max_memory_allocated()
        max_mem = max(max_mem, mem)
        
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / iterations * 1000
    peak_mem_mb = (max_mem - start_mem) / 1024 / 1024
    
    print(f"[{name}] Input: {input_shape}")
    print(f"  Avg Time: {avg_time:.2f} ms")
    print(f"  Peak Mem Overhead: {peak_mem_mb:.2f} MB")
    return avg_time

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Benchmarking on {device}")
    
    input_shape = (32, 3, 32, 32) # CIFAR-like batch
    
    benchmark_layer("ConvSTE2d (Baseline)", ConvSTE2d, input_shape, device)
    benchmark_layer("ConvSTCM2d (Baseline)", ConvSTCM2d, input_shape, device)

    # Check for optimized version if available
    try:
        from fptm_ste.conv_tm import ConvTM2dOptimized
        benchmark_layer("ConvTM2dOptimized", ConvTM2dOptimized, input_shape, device)
    except ImportError:
        print("ConvTM2dOptimized not implemented yet.")

