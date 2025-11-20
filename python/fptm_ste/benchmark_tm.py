import time
import torch
import torch.nn as nn
from fptm_ste.tm import FuzzyPatternTM_STE, FuzzyPatternTM_STCM, FuzzyPatternTMFPTM, prepare_tm_input

def benchmark_forward(model, input_data, name, batch_size=512, iterations=100):
    device = input_data.device
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_data)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(input_data)
    torch.cuda.synchronize()
    end = time.time()
    
    total_time = end - start
    avg_time_ms = (total_time / iterations) * 1000
    throughput = (batch_size * iterations) / total_time
    print(f"{name:25s} | Batch: {batch_size} | Avg Time: {avg_time_ms:.2f} ms | Throughput: {throughput:.0f} samples/s")
    return throughput

def run_benchmarks():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmarks.")
        return

    device = torch.device("cuda")
    batch_size = 2048
    n_features = 28 * 28 * 1
    n_clauses = 2000
    n_classes = 10
    
    print(f"Benchmarking TM variants (Features={n_features}, Clauses={n_clauses}, Classes={n_classes}, Batch={batch_size})...")
    print("-" * 90)
    
    # Create dummy input
    x = torch.rand(batch_size, n_features, device=device)
    
    # 1. FuzzyPatternTM_STE
    ste_model = FuzzyPatternTM_STE(n_features, n_clauses, n_classes).to(device)
    benchmark_forward(ste_model, x, "FuzzyPatternTM_STE", batch_size)
    
    # 2. FuzzyPatternTM_STCM
    stcm_model = FuzzyPatternTM_STCM(n_features, n_clauses, n_classes).to(device)
    benchmark_forward(stcm_model, x, "FuzzyPatternTM_STCM", batch_size)
    
    # 3. FuzzyPatternTMFPTM
    fptm_model = FuzzyPatternTMFPTM(n_features, n_clauses, n_classes).to(device)
    benchmark_forward(fptm_model, x, "FuzzyPatternTMFPTM", batch_size)
    
    print("-" * 90)

if __name__ == "__main__":
    run_benchmarks()

