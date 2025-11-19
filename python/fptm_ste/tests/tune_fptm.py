import argparse
import json
import os
import subprocess
import sys
import itertools
from datetime import datetime

def run_tuning_sweep():
    print("--- Starting Hyperparameter Tuning for FPTM-Equiv ---")
    
    t_values = [10.0, 50.0, 100.0, 200.0]
    lr_values = [0.01, 0.05, 0.1, 0.5]
    
    best_acc = 0.0
    best_config = {}
    results_log = []
    
    os.makedirs("tmp", exist_ok=True)
    log_file = "tmp/fptm_tuning_results.json"
    
    for t, lr in itertools.product(t_values, lr_values):
        print(f"\n[Testing Config] T={t}, LR={lr}")
        
        cmd = [
            sys.executable, "python/fptm_ste/tests/run_mnist_equiv.py",
            "--dataset", "mnist",
            "--models", "fptm_equiv",
            "--epochs", "5",
            "--batch-size", "256",
            "--tm-feature-mode", "conv",
            "--tm-feature-cache", "/tmp/tm_features",
            "--tm-feature-grayscale",
            "--tm-feature-batch", "512",
            "--margin-loss",
            "--l1-lambda", "0.00001",
            "--prune-threshold", "0.05",
            "--tm-T", str(t),
            "--lr", str(lr),
            "--tm-n-clauses", "200"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stdout
            print(result.stderr)
            
            acc = 0.0
            for line in output.splitlines():
                if "FPTM-Equiv" in line and "test_acc=" in line:
                    parts = line.split()
                    for part in parts:
                        if part.startswith("test_acc="):
                            try:
                                val = float(part.split("=")[1])
                                if val > acc: acc = val
                            except ValueError:
                                pass
            
            print(f"  -> Result: Test Acc = {acc*100:.2f}%")
            
            results_log.append({
                "T": t,
                "lr": lr,
                "accuracy": acc
            })
            
            if acc > best_acc:
                best_acc = acc
                best_config = {"T": t, "lr": lr}
                print(f"  -> New Best!")
                
        except Exception as e:
            print(f"  -> Error: {e}")
            
    print("\n--- Tuning Completed ---")
    print(f"Best Accuracy: {best_acc*100:.2f}%")
    print(f"Best Config: {best_config}")
    
    with open(log_file, "w") as f:
        json.dump(results_log, f, indent=2)
    print(f"Results saved to {log_file}")

if __name__ == "__main__":
    run_tuning_sweep()

