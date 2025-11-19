
import subprocess
import sys
import os

def run_test():
    print("Running regression test for dimension mismatch...")
    
    if not os.path.exists("python/fptm_ste/tests/run_mnist_equiv.py"):
        print("Error: Run from project root.")
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("python") + ":" + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        "python/fptm_ste/tests/run_mnist_equiv.py",
        "--dataset", "mnist",
        "--models", "transformer",
        "--epochs", "1",
        "--batch-size", "32",
        "--transformer-clause-attention",
        "--transformer-auto-clause",
        "--transformer-auto-clause-batches", "2",
        "--report-train-acc"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        print("Success!")
    except subprocess.CalledProcessError as e:
        print("Failed!")
        print(e.stderr)
        sys.exit(1)

if __name__ == "__main__":
    run_test()

