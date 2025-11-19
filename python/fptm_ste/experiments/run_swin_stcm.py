
import os
import sys
import torch

# Add python directory to sys.path to ensure imports work
sys.path.append(os.path.join(os.getcwd(), "python"))

from fptm_ste.tests.run_mnist_equiv import main

if __name__ == "__main__":
    # Force set arguments for Swin + STCM experiment
    # We simulate command line arguments
    
    print("=== Running Swin + STCM SOTA Experiment ===")
    
    # Common args
    base_args = [
        "--dataset", "mnist",
        "--epochs", "10",
        "--batch-size", "64",
        "--tm-feature-mode", "none", # We want end-to-end training, not fixed features
        "--report-train-acc", 
        "--report-epoch-acc", 
        "--report-epoch-test"
    ]
    
    # Swin configuration
    # SOTA Swin: hierarchical, shifted windows. 
    # We use a small config for speed but sufficient for MNIST.
    swin_args = [
        "--models", "transformer",
        "--transformer-arch", "swin",
        "--transformer-patch", "2",       # Small patch for 28x28
        "--transformer-window", "4",
        "--transformer-depths", "2,2",
        "--transformer-embed-dims", "48,96", # Small embedding
        "--transformer-stage-heads", "3,6",
        "--transformer-mlp-ratio", "2.0,2.0",
    ]
    
    # STCM SOTA configuration
    # backend=stcm triggers FuzzyPatternTM_STCM
    stcm_args = [
        "--transformer-backend", "stcm", # Now supported in our modified code
        # We assume our modified UnifiedTMTransformer defaults to operator="capacity" and ternary_voting=False
        # But we want ternary_voting=True for SOTA.
        # run_mnist_equiv.py doesn't expose --ternary-voting arg yet.
        # However, we can rely on defaults if we modified them, or we must modify run_mnist_equiv.py to pass them.
        # I modified TMFeedForward to accept them, but UnifiedTMTransformer needs to receive them from run_mnist_equiv.py
    ]

    # Wait, I updated UnifiedTMTransformer to accept 'tm_operator' and 'ternary_voting'.
    # But I didn't update run_mnist_equiv.py to parse these args and pass them.
    # So currently they will use defaults (capacity, False).
    # I should update run_mnist_equiv.py first or I won't get ternary voting.
    
    # Actually, let's check run_mnist_equiv.py again to see where it instantiates UnifiedTMTransformer.
    pass


