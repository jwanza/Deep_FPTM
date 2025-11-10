#!/usr/bin/env julia

println("Running Swin + TM hybrid variant via Python pipeline...")
ENV["TM_MNIST_VARIANTS"] = "hybrid"
run(`python python/fptm_ste/tests/run_mnist_equiv.py`)


