#!/usr/bin/env julia

println("Running TM Transformer variant via Python pipeline...")
ENV["TM_MNIST_VARIANTS"] = "transformer"
run(`python python/fptm_ste/tests/run_mnist_equiv.py`)


