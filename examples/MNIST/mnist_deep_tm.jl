#!/usr/bin/env julia

println("Running Deep TM variant via Python pipeline...")
ENV["TM_MNIST_VARIANTS"] = "deep_tm"
run(`python python/fptm_ste/tests/run_mnist_equiv.py`)


