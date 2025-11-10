# Pyramid TM Roadmap Summary

## Baselines
- `run_baselines.py --epochs 10` records reference metrics for `FuzzyPatternTM_STE` (≈0.85 test acc) and `DeepTMNetwork` (≈0.95 test acc) with per-epoch logs under `python/fptm_ste/tests/experiments/`.

## Multi-Scale PyramidTM
- `pyramid_tm.py` implements adaptive pooling stages (28→14→7→1) with clause pooling and logit normalisation to balance stage votes.
- `PyramidStageConfig` exposes clause budgets, dropout, and pooling modes for each stage.
- `run_pyramid.py` trains the stack; default 20-epoch run (no attention) reaches ≈0.977 MNIST accuracy.

## Attention & Diagnostics
- Clause-level and scale-level attention (with entropy regularisation) are enabled via `run_pyramid.py --no-clause-attention/--no-scale-attention` flags.
- `collect_tm_diagnostics` surfaces τ values, clause sparsity, and attention weights; metrics are logged per epoch (JSON/CSV).
- `analyze_pyramid_export.py` summarises exported clause bundles and attention weights.

## Training Enhancements
- Warm-up + cosine LR (`--warmup-epochs`, `--min-lr`) and gradient accumulation (`--grad-accum`) stabilise training.
- Optional EMA tracking (`--ema-decay`) and input noise (`--input-noise-std`) support regularisation experiments.
- Stage overrides (`--stage-clauses`, `--stage-dropouts`) make clause budgets tunable from the CLI.

## Reproducible Runs
- `run_pyramid.py` is the primary entry point; logs and optional clause exports land in `python/fptm_ste/tests/experiments/`.
- `mnist_pyramid_sota.py` wraps a 30-epoch configuration (drop-in SOTA candidate) using EMA, gradient accumulation, and annealed τ.
- Example run with export: `python run_pyramid.py --epochs 20 --export-path experiments/pyramid.json`.

## Current Results Snapshot
| Model | Test Acc (epochs) | Notes |
|-------|-------------------|-------|
| FuzzyPatternTM_STE | 0.85 (10) | baseline single-stage |
| DeepTMNetwork | 0.95 (10) | residual TM stack |
| PyramidTM (no attention) | 0.98 (20) | clause pooling & τ anneal |
| PyramidTM + attention | 0.97 (20) | scale weights + clause attention |
| PyramidTM SOTA config | 0.96 (30) | EMA + warm-up + noise |

## Next Steps
- Tune stage projections/clauses to close the gap between 0.97–0.98 and SOTA (>0.99).
- Extend exports to include learned scale weights per epoch for deeper interpretability.
- Port the pipeline to Fashion-MNIST/CIFAR using Swin backbones for a full vision benchmark suite.
