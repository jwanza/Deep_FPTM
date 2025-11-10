# Pyramid TM Roadmap Summary

## Baselines
- `run_baselines.py --epochs 10` records reference metrics for `FuzzyPatternTM_STE` (≈0.85 test acc) and `DeepTMNetwork` (≈0.95 test acc) with per-epoch logs under `python/fptm_ste/tests/experiments/`.

## Multi-Scale PyramidTM
- `pyramid_tm.py` implements adaptive pooling stages (28→14→7→1) with clause pooling and logit normalisation to balance stage votes.
- `PyramidStageConfig` exposes clause budgets, dropout, and pooling modes for each stage.
- `run_pyramid.py --tm-variant pyramid` trains the stack; a 5-epoch MNIST run reached ≈0.91 test accuracy in ~23 s/epoch on GPU.

## Swin-Like Window TM
- `swin_pyramid_tm.py` adds learnable patch embedding, shifted-window TM blocks, clause pooling, and optional scale attention while staying pure TM.
- Stages reuse `TM_Attention_Oracle`; `run_pyramid.py --tm-variant swin` exposes CLI knobs for window size, depths, clause budgets, and patch size.
- A 5-epoch MNIST trial (CPU, no scale attention, 128 clauses per stage) delivered ≈0.92 test accuracy, albeit at ~150 s/epoch due to window-level TM blocks; accuracy improved ~1.3 pts over the pooling pyramid under comparable settings.

## Attention & Diagnostics
- Clause-level and scale-level attention (with entropy regularisation) are enabled via `run_pyramid.py --no-clause-attention/--no-scale-attention` flags.
- `collect_tm_diagnostics` surfaces τ values, clause sparsity, and attention weights; metrics are logged per epoch (JSON/CSV).
- `analyze_pyramid_export.py` summarises exported clause bundles and attention weights (works for both pyramid and Swin variants).

## Training Enhancements
- Warm-up + cosine LR (`--warmup-epochs`, `--min-lr`) and gradient accumulation (`--grad-accum`) stabilise training.
- Optional EMA tracking (`--ema-decay`) and input noise (`--input-noise-std`) support regularisation experiments.
- Stage overrides (`--stage-clauses`, `--stage-dropouts`) and Swin-specific overrides (`--swin-embed-dims`, `--swin-depths`, `--swin-*clauses`) expose architecture tuning from the CLI.

## Reproducible Runs
- `run_pyramid.py --tm-variant {pyramid,swin}` is the primary entry point; logs and optional clause exports land in `python/fptm_ste/tests/experiments/`.
- `mnist_pyramid_sota.py --variant {pyramid,swin}` wraps 30-epoch presets (pyramid uses EMA + warm-up + noise; Swin mirrors those plus window configs).
- Example Swin run with export (CPU):
  ```bash
  CUDA_VISIBLE_DEVICES="" python python/fptm_ste/tests/experiments/run_pyramid.py \
      --tm-variant swin --epochs 5 --swin-embed-dims 96,192,384,768 \
      --swin-depths 2,2,4,2 --swin-window-size 4 --swin-clauses 128,128,128,128 \
      --no-scale-attention --export-path python/fptm_ste/tests/experiments/swin_export.json
  ```

## Current Results Snapshot
| Model | Test Acc (epochs) | Notes |
|-------|-------------------|-------|
| FuzzyPatternTM_STE | 0.85 (10) | baseline single-stage |
| DeepTMNetwork | 0.95 (10) | residual TM stack |
| PyramidTM (no attention) | 0.98 (20) | clause pooling & τ anneal |
| PyramidTM + attention | 0.97 (20) | scale weights + clause attention |
| PyramidTM SOTA config | 0.96 (30) | EMA + warm-up + noise |
| Swin-Like TM (window=4) | 0.92 (5, CPU) | shifted windows, window TM blocks |

## Next Steps
- Tune Swin window sizes / patch sizes to narrow runtime gap versus pooling pyramid.
- Extend exports to include learned scale weights per epoch for deeper interpretability.
- Port both variants to Fashion-MNIST/CIFAR using the Swin-style TM for a unified benchmark suite.
