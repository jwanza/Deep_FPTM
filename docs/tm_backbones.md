# TM Vision Backbone Design Notes

## SwinTM Backbone
- Replaces Swin feed-forward layers with `FuzzyPatternTM_STE` blocks inside each window attention block.
- Pipeline: patch embed (stride 4) → windowed MHSA (with padding + shift support) → TM feed-forward (`WindowTMFeedForward`) → residual/drop path.
- Handles non-divisible spatial dims by padding before window partition and trimming afterward; supports MNIST-scale inputs.
- Stage heads use TM classifiers after global pooling; per-stage logits/clauses recorded via `last_stage_logits` and `last_attention_weights`.
- Configured via `SwinTMStageConfig` dataclass (embed dim, depth, heads, window size, clause budgets, τ, drop path, dropout).
- Presets (`build_swin_stage_configs`) mirror Tiny/Small/Base/Large but CLI allows custom hidden/clause counts suitable for small images.
- Gradients flow through MHSA → Sigmoid projections → TM STE → residual gating; all residuals use out-of-place additions to avoid autograd conflicts.

## ResNetTM Backbone
- Mirrors ResNet architecture with TM-enhanced residual blocks (`BasicTMBlock`, `BottleneckTMBlock`).
- Each block: conv+BN stack → project to TM hidden → TM clauses → linear projection back → residual add + ReLU.
- Stage outputs pooled to `tm_hidden` via linear+sigmoid and classified by TM heads; final TM head mirrors stage config.
- `ResNetTMConfig` controls layer counts, channels, TM hidden size, clause budget, τ; helper constructors expose ResNet18/34/50/101 variants.
- `last_stage_logits`/`last_stage_clauses` store per-stage results for accuracy breakdown and diagnostics.
- Avoids in-place residual modifications to maintain STE gradient correctness.

## Input Channel Handling
- `run_pyramid.py` auto-detects dataset channels and records them in run metadata (`dataset_channels`, `target_channels`).
- New CLI flags:
  - `--force-channels`: override detected channel count (e.g., force 3 for RGB-style backbones).
  - `--auto-expand-grayscale/--no-auto-expand-grayscale`: control whether single-channel tensors are duplicated to match the requested count.
- `fptm_ste.utils.data.ChannelAdjust` powers the duplication/reduction logic and is covered by unit tests in `test_data_utils.py`.
- Grayscale datasets can now drive color-aware backbones without manual preprocessing; RGB datasets pass through unchanged.
- When `tm-variant` is `swin`, `swin_tm`, or `resnet_tm`, inputs are automatically resized to 224×224 with bicubic filtering so stage geometries match native backbones.

## Instrumentation & Logging
- `run_pyramid.py` extended with helper utilities:
  - `extract_final_and_stage_logits` to normalize outputs across Pyramid, Swin-like, SwinTM, and ResNetTM.
  - `summarize_attention_weights` + `to_serializable` to surface attention/diagnostic tensors as JSON-safe scalars.
  - `evaluate` now returns overall accuracy, per-stage accuracies, and averaged attention metrics for all variants.
- CLI options cover new variants (`--tm-variant {pyramid,swin,swin_tm,resnet_tm}`) plus Swin/ResNet-specific knobs (presets, clause budgets, drop path, variant selection).
- Experiment logger stores per-stage accuracy fields (`stage_acc_i`) and attention summaries in CSV/JSON for downstream analysis.

## Gradient Flow Considerations
- SwinTM: padding/shift operations maintain differentiability; STE gradients backprop through TM clause banks and gating parameters; drop paths control residual blend.
- ResNetTM: Sigmoid-projected TM outputs combined with linear projectors ensure smooth gradients; residual paths remain contiguous to avoid version skew.
- Both backbones share the STE annealing utilities (`anneal_ste_factor`) so τ schedules impact every TM layer uniformly.

## Current Status
- Unit tests (`test_tm_backbones.py`) validate forward/backward consistency for minimal SwinTM and ResNetTM instances.
- Smoke runs (`run_pyramid.py`) confirm training loop compatibility; stage accuracy and attention metrics logged for SwinTM and ResNetTM variants.
