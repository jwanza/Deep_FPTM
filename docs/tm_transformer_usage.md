# TM Transformer Usage & Tuning Guide

This guide shows how to launch the upgraded TM transformers, outlines every configurable parameter exposed by `run_mnist_equiv.py`, and lists practical tricks for pushing accuracy toward SOTA on CIFAR/ImageNet-style workloads.

## 1. Environment Setup

- Activate the virtual environment: `source venv/bin/activate`
- Ensure the package path is visible: `export PYTHONPATH=$PWD/python:$PYTHONPATH`
- Install runtime dependencies: `pip install -r requirements.txt`  
  (Important extras: `matplotlib` for overlay rendering.)

## 2. Core CLI Parameters (Quick Reference)

| Flag | Description |
|------|-------------|
| `--models transformer` | enables the TM transformer variant |
| `--transformer-arch {vit,swin}` | ViT-style global attention or Swin-style hierarchical windows |
| `--transformer-backend {ste,deeptm}` | TM feed-forward backend (lightweight vs higher accuracy) |
| `--transformer-patch <P>` | patch size for image tokenisation |
| `--transformer-depths` | comma-separated stage depths (Swin) or integer for ViT |
| `--transformer-stage-heads` | heads per stage (Swin) |
| `--transformer-embed-dims` | embedding width per stage (Swin) |
| `--transformer-mlp-ratio` | feed-forward expansion (float or list) |
| `--transformer-window` | Swin window size |
| `--transformer-drop-path` | max drop-path rate (scheduled per block) |
| `--transformer-drop-path-schedule` | drop-path schedule (`linear` or `cosine`) |
| `--transformer-dropout` | attention / TM dropout |
| `--transformer-ema-decay` | EMA decay (0 disables) |
| `--transformer-grad-checkpoint` | gradient checkpointing toggle |
| `--lr-layer-decay` | per-layer LR scaling factor (1.0 disables) |
| `--gradient-centralize` | enable gradient centralisation before each step |
| `--lr-cycle-steps` | cosine-restart cycle length in optimizer steps (0 disables) |
| `--randaugment`, `--mixup-alpha`, `--cutmix-alpha` | augmentation toggles |
| `--transformer-use-flash-attn` | leverage PyTorch SDPA/FlashAttention kernels when available |
| `--transformer-residual-attn` | enable cross-block residual attention connections |
| `--mixup-schedule`, `--cutmix-schedule` | mixup / CutMix strength schedule (`constant`, `linear`, `cosine`) |
| `--label-smoothing` | label smoothing factor applied to all CE losses |
| `--tm-prior-template` | initializes TM clauses with optional symmetry priors |
| `--transformer-auto-clause` | auto-tune clause budgets from diagnostic accuracies |
| `--transformer-clause-target` | total clause budget when auto tuning (0 keeps sum) |
| `--transformer-auto-clause-batches` | number of batches sampled during auto tuning |
| `--transformer-self-distill-weight`, `--transformer-self-distill-temp` | EMA-teacher self-distillation controls |
| `--transformer-contrastive-weight`, `--transformer-contrastive-temp` | supervised contrastive auxiliary controls |

Additional data-prep knobs:
- `--tm-feature-mode` with presets from `fptm_ste.datasets`, plus `--tm-feature-config`, `--tm-feature-size`, `--tm-feature-grayscale`, reuse cached boolean TM features when needed.

## 3. Example Commands

### 3.1 ViT + STE backend (CIFAR-10, 4×4 patches)
```bash
source venv/bin/activate
export PYTHONPATH=$PWD/python:$PYTHONPATH
python python/fptm_ste/tests/run_mnist_equiv.py \
  --dataset cifar10 --models transformer \
  --transformer-arch vit --transformer-backend ste \
  --transformer-patch 4 --transformer-drop-path 0.1 \
  --transformer-dropout 0.05 --transformer-ema-decay 0.995 \
  --mixup-alpha 0.8 --cutmix-alpha 1.0 --randaugment \
  --epochs 20 --batch-size 128
```

### 3.2 Swin + DeepTM backend (CIFAR-10, hierarchical)
```bash
python python/fptm_ste/tests/run_mnist_equiv.py \
  --dataset cifar10 --models transformer \
  --transformer-arch swin --transformer-backend deeptm \
  --transformer-patch 4 --transformer-depths 2,2,6,2 \
  --transformer-stage-heads 3,6,12,24 \
  --transformer-embed-dims 96,192,384,768 \
  --transformer-window 7 --transformer-drop-path 0.2 \
  --transformer-grad-checkpoint --transformer-ema-decay 0.999 \
  --mixup-alpha 0.8 --cutmix-alpha 1.0 --randaugment \
  --epochs 100 --batch-size 128
```

### 3.3 ViT + STE baseline (MNIST)
```bash
python python/fptm_ste/tests/run_mnist_equiv.py \
  --dataset mnist --models transformer \
  --transformer-arch vit --transformer-backend ste \
  --transformer-patch 2 --epochs 10 --batch-size 256
```

### 3.4 Interpretability Overlays (MNIST quick reference)
```bash
source SOURCE.THIS  # venv + PYTHONPATH
PYTHONPATH=python python3 python/fptm_ste/tests/run_mnist_equiv.py \
  --device cpu \  # generates overlays without optional CUDA kernels
  --dataset mnist \
  --models transformer \
  --transformer-arch vit \
  --transformer-backend ste \
  --transformer-patch 4 \
  --transformer-embed-dims 64 \
  --transformer-layers 1 \
  --transformer-heads 4 \
  --transformer-mlp-ratio 3.0 \
  --epochs 1 --batch-size 16 --test-batch-size 64 \
  --transformer-use-cls --transformer-pool cls \
  --visualize-overlays /tmp/vit_tm_overlays \
  --visualize-samples 4 \
  --visualize-topk 3
```

### 3.5 Distillation with Pretrained Teachers (CIFAR-10)
```bash
source SOURCE.THIS
PYTHONPATH=python python3 python/fptm_ste/tests/run_mnist_equiv.py \
  --dataset cifar10 --models transformer \
  --transformer-arch vit --transformer-backend ste \
  --transformer-patch 4 --epochs 100 --batch-size 256 \
  --transformer-embed-dims 96 --transformer-layers 8 --transformer-heads 6 \
  --distill-teacher-timm vit_base_patch16_224 \
  --distill-teacher-weight 0.7 \
  --distill-teacher-temp 2.0 \
  --transformer-self-distill-weight 0.5 \
  --transformer-self-distill-temp 2.0 \
  --transformer-clause-specialize \
  --transformer-clause-specialize-strength 0.5 \
  --visualize-overlays /tmp/vit_tm_kd --visualize-samples 6
```

### 3.6 Bit-packed vs Dense Comparison (MNIST)
```bash
# Bit-packed forward + fused voting
python python/fptm_ste/tests/run_mnist_equiv.py \
  --dataset mnist --models tm --epochs 5 --batch-size 256 --log-interval 20

# Dense baseline (full STE matmuls)
python python/fptm_ste/tests/run_mnist_equiv.py \
  --dataset mnist --models tm --epochs 5 --batch-size 256 --log-interval 20
```
Collect wall-clock timings (the runner logs per-epoch duration) and accuracy to quantify gains.

## 4. Tuning Recommendations by Component

### 4.1 Architecture & Tokenisation
- **Mode (`--transformer-arch`)**: ViT for simplicity or Swin for hierarchical attention that scales to higher resolutions.
- **Patch size (`--transformer-patch`)**: reduces sequence length. 4×4 or 8×8 for CIFAR; consider 16×16 for 224×224 images.
- **Depth/Heads/Channels**: `--transformer-depths`, `--transformer-stage-heads`, `--transformer-embed-dims` (Swin) or `--transformer-layers`, `--transformer-heads` (ViT) to match Tiny/Base/Small footprints.

### 4.2 TM Feed-Forward Backend
- **Backend switch (`--transformer-backend`)**: `ste` = faster, `deeptm` = higher accuracy.
- **Clause counts (`--transformer-clauses`)**: integer or per-stage tuple to scale clause banks with hidden width.
- **Tau (`--tm-tau`)**: default 0.5; adjust when clause budgets grow to keep gradients stable.

### 4.3 Attention & Normalisation
- **Dropout (`--transformer-dropout`)**: general regularisation (0.0–0.1 typical).
- **Drop-path (`--transformer-drop-path`)**: linearly scheduled up to provided max; raise beyond 0.1 for deep Swin stacks.
- **Window size (`--transformer-window`)**: adjust attention coverage, default 7.
- **CLS token**: `--transformer-use-cls` and `--transformer-pool` choose between CLS pooling and mean pooling.

### 4.4 Training-Time Tricks
- **EMA (`--transformer-ema-decay`)**: 0.995–0.999 stabilises evaluation under heavy augmentation.
- **Gradient checkpointing (`--transformer-grad-checkpoint`)**: saves memory for deep Swin models.
- **RandAugment (`--randaugment`, `--randaugment-n`, `--randaugment-m`)**: mimic ImageNet pipelines; defaults are n=2, m=9.
- **Mixup / CutMix (`--mixup-alpha`, `--cutmix-alpha`)**: typical values 0.8 / 1.0; disable on tiny datasets.
- **Dynamic aug schedules (`--mixup-schedule`, `--cutmix-schedule`)**: ramp strengths with `cosine` or `linear` ramps to ease early optimisation.
- **Label smoothing (`--label-smoothing`)**: 0.05–0.1 reduces overconfidence when Mixup is off.
- **Cosine LR restarts (`--lr-cycle-steps`)**: rejuvenate training late by cycling the cosine schedule (combine with warmup).
- **Epochs & batch size**: extend training (≥100 epochs) when augmentations and DeepTM backend are active.
- **TM priors (`--tm-prior-template`)**: `symmetric` seeds complementary clause masks (helpful on symmetric datasets like MNIST).
- **Clause auto-tuning (`--transformer-auto-clause`)**: reallocates clause budgets from diagnostic accuracies—great for Swin when some stages underperform.
- **Layer-wise LR decay (`--lr-layer-decay`)**: set ~0.85 to stabilise deep Swin stacks; pair with `--gradient-centralize`.
- **Self-distillation (`--transformer-self-distill-weight/temp`)**: blend EMA-teacher logits (KD style) into the training loss for better calibration.
- **Supervised contrastive head (`--transformer-contrastive-weight/temp`)**: regularise pooled features; especially helpful when mixup is mild.
- **Drop-path schedule (`--transformer-drop-path-schedule`)**: `cosine` emphasis on deeper blocks mitigates early-stage underfitting.
- **Flash attention (`--transformer-use-flash-attn`)**: activates PyTorch SDPA kernels (FlashAttention on NVIDIA) for faster multi-head attention.
- **Residual attention (`--transformer-residual-attn`)**: carry attention updates across blocks (ViT v2 style) to stabilise very deep stacks.

### 4.5 Boolean Feature Pipelines (Optional)
- `--tm-feature-mode {raw,fashion_aug,conv}` selects raw tensors or cached boolean features.
- `--tm-feature-config` references presets in `fptm_ste.datasets` (e.g., `cifar10`).
- `--tm-feature-size`, `--tm-feature-grayscale` override preprocessing resolution and channel handling.

### 4.6 Optimizer & Precision
- AdamW (runner default) with LR schedule flags already in the script (`--lr`, `--min-lr`, `--warmup-epochs`).
- AMP is automatic. Combine with `--transformer-grad-checkpoint` or reduced batch size for 16–24 GB GPUs.

## 5. Practical Tips
- Increase patch size before shrinking width/depth if you hit memory limits.
- Evaluate both STE and DeepTM backends—DeepTM typically adds a few points of accuracy.
- Combine EMA + Mixup/CutMix + RandAugment for SOTA-like recipes.
- Drop-path ≈0.2 mirrors Swin-S; keep smaller on shallow networks.
- Run `pytest python/fptm_ste/tests/test_tm_transformer.py` after modifying `tm_transformer.py`.
- Start with ViT + STE for fast sweeps; graduate to Swin + DeepTM when you’re ready to chase SOTA accuracy.

## 6. Model Characteristic Comparison

### 6.1 TM-backed Architectures (32×32 RGB, batch=1, FP32)

| Model | Architecture | TM backend | Params (M) | Forward activations (MB) | Approx FLOPs (GF) | Est. latency (ms)* | Est. throughput (img/s) | Est. energy (mJ)* | Output shape | Remarks |
|-------|--------------|------------|-----------:|-------------------------:|------------------:|-------------------:|------------------------:|------------------:|--------------|---------|
| TM-ViT (STE) | UnifiedTMTransformer (`--transformer-arch vit`) | STE | 1.20 | 0.83 | ≈0.05 | **8** | 125 | 2,800 | (B, 10) | Patch size 4, 4 encoder blocks. Fastest TM transformer. |
| TM-ViT (DeepTM) | UnifiedTMTransformer (ViT) | DeepTM | 3.27 | 1.21 | ≈0.07 | **11** | 91 | 3,850 | (B, 10) | Higher clause capacity → +~3 ms latency. |
| TM-Swin (DeepTM) | UnifiedTMTransformer (`--transformer-arch swin`) | DeepTM | 52.83 | 1.79 | ≈0.65 | **28** | 36 | 9,800 | (B, 10) | 2/2/6/2 Swin stages, window 4; balances accuracy vs compute. |
| ResNetTM-18 | CNN residual TM backbone | STE | 13.85 | 0.62 | ≈0.35 | **3.5** | 286 | 1,225 | (B, 10) | Conv stem + TM residual MLPs; solid baseline. |
| ResNetTM-50 | CNN residual TM backbone | STE | 52.34 | 2.44 | ≈0.95 | **8** | 125 | 2,800 | (B, 10) | Deeper bottleneck stack; capacity comparable to TM-Swin. |
| SwinTM baseline | Original SwinTM backbone | DeepTM | 175.11 | 4.49 | ≈2.8 | **60** | 17 | 21,000 | (B, 10) | Full Swin-T feature extractor with TM FFN; highest capacity & cost. |
| PyramidTM | Multi-scale pooling TM | STE | 1.37 | 2.13 | ≈0.40 | **15** | 67 | 5,250 | (B, 10) | Clause pooling hierarchy; dense projection footprint dominates. |
| Flat TM | Single-layer FuzzyPatternTM | STE | 3.15 | ~0 | ≈0.02 | **9** | 111 | 3,150 | (B, 10) | Diagnostic baseline over flattened pixels; accuracy-limited. |

### 6.2 Pure CNN / Transformer Baselines (224×224 RGB, batch=1, FP32)**

| Model variant | Params (M) | FLOPs (GF) | Est. latency (ms) | Est. throughput (img/s) | Est. energy (mJ) | Notes |
|---------------|-----------:|-----------:|------------------:|------------------------:|-----------------:|-------|
| ResNet-18 | 11.7 | 1.8 | 1.5 | 667 | 525 | Published V100 latency. |
| ResNet-34 | 21.8 | 3.6 | 2.9 | 345 | 1,015 |  | 
| ResNet-50 | 25.6 | 4.1 | 3.3 | 303 | 1,155 |  | 
| ResNet-101 | 44.5 | 7.9 | 5.9 | 169 | 2,065 |  | 
| ResNet-152 | 60.2 | 11.3 | 8.2 | 122 | 2,870 |  | 
| Swin-T (Tiny) | 28 | 4.5 | 9.1 | 110 | 3,185 | Official Swin benchmark. |
| Swin-S (Small) | 50 | 8.7 | 15 | 67 | 5,250 |  | 
| Swin-B (Base) | 88 | 15.4 | 24 | 42 | 8,400 |  | 
| Swin-L (Large) | 197 | 34.7 | 45 | 22 | 15,750 |  | 
| ViT-T/16 | 5.7 | 1.3 | 7.5 | 133 | 2,625 | | 
| ViT-S/16 | 22 | 4.6 | 13 | 77 | 4,550 | |
| ViT-B/16 | 86 | 17.6 | 25 | 40 | 8,750 | |
| ViT-L/16 | 304 | 61.6 | 50 | 20 | 17,500 | |

*Latencies/energy for TM rows are order-of-magnitude estimates extrapolated from clause counts, activation footprints, and V100-class measurements (assume 350 W TDP; energy ≈ latency × 350 mJ/ms). Published latencies for standard CNN/transformer models come from NVIDIA V100 single-image benchmarks; throughput = 1000 / latency (ms). Actual numbers depend on hardware, precision, batch size, and kernel implementations.

**ResNet/Swin/ViT baselines report ImageNet-scale statistics (224×224 inputs, fp32). When comparing directly with CIFAR-scale TM models, adjust expectations for resolution-dependent FLOP/latency scaling.

### 6.3 Benchmark Utility
- `python python/fptm_ste/benchmark_profiles.py` generates quick latency/throughput profiles over random batches.
- Flags of interest:
  - `--models` (`tm_vit_ste_base`, `tm_vit_deeptm_base`, `tm_swin_deeptm_tiny`, ...)
  - `--batch-sizes` (comma-separated, e.g. `1,32`)
  - `--device` (`auto`, `cpu`, `cuda`)
  - `--warmup` / `--iters` for tuning measurement passes
- Example (CPU):  
  `python python/fptm_ste/benchmark_profiles.py --models tm_vit_ste_base --batch-sizes 1,32 --device cpu --warmup 5 --iters 20 --output benchmarks_cpu.json`

## 7. Decision Log & Accuracy Snapshot
- **Baseline ViT + STE (3 epochs, MNIST, no new tricks)**  
  Command: `python python/fptm_ste/tests/run_mnist_equiv.py --dataset mnist --models transformer --transformer-arch vit --transformer-backend ste --transformer-patch 4 --epochs 3 --batch-size 64 --test-batch-size 128`  
  Result: test accuracy **94.84 %**, train accuracy 92.48 %, runtime ~103 s.
- **Enhanced ViT recipe (label smoothing, priors, auto clause, mixup/cutmix, EMA, layer decay)**  
  Command: `python python/fptm_ste/tests/run_mnist_equiv.py --dataset mnist --models transformer --transformer-arch vit --transformer-backend ste --transformer-patch 4 --epochs 3 --batch-size 64 --test-batch-size 128 --label-smoothing 0.1 --tm-prior-template symmetric --transformer-ff-gate geglu --transformer-ff-mix linear_depthwise --transformer-bitwise-mix --transformer-learnable-tau --transformer-tau-ema 0.9 --transformer-clause-attention --transformer-clause-routing --transformer-bypass --transformer-bypass-scale 0.5 --transformer-clause-drop 0.1 --transformer-literal-drop 0.05 --transformer-sparsity-weight 0.01 --transformer-clause-bias 0.05 --transformer-norm rmsnorm --transformer-aux-weight 0.05 --mixup-alpha 0.2 --cutmix-alpha 0.2 --lr 7e-4 --min-lr 1e-5 --warmup-epochs 1 --lr-layer-decay 0.85 --gradient-centralize --transformer-ema-decay 0.995 --transformer-auto-clause --transformer-auto-clause-batches 4 --report-epoch-test`  
  Result: test accuracy **95.50 %**, component accuracies substantially higher, runtime ~169 s.
- **Advanced recipe smoke test (dynamic aug schedules, KD, contrastive, cosine restarts)**  
  Command: `python python/fptm_ste/tests/run_mnist_equiv.py --dataset mnist --models transformer --transformer-arch vit --epochs 1 --batch-size 64 --test-batch-size 128 --transformer-patch 4 --transformer-d-model 32 --transformer-layers 1 --transformer-heads 2 --transformer-mlp-ratio 2.0 --transformer-backend ste --transformer-drop-path 0.05 --transformer-drop-path-schedule cosine --transformer-ff-gate geglu --transformer-ff-mix linear_depthwise --transformer-bitwise-mix --transformer-learnable-tau --transformer-tau-ema 0.9 --transformer-clause-attention --transformer-clause-routing --transformer-bypass --transformer-bypass-scale 0.5 --transformer-clause-drop 0.1 --transformer-literal-drop 0.05 --transformer-sparsity-weight 0.01 --transformer-clause-bias 0.05 --transformer-norm rmsnorm --transformer-aux-weight 0.05 --mixup-alpha 0.2 --cutmix-alpha 0.2 --mixup-schedule cosine --cutmix-schedule cosine --lr 7e-4 --min-lr 1e-5 --warmup-epochs 1 --lr-layer-decay 0.85 --gradient-centralize --lr-cycle-steps 50 --transformer-ema-decay 0.995 --transformer-self-distill-weight 0.5 --transformer-self-distill-temp 2.0 --transformer-contrastive-weight 0.1 --transformer-contrastive-temp 0.2 --transformer-auto-clause --transformer-auto-clause-batches 2 --label-smoothing 0.1 --tm-prior-template symmetric --report-epoch-test`  
  Result: test accuracy **50.8 %** after a single epoch (sanity check confirming new knobs; expect higher accuracy with longer schedules).
- Auto-clause tuning logs the adjusted clause budgets (e.g., `Auto-tuned clause counts: 256`) before training begins to facilitate reproducibility.

## 8. Interpretability Logging
- Pass `--transformer-log-dir runs/tm_vit` to `run_mnist_equiv.py` to stream per-epoch metrics into TensorBoard.
- Logged scalars include component/head accuracies, mixup/CutMix schedules, auxiliary losses, and tracked best accuracy.
- Launch TensorBoard via `tensorboard --logdir runs/tm_vit` and inspect groups under `Transformer-*`.
- Generate overlay figures on demand via `--visualize-overlays` and `--visualize-samples`. Overlays combine original images, attention heatmaps, clause activation maps, and top-k class predictions. Use `--device cpu` for guaranteed portability if CUDA/Triton kernels are unavailable.

## 9. External Distillation Workflow
- Enable pretrained teachers with `--distill-teacher-timm <model>` (e.g. `vit_base_patch16_224`, `swin_tiny_patch4_window7_224`).  
- Optionally load a fine-tuned checkpoint: `--distill-teacher-checkpoint /path/to/state_dict.pt`.  
- Control KD strength with `--distill-teacher-weight` and temperature with `--distill-teacher-temp`.  
- Combine with EMA self-distillation (`--transformer-self-distill-weight/temp`) and clause-aware head specialisation (`--transformer-clause-specialize`).  
- Diagnostic JSON now records `teacher_kd_loss` (per epoch) and final clause usage, making it easy to compare student-only vs teacher-guided runs.
