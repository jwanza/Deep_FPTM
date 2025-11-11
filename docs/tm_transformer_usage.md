# TM Transformer Usage & Tuning Guide

This guide shows how to launch the upgraded TM transformers, outlines every configurable parameter exposed by `run_mnist_equiv.py`, and lists practical tricks for pushing accuracy toward SOTA on CIFAR/ImageNet-style workloads.

## 1. Environment Setup

- Activate the virtual environment: `source venv/bin/activate`
- Ensure the package path is visible: `export PYTHONPATH=$PWD/python:$PYTHONPATH`

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
| `--transformer-dropout` | attention / TM dropout |
| `--transformer-ema-decay` | EMA decay (0 disables) |
| `--transformer-grad-checkpoint` | gradient checkpointing toggle |
| `--randaugment`, `--mixup-alpha`, `--cutmix-alpha` | augmentation toggles |

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
- **Epochs & batch size**: extend training (≥100 epochs) when augmentations and DeepTM backend are active.

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
