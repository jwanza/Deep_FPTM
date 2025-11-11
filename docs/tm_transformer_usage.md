# TM Transformer Usage Guide

This document summarizes how to launch the upgraded TM transformers, configure ViT/Swin backbones, and enable the optional training tricks (RandAugment, Mixup/CutMix, EMA, checkpointing).

## 1. Environment Setup

- Activate the project virtual environment: `source venv/bin/activate`
- Ensure the Python package path is visible when running scripts: `export PYTHONPATH=$PWD/python:$PYTHONPATH`

## 2. Core CLI Parameters

| Flag | Description |
|------|-------------|
| `--models transformer` | selects the TM transformer variant |
| `--transformer-arch {vit,swin}` | ViT-style global attention or Swin-style hierarchical windows |
| `--transformer-backend {ste,deeptm}` | chooses the TM feed-forward backend |
| `--transformer-patch <P>` | patch size (e.g. 4 or 8) for image tokenization |
| `--transformer-depths` | comma-separated stage depths (Swin) or omitted for ViT |
| `--transformer-stage-heads` | per-stage attention heads (Swin) |
| `--transformer-embed-dims` | per-stage channel widths (Swin) |
| `--transformer-drop-path` | maximum drop-path probability |
| `--transformer-dropout` | attention/FF dropout |
| `--transformer-ema-decay` | EMA decay (0 disables) |
| `--transformer-grad-checkpoint` | enables gradient checkpointing |
| `--randaugment`, `--mixup-alpha`, `--cutmix-alpha` | apply modern data augmentations |

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
source venv/bin/activate
export PYTHONPATH=$PWD/python:$PYTHONPATH
python python/fptm_ste/tests/run_mnist_equiv.py \
  --dataset cifar10 --models transformer \
  --transformer-arch swin --transformer-backend deeptm \
  --transformer-patch 4 --transformer-depths 2,2,6,2 \
  --transformer-stage-heads 3,6,12,24 \
  --transformer-embed-dims 96,192,384,768 \
  --transformer-window 7 --transformer-drop-path 0.2 \
  --transformer-grad-checkpoint --epochs 30 --batch-size 64
```

### 3.3 ViT + STE without augmentations (MNIST baseline)
```bash
source venv/bin/activate
export PYTHONPATH=$PWD/python:$PYTHONPATH
python python/fptm_ste/tests/run_mnist_equiv.py \
  --dataset mnist --models transformer \
  --transformer-arch vit --transformer-backend ste \
  --transformer-patch 2 --epochs 10 --batch-size 256
```

## 4. Tips & Notes

- **Patch size**: reduce tokens by increasing patch size (e.g., 8×8) to keep memory manageable.
- **DEEPTM backend**: higher accuracy but higher compute; start with STE to benchmark.
- **Drop-path schedules**: `--transformer-drop-path 0.2` mirrors Swin-S small networks; set to 0 for baselines.
- **EMA**: values in the range `0.995-0.999` work well; disable with `--transformer-ema-decay 0`.
- **Mixup/CutMix**: enable on larger datasets (CIFAR/ImageNet) for regularization; skip on MNIST.
- **Checkpointing**: `--transformer-grad-checkpoint` saves memory at slight compute cost, recommended on GPUs with <24 GB.
- **RandAugment**: defaults to two operations, magnitude 9; adjust via `--randaugment-n`/`--randaugment-m`.
- **Boolean TM features**: the `--tm-feature-mode` pipeline now has presets in `fptm_ste.datasets`; set `--tm-feature-config cifar10` to reuse cached boolean features if required.
