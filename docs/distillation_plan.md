# Advanced Distillation Blueprint

## Overview

This document captures the recommended workflow for transferring pretrained knowledge from canonical backbones into the TM-enhanced transformer variants. The goals are:

- Initialise teacher models from strong pretrained checkpoints (no training from scratch).
- Support one-to-one mappings: `MLP/FFN → TM`, `ViT → ViT-TM`, `Swin → Swin-TM`.
- Combine multiple distillation signals (logit KD, feature/context alignment, clause-aware routing).
- Define validation checkpoints and proofs to ensure the pipeline delivers measurable gains.

## Teacher Selection

| Student Variant | Recommended Teacher | Acquisition | Notes |
|-----------------|---------------------|-------------|-------|
| `UnifiedTMTransformer (vit, ste/deeptm)` | `timm` ViT models (`vit_tiny_patch16_224`, `vit_base_patch16_224`, …) | `--distill-teacher-timm` | Uses ImageNet-pretrained weights. Classifier head is reset to match `--num-classes`. |
| `UnifiedTMTransformer (swin, ste/deeptm)` | `timm` Swin models (`swin_tiny_patch4_window7_224`, …) | `--distill-teacher-timm` | Supports V1/V2 variants. Matches input channels automatically and honours the student’s image size. |
| CNN/Hybrid + TM heads | `torchvision` ResNet / custom MLP mixer checkpoints | `--distill-teacher-checkpoint` | Provide an explicit state_dict for the baseline you want to mimic. |

> **Tip:** When both `--distill-teacher-timm` and `--distill-teacher-checkpoint` are supplied, the checkpoint is loaded *after* the pretrained timm weights, allowing fine-tuned teachers to override the default parameters.

Pretrained heads are frozen internally; gradients never flow into the teacher model.

## Loss Composition

The training loop now exposes the following knobs:

| Flag | Description |
|------|-------------|
| `--distill-teacher-weight` | Multiplier for logit KD against the external teacher. |
| `--distill-teacher-temp` | Temperature used for KL divergence. Compensates for different entropy profiles. |
| `--transformer-self-distill-weight` / `--transformer-self_distill_temp` | EMA teacher self-distillation (already available). |
| `--transformer-clause-specialize` / `--transformer-clause-specialize-strength` | Reweights attention heads using clause diagnostics, leveraging clause signals during distillation. |
| `--transformer-save-path` | Persist the distilled transformer (state dict + config + metrics) for standalone evaluation. |

The effective student loss becomes:

```
L_total = CE(student, labels)
        + λ_teacher * KD(student, teacher_pretrained)
        + λ_ema * KD(student, EMA(student))
        + λ_aux * AuxHeads(student)
        + λ_contrastive * SupCon(student_features)
        + λ_clause * ClauseRegularisation
```

The KD term uses `F.kl_div(log_softmax(s/T), softmax(t/T)) * T²` as standard.

## Validation & Proof Checklist

1. **Unit Regression**
   - `pytest python/fptm_ste/tests/test_tm_transformer.py` (verifies clause metrics & specialisation).
   - `pytest python/fptm_ste/tests/test_visualization.py` (overlay utilities; skipped automatically if matplotlib is unavailable).

2. **Numerical Sanity**
   - Enable `--report-epoch-acc` and monitor KD loss (`teacher_kd_loss`) in logs/TensorBoard.
   - Ensure the teacher logits agree dimensionally with the student (`num_classes` match).

3. **Benchmark Proofs**
   - CIFAR-10: run baseline and TM student with teacher KD to compare accuracy/FLOPs.
   - ImageNet-1k: use `python/fptm_ste/imagenet_sweeps.py` with `--distill-teacher-timm` to confirm large-scale viability.
   - Track clause usage before/after specialisation to demonstrate head re-weighting effects.

4. **Reproducibility Artifacts**
   - Store command invocations and resulting `results.json` outputs.
   - When exporting overlays (`--visualize-overlays`), archive interpretability bundles alongside benchmark results to demonstrate qualitative improvements.

## Example Commands

Before launching any distillation run:
```bash
source SOURCE.THIS           # activates the shared venv and exports PYTHONPATH
export PYTHONPATH=$PWD/python:$PYTHONPATH
```

### ViT-TM distilled from pretrained ViT-B/16

```bash
PYTHONPATH=python python3 python/fptm_ste/tests/run_mnist_equiv.py \
  --dataset cifar10 \
  --models transformer \
  --transformer-arch vit \
  --transformer-backend ste \
  --transformer-patch 4 \
  --epochs 100 \
  --batch-size 256 \
  --transformer-self-distill-weight 0.5 \
  --distill-teacher-timm vit_base_patch16_224 \
  --distill-teacher-weight 0.7 \
  --distill-teacher-temp 2.0 \
  --transformer-clause-specialize \
  --transformer-clause-specialize-strength 0.6 \
  --visualize-overlays /tmp/vit_tm_overlays \
  --visualize-samples 8 \
  --transformer-save-path checkpoints/cifar10_vit_tm_distilled.pth
```

### Swin-TM distilled from pretrained Swin-T

```bash
PYTHONPATH=python python3 python/fptm_ste/tests/run_mnist_equiv.py \
  --dataset cifar10 \
  --models transformer \
  --transformer-arch swin \
  --transformer-backend deeptm \
  --transformer-patch 4 \
  --transformer-depths 2,2,6,2 \
  --transformer-stage-heads 3,6,12,24 \
  --transformer-embed-dims 96,192,384,768 \
  --distill-teacher-timm swin_tiny_patch4_window7_224 \
  --distill-teacher-weight 0.5 \
  --distill-teacher-temp 1.5 \
  --transformer-clause-specialize \
  --transformer-clause-specialize-strength 0.5 \
  --transformer-save-path checkpoints/cifar10_swin_tm_distilled.pth
```

### Quick MNIST sanity checks (used for pipeline validation)

These 1-epoch CPU runs confirm end-to-end wiring, teacher loading, and checkpoint export.

**ViT teacher → TM-ViT student**
```bash
python3 python/fptm_ste/tests/run_mnist_equiv.py \
  --device cpu \
  --dataset mnist \
  --models transformer \
  --transformer-arch vit \
  --transformer-backend ste \
  --transformer-patch 4 \
  --transformer-embed-dims 64 \
  --transformer-layers 1 \
  --transformer-heads 4 \
  --transformer-mlp-ratio 3.0 \
  --epochs 1 \
  --batch-size 512 \
  --test-batch-size 512 \
  --tm-target-size 64 64 \
  --distill-teacher-timm vit_small_patch16_224 \
  --distill-teacher-weight 0.5 \
  --distill-teacher-temp 2.0 \
  --transformer-save-path /tmp/mnist_vit_tm_distilled.pth
```
Result snapshot: `test_acc ≈ 0.41`, `teacher_kd_loss ≈ 0.36`, checkpoint saved to `/tmp/mnist_vit_tm_distilled.pth`.

**Swin teacher → TM-Swin student**
```bash
python3 python/fptm_ste/tests/run_mnist_equiv.py \
  --device cpu \
  --dataset mnist \
  --models transformer \
  --transformer-arch swin \
  --transformer-backend ste \
  --transformer-patch 2 \
  --transformer-depths 1,1 \
  --transformer-stage-heads 2,4 \
  --transformer-embed-dims 48,96 \
  --transformer-mlp-ratio 3.0,3.0 \
  --transformer-window 4 \
  --epochs 1 \
  --batch-size 512 \
  --test-batch-size 512 \
  --tm-target-size 64 64 \
  --distill-teacher-timm swin_tiny_patch4_window7_224 \
  --distill-teacher-weight 0.5 \
  --distill-teacher-temp 2.0 \
  --transformer-save-path /tmp/mnist_swin_tm_distilled.pth
```
Result snapshot: `test_acc ≈ 0.36`, `teacher_kd_loss ≈ 0.12`, checkpoint saved to `/tmp/mnist_swin_tm_distilled.pth`.

Checkpoint contents can be reloaded with:
```python
import torch
from fptm_ste.tm_transformer import UnifiedTMTransformer

ckpt = torch.load("/tmp/mnist_vit_tm_distilled.pth", map_location="cpu")
student = UnifiedTMTransformer(**ckpt["model_kwargs"])
student.load_state_dict(ckpt["model_state"])
student.eval()
```

## Future Extensions

- Feature matching (e.g. aligning token embeddings or clause activations) can be added by extracting intermediate tensors and computing L₂ penalties.
- Clause-level KD: convert teacher activations into pseudo clause targets using attention maps as guidance.
- Teacher ensembles: average logits from multiple pretrained sources to create a stronger supervisory signal.

This blueprint ensures every distillation experiment starts from a pretrained teacher, exploits TM-specific diagnostics, and includes a validation matrix to prove impact.


