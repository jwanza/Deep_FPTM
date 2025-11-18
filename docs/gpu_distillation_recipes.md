# GPU Distillation Recipes (MNIST sanity checks)

This note captures the exact steps, commands, and quick verification scripts used to distil TM-backed transformers on GPU from pretrained TIMM teachers. Both experiments resize MNIST to 64×64 to remain compatible with the teachers’ patch embeddings; swap dataset/resize settings as needed for CIFAR-10 or ImageNet.

## 1. Prerequisites

- Activate the shared environment and expose the package path:
  ```bash
  cd /nvme0n1-disk/shared/joel/FuzzyPatternTM_4GTM
  source SOURCE.THIS
  export PYTHONPATH=$PWD/python:$PYTHONPATH
  ```
- Ensure the MNIST boolean caches live under `/tmp/` (created automatically the first run).
- A CUDA-capable GPU (commands fall back to CPU if `--device cuda` is omitted).

## 2. ViT teacher → TM-ViT student (10 epochs, GPU)

### 2.1 Training command

```bash
python3 python/fptm_ste/tests/run_mnist_equiv.py \
  --device cuda \
  --dataset mnist \
  --models transformer \
  --transformer-arch vit \
  --transformer-backend ste \
  --transformer-patch 4 \
  --transformer-embed-dims 64 \
  --transformer-layers 1 \
  --transformer-heads 4 \
  --transformer-mlp-ratio 3.0 \
  --epochs 10 \
  --batch-size 512 \
  --test-batch-size 512 \
  --tm-target-size 64 64 \
  --test-holdout-fraction 0.5 \
  --distill-teacher-timm vit_small_patch16_224 \
  --distill-teacher-weight 0.5 \
  --distill-teacher-temp 2.0 \
  --distill-teacher-trainable \
  --transformer-save-path /tmp/mnist_vit_tm_distilled_gpu.pth
```

### 2.2 Observed results
- Validation accuracy (50 % split): **0.9384**, holdout accuracy: **0.9364**.
- Teacher accuracy on the same splits: ~0.987 (logged as `teacher_acc`).
- Training wall time: ~91 s (NVIDIA GPU, AMP enabled).
- Artifact: `/tmp/mnist_vit_tm_distilled_gpu_holdout.pth` containing `model_state`, `model_kwargs`, training metrics, and metadata (`image_size=(64, 64)`).

### 2.3 Inference sanity check

```bash
python3 - <<'PY'
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from fptm_ste.tm_transformer import UnifiedTMTransformer

ckpt = torch.load("/tmp/mnist_vit_tm_distilled_gpu_holdout.pth", map_location="cpu")
model = UnifiedTMTransformer(**ckpt["model_kwargs"])
model.load_state_dict(ckpt["model_state"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    transforms.ToTensor(),
])
test_ds = datasets.MNIST("/tmp/mnist", train=False, download=True, transform=transform)
loader = DataLoader(test_ds, batch_size=256, shuffle=False)

correct = 0
total = 0
with torch.no_grad():
    for x, y in loader:
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

print(f"Inference accuracy: {correct / total:.4f} ({correct}/{total})")
print("Checkpoint meta:", ckpt["meta"])
PY
```

Expected output:
```
Inference accuracy: 0.9374 (9374/10000)
Checkpoint meta: {'architecture': 'vit', 'backend': 'ste', 'timestamp': ..., 'image_size': (64, 64), 'num_classes': 10}
```

## 3. Swin teacher → TM-Swin student (10 epochs, GPU)

### 3.1 Training command

```bash
python3 python/fptm_ste/tests/run_mnist_equiv.py \
  --device cuda \
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
  --epochs 10 \
  --batch-size 512 \
  --test-batch-size 512 \
  --tm-target-size 64 64 \
  --test-holdout-fraction 0.5 \
  --distill-teacher-timm swin_tiny_patch4_window7_224 \
  --distill-teacher-weight 0.5 \
  --distill-teacher-temp 2.0 \
  --distill-teacher-trainable \
  --transformer-save-path /tmp/mnist_swin_tm_distilled_gpu.pth
```

### 3.2 Observed results
- Validation accuracy (50 % split): **0.7122**, holdout accuracy: **0.7220**.
- Teacher accuracy on the same splits: ~0.994.
- Training wall time: ~146 s (Swin window attention and clause-aware routing are heavier).
- Artifact: `/tmp/mnist_swin_tm_distilled_gpu_holdout.pth`.

### 3.3 Inference sanity check

```bash
python3 - <<'PY'
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from fptm_ste.tm_transformer import UnifiedTMTransformer

ckpt = torch.load("/tmp/mnist_swin_tm_distilled_gpu_holdout.pth", map_location="cpu")
model = UnifiedTMTransformer(**ckpt["model_kwargs"])
model.load_state_dict(ckpt["model_state"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    transforms.ToTensor(),
])
test_ds = datasets.MNIST("/tmp/mnist", train=False, download=True, transform=transform)
loader = DataLoader(test_ds, batch_size=256, shuffle=False)

correct = 0
total = 0
with torch.no_grad():
    for x, y in loader:
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

print(f"Inference accuracy: {correct / total:.4f} ({correct}/{total})")
print("Checkpoint meta:", ckpt["meta"])
PY
```

Expected output:
```
Inference accuracy: 0.7171 (7171/10000)
Checkpoint meta: {'architecture': 'swin', 'backend': 'ste', 'timestamp': ..., 'image_size': (64, 64), 'num_classes': 10}
```

## 4. Notes for scaling

- **Dataset swaps:** replace `--dataset mnist` with `cifar10` (and drop `--tm-target-size` if you want to retain native 32×32 crops). For ImageNet, point `--dataset-root` to your data folder and adjust `--transformer-depths/heads/embeds`.
- **Augmentation:** enable `--randaugment`, `--mixup-alpha`, and `--cutmix-alpha` to mirror the heavier recipes captured in `docs/tm_transformer_usage.md`.
- **Backends:** switching `--transformer-backend` to `deeptm` raises accuracy at additional cost; the checkpoint structure remains the same.
- **Checkpoint reuse:** each artifact stores `model_kwargs` so the student reinstantiates without manual bookkeeping. Pair checkpoints with overlay generation (`--visualize-overlays ...`) if you want clause/attention diagnostics post-training.

These commands were executed on November 13, 2025, and verified by running the inference snippets above.


## 5. Advanced distillation controls

- **Layer/token hints & relational KD**  
  Activate FitNet-style guidance with `--transformer-hint-weight` (temperature via `--transformer-hint-temp`).  
  Preserve token geometry using `--transformer-relational-weight`, which matches pairwise similarities between teacher and student diagnostics.

- **Clause & attention alignment**  
  Map teacher features into clause space using `--transformer-clause-align-weight`/`--transformer-clause-align-temp`.  
  Encourage clause gates to follow teacher saliency with `--transformer-attn-guidance-weight`/`--transformer-attn-guidance-temp`.

- **Two-stage KD scheduling**  
  `--transformer-kd-stage-epochs` triggers the late-stage schedule. Adjust component strengths with `--transformer-kd-stage2-*-scale` and temperatures via `--transformer-kd-stage2-*-temp-scale`.  
  Harden clauses gradually by setting `--transformer-kd-stage2-tau`.

- **Clause warm start**  
  Pre-adapt literals with the teacher for a few batches using `--transformer-clause-init-batches`, `--transformer-clause-init-lr`, and `--transformer-clause-init-temp`.

- **Teacher adapters & mixup safety**  
  For frozen teachers mixed with aggressive augmentations, enable `--distill-teacher-adapter` (with optional `--distill-teacher-adapter-lr`) so logits are calibrated after mixup/cutmix.

- **Contrastive + KD hybrid**  
  Combine KD with supervised contrastive training using `--transformer-contrastive-weight`, `--transformer-contrastive-temp`, and `--transformer-contrastive-use-teacher`.

- **Offline teacher augmentation**  
  Load pseudo-labelled batches with `--teacher-aug-path` and control how often they are sampled using `--teacher-aug-batches`.

- **Teacher baseline fine-tuning**  
  Run `--teacher-baseline` (optionally overriding the model with `--teacher-baseline-model`) to fine-tune the timm teacher standalone. The script auto-selects 224×224 ImageNet-style transforms for ViT/Swin, logs metrics to TensorBoard if `--teacher-baseline-log-dir` is set, and saves checkpoints via `--teacher-baseline-save-path`.

Quick start (CIFAR-10 baseline with a Swin-B teacher):

```bash
python3 python/fptm_ste/tests/run_mnist_equiv.py \
  --dataset cifar10 \
  --teacher-baseline \
  --teacher-baseline-model swin_base_patch4_window7_224 \
  --teacher-baseline-epochs 30 \
  --teacher-baseline-lr 3e-5 \
  --teacher-baseline-save-path /tmp/cifar10_swin_teacher.pt
```

Minimal example (ViT hints + clause alignment + stage schedule):


```bash
python3 python/fptm_ste/tests/run_mnist_equiv.py \
  --models transformer \
  --transformer-arch vit \
  --distill-teacher-timm vit_base_patch16_224 \
  --transformer-hint-weight 0.4 \
  --transformer-clause-align-weight 0.2 \
  --transformer-kd-stage-epochs 5 \
  --transformer-kd-stage2-teacher-scale 0.5 \
  --transformer-kd-stage2-tau 0.35
```
