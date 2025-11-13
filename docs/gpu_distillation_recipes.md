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

