# FPTM-STE (PyTorch)

Differentiable Fuzzy-Pattern Tsetlin Machines with Straight-Through Estimation (STE), deep TM stacks, multi-scale Swin backbones, attention over clauses/logits, and a simple TM-Transformer block.

## Install

```bash
pip install -r python/fptm_ste/requirements.txt
```

## Quickstart

```python
import torch
from fptm_ste import FuzzyPatternTM_STE

x = torch.rand(8, 128)  # [batch, features] in [0,1]
tm = FuzzyPatternTM_STE(n_features=128, n_clauses=200, n_classes=10)
logits, clauses = tm(x, use_ste=True)
```

### Swin + Multi-Scale TM

```python
import torch
from fptm_ste import SwinFeatureExtractor, MultiScaleTMEnsemble

backbone = SwinFeatureExtractor(pretrained=True, freeze=True)
feats = backbone(torch.randn(2, 3, 224, 224))
ens = MultiScaleTMEnsemble([f.shape[1] for f in feats], n_classes=1000, n_clauses_per_scale=[500,300,200,100])
final_logits, scale_logits, clause_outputs = ens(feats, use_ste=True)
```

### Export literals to JSON (for Julia)

```python
from fptm_ste import export_compiled_to_json
bundle = tm.discretize(threshold=0.5)
export_compiled_to_json(bundle, class_labels=["0","1"], path="/tmp/tm.json", clauses_num=tm.n_clauses, L=16, LF=4)
```

### Deep TM Network

```python
from fptm_ste import DeepTMNetwork
model = DeepTMNetwork(input_dim=128, hidden_dims=[64, 64], n_classes=10, n_clauses=100)
```

## MNIST Equivalence Runner

We ship a convenience script (`python/fptm_ste/tests/run_mnist_equiv.py`) that runs the latest TM variants on MNIST (and can be adapted to CIFAR) with:

- straight-through TM (conv projector + clause annealing)
- deeper TM stacks with noise and layer-wise τ scheduling
- Swin-T hybrid ensemble (spatial + scale attention, anti-collapse gating)
- TM transformer with token downsampling
- gradient accumulation, EMA smoothing, and warm-up + cosine LR schedules

```bash
python python/fptm_ste/tests/run_mnist_equiv.py
```

Environment variables (set before invoking the script):

| Variable | Meaning | Default |
|----------|---------|---------|
| `TM_MNIST_EPOCHS` | Epochs per model | `50` |
| `TM_MNIST_ACCUM` | Gradient-accumulation steps | `4` |
| `TM_MNIST_EMA` | EMA decay | `0.999` |
| `TM_MNIST_WARMUP` | Warm-up epochs before cosine anneal | `5` |
| `TM_MNIST_VARIANTS` | Comma-separated subset (`tm`, `deep_tm`, `hybrid`, `transformer`) | `tm,deep_tm,hybrid,transformer` |
| `TM_MNIST_REPORT_EPOCH` | Log per-epoch train accuracy (`0`/`1`) | `0` |

Examples:

```bash
# Just the deep TM variant with detailed logging
TM_MNIST_VARIANTS=deep_tm TM_MNIST_REPORT_EPOCH=1 python python/fptm_ste/tests/run_mnist_equiv.py

# Swin hybrid only, longer warm-up, more accumulation steps
TM_MNIST_VARIANTS=hybrid TM_MNIST_WARMUP=8 TM_MNIST_ACCUM=6 python python/fptm_ste/tests/run_mnist_equiv.py
```

Each run produces `/tmp/mnist_equiv_results.json` plus model-specific JSON exports (for the TM flavours that can be discretised). These exports load directly into the Julia `JsonBridge` pipeline.

Julia wrappers (`examples/MNIST/mnist_deep_tm.jl`, `mnist_hybrid_swin.jl`, `mnist_tm_transformer.jl`) forward to this script so you can invoke the variants from Julia with the same environment variables.


