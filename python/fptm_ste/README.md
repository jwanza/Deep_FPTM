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


