# Advanced FuzzyPatternTM Modules

This document describes the advanced modules implemented in FuzzyPatternTM to address
the booleanization bottleneck and improve learning capabilities.

## Table of Contents

1. [Booleanization Solutions](#booleanization-solutions)
2. [Ultimate Hybrid Architecture](#ultimate-hybrid-architecture)
3. [Temporal Processing](#temporal-processing)
4. [Continual Learning](#continual-learning)
5. [Sparse Routing & MoE](#sparse-routing--moe)
6. [Hyperbolic Geometry](#hyperbolic-geometry)
7. [Advanced Optimizers](#advanced-optimizers)
8. [Data Augmentation](#data-augmentation)

---

## Booleanization Solutions

The fundamental challenge in Tsetlin Machines is the conversion of continuous features
to binary, which loses information. We implement six complementary solutions:

### 1. Continuous Residual Clause Machine (CRCM)

**File:** `fptm_ste/booleanization/continuous_residual.py`

**Key Idea:** Maintain a parallel continuous stream alongside the binary TM stream.

```python
from fptm_ste.booleanization import ContinuousResidualClauseMachine

model = ContinuousResidualClauseMachine(
    n_features=784,
    n_clauses=64,
    n_classes=10,
    hidden_dim=128,
    fusion_type="sigmoid",  # or "softmax", "attention"
    reconstruction_weight=0.1,
)

# Forward pass returns (logits, clause_outputs)
logits, clauses = model(x)

# Get detailed outputs including reconstruction
details = model(x, return_details=True)
recon_loss = model.information_preservation_loss(x, details["reconstruction"])
```

### 2. Probabilistic Literal Clause Machine

**File:** `fptm_ste/booleanization/probabilistic.py`

**Key Idea:** Represent literals as probability distributions, preserving uncertainty.

```python
from fptm_ste.booleanization import ProbabilisticLiteralClauseMachine

model = ProbabilisticLiteralClauseMachine(
    n_features=784,
    n_clauses=64,
    n_classes=10,
)

# Get predictions with uncertainty
logits, clauses, uncertainty = model(x, return_uncertainty=True)
```

### 3. Hyperdimensional Clause Machine

**File:** `fptm_ste/booleanization/hyperdimensional.py`

**Key Idea:** Encode continuous features into high-dimensional binary vectors that preserve similarity.

```python
from fptm_ste.booleanization import HyperdimensionalClauseMachine

model = HyperdimensionalClauseMachine(
    n_features=784,
    n_clauses=64,
    n_classes=10,
    hd_dim=10000,  # HD vector dimension
    n_levels=16,    # Quantization levels
)
```

### 4. Information Bottleneck Binarizer

**File:** `fptm_ste/booleanization/information_bottleneck.py`

**Key Idea:** Learn optimal binary representation by maximizing relevant information while compressing irrelevant details.

```python
from fptm_ste.booleanization import InformationPreservingClauseMachine

model = InformationPreservingClauseMachine(
    n_features=784,
    n_clauses=64,
    n_classes=10,
    latent_dim=16,
)

# IB loss components are accessible
ib_loss = model.ib_kl_loss + model.ib_relevance_loss
```

### 5. Hierarchical Multi-Resolution TM

**File:** `fptm_ste/booleanization/hierarchical.py`

**Key Idea:** Process features at multiple resolution levels with different binarization granularities.

```python
from fptm_ste.booleanization import HierarchicalMultiResolutionTM

model = HierarchicalMultiResolutionTM(
    n_features=784,  # Must be square (28x28)
    n_clauses=64,
    n_classes=10,
    n_levels=3,
    resolution_factors=[1.0, 0.5, 0.25],
)
```

### 6. Neural Symbolic Transformer

**File:** `fptm_ste/booleanization/attention_adaptive.py`

**Key Idea:** Use attention to dynamically learn optimal binarization per sample.

```python
from fptm_ste.booleanization import NeuralSymbolicTransformer

model = NeuralSymbolicTransformer(
    n_features=784,
    n_clauses=64,
    n_classes=10,
    embed_dim=64,
    n_heads=4,
    n_layers=2,
)
```

---

## Ultimate Hybrid Architecture

**File:** `fptm_ste/ultimate_hybrid.py`

Combines multiple streams into a single powerful architecture:

```python
from fptm_ste.ultimate_hybrid import UltimateHybridTM

model = UltimateHybridTM(
    n_features=784,
    n_clauses=64,
    n_classes=10,
    
    # Enable/disable streams
    use_binary_stream=True,
    use_continuous_stream=True,
    use_hd_stream=True,
    use_ib_stream=False,
    use_probabilistic_stream=False,
    
    # Fusion type
    fusion_type="adaptive",  # or "sum", "concat"
    use_clause_attention=True,
)

# Training with auxiliary losses
logits, clauses = model(x)
aux_losses = model.get_auxiliary_losses()
total_loss = F.cross_entropy(logits, y) + sum(aux_losses.values())
```

### Pre-configured Architectures

```python
from fptm_ste.ultimate_hybrid import (
    create_light_hybrid,       # Binary + Continuous only
    create_full_hybrid,        # All streams
    create_fast_inference_hybrid,  # Optimized for speed
    create_interpretable_hybrid,   # With uncertainty
)
```

---

## Temporal Processing

**File:** `fptm_ste/temporal.py`

Process sequences with stateful clauses:

```python
from fptm_ste.temporal import TemporalClauseMachine

model = TemporalClauseMachine(
    n_features=16,
    n_clauses=32,
    n_classes=5,
    state_dim=64,
    state_update="gru",  # or "lstm"
    use_temporal_attention=True,
    pooling="attention",  # or "last", "mean", "max"
)

# Input: [batch, seq_len, n_features]
x_seq = torch.randn(4, 10, 16)
logits, hidden = model(x_seq)

# Bidirectional variant
from fptm_ste.temporal import BidirectionalTemporalClauseMachine

bidir_model = BidirectionalTemporalClauseMachine(
    n_features=16,
    n_clauses=32,
    n_classes=5,
)
```

---

## Continual Learning

**File:** `fptm_ste/continual.py`

Prevent catastrophic forgetting:

### Elastic Weight Consolidation (EWC)

```python
from fptm_ste.continual import EWCWrapper

model = MyTMModel(...)
ewc_model = EWCWrapper(model, lambda_=1000.0)

# After training on task 1
ewc_model.compute_fisher(train_loader)
ewc_model.consolidate()

# Training on task 2
loss = F.cross_entropy(logits, y) + ewc_model.ewc_penalty()
```

### Synaptic Intelligence (SI)

```python
from fptm_ste.continual import SynapticIntelligence

si = SynapticIntelligence(model, c=1.0)

# During training
loss = F.cross_entropy(logits, y) + si.compute_penalty()
si.update_omega()

# After task
si.consolidate()
```

### Experience Replay

```python
from fptm_ste.continual import ExperienceReplayBuffer

buffer = ExperienceReplayBuffer(capacity=1000)

# Add samples
buffer.add(x, y)

# Sample for replay
replay_x, replay_y = buffer.sample(batch_size=32)
```

---

## Sparse Routing & MoE

**File:** `fptm_ste/sparse_routing.py`

Mixture of Experts for dynamic clause activation:

```python
from fptm_ste.sparse_routing import SparseMoEClauseMachine

model = SparseMoEClauseMachine(
    n_features=784,
    n_clauses_per_expert=16,
    n_classes=10,
    n_experts=8,
    top_k=2,  # Activate 2 experts per sample
    use_l0_pruning=True,
)

logits, clauses = model(x)
aux_loss = model.router_aux_loss + model.l0_reg_loss
```

### L0 Clause Pruning

```python
from fptm_ste.sparse_routing import L0ClauseMask

mask = L0ClauseMask(n_clauses=64)
gated_clauses, l0_loss = mask(clause_outputs)
```

---

## Hyperbolic Geometry

**File:** `fptm_ste/hyperbolic.py`

Non-Euclidean voting for hierarchical class relationships:

```python
from fptm_ste.hyperbolic import HyperbolicClauseVoting

voting = HyperbolicClauseVoting(
    n_clauses=64,
    n_classes=10,
    embed_dim=16,
    c=1.0,  # Curvature
)

logits = voting(clause_outputs)
```

---

## Advanced Optimizers

### Sharpness-Aware Minimization (SAM)

**File:** `fptm_ste/sam_optimizer.py`

Find flat minima for better generalization:

```python
from fptm_ste.sam_optimizer import SAM

optimizer = SAM(model.parameters(), torch.optim.Adam, lr=0.001, rho=0.05)

# Training step
loss.backward()
optimizer.first_step(zero_grad=True)

loss.backward()  # Recompute loss at perturbed point
optimizer.second_step(zero_grad=True)
```

---

## Data Augmentation

**File:** `fptm_ste/augmentation.py`

Advanced mixing augmentations:

```python
from fptm_ste.augmentation import mixup_data, cutmix_data, mixup_criterion

# Mixup
mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.4)
loss = mixup_criterion(F.cross_entropy, logits, y_a, y_b, lam)

# CutMix
mixed_x, y_a, y_b, lam = cutmix_data(x, y, alpha=0.4)

# Pipeline
from fptm_ste.augmentation import AugmentationPipeline

aug = AugmentationPipeline(
    use_mixup=True,
    use_cutmix=True,
    mixup_alpha=0.4,
)
aug_x, y_a, y_b, lam = aug(x, y)
```

---

## CLI Usage

Run experiments with the advanced CLI:

```bash
# Train CRCM on MNIST
python run_advanced_tm.py --model crcm --dataset mnist --epochs 10

# Train Ultimate Hybrid on CIFAR-10 with SAM
python run_advanced_tm.py --model ultimate_hybrid --dataset cifar10 \
    --optimizer sam --epochs 30

# Train with continual learning (EWC)
python run_advanced_tm.py --model stcm --dataset mnist \
    --continual ewc --epochs 20

# Train with augmentation
python run_advanced_tm.py --model crcm --dataset cifar10 \
    --augmentation mixup --mixup-alpha 0.4
```

---

## Quick Start Examples

### Example 1: Basic Booleanization Solution

```python
import torch
from fptm_ste.booleanization import ContinuousResidualClauseMachine

# Create model
model = ContinuousResidualClauseMachine(
    n_features=784,
    n_clauses=64,
    n_classes=10,
)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for x, y in train_loader:
    logits, _ = model(x)
    loss = F.cross_entropy(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Example 2: Full Pipeline

```python
import torch
from fptm_ste.ultimate_hybrid import create_light_hybrid
from fptm_ste.sam_optimizer import SAM
from fptm_ste.augmentation import AugmentationPipeline

# Model
model = create_light_hybrid(n_features=784, n_clauses=64, n_classes=10)

# SAM Optimizer
optimizer = SAM(model.parameters(), torch.optim.Adam, lr=0.001)

# Augmentation
augment = AugmentationPipeline(use_mixup=True)

# Training loop
for x, y in train_loader:
    aug_x, y_a, y_b, lam = augment(x, y)
    
    # First SAM step
    logits, _ = model(aug_x)
    loss = lam * F.cross_entropy(logits, y_a) + (1-lam) * F.cross_entropy(logits, y_b)
    loss.backward()
    optimizer.first_step(zero_grad=True)
    
    # Second SAM step
    logits, _ = model(aug_x)
    loss = lam * F.cross_entropy(logits, y_a) + (1-lam) * F.cross_entropy(logits, y_b)
    loss.backward()
    optimizer.second_step(zero_grad=True)
```

---

## Testing

Run the test suite:

```bash
# Unit tests
pytest python/tests/test_booleanization_unit.py -v
pytest python/tests/test_integration.py -v

# E2E tests
pytest python/tests/test_booleanization_e2e.py -v
pytest python/tests/test_sota_validation.py -v
```



