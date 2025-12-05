# TM SOTA Improvements Roadmap

**Analysis Date:** December 2, 2025

This document identifies the major gaps between the current TM implementation and modern ANN SOTA practices, with specific code-level recommendations.

---

## 1. DATA AUGMENTATION (Critical Gap)

### Current State
- Minimal augmentation (basic normalization)
- No RandAugment, CutMix, MixUp, AutoAugment

### SOTA Practice
ANNs achieve +5-10% accuracy gains from strong augmentation on CIFAR-10.

### Implementation

```python
# Add to run_mnist_equiv.py or create fptm_ste/augmentation.py

class TMDataAugmentation:
    """SOTA augmentation pipeline for TM training."""
    
    def __init__(self, mode='strong'):
        self.mode = mode
        
        if mode == 'strong':
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=14),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                CutMixMixUp(alpha=1.0),  # Mix strategies
            ])
    
    def cutmix_mixup(self, batch, labels):
        """Apply CutMix or MixUp randomly."""
        if random.random() < 0.5:
            return cutmix(batch, labels, alpha=1.0)
        return mixup(batch, labels, alpha=0.8)
```

**Files to modify:** `run_mnist_equiv.py` lines 3098-3120

---

## 2. NORMALIZATION (Moderate Gap)

### Current State
- Using `BatchNorm2d` and `LayerNorm`
- Post-norm architecture

### SOTA Practice
- Pre-norm (more stable training)
- RMSNorm (faster, simpler)
- No affine parameters sometimes

### Implementation

```python
# Add to fptm_ste/layers.py

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (faster than LayerNorm)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class PreNormResidual(nn.Module):
    """Pre-normalization residual block (more stable)."""
    def __init__(self, dim, fn, norm=RMSNorm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x):
        return x + self.fn(self.norm(x))
```

**Files to modify:** 
- `deep_tm.py` line 137: Change from post-norm to pre-norm
- `deep_ctm.py` line 166: Use RMSNorm instead of BatchNorm

---

## 3. ATTENTION IMPROVEMENTS (Significant Gap)

### Current State
- Basic `nn.MultiheadAttention`
- No flash attention
- No relative position bias

### SOTA Practice
- Flash Attention / Memory-efficient attention
- Relative position bias (Swin-style)
- SwiGLU feed-forward

### Implementation

```python
# Add to fptm_ste/attention.py

class FlashSelfAttention(nn.Module):
    """Flash attention with scaled dot product."""
    def __init__(self, dim, heads=8, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Use PyTorch 2.0 scaled_dot_product_attention (uses Flash when available)
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        
        return self.proj(attn_out.transpose(1, 2).reshape(B, N, C))


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network (better than GELU)."""
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8/3)  # SwiGLU uses 8/3 expansion
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))
```

**Files to modify:**
- `clause_attention.py`: Replace MultiHeadAttention with FlashSelfAttention
- `deep_ctm.py` lines 435-440: Use SwiGLU instead of GELU MLP

---

## 4. RESIDUAL CONNECTIONS (Minor Gap)

### Current State
- Standard residual connections
- No stochastic depth

### SOTA Practice
- Stochastic depth / DropPath
- ReZero initialization
- Pre-activation patterns

### Implementation

```python
# Add to fptm_ste/layers.py

class DropPath(nn.Module):
    """Stochastic depth regularization."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x / keep_prob * random_tensor.floor_()


class ReZeroResidual(nn.Module):
    """ReZero: All you need is a good init."""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.alpha = nn.Parameter(torch.zeros(1))  # Start at 0

    def forward(self, x):
        return x + self.alpha * self.fn(x)
```

**Files to modify:**
- `deep_ctm.py` line 258: Add DropPath with linearly increasing probability
- `deep_tm.py` line 137: Add DropPath

---

## 5. FEATURE EXTRACTION (Critical Gap)

### Current State
- Simple conv stem or flat input
- No pretrained backbone
- No multi-scale features

### SOTA Practice
- Use pretrained CNN/ViT features
- Multi-scale feature pyramid
- Overlapping patch embedding

### Implementation

```python
# Add to fptm_ste/backbones.py

class PretrainedBackbone(nn.Module):
    """Use pretrained backbone for feature extraction."""
    def __init__(self, model_name='convnext_tiny', pretrained=True, freeze=False):
        super().__init__()
        import timm
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.out_dim = self.backbone.num_features

    def forward(self, x):
        return self.backbone(x)


class HybridTMWithBackbone(nn.Module):
    """TM classifier on top of pretrained features."""
    def __init__(self, backbone='convnext_tiny', n_clauses=512, n_classes=10):
        super().__init__()
        self.backbone = PretrainedBackbone(backbone, pretrained=True, freeze=True)
        self.tm_head = FuzzyPatternTM_STCM(
            n_features=self.backbone.out_dim,
            n_clauses=n_clauses,
            n_classes=n_classes,
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.tm_head(features)
```

**Expected Impact:** +10-20% accuracy using pretrained features

---

## 6. OPTIMIZER & SCHEDULER (Moderate Gap)

### Current State
- AdamW with cosine decay
- Basic warmup

### SOTA Practice
- SAM (Sharpness Aware Minimization)
- LAMB for large batch
- Gradient accumulation
- EMA (already partially present)

### Implementation

```python
# Add to fptm_ste/optimizers.py

class SAM(torch.optim.Optimizer):
    """Sharpness Aware Minimization - finds flatter minima."""
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        self.base_optimizer = base_optimizer(params, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.rho = rho

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                e_w = p.grad * self.rho / (grad_norm + 1e-12)
                p.add_(e_w)  # Climb to local max
                self.state[p]['e_w'] = e_w

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                p.sub_(self.state[p]['e_w'])  # Go back
        self.base_optimizer.step()

    def _grad_norm(self):
        norm = torch.norm(torch.stack([
            p.grad.norm(p=2) for group in self.param_groups 
            for p in group['params'] if p.grad is not None
        ]), p=2)
        return norm


# Usage in training loop:
# optimizer = SAM(model.parameters(), torch.optim.AdamW, rho=0.05, lr=1e-3)
# for x, y in loader:
#     loss = criterion(model(x), y)
#     loss.backward()
#     optimizer.first_step()  # Ascend
#     criterion(model(x), y).backward()
#     optimizer.second_step()  # Descend
```

**Files to modify:** `run_mnist_equiv.py` line 3057

---

## 7. CLAUSE INITIALIZATION (Important Gap)

### Current State
- Random initialization with small std
- No structured initialization

### SOTA Practice
- Xavier/Kaiming initialization
- Structured patterns for clauses
- Meta-learned initialization

### Implementation

```python
# Add to fptm_ste/initialization.py

def init_clauses_structured(n_clauses, n_features, n_classes, pattern='orthogonal'):
    """Structured clause initialization."""
    if pattern == 'orthogonal':
        # Orthogonal initialization for diversity
        weights = torch.empty(n_clauses, n_features)
        nn.init.orthogonal_(weights)
        return weights * 0.1
    
    elif pattern == 'prototype':
        # Initialize clauses as prototypes
        clauses_per_class = n_clauses // n_classes
        weights = torch.zeros(n_clauses, n_features)
        for c in range(n_classes):
            start = c * clauses_per_class
            end = start + clauses_per_class
            # Initialize each class's clauses around a learned prototype
            weights[start:end] = torch.randn(clauses_per_class, n_features) * 0.05
        return weights
    
    elif pattern == 'sparse':
        # Sparse initialization - each clause focuses on few features
        weights = torch.zeros(n_clauses, n_features)
        sparsity = 0.1  # 10% of features per clause
        for i in range(n_clauses):
            mask = torch.rand(n_features) < sparsity
            weights[i, mask] = torch.randn(mask.sum()) * 0.1
        return weights
    
    return torch.randn(n_clauses, n_features) * 0.05
```

**Files to modify:** `tm.py` lines 1262-1263

---

## 8. TRAINING STRATEGIES (Significant Gap)

### Current State
- Single-stage training
- No progressive resizing
- Limited distillation

### SOTA Practice
- Progressive training (start small, increase)
- Knowledge distillation from larger models
- Self-supervised pretraining then fine-tune

### Implementation

```python
# Add to fptm_ste/training_strategies.py

class ProgressiveTrainer:
    """Progressive training with increasing difficulty."""
    
    def __init__(self, model, stages=[
        {'epochs': 10, 'img_size': 16, 'clauses_active': 0.25},
        {'epochs': 15, 'img_size': 24, 'clauses_active': 0.50},
        {'epochs': 25, 'img_size': 32, 'clauses_active': 1.00},
    ]):
        self.model = model
        self.stages = stages
    
    def train(self, train_loader, test_loader):
        for stage in self.stages:
            # Resize images
            resized_loader = self._resize_loader(train_loader, stage['img_size'])
            
            # Activate subset of clauses
            self._set_clause_activation(stage['clauses_active'])
            
            # Train
            train_epochs(self.model, resized_loader, stage['epochs'])


class DistillationTrainer:
    """Knowledge distillation from teacher model."""
    
    def __init__(self, student, teacher, temperature=4.0, alpha=0.5):
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha
    
    def distill_loss(self, student_logits, teacher_logits, labels):
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        hard_loss = F.cross_entropy(student_logits, labels)
        
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
```

---

## 9. ARCHITECTURE: MODERN BLOCK DESIGN

### Current State
- Conv -> BatchNorm -> Activation -> Residual
- Basic mixing modules

### SOTA Practice
- ConvNeXt-style blocks
- Inverted bottleneck
- Depthwise separable

### Implementation

```python
# Add to fptm_ste/blocks.py

class ConvNeXtBlock(nn.Module):
    """Modern ConvNeXt-style block for TM."""
    def __init__(self, dim, drop_path=0.0, layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        return shortcut + self.drop_path(x)
```

**Files to modify:** `deep_ctm.py` - Add ConvNeXt option for blocks

---

## 10. VOTING MECHANISM (TM-Specific)

### Current State
- Linear voting or simple attention
- No uncertainty-aware voting

### SOTA Practice (TM-specific improvements)
- Learnable temperature per clause
- Top-k sparse voting
- Clause confidence weighting

### Implementation

```python
# Already partially in tm.py, enhance:

class SparseTopKVoting(nn.Module):
    """Only top-k most confident clauses vote."""
    def __init__(self, n_clauses, n_classes, top_k=32):
        super().__init__()
        self.top_k = top_k
        self.voting = nn.Parameter(torch.randn(n_clauses, n_classes) * 0.1)
        self.temperature = nn.Parameter(torch.ones(n_clauses))

    def forward(self, clause_outputs):
        # Scale by learned temperature
        scaled = clause_outputs * self.temperature
        
        # Select top-k clauses
        _, top_indices = torch.topk(scaled.abs(), self.top_k, dim=1)
        mask = torch.zeros_like(clause_outputs).scatter_(1, top_indices, 1.0)
        sparse_outputs = clause_outputs * mask
        
        return torch.mm(sparse_outputs, self.voting)
```

---

## Priority Implementation Order

| Priority | Improvement | Expected Gain | Effort |
|----------|------------|---------------|--------|
| 🔴 1 | Pretrained Backbone | +10-20% | Medium |
| 🔴 2 | Strong Augmentation | +5-10% | Low |
| 🟡 3 | Flash Attention + SwiGLU | +2-3% | Medium |
| 🟡 4 | SAM Optimizer | +1-2% | Low |
| 🟡 5 | Stochastic Depth | +1-2% | Low |
| 🟢 6 | Pre-norm Architecture | +0.5-1% | Low |
| 🟢 7 | Structured Initialization | +0.5-1% | Low |
| 🟢 8 | Progressive Training | +1-2% | Medium |

---

## Quick Wins (Implement First)

1. **Add RandAugment**: Single line change in data loading
2. **Add DropPath**: Few lines in residual blocks  
3. **Use pretrained features**: Replace flat input with `timm` backbone
4. **Enable Flash Attention**: Set `torch.backends.cuda.sdp_kernel(enable_flash=True)`

---

## Summary

The biggest gaps are:
1. **Feature extraction** - Using raw pixels instead of pretrained features
2. **Data augmentation** - Missing modern augmentation stack
3. **Attention efficiency** - Not using Flash Attention
4. **Modern blocks** - Not using ConvNeXt-style design

Implementing these could push CIFAR-10 accuracy from **75% → 90%+**.



