"""
SOTA layers and components for TM improvements.
Implements modern ANN practices: RMSNorm, DropPath, SwiGLU, FlashAttention, ConvNeXt
"""

from __future__ import annotations
import math
from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - faster than LayerNorm."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class DropPath(nn.Module):
    """Stochastic Depth per sample."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * rand


class SwiGLU(nn.Module):
    """SwiGLU FFN - used in LLaMA, better than GELU."""
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class FlashAttention(nn.Module):
    """Flash Attention using PyTorch 2.0 scaled_dot_product_attention."""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop if self.training else 0.0)
        return self.proj_drop(self.proj(x.transpose(1, 2).reshape(B, N, C)))


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block with modern design."""
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim)) if layer_scale > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x).permute(0, 2, 3, 1)
        x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
        if self.gamma is not None: x = self.gamma * x
        return shortcut + self.drop_path(x.permute(0, 3, 1, 2))


class SparseTopKVoting(nn.Module):
    """Sparse voting using only top-k most confident clauses."""
    def __init__(self, n_clauses: int, n_classes: int, top_k: int = 32):
        super().__init__()
        self.top_k = min(top_k, n_clauses)
        self.voting = nn.Parameter(torch.randn(n_clauses, n_classes) * 0.1)
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, clause_outputs: torch.Tensor) -> torch.Tensor:
        _, idx = torch.topk(clause_outputs.abs(), self.top_k, dim=1)
        mask = torch.zeros_like(clause_outputs).scatter_(1, idx, 1.0)
        return F.linear(clause_outputs * mask / self.temperature, self.voting.t())


class PreNormResidual(nn.Module):
    """Pre-norm residual block."""
    def __init__(self, dim: int, fn: nn.Module, drop_path: float = 0.0):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.fn = fn
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.fn(self.norm(x)))
