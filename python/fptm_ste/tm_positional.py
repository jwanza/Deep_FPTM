from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class RelativePositionBias2D(nn.Module):
    """Learned 2D relative position bias for attention."""

    def __init__(self, num_heads: int, grid_size: Tuple[int, int], has_cls_token: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.grid_size = grid_size
        self.has_cls_token = has_cls_token
        height, width = grid_size
        relative_coords = self._build_relative_coords(height, width)
        self.register_buffer("relative_position_index", relative_coords, persistent=False)
        num_relative_positions = (2 * height - 1) * (2 * width - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_relative_positions, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    @staticmethod
    def _build_relative_coords(height: int, width: int) -> torch.Tensor:
        coords_h = torch.arange(height)
        coords_w = torch.arange(width)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2 x H x W
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2 x HW x HW
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # HW x HW x 2
        relative_coords[:, :, 0] += height - 1
        relative_coords[:, :, 1] += width - 1
        relative_coords[:, :, 0] *= 2 * width - 1
        relative_position_index = relative_coords.sum(-1)  # HW x HW
        return relative_position_index

    def forward(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        height, width = self.grid_size
        token_count = height * width
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(token_count, token_count, self.num_heads)
        bias = bias.permute(2, 0, 1).contiguous()  # num_heads x N x N
        bias = bias.to(dtype=dtype, device=device)
        if self.has_cls_token:
            full = torch.zeros(self.num_heads, token_count + 1, token_count + 1, device=device, dtype=dtype)
            full[:, 1:, 1:] = bias
            bias = full
        return bias


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding helper (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 1024, base: int = 10000) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Rotary embedding dimension must be even.")
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cached_seq_len", torch.tensor(0, dtype=torch.int64), persistent=False)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        if int(self.cached_seq_len) >= seq_len and self.cos_cached.device == device and self.cos_cached.dtype == dtype:
            return
        seq = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", seq, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        self.register_buffer("cached_seq_len", torch.tensor(seq_len, device=device), persistent=False)

    def get_sin_cos(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = min(seq_len, self.max_seq_len)
        self._build_cache(seq_len, device, dtype)
        return (
            self.sin_cached[:seq_len].view(1, 1, seq_len, self.dim),
            self.cos_cached[:seq_len].view(1, 1, seq_len, self.dim),
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.size(-1) // 2], x[..., x.size(-1) // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k

