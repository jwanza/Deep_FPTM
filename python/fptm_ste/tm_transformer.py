import math
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .tm import FuzzyPatternTM_STE
from .deep_tm import DeepTMNetwork
from .swin_tm import DropPath, WindowAttention
from .swin_pyramid_tm import PatchMerging, window_partition, window_reverse


def _to_sequence(value: Union[int, float, Sequence[Union[int, float]]], length: int) -> Tuple[Union[int, float], ...]:
    if isinstance(value, Sequence):
        if len(value) != length:
            raise ValueError(f"Expected sequence of length {length}, got {len(value)}.")
        return tuple(value)
    return tuple(value for _ in range(length))


class PatchEmbed(nn.Module):
    """Image to patch embedding."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int, *, flatten: bool = True) -> None:
        super().__init__()
        self.flatten = flatten
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if not self.flatten:
            x = x.view(B, H, W, C)
        return x


class TMFeedForward(nn.Module):
    """Feed-forward module backed by TM / DeepTM."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        backend: str,
        n_clauses: int,
        tm_tau: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backend = backend.lower()
        self.norm = nn.LayerNorm(dim)
        self.proj_in = nn.Linear(dim, hidden_dim)
        self.act = nn.Sigmoid()
        if self.backend == "ste":
            self.core = FuzzyPatternTM_STE(hidden_dim, n_clauses, dim, tau=tm_tau)
        elif self.backend in {"deep", "deeptm"}:
            self.core = DeepTMNetwork(
                input_dim=hidden_dim,
                hidden_dims=[hidden_dim],
                n_classes=dim,
                n_clauses=n_clauses,
                dropout=dropout,
                tau=tm_tau,
                input_shape=None,
                auto_expand_grayscale=False,
                allow_channel_reduce=False,
            )
        else:
            raise ValueError(f"Unsupported TM backend '{backend}'.")
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, use_ste: bool = True) -> torch.Tensor:
        B, T, C = x.shape
        y = self.norm(x)
        y = self.act(self.proj_in(y))
        flat = y.reshape(B * T, -1)
        logits, _ = self.core(flat, use_ste=use_ste)
        logits = logits.view(B, T, C)
        gate = torch.sigmoid(self.gate)
        logits = gate * logits + (1 - gate) * torch.sigmoid(logits)
        return self.dropout(logits)


class TMEncoderBlock(nn.Module):
    """ViT-style encoder block with TM/DeepTM feed-forward."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        mlp_ratio: float,
        backend: str,
        n_clauses: int,
        tm_tau: float,
        drop: float,
        attn_drop: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.attn_drop = nn.Dropout(drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = TMFeedForward(
            dim,
            hidden_dim,
            backend=backend,
            n_clauses=n_clauses,
            tm_tau=tm_tau,
            dropout=drop,
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, use_ste: bool = True) -> torch.Tensor:
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + self.drop_path(self.attn_drop(attn_out))
        x = x + self.drop_path(self.ffn(self.norm2(x), use_ste=use_ste))
        return x


class TMSwinBlock(nn.Module):
    """Windowed attention block with TM feed-forward."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        *,
        mlp_ratio: float,
        backend: str,
        n_clauses: int,
        tm_tau: float,
        drop: float,
        attn_drop: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads)
        self.attn_drop = nn.Dropout(drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = TMFeedForward(
            dim,
            hidden_dim,
            backend=backend,
            n_clauses=n_clauses,
            tm_tau=tm_tau,
            dropout=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, H: int, W: int, use_ste: bool = True) -> Tuple[torch.Tensor, int, int]:
        B, L, C = x.shape
        assert L == H * W, "Unexpected token count for Swin block"
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        H_pad, W_pad = x.shape[1:3]

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows = window_partition(x, self.window_size).view(-1, self.window_size * self.window_size, C)
        attn_out, _ = self.attn(windows)
        attn_out = attn_out.view(-1, self.window_size, self.window_size, C)
        if self.shift_size > 0:
            attn_out = torch.roll(attn_out, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = window_reverse(attn_out, self.window_size, H_pad, W_pad)
        if pad_h or pad_w:
            x = x[:, :H, :W, :]
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path1(self.attn_drop(x))

        x = x + self.drop_path2(self.ffn(self.norm2(x), use_ste=use_ste))
        return x, H, W


class TMSwinStage(nn.Module):
    """Stack of Swin blocks followed by optional patch merging."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        *,
        backend: str,
        mlp_ratio: float,
        n_clauses: int,
        tm_tau: float,
        drop: float,
        attn_drop: float,
        drop_path: Sequence[float],
        downsample: bool,
        use_checkpoint: bool,
    ) -> None:
        super().__init__()
        blocks = []
        for idx in range(depth):
            shift = 0 if idx % 2 == 0 else window_size // 2
            blocks.append(
                TMSwinBlock(
                    dim,
                    num_heads,
                    window_size,
                    shift,
                    mlp_ratio=mlp_ratio,
                    backend=backend,
                    n_clauses=n_clauses,
                    tm_tau=tm_tau,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[idx],
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.downsample = PatchMerging(dim, dim * 2) if downsample else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor, H: int, W: int, use_ste: bool = True) -> Tuple[torch.Tensor, int, int]:
        for block in self.blocks:
            if self.use_checkpoint and self.training and x.requires_grad:
                def _forward(inp, blk=block, h=H, w=W):  # type: ignore
                    out, _, _ = blk(inp, h, w, use_ste=use_ste)
                    return out

                x = checkpoint(_forward, x)
            else:
                x, H, W = block(x, H, W, use_ste=use_ste)
        if self.downsample is not None:
            x = x.view(-1, H, W, x.shape[-1])
            x = self.downsample(x)
            H //= 2
            W //= 2
            x = x.view(x.shape[0], -1, x.shape[-1])
        return x, H, W


class UnifiedTMTransformer(nn.Module):
    """Vision transformer with TM/DeepTM feed-forward options."""

    def __init__(
        self,
        *,
        num_classes: int,
        architecture: str = "vit",
        backend: str = "ste",
        image_size: Tuple[int, int] = (32, 32),
        in_channels: int = 3,
        patch_size: int = 4,
        embed_dim: Union[int, Sequence[int]] = 96,
        depths: Union[int, Sequence[int]] = 4,
        num_heads: Union[int, Sequence[int]] = 3,
        mlp_ratio: Union[float, Sequence[float]] = 4.0,
        tm_clauses: Union[int, Sequence[int]] = 256,
        tm_tau: float = 0.5,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        window_size: int = 7,
        use_cls_token: bool = True,
        pool: str = "cls",
        grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.architecture = architecture.lower()
        self.backend = backend.lower()
        self.num_classes = num_classes
        self.pool = pool
        self.grad_checkpoint = grad_checkpoint

        if self.architecture not in {"vit", "swin"}:
            raise ValueError("architecture must be 'vit' or 'swin'.")

        if self.architecture == "vit":
            self.patch_embed = PatchEmbed(in_channels, int(embed_dim), patch_size, flatten=True)
            num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, int(embed_dim))) if use_cls_token else None
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + (1 if self.cls_token is not None else 0), int(embed_dim)))
            self.pos_drop = nn.Dropout(drop_rate)
            depth = depths if isinstance(depths, int) else sum(depths)
            head = num_heads if isinstance(num_heads, int) else num_heads[0]
            mlp = mlp_ratio if isinstance(mlp_ratio, (int, float)) else mlp_ratio[0]
            clauses = tm_clauses if isinstance(tm_clauses, int) else tm_clauses[0]
            drop_path = torch.linspace(0, drop_path_rate, depth).tolist()
            self.blocks = nn.ModuleList(
                [
                    TMEncoderBlock(
                        int(embed_dim),
                        head,
                        mlp_ratio=mlp,
                        backend=self.backend,
                        n_clauses=int(clauses),
                        tm_tau=tm_tau,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=drop_path[i],
                    )
                    for i in range(depth)
                ]
            )
            self.norm = nn.LayerNorm(int(embed_dim))
            self.head = nn.Linear(int(embed_dim), num_classes)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            if self.cls_token is not None:
                nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            if isinstance(depths, int):
                raise ValueError("depths must be a sequence for Swin architecture.")
            stage_depths = tuple(depths)
            stage_heads = _to_sequence(num_heads, len(stage_depths))
            if isinstance(embed_dim, Sequence):
                stage_dims = tuple(embed_dim)
            else:
                stage_dims = tuple(int(embed_dim) * (2 ** i) for i in range(len(stage_depths)))
            stage_mlp = _to_sequence(mlp_ratio, len(stage_depths))
            stage_clauses = _to_sequence(tm_clauses, len(stage_depths))
            total_blocks = sum(stage_depths)
            drop_path = torch.linspace(0, drop_path_rate, total_blocks).tolist()
            self.patch_embed = PatchEmbed(in_channels, stage_dims[0], patch_size, flatten=False)
            stages = []
            dp_offset = 0
            for idx, depth in enumerate(stage_depths):
                stage = TMSwinStage(
                    dim=stage_dims[idx],
                    depth=depth,
                    num_heads=stage_heads[idx],
                    window_size=window_size,
                    backend=self.backend,
                    mlp_ratio=stage_mlp[idx],
                    n_clauses=int(stage_clauses[idx]),
                    tm_tau=tm_tau,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path[dp_offset : dp_offset + depth],
                    downsample=idx < len(stage_depths) - 1,
                    use_checkpoint=grad_checkpoint,
                )
                stages.append(stage)
                dp_offset += depth
            self.stages = nn.ModuleList(stages)
            self.norm = nn.LayerNorm(stage_dims[-1])
            self.head = nn.Linear(stage_dims[-1], num_classes)

        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, use_ste: bool = True) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("UnifiedTMTransformer expects image tensor of shape (B, C, H, W).")

        if self.architecture == "vit":
            x = self.patch_embed(x)
            if hasattr(self, "cls_token") and self.cls_token is not None:
                cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed[:, : x.size(1), :]
            x = self.pos_drop(x)
            for block in self.blocks:
                if self.grad_checkpoint and self.training and x.requires_grad:
                    x = checkpoint(lambda inp, blk=block: blk(inp, use_ste=use_ste), x)
                else:
                    x = block(x, use_ste=use_ste)
            x = self.norm(x)
            if hasattr(self, "cls_token") and self.cls_token is not None and self.pool == "cls":
                feats = x[:, 0]
            else:
                feats = x.mean(dim=1)
            logits = self.head(feats)
            return logits

        x = self.patch_embed(x)
        B, H, W, C = x.shape
        x = x.view(B, H * W, C)
        for stage in self.stages:
            x, H, W = stage(x, H, W, use_ste=use_ste)
        x = self.norm(x)
        feats = x.mean(dim=1)
        logits = self.head(feats)
        return logits


class TM_TransformerBlock(TMEncoderBlock):
    """Backwards-compatible alias of the legacy token block."""

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int = 512, dropout: float = 0.1, n_clauses: int = 200):
        super().__init__(
            dim=d_model,
            num_heads=n_heads,
            mlp_ratio=dim_feedforward / d_model,
            backend="ste",
            n_clauses=n_clauses,
            tm_tau=0.5,
            drop=dropout,
            attn_drop=dropout,
            drop_path=0.0,
        )

    def forward(self, x: torch.Tensor, use_ste: bool = True) -> torch.Tensor:
        return super().forward(x, use_ste=use_ste)


