"""Swin-style pyramid Tsetlin Machine architecture."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_oracle import TM_Attention_Oracle
from .tm import FuzzyPatternTM_STE


@dataclass
class SwinStageConfig:
    """Configuration for a Swin-like stage."""

    embed_dim: int
    depth: int
    window_size: int = 4
    shift_size: int = 0
    tm_hidden_dim: int = 256
    tm_clauses: int = 256
    head_clauses: int = 128
    tau: float = 0.5
    dropout: float = 0.0
    clause_pool: str = "mean"
    logit_norm: str = "sqrt"


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition (B, H, W, C) tensor into windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse window partition."""
    num_windows = windows.shape[0]
    B = num_windows // ((H // window_size) * (W // window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
    return x


class PatchEmbedding(nn.Module):
    """Initial patch embedding via configurable stride convolution."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 4, norm_layer: Optional[nn.Module] = nn.LayerNorm):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    """Patch merging operation reducing resolution by 2 and doubling channels."""

    def __init__(self, input_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.reduction = nn.Linear(4 * input_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        x = self.norm(x)
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = x.permute(0, 3, 1, 2)
            x = F.pad(x, (0, pad_w, 0, pad_h))
            x = x.permute(0, 2, 3, 1)
            H += pad_h
            W += pad_w
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.reduction(x)
        return x


class _ClausePooler(nn.Module):
    def __init__(self, mode: str, n_clauses: int) -> None:
        super().__init__()
        mode = mode.lower()
        valid = {"none", "mean", "max", "mean_max", "learned"}
        if mode not in valid:
            raise ValueError(f"Unknown clause_pool mode '{mode}'. Choose from {sorted(valid)}")
        self.mode = mode
        self.n_clauses = n_clauses
        if mode == "learned":
            weight = torch.ones(n_clauses) / n_clauses
            self.weight = nn.Parameter(weight)

    def forward(self, clauses: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return clauses
        if self.mode == "mean":
            return clauses.mean(dim=1, keepdim=True)
        if self.mode == "max":
            return clauses.max(dim=1, keepdim=True)[0]
        if self.mode == "mean_max":
            mean = clauses.mean(dim=1, keepdim=True)
            maxv = clauses.max(dim=1, keepdim=True)[0]
            return torch.cat([mean, maxv], dim=1)
        weights = torch.softmax(self.weight, dim=0)
        return clauses @ weights.unsqueeze(-1)


class WindowTMBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        window_size: int,
        shift_size: int,
        tm_hidden_dim: int,
        tm_clauses: int,
        tau: float,
        dropout: float,
        clause_pool: str,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm = nn.LayerNorm(embed_dim)
        in_dim = window_size * window_size * embed_dim
        self.pre_tm = nn.Sequential(
            nn.Linear(in_dim, tm_hidden_dim),
            nn.Sigmoid(),
        )
        self.tm = FuzzyPatternTM_STE(tm_hidden_dim, tm_clauses, embed_dim, tau=tau)
        self.post_tm = nn.Sequential(
            nn.Linear(embed_dim, self.window_size * self.window_size * embed_dim),
            nn.Sigmoid(),
        )
        self.pooler = _ClausePooler(clause_pool, tm_clauses)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, use_ste: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, H, W, C = x.shape
        residual = x
        x_norm = self.norm(x)
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h or pad_w:
            residual = residual.permute(0, 3, 1, 2)
            residual = F.pad(residual, (0, pad_w, 0, pad_h))
            residual = residual.permute(0, 2, 3, 1)
            x_norm = x_norm.permute(0, 3, 1, 2)
            x_norm = F.pad(x_norm, (0, pad_w, 0, pad_h))
            x_norm = x_norm.permute(0, 2, 3, 1)
            H += pad_h
            W += pad_w
        if self.shift_size > 0:
            x_norm = torch.roll(x_norm, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows = window_partition(x_norm, self.window_size)
        windows_flat = windows.reshape(-1, self.window_size * self.window_size * C)
        tm_input = self.pre_tm(windows_flat)
        logits, clause_outputs = self.tm(tm_input, use_ste=use_ste)
        clause_summary = self.pooler(clause_outputs)
        window_features = self.post_tm(logits).view(-1, self.window_size, self.window_size, C)
        merged = window_reverse(window_features, self.window_size, H, W)
        if self.shift_size > 0:
            merged = torch.roll(merged, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        if pad_h or pad_w:
            merged = merged[:, :H - pad_h, :W - pad_w, :]
            residual = residual[:, :H - pad_h, :W - pad_w, :]

        gate = torch.sigmoid(self.gate)
        fused = gate * merged + (1.0 - gate) * torch.sigmoid(merged)
        x = residual + self.dropout(fused)
        clause_summary = clause_summary.view(-1, clause_summary.shape[-1])
        return x, clause_outputs, clause_summary


class SwinLikeStage(nn.Module):
    def __init__(
        self,
        config: SwinStageConfig,
        num_classes: int,
        downsample: bool,
    ) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList()
        for i in range(config.depth):
            shift = config.shift_size if (i % 2 == 1 and config.shift_size > 0) else 0
            self.blocks.append(
                WindowTMBlock(
                    embed_dim=config.embed_dim,
                    window_size=config.window_size,
                    shift_size=shift,
                    tm_hidden_dim=config.tm_hidden_dim,
                    tm_clauses=config.tm_clauses,
                    tau=config.tau,
                    dropout=config.dropout,
                    clause_pool=config.clause_pool,
                )
            )
        self.merge = PatchMerging(config.embed_dim, config.embed_dim * 2) if downsample else None
        self.head_norm = nn.LayerNorm(config.embed_dim)
        self.head_proj = nn.Sequential(
            nn.Linear(config.embed_dim, config.tm_hidden_dim),
            nn.Sigmoid(),
        )
        self.head_tm = FuzzyPatternTM_STE(
            config.tm_hidden_dim,
            config.head_clauses,
            num_classes,
            tau=config.tau,
        )

    def forward(self, x: torch.Tensor, use_ste: bool = True):
        clause_outputs: List[torch.Tensor] = []
        clause_summaries: List[torch.Tensor] = []

        for block in self.blocks:
            x, block_clause, block_summary = block(x, use_ste=use_ste)
            clause_outputs.append(block_clause)
            clause_summaries.append(block_summary)

        pooled = self.head_norm(x).mean(dim=(1, 2))
        head_inp = self.head_proj(pooled)
        stage_logits, stage_clause = self.head_tm(head_inp, use_ste=use_ste)

        if self.merge is not None:
            x = self.merge(x)

        return x, stage_logits, stage_clause, clause_outputs, clause_summaries


class SwinLikePyramidTM(nn.Module):
    def __init__(
        self,
        image_size: Tuple[int, int] = (28, 28),
        in_channels: int = 1,
        num_classes: int = 10,
        stage_configs: Optional[Sequence[SwinStageConfig]] = None,
        use_scale_attention: bool = True,
        scale_entropy_weight: float = 0.01,
        patch_size: int = 4,
    ) -> None:
        super().__init__()
        if stage_configs is None:
            stage_configs = (
                SwinStageConfig(embed_dim=96, depth=2, window_size=4, shift_size=0),
                SwinStageConfig(embed_dim=192, depth=2, window_size=4, shift_size=2),
                SwinStageConfig(embed_dim=384, depth=4, window_size=4, shift_size=0),
                SwinStageConfig(embed_dim=768, depth=2, window_size=4, shift_size=2),
            )

        self.image_size = image_size
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.patch_embed = PatchEmbedding(in_channels, stage_configs[0].embed_dim, patch_size=patch_size)
        stages: List[SwinLikeStage] = []
        for idx, cfg in enumerate(stage_configs):
            downsample = idx < len(stage_configs) - 1
            stages.append(SwinLikeStage(cfg, num_classes, downsample))
        self.stages = nn.ModuleList(stages)
        final_embed = stages[-1].config.embed_dim if self.stages else stage_configs[-1].embed_dim
        self.final_norm = nn.LayerNorm(final_embed)
        self.final_proj = nn.Sequential(
            nn.Linear(final_embed, stage_configs[-1].tm_hidden_dim),
            nn.Sigmoid(),
        )
        self.head = FuzzyPatternTM_STE(stage_configs[-1].tm_hidden_dim, stage_configs[-1].head_clauses, num_classes, tau=stage_configs[-1].tau)

        self.use_scale_attention = use_scale_attention
        self.scale_entropy_weight = scale_entropy_weight if use_scale_attention else 0.0
        if use_scale_attention:
            self.scale_logits_param = nn.Parameter(torch.zeros(len(self.stages) + 1))
            self.attention_gate = nn.Parameter(torch.tensor(0.5))
            self.attention_oracle = TM_Attention_Oracle(len(self.stages) + 1, num_classes, d_model=num_classes, n_heads=max(1, num_classes // 2))
        else:
            self.register_parameter("scale_logits_param", None)
            self.attention_gate = None
            self.attention_oracle = None
        self._scale_entropy_loss: Optional[torch.Tensor] = None
        self.last_attention_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, use_ste: bool = True):
        if x.dim() == 2:
            H, W = self.image_size
            expected = self.in_channels * H * W
            if x.shape[1] != expected:
                raise ValueError(f"Expected flattened input of size {expected}, got {x.shape[1]}")
            x = x.view(x.size(0), self.in_channels, H, W)
        elif x.dim() == 4:
            pass
        else:
            raise ValueError("Input must be flat [B, F] or image tensor [B, C, H, W]")

        features = self.patch_embed(x)
        stage_logits_list: List[torch.Tensor] = []
        clause_outputs: List[torch.Tensor] = []

        for stage in self.stages:
            features, s_logits, s_clause, _, _ = stage(features, use_ste=use_ste)
            stage_logits_list.append(s_logits)
            clause_outputs.append(s_clause)

        pooled = self.final_norm(features).mean(dim=(1, 2))
        head_inp = self.final_proj(pooled)
        final_logits, final_clause = self.head(head_inp, use_ste=use_ste)

        scale_logits = stage_logits_list + [final_logits]
        all_clauses = clause_outputs + [final_clause]

        if self.scale_logits_param is not None:
            weights = torch.softmax(self.scale_logits_param, dim=0)
            stacked = torch.stack(scale_logits, dim=0)
            weighted = torch.sum(stacked * weights.view(-1, 1, 1), dim=0)
            if self.scale_entropy_weight > 0:
                entropy = -(weights * torch.log(weights + 1e-8)).sum() * self.scale_entropy_weight
                self._scale_entropy_loss = entropy
            else:
                self._scale_entropy_loss = None
            self.last_attention_weights = weights.detach()
            final_output = weighted
        else:
            stacked = torch.stack(scale_logits, dim=0)
            final_output = stacked.mean(dim=0)
            self.last_attention_weights = None
            self._scale_entropy_loss = None

        if self.attention_oracle is not None:
            gate = torch.sigmoid(self.attention_gate)
            attn_out = self.attention_oracle(scale_logits, all_clauses)
            final_output = gate * attn_out + (1.0 - gate) * final_output

        return final_output, scale_logits, all_clauses

    def attention_entropy_loss(self) -> Optional[torch.Tensor]:
        return self._scale_entropy_loss
