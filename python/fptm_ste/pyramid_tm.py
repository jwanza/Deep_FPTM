from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tm import FuzzyPatternTM_STE


@dataclass
class PyramidStageConfig:
    """Configuration for a single pyramid stage."""

    pool_size: int
    projection_dim: int
    n_clauses: int
    tau: float = 0.5
    dropout: float = 0.0


class _PyramidStage(nn.Module):
    def __init__(
        self,
        input_channels: int,
        pool_size: int,
        projection_dim: int,
        n_clauses: int,
        n_classes: int,
        tau: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pool = nn.Identity() if pool_size is None else nn.AdaptiveAvgPool2d(pool_size)
        flat_dim = input_channels * (pool_size ** 2 if pool_size is not None else 0)
        dense_dim = max(32, projection_dim)
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, dense_dim),
            nn.LayerNorm(dense_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, projection_dim),
            nn.Sigmoid(),
        )
        self.tm = FuzzyPatternTM_STE(projection_dim, n_clauses, n_classes, tau=tau)

    def forward(self, x: torch.Tensor, use_ste: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = self.pool(x)
        features = self.projector(pooled)
        logits, clauses = self.tm(features, use_ste=use_ste)
        return logits, clauses


class PyramidTM(nn.Module):
    """Multi-scale TM architecture using adaptive pooling and shared input."""

    def __init__(
        self,
        input_resolution: int = 28,
        input_channels: int = 1,
        n_classes: int = 10,
        stage_configs: Sequence[PyramidStageConfig] | None = None,
    ) -> None:
        super().__init__()
        if stage_configs is None:
            stage_configs = (
                PyramidStageConfig(pool_size=28, projection_dim=256, n_clauses=256, tau=0.5, dropout=0.05),
                PyramidStageConfig(pool_size=14, projection_dim=192, n_clauses=192, tau=0.5, dropout=0.05),
                PyramidStageConfig(pool_size=7, projection_dim=128, n_clauses=128, tau=0.5, dropout=0.05),
                PyramidStageConfig(pool_size=1, projection_dim=64, n_clauses=64, tau=0.5, dropout=0.0),
            )

        self.input_resolution = input_resolution
        self.input_channels = input_channels
        self.n_classes = n_classes

        stages: List[_PyramidStage] = []
        for cfg in stage_configs:
            if cfg.pool_size <= 0:
                raise ValueError("pool_size must be positive")
            stage = _PyramidStage(
                input_channels=input_channels,
                pool_size=cfg.pool_size,
                projection_dim=cfg.projection_dim,
                n_clauses=cfg.n_clauses,
                n_classes=n_classes,
                tau=cfg.tau,
                dropout=cfg.dropout,
            )
            stages.append(stage)
        self.stage_configs = list(stage_configs)
        self.stages = nn.ModuleList(stages)

        self.last_logits: List[torch.Tensor] | None = None
        self.last_clauses: List[torch.Tensor] | None = None

    def forward(self, x: torch.Tensor, use_ste: bool = True):
        if x.dim() == 2:
            expected = self.input_channels * self.input_resolution * self.input_resolution
            if x.shape[1] != expected:
                raise ValueError(f"Expected feature dimension {expected}, got {x.shape[1]}")
            feats = x.view(x.size(0), self.input_channels, self.input_resolution, self.input_resolution)
        elif x.dim() == 4:
            feats = x
        else:
            raise ValueError("Input must be flat [B, F] or image tensor [B, C, H, W]")

        logits_list: List[torch.Tensor] = []
        clauses_list: List[torch.Tensor] = []
        for stage in self.stages:
            logits, clauses = stage(feats, use_ste=use_ste)
            logits_list.append(logits)
            clauses_list.append(clauses)

        stacked = torch.stack(logits_list, dim=0)
        final_logits = stacked.mean(dim=0)
        self.last_logits = logits_list
        self.last_clauses = clauses_list
        return final_logits, logits_list, clauses_list
