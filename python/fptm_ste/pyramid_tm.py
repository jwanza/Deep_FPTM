from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from .attention_oracle import TM_Attention_Oracle
from .tm import FuzzyPatternTM_STE


@dataclass
class PyramidStageConfig:
    pool_size: int
    projection_dim: int
    n_clauses: int
    tau: float = 0.5
    dropout: float = 0.0
    clause_pool: str = "mean"
    logit_norm: str = "sqrt"


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
        self.output_dim = self._infer_output_dim()

    def _infer_output_dim(self) -> int:
        if self.mode == "none":
            return self.n_clauses
        if self.mode == "mean_max":
            return 2
        return 1

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
        return torch.matmul(clauses, weights.unsqueeze(-1))


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
        clause_pool: str,
        logit_norm: str,
    ) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        flat_dim = input_channels * pool_size * pool_size
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
        self.pooler = _ClausePooler(clause_pool, n_clauses)
        if logit_norm == "sqrt":
            self.logit_scale = 1.0 / math.sqrt(n_clauses)
        elif logit_norm == "count":
            self.logit_scale = 1.0 / n_clauses
        elif logit_norm == "none":
            self.logit_scale = 1.0
        else:
            raise ValueError("logit_norm must be 'sqrt', 'count', or 'none'")

    def forward(self, x: torch.Tensor, use_ste: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = self.pool(x)
        features = self.projector(pooled)
        logits, clauses = self.tm(features, use_ste=use_ste)
        logits = logits * self.logit_scale
        summary = self.pooler(clauses)
        return logits, clauses, summary


class PyramidTM(nn.Module):
    """Multi-scale TM architecture with clause and scale attention options."""

    def __init__(
        self,
        input_resolution: int = 28,
        input_channels: int = 1,
        n_classes: int = 10,
        stage_configs: Sequence[PyramidStageConfig] | None = None,
        use_clause_attention: bool = True,
        attention_heads: int = 2,
        use_scale_attention: bool = True,
        scale_entropy_weight: float = 0.01,
        input_noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        if stage_configs is None:
            stage_configs = (
                PyramidStageConfig(pool_size=28, projection_dim=256, n_clauses=256, tau=0.55, dropout=0.05),
                PyramidStageConfig(pool_size=14, projection_dim=192, n_clauses=192, tau=0.55, dropout=0.05),
                PyramidStageConfig(pool_size=7, projection_dim=128, n_clauses=160, tau=0.55, dropout=0.05),
                PyramidStageConfig(pool_size=1, projection_dim=96, n_clauses=128, tau=0.5, dropout=0.0),
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
                clause_pool=cfg.clause_pool,
                logit_norm=cfg.logit_norm,
            )
            stages.append(stage)
        self.stage_configs = list(stage_configs)
        self.stages = nn.ModuleList(stages)

        self.use_clause_attention = use_clause_attention
        self.use_scale_attention = use_scale_attention
        self.scale_entropy_weight = scale_entropy_weight if use_scale_attention else 0.0
        self.input_noise_std = input_noise_std

        if use_clause_attention and n_classes % attention_heads != 0:
            raise ValueError("n_classes must be divisible by attention_heads for clause attention.")

        if use_clause_attention:
            self.attention_oracle = TM_Attention_Oracle(len(stages), n_classes, d_model=n_classes, n_heads=attention_heads)
            self.attention_gate = nn.Parameter(torch.tensor(0.5))
        else:
            self.attention_oracle = None
            self.attention_gate = None

        if use_scale_attention:
            self.scale_logits_param = nn.Parameter(torch.zeros(len(stages)))
        else:
            self.register_parameter("scale_logits_param", None)

        self.last_logits: List[torch.Tensor] | None = None
        self.last_clauses: List[torch.Tensor] | None = None
        self.last_clause_summaries: List[torch.Tensor] | None = None
        self.last_attention_weights: torch.Tensor | None = None
        self._scale_entropy_loss: torch.Tensor | None = None

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
        summaries: List[torch.Tensor] = []
        for stage in self.stages:
            logits, clauses, summary = stage(feats, use_ste=use_ste)
            logits_list.append(logits)
            clauses_list.append(clauses)
            summaries.append(summary)

        stacked = torch.stack(logits_list, dim=0)

        if self.scale_logits_param is not None:
            weights = torch.softmax(self.scale_logits_param, dim=0)
            final_logits = torch.sum(stacked * weights.view(-1, 1, 1), dim=0)
            self.last_attention_weights = weights.detach()
            if self.scale_entropy_weight > 0:
                entropy = -(weights * torch.log(weights + 1e-8)).sum()
                self._scale_entropy_loss = entropy * self.scale_entropy_weight
            else:
                self._scale_entropy_loss = None
        else:
            final_logits = stacked.mean(dim=0)
            self.last_attention_weights = None
            self._scale_entropy_loss = None

        if self.attention_oracle is not None:
            attn_logits = self.attention_oracle(logits_list, clauses_list)
            gate = torch.sigmoid(self.attention_gate)
            final_logits = gate * attn_logits + (1 - gate) * final_logits

        self.last_logits = logits_list
        self.last_clauses = clauses_list
        self.last_clause_summaries = summaries
        return final_logits, logits_list, clauses_list

    def attention_entropy_loss(self) -> torch.Tensor | None:
        return self._scale_entropy_loss
