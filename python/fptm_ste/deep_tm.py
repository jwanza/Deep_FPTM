import torch
import torch.nn as nn
import torch.nn.functional as F
from .tm import FuzzyPatternTM_STE


class DeepTMNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], n_classes: int,
                 n_clauses: int = 100, dropout: float = 0.1, tau: float = 0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residuals = nn.ModuleList()

        prev = input_dim
        for h in hidden_dims:
            self.layers.append(FuzzyPatternTM_STE(prev, n_clauses, h, tau=tau))
            self.norms.append(nn.LayerNorm(h))
            self.residuals.append(nn.Linear(prev, h, bias=False) if prev != h else nn.Identity())
            prev = h

        self.classifier = FuzzyPatternTM_STE(prev, n_clauses * 2, n_classes, tau=tau)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, use_ste: bool = True):
        # x in [0,1]
        for layer, norm, res in zip(self.layers, self.norms, self.residuals):
            identity = res(x)
            logits, _ = layer(x, use_ste=use_ste)   # [B, h]
            x = norm(self.dropout(torch.sigmoid(logits)) + identity)
        logits, clauses = self.classifier(x, use_ste=use_ste)
        return logits, clauses


