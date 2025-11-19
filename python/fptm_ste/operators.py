from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class _BaseTernaryOperator(nn.Module):
    """Base class for ternary clause operators."""

    name: str = "base"

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class TernaryQuantumAND(_BaseTernaryOperator):
    """
    Quantum-inspired ternary AND (TQAND).

    Normalizes the multiplicative interaction of three inputs to stay numerically
    stable while preserving relative magnitudes.
    """

    name = "tqand"

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) < 3:
            raise ValueError("TernaryQuantumAND expects at least three inputs.")
        a, b, c = inputs[:3]
        denom = torch.sqrt(torch.clamp(a**2 + b**2 + c**2, min=self.eps))
        return (a * b * c) / denom


class TernaryXOR(_BaseTernaryOperator):
    """
    Ternary XOR (TXOR).

    Emphasizes disagreement between two inputs using a quadratic remainder that
    stays inside the [0, 1] interval.
    """

    name = "txor"

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) < 2:
            raise ValueError("TernaryXOR expects at least two inputs.")
        a, b = inputs[:2]
        diff_sq = (a - b) ** 2
        return torch.remainder(diff_sq, 3.0) / 3.0


class TernaryMajority(_BaseTernaryOperator):
    """
    Ternary majority gate (TMAJ).

    Returns the median across all inputs which corresponds to the balanced
    ternary majority vote.
    """

    name = "tmaj"

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if not inputs:
            raise ValueError("TernaryMajority expects at least one input.")
        stacked = torch.stack(inputs)
        return torch.median(stacked, dim=0).values


_REGISTERED_OPERATORS = {
    cls.name: cls for cls in (TernaryQuantumAND, TernaryXOR, TernaryMajority)
}


def available_ternary_operators() -> Tuple[str, ...]:
    """Returns the tuple of operator identifiers that can be requested."""

    return tuple(sorted(_REGISTERED_OPERATORS))


def build_ternary_operator(name: Optional[str]) -> Optional[_BaseTernaryOperator]:
    """
    Returns an instantiated operator module if ``name`` refers to a custom ternary
    operator. ``None`` means the caller should fall back to the legacy capacity /
    product logic implemented inside the STCM module.
    """

    if name is None:
        return None
    key = name.lower()
    cls = _REGISTERED_OPERATORS.get(key)
    if cls is None:
        return None
    return cls()

