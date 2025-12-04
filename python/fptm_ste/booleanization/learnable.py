from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn


class LearnableBinarizer(nn.Module):
    """
    General-purpose learnable binarizer supporting single or dual channels.

    Parameters
    ----------
    in_channels:
        Number of input feature channels.
    num_thresholds:
        Number of learnable thresholds per channel.
    init_temperature:
        Initial temperature used before annealing.
    mode:
        ``\"single\"`` for standard sigmoid binarisation (non-negative inputs),
        ``\"dual\"`` for paired sigmoid heads that capture positive/negative
        evidence separately (zero-centred inputs such as transformer features).
    """

    def __init__(
        self,
        in_channels: int,
        num_thresholds: int = 16,
        init_temperature: float = 1.0,
        *,
        mode: Literal["single", "dual"] = "single",
        stat_momentum: float = 0.05,
    ) -> None:
        super().__init__()
        if num_thresholds <= 0:
            raise ValueError("num_thresholds must be positive.")
        if mode not in {"single", "dual"}:
            raise ValueError(f"Unsupported mode '{mode}'.")

        self.num_thresholds = num_thresholds
        self.temperature = float(init_temperature)
        self.mode = mode
        self.stat_momentum = stat_momentum

        self.reduce = nn.Conv2d(in_channels, num_thresholds, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.reduce.weight, gain=0.5)
        nn.init.zeros_(self.reduce.bias)

        self.register_buffer("running_mean", torch.zeros(1, num_thresholds, 1, 1))
        self.register_buffer("running_std", torch.ones(1, num_thresholds, 1, 1))
        self.register_buffer("initialized", torch.tensor(False))

        init_offsets = torch.linspace(-1.0, 1.0, num_thresholds).view(1, num_thresholds, 1, 1)
        self.threshold_offsets = nn.Parameter(init_offsets)

    @property
    def output_channels(self) -> int:
        return self.num_thresholds * 2 if self.mode == "dual" else self.num_thresholds

    def set_temperature(self, value: float) -> None:
        self.temperature = float(value)

    def anneal_temperature(self, value: float) -> None:
        self.temperature = max(0.01, float(value))

    def reset_running_stats(self) -> None:
        self.running_mean.zero_()
        self.running_std.fill_(1.0)
        self.initialized.fill_(False)

    def _update_stats(self, tensor: torch.Tensor) -> None:
        if not self.training:
            return
        with torch.no_grad():
            mean = tensor.mean(dim=[0, 2, 3], keepdim=True)
            std = tensor.std(dim=[0, 2, 3], keepdim=True).clamp(min=0.1)
            momentum = self.stat_momentum
            self.running_mean.mul_(1 - momentum).add_(mean, alpha=momentum)
            self.running_std.mul_(1 - momentum).add_(std, alpha=momentum)
            self.running_mean.clamp_(-10.0, 10.0)
            self.running_std.clamp_(0.01, 10.0)

    def forward(self, x: torch.Tensor, use_discrete: bool = False) -> torch.Tensor:
        x = x.float()
        reduced = self.reduce(x)

        if not self.initialized and self.training:
            with torch.no_grad():
                self.running_mean.copy_(reduced.mean(dim=[0, 2, 3], keepdim=True))
                self.running_std.copy_(reduced.std(dim=[0, 2, 3], keepdim=True).clamp(min=0.1))
                self.initialized.fill_(True)

        self._update_stats(reduced)
        thresholds = (self.running_mean + self.threshold_offsets * self.running_std).clamp(-10.0, 10.0)
        temp = self.temperature + 1e-8

        if self.mode == "dual":
            diff_neg = torch.clamp((-(reduced) - thresholds) / temp * 5.0, -20.0, 20.0)
            diff_pos = torch.clamp(((reduced) - thresholds) / temp * 5.0, -20.0, 20.0)
            logits = torch.cat([diff_neg, diff_pos], dim=1)
        else:
            logits = torch.clamp(((reduced - thresholds) / temp) * 5.0, -20.0, 20.0)

        probs = torch.sigmoid(logits)
        if use_discrete or not self.training:
            with torch.no_grad():
                return (probs > 0.5).float()
        return probs


__all__ = ["LearnableBinarizer"]

