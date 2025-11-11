from __future__ import annotations

from dataclasses import dataclass

import torch


def _ensure_3d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2:
        return tensor.unsqueeze(0)
    if tensor.dim() == 3:
        return tensor
    raise ValueError(f"Expected tensor with 2 or 3 dims (got shape {tuple(tensor.shape)})")


def adjust_channels(
    tensor: torch.Tensor,
    target_channels: int,
    *,
    allow_expand: bool = True,
    allow_reduce: bool = True,
) -> torch.Tensor:
    if target_channels <= 0:
        raise ValueError(f"target_channels must be positive (got {target_channels})")

    tensor = _ensure_3d(tensor)
    channels = tensor.shape[0]

    if channels == target_channels:
        return tensor

    if channels == 1 and target_channels > 1:
        if not allow_expand:
            raise ValueError(
                "Cannot expand single-channel input to multiple channels when allow_expand=False."
            )
        return tensor.repeat(target_channels, 1, 1)

    if channels > 1 and target_channels == 1:
        if not allow_reduce:
            raise ValueError(
                "Cannot reduce multi-channel input to single channel when allow_reduce=False."
            )
        return tensor.mean(dim=0, keepdim=True)

    raise ValueError(
        f"Unsupported channel adjustment from {channels} -> {target_channels}. "
        "Only single-channel expansion and simple reduction are implemented."
    )


@dataclass
class ChannelAdjust:
    """Transform that adjusts an image tensor to the desired channel count."""

    target_channels: int
    allow_expand: bool = True
    allow_reduce: bool = True

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.target_channels is None:
            return tensor
        return adjust_channels(
            tensor,
            self.target_channels,
            allow_expand=self.allow_expand,
            allow_reduce=self.allow_reduce,
        )

