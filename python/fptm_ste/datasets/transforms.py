"""
Utility transforms for backbone-aligned preprocessing.

These helpers provide torchvision pipelines that respect backbone-specific
normalisation statistics and expose optional mixup / cutmix callables that can
be applied to mini-batches after augmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from ..backbones import get_backbone_normalization

Tensor = torch.Tensor
MixFn = Optional[Callable[[Tensor, torch.Tensor], Tuple[Tensor, torch.Tensor]]]


def _sample_beta(alpha: float) -> float:
    if alpha <= 0:
        return 1.0
    return torch.distributions.Beta(alpha, alpha).sample().item()


def build_mixup(alpha: float) -> MixFn:
    if alpha <= 0:
        return None

    def _apply(data: Tensor, target: torch.Tensor) -> Tuple[Tensor, torch.Tensor]:
        lam = _sample_beta(alpha)
        index = torch.randperm(data.size(0), device=data.device)
        mixed = lam * data + (1 - lam) * data[index]
        if target.ndim == 1:
            target_one_hot = torch.nn.functional.one_hot(target, num_classes=target.max().item() + 1).float()
        else:
            target_one_hot = target.float()
        mixed_target = lam * target_one_hot + (1 - lam) * target_one_hot[index]
        return mixed, mixed_target

    return _apply


def build_cutmix(alpha: float) -> MixFn:
    if alpha <= 0:
        return None

    def _apply(data: Tensor, target: torch.Tensor) -> Tuple[Tensor, torch.Tensor]:
        lam = _sample_beta(alpha)
        index = torch.randperm(data.size(0), device=data.device)
        bbx1, bby1, bbx2, bby2 = _rand_bbox(data.size(-2), data.size(-1), lam)
        cut_data = data.clone()
        cut_data[:, :, bby1:bby2, bbx1:bbx2] = data[index, :, bby1:bby2, bbx1:bbx2]
        lam_adj = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size(-1) * data.size(-2)))
        if target.ndim == 1:
            target_one_hot = torch.nn.functional.one_hot(target, num_classes=target.max().item() + 1).float()
        else:
            target_one_hot = target.float()
        mixed_target = lam_adj * target_one_hot + (1 - lam_adj) * target_one_hot[index]
        return cut_data, mixed_target

    return _apply


def _rand_bbox(width: int, height: int, lam: float) -> Tuple[int, int, int, int]:
    cut_ratio = torch.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    cx = torch.randint(0, width, (1,)).item()
    cy = torch.randint(0, height, (1,)).item()

    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, width)
    bby2 = min(cy + cut_h // 2, height)
    return bbx1, bby1, bbx2, bby2


@dataclass
class BackboneTransformBundle:
    train_transform: transforms.Compose
    eval_transform: transforms.Compose
    mixup_fn: MixFn
    cutmix_fn: MixFn
    mean: Tuple[float, ...]
    std: Tuple[float, ...]


def build_backbone_transforms(
    backbone_type: str,
    *,
    input_size: int,
    input_channels: int = 3,
    augment: Optional[str] = "randaugment",
    normalize: bool = True,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
) -> BackboneTransformBundle:
    """
    Create torchvision transforms aligned with the target backbone.

    Parameters
    ----------
    backbone_type:
        Identifier passed to :func:`get_backbone_normalization`.
    input_size:
        Final spatial resolution expected by the backbone.
    input_channels:
        Number of input channels fed into the backbone.
    augment:
        Optional augmentation policy (``randaugment`` or ``autoaugment``).
    normalize:
        If ``True`` apply per-channel normalisation using backbone stats.
    mixup_alpha / cutmix_alpha:
        If greater than zero the returned bundle will include callable hooks
        that apply mixup / cutmix to mini-batches.
    """

    mean, std = get_backbone_normalization(backbone_type, input_channels=input_channels)

    aug_ops: List[Callable] = []
    if augment == "randaugment":
        aug_ops.append(
            transforms.RandAugment(num_ops=2, magnitude=9, interpolation=InterpolationMode.BICUBIC)
        )
    elif augment == "autoaugment":
        aug_ops.append(
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET)
        )

    resize_op = transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC)
    to_tensor = transforms.Lambda(lambda img: img if isinstance(img, torch.Tensor) else transforms.functional.to_tensor(img))

    train_transforms = transforms.Compose(
        [
            resize_op,
            *aug_ops,
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean, std) if normalize else transforms.Lambda(lambda x: x),
        ]
    )
    eval_transforms = transforms.Compose(
        [
            resize_op,
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean, std) if normalize else transforms.Lambda(lambda x: x),
        ]
    )

    mixup_fn = build_mixup(mixup_alpha)
    cutmix_fn = build_cutmix(cutmix_alpha)
    return BackboneTransformBundle(
        train_transform=train_transforms,
        eval_transform=eval_transforms,
        mixup_fn=mixup_fn,
        cutmix_fn=cutmix_fn,
        mean=mean,
        std=std,
    )


__all__ = [
    "BackboneTransformBundle",
    "build_backbone_transforms",
    "build_mixup",
    "build_cutmix",
]

