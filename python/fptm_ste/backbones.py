"""
Backbone utilities and adapters for TM hybrids.

Besides the original lightweight wrappers, this module now exposes a
universal factory that returns multi-scale feature extractors with consistent
metadata (channels, reduction factors, normalisation stats).  This makes it
trivial for TM ensembles to align clause budgets with arbitrary CNN/ViT
backbones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # pragma: no cover - optional dependency
    import timm

    HAS_TIMM = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_TIMM = False

from .tm import FuzzyPatternTM_STCM

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

BACKBONE_STATS: Dict[str, Tuple[Tuple[float, ...], Tuple[float, ...]]] = {
    "swin": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    "convnext": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    "resnet": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    "efficientnet": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    "mobilenet": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    "vit": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    "simple": ((0.5,), (0.5,)),
}


@dataclass
class BackboneMetadata:
    """Structural information emitted by a backbone adapter."""

    backbone_type: str
    variant: str
    num_scales: int
    channels: List[int]
    reductions: List[int]
    input_size: int


def _ensure_stats_dim(stats: Sequence[float], channels: int) -> Tuple[float, ...]:
    if len(stats) == channels:
        return tuple(stats)
    if len(stats) == 1:
        return tuple(stats * channels)
    if channels == 1:
        return (float(sum(stats) / len(stats)),)
    return tuple(stats[:channels])


def get_backbone_normalization(
    backbone_type: str,
    *,
    input_channels: int = 3,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """
    Return `(mean, std)` tuples appropriate for the requested backbone.

    Unknown types fall back to ImageNet statistics.  Channel counts are
    adapted automatically so that grayscale datasets can reuse RGB stats.
    """

    base_mean, base_std = BACKBONE_STATS.get(
        backbone_type.lower(), (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    )
    return (
        _ensure_stats_dim(base_mean, input_channels),
        _ensure_stats_dim(base_std, input_channels),
    )


class BackboneAdapter(nn.Module):
    """Common interface for multi-scale feature extractors."""

    def get_output_channels(self) -> List[int]:
        raise NotImplementedError

    def get_reduction_factors(self) -> List[int]:
        raise NotImplementedError

    def num_scales(self) -> int:
        return len(self.get_output_channels())

    def metadata(self) -> BackboneMetadata:
        raise NotImplementedError
        
    def get_feature_info(self) -> Dict[str, Any]:
        """Expose detailed feature info (channels, reductions)."""
        meta = self.metadata()
        return {
            "channels": meta.channels,
            "reductions": meta.reductions,
            "num_scales": meta.num_scales,
            "input_size": meta.input_size,
            "backbone_type": meta.backbone_type,
            "variant": meta.variant
        }


class TimmFeatureBackbone(BackboneAdapter):
    """
    Adapter around timm feature extractors (`features_only=True`).
    """

    def __init__(
        self,
        model_name: str,
        *,
        num_scales: int,
        pretrained: bool = True,
        out_indices: Optional[Sequence[int]] = None,
        input_size: int = 224,
        freeze_stages: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if not HAS_TIMM:
            raise ImportError("timm is required to instantiate backbone models.")
        if out_indices is None:
            out_indices = tuple(range(num_scales))
        self.model_name = model_name
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=tuple(out_indices),
            **kwargs,
        )
        info = self.model.feature_info
        self._channels = list(info.channels()[: len(out_indices)])
        self._reductions = list(info.reduction()[: len(out_indices)])
        self._input_size = input_size
        
        if freeze_stages > 0:
            self._freeze_stages(freeze_stages)

    def _freeze_stages(self, freeze_stages: int):
        """
        Freeze the first N stages of the backbone.
        This is a best-effort implementation depending on timm model structure.
        """
        # Freeze patch embedding if it exists
        if hasattr(self.model, 'patch_embed'):
            for param in self.model.patch_embed.parameters():
                param.requires_grad = False
                
        # Try to freeze blocks/stages
        # Many timm models expose stages as children or blocks
        # This generic approach iterates over top-level children
        children = list(self.model.children())
        # Filter out patch_embed if we already froze it or it's not a stage
        stages = [c for c in children if not isinstance(c, (nn.Identity, nn.Dropout))]
        
        # Heuristic: assume stages are the main computational blocks
        # For Swin, it might be 'layers'. For ResNet, 'layer1', 'layer2', etc.
        # We'll try to freeze the first `freeze_stages` children that look like stages.
        
        frozen_count = 0
        for i, child in enumerate(stages):
            if frozen_count >= freeze_stages:
                break
            # Skip initial conv/stem if it's considered stage 0
            # Often stem is handled separately, but let's include it in freezing count
            for param in child.parameters():
                param.requires_grad = False
            frozen_count += 1

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = list(self.model(x))
        # Check if output is NHWC (Swin/ViT often are) and convert to NCHW if so
        # Heuristic: if last dim is channels (from metadata)
        out = []
        for i, feat in enumerate(features):
            if feat.dim() == 4:
                # If channels is the last dimension, permute
                # We can check against expected channels from metadata if available, 
                # but simplistic check: if shape[1] != self._channels[i] and shape[-1] == self._channels[i]
                if feat.shape[1] != self._channels[i] and feat.shape[-1] == self._channels[i]:
                    feat = feat.permute(0, 3, 1, 2)
            out.append(feat)
        return out

    def get_output_channels(self) -> List[int]:
        return list(self._channels)

    def get_reduction_factors(self) -> List[int]:
        return list(self._reductions)

    def metadata(self) -> BackboneMetadata:
        return BackboneMetadata(
            backbone_type="timm",
            variant=self.model_name,
            num_scales=len(self._channels),
            channels=self.get_output_channels(),
            reductions=self.get_reduction_factors(),
            input_size=self._input_size,
        )


class SimpleMultiScaleBackbone(BackboneAdapter):
    """Pure PyTorch fallback that returns a pyramid of feature maps."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64, num_stages: int = 4):
        super().__init__()
        self.stages = nn.ModuleList()
        self._channels: List[int] = []
        self._reductions: List[int] = []
        c_in = in_channels
        c_out = base_channels
        
        for idx in range(num_stages):
            block = nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.GELU(),
                nn.Conv2d(c_out, c_out, 3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(c_out),
                nn.GELU(),
            )
            self.stages.append(block)
            self._channels.append(c_out)
            self._reductions.append(2 ** (idx + 1))
            c_in = c_out
            if idx < num_stages - 1:
                c_out = min(c_out * 2, 512)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        current = x
        for stage in self.stages:
            current = stage(current)
            outputs.append(current)
        return outputs

    def get_output_channels(self) -> List[int]:
        return list(self._channels)

    def get_reduction_factors(self) -> List[int]:
        return list(self._reductions)

    def metadata(self) -> BackboneMetadata:
        return BackboneMetadata(
            backbone_type="simple",
            variant="cnn",
            num_scales=len(self._channels),
            channels=self.get_output_channels(),
            reductions=self.get_reduction_factors(),
            input_size=0,
        )


def _import_swin_feature_extractor():
    from .swin_tm import SwinFeatureExtractor

    return SwinFeatureExtractor


class SwinBackboneAdapter(BackboneAdapter):
    """Adapter around the project-native SwinFeatureExtractor."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.variant = kwargs.get("variant", "tiny")
        self._input_size = kwargs.get("input_size", 224)
        extractor_cls = _import_swin_feature_extractor()
        self.extractor = extractor_cls(**kwargs)
        feature_info = getattr(self.extractor.backbone, "feature_info", None)
        if feature_info is not None and hasattr(feature_info, "reduction"):
            self._reductions = list(feature_info.reduction())
        else:
            num_scales = len(self.extractor.get_output_channels())
            self._reductions = [4 * (2 ** idx) for idx in range(num_scales)]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.extractor(x)

    def get_output_channels(self) -> List[int]:
        return self.extractor.get_output_channels()

    def get_reduction_factors(self) -> List[int]:
        return list(self._reductions[: len(self.get_output_channels())])

    def metadata(self) -> BackboneMetadata:
        return BackboneMetadata(
            backbone_type="swin",
            variant=self.variant,
            num_scales=len(self.get_output_channels()),
            channels=self.get_output_channels(),
            reductions=self.get_reduction_factors(),
            input_size=self._input_size,
        )


class UniversalBackboneFactory:
    """Factory object that builds backbone adapters from a simple spec."""

    SWIN_VARIANTS = {"tiny", "small", "base", "large"}
    RESNET_VARIANTS = {"18", "34", "50", "101", "152"}
    CONVNEXT_VARIANTS = {"tiny", "small", "base", "large"}
    EFFICIENTNET_VARIANTS = {"b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"}

    @classmethod
    def _resolve_timm_name(cls, backbone_type: str, variant: str) -> str:
        backbone_type = backbone_type.lower()
        variant = variant.lower()
        if backbone_type == "resnet":
            return f"resnet{variant}"
        if backbone_type == "convnext":
            return f"convnext_{variant}"
        if backbone_type == "efficientnet":
            return f"efficientnet_{variant}"
        if backbone_type == "mobilenet":
            suffix = "" if variant.endswith("100") else f"_{variant}"
            return f"mobilenetv3_large{suffix}"
        return variant

    @classmethod
    def create(
        cls,
        backbone_type: str = "swin",
        backbone_variant: str = "tiny",
        *,
        num_scales: int = 4,
        pretrained: bool = True,
        input_size: int = 224,
        freeze_stages: int = 0,
        **kwargs: Any,
    ) -> BackboneAdapter:
        backbone_type = backbone_type.lower()
        
        # Use TimmFeatureBackbone for Swin if available and requested via specific kwargs or default
        # But for backward compatibility we might want to keep SwinBackboneAdapter if 'use_timm' isn't explicitly set
        # Actually, let's prefer TimmFeatureBackbone for consistency if we are refactoring.
        # SwinBackboneAdapter uses 'swin_tm' which might be our custom implementation.
        # The plan implies standardizing.
        
        if backbone_type == "swin" and kwargs.get("use_custom_swin", False):
             if backbone_variant not in cls.SWIN_VARIANTS:
                raise ValueError(
                    f"Unsupported Swin variant '{backbone_variant}'. "
                    f"Choose from {sorted(cls.SWIN_VARIANTS)}."
                )
             return SwinBackboneAdapter(
                variant=backbone_variant,
                num_scales=num_scales,
                pretrained=pretrained,
                input_size=input_size,
                freeze_stages=freeze_stages,
                **kwargs,
            )

        if backbone_type in {"swin", "resnet", "convnext", "efficientnet", "mobilenet", "vit", "timm"}:
            # Resolve model name for timm
            if backbone_type == "swin":
                 # Map swin variants to timm names
                 # e.g. swin_tiny_patch4_window7_224
                 window_size = kwargs.get("window_size", 7)
                 patch_size = kwargs.get("patch_size", 4)
                 model_name = f"swin_{backbone_variant}_patch{patch_size}_window{window_size}_{input_size}"
                 # Fallback/Override logic
                 if "timm_model_name" in kwargs:
                     model_name = kwargs.pop("timm_model_name")
            else:
                model_name = kwargs.pop(
                    "timm_model_name", cls._resolve_timm_name(backbone_type, backbone_variant)
                )
            
            return TimmFeatureBackbone(
                model_name=model_name,
                num_scales=num_scales,
                pretrained=pretrained,
                input_size=input_size,
                freeze_stages=freeze_stages,
                **kwargs,
            )

        if backbone_type in {"simple", "cnn"}:
            return SimpleMultiScaleBackbone(
                in_channels=kwargs.get("in_channels", 3),
                base_channels=kwargs.get("base_channels", 64),
                num_stages=num_scales,
            )

        raise ValueError(
            f"Unknown backbone type '{backbone_type}'. "
            "Supported types: swin, resnet, convnext, efficientnet, mobilenet, vit, simple."
        )


# ---------------------------------------------------------------------------
# Backwards compatible wrappers (kept for existing imports)
# ---------------------------------------------------------------------------


class PretrainedBackbone(nn.Module):
    """Generic pretrained backbone using timm (single-scale pooled output)."""

    SMALL_MODELS = {
        "convnext_tiny": 768,
        "convnext_small": 768,
        "efficientnet_b0": 1280,
        "efficientnet_b1": 1280,
        "mobilenetv3_large_100": 960,
        "resnet18": 512,
        "resnet34": 512,
        "resnet50": 2048,
    }

    def __init__(
        self,
        model_name: str = "convnext_tiny",
        pretrained: bool = True,
        freeze: bool = False,
        freeze_bn: bool = True,
    ):
        super().__init__()
        if not HAS_TIMM:
            raise ImportError("timm required: pip install timm")

        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.out_dim = self.backbone.num_features

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if freeze_bn:
            for m in self.backbone.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class HybridTMWithBackbone(nn.Module):
    """TM classifier on pretrained flattened features."""

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        n_clauses: int = 512,
        n_classes: int = 10,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        operator: str = "capacity",
    ):
        super().__init__()
        self.backbone = PretrainedBackbone(backbone, pretrained=pretrained, freeze=freeze_backbone)

        self.tm_head = FuzzyPatternTM_STCM(
            n_features=self.backbone.out_dim,
            n_clauses=n_clauses,
            n_classes=n_classes,
            operator=operator,
        )

    def forward(self, x: torch.Tensor, use_ste: bool = True):
        features = self.backbone(x)
        return self.tm_head(features, use_ste=use_ste)


class MultiScaleBackbone(nn.Module):
    """Legacy helper that exposes the channels produced by a timm feature extractor."""

    def __init__(self, model_name: str = "convnext_tiny", pretrained: bool = True, num_scales: int = 4):
        super().__init__()
        self.adapter = TimmFeatureBackbone(model_name, num_scales=num_scales, pretrained=pretrained)
        self.out_channels = self.adapter.get_output_channels()

    def forward(self, x: torch.Tensor):
        return self.adapter(x)


class SimpleCNNBackbone(nn.Module):
    """Simple CNN backbone for when timm is not available (single pooled output)."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64, num_stages: int = 4):
        super().__init__()

        layers = []
        c_in = in_channels
        c_out = base_channels

        for i in range(num_stages):
            layers.extend(
                [
                    nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                    nn.BatchNorm2d(c_out),
                    nn.GELU(),
                    nn.Conv2d(c_out, c_out, 3, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(c_out),
                    nn.GELU(),
                ]
            )
            c_in = c_out
            if i < num_stages - 1:
                c_out = min(c_out * 2, 512)

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = c_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)


class HybridTMSimple(nn.Module):
    """TM with simple CNN backbone (no timm dependency)."""

    def __init__(self, n_classes: int = 10, n_clauses: int = 512, base_channels: int = 64):
        super().__init__()
        self.backbone = SimpleCNNBackbone(base_channels=base_channels)
        self.tm_head = FuzzyPatternTM_STCM(
            n_features=self.backbone.out_dim,
            n_clauses=n_clauses,
            n_classes=n_classes,
        )

    def forward(self, x: torch.Tensor, use_ste: bool = True):
        features = self.backbone(x)
        return self.tm_head(features, use_ste=use_ste)
