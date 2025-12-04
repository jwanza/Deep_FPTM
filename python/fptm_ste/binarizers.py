from __future__ import annotations

from .booleanization.learnable import LearnableBinarizer


class SwinDualBinarizer(LearnableBinarizer):
    """
    Dual-sigmoid learnable binarizer for transformer backbones with
    zero-centred activations.
    """

    def __init__(
        self,
        in_channels: int,
        num_thresholds: int = 16,
        init_temperature: float = 1.0,
        backbone_type: str = "swin",
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            num_thresholds=num_thresholds,
            init_temperature=init_temperature,
            mode="dual",
        )
        self.backbone_type = backbone_type


class CNNSingleBinarizer(LearnableBinarizer):
    """
    Single-sigmoid learnable binarizer for non-negative CNN feature maps.
    """

    def __init__(
        self,
        in_channels: int,
        num_thresholds: int = 16,
        init_temperature: float = 1.0,
        backbone_type: str = "cnn",
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            num_thresholds=num_thresholds,
            init_temperature=init_temperature,
            mode="single",
        )
        self.backbone_type = backbone_type


__all__ = ["SwinDualBinarizer", "CNNSingleBinarizer", "LearnableBinarizer"]
