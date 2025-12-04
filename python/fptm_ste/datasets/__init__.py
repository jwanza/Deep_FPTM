"""
Dataset preparation utilities for FPTM Python tooling.

Exposes the Fashion-MNIST augmentation + booleanisation pipeline necessary to
mirror the Julia examples as well as backbone-aligned torchvision transforms
for hybrid models.
"""

from .fashion_augmented import (  # noqa: F401
    AugmentationRecipe,
    FashionAugmentedBundle,
    PackedBooleanDataset,
    PreprocessConfig,
    DEFAULT_PREPROCESS_CONFIGS,
    prepare_boolean_feature_bundle,
    prepare_fashion_augmented_bundle,
)
from .transforms import (  # noqa: F401
    BackboneTransformBundle,
    build_backbone_transforms,
    build_cutmix,
    build_mixup,
)

__all__ = [
    "AugmentationRecipe",
    "FashionAugmentedBundle",
    "PackedBooleanDataset",
    "PreprocessConfig",
    "DEFAULT_PREPROCESS_CONFIGS",
    "prepare_boolean_feature_bundle",
    "prepare_fashion_augmented_bundle",
    "BackboneTransformBundle",
    "build_backbone_transforms",
    "build_mixup",
    "build_cutmix",
]

