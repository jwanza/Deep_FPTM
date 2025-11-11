"""
Dataset preparation utilities for FPTM Python tooling.

Currently exposes the Fashion-MNIST augmentation + booleanisation pipeline
necessary to mirror the Julia examples.
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

__all__ = [
    "AugmentationRecipe",
    "FashionAugmentedBundle",
    "PackedBooleanDataset",
    "PreprocessConfig",
    "DEFAULT_PREPROCESS_CONFIGS",
    "prepare_boolean_feature_bundle",
    "prepare_fashion_augmented_bundle",
]


