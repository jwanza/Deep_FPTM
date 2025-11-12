"""
MNIST equivalence runner with profiling and accelerated training.
- Trains STE TM, Deep TM, CNN+TM Hybrid, and TM-Transformer (5 epochs by default)
- Uses fuzzy (float) inputs for learning stability
- Skips regenerating boolean datasets if cached
- Exports compiled literals to JSON for Julia bridge
- Supports MNIST, Fashion-MNIST, and CIFAR-10 dataset options
"""

import argparse
import json
import os
import resource
import time
import math
import copy
from collections import defaultdict
from dataclasses import replace
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from fptm_ste import FuzzyPatternTM_STE, FuzzyPatternTMFPTM
from fptm_ste.binarizers import CNNSingleBinarizer
from fptm_ste.deep_tm import DeepTMNetwork
from fptm_ste.export import export_compiled_to_json
from fptm_ste.tm_transformer import UnifiedTMTransformer
from fptm_ste.datasets import (
    DEFAULT_PREPROCESS_CONFIGS,
    PreprocessConfig,
    prepare_boolean_feature_bundle,
    prepare_fashion_augmented_bundle,
)

DEFAULT_MODELS = ("tm", "deep_tm", "hybrid", "transformer")
EXPORT_SLUGS = {
    "tm": "tm",
    "deep_tm": "deeptm",
    "hybrid": "hybrid",
    "transformer": "transformer",
}
DEFAULT_DATA_ROOT = os.environ.get("TM_DATA_ROOT", os.environ.get("TM_MNIST_ROOT", "/tmp/mnist"))
DEFAULT_BOOL_TRAIN_PATH = os.environ.get("TM_BOOL_TRAIN_PATH", "/tmp/MNISTTrainingData.txt")
DEFAULT_BOOL_TEST_PATH = os.environ.get("TM_BOOL_TEST_PATH", "/tmp/MNISTTestData.txt")
DEFAULT_OUTPUT_JSON = os.environ.get("TM_OUTPUT_JSON", "/tmp/mnist_equiv_results.json")
DEFAULT_EXPORT_PREFIX = os.environ.get("TM_EXPORT_PREFIX", "tm_mnist")

DATASET_CONFIGS = {
    "mnist": {
        "train_class": datasets.MNIST,
        "test_class": datasets.MNIST,
        "transform": transforms.ToTensor(),
        "input_channels": 1,
        "image_size": (28, 28),
        "num_classes": 10,
    },
    "fashionmnist": {
        "train_class": datasets.FashionMNIST,
        "test_class": datasets.FashionMNIST,
        "transform": transforms.ToTensor(),
        "input_channels": 1,
        "image_size": (28, 28),
        "num_classes": 10,
    },
    "cifar10": {
        "train_class": datasets.CIFAR10,
        "test_class": datasets.CIFAR10,
        "transform": transforms.ToTensor(),
        "input_channels": 3,
        "image_size": (32, 32),
        "num_classes": 10,
    },
}

BASELINE_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "mnist_vit_patch4": {
        "description": "MNIST ViT baseline with STE TM feed-forward (quick smoke test).",
        "overrides": {
            "dataset": "mnist",
            "models": ["transformer"],
            "seed": 1337,
            "epochs": 3,
            "batch_size": 64,
            "test_batch_size": 128,
            "transformer_arch": "vit",
            "transformer_patch": 4,
            "transformer_d_model": 64,
            "transformer_layers": 2,
            "transformer_heads": 4,
            "transformer_mlp_ratio": "3.0",
            "transformer_backend": "ste",
            "transformer_ff": 256,
            "transformer_drop_path": 0.0,
            "transformer_dropout": 0.05,
            "transformer_aux_weight": 0.0,
            "randaugment": False,
            "mixup_alpha": 0.0,
            "cutmix_alpha": 0.0,
        },
    },
    "mnist_swin_window4": {
        "description": "MNIST Swin baseline with STE TM feed-forward (window=4).",
        "overrides": {
            "dataset": "mnist",
            "models": ["transformer"],
            "seed": 2024,
            "epochs": 3,
            "batch_size": 64,
            "test_batch_size": 128,
            "transformer_arch": "swin",
            "transformer_patch": 2,
            "transformer_depths": "1,1",
            "transformer_stage_heads": "2,4",
            "transformer_embed_dims": "48,96",
            "transformer_mlp_ratio": "3.0,3.0",
            "transformer_backend": "ste",
            "transformer_window": 4,
            "transformer_drop_path": 0.0,
            "transformer_dropout": 0.05,
            "transformer_aux_weight": 0.0,
            "randaugment": False,
            "mixup_alpha": 0.0,
            "cutmix_alpha": 0.0,
        },
    },
    "cifar10_swin_mini": {
        "description": "CIFAR-10 Swin mini-run with DeepTM feed-forward (diagnostic baseline).",
        "overrides": {
            "dataset": "cifar10",
            "models": ["transformer"],
            "seed": 42,
            "epochs": 2,
            "batch_size": 64,
            "test_batch_size": 128,
            "transformer_arch": "swin",
            "transformer_patch": 4,
            "transformer_depths": "2,2",
            "transformer_stage_heads": "3,6",
            "transformer_embed_dims": "96,192",
            "transformer_mlp_ratio": "3.0,3.0",
            "transformer_backend": "deeptm",
            "transformer_window": 4,
            "transformer_drop_path": 0.1,
            "transformer_dropout": 0.1,
            "transformer_aux_weight": 0.05,
            "randaugment": True,
            "randaugment_n": 1,
            "randaugment_m": 7,
            "mixup_alpha": 0.4,
            "cutmix_alpha": 0.5,
            "warmup_epochs": 2,
        },
    },
}


def bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def int_or_none(value: Optional[str], default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    value = value.strip()
    if not value:
        return default
    return int(value)


def float_or_none(value: Optional[str], default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    value = value.strip()
    if not value:
        return default
    return float(value)


def apply_gradient_centralization(module: nn.Module) -> None:
    for param in module.parameters():
        if param.grad is None:
            continue
        if param.grad.dim() > 1:
            dims = tuple(range(1, param.grad.dim()))
            mean = param.grad.mean(dim=dims, keepdim=True)
            param.grad.sub_(mean)


def build_param_groups_with_layer_decay(model: nn.Module, base_lr: float, weight_decay: float, layer_decay: float) -> List[Dict[str, Any]]:
    if layer_decay is None or layer_decay == 1.0 or not hasattr(model, "layer_param_groups"):
        return [{"params": list(model.parameters()), "lr": base_lr, "weight_decay": weight_decay}]
    param_groups: List[Dict[str, Any]] = []
    layer_groups = model.layer_param_groups()
    total_layers = len(layer_groups)
    for layer_id, params in enumerate(layer_groups):
        scale = layer_decay ** (total_layers - layer_id - 1)
        group_lr = base_lr * scale
        param_groups.append({
            "params": params,
            "lr": group_lr,
            "weight_decay": weight_decay,
        })
    return param_groups


def optional_int_arg(value: str) -> Optional[int]:
    value = value.strip().lower()
    if value in {"none", "null", ""}:
        return None
    return int(value)


def optional_float_arg(value: str) -> Optional[float]:
    value = value.strip().lower()
    if value in {"none", "null", ""}:
        return None
    return float(value)


def parse_int_list(value: str) -> List[int]:
    parts = [p.strip() for p in value.split(",")]
    return [int(p) for p in parts if p]


def parse_float_list(value: str) -> List[float]:
    parts = [p.strip() for p in value.split(",")]
    return [float(p) for p in parts if p]


def maybe_set_seed(seed: Optional[int]):
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MNIST equivalence runner with configurable variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=DEFAULT_MODELS,
        default=None,
        help="Subset of model variants to train. Defaults to all variants.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_CONFIGS.keys()),
        default=os.environ.get("TM_DATASET", "mnist"),
        help="Dataset to use for training and evaluation.",
    )
    parser.add_argument("--epochs", type=int, default=int_env("TM_MNIST_EPOCHS", 25), help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=int_env("TM_MNIST_BATCH", 128), help="Training batch size.")
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=int_env("TM_MNIST_TEST_BATCH", 512),
        help="Evaluation batch size.",
    )
    parser.add_argument("--lr", type=float, default=None, help="Base learning rate for AdamW optimizers.")
    parser.add_argument("--min-lr", type=float, default=None, help="Minimum learning rate for cosine decay.")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Number of warmup epochs before cosine decay.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay value for AdamW optimizers.")
    parser.add_argument("--lr-layer-decay", type=float, default=1.0, help="Layer-wise LR decay factor (set <1 to scale deeper layers).")
    parser.add_argument("--gradient-centralize", action=argparse.BooleanOptionalAction, default=False, help="Apply gradient centralization before optimizer step.")
    parser.add_argument("--num-classes", type=int, default=None, help="Number of target classes.")
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATA_ROOT,
        help="Root directory where the selected dataset will be stored.",
    )
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow torchvision to download the dataset if missing.",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker processes for DataLoaders.")
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override automatic pin_memory selection.",
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override automatic persistent_workers selection.",
    )
    parser.add_argument(
        "--write-bool-dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Toggle writing cached boolean dataset dumps to disk.",
    )
    parser.add_argument(
        "--bool-threshold",
        type=float,
        default=0.5,
        help="Threshold applied when binarizing input tensors.",
    )
    parser.add_argument(
        "--bool-train-path",
        default=DEFAULT_BOOL_TRAIN_PATH,
        help="File path for the boolean training dataset export.",
    )
    parser.add_argument(
        "--bool-test-path",
        default=DEFAULT_BOOL_TEST_PATH,
        help="File path for the boolean test dataset export.",
    )
    parser.add_argument(
        "--output-json",
        default=DEFAULT_OUTPUT_JSON,
        help="Path to the metrics summary JSON file.",
    )
    parser.add_argument(
        "--export-compiled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export compiled literals for models that support discretization.",
    )
    parser.add_argument(
        "--export-prefix",
        default=DEFAULT_EXPORT_PREFIX,
        help="Prefix used for compiled literal artifact filenames.",
    )
    parser.add_argument(
        "--export-dir",
        default="/tmp",
        help="Directory where compiled literal artifacts will be stored.",
    )
    parser.add_argument(
        "--report-train-acc",
        action=argparse.BooleanOptionalAction,
        default=bool_env("TM_MNIST_REPORT_TRAIN", True),
        help="Log running training accuracy during epochs.",
    )
    parser.add_argument(
        "--report-epoch-acc",
        action=argparse.BooleanOptionalAction,
        default=bool_env("TM_MNIST_REPORT_EPOCH", True),
        help="Include training accuracy in epoch summary lines.",
    )
    parser.add_argument(
        "--report-epoch-test",
        action=argparse.BooleanOptionalAction,
        default=bool_env("TM_MNIST_REPORT_EPOCH_TEST", True),
        help="Evaluate on the test set and log test accuracy after each epoch.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default=os.environ.get("TM_MNIST_DEVICE", "auto"),
        help="Preferred compute device.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for torch.",
    )
    parser.add_argument(
        "--baseline-scenarios",
        nargs="+",
        choices=sorted(BASELINE_SCENARIOS.keys()),
        default=None,
        help="Run predefined baseline scenario(s) prior to the main invocation.",
    )
    parser.add_argument(
        "--baseline-output-dir",
        default=None,
        help="Directory for baseline metrics (defaults to derived path under output-json).",
    )
    parser.add_argument(
        "--baseline-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only execute baseline scenario(s) and skip the main command.",
    )
    parser.add_argument(
        "--list-baselines",
        action="store_true",
        help="List available baseline scenarios and exit.",
    )

    # TM specific options
    parser.add_argument(
        "--tm-impl",
        choices=["ste", "fptm"],
        default=os.environ.get("TM_MNIST_TM_IMPL", "ste"),
        help="Implementation for the TM baseline.",
    )
    parser.add_argument(
        "--tm-feature-mode",
        choices=["raw", "fashion_aug", "conv"],
        default=os.environ.get("TM_MNIST_FEATURE_MODE", "raw"),
        help="Feature preprocessing pipeline for TM variants.",
    )
    parser.add_argument(
        "--tm-feature-cache",
        default=os.environ.get("TM_FEATURE_CACHE_DIR", "/tmp/fptm_features"),
        help="Directory used to cache booleanised TM feature datasets.",
    )
    parser.add_argument(
        "--tm-feature-config",
        default=None,
        help="Name of predefined TM feature preprocessing config (defaults to dataset).",
    )
    parser.add_argument(
        "--tm-feature-size",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Override TM feature preprocessing target size.",
    )
    parser.add_argument(
        "--tm-feature-grayscale",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force grayscale conversion in TM feature preprocessing.",
    )
    parser.add_argument(
        "--tm-feature-batch",
        type=int,
        default=int_env("TM_FEATURE_BATCH", 256),
        help="Batch size used when materialising TM feature caches.",
    )
    parser.add_argument(
        "--tm-feature-force",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force regeneration of cached TM feature datasets.",
    )
    parser.add_argument(
        "--tm-augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable geometric augmentations when preparing TM feature datasets (Fashion-MNIST only).",
    )
    parser.add_argument(
        "--tm-tau",
        type=float,
        default=float_env("TM_MNIST_TAU", 0.4),
        help="Softmax temperature parameter for the TM variants.",
    )
    parser.add_argument(
        "--tm-lf",
        type=int,
        default=int_env("TM_MNIST_LF", 4),
        help="Lookahead parameter lf for FPTM.",
    )
    parser.add_argument(
        "--tm-literal-budget",
        type=optional_int_arg,
        default=int_or_none(os.environ.get("TM_MNIST_L")),
        help="Optional literal budget for FPTM (set to 'none' to disable).",
    )
    parser.add_argument(
        "--tm-vote-clamp",
        type=optional_float_arg,
        default=float_or_none(os.environ.get("TM_MNIST_VOTE_CLAMP")),
        help="Optional vote clamp for FPTM (set to 'none' to disable).",
    )
    parser.add_argument("--tm-n-clauses", type=int, default=256, help="Number of clauses for TM variants.")
    parser.add_argument(
        "--tm-L",
        type=int,
        default=None,
        help="Literal budget L embedded into exported TM JSON metadata (defaults to --tm-literal-budget or 16).",
    )
    parser.add_argument(
        "--tm-LF",
        type=int,
        default=None,
        help="LF parameter embedded into exported TM JSON metadata (defaults to --tm-lf).",
    )
    parser.add_argument(
        "--tm-target-channels",
        type=optional_int_arg,
        default=None,
        help="Override the logical TM input channel count (default: dataset channels).",
    )
    parser.add_argument(
        "--tm-target-size",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Override the logical TM input spatial size (default: dataset image size).",
    )
    parser.add_argument(
        "--tm-auto-expand-grayscale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically expand single-channel inputs to the requested channel count.",
    )
    parser.add_argument(
        "--tm-allow-channel-reduce",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow collapsing multi-channel inputs to single-channel if requested.",
    )

    # Deep TM options
    parser.add_argument(
        "--deeptm-hidden-dims",
        default="256,128",
        help="Comma separated list of hidden dimensions for Deep TM.",
    )
    parser.add_argument("--deeptm-n-clauses", type=int, default=256, help="Number of clauses for Deep TM.")
    parser.add_argument("--deeptm-dropout", type=float, default=0.1, help="Dropout for Deep TM network.")
    parser.add_argument("--deeptm-tau", type=float, default=0.5, help="Tau for Deep TM classifier.")

    # Hybrid options
    parser.add_argument("--hybrid-thresholds", type=int, default=32, help="Number of thresholds for CNN binarizer.")
    parser.add_argument("--hybrid-clauses", type=int, default=384, help="Number of clauses in hybrid TM head.")
    parser.add_argument(
        "--hybrid-classes",
        type=optional_int_arg,
        default=None,
        help="Optional override for hybrid output classes (defaults to --num-classes).",
    )

    # Transformer options
    parser.add_argument("--transformer-vocab", type=int, default=256, help="Token vocabulary size for transformer.")
    parser.add_argument("--transformer-d-model", type=int, default=96, help="Model dimension for transformer.")
    parser.add_argument("--transformer-heads", type=int, default=3, help="Attention heads in transformer.")
    parser.add_argument("--transformer-layers", type=int, default=2, help="Number of transformer layers.")
    parser.add_argument(
        "--transformer-clauses",
        type=int,
        default=192,
        help="Number of clauses used in transformer TM heads.",
    )
    parser.add_argument(
        "--transformer-arch",
        choices=["vit", "swin"],
        default="vit",
        help="Top-level transformer architecture to instantiate.",
    )
    parser.add_argument(
        "--transformer-patch",
        type=int,
        default=4,
        help="Patch size for vision transformer variants.",
    )
    parser.add_argument(
        "--transformer-depths",
        default=None,
        help="Comma separated stage depths (Swin) or integer for ViT.",
    )
    parser.add_argument(
        "--transformer-stage-heads",
        default=None,
        help="Comma separated attention heads per stage (Swin only).",
    )
    parser.add_argument(
        "--transformer-embed-dims",
        default=None,
        help="Comma separated embed dimensions per stage (Swin only).",
    )
    parser.add_argument(
        "--transformer-mlp-ratio",
        default="4.0",
        help="MLP ratio (float or comma separated per stage).",
    )
    parser.add_argument(
        "--transformer-window",
        type=int,
        default=7,
        help="Window size for Swin-style attention.",
    )
    parser.add_argument(
        "--transformer-drop-path",
        type=float,
        default=0.1,
        help="Maximum drop-path rate across transformer layers.",
    )
    parser.add_argument(
        "--transformer-use-cls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CLS token for ViT pooling.",
    )
    parser.add_argument(
        "--transformer-pool",
        choices=["cls", "mean"],
        default="cls",
        help="Pooling strategy for ViT outputs.",
    )
    parser.add_argument(
        "--transformer-backend",
        choices=["ste", "deeptm"],
        default="ste",
        help="Choose TM backend for transformer feed-forward replacements.",
    )
    parser.add_argument(
        "--transformer-ff",
        type=int,
        default=512,
        help="Hidden dimension used inside TM-based feed-forward replacements.",
    )
    parser.add_argument(
        "--transformer-ff-gate",
        choices=["none", "linear", "geglu", "swiglu", "tm", "deeptm"],
        default="none",
        help="Gate style applied inside TM feed-forward blocks (e.g., linear/GEGLU/SwiGLU/TM-based).",
    )
    parser.add_argument(
        "--transformer-ff-gate-activation",
        choices=["sigmoid", "tanh", "relu"],
        default="sigmoid",
        help="Activation applied to feed-forward gates when enabled.",
    )
    parser.add_argument(
        "--transformer-layerscale-init",
        type=float,
        default=1e-4,
        help="Initial value for LayerScale multipliers applied to TM residual branches.",
    )
    parser.add_argument(
        "--transformer-use-layerscale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable learnable LayerScale multipliers on transformer residual paths.",
    )
    parser.add_argument(
        "--transformer-clause-drop",
        type=float,
        default=0.0,
        help="Clause dropout probability applied within TM feed-forward modules.",
    )
    parser.add_argument(
        "--transformer-literal-drop",
        type=float,
        default=0.0,
        help="Literal dropout probability applied to TM inputs before clause evaluation.",
    )
    parser.add_argument(
        "--transformer-sparsity-weight",
        type=float,
        default=0.0,
        help="L1 penalty weight applied to TM clause activations.",
    )
    parser.add_argument(
        "--transformer-clause-bias",
        type=float,
        default=0.0,
        help="Initial additive bias applied to TM clause outputs.",
    )
    parser.add_argument(
        "--transformer-norm",
        choices=["layernorm", "rmsnorm", "scalenorm"],
        default="layernorm",
        help="Normalization applied inside TM transformer blocks.",
    )
    parser.add_argument(
        "--transformer-ff-mix",
        choices=["none", "linear", "depthwise", "linear_depthwise"],
        default="none",
        help="Optional continuous mixing applied around TM feed-forward layers.",
    )
    parser.add_argument(
        "--transformer-bitwise-mix",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable XOR-style bitwise mixing of clause inputs before TM evaluation.",
    )
    parser.add_argument(
        "--transformer-learnable-tau",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Make TM tau learnable within transformer feed-forward blocks.",
    )
    parser.add_argument(
        "--transformer-tau-min",
        type=float,
        default=0.05,
        help="Lower bound for learnable TM tau.",
    )
    parser.add_argument(
        "--transformer-tau-max",
        type=float,
        default=0.95,
        help="Upper bound for learnable TM tau.",
    )
    parser.add_argument(
        "--transformer-tau-ema",
        type=float,
        default=None,
        help="Optional EMA beta for smoothing tau updates (set between 0 and 1).",
    )
    parser.add_argument(
        "--transformer-clause-attention",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable clause-level attention projections inside TM feed-forwards.",
    )
    parser.add_argument(
        "--transformer-clause-routing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable neuro-symbolic clause gating of feed-forward outputs.",
    )
    parser.add_argument(
        "--transformer-bypass",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add a continuous MLP bypass alongside TM outputs.",
    )
    parser.add_argument(
        "--transformer-bypass-scale",
        type=float,
        default=1.0,
        help="Initial gain applied to the continuous bypass pathway.",
    )
    parser.add_argument(
        "--transformer-dropout",
        type=float,
        default=0.1,
        help="Dropout probability for transformer attention and TM replacements.",
    )
    parser.add_argument(
        "--transformer-aux-weight",
        type=float,
        default=0.0,
        help="Auxiliary cross-entropy weight for transformer diagnostic heads.",
    )
    parser.add_argument(
        "--transformer-ema-decay",
        type=float,
        default=0.0,
        help="EMA decay for transformer weights (0 disables EMA).",
    )
    parser.add_argument(
        "--transformer-grad-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable gradient checkpointing inside transformer blocks.",
    )
    parser.add_argument(
        "--randaugment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply RandAugment to training images.",
    )
    parser.add_argument(
        "--randaugment-n",
        type=int,
        default=2,
        help="RandAugment number of transformations per image.",
    )
    parser.add_argument(
        "--randaugment-m",
        type=int,
        default=9,
        help="RandAugment magnitude (severity).",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.0,
        help="Mixup alpha value (0 disables Mixup).",
    )
    parser.add_argument(
        "--cutmix-alpha",
        type=float,
        default=0.0,
        help="CutMix alpha value (0 disables CutMix).",
    )

    return parser


def list_baseline_scenarios() -> None:
    print("Available baseline scenarios:")
    for name in sorted(BASELINE_SCENARIOS):
        info = BASELINE_SCENARIOS[name]
        description = info.get("description", "")
        print(f"  - {name}: {description}")


def build_baseline_args(
    base_args: argparse.Namespace,
    scenario_name: str,
    *,
    output_dir: Optional[str] = None,
) -> argparse.Namespace:
    if scenario_name not in BASELINE_SCENARIOS:
        raise KeyError(f"Unknown baseline scenario '{scenario_name}'.")
    scenario = BASELINE_SCENARIOS[scenario_name]
    overrides = scenario.get("overrides", {})
    scenario_args = copy.deepcopy(base_args)

    # Reset baseline-management flags to avoid recursion.
    scenario_args.baseline_scenarios = None
    scenario_args.baseline_output_dir = None
    scenario_args.baseline_only = False
    scenario_args.list_baselines = False

    for key, value in overrides.items():
        setattr(scenario_args, key, value)

    # Derive per-scenario JSON output path.
    default_output = getattr(base_args, "output_json", DEFAULT_OUTPUT_JSON)
    base_dir = output_dir or os.path.join(os.path.dirname(default_output) or ".", "baselines")
    os.makedirs(base_dir, exist_ok=True)
    scenario_args.output_json = os.path.join(base_dir, f"{scenario_name}.json")

    # Avoid export collisions when compiling TM artifacts.
    if hasattr(scenario_args, "export_prefix"):
        scenario_args.export_prefix = f"{scenario_name}_{scenario_args.export_prefix}"

    return scenario_args


def current_device(preferred: str = "auto") -> torch.device:
    preferred = (preferred or "auto").lower()
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    if preferred == "cuda":
        print("Requested CUDA device is unavailable; falling back to CPU.")
        return torch.device("cpu")
    if preferred == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "mps":
        print("Requested MPS device is unavailable; falling back to CPU.")
        return torch.device("cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@contextmanager
def profile_block(label: str):
    wall_start = time.time()
    cpu_start = resource.getrusage(resource.RUSAGE_SELF).ru_utime
    print(f"\n[{datetime.now():%H:%M:%S}] ▶ {label}")
    try:
        yield
    finally:
        wall_end = time.time()
        cpu_end = resource.getrusage(resource.RUSAGE_SELF).ru_utime
        print(f"[{datetime.now():%H:%M:%S}] ◀ {label} | wall {wall_end - wall_start:5.1f}s | cpu {cpu_end - cpu_start:5.1f}s")


def optimizer_lr(optimizer: Optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def set_optimizer_lr(optimizer: Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def cosine_warmup_lr(epoch: int, total_epochs: int, base_lr: float, min_lr: float, warmup_epochs: int) -> float:
    if total_epochs <= 0:
        return base_lr
    if warmup_epochs > 0 and epoch < warmup_epochs:
        warm_progress = float(epoch + 1) / float(max(1, warmup_epochs))
        return base_lr * warm_progress
    if total_epochs <= warmup_epochs:
        return min_lr
    progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def update_cosine_warmup_lr(
    optimizer: Optimizer,
    epoch: int,
    total_epochs: int,
    base_lr: float,
    min_lr: float,
    warmup_epochs: int,
) -> float:
    lr = cosine_warmup_lr(epoch, total_epochs, base_lr, min_lr, warmup_epochs)
    set_optimizer_lr(optimizer, lr)
    return lr


def format_epoch_log(
    label: str,
    epoch: int,
    total_epochs: int,
    *,
    loss: float,
    train_acc: Optional[float],
    test_acc: Optional[float],
    best_acc: Optional[float],
    lr: Optional[float],
    duration: Optional[float],
    extras: Optional[Dict[str, Union[int, float, str]]] = None,
) -> str:
    parts = [
        f"{label:12s}",
        f"epoch {epoch:02d}/{total_epochs:02d}",
        f"loss={loss:.4f}",
    ]
    if train_acc is not None:
        parts.append(f"train_acc={train_acc:.4f}")
    if test_acc is not None:
        parts.append(f"test_acc={test_acc:.4f}")
    if best_acc is not None:
        parts.append(f"best_acc={best_acc:.4f}")
    if lr is not None:
        parts.append(f"lr={lr:.5f}")
    if duration is not None:
        parts.append(f"{duration:4.1f}s")
    if extras:
        extras_str = ", ".join(f"{key}={value}" if isinstance(value, str) else f"{key}={value:.3f}" for key, value in extras.items())
        if extras_str:
            parts.append(extras_str)
    return " | ".join(parts)


def flatten_images(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size(0), -1)


def rand_bbox(size: torch.Size, lam: float, device: torch.device) -> Tuple[int, int, int, int]:
    B, C, H, W = size
    cut_rat = math.sqrt(1.0 - lam)
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)
    cy = torch.randint(0, H, (1,), device=device).item()
    cx = torch.randint(0, W, (1,), device=device).item()
    y1 = max(0, cy - cut_h // 2)
    y2 = min(H, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(W, cx + cut_w // 2)
    return y1, y2, x1, x2


def apply_mixup_cutmix(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mixup_alpha: float,
    cutmix_alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[float], str]:
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return inputs, targets, targets, None, "none"

    use_cuda = inputs.is_cuda
    if mixup_alpha > 0 and cutmix_alpha > 0:
        choose_mixup = torch.rand(1, device=inputs.device if use_cuda else None).item() < 0.5
    else:
        choose_mixup = mixup_alpha > 0

    perm = torch.randperm(inputs.size(0), device=inputs.device if use_cuda else None)
    if choose_mixup:
        lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
        mixed = lam * inputs + (1 - lam) * inputs[perm]
        return mixed, targets, targets[perm], lam, "mixup"

    lam = torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample().item()
    y1, y2, x1, x2 = rand_bbox(inputs.size(), lam, inputs.device)
    inputs_clone = inputs.clone()
    inputs_clone[:, :, y1:y2, x1:x2] = inputs[perm, :, y1:y2, x1:x2]
    lam = 1 - ((y2 - y1) * (x2 - x1) / (inputs.size(-1) * inputs.size(-2)))
    return inputs_clone, targets, targets[perm], lam, "cutmix"


def save_bool_dataset(train_loader, test_loader, train_path: str, test_path: str, thr: float = 0.5):
    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"  boolean datasets cached: {train_path}, {test_path}")
        return
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    with open(train_path, "w") as ft:
        for x, y in train_loader:
            xb = (x > thr).int().view(x.size(0), -1)
            for bits, label in zip(xb, y):
                ft.write(" ".join(str(int(v)) for v in bits.tolist()) + f" {int(label.item())}\n")
    with open(test_path, "w") as fe:
        for x, y in test_loader:
            xb = (x > thr).int().view(x.size(0), -1)
            for bits, label in zip(xb, y):
                fe.write(" ".join(str(int(v)) for v in bits.tolist()) + f" {int(label.item())}\n")


def evaluate_model(model: nn.Module,
                   prepare_fn: Callable[[torch.Tensor], torch.Tensor],
                   loader: DataLoader,
                   device: torch.device,
                   use_ste: bool,
                   *,
                   collect_preds: bool = True) -> Tuple[float, List[int]]:
    was_training = model.training
    model.eval()
    preds: List[int] = [] if collect_preds else []
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            xb = prepare_fn(x)
            outputs = model(xb, use_ste=use_ste)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            if hasattr(model, "pop_regularization_loss"):
                model.pop_regularization_loss()
            pred = logits.argmax(dim=1)
            if collect_preds: preds.extend(pred.cpu().tolist())
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total
    if was_training:
        model.train()
    return acc, preds


def evaluate_transformer_components(
    model: UnifiedTMTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    collect_preds: bool = True,
) -> Tuple[float, Dict[str, float], List[int]]:
    was_training = model.training
    model.eval()
    total_correct = 0
    total = 0
    component_correct: Dict[str, int] = defaultdict(int)
    component_total: Dict[str, int] = defaultdict(int)
    preds: List[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            outputs = model(x, use_ste=True, collect_diagnostics=True)
            if isinstance(outputs, tuple):
                logits, diagnostics = outputs
            else:
                logits, diagnostics = outputs, {}
            batch_preds = logits.argmax(dim=1)
            if collect_preds:
                preds.extend(batch_preds.cpu().tolist())
            total_correct += (batch_preds == y).sum().item()
            total += y.size(0)
            for name, comp_logits in diagnostics.items():
                comp_pred = comp_logits.argmax(dim=1)
                component_correct[name] += (comp_pred == y).sum().item()
                component_total[name] += y.size(0)
    if was_training:
        model.train()
    final_acc = total_correct / total if total > 0 else 0.0
    component_acc = {
        name: (component_correct[name] / component_total[name]) if component_total[name] > 0 else 0.0
        for name in sorted(component_correct)
    }
    return final_acc, component_acc, preds


def train_tm_model(model: nn.Module,
                   prepare_fn: Callable[[torch.Tensor], torch.Tensor],
                   train_loader: DataLoader,
                   test_loader: DataLoader,
                   device: torch.device,
                   label: str,
                   use_ste_train: bool,
                   use_ste_eval: bool,
                   epochs: int,
                   report_train_acc: bool,
                   report_epoch_acc: bool,
                   report_epoch_test: bool,
                   *,
                   base_lr: float,
                   min_lr: float,
                   warmup_epochs: int,
                   weight_decay: float,
                   gradient_centralize: bool = False) -> Tuple[Optional[float], float, float, List[int], Optional[float]]:
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    start = time.time()
    last_train_acc: Optional[float] = None
    best_test_acc: Optional[float] = None
    for epoch in range(epochs):
        epoch_lr = update_cosine_warmup_lr(opt, epoch, epochs, base_lr, min_lr, warmup_epochs)
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        num_batches = 0
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            xb = prepare_fn(x)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(xb, use_ste=use_ste_train)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if gradient_centralize:
                apply_gradient_centralization(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(opt)
            scaler.update()
            running_loss += loss.item()
            num_batches += 1
            if report_train_acc:
                preds = logits.argmax(dim=1)
                epoch_correct += (preds == y).sum().item()
                epoch_total += y.size(0)
        dur = time.time() - epoch_start
        avg_loss = running_loss / max(1, num_batches)
        epoch_acc = None
        epoch_test_acc = None
        if report_train_acc and epoch_total > 0:
            epoch_acc = epoch_correct / epoch_total
            last_train_acc = epoch_acc
        if report_epoch_test:
            epoch_test_acc, _ = evaluate_model(
                model,
                prepare_fn,
                test_loader,
                device,
                use_ste=use_ste_eval,
                collect_preds=False,
            )
            if epoch_test_acc is not None:
                best_test_acc = epoch_test_acc if best_test_acc is None else max(best_test_acc, epoch_test_acc)
        log_msg = format_epoch_log(
            label=label,
            epoch=epoch + 1,
            total_epochs=epochs,
            loss=avg_loss,
            train_acc=epoch_acc if report_epoch_acc else None,
            test_acc=epoch_test_acc,
            best_acc=best_test_acc,
            lr=epoch_lr,
            duration=dur,
            extras=None,
        )
        print("  " + log_msg)
    train_time = time.time() - start
    test_acc, preds = evaluate_model(model, prepare_fn, test_loader, device, use_ste=use_ste_eval)
    if best_test_acc is None or test_acc > best_test_acc:
        best_test_acc = test_acc
    return last_train_acc, test_acc, train_time, preds, best_test_acc


class CNNHybrid(nn.Module):
    """Small CNN backbone + learnable binarizer feeding a TM head."""

    def __init__(
        self,
        in_channels: int = 1,
        thresholds: int = 32,
        tm_clauses: int = 384,
        n_classes: int = 10,
        tm_tau: float = 0.5,
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.binarizer = CNNSingleBinarizer(in_channels=64, num_thresholds=thresholds, init_temperature=1.0)
        fused_dim = thresholds
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fused_dim, max(64, fused_dim)), nn.GELU(),
            nn.Linear(max(64, fused_dim), max(32, fused_dim // 2)), nn.Sigmoid(),
        )
        self.tm = FuzzyPatternTM_STE(max(32, fused_dim // 2), tm_clauses, n_classes, tau=tm_tau)

    def forward(self, x: torch.Tensor, use_ste: bool = True):
        feats = self.backbone(x)
        b = self.binarizer(feats, use_discrete=not self.training)
        b = F.adaptive_avg_pool2d(b, 1)
        proj = self.projector(b)
        logits, clauses = self.tm(proj, use_ste=use_ste)
        return logits, clauses


def run_variant_tm(train_loader,
                   test_loader,
                   device,
                   *,
                   epochs: int,
                   report_train_acc: bool,
                   report_epoch_acc: bool,
                   report_epoch_test: bool,
                   n_features: int,
                   prepare_fn: Callable[[torch.Tensor], torch.Tensor],
                   tm_impl: str,
                   tm_tau: float,
                   tm_lf: int,
                   tm_literal_budget: Optional[int],
                   tm_vote_clamp: Optional[float],
                   tm_n_clauses: int,
                   input_shape: Optional[Tuple[int, int, int]],
                   auto_expand_grayscale: bool,
                   allow_channel_reduce: bool,
                   n_classes: int = 10,
                   base_lr: float = 1e-3,
                   min_lr: float = 1e-3,
                   warmup_epochs: int = 0,
                   weight_decay: float = 1e-5,
                   gradient_centralize: bool = False) -> Tuple[str, Optional[float], float, float, List[int], Dict[str, Any]]:
    tm_impl = tm_impl.lower()
    if tm_impl == "fptm":
        model = FuzzyPatternTMFPTM(
            n_features=n_features,
            n_clauses=tm_n_clauses,
            n_classes=n_classes,
            tau=tm_tau,
            input_shape=input_shape,
            auto_expand_grayscale=auto_expand_grayscale,
            allow_channel_reduce=allow_channel_reduce,
            lf=tm_lf,
            literal_budget=tm_literal_budget,
            vote_clamp=tm_vote_clamp,
        ).to(device)
        use_ste_train = True
        label = "FPTM"
    else:
        model = FuzzyPatternTM_STE(
            n_features=n_features,
            n_clauses=tm_n_clauses,
            n_classes=n_classes,
            tau=tm_tau,
            input_shape=input_shape,
            auto_expand_grayscale=auto_expand_grayscale,
            allow_channel_reduce=allow_channel_reduce,
        ).to(device)
        use_ste_train = False
        label = "STE-TM"

    train_acc, test_acc, ttrain, preds, best_test_acc = train_tm_model(
        model,
        prepare_fn,
        train_loader,
        test_loader,
        device,
        label,
        use_ste_train=use_ste_train,
        use_ste_eval=True,
        epochs=epochs,
        report_train_acc=report_train_acc,
        report_epoch_acc=report_epoch_acc,
        report_epoch_test=report_epoch_test,
        base_lr=base_lr,
        min_lr=min_lr,
        warmup_epochs=warmup_epochs,
        weight_decay=weight_decay,
        gradient_centralize=gradient_centralize,
    )
    bundle = model.discretize(threshold=0.5)
    return label, train_acc, test_acc, ttrain, preds, bundle, best_test_acc


def run_variant_deeptm(train_loader,
                       test_loader,
                       device,
                       *,
                       epochs: int,
                       report_train_acc: bool,
                       report_epoch_acc: bool,
                       report_epoch_test: bool,
                       prepare_fn: Callable[[torch.Tensor], torch.Tensor],
                       input_dim: int,
                       hidden_dims: List[int],
                       n_clauses: int,
                       dropout: float,
                       tau: float,
                       input_shape: Optional[Tuple[int, int, int]],
                       auto_expand_grayscale: bool,
                       allow_channel_reduce: bool,
                       n_classes: int = 10,
                       base_lr: float = 1e-3,
                       min_lr: float = 1e-3,
                       warmup_epochs: int = 0,
                       weight_decay: float = 1e-5,
                       gradient_centralize: bool = False) -> Tuple[str, Optional[float], float, float, List[int], Dict[str, Any]]:
    model = DeepTMNetwork(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        n_classes=n_classes,
        n_clauses=n_clauses,
        dropout=dropout,
        tau=tau,
        input_shape=input_shape,
        auto_expand_grayscale=auto_expand_grayscale,
        allow_channel_reduce=allow_channel_reduce,
    ).to(device)
    label = "Deep-TM"
    train_acc, test_acc, ttrain, preds, best_test_acc = train_tm_model(
        model,
        prepare_fn,
        train_loader,
        test_loader,
        device,
        label,
        use_ste_train=False,
        use_ste_eval=True,
        epochs=epochs,
        report_train_acc=report_train_acc,
        report_epoch_acc=report_epoch_acc,
        report_epoch_test=report_epoch_test,
        base_lr=base_lr,
        min_lr=min_lr,
        warmup_epochs=warmup_epochs,
        weight_decay=weight_decay,
        gradient_centralize=gradient_centralize,
    )
    bundle = model.classifier.discretize(threshold=0.5)
    return label, train_acc, test_acc, ttrain, preds, bundle, best_test_acc


def run_variant_hybrid(train_loader,
                       test_loader,
                       device,
                       *,
                       epochs: int,
                       report_train_acc: bool,
                       report_epoch_acc: bool,
                       report_epoch_test: bool,
                       in_channels: int,
                       thresholds: int,
                       tm_clauses: int,
                       tm_tau: float,
                       n_classes: int = 10,
                       base_lr: float = 1e-3,
                       min_lr: float = 1e-3,
                       warmup_epochs: int = 0,
                       weight_decay: float = 1e-5) -> Tuple[str, Optional[float], float, float, List[int], Dict[str, Any]]:
    model = CNNHybrid(
        in_channels=in_channels,
        thresholds=thresholds,
        tm_clauses=tm_clauses,
        n_classes=n_classes,
        tm_tau=tm_tau,
    ).to(device)
    param_groups = build_param_groups_with_layer_decay(model, base_lr, weight_decay, layer_decay)
    opt = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    start = time.time()
    last_train_acc: Optional[float] = None
    best_test_acc: Optional[float] = None
    label = "Hybrid"
    for epoch in range(epochs):
        epoch_lr = update_cosine_warmup_lr(opt, epoch, epochs, base_lr, min_lr, warmup_epochs)
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        num_batches = 0
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits, _ = model(x, use_ste=False)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if gradient_centralize:
                apply_gradient_centralization(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(opt)
            scaler.update()
            running_loss += loss.item()
            num_batches += 1
            if report_train_acc:
                preds = logits.argmax(dim=1)
                epoch_correct += (preds == y).sum().item()
                epoch_total += y.size(0)
        dur = time.time() - epoch_start
        avg_loss = running_loss / max(1, num_batches)
        epoch_acc = None
        epoch_test_acc = None
        if report_train_acc and epoch_total > 0:
            epoch_acc = epoch_correct / epoch_total
            last_train_acc = epoch_acc
        if report_epoch_test:
            epoch_test_acc, _ = evaluate_model(
                model,
                lambda t: t,
                test_loader,
                device,
                use_ste=True,
                collect_preds=False,
            )
            if epoch_test_acc is not None:
                best_test_acc = epoch_test_acc if best_test_acc is None else max(best_test_acc, epoch_test_acc)
        log_msg = format_epoch_log(
            label=label,
            epoch=epoch + 1,
            total_epochs=epochs,
            loss=avg_loss,
            train_acc=epoch_acc if report_epoch_acc else None,
            test_acc=epoch_test_acc,
            best_acc=best_test_acc,
            lr=epoch_lr,
            duration=dur,
            extras=None,
        )
        print("  " + log_msg)
    train_time = time.time() - start
    test_acc, preds = evaluate_model(model, lambda t: t, test_loader, device, use_ste=True)
    if best_test_acc is None or test_acc > best_test_acc:
        best_test_acc = test_acc
    return label, last_train_acc, test_acc, train_time, preds, best_test_acc


def run_variant_transformer(train_loader,
                            test_loader,
                            device,
                            *,
                            epochs: int,
                            report_train_acc: bool,
                            report_epoch_acc: bool,
                            report_epoch_test: bool,
                            num_classes: int,
                            architecture: str,
                            backend: str,
                            image_size: Tuple[int, int],
                            in_channels: int,
                            patch_size: int,
                            depths: Union[int, Sequence[int]],
                            embed_dims: Union[int, Sequence[int]],
                            num_heads: Union[int, Sequence[int]],
                            mlp_ratio: Union[float, Sequence[float]],
                            window_size: int,
                            drop_path: float,
                            dropout: float,
                            tm_clauses: Union[int, Sequence[int]],
                            tm_tau: float,
                            use_cls_token: bool,
                            pool: str,
                            grad_checkpoint: bool,
                            ema_decay: float,
                            mixup_alpha: float,
                            cutmix_alpha: float,
                            base_lr: float = 7e-4,
                            min_lr: float = 7e-4,
                            warmup_epochs: int = 0,
                            weight_decay: float = 1e-5,
                            layer_decay: float = 1.0,
                            gradient_centralize: bool = False,
                            aux_weight: float = 0.0,
                            ff_gate: str = "none",
                            ff_gate_activation: str = "sigmoid",
                            layerscale_init: float = 1e-4,
                            use_layerscale: bool = True,
                            clause_dropout: float = 0.0,
                            literal_dropout: float = 0.0,
                            sparsity_weight: float = 0.0,
                            clause_bias_init: float = 0.0,
                            norm_type: str = "layernorm",
                            mix_type: str = "none",
                            bitwise_mix: bool = False,
                            learnable_tau: bool = False,
                            tau_min: float = 0.05,
                            tau_max: float = 0.95,
                            tau_ema_beta: Optional[float] = None,
                            clause_attention: bool = False,
                            clause_routing: bool = False,
                            continuous_bypass: bool = False,
                            bypass_scale: float = 1.0,
                            ) -> Tuple[str, Optional[float], float, float, List[int], Optional[Dict[str, Any]]]:
    model = UnifiedTMTransformer(
        num_classes=num_classes,
        architecture=architecture,
        backend=backend,
        image_size=image_size,
        in_channels=in_channels,
        patch_size=patch_size,
        embed_dim=embed_dims,
        depths=depths,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        tm_clauses=tm_clauses,
        tm_tau=tm_tau,
        drop_rate=dropout,
        attn_drop_rate=dropout,
        drop_path_rate=drop_path,
        window_size=window_size,
        use_cls_token=use_cls_token,
        pool=pool,
        grad_checkpoint=grad_checkpoint,
        ff_gate=ff_gate,
        ff_gate_activation=ff_gate_activation,
        layerscale_init=layerscale_init,
        use_layerscale=use_layerscale,
        clause_dropout=clause_dropout,
        literal_dropout=literal_dropout,
        ff_sparsity_weight=sparsity_weight,
        clause_bias_init=clause_bias_init,
        norm_type=norm_type,
        ff_mix_type=mix_type,
        ff_bitwise_mix=bitwise_mix,
        learnable_tau=learnable_tau,
        tau_min=tau_min,
        tau_max=tau_max,
        tau_ema_beta=tau_ema_beta,
        clause_attention=clause_attention,
        clause_routing=clause_routing,
        continuous_bypass=continuous_bypass,
        bypass_scale=bypass_scale,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    ema_model = copy.deepcopy(model) if ema_decay > 0 else None
    start = time.time()
    last_train_acc: Optional[float] = None
    best_test_acc: Optional[float] = None
    label = f"Transformer-{architecture.upper()}"
    collect_aux = aux_weight > 0
    component_history: List[Dict[str, Any]] = []

    for epoch in range(epochs):
        epoch_lr = update_cosine_warmup_lr(opt, epoch, epochs, base_lr, min_lr, warmup_epochs)
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        num_batches = 0
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x_aug, targets_a, targets_b, lam, aug_mode = apply_mixup_cutmix(x, y, mixup_alpha, cutmix_alpha)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(x_aug, use_ste=True, collect_diagnostics=collect_aux)
                if isinstance(outputs, tuple):
                    logits, aux_outputs = outputs
                else:
                    logits, aux_outputs = outputs, {}
                if lam is not None:
                    loss = lam * F.cross_entropy(logits, targets_a) + (1 - lam) * F.cross_entropy(logits, targets_b)
                else:
                    loss = F.cross_entropy(logits, y)
                if collect_aux and aux_outputs:
                    aux_loss = 0.0
                    for aux_logits in aux_outputs.values():
                        if lam is not None:
                            aux_loss += lam * F.cross_entropy(aux_logits, targets_a) + (1 - lam) * F.cross_entropy(aux_logits, targets_b)
                        else:
                            aux_loss += F.cross_entropy(aux_logits, y)
                    aux_loss = aux_loss / max(len(aux_outputs), 1)
                    loss = loss + aux_weight * aux_loss
                reg_loss = model.pop_regularization_loss()
                if reg_loss is not None:
                    loss = loss + reg_loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if gradient_centralize:
                apply_gradient_centralization(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(opt)
            scaler.update()
            if ema_model is not None:
                with torch.no_grad():
                    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                        ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)
            running_loss += loss.item()
            num_batches += 1
            if report_train_acc:
                preds = logits.argmax(dim=1)
                epoch_correct += (preds == y).sum().item()
                epoch_total += y.size(0)
        dur = time.time() - epoch_start
        avg_loss = running_loss / max(1, num_batches)
        epoch_acc = None
        if report_train_acc and epoch_total > 0:
            epoch_acc = epoch_correct / epoch_total
            last_train_acc = epoch_acc

        eval_model = ema_model if ema_model is not None else model
        epoch_test_acc, component_acc, _ = evaluate_transformer_components(
            eval_model,
            test_loader,
            device,
            collect_preds=False,
        )
        best_test_acc = epoch_test_acc if best_test_acc is None else max(best_test_acc, epoch_test_acc)
        component_history.append(
            {
                "epoch": epoch + 1,
                "test_accuracy": epoch_test_acc,
                "component_accuracy": component_acc,
            }
        )

        extras = {f"{name}_acc": acc for name, acc in component_acc.items()}
        log_msg = format_epoch_log(
            label=label,
            epoch=epoch + 1,
            total_epochs=epochs,
            loss=avg_loss,
            train_acc=epoch_acc if report_epoch_acc else None,
            test_acc=epoch_test_acc if report_epoch_test else None,
            best_acc=best_test_acc,
            lr=epoch_lr,
            duration=dur,
            extras=extras if extras else None,
        )
        print("  " + log_msg)

    train_time = time.time() - start
    eval_model = ema_model if ema_model is not None else model
    test_acc, final_component_acc, preds = evaluate_transformer_components(
        eval_model,
        test_loader,
        device,
        collect_preds=True,
    )
    if best_test_acc is None or test_acc > best_test_acc:
        best_test_acc = test_acc
    diagnostics = {
        "per_epoch": component_history,
        "final": final_component_acc,
    }
    return label, last_train_acc, test_acc, train_time, preds, diagnostics, best_test_acc


def run_experiment_with_args(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    maybe_set_seed(args.seed)
    if args.tm_L is None:
        args.tm_L = args.tm_literal_budget if args.tm_literal_budget is not None else 16
    if args.tm_LF is None:
        args.tm_LF = args.tm_lf

    dataset_key = args.dataset.lower()
    if dataset_key not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset '{args.dataset}'. Available: {', '.join(sorted(DATASET_CONFIGS))}")
    dataset_cfg = DATASET_CONFIGS[dataset_key]
    if args.num_classes is None:
        args.num_classes = dataset_cfg["num_classes"]

    if dataset_key != "mnist":
        dataset_upper = dataset_key.upper()
        if args.bool_train_path == DEFAULT_BOOL_TRAIN_PATH:
            args.bool_train_path = f"/tmp/{dataset_upper}TrainingData.txt"
        if args.bool_test_path == DEFAULT_BOOL_TEST_PATH:
            args.bool_test_path = f"/tmp/{dataset_upper}TestData.txt"
        if args.output_json == DEFAULT_OUTPUT_JSON:
            args.output_json = f"/tmp/{dataset_key}_equiv_results.json"
        if args.export_prefix == DEFAULT_EXPORT_PREFIX:
            args.export_prefix = f"tm_{dataset_key}"

    device = current_device(args.device)

    pin_memory = args.pin_memory
    if pin_memory is None:
        pin_memory = device.type != "cpu"

    persistent_workers = args.persistent_workers
    if persistent_workers is None:
        persistent_workers = args.num_workers > 0
    if args.num_workers == 0:
        persistent_workers = False

    transform = dataset_cfg["transform"]
    target_channels = args.tm_target_channels if args.tm_target_channels is not None else dataset_cfg["input_channels"]
    target_size = tuple(args.tm_target_size) if args.tm_target_size is not None else tuple(dataset_cfg["image_size"])
    if transform is None:
        transform_steps: List[Any] = []
    elif isinstance(transform, transforms.Compose):
        transform_steps = list(transform.transforms)
    else:
        transform_steps = [transform]
    if tuple(dataset_cfg["image_size"]) != target_size:
        transform_steps.insert(0, transforms.Resize(target_size, interpolation=InterpolationMode.BICUBIC, antialias=True))
    if args.randaugment:
        transform_steps.insert(0, transforms.RandAugment(num_ops=args.randaugment_n, magnitude=args.randaugment_m))
    has_tensor = any(isinstance(t, transforms.ToTensor) for t in transform_steps)
    if not has_tensor:
        transform_steps.append(transforms.ToTensor())
    transform = transforms.Compose(transform_steps)

    with profile_block("load-dataset"):
        train_ds = dataset_cfg["train_class"](
            root=args.dataset_root,
            train=True,
            download=args.download,
            transform=transform,
        )
        test_ds = dataset_cfg["test_class"](
            root=args.dataset_root,
            train=False,
            download=args.download,
            transform=transform,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    if args.write_bool_dataset:
        with profile_block("write-boolean-datasets"):
            save_bool_dataset(
                train_loader,
                test_loader,
                args.bool_train_path,
                args.bool_test_path,
                thr=args.bool_threshold,
            )

    tm_feature_mode = args.tm_feature_mode.lower()
    tm_bundle = None
    tm_train_loader = train_loader
    tm_test_loader = test_loader
    tm_prepare_fn: Callable[[torch.Tensor], torch.Tensor] = lambda t: t

    selected_models = tuple(args.models or DEFAULT_MODELS)
    if not selected_models:
        raise ValueError("No model variants selected.")

    input_channels = target_channels
    image_h, image_w = target_size
    flat_dim = input_channels * image_h * image_w
    tm_input_shape = (input_channels, image_h, image_w)
    tm_input_shape_active: Optional[Tuple[int, int, int]] = tm_input_shape
    tm_auto_expand = args.tm_auto_expand_grayscale
    tm_allow_reduce = args.tm_allow_channel_reduce
    tm_n_features = flat_dim

    feature_mode = "conv" if tm_feature_mode == "fashion_aug" else tm_feature_mode

    if feature_mode == "conv":
        config_name = args.tm_feature_config or dataset_key
        if config_name not in DEFAULT_PREPROCESS_CONFIGS:
            raise ValueError(
                f"No TM preprocessing config named '{config_name}'. Available: {', '.join(DEFAULT_PREPROCESS_CONFIGS)}."
            )
        base_config = DEFAULT_PREPROCESS_CONFIGS[config_name]
        if args.tm_feature_size is not None:
            base_config = replace(base_config, image_size=tuple(args.tm_feature_size))
        if args.tm_feature_grayscale is not None:
            base_config = replace(base_config, to_grayscale=args.tm_feature_grayscale)
        cache_dir = Path(args.tm_feature_cache)
        with profile_block("prepare-fashion-augmented-features"):
            tm_bundle = prepare_boolean_feature_bundle(
                train_ds,
                test_ds,
                cache_dir,
                config=base_config,
                batch_size=args.tm_feature_batch,
                force=args.tm_feature_force,
                apply_augmentations=args.tm_augment,
            )
        tm_train_loader = DataLoader(
            tm_bundle.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        tm_test_loader = DataLoader(
            tm_bundle.test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        tm_prepare_fn = lambda t: t
        tm_n_features = tm_bundle.num_bits
        tm_input_shape_active = None
    elif feature_mode == "raw":
        tm_input_shape_active = tm_input_shape
        tm_prepare_fn = flatten_images
    else:
        raise ValueError(f"Unknown tm feature mode '{tm_feature_mode}'.")

    deeptm_train_loader = tm_train_loader
    deeptm_test_loader = tm_test_loader
    deeptm_prepare_fn = tm_prepare_fn
    deeptm_input_dim = tm_n_features
    deeptm_input_shape_active = tm_input_shape_active

    if args.transformer_arch == "vit":
        mlp_vals = parse_float_list(args.transformer_mlp_ratio)
        transformer_mlp_cfg: Union[float, Tuple[float, ...]] = mlp_vals[0] if mlp_vals else float(args.transformer_mlp_ratio)
        transformer_depths_cfg: Union[int, Tuple[int, ...]] = args.transformer_layers
        transformer_heads_cfg: Union[int, Tuple[int, ...]] = args.transformer_heads
        transformer_embed_cfg: Union[int, Tuple[int, ...]] = int(args.transformer_d_model)
        transformer_tm_clauses_cfg: Union[int, Tuple[int, ...]] = args.transformer_clauses
    else:
        depth_list = parse_int_list(args.transformer_depths) if args.transformer_depths else [2, 2, 6, 2]
        head_list = parse_int_list(args.transformer_stage_heads) if args.transformer_stage_heads else [3, 6, 12, 24][: len(depth_list)]
        embed_list = (
            parse_int_list(args.transformer_embed_dims)
            if args.transformer_embed_dims
            else [int(args.transformer_d_model) * (2 ** i) for i in range(len(depth_list))]
        )
        mlp_list = parse_float_list(args.transformer_mlp_ratio)
        if not mlp_list:
            mlp_list = [4.0] * len(depth_list)
        while len(mlp_list) < len(depth_list):
            mlp_list.append(mlp_list[-1])
        transformer_depths_cfg = tuple(depth_list)
        transformer_heads_cfg = tuple(head_list)
        transformer_embed_cfg = tuple(embed_list)
        transformer_mlp_cfg = tuple(mlp_list)
        transformer_tm_clauses_cfg = tuple([int(args.transformer_clauses)] * len(depth_list))

    hidden_dims = parse_int_list(args.deeptm_hidden_dims)
    if not hidden_dims:
        hidden_dims = [256, 128]

    warmup_epochs = args.warmup_epochs
    weight_decay = args.weight_decay

    def resolve_lr(default_base: float) -> Tuple[float, float]:
        base = args.lr if args.lr is not None else default_base
        min_lr = args.min_lr if args.min_lr is not None else base
        return base, min_lr

    tm_base_lr, tm_min_lr = resolve_lr(1e-3)
    deeptm_base_lr, deeptm_min_lr = resolve_lr(1e-3)
    hybrid_base_lr, hybrid_min_lr = resolve_lr(1e-3)
    transformer_base_lr, transformer_min_lr = resolve_lr(7e-4)

    results: Dict[str, Dict[str, Any]] = {}

    for model_key in selected_models:
        with profile_block(f"variant-{model_key}"):
            diagnostics: Optional[Dict[str, Any]] = None
            if model_key == "tm":
                label, train_acc, test_acc, train_time, preds, bundle, best_epoch_test_acc = run_variant_tm(
                    tm_train_loader,
                    tm_test_loader,
                    device,
                    epochs=args.epochs,
                    report_train_acc=args.report_train_acc,
                    report_epoch_acc=args.report_epoch_acc,
                    report_epoch_test=args.report_epoch_test,
                    n_features=tm_n_features,
                    prepare_fn=tm_prepare_fn,
                    tm_impl=args.tm_impl,
                    tm_tau=args.tm_tau,
                    tm_lf=args.tm_lf,
                    tm_literal_budget=args.tm_literal_budget,
                    tm_vote_clamp=args.tm_vote_clamp,
                    tm_n_clauses=args.tm_n_clauses,
                    input_shape=tm_input_shape_active,
                    auto_expand_grayscale=tm_auto_expand,
                    allow_channel_reduce=tm_allow_reduce,
                    n_classes=args.num_classes,
                    base_lr=tm_base_lr,
                    min_lr=tm_min_lr,
                    warmup_epochs=warmup_epochs,
                    weight_decay=weight_decay,
                    gradient_centralize=args.gradient_centralize,
                )
                variant_classes = args.num_classes
            elif model_key == "deep_tm":
                label, train_acc, test_acc, train_time, preds, bundle, best_epoch_test_acc = run_variant_deeptm(
                    deeptm_train_loader,
                    deeptm_test_loader,
                    device,
                    epochs=args.epochs,
                    report_train_acc=args.report_train_acc,
                    report_epoch_acc=args.report_epoch_acc,
                    report_epoch_test=args.report_epoch_test,
                    prepare_fn=deeptm_prepare_fn,
                    input_dim=deeptm_input_dim,
                    hidden_dims=hidden_dims,
                    n_clauses=args.deeptm_n_clauses,
                    dropout=args.deeptm_dropout,
                    tau=args.deeptm_tau,
                    input_shape=deeptm_input_shape_active,
                    auto_expand_grayscale=tm_auto_expand,
                    allow_channel_reduce=tm_allow_reduce,
                    n_classes=args.num_classes,
                    base_lr=deeptm_base_lr,
                    min_lr=deeptm_min_lr,
                    warmup_epochs=warmup_epochs,
                    weight_decay=weight_decay,
                    gradient_centralize=args.gradient_centralize,
                )
                variant_classes = args.num_classes
            elif model_key == "hybrid":
                hybrid_classes = args.hybrid_classes if args.hybrid_classes is not None else args.num_classes
                label, train_acc, test_acc, train_time, preds, bundle, best_epoch_test_acc = run_variant_hybrid(
                    train_loader,
                    test_loader,
                    device,
                    epochs=args.epochs,
                    report_train_acc=args.report_train_acc,
                    report_epoch_acc=args.report_epoch_acc,
                    report_epoch_test=args.report_epoch_test,
                    in_channels=input_channels,
                    thresholds=args.hybrid_thresholds,
                    tm_clauses=args.hybrid_clauses,
                    tm_tau=args.tm_tau,
                    n_classes=hybrid_classes,
                    base_lr=hybrid_base_lr,
                    min_lr=hybrid_min_lr,
                    warmup_epochs=warmup_epochs,
                    weight_decay=weight_decay,
                    gradient_centralize=args.gradient_centralize,
                )
                variant_classes = hybrid_classes
            elif model_key == "transformer":
                label, train_acc, test_acc, train_time, preds, diagnostics, best_epoch_test_acc = run_variant_transformer(
                    train_loader,
                    test_loader,
                    device,
                    epochs=args.epochs,
                    report_train_acc=args.report_train_acc,
                    report_epoch_acc=args.report_epoch_acc,
                    report_epoch_test=args.report_epoch_test,
                    num_classes=args.num_classes,
                    architecture=args.transformer_arch,
                    backend=args.transformer_backend,
                    image_size=target_size,
                    in_channels=input_channels,
                    patch_size=args.transformer_patch,
                    depths=transformer_depths_cfg,
                    embed_dims=transformer_embed_cfg,
                    num_heads=transformer_heads_cfg,
                    mlp_ratio=transformer_mlp_cfg,
                    window_size=args.transformer_window,
                    drop_path=args.transformer_drop_path,
                    dropout=args.transformer_dropout,
                    tm_clauses=transformer_tm_clauses_cfg,
                    tm_tau=args.tm_tau,
                    use_cls_token=args.transformer_use_cls,
                    pool=args.transformer_pool,
                    grad_checkpoint=args.transformer_grad_checkpoint,
                    ema_decay=args.transformer_ema_decay,
                    mixup_alpha=args.mixup_alpha,
                    cutmix_alpha=args.cutmix_alpha,
                    base_lr=transformer_base_lr,
                    min_lr=transformer_min_lr,
                    warmup_epochs=warmup_epochs,
                    weight_decay=weight_decay,
                    layer_decay=args.lr_layer_decay,
                    gradient_centralize=args.gradient_centralize,
                    aux_weight=args.transformer_aux_weight,
                    ff_gate=args.transformer_ff_gate,
                    ff_gate_activation=args.transformer_ff_gate_activation,
                    layerscale_init=args.transformer_layerscale_init,
                    use_layerscale=args.transformer_use_layerscale,
                    clause_dropout=args.transformer_clause_drop,
                    literal_dropout=args.transformer_literal_drop,
                    sparsity_weight=args.transformer_sparsity_weight,
                    clause_bias_init=args.transformer_clause_bias,
                    norm_type=args.transformer_norm,
                    mix_type=args.transformer_ff_mix,
                    bitwise_mix=args.transformer_bitwise_mix,
                    learnable_tau=args.transformer_learnable_tau,
                    tau_min=args.transformer_tau_min,
                    tau_max=args.transformer_tau_max,
                    tau_ema_beta=args.transformer_tau_ema,
                    clause_attention=args.transformer_clause_attention,
                    clause_routing=args.transformer_clause_routing,
                    continuous_bypass=args.transformer_bypass,
                    bypass_scale=args.transformer_bypass_scale,
                )
                bundle = None
                variant_classes = args.num_classes
            else:
                raise ValueError(f"Unknown model variant '{model_key}'.")

        export_path = None
        if model_key in {"tm", "deep_tm", "hybrid"} and args.export_compiled and bundle is not None:
            os.makedirs(args.export_dir, exist_ok=True)
            export_base = os.path.join(
                args.export_dir,
                f"{args.export_prefix}_{EXPORT_SLUGS.get(model_key, model_key)}",
            )
            export_path = export_compiled_to_json(
                bundle,
                [str(i) for i in range(variant_classes)],
                export_base,
                bundle["clauses_num"],
                L=int(args.tm_L),
                LF=int(args.tm_LF),
            )

        results[model_key] = {
            "label": label,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "train_time_s": train_time,
            "json": export_path,
            "preds": preds,
            "best_epoch_test_accuracy": best_epoch_test_acc,
        }
        if model_key == "tm":
            results[model_key]["feature_mode"] = tm_feature_mode
            results[model_key]["n_features"] = tm_n_features
            if tm_bundle is not None:
                results[model_key]["feature_cache"] = {
                    "train_path": str(tm_bundle.train_dataset.data_path),
                    "test_path": str(tm_bundle.test_dataset.data_path),
                    "num_bits": tm_bundle.num_bits,
                    "augmentations": tm_bundle.augmentation_names,
                    "config": tm_bundle.config_name,
                }
        if model_key == "transformer" and diagnostics is not None:
            results[model_key]["diagnostics"] = diagnostics

    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f)

    print("\n=== Summary ===")
    for key, info in results.items():
        train_acc = info.get("train_accuracy")
        test_acc = info.get("test_accuracy")
        acc_parts = []
        if train_acc is not None:
            acc_parts.append(f"train_acc={train_acc:.4f}")
        if test_acc is not None:
            acc_parts.append(f"test_acc={test_acc:.4f}")
        acc_str = " ".join(acc_parts)
        label = info.get("label", key.upper())
        print(f"{label:12s} {acc_str} train_time={info['train_time_s']:.1f}s json={info['json']}")

    if args.write_bool_dataset:
        print("Boolean datasets:", args.bool_train_path, args.bool_test_path)
    else:
        print("Boolean datasets: skipped (flag disabled)")
    print("Summary JSON:", args.output_json)
    return results


def main(argv: Optional[List[str]] = None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.list_baselines:
        list_baseline_scenarios()
        return

    baseline_names = tuple(args.baseline_scenarios or [])
    target_dir = args.baseline_output_dir
    if target_dir is not None:
        os.makedirs(target_dir, exist_ok=True)

    for scenario_name in baseline_names:
        scenario_args = build_baseline_args(args, scenario_name, output_dir=target_dir)
        print(f"\n>>> Running baseline scenario: {scenario_name}")
        run_experiment_with_args(scenario_args)

    if baseline_names and args.baseline_only:
        return

    args.baseline_scenarios = None
    args.baseline_output_dir = None
    args.baseline_only = False
    args.list_baselines = False

    run_experiment_with_args(args)


if __name__ == "__main__":
    main()

