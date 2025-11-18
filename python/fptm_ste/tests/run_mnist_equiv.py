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
import inspect
from collections import defaultdict
from dataclasses import replace
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

try:  # pragma: no cover - optional dependency
    import timm
except ImportError:  # pragma: no cover
    timm = None

from fptm_ste import FuzzyPatternTM_STE, FuzzyPatternTMFPTM, FuzzyPatternTM_STCM
from fptm_ste.binarizers import CNNSingleBinarizer
from fptm_ste.deep_tm import DeepTMNetwork
from fptm_ste.export import export_compiled_to_json
from fptm_ste.tm_transformer import UnifiedTMTransformer
from fptm_ste.tm_priors import apply_tm_prior_template
from fptm_ste.datasets import (
    DEFAULT_PREPROCESS_CONFIGS,
    PreprocessConfig,
    prepare_boolean_feature_bundle,
    prepare_fashion_augmented_bundle,
)
from fptm_ste.visualization import generate_transformer_overlay
from fptm_ste.trainers import anneal_ste_factor

AVAILABLE_MODELS = ("tm", "stcm", "deep_tm", "deep_stcm", "hybrid", "transformer")
DEFAULT_MODELS = ("tm", "deep_tm", "hybrid", "transformer")
EXPORT_SLUGS = {
    "tm": "tm",
    "stcm": "stcm",
    "deep_tm": "deeptm",
    "deep_stcm": "deepstcm",
    "hybrid": "hybrid",
    "transformer": "transformer",
}
DEFAULT_DATA_ROOT = os.environ.get("TM_DATA_ROOT", os.environ.get("TM_MNIST_ROOT", "/tmp/mnist"))
DEFAULT_BOOL_TRAIN_PATH = os.environ.get("TM_BOOL_TRAIN_PATH", "/tmp/MNISTTrainingData.txt")
DEFAULT_BOOL_TEST_PATH = os.environ.get("TM_BOOL_TEST_PATH", "/tmp/MNISTTestData.txt")
DEFAULT_OUTPUT_JSON = os.environ.get("TM_OUTPUT_JSON", "/tmp/mnist_equiv_results.json")
DEFAULT_EXPORT_PREFIX = os.environ.get("TM_EXPORT_PREFIX", "tm_mnist")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_IMAGENET_ROOT = os.environ.get("TM_IMAGENET_ROOT", "/datasets/imagenet")

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
    "imagenet": {
        "train_class": datasets.ImageFolder,
        "test_class": datasets.ImageFolder,
        "train_transform": None,
        "test_transform": None,
        "input_channels": 3,
        "image_size": (224, 224),
        "num_classes": 1000,
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


class TeacherFeatureCollector:
    """
    Helper to capture intermediate teacher representations for distillation.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._module_map = dict(model.named_modules())
        self.alias_map = self._build_alias_map()

    def _build_alias_map(self) -> Dict[str, str]:
        alias: Dict[str, str] = {}
        if "patch_embed" in self._module_map:
            alias["patch_embed"] = "patch_embed"
        if hasattr(self.model, "blocks"):
            blocks = getattr(self.model, "blocks")
            try:
                block_len = len(blocks)
            except TypeError:
                block_len = 0
            for idx in range(block_len):
                alias[f"blocks.{idx}"] = f"block_{idx + 1}"
        if hasattr(self.model, "layers"):
            layers = getattr(self.model, "layers")
            for stage_idx, layer in enumerate(layers):
                if hasattr(layer, "blocks"):
                    try:
                        block_count = len(layer.blocks)
                    except TypeError:
                        block_count = 0
                    for block_idx in range(block_count):
                        alias[f"layers.{stage_idx}.blocks.{block_idx}"] = (
                            f"stage{stage_idx + 1}_block{block_idx + 1}"
                        )
        if "norm" in self._module_map:
            alias["norm"] = "pre_head"
        return alias

    @staticmethod
    def _extract_tensor(output: Any) -> Optional[torch.Tensor]:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (list, tuple)) and output:
            for item in output:
                if isinstance(item, torch.Tensor):
                    return item
        return None

    def _make_hook(self, alias: str, store: Dict[str, torch.Tensor]):
        def hook(_: nn.Module, __: Tuple[torch.Tensor, ...], output: Any):
            tensor = self._extract_tensor(output)
            if tensor is not None:
                store[alias] = tensor.detach()

        return hook

    def collect(
        self,
        x: torch.Tensor,
        *,
        retain_logits_grad: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        features: Dict[str, torch.Tensor] = {}
        handles: List[Any] = []
        for name, alias in self.alias_map.items():
            module = self._module_map.get(name)
            if module is None:
                continue
            handles.append(module.register_forward_hook(self._make_hook(alias, features)))
        try:
            result = self.model(x)
        finally:
            for handle in handles:
                handle.remove()
        logits = result[0] if isinstance(result, (tuple, list)) else result
        logits_for_loss = logits if retain_logits_grad else logits.detach()
        return logits, features, logits_for_loss


class TeacherHintAdapter(nn.Module):
    """
    Lightweight projection heads that map teacher features to student logits.
    """

    def __init__(self, num_classes: int, device: torch.device):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.adapters = nn.ModuleDict()
        self._new_layers: List[nn.Module] = []

    def _pool_feature(self, alias: str, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        if tensor.dim() == 4:
            tensor = tensor.flatten(2).transpose(1, 2)
        if tensor.dim() == 3:
            if alias.startswith("stage"):
                return tensor.mean(dim=1)
            if tensor.size(1) > 0:
                return tensor[:, 0]
            return tensor.mean(dim=1)
        if tensor.dim() == 2:
            return tensor
        if tensor.dim() > 2:
            return tensor.view(tensor.size(0), -1)
        return None

    def _get_adapter(self, name: str, in_dim: int) -> nn.Linear:
        if name not in self.adapters:
            layer = nn.Linear(in_dim, self.num_classes)
            nn.init.trunc_normal_(layer.weight, std=0.02)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            layer = layer.to(self.device)
            self.adapters[name] = layer
            self._new_layers.append(layer)
        return self.adapters[name]

    def forward(self, feature_map: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        for name, tensor in feature_map.items():
            pooled = self._pool_feature(name, tensor)
            if pooled is None:
                continue
            pooled = pooled.to(self.device)
            adapter = self._get_adapter(name, pooled.shape[-1])
            outputs[name] = adapter(pooled)
        return outputs

    def consume_new_layers(self) -> List[nn.Module]:
        created = self._new_layers
        self._new_layers = []
        return created


class TeacherClauseProjector(nn.Module):
    """
    Projects teacher features into clause activation space for alignment.
    """

    def __init__(self, clause_dims: Dict[str, int], device: torch.device):
        super().__init__()
        self.clause_dims = clause_dims
        self.device = device
        self.adapters = nn.ModuleDict()
        self._new_layers: List[nn.Module] = []

    def _pool_feature(self, alias: str, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        if tensor.dim() == 4:
            tensor = tensor.flatten(2).transpose(1, 2)
        if tensor.dim() == 3:
            if alias.startswith("stage"):
                return tensor.mean(dim=1)
            if tensor.size(1) > 0:
                return tensor[:, 0]
            return tensor.mean(dim=1)
        if tensor.dim() == 2:
            return tensor
        if tensor.dim() > 2:
            return tensor.view(tensor.size(0), -1)
        return None

    def _get_adapter(self, name: str, in_dim: int, out_dim: int) -> nn.Linear:
        if name not in self.adapters:
            layer = nn.Linear(in_dim, out_dim)
            nn.init.trunc_normal_(layer.weight, std=0.02)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            layer = layer.to(self.device)
            self.adapters[name] = layer
            self._new_layers.append(layer)
        return self.adapters[name]

    def forward(self, feature_map: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        for name, tensor in feature_map.items():
            if name not in self.clause_dims:
                continue
            pooled = self._pool_feature(name, tensor)
            if pooled is None:
                continue
            pooled = pooled.to(self.device)
            adapter = self._get_adapter(name, pooled.shape[-1], self.clause_dims[name])
            outputs[name] = adapter(pooled)
        return outputs

    def consume_new_layers(self) -> List[nn.Module]:
        created = self._new_layers
        self._new_layers = []
        return created


def infer_clause_dimensions(model: UnifiedTMTransformer) -> Dict[str, int]:
    dims: Dict[str, int] = {}
    if model.architecture == "vit":
        for idx, block in enumerate(model.blocks):
            ffn = getattr(block, "ffn", None)
            if ffn is None:
                continue
            core = getattr(ffn, "core", None)
            if core is None or not hasattr(core, "n_clauses"):
                continue
            dims[f"block_{idx + 1}"] = int(getattr(core, "n_clauses"))
    else:
        for stage_idx, stage in enumerate(model.stages):
            for block_idx, block in enumerate(stage.blocks):
                ffn = getattr(block, "ffn", None)
                if ffn is None:
                    continue
                core = getattr(ffn, "core", None)
                if core is None or not hasattr(core, "n_clauses"):
                    continue
                dims[f"stage{stage_idx + 1}_block{block_idx + 1}"] = int(getattr(core, "n_clauses"))
    return dims


def collect_clause_outputs(model: UnifiedTMTransformer) -> Dict[str, torch.Tensor]:
    outputs: Dict[str, torch.Tensor] = {}
    if model.architecture == "vit":
        for idx, block in enumerate(model.blocks):
            ffn = getattr(block, "ffn", None)
            if ffn is None:
                continue
            clause_tensor = getattr(ffn, "last_clause_outputs_raw", None)
            if clause_tensor is None:
                continue
            outputs[f"block_{idx + 1}"] = clause_tensor
            ffn.last_clause_outputs_raw = None
    else:
        for stage_idx, stage in enumerate(model.stages):
            for block_idx, block in enumerate(stage.blocks):
                ffn = getattr(block, "ffn", None)
                if ffn is None:
                    continue
                clause_tensor = getattr(ffn, "last_clause_outputs_raw", None)
                if clause_tensor is None:
                    continue
                outputs[f"stage{stage_idx + 1}_block{block_idx + 1}"] = clause_tensor
                ffn.last_clause_outputs_raw = None
    return outputs


def _flatten_teacher_tokens(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 4:
        tensor = tensor.flatten(2).transpose(1, 2)
    return tensor


def compute_teacher_token_attention(alias: str, tensor: torch.Tensor) -> Optional[torch.Tensor]:
    tokens = _flatten_teacher_tokens(tensor)
    if tokens.dim() != 3 or tokens.size(1) == 0:
        return None
    if alias.startswith("block_") and tokens.size(1) > 1:
        cls_token = tokens[:, 0:1, :]
        other_tokens = tokens[:, 1:, :]
        attn_logits = torch.matmul(other_tokens, cls_token.transpose(1, 2)).squeeze(-1)
        attention = torch.softmax(attn_logits, dim=1)
        return attention
    norms = tokens.norm(dim=-1)
    norms = norms.clamp_min(1e-6)
    attention = norms / norms.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return attention


def compute_student_clause_token_distribution(
    clause_tensor: torch.Tensor,
    *,
    drop_first_token: bool,
) -> torch.Tensor:
    if clause_tensor.dim() == 2:
        clause_tensor = clause_tensor.unsqueeze(1)
    if drop_first_token and clause_tensor.size(1) > 1:
        clause_tensor = clause_tensor[:, 1:, :]
    token_strength = clause_tensor.abs().mean(dim=2)
    token_strength = token_strength.clamp_min(1e-6)
    distribution = token_strength / token_strength.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return distribution


def _select_clause_parameters(model: UnifiedTMTransformer) -> List[nn.Parameter]:
    clause_params: List[nn.Parameter] = []
    clause_keywords = ("ta_", "clause_bias", "core.voting", "core.tau", "core.clause")
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in clause_keywords):
            clause_params.append(param)
    return clause_params


def warm_start_clauses_with_teacher(
    model: UnifiedTMTransformer,
    teacher_collector: TeacherFeatureCollector,
    dataloader: DataLoader,
    device: torch.device,
    *,
    batches: int,
    lr: float,
    temperature: float,
) -> None:
    clause_params = _select_clause_parameters(model)
    if not clause_params or batches <= 0:
        return
    optimizer = torch.optim.Adam(clause_params, lr=lr)
    model.train()
    processed = 0
    for x, _ in dataloader:
        if processed >= batches:
            break
        x = x.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_logits, _, _ = teacher_collector.collect(x, retain_logits_grad=False)
            teacher_logits = teacher_logits.detach()
        outputs = model(x, use_ste=True)
        student_logits = outputs[0] if isinstance(outputs, tuple) else outputs
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature ** 2)
        kd_loss.backward()
        optimizer.step()
        processed += 1


class DistillationStageScheduler:
    """
    Handles two-stage scheduling for KD weights, temperatures, and tm_tau annealing.
    """

    def __init__(
        self,
        *,
        stage_epoch: int,
        total_epochs: int,
        base_weights: Dict[str, float],
        base_temps: Dict[str, float],
        weight_scales: Dict[str, float],
        temp_scales: Dict[str, float],
        base_tau: Optional[float],
        target_tau: Optional[float],
    ):
        self.stage_epoch = max(0, stage_epoch)
        self.total_epochs = max(1, total_epochs)
        self.base_weights = base_weights
        self.base_temps = base_temps
        self.weight_scales = weight_scales
        self.temp_scales = temp_scales
        self.base_tau = base_tau
        self.target_tau = target_tau

    def _in_stage2(self, epoch: int) -> bool:
        return self.stage_epoch > 0 and epoch >= self.stage_epoch

    def compute(self, epoch: int) -> Tuple[Dict[str, float], Dict[str, float], Optional[float]]:
        stage2 = self._in_stage2(epoch)
        weights = {}
        for key, value in self.base_weights.items():
            scale = self.weight_scales.get(key, 1.0) if stage2 else 1.0
            weights[key] = value * scale
        temps = {}
        for key, value in self.base_temps.items():
            scale = self.temp_scales.get(key, 1.0) if stage2 else 1.0
            temps[key] = value * scale
        tau_value: Optional[float] = None
        if (
            stage2
            and self.base_tau is not None
            and self.target_tau is not None
        ):
            span = max(1, self.total_epochs - self.stage_epoch + 1)
            progress = min(1.0, (epoch - self.stage_epoch + 1) / span)
            tau_value = self.base_tau + (self.target_tau - self.base_tau) * progress
        return weights, temps, tau_value


class TeacherLogitAdapter(nn.Module):
    """
    Lightweight adapter to transform frozen teacher logits.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.proj = nn.Linear(num_classes, num_classes, bias=True)
        nn.init.eye_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.proj(logits)


class TeacherContrastiveAdapter(nn.Module):
    """
    Projects teacher logits into student feature space for contrastive KD.
    """

    def __init__(self, num_classes: int, feature_dim: int):
        super().__init__()
        self.proj = nn.Linear(num_classes, feature_dim, bias=True)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.proj(logits)


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


def _coerce_transform_spec(transform_spec: Optional[Any]) -> List[Any]:
    if transform_spec is None:
        return []
    if isinstance(transform_spec, transforms.Compose):
        return list(transform_spec.transforms)
    if isinstance(transform_spec, (list, tuple)):
        return list(transform_spec)
    return [transform_spec]


def _build_transform(
    transform_spec: Optional[Any],
    *,
    target_size: Optional[Tuple[int, int]],
    apply_randaugment: bool,
    randaugment_n: int,
    randaugment_m: int,
    add_normalize: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None,
) -> transforms.Compose:
    steps = _coerce_transform_spec(transform_spec)
    if target_size is not None:
        resize = transforms.Resize(target_size, interpolation=InterpolationMode.BICUBIC, antialias=True)
        steps = [resize] + steps
    if apply_randaugment:
        steps.insert(0, transforms.RandAugment(num_ops=randaugment_n, magnitude=randaugment_m))
    has_tensor = any(isinstance(t, transforms.ToTensor) for t in steps)
    if not has_tensor:
        steps.append(transforms.ToTensor())
    if add_normalize is not None:
        mean, std = add_normalize
        steps.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(steps)


def _extract_normalize(transform: transforms.Compose) -> Optional[Tuple[Sequence[float], Sequence[float]]]:
    if not isinstance(transform, transforms.Compose):
        return None
    for step in transform.transforms:
        if isinstance(step, transforms.Normalize):
            return (tuple(step.mean), tuple(step.std))
    return None


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
        choices=AVAILABLE_MODELS,
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
    parser.add_argument(
        "--test-holdout-fraction",
        type=float,
        default=0.0,
        help="Fraction of the evaluation dataset reserved for blind holdout (0 disables the split).",
    )
    parser.add_argument(
        "--test-holdout-seed",
        type=int,
        default=None,
        help="Optional random seed controlling evaluation/holdout split when --test-holdout-fraction > 0.",
    )
    parser.add_argument(
        "--imagenet-train-dir",
        type=str,
        default=None,
        help="Override ImageNet training directory (defaults to <dataset_root>/train).",
    )
    parser.add_argument(
        "--imagenet-val-dir",
        type=str,
        default=None,
        help="Override ImageNet validation directory (defaults to <dataset_root>/val).",
    )
    parser.add_argument(
        "--imagenet-resize",
        type=int,
        default=256,
        help="Resize shorter side to this size for ImageNet evaluation pipeline.",
    )
    parser.add_argument(
        "--imagenet-crop",
        type=int,
        default=224,
        help="Crop size for ImageNet training and evaluation.",
    )
    parser.add_argument("--lr", type=float, default=None, help="Base learning rate for AdamW optimizers.")
    parser.add_argument("--min-lr", type=float, default=None, help="Minimum learning rate for cosine decay.")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Number of warmup epochs before cosine decay.")
    parser.add_argument(
        "--lr-cycle-steps",
        type=int,
        default=0,
        help="Optional cosine restart cycle length in optimizer steps (0 disables restarts).",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay value for AdamW optimizers.")
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor applied to cross-entropy losses.",
    )
    parser.add_argument(
        "--tm-prior-template",
        default="none",
        choices=["none", "symmetric", "zero"],
        help="Optional TM prior template applied to clause initialisation.",
    )
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
        choices=["ste", "fptm", "stcm"],
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
    parser.add_argument("--tm-clause-dropout", type=float, default=0.0, help="Clause dropout applied in STE/STCM layers.")
    parser.add_argument("--tm-literal-dropout", type=float, default=0.0, help="Literal dropout applied in STE/STCM layers.")
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
    parser.add_argument(
        "--stcm-operator",
        choices=["capacity", "product"],
        default="capacity",
        help="Clause strength operator used by STCM variants.",
    )
    parser.add_argument(
        "--stcm-ternary-voting",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable ternary voting weights in STCM models.",
    )
    parser.add_argument(
        "--stcm-ternary-band",
        type=float,
        default=0.0,
        help="Neutral band for ternary literal selection in STCM.",
    )
    parser.add_argument(
        "--stcm-ste-temperature",
        type=float,
        default=1.0,
        help="Temperature for STCM STE ternary masks.",
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
        "--transformer-drop-path-schedule",
        choices=["linear", "cosine"],
        default="linear",
        help="Drop-path schedule across blocks (linear default or cosine).",
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
        "--transformer-auto-clause",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Auto-tune transformer clause counts based on diagnostics before training.",
    )
    parser.add_argument(
        "--transformer-clause-target",
        type=int,
        default=0,
        help="Optional total clause budget to enforce during auto clause tuning (0 keeps scaled totals).",
    )
    parser.add_argument(
        "--transformer-auto-clause-batches",
        type=int,
        default=4,
        help="Number of mini-batches sampled when estimating diagnostics for clause auto-tuning.",
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
        "--transformer-self-distill-weight",
        type=float,
        default=0.0,
        help="KL divergence weight for self-distillation from EMA teacher.",
    )
    parser.add_argument(
        "--transformer-self-distill-temp",
        type=float,
        default=1.0,
        help="Temperature used for self-distillation logits.",
    )
    parser.add_argument(
        "--distill-teacher-timm",
        type=str,
        default="",
        help="Optional timm model name to use as external teacher (pretrained).",
    )
    parser.add_argument(
        "--distill-teacher-checkpoint",
        type=str,
        default="",
        help="Path to a teacher checkpoint (torch.load state_dict).",
    )
    parser.add_argument(
        "--distill-teacher-weight",
        type=float,
        default=0.0,
        help="Weight for external teacher KD loss.",
    )
    parser.add_argument(
        "--distill-teacher-temp",
        type=float,
        default=1.0,
        help="Temperature for external teacher KD.",
    )
    parser.add_argument(
        "--distill-teacher-trainable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow external teacher weights to update during distillation.",
    )
    parser.add_argument(
        "--distill-teacher-lr",
        type=float,
        default=None,
        help="Optional learning rate for trainable teacher parameters (defaults to transformer base LR).",
    )
    parser.add_argument(
        "--transformer-hint-weight",
        type=float,
        default=0.0,
        help="Weight for FitNet-style hint distillation on transformer diagnostics.",
    )
    parser.add_argument(
        "--transformer-hint-temp",
        type=float,
        default=1.0,
        help="Temperature for hint distillation logits.",
    )
    parser.add_argument(
        "--transformer-clause-align-weight",
        type=float,
        default=0.0,
        help="Weight for clause activation alignment with teacher projections.",
    )
    parser.add_argument(
        "--transformer-clause-align-temp",
        type=float,
        default=1.0,
        help="Temperature for clause alignment distributions.",
    )
    parser.add_argument(
        "--transformer-attn-guidance-weight",
        type=float,
        default=0.0,
        help="Weight for aligning teacher attention with clause token usage.",
    )
    parser.add_argument(
        "--transformer-attn-guidance-temp",
        type=float,
        default=1.0,
        help="Temperature applied when matching attention distributions.",
    )
    parser.add_argument(
        "--transformer-relational-weight",
        type=float,
        default=0.0,
        help="Weight for relational KD on component logits.",
    )
    parser.add_argument(
        "--distill-teacher-adapter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable a trainable adapter on frozen teacher logits.",
    )
    parser.add_argument(
        "--distill-teacher-adapter-lr",
        type=float,
        default=1e-3,
        help="Learning rate for the teacher logit adapter when enabled.",
    )
    parser.add_argument(
        "--transformer-kd-stage-epochs",
        type=int,
        default=0,
        help="Epoch index (1-based) to switch to KD stage 2 (0 disables staged scheduling).",
    )
    parser.add_argument(
        "--transformer-kd-stage2-teacher-scale",
        type=float,
        default=1.0,
        help="Scale applied to external teacher KD weight after the stage boundary.",
    )
    parser.add_argument(
        "--transformer-kd-stage2-teacher-temp-scale",
        type=float,
        default=1.0,
        help="Scale applied to external teacher KD temperature after the stage boundary.",
    )
    parser.add_argument(
        "--transformer-kd-stage2-self-scale",
        type=float,
        default=1.0,
        help="Scale applied to EMA self-distillation weight after the stage boundary.",
    )
    parser.add_argument(
        "--transformer-kd-stage2-self-temp-scale",
        type=float,
        default=1.0,
        help="Scale applied to EMA self-distillation temperature after the stage boundary.",
    )
    parser.add_argument(
        "--transformer-kd-stage2-hint-scale",
        type=float,
        default=1.0,
        help="Scale applied to hint distillation weight after the stage boundary.",
    )
    parser.add_argument(
        "--transformer-kd-stage2-hint-temp-scale",
        type=float,
        default=1.0,
        help="Scale applied to hint distillation temperature after the stage boundary.",
    )
    parser.add_argument(
        "--transformer-kd-stage2-clause-scale",
        type=float,
        default=1.0,
        help="Scale applied to clause alignment weight after the stage boundary.",
    )
    parser.add_argument(
        "--transformer-kd-stage2-clause-temp-scale",
        type=float,
        default=1.0,
        help="Scale applied to clause alignment temperature after the stage boundary.",
    )
    parser.add_argument(
        "--transformer-kd-stage2-attn-scale",
        type=float,
        default=1.0,
        help="Scale applied to attention guidance weight after the stage boundary.",
    )
    parser.add_argument(
        "--transformer-kd-stage2-attn-temp-scale",
        type=float,
        default=1.0,
        help="Scale applied to attention guidance temperature after the stage boundary.",
    )
    parser.add_argument(
        "--transformer-kd-stage2-rel-scale",
        type=float,
        default=1.0,
        help="Scale applied to relational KD weight after the stage boundary.",
    )
    parser.add_argument(
        "--transformer-kd-stage2-tau",
        type=float,
        default=None,
        help="Target tm_tau value reached by the end of training (linear schedule starting at stage boundary).",
    )
    parser.add_argument(
        "--transformer-clause-init-batches",
        type=int,
        default=0,
        help="Calibration batches used for teacher-guided clause warm start (0 disables).",
    )
    parser.add_argument(
        "--transformer-clause-init-lr",
        type=float,
        default=1e-3,
        help="Learning rate used during clause warm start calibration.",
    )
    parser.add_argument(
        "--transformer-clause-init-temp",
        type=float,
        default=2.0,
        help="Temperature applied during clause warm start KD.",
    )
    parser.add_argument(
        "--teacher-aug-path",
        type=str,
        default="",
        help="Optional path to pseudo-labeled dataset generated by a teacher (torch .pt).",
    )
    parser.add_argument(
        "--teacher-aug-batches",
        type=int,
        default=0,
        help="Number of pseudo-labeled batches to use per epoch.",
    )
    parser.add_argument(
        "--transformer-contrastive-weight",
        type=float,
        default=0.0,
        help="Supervised contrastive loss weight on pooled transformer features.",
    )
    parser.add_argument(
        "--transformer-contrastive-temp",
        type=float,
        default=0.1,
        help="Temperature for supervised contrastive objective.",
    )
    parser.add_argument(
        "--transformer-contrastive-use-teacher",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Augment contrastive loss with teacher anchors.",
    )
    parser.add_argument(
        "--transformer-grad-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable gradient checkpointing inside transformer blocks.",
    )
    parser.add_argument(
        "--transformer-use-flash-attn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use PyTorch scaled_dot_product_attention (FlashAttention when available).",
    )
    parser.add_argument(
        "--transformer-residual-attn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable cross-block residual attention (adds previous attention output to the next block).",
    )
    parser.add_argument(
        "--transformer-relative-pos",
        choices=["none", "learned", "rotary"],
        default="none",
        help="Relative positional encoding type for transformer attention.",
    )
    parser.add_argument(
        "--transformer-clause-specialize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rescale attention heads based on clause usage diagnostics.",
    )
    parser.add_argument(
        "--transformer-clause-specialize-strength",
        type=float,
        default=0.5,
        help="Smoothing factor (0-1) for clause-aware head specialization.",
    )
    parser.add_argument(
        "--transformer-log-dir",
        type=str,
        default="",
        help="Optional TensorBoard log directory for transformer diagnostics.",
    )
    parser.add_argument(
        "--transformer-save-path",
        type=str,
        default="",
        help="Optional filepath to save the trained transformer checkpoint (state_dict + config).",
    )
    parser.add_argument(
        "--visualize-overlays",
        type=str,
        default="",
        help="Directory to save interpretability overlays (empty to disable).",
    )
    parser.add_argument(
        "--visualize-samples",
        type=int,
        default=0,
        help="Number of test samples to visualize with overlays.",
    )
    parser.add_argument(
        "--visualize-topk",
        type=int,
        default=3,
        help="Top-K classes to display in overlay reports.",
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
    parser.add_argument(
        "--mixup-schedule",
        choices=["constant", "linear", "cosine"],
        default="cosine",
        help="Schedule for mixup strength across epochs.",
    )
    parser.add_argument(
        "--cutmix-schedule",
        choices=["constant", "linear", "cosine"],
        default="cosine",
        help="Schedule for CutMix strength across epochs.",
    )
    # Adapter learning-rate flag defined above with other KD controls.
    parser.add_argument(
        "--teacher-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fine-tune and evaluate the teacher standalone to establish a baseline.",
    )
    parser.add_argument(
        "--teacher-baseline-model",
        type=str,
        default="",
        help="timm model name for teacher baseline (defaults to --distill-teacher-timm).",
    )
    parser.add_argument(
        "--teacher-baseline-checkpoint",
        type=str,
        default="",
        help="Optional checkpoint path for teacher baseline initialization.",
    )
    parser.add_argument(
        "--teacher-baseline-img-size",
        nargs="+",
        type=int,
        default=None,
        help="Override teacher input image size (single value or H W). Defaults to teacher config.",
    )
    parser.add_argument(
        "--teacher-baseline-epochs",
        type=int,
        default=50,
        help="Training epochs for teacher baseline fine-tuning.",
    )
    parser.add_argument(
        "--teacher-baseline-batch-size",
        type=int,
        default=None,
        help="Mini-batch size for teacher baseline (defaults to --batch-size).",
    )
    parser.add_argument(
        "--teacher-baseline-lr",
        type=float,
        default=1e-4,
        help="Base learning rate for teacher baseline optimizer.",
    )
    parser.add_argument(
        "--teacher-baseline-min-lr",
        type=float,
        default=None,
        help="Minimum learning rate for teacher baseline cosine schedule (defaults to lr).",
    )
    parser.add_argument(
        "--teacher-baseline-warmup",
        type=int,
        default=5,
        help="Warmup epochs for teacher baseline cosine schedule.",
    )
    parser.add_argument(
        "--teacher-baseline-weight-decay",
        type=float,
        default=0.05,
        help="Weight decay applied during teacher baseline optimization.",
    )
    parser.add_argument(
        "--teacher-baseline-label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing coefficient for teacher baseline cross-entropy.",
    )
    parser.add_argument(
        "--teacher-baseline-log-dir",
        type=str,
        default="",
        help="Optional TensorBoard directory for teacher baseline diagnostics.",
    )
    parser.add_argument(
        "--teacher-baseline-save-path",
        type=str,
        default="",
        help="Optional checkpoint path for saving the fine-tuned teacher baseline.",
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


def cosine_restart_lr(step: int, total_steps: int, cycle_steps: int, base_lr: float, min_lr: float) -> float:
    if cycle_steps <= 0:
        return cosine_warmup_lr(step, total_steps, base_lr, min_lr, 0)
    cycle_pos = step % cycle_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * cycle_pos / float(max(1, cycle_steps))))
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


def cosine_with_restarts_lr(
    step: int,
    warmup_steps: int,
    base_lr: float,
    min_lr: float,
    cycle_steps: int,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        warm_progress = float(step + 1) / float(max(1, warmup_steps))
        return base_lr * warm_progress
    if cycle_steps <= 0:
        return min_lr
    step_after = step - warmup_steps
    cycle_pos = step_after % cycle_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * cycle_pos / float(max(1, cycle_steps))))
    return min_lr + (base_lr - min_lr) * cosine


def schedule_scalar(epoch: int, total_epochs: int, value: float, schedule: str) -> float:
    if value <= 0.0:
        return 0.0
    if schedule == "constant" or total_epochs <= 1:
        return value
    progress = float(epoch) / float(max(1, total_epochs - 1))
    if schedule == "linear":
        factor = progress
    elif schedule == "cosine":
        factor = 0.5 * (1.0 - math.cos(math.pi * progress))
    else:
        factor = 1.0
    return value * factor


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[float], str, Optional[torch.Tensor]]:
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return inputs, targets, targets, None, "none", None

    use_cuda = inputs.is_cuda
    if mixup_alpha > 0 and cutmix_alpha > 0:
        choose_mixup = torch.rand(1, device=inputs.device if use_cuda else None).item() < 0.5
    else:
        choose_mixup = mixup_alpha > 0

    perm = torch.randperm(inputs.size(0), device=inputs.device if use_cuda else None)
    if choose_mixup:
        lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
        mixed = lam * inputs + (1 - lam) * inputs[perm]
        return mixed, targets, targets[perm], lam, "mixup", perm

    lam = torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample().item()
    y1, y2, x1, x2 = rand_bbox(inputs.size(), lam, inputs.device)
    inputs_clone = inputs.clone()
    inputs_clone[:, :, y1:y2, x1:x2] = inputs[perm, :, y1:y2, x1:x2]
    lam = 1 - ((y2 - y1) * (x2 - x1) / (inputs.size(-1) * inputs.size(-2)))
    return inputs_clone, targets, targets[perm], lam, "cutmix", perm


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    if features.dim() > 2:
        features = features.view(features.size(0), -1)
    if features.size(0) <= 1:
        return features.new_zeros(())
    feats = F.normalize(features, dim=1)
    similarity = torch.matmul(feats, feats.t()) / max(temperature, 1e-6)
    logits_mask = torch.ones_like(similarity, dtype=torch.bool)
    logits_mask.fill_diagonal_(False)
    labels = labels.view(-1, 1)
    positive_mask = (labels == labels.t()) & logits_mask
    if positive_mask.sum() == 0:
        return features.new_zeros(())
    exp_sim = torch.exp(similarity) * logits_mask
    log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp(min=1)
    loss = -mean_log_prob_pos[positive_mask.sum(dim=1) > 0].mean()
    if torch.isnan(loss):
        return features.new_zeros(())
    return loss


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


def evaluate_classifier(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / total if total > 0 else 0.0
    if was_training:
        model.train()
    return acc


def evaluate_transformer_components(
    model: UnifiedTMTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    collect_preds: bool = True,
    max_batches: Optional[int] = None,
) -> Tuple[float, Dict[str, float], List[int], Dict[str, List[float]]]:
    was_training = model.training
    model.eval()
    total_correct = 0
    total = 0
    component_correct: Dict[str, int] = defaultdict(int)
    component_total: Dict[str, int] = defaultdict(int)
    preds: List[int] = []
    clause_sum: Dict[str, torch.Tensor] = {}
    clause_count: Dict[str, int] = defaultdict(int)
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader, start=1):
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
            batch_clause_metrics = model.consume_clause_metrics()
            for key, tensor in batch_clause_metrics.items():
                if key not in clause_sum:
                    clause_sum[key] = tensor.clone()
                else:
                    clause_sum[key] += tensor
                clause_count[key] += 1
            if max_batches is not None and batch_idx >= max_batches:
                break
    if was_training:
        model.train()
    final_acc = total_correct / total if total > 0 else 0.0
    component_acc = {
        name: (component_correct[name] / component_total[name]) if component_total[name] > 0 else 0.0
        for name in sorted(component_correct)
    }
    clause_metrics = {
        key: (clause_sum[key] / clause_count[key]).tolist()
        for key in clause_sum
        if clause_count[key] > 0
    }
    return final_acc, component_acc, preds, clause_metrics


def _vit_clause_list(tm_clauses: Union[int, Sequence[int]], depth: int) -> List[int]:
    if isinstance(tm_clauses, Sequence) and not isinstance(tm_clauses, (str, bytes)):
        clause_list = [int(max(1, c)) for c in tm_clauses]
        if len(clause_list) != depth:
            if len(clause_list) == 0:
                clause_list = [1] * depth
            else:
                clause_list = (clause_list + [clause_list[-1]])[:depth]
    else:
        clause_list = [int(max(1, tm_clauses))] * depth
    return clause_list


def _format_vit_clause_counts(new_counts: List[int], original: Union[int, Sequence[int]]) -> Union[int, Tuple[int, ...]]:
    new_counts = [int(max(1, c)) for c in new_counts]
    if isinstance(original, Sequence) and not isinstance(original, (str, bytes)):
        return tuple(new_counts)
    return int(sum(new_counts) / max(1, len(new_counts)))


def _collect_stage_accuracies(
    component_acc: Dict[str, float],
    architecture: str,
    depths: Union[int, Sequence[int]],
) -> List[float]:
    if architecture == "vit":
        depth = depths if isinstance(depths, int) else (sum(depths) if depths else 0)
        return [component_acc.get(f"block_{i + 1}", 0.0) for i in range(depth)]
    stage_depths = list(depths) if isinstance(depths, Sequence) else [int(depths)]
    stage_scores: List[float] = []
    for stage_idx, stage_depth in enumerate(stage_depths):
        block_scores = [
            component_acc.get(f"stage{stage_idx + 1}_block{block_idx + 1}", 0.0) for block_idx in range(stage_depth)
        ]
        if block_scores:
            stage_scores.append(sum(block_scores) / len(block_scores))
        else:
            stage_scores.append(0.0)
    return stage_scores


def _collect_clause_usage(
    clause_metrics: Dict[str, List[float]],
    architecture: str,
    depths: Union[int, Sequence[int]],
) -> List[float]:
    if architecture == "vit":
        depth = depths if isinstance(depths, int) else (sum(depths) if depths else 0)
        usage_vals: List[float] = []
        for i in range(depth):
            vals = clause_metrics.get(f"block_{i + 1}_clause_mean")
            if vals:
                usage_vals.append(float(sum(vals) / len(vals)))
            else:
                usage_vals.append(0.0)
        return usage_vals
    stage_depths = list(depths) if isinstance(depths, Sequence) else [int(depths)]
    usage_vals = []
    for stage_idx, stage_depth in enumerate(stage_depths):
        block_usages: List[float] = []
        for block_idx in range(stage_depth):
            vals = clause_metrics.get(f"stage{stage_idx + 1}_block{block_idx + 1}_clause_mean")
            if vals:
                block_usages.append(float(sum(vals) / len(vals)))
        usage_vals.append(float(sum(block_usages) / len(block_usages)) if block_usages else 0.0)
    return usage_vals


def build_distillation_teacher(
    timm_name: str,
    checkpoint_path: str,
    num_classes: int,
    in_channels: int,
    device: torch.device,
    img_size: Optional[Tuple[int, int]],
    *,
    trainable: bool = False,
) -> nn.Module:
    if not timm_name and not checkpoint_path:
        raise ValueError("No teacher configuration provided.")
    if timm is None:
        raise ImportError("timm is required for external teacher distillation.")
    if not timm_name:
        raise ValueError("A timm model name is required when loading an external teacher.")
    model_kwargs: Dict[str, Any] = {
        "pretrained": True,
        "num_classes": num_classes,
        "in_chans": in_channels,
    }
    if img_size is not None:
        model_kwargs["img_size"] = img_size
    teacher = timm.create_model(
        timm_name,
        **model_kwargs,
    )
    if hasattr(teacher, "reset_classifier"):
        teacher.reset_classifier(num_classes)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        teacher.load_state_dict(state, strict=False)
    teacher.to(device)
    if trainable:
        teacher.train()
    else:
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
    return teacher





def _resolve_teacher_input_size(
    override: Optional[Sequence[int]],
    default_cfg: Dict[str, Any],
    dataset_size: Tuple[int, int],
) -> Tuple[int, int]:
    if override:
        values = [int(v) for v in override if v is not None]
        if len(values) == 1:
            return (values[0], values[0])
        if len(values) >= 2:
            return (values[0], values[1])
    input_size = default_cfg.get("input_size")
    if isinstance(input_size, (list, tuple)):
        if len(input_size) >= 3:
            return (int(input_size[-2]), int(input_size[-1]))
        if len(input_size) == 2:
            return (int(input_size[0]), int(input_size[1]))
    if dataset_size and len(dataset_size) == 2:
        return (int(dataset_size[0]), int(dataset_size[1]))
    return (224, 224)


def _build_teacher_baseline_transforms(
    image_size: Tuple[int, int],
    *,
    mean: Sequence[float],
    std: Sequence[float],
    interpolation: str,
    crop_pct: float,
) -> Tuple[transforms.Compose, transforms.Compose]:
    height, width = int(image_size[0]), int(image_size[1])
    train_size: Union[int, Tuple[int, int]] = (height, width) if height != width else height
    interp_name = (interpolation or "bicubic").upper()
    interpolation_mode = getattr(InterpolationMode, interp_name, InterpolationMode.BICUBIC)
    crop_pct = float(crop_pct) if crop_pct else 0.875
    crop_pct = min(max(crop_pct, 0.1), 1.0)
    if crop_pct < 1.0:
        resize_h = max(1, math.ceil(height / crop_pct))
        resize_w = max(1, math.ceil(width / crop_pct))
    else:
        resize_h, resize_w = height, width
    resize_size = (resize_h, resize_w)
    mean = tuple(mean) if mean else IMAGENET_MEAN
    std = tuple(std) if std else IMAGENET_STD
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(train_size, interpolation=interpolation_mode),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=interpolation_mode, antialias=True),
            transforms.CenterCrop((height, width) if height != width else height),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


def run_teacher_baseline(
    *,
    args: argparse.Namespace,
    dataset_key: str,
    dataset_cfg: Dict[str, Any],
    device: torch.device,
    data_root: str,
    download: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    imagenet_mode: bool,
    imagenet_train_dir: Optional[str],
    imagenet_val_dir: Optional[str],
) -> Dict[str, Any]:
    if timm is None:
        raise ImportError("Teacher baseline requires timm to be installed.")
    teacher_name = args.teacher_baseline_model or args.distill_teacher_timm
    teacher_checkpoint = args.teacher_baseline_checkpoint or args.distill_teacher_checkpoint
    if not teacher_name:
        raise ValueError("Teacher baseline requested but no timm model name provided.")
    override_size = None
    if args.teacher_baseline_img_size:
        override_vals = [int(v) for v in args.teacher_baseline_img_size]
        if len(override_vals) == 1:
            override_size = (override_vals[0], override_vals[0])
        elif len(override_vals) >= 2:
            override_size = (override_vals[0], override_vals[1])
    teacher_model = build_distillation_teacher(
        teacher_name,
        teacher_checkpoint,
        args.num_classes,
        dataset_cfg["input_channels"],
        device,
        override_size,
        trainable=True,
    )
    default_cfg = getattr(teacher_model, "default_cfg", {}) or {}
    image_size = _resolve_teacher_input_size(args.teacher_baseline_img_size, default_cfg, dataset_cfg["image_size"])
    mean = default_cfg.get("mean", IMAGENET_MEAN)
    std = default_cfg.get("std", IMAGENET_STD)
    interpolation = default_cfg.get("interpolation", "bicubic")
    crop_pct = default_cfg.get("crop_pct", 0.875)
    train_transform, eval_transform = _build_teacher_baseline_transforms(
        image_size,
        mean=mean,
        std=std,
        interpolation=interpolation,
        crop_pct=crop_pct,
    )

    if imagenet_mode:
        if imagenet_train_dir is None or imagenet_val_dir is None:
            raise ValueError("ImageNet paths are required for teacher baseline in ImageNet mode.")
        train_dataset = dataset_cfg["train_class"](imagenet_train_dir, transform=train_transform)
        test_dataset = dataset_cfg["test_class"](imagenet_val_dir, transform=eval_transform)
    else:
        train_dataset = dataset_cfg["train_class"](
            root=data_root,
            train=True,
            download=download,
            transform=train_transform,
        )
        test_dataset = dataset_cfg["test_class"](
            root=data_root,
            train=False,
            download=download,
            transform=eval_transform,
        )

    baseline_batch_size = args.teacher_baseline_batch_size or args.batch_size
    test_batch_size = args.test_batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=baseline_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    total_epochs = args.teacher_baseline_epochs
    base_lr = args.teacher_baseline_lr
    min_lr = args.teacher_baseline_min_lr if args.teacher_baseline_min_lr is not None else base_lr
    weight_decay = args.teacher_baseline_weight_decay
    label_smoothing = args.teacher_baseline_label_smoothing

    optimizer = torch.optim.AdamW(teacher_model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    steps_per_epoch = max(1, len(train_loader))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    writer = None
    if args.teacher_baseline_log_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=args.teacher_baseline_log_dir)
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] TensorBoard writer unavailable for teacher baseline: {exc}")
            writer = None

    start_time = time.time()
    best_test_acc: Optional[float] = None
    last_train_acc: Optional[float] = None

    for epoch in range(total_epochs):
        epoch_lr = update_cosine_warmup_lr(
            optimizer,
            epoch,
            total_epochs,
            base_lr,
            min_lr,
            args.teacher_baseline_warmup,
        )
        teacher_model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_start = time.time()
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = teacher_model(inputs)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(teacher_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            preds = logits.argmax(dim=1)
            epoch_correct += (preds == targets).sum().item()
            epoch_total += targets.size(0)
        duration = time.time() - epoch_start
        avg_loss = epoch_loss / max(1, len(train_loader))
        train_acc = epoch_correct / max(1, epoch_total)
        last_train_acc = train_acc
        test_acc = evaluate_classifier(teacher_model, test_loader, device)
        if best_test_acc is None or test_acc > best_test_acc:
            best_test_acc = test_acc
        log_msg = format_epoch_log(
            label="TeacherBase",
            epoch=epoch + 1,
            total_epochs=total_epochs,
            loss=avg_loss,
            train_acc=train_acc,
            test_acc=test_acc,
            best_acc=best_test_acc,
            lr=epoch_lr,
            duration=duration,
            extras=None,
        )
        print("  " + log_msg)
        if writer is not None:
            writer.add_scalar("train/loss", avg_loss, epoch)
            writer.add_scalar("train/accuracy", train_acc, epoch)
            writer.add_scalar("eval/accuracy", test_acc, epoch)
            writer.add_scalar("lr", epoch_lr, epoch)

    train_time = time.time() - start_time
    final_test_acc = evaluate_classifier(teacher_model, test_loader, device)
    if best_test_acc is None or final_test_acc > best_test_acc:
        best_test_acc = final_test_acc
    if writer is not None:
        writer.add_scalar("eval/final_accuracy", final_test_acc, total_epochs)
        writer.close()

    total_samples = len(train_dataset)
    profile = {
        "num_samples": total_samples,
        "epochs": total_epochs,
        "total_steps": total_epochs * steps_per_epoch,
        "samples_per_second": (total_samples * total_epochs) / train_time if train_time > 0 else None,
        "max_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }

    if args.teacher_baseline_save_path:
        checkpoint_path = Path(args.teacher_baseline_save_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state": teacher_model.state_dict(),
            "model_name": teacher_name,
            "default_cfg": default_cfg,
            "image_size": image_size,
            "train_args": {
                "epochs": total_epochs,
                "base_lr": base_lr,
                "min_lr": min_lr,
                "weight_decay": weight_decay,
                "label_smoothing": label_smoothing,
                "batch_size": baseline_batch_size,
            },
            "metrics": {
                "train_accuracy": last_train_acc if last_train_acc is not None else 0.0,
                "best_test_accuracy": best_test_acc,
                "final_test_accuracy": final_test_acc,
                "train_time": train_time,
            },
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"  Saved teacher baseline checkpoint to {checkpoint_path}")

    label = f"Teacher-{teacher_name}"
    if last_train_acc is None:
        last_train_acc = 0.0
    result = {
        "label": label,
        "model_name": teacher_name,
        "train_accuracy": last_train_acc,
        "test_accuracy": final_test_acc,
        "best_epoch_test_accuracy": best_test_acc,
        "train_time_s": train_time,
        "json": None,
        "preds": [],
        "profile": profile,
        "image_size": image_size,
    }
    return result


def _scale_clause_counts(
    base_counts: List[int],
    accuracies: List[float],
    clause_target: int,
    usage: Optional[List[float]] = None,
) -> List[int]:
    target_acc = 0.75
    target_usage = 0.25
    new_counts: List[int] = []
    for idx, (base, acc) in enumerate(zip(base_counts, accuracies)):
        factor = 1.0 + (target_acc - acc)
        factor = max(0.5, min(1.5, factor))
        usage_factor = 1.0
        if usage is not None and idx < len(usage):
            usage_val = max(0.0, min(1.0, usage[idx]))
            usage_factor = 1.0 + (target_usage - usage_val)
            usage_factor = max(0.7, min(1.3, usage_factor))
        combined = factor * usage_factor
        new_counts.append(max(8, int(round(base * combined))))
    if clause_target > 0:
        total = sum(new_counts)
        if total > 0:
            scale = clause_target / total
            new_counts = [max(8, int(round(c * scale))) for c in new_counts]
    return new_counts


def auto_tune_transformer_clauses(
    *,
    model_kwargs: Dict[str, Any],
    tm_clauses: Union[int, Sequence[int]],
    depths: Union[int, Sequence[int]],
    architecture: str,
    train_loader: DataLoader,
    device: torch.device,
    auto_batches: int,
    clause_target: int,
    tm_prior_template: str,
) -> Union[int, Tuple[int, ...]]:
    if not auto_batches or auto_batches <= 0:
        return tm_clauses
    temp_kwargs = dict(model_kwargs)
    temp_kwargs["tm_clauses"] = tm_clauses
    temp_model = UnifiedTMTransformer(**temp_kwargs).to(device)
    apply_tm_prior_template(temp_model, tm_prior_template)
    _, component_acc, _, clause_metrics = evaluate_transformer_components(
        temp_model, train_loader, device, collect_preds=False, max_batches=auto_batches
    )
    del temp_model
    torch.cuda.empty_cache()

    if architecture == "vit":
        depth = depths if isinstance(depths, int) else (sum(depths) if depths else 0)
        base_counts = _vit_clause_list(tm_clauses, depth)
        accuracies = _collect_stage_accuracies(component_acc, architecture, depths)
        clause_usage_vals = _collect_clause_usage(clause_metrics, architecture, depths)
        new_counts = _scale_clause_counts(base_counts, accuracies, clause_target, usage=clause_usage_vals)
        return _format_vit_clause_counts(new_counts, tm_clauses)

    # Swin-style: per-stage clause counts
    stage_depths = list(depths) if isinstance(depths, Sequence) else [int(depths)]
    if isinstance(tm_clauses, Sequence) and not isinstance(tm_clauses, (str, bytes)):
        base_counts = [int(max(1, c)) for c in tm_clauses]
        if len(base_counts) != len(stage_depths):
            if len(base_counts) == 0:
                base_counts = [64] * len(stage_depths)
            else:
                base_counts = (base_counts + [base_counts[-1]])[: len(stage_depths)]
    else:
        base_counts = [int(max(1, tm_clauses))] * len(stage_depths)
    accuracies = _collect_stage_accuracies(component_acc, architecture, stage_depths)
    clause_usage_vals = _collect_clause_usage(clause_metrics, architecture, stage_depths)
    new_counts = _scale_clause_counts(base_counts, accuracies, clause_target, usage=clause_usage_vals)
    return tuple(int(max(1, c)) for c in new_counts)


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
                   gradient_centralize: bool = False,
                   label_smoothing: float = 0.0,
                   lr_cycle_steps: int = 0,
                   kd_scheduler: Optional["DistillationStageScheduler"] = None,
                   ema_model: Optional[nn.Module] = None,
                   teacher_aug_loader: Optional[DataLoader] = None,
                   teacher_aug_batches: int = 0,
                   teacher_logit_adapter: Optional[nn.Module] = None) -> Tuple[Optional[float], float, float, List[int], Optional[float]]:
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    start = time.time()
    last_train_acc: Optional[float] = None
    best_test_acc: Optional[float] = None
    steps_per_epoch = max(1, len(train_loader))
    warmup_steps = warmup_epochs * steps_per_epoch
    cycle_steps = max(0, lr_cycle_steps)
    for epoch in range(epochs):
        global_step = epoch * steps_per_epoch
        if cycle_steps > 0:
            epoch_lr = cosine_with_restarts_lr(global_step, warmup_steps, base_lr, min_lr, cycle_steps)
            set_optimizer_lr(opt, epoch_lr)
        else:
            epoch_lr = update_cosine_warmup_lr(opt, epoch, epochs, base_lr, min_lr, warmup_epochs)
        model.train()
        epoch_start = time.time()
        teacher_kd_weight = 0.0
        teacher_temp_epoch = 1.0
        if kd_scheduler is not None:
            weight_schedule, temp_schedule, tau_override = kd_scheduler.compute(epoch)
            teacher_kd_weight = weight_schedule.get("teacher", 0.0)
            teacher_temp_epoch = temp_schedule.get("teacher", 1.0)
            if tau_override is not None:
                anneal_ste_factor(model, tau_override)
                if ema_model is not None:
                    anneal_ste_factor(ema_model, tau_override)
        running_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        num_batches = 0
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            if cycle_steps > 0:
                current_lr = cosine_with_restarts_lr(global_step, warmup_steps, base_lr, min_lr, cycle_steps)
                set_optimizer_lr(opt, current_lr)
                epoch_lr = current_lr
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            xb = prepare_fn(x)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(xb, use_ste=use_ste_train)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
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
            global_step += 1
        if (
            teacher_aug_loader is not None
            and teacher_aug_batches > 0
            and teacher_kd_weight > 0.0
        ):
            pseudo_iter = iter(teacher_aug_loader)
            for _ in range(teacher_aug_batches):
                try:
                    px, plogits = next(pseudo_iter)
                except StopIteration:
                    pseudo_iter = iter(teacher_aug_loader)
                    px, plogits = next(pseudo_iter)
                px = px.to(device, non_blocking=True)
                kd_targets = plogits.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    outputs = model(
                        px,
                        use_ste=True,
                        collect_diagnostics=False,
                        return_features=False,
                    )
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    if teacher_logit_adapter is not None:
                        kd_targets = teacher_logit_adapter(kd_targets)
                    kd_loss = F.kl_div(
                        F.log_softmax(logits / teacher_temp_epoch, dim=1),
                        F.softmax(kd_targets / teacher_temp_epoch, dim=1),
                        reduction="batchmean",
                    ) * (teacher_temp_epoch ** 2)
                    loss = teacher_kd_weight * kd_loss
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(opt)
                scaler.update()
                running_loss += loss.detach().item()
                num_batches += 1
                global_step += 1
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
                   gradient_centralize: bool = False,
                   label_smoothing: float = 0.0,
                   tm_prior_template: str = "none",
                   lr_cycle_steps: int = 0,
                   tm_clause_dropout: float = 0.0,
                   tm_literal_dropout: float = 0.0,
                   stcm_operator: str = "capacity",
                   stcm_ternary_voting: bool = False,
                   stcm_ternary_band: float = 0.05,
                   stcm_ste_temperature: float = 1.0) -> Tuple[str, Optional[float], float, float, List[int], Dict[str, Any], Dict[str, Any]]:
    tm_impl = tm_impl.lower()
    use_ste_eval = True
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
    elif tm_impl == "stcm":
        model = FuzzyPatternTM_STCM(
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
            clause_dropout=tm_clause_dropout,
            literal_dropout=tm_literal_dropout,
            operator=stcm_operator,
            ternary_voting=stcm_ternary_voting,
            ternary_band=stcm_ternary_band,
            ste_temperature=stcm_ste_temperature,
        ).to(device)
        use_ste_train = False
        use_ste_eval = False
        label = "STCM"
    else:
        model = FuzzyPatternTM_STE(
            n_features=n_features,
            n_clauses=tm_n_clauses,
            n_classes=n_classes,
            tau=tm_tau,
            input_shape=input_shape,
            auto_expand_grayscale=auto_expand_grayscale,
            allow_channel_reduce=allow_channel_reduce,
            clause_dropout=tm_clause_dropout,
            literal_dropout=tm_literal_dropout,
        ).to(device)
        use_ste_train = False
        label = "STE-TM"

    apply_tm_prior_template(model, tm_prior_template)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_acc, test_acc, ttrain, preds, best_test_acc = train_tm_model(
        model,
        prepare_fn,
        train_loader,
        test_loader,
        device,
        label,
        use_ste_train=use_ste_train,
        use_ste_eval=use_ste_eval,
        epochs=epochs,
        report_train_acc=report_train_acc,
        report_epoch_acc=report_epoch_acc,
        report_epoch_test=report_epoch_test,
        base_lr=base_lr,
        min_lr=min_lr,
        warmup_epochs=warmup_epochs,
        weight_decay=weight_decay,
        gradient_centralize=gradient_centralize,
        label_smoothing=label_smoothing,
        lr_cycle_steps=lr_cycle_steps,
    )
    bundle = model.discretize(threshold=0.5)
    total_samples = len(train_loader.dataset)
    total_steps = epochs * max(1, len(train_loader))
    throughput = (total_samples * epochs) / ttrain if ttrain > 0 else None
    profile = {
        "num_samples": total_samples,
        "epochs": epochs,
        "total_steps": total_steps,
        "samples_per_second": throughput,
        "max_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }
    return label, train_acc, test_acc, ttrain, preds, bundle, best_test_acc, profile


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
                       gradient_centralize: bool = False,
                       label_smoothing: float = 0.0,
                       tm_prior_template: str = "none",
                       lr_cycle_steps: int = 0,
                       layer_cls: Type[nn.Module] = FuzzyPatternTM_STE,
                       layer_operator: Optional[str] = None,
                       layer_ternary_voting: Optional[bool] = None,
                       layer_extra_kwargs: Optional[Dict[str, Any]] = None,
                       label_override: Optional[str] = None,
                       clause_dropout: float = 0.0,
                       literal_dropout: float = 0.0,
                       use_ste_train: bool = False,
                       use_ste_eval: bool = True) -> Tuple[str, Optional[float], float, float, List[int], Dict[str, Any], Dict[str, Any]]:
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
        clause_dropout=clause_dropout,
        literal_dropout=literal_dropout,
        layer_cls=layer_cls,
        layer_operator=layer_operator,
        layer_ternary_voting=layer_ternary_voting,
        layer_extra_kwargs=layer_extra_kwargs,
    ).to(device)
    apply_tm_prior_template(model, tm_prior_template)
    label = label_override or "Deep-TM"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_acc, test_acc, ttrain, preds, best_test_acc = train_tm_model(
        model,
        prepare_fn,
        train_loader,
        test_loader,
        device,
        label,
        use_ste_train=use_ste_train,
        use_ste_eval=use_ste_eval,
        epochs=epochs,
        report_train_acc=report_train_acc,
        report_epoch_acc=report_epoch_acc,
        report_epoch_test=report_epoch_test,
        base_lr=base_lr,
        min_lr=min_lr,
        warmup_epochs=warmup_epochs,
        weight_decay=weight_decay,
        gradient_centralize=gradient_centralize,
        label_smoothing=label_smoothing,
        lr_cycle_steps=lr_cycle_steps,
    )
    bundle = model.classifier.discretize(threshold=0.5)
    total_samples = len(train_loader.dataset)
    total_steps = epochs * max(1, len(train_loader))
    throughput = (total_samples * epochs) / ttrain if ttrain > 0 else None
    profile = {
        "num_samples": total_samples,
        "epochs": epochs,
        "total_steps": total_steps,
        "samples_per_second": throughput,
        "max_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }
    return label, train_acc, test_acc, ttrain, preds, bundle, best_test_acc, profile


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
                       weight_decay: float = 1e-5,
                       gradient_centralize: bool = False,
                       label_smoothing: float = 0.0,
                       tm_prior_template: str = "none",
                       lr_cycle_steps: int = 0) -> Tuple[str, Optional[float], float, float, List[int], Dict[str, Any], Dict[str, Any]]:
    model = CNNHybrid(
        in_channels=in_channels,
        thresholds=thresholds,
        tm_clauses=tm_clauses,
        n_classes=n_classes,
        tm_tau=tm_tau,
    ).to(device)
    apply_tm_prior_template(model.tm, tm_prior_template)
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    start = time.time()
    last_train_acc: Optional[float] = None
    best_test_acc: Optional[float] = None
    label = "Hybrid"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    steps_per_epoch = max(1, len(train_loader))
    warmup_steps = warmup_epochs * steps_per_epoch
    cycle_steps = max(0, lr_cycle_steps)
    for epoch in range(epochs):
        global_step = epoch * steps_per_epoch
        if cycle_steps > 0:
            epoch_lr = cosine_with_restarts_lr(global_step, warmup_steps, base_lr, min_lr, cycle_steps)
            set_optimizer_lr(opt, epoch_lr)
        else:
            epoch_lr = update_cosine_warmup_lr(opt, epoch, epochs, base_lr, min_lr, warmup_epochs)
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        num_batches = 0
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            if cycle_steps > 0:
                current_lr = cosine_with_restarts_lr(global_step, warmup_steps, base_lr, min_lr, cycle_steps)
                set_optimizer_lr(opt, current_lr)
                epoch_lr = current_lr
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits, _ = model(x, use_ste=False)
                loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
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
            global_step += 1
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
    total_samples = len(train_loader.dataset)
    total_steps = epochs * max(1, len(train_loader))
    throughput = (total_samples * epochs) / train_time if train_time > 0 else None
    profile = {
        "num_samples": total_samples,
        "epochs": epochs,
        "total_steps": total_steps,
        "samples_per_second": throughput,
        "max_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }
    return label, last_train_acc, test_acc, train_time, preds, best_test_acc, profile


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
                            drop_path_schedule: str,
                            dropout: float,
                            tm_clauses: Union[int, Sequence[int]],
                            tm_tau: float,
                            use_cls_token: bool,
                            pool: str,
                            grad_checkpoint: bool,
                            ema_decay: float,
                            mixup_alpha: float,
                            cutmix_alpha: float,
                            mixup_schedule: str,
                            cutmix_schedule: str,
                            base_lr: float = 7e-4,
                            min_lr: float = 7e-4,
                            warmup_epochs: int = 0,
                            weight_decay: float = 1e-5,
                            layer_decay: float = 1.0,
                            lr_cycle_steps: int = 0,
                            gradient_centralize: bool = False,
                            aux_weight: float = 0.0,
                            label_smoothing: float = 0.0,
                            auto_clause: bool = False,
                            clause_target: int = 0,
                            auto_clause_batches: int = 4,
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
                            tm_prior_template: str = "none",
                            self_distill_weight: float = 0.0,
                            self_distill_temp: float = 1.0,
                            hint_distill_weight: float = 0.0,
                            hint_distill_temp: float = 1.0,
                            clause_align_weight: float = 0.0,
                            clause_align_temp: float = 1.0,
                            attn_guidance_weight: float = 0.0,
                            attn_guidance_temp: float = 1.0,
                            relational_weight: float = 0.0,
                            kd_stage_epochs: int = 0,
                            kd_stage2_teacher_scale: float = 1.0,
                            kd_stage2_teacher_temp_scale: float = 1.0,
                            kd_stage2_self_scale: float = 1.0,
                            kd_stage2_self_temp_scale: float = 1.0,
                            kd_stage2_hint_scale: float = 1.0,
                            kd_stage2_hint_temp_scale: float = 1.0,
                            kd_stage2_clause_scale: float = 1.0,
                            kd_stage2_clause_temp_scale: float = 1.0,
                            kd_stage2_attn_scale: float = 1.0,
                            kd_stage2_attn_temp_scale: float = 1.0,
                            kd_stage2_rel_scale: float = 1.0,
                            kd_stage2_tau: Optional[float] = None,
                            clause_init_batches: int = 0,
                            clause_init_lr: float = 1e-3,
                            clause_init_temp: float = 2.0,
                            contrastive_weight: float = 0.0,
                            contrastive_temp: float = 0.1,
                            contrastive_use_teacher: bool = True,
                            teacher_aug_path: str = "",
                            teacher_aug_batches: int = 0,
                            use_flash_attention: bool = False,
                            use_residual_attention: bool = False,
                            log_dir: str = "",
                            relative_position_type: str = "none",
                            visualize_dir: str = "",
                            visualize_samples: int = 0,
                            visualize_topk: int = 3,
                            class_names: Optional[Sequence[str]] = None,
                            test_dataset: Optional[Dataset] = None,
                            holdout_loader: Optional[DataLoader] = None,
                            normalize_stats: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
                            clause_specialize: bool = False,
                            clause_specialize_strength: float = 0.5,
                            distill_teacher_timm: str = "",
                            distill_teacher_checkpoint: str = "",
                            distill_teacher_weight: float = 0.0,
                            distill_teacher_temp: float = 1.0,
                            distill_teacher_trainable: bool = False,
                            distill_teacher_lr: Optional[float] = None,
                            distill_teacher_adapter: bool = False,
                            distill_teacher_adapter_lr: float = 1e-3,
                            save_path: str = "",
                            ) -> Tuple[str, Optional[float], float, float, List[int], Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
    base_model_kwargs: Dict[str, Any] = {
        "num_classes": num_classes,
        "architecture": architecture,
        "backend": backend,
        "image_size": image_size,
        "in_channels": in_channels,
        "patch_size": patch_size,
        "embed_dim": embed_dims,
        "depths": depths,
        "num_heads": num_heads,
        "mlp_ratio": mlp_ratio,
        "tm_tau": tm_tau,
        "drop_rate": dropout,
        "attn_drop_rate": dropout,
        "drop_path_rate": drop_path,
        "drop_path_schedule": drop_path_schedule,
        "window_size": window_size,
        "use_cls_token": use_cls_token,
        "pool": pool,
        "grad_checkpoint": grad_checkpoint,
        "ff_gate": ff_gate,
        "ff_gate_activation": ff_gate_activation,
        "layerscale_init": layerscale_init,
        "use_layerscale": use_layerscale,
        "clause_dropout": clause_dropout,
        "literal_dropout": literal_dropout,
        "ff_sparsity_weight": sparsity_weight,
        "clause_bias_init": clause_bias_init,
        "norm_type": norm_type,
        "ff_mix_type": mix_type,
        "ff_bitwise_mix": bitwise_mix,
        "learnable_tau": learnable_tau,
        "tau_min": tau_min,
        "tau_max": tau_max,
        "tau_ema_beta": tau_ema_beta,
        "clause_attention": clause_attention,
        "clause_routing": clause_routing,
        "continuous_bypass": continuous_bypass,
        "bypass_scale": bypass_scale,
        "use_flash_attention": use_flash_attention,
        "use_residual_attention": use_residual_attention,
        "relative_position_type": relative_position_type,
    }

    if auto_clause:
        tm_clauses = auto_tune_transformer_clauses(
            model_kwargs=base_model_kwargs,
            tm_clauses=tm_clauses,
            depths=depths,
            architecture=architecture,
            train_loader=train_loader,
            device=device,
            auto_batches=auto_clause_batches,
            clause_target=clause_target,
            tm_prior_template=tm_prior_template,
        )
        print(f"  Auto-tuned clause counts: {tm_clauses}")

    base_model_kwargs["tm_clauses"] = tm_clauses
    model = UnifiedTMTransformer(**base_model_kwargs).to(device)
    apply_tm_prior_template(model, tm_prior_template)
    teacher_model = None
    if (distill_teacher_weight > 0.0 or distill_teacher_trainable) and (distill_teacher_timm or distill_teacher_checkpoint):
        teacher_model = build_distillation_teacher(
            distill_teacher_timm,
            distill_teacher_checkpoint,
            num_classes,
            in_channels,
            device,
            image_size,
            trainable=distill_teacher_trainable,
        )
    teacher_lr_value = distill_teacher_lr if distill_teacher_lr is not None else base_lr
    if teacher_model is not None:
        print(f"Loaded teacher model: {distill_teacher_timm or distill_teacher_checkpoint}")
    param_groups = build_param_groups_with_layer_decay(model, base_lr, weight_decay, layer_decay)
    opt = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)
    teacher_logit_adapter = None
    teacher_contrastive_adapter = None
    if teacher_model is not None and distill_teacher_trainable:
        teacher_params = [p for p in teacher_model.parameters() if p.requires_grad]
        if teacher_params:
            opt.add_param_group({"params": teacher_params, "lr": teacher_lr_value, "weight_decay": weight_decay})
    if (
        teacher_model is not None
        and not distill_teacher_trainable
        and distill_teacher_adapter
    ):
        teacher_logit_adapter = TeacherLogitAdapter(num_classes).to(device)
    if (
        teacher_model is not None
        and contrastive_weight > 0.0
        and contrastive_use_teacher
    ):
        feature_dim = model.head.weight.shape[1] if hasattr(model.head, "weight") else num_classes
        teacher_contrastive_adapter = TeacherContrastiveAdapter(num_classes, feature_dim).to(device)
    if teacher_logit_adapter is not None:
        opt.add_param_group(
            {"params": teacher_logit_adapter.parameters(), "lr": distill_teacher_adapter_lr, "weight_decay": 0.0}
        )
    if teacher_contrastive_adapter is not None:
        opt.add_param_group(
            {"params": teacher_contrastive_adapter.parameters(), "lr": base_lr, "weight_decay": weight_decay}
        )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    ema_model = copy.deepcopy(model) if ema_decay > 0 else None
    teacher_feature_collector = TeacherFeatureCollector(teacher_model) if teacher_model is not None else None
    teacher_hint_adapter = (
        TeacherHintAdapter(num_classes, device)
        if teacher_feature_collector is not None and (hint_distill_weight > 0.0 or relational_weight > 0.0)
        else None
    )
    clause_dims = infer_clause_dimensions(model)
    teacher_clause_projector = (
        TeacherClauseProjector(clause_dims, device)
        if teacher_feature_collector is not None and clause_align_weight > 0.0
        else None
    )
    registered_hint_layers: Set[int] = set()
    registered_clause_layers: Set[int] = set()
    base_weights = {
        "teacher": distill_teacher_weight,
        "self": self_distill_weight,
        "hint": hint_distill_weight,
        "clause": clause_align_weight,
        "attn": attn_guidance_weight,
        "relational": relational_weight,
    }
    base_temps = {
        "teacher": distill_teacher_temp,
        "self": self_distill_temp,
        "hint": hint_distill_temp,
        "clause": clause_align_temp,
        "attn": attn_guidance_temp,
    }
    weight_scales = {
        "teacher": kd_stage2_teacher_scale,
        "self": kd_stage2_self_scale,
        "hint": kd_stage2_hint_scale,
        "clause": kd_stage2_clause_scale,
        "attn": kd_stage2_attn_scale,
        "relational": kd_stage2_rel_scale,
    }
    temp_scales = {
        "teacher": kd_stage2_teacher_temp_scale,
        "self": kd_stage2_self_temp_scale,
        "hint": kd_stage2_hint_temp_scale,
        "clause": kd_stage2_clause_temp_scale,
        "attn": kd_stage2_attn_temp_scale,
    }
    kd_scheduler = DistillationStageScheduler(
        stage_epoch=kd_stage_epochs,
        total_epochs=epochs,
        base_weights=base_weights,
        base_temps=base_temps,
        weight_scales=weight_scales,
        temp_scales=temp_scales,
        base_tau=tm_tau,
        target_tau=kd_stage2_tau,
    )
    current_tau_value = tm_tau
    teacher_aug_loader: Optional[DataLoader] = None
    if teacher_aug_path:
        pseudo_data = torch.load(teacher_aug_path, map_location="cpu")
        images = pseudo_data.get("images")
        logits_store = pseudo_data.get("logits")
        if images is None or logits_store is None:
            raise ValueError(f"Pseudo-label dataset at {teacher_aug_path} missing 'images' or 'logits'.")
        pseudo_dataset = TensorDataset(images, logits_store)
        teacher_aug_loader = DataLoader(
            pseudo_dataset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
    if clause_init_batches > 0 and teacher_feature_collector is not None:
        print(f"  Warming clause parameters with teacher for {clause_init_batches} batches...")
        warm_start_clauses_with_teacher(
            model,
            teacher_feature_collector,
            train_loader,
            device,
            batches=clause_init_batches,
            lr=clause_init_lr,
            temperature=clause_init_temp,
        )
        if ema_model is not None:
            ema_model.load_state_dict(model.state_dict())
    start = time.time()
    last_train_acc: Optional[float] = None
    best_test_acc: Optional[float] = None
    label = f"Transformer-{architecture.upper()}"
    collect_aux = aux_weight > 0
    need_component_hints = hint_distill_weight > 0.0
    need_relational = relational_weight > 0.0
    need_attention_guidance = attn_guidance_weight > 0.0
    collect_components = collect_aux or need_component_hints or need_relational
    component_history: List[Dict[str, Any]] = []
    writer = None
    if log_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=log_dir)
        except Exception as exc:  # pragma: no cover - tensorboard optional
            print(f"[WARN] TensorBoard writer unavailable: {exc}")
            writer = None

    steps_per_epoch = max(1, len(train_loader))
    warmup_steps = warmup_epochs * steps_per_epoch
    cycle_steps = max(0, lr_cycle_steps)
    need_features = contrastive_weight > 0.0
    mixup_schedule = (mixup_schedule or "constant").lower()
    cutmix_schedule = (cutmix_schedule or "constant").lower()

    for epoch in range(epochs):
        weight_schedule, temp_schedule, tau_override = kd_scheduler.compute(epoch)
        teacher_kd_weight = weight_schedule.get("teacher", distill_teacher_weight)
        self_kd_weight = weight_schedule.get("self", self_distill_weight)
        hint_weight_epoch = weight_schedule.get("hint", hint_distill_weight)
        clause_align_weight_epoch = weight_schedule.get("clause", clause_align_weight)
        attn_guidance_weight_epoch = weight_schedule.get("attn", attn_guidance_weight)
        relational_weight_epoch = weight_schedule.get("relational", relational_weight)
        teacher_temp_epoch = temp_schedule.get("teacher", distill_teacher_temp)
        self_temp_epoch = temp_schedule.get("self", self_distill_temp)
        hint_temp_epoch = temp_schedule.get("hint", hint_distill_temp)
        clause_temp_epoch = temp_schedule.get("clause", clause_align_temp)
        attn_temp_epoch = temp_schedule.get("attn", attn_guidance_temp)
        if tau_override is not None:
            anneal_ste_factor(model, tau_override)
            if ema_model is not None:
                anneal_ste_factor(ema_model, tau_override)
            current_tau_value = tau_override
        mixup_epoch_alpha = schedule_scalar(epoch, epochs, mixup_alpha, mixup_schedule)
        cutmix_epoch_alpha = schedule_scalar(epoch, epochs, cutmix_alpha, cutmix_schedule)
        global_step = epoch * steps_per_epoch
        if cycle_steps > 0:
            epoch_lr = cosine_with_restarts_lr(global_step, warmup_steps, base_lr, min_lr, cycle_steps)
            set_optimizer_lr(opt, epoch_lr)
        else:
            epoch_lr = update_cosine_warmup_lr(opt, epoch, epochs, base_lr, min_lr, warmup_epochs)
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        num_batches = 0
        total_distill = 0.0
        distill_count = 0
        total_contrast = 0.0
        contrast_count = 0
        total_teacher_kd = 0.0
        teacher_count = 0
        total_teacher_ce = 0.0
        teacher_ce_count = 0
        total_hint = 0.0
        hint_count = 0
        hint_component_totals: Dict[str, float] = defaultdict(float)
        hint_component_counts: Dict[str, int] = defaultdict(int)
        total_clause_align = 0.0
        clause_align_count = 0
        clause_align_component_totals: Dict[str, float] = defaultdict(float)
        clause_align_component_counts: Dict[str, int] = defaultdict(int)
        total_attn_guidance = 0.0
        attn_guidance_count = 0
        attn_guidance_component_totals: Dict[str, float] = defaultdict(float)
        attn_guidance_component_counts: Dict[str, int] = defaultdict(int)
        total_relational = 0.0
        relational_count = 0
        relational_component_totals: Dict[str, float] = defaultdict(float)
        relational_component_counts: Dict[str, int] = defaultdict(int)
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            if cycle_steps > 0:
                current_lr = cosine_with_restarts_lr(global_step, warmup_steps, base_lr, min_lr, cycle_steps)
                set_optimizer_lr(opt, current_lr)
                epoch_lr = current_lr
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x_aug, targets_a, targets_b, lam, aug_mode, mix_perm = apply_mixup_cutmix(
                x, y, mixup_epoch_alpha, cutmix_epoch_alpha
            )
            teacher_inputs = x_aug if distill_teacher_trainable else x
            opt.zero_grad(set_to_none=True)
            distill_term: Optional[torch.Tensor] = None
            contrast_term: Optional[torch.Tensor] = None
            teacher_ce_term: Optional[torch.Tensor] = None
            external_teacher_logits: Optional[torch.Tensor] = None
            teacher_features_batch: Dict[str, torch.Tensor] = {}
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                teacher_logits_for_loss: Optional[torch.Tensor] = None
                if teacher_model is not None and teacher_feature_collector is not None:
                    if distill_teacher_trainable:
                        teacher_logits, teacher_features_batch, teacher_logits_for_loss = teacher_feature_collector.collect(
                            teacher_inputs, retain_logits_grad=True
                        )
                        external_teacher_logits = teacher_logits.detach()
                    else:
                        with torch.no_grad():
                            primary_logits, primary_features, primary_logits_for_loss = teacher_feature_collector.collect(
                                x, retain_logits_grad=False
                            )
                            teacher_logits_for_loss = None
                            if lam is not None and mix_perm is not None:
                                secondary_logits, secondary_features, _ = teacher_feature_collector.collect(
                                    x[mix_perm], retain_logits_grad=False
                                )
                                external_teacher_logits = lam * primary_logits_for_loss + (1 - lam) * secondary_logits
                                teacher_features_batch = {}
                                for name, feat_a in primary_features.items():
                                    feat_b = secondary_features.get(name)
                                    if feat_b is None:
                                        teacher_features_batch[name] = feat_a
                                    else:
                                        teacher_features_batch[name] = lam * feat_a + (1 - lam) * feat_b
                            else:
                                external_teacher_logits = primary_logits_for_loss
                                teacher_features_batch = primary_features
                if teacher_logit_adapter is not None and external_teacher_logits is not None:
                    external_teacher_logits = teacher_logit_adapter(external_teacher_logits)
                outputs = model(
                    x_aug,
                    use_ste=True,
                    collect_diagnostics=collect_components,
                    return_features=need_features,
                )
                component_logits: Dict[str, torch.Tensor] = {}
                features: Optional[torch.Tensor]
                if collect_components and need_features:
                    logits, component_logits, features = outputs  # type: ignore[assignment]
                elif collect_components:
                    logits, component_logits = outputs  # type: ignore[assignment]
                    features = None
                elif need_features:
                    logits, features = outputs  # type: ignore[assignment]
                else:
                    logits = outputs  # type: ignore[assignment]
                    features = None
                clause_output_map: Dict[str, torch.Tensor] = (
                    collect_clause_outputs(model)
                    if (clause_align_weight_epoch > 0.0 or attn_guidance_weight_epoch > 0.0)
                    else {}
                )
                if lam is not None:
                    loss = lam * F.cross_entropy(logits, targets_a, label_smoothing=label_smoothing) + (
                        1 - lam
                    ) * F.cross_entropy(logits, targets_b, label_smoothing=label_smoothing)
                else:
                    loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
                if collect_aux and component_logits:
                    aux_loss = 0.0
                    for aux_logits in component_logits.values():
                        if lam is not None:
                            aux_loss += lam * F.cross_entropy(
                                aux_logits, targets_a, label_smoothing=label_smoothing
                            ) + (1 - lam) * F.cross_entropy(
                                aux_logits, targets_b, label_smoothing=label_smoothing
                            )
                        else:
                            aux_loss += F.cross_entropy(aux_logits, y, label_smoothing=label_smoothing)
                    aux_loss = aux_loss / max(len(component_logits), 1)
                    loss = loss + aux_weight * aux_loss
                if teacher_logits_for_loss is not None:
                    if lam is not None:
                        teacher_ce_term = lam * F.cross_entropy(
                            teacher_logits_for_loss, targets_a, label_smoothing=label_smoothing
                        ) + (1 - lam) * F.cross_entropy(
                            teacher_logits_for_loss, targets_b, label_smoothing=label_smoothing
                        )
                    else:
                        teacher_ce_term = F.cross_entropy(
                            teacher_logits_for_loss, y, label_smoothing=label_smoothing
                        )
                    loss = loss + teacher_ce_term
                teacher_hint_logits_map: Dict[str, torch.Tensor] = {}
                if (
                    teacher_hint_adapter is not None
                    and teacher_features_batch
                    and component_logits
                    and (need_component_hints or need_relational)
                ):
                    teacher_hint_logits_map = teacher_hint_adapter(teacher_features_batch)
                    new_hint_layers = teacher_hint_adapter.consume_new_layers()
                    for layer in new_hint_layers:
                        layer_id = id(layer)
                        if layer_id in registered_hint_layers:
                            continue
                        params = [p for p in layer.parameters() if p.requires_grad]
                        if params:
                            opt.add_param_group({"params": params, "lr": epoch_lr, "weight_decay": weight_decay})
                        registered_hint_layers.add(layer_id)
                hint_term_value: Optional[torch.Tensor] = None
                if hint_weight_epoch > 0.0 and teacher_hint_logits_map:
                    hint_losses: List[torch.Tensor] = []
                    for name, student_component in component_logits.items():
                        teacher_component = teacher_hint_logits_map.get(name)
                        if teacher_component is None:
                            continue
                        hint_loss_component = F.kl_div(
                            F.log_softmax(student_component / hint_temp_epoch, dim=1),
                            F.softmax(teacher_component / hint_temp_epoch, dim=1),
                            reduction="batchmean",
                        ) * (hint_temp_epoch ** 2)
                        hint_losses.append(hint_loss_component)
                        hint_component_totals[name] += hint_loss_component.detach().item()
                        hint_component_counts[name] += 1
                    if hint_losses:
                        hint_term_value = sum(hint_losses) / len(hint_losses)
                        loss = loss + hint_weight_epoch * hint_term_value
                relational_term: Optional[torch.Tensor] = None
                if relational_weight_epoch > 0.0 and teacher_hint_logits_map and component_logits:
                    relational_losses: List[torch.Tensor] = []
                    for name, student_component in component_logits.items():
                        teacher_component = teacher_hint_logits_map.get(name)
                        if teacher_component is None or student_component.size(0) < 2:
                            continue
                        student_feat = F.normalize(student_component, dim=1)
                        teacher_feat = F.normalize(teacher_component, dim=1)
                        student_sim = torch.matmul(student_feat, student_feat.transpose(0, 1))
                        teacher_sim = torch.matmul(teacher_feat, teacher_feat.transpose(0, 1))
                        relational_loss_component = F.mse_loss(student_sim, teacher_sim)
                        relational_losses.append(relational_loss_component)
                        relational_component_totals[name] += relational_loss_component.detach().item()
                        relational_component_counts[name] += 1
                    if relational_losses:
                        relational_term = sum(relational_losses) / len(relational_losses)
                        loss = loss + relational_weight_epoch * relational_term
                clause_align_term: Optional[torch.Tensor] = None
                if (
                    clause_align_weight_epoch > 0.0
                    and teacher_clause_projector is not None
                    and teacher_features_batch
                    and clause_output_map
                ):
                    teacher_clause_logits = teacher_clause_projector(teacher_features_batch)
                    new_clause_layers = teacher_clause_projector.consume_new_layers()
                    for layer in new_clause_layers:
                        layer_id = id(layer)
                        if layer_id in registered_clause_layers:
                            continue
                        params = [p for p in layer.parameters() if p.requires_grad]
                        if params:
                            opt.add_param_group({"params": params, "lr": epoch_lr, "weight_decay": weight_decay})
                        registered_clause_layers.add(layer_id)
                    clause_losses: List[torch.Tensor] = []
                    for name, student_clause in clause_output_map.items():
                        teacher_clause = teacher_clause_logits.get(name)
                        if teacher_clause is None:
                            continue
                        if student_clause.dim() == 3:
                            student_clause_flat = student_clause.mean(dim=1)
                        else:
                            student_clause_flat = student_clause.view(student_clause.size(0), -1)
                        student_probs = student_clause_flat.clamp_min(1e-6)
                        student_probs = student_probs / student_probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
                        student_log = torch.log(student_probs)
                        teacher_probs = F.softmax(teacher_clause / clause_temp_epoch, dim=1)
                        clause_loss_component = F.kl_div(
                            student_log,
                            teacher_probs,
                            reduction="batchmean",
                        ) * (clause_temp_epoch ** 2)
                        clause_losses.append(clause_loss_component)
                        clause_align_component_totals[name] += clause_loss_component.detach().item()
                        clause_align_component_counts[name] += 1
                    if clause_losses:
                        clause_align_term = sum(clause_losses) / len(clause_losses)
                        loss = loss + clause_align_weight_epoch * clause_align_term
                attn_guidance_term: Optional[torch.Tensor] = None
                if (
                    attn_guidance_weight_epoch > 0.0
                    and teacher_features_batch
                    and clause_output_map
                ):
                    attn_losses: List[torch.Tensor] = []
                    for name, student_clause in clause_output_map.items():
                        teacher_tensor = teacher_features_batch.get(name)
                        if teacher_tensor is None:
                            continue
                        teacher_attention = compute_teacher_token_attention(name, teacher_tensor)
                        if teacher_attention is None or teacher_attention.size(1) == 0:
                            continue
                        drop_first = student_clause.size(1) == teacher_attention.size(1) + 1
                        student_dist = compute_student_clause_token_distribution(
                            student_clause, drop_first_token=drop_first
                        )
                        if student_dist.size(1) != teacher_attention.size(1):
                            continue
                        teacher_probs = teacher_attention.detach().clamp_min(1e-6)
                        if attn_temp_epoch != 1.0:
                            teacher_probs = teacher_probs.pow(1.0 / attn_temp_epoch)
                            teacher_probs = teacher_probs / teacher_probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
                        student_dist = student_dist.clamp_min(1e-6)
                        student_log = torch.log(student_dist)
                        attn_loss_component = F.kl_div(
                            student_log,
                            teacher_probs,
                            reduction="batchmean",
                        )
                        attn_losses.append(attn_loss_component)
                        attn_guidance_component_totals[name] += attn_loss_component.detach().item()
                        attn_guidance_component_counts[name] += 1
                    if attn_losses:
                        attn_guidance_term = sum(attn_losses) / len(attn_losses)
                        loss = loss + attn_guidance_weight_epoch * attn_guidance_term
                if self_kd_weight > 0.0 and ema_model is not None:
                    with torch.no_grad():
                        teacher_out = ema_model(x_aug, use_ste=True)
                        teacher_logits = teacher_out[0] if isinstance(teacher_out, tuple) else teacher_out
                    distill_term = F.kl_div(
                        F.log_softmax(logits / self_temp_epoch, dim=1),
                        F.softmax(teacher_logits / self_temp_epoch, dim=1),
                        reduction="batchmean",
                    ) * (self_temp_epoch ** 2)
                    loss = loss + self_kd_weight * distill_term
                if external_teacher_logits is not None and teacher_kd_weight > 0.0:
                    kd_loss = F.kl_div(
                        F.log_softmax(logits / teacher_temp_epoch, dim=1),
                        F.softmax(external_teacher_logits / teacher_temp_epoch, dim=1),
                        reduction="batchmean",
                    ) * (teacher_temp_epoch ** 2)
                    loss = loss + teacher_kd_weight * kd_loss
                    total_teacher_kd += kd_loss.item()
                    teacher_count += 1
                if contrastive_weight > 0.0 and features is not None and features.numel() > 0:
                    contrast_inputs = features
                    contrast_labels = y
                    if (
                        teacher_contrastive_adapter is not None
                        and external_teacher_logits is not None
                        and (lam is None or lam <= 0.0 or lam >= 1.0)
                    ):
                        teacher_embeds = teacher_contrastive_adapter(external_teacher_logits)
                        contrast_inputs = torch.cat([features, teacher_embeds], dim=0)
                        contrast_labels = torch.cat([y, y], dim=0)
                    contrast_term = supervised_contrastive_loss(contrast_inputs, contrast_labels, temperature=contrastive_temp)
                    loss = loss + contrastive_weight * contrast_term
                reg_loss = model.pop_regularization_loss()
                if reg_loss is not None:
                    loss = loss + reg_loss
            if teacher_ce_term is not None:
                total_teacher_ce += teacher_ce_term.detach().item()
                teacher_ce_count += 1
            if hint_term_value is not None:
                total_hint += hint_term_value.detach().item()
                hint_count += 1
            if clause_align_term is not None:
                total_clause_align += clause_align_term.detach().item()
                clause_align_count += 1
            if attn_guidance_term is not None:
                total_attn_guidance += attn_guidance_term.detach().item()
                attn_guidance_count += 1
            if relational_term is not None:
                total_relational += relational_term.detach().item()
                relational_count += 1
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
            if distill_term is not None:
                total_distill += distill_term.detach().item()
                distill_count += 1
            if contrast_term is not None:
                total_contrast += contrast_term.detach().item()
                contrast_count += 1
            if report_train_acc:
                preds = logits.argmax(dim=1)
                epoch_correct += (preds == y).sum().item()
                epoch_total += y.size(0)
            global_step += 1
        dur = time.time() - epoch_start
        avg_loss = running_loss / max(1, num_batches)
        epoch_acc = None
        if report_train_acc and epoch_total > 0:
            epoch_acc = epoch_correct / epoch_total
            last_train_acc = epoch_acc

        eval_model = ema_model if ema_model is not None else model
        epoch_test_acc, component_acc, _, clause_usage = evaluate_transformer_components(
            eval_model,
            test_loader,
            device,
            collect_preds=False,
        )
        best_test_acc = epoch_test_acc if best_test_acc is None else max(best_test_acc, epoch_test_acc)
        avg_distill = (total_distill / distill_count) if distill_count > 0 else None
        avg_contrast = (total_contrast / contrast_count) if contrast_count > 0 else None
        avg_teacher_kd = (total_teacher_kd / teacher_count) if teacher_count > 0 else None
        avg_teacher_ce = (total_teacher_ce / teacher_ce_count) if teacher_ce_count > 0 else None
        avg_hint = (total_hint / hint_count) if hint_count > 0 else None
        hint_component_avg = {
            name: hint_component_totals[name] / hint_component_counts[name]
            for name in hint_component_totals
            if hint_component_counts[name] > 0
        }
        avg_clause_align = (total_clause_align / clause_align_count) if clause_align_count > 0 else None
        clause_align_component_avg = {
            name: clause_align_component_totals[name] / clause_align_component_counts[name]
            for name in clause_align_component_totals
            if clause_align_component_counts[name] > 0
        }
        avg_attn_guidance = (total_attn_guidance / attn_guidance_count) if attn_guidance_count > 0 else None
        attn_guidance_component_avg = {
            name: attn_guidance_component_totals[name] / attn_guidance_component_counts[name]
            for name in attn_guidance_component_totals
            if attn_guidance_component_counts[name] > 0
        }
        avg_relational = (total_relational / relational_count) if relational_count > 0 else None
        relational_component_avg = {
            name: relational_component_totals[name] / relational_component_counts[name]
            for name in relational_component_totals
            if relational_component_counts[name] > 0
        }
        teacher_eval_acc = None
        if teacher_model is not None:
            teacher_eval_acc = evaluate_classifier(teacher_model, test_loader, device)
        stage_weights = {
            "teacher": teacher_kd_weight,
            "self": self_kd_weight,
            "hint": hint_weight_epoch,
            "clause": clause_align_weight_epoch,
            "attn": attn_guidance_weight_epoch,
            "relational": relational_weight_epoch,
            "tau": current_tau_value,
        }
        stage_temps = {
            "teacher": teacher_temp_epoch,
            "self": self_temp_epoch,
            "hint": hint_temp_epoch,
            "clause": clause_temp_epoch,
            "attn": attn_temp_epoch,
        }
        component_history.append(
            {
                "epoch": epoch + 1,
                "test_accuracy": epoch_test_acc,
                "component_accuracy": component_acc,
                "mixup_alpha": mixup_epoch_alpha,
                "cutmix_alpha": cutmix_epoch_alpha,
                "distill_loss": avg_distill,
                "contrastive_loss": avg_contrast,
                "hint_loss": avg_hint,
                "clause_align_loss": avg_clause_align,
                "hint_components": hint_component_avg,
                "clause_align_components": clause_align_component_avg,
                "attn_guidance_loss": avg_attn_guidance,
                "attn_guidance_components": attn_guidance_component_avg,
                "relational_loss": avg_relational,
                "relational_components": relational_component_avg,
                "clause_usage": clause_usage,
                "teacher_kd_loss": avg_teacher_kd,
                "teacher_ce_loss": avg_teacher_ce,
                "teacher_accuracy": teacher_eval_acc,
                "stage_weights": stage_weights,
                "stage_temps": stage_temps,
            }
        )

        extras = {f"{name}_acc": acc for name, acc in component_acc.items()}
        extras["mixup_alpha"] = mixup_epoch_alpha
        extras["cutmix_alpha"] = cutmix_epoch_alpha
        extras["teacher_kd_weight"] = teacher_kd_weight
        extras["self_kd_weight"] = self_kd_weight
        extras["tau"] = current_tau_value
        if avg_distill is not None:
            extras["distill_loss"] = avg_distill
        if avg_contrast is not None:
            extras["contrast_loss"] = avg_contrast
        if avg_hint is not None:
            extras["hint_loss"] = avg_hint
        if avg_clause_align is not None:
            extras["clause_align_loss"] = avg_clause_align
        if avg_attn_guidance is not None:
            extras["attn_guidance_loss"] = avg_attn_guidance
        if avg_relational is not None:
            extras["relational_loss"] = avg_relational
        if avg_teacher_kd is not None:
            extras["teacher_kd_loss"] = avg_teacher_kd
        if avg_teacher_ce is not None:
            extras["teacher_ce_loss"] = avg_teacher_ce
        if teacher_eval_acc is not None:
            extras["teacher_acc"] = teacher_eval_acc
        if writer is not None:
            writer.add_scalar(f"{label}/loss/train", avg_loss, epoch)
            if epoch_acc is not None:
                writer.add_scalar(f"{label}/train/accuracy", epoch_acc, epoch)
            writer.add_scalar(f"{label}/test/accuracy", epoch_test_acc, epoch)
            writer.add_scalar(f"{label}/test/best_accuracy", best_test_acc, epoch)
            writer.add_scalar(f"{label}/augment/mixup_alpha", mixup_epoch_alpha, epoch)
            writer.add_scalar(f"{label}/augment/cutmix_alpha", cutmix_epoch_alpha, epoch)
            if avg_distill is not None:
                writer.add_scalar(f"{label}/aux/distillation_loss", avg_distill, epoch)
            if avg_contrast is not None:
                writer.add_scalar(f"{label}/aux/contrastive_loss", avg_contrast, epoch)
            if avg_hint is not None:
                writer.add_scalar(f"{label}/aux/hint_loss", avg_hint, epoch)
            if hint_component_avg:
                for name, value in hint_component_avg.items():
                    writer.add_scalar(f"{label}/aux/hint_components/{name}", value, epoch)
            if avg_clause_align is not None:
                writer.add_scalar(f"{label}/aux/clause_align_loss", avg_clause_align, epoch)
            if clause_align_component_avg:
                for name, value in clause_align_component_avg.items():
                    writer.add_scalar(f"{label}/aux/clause_align_components/{name}", value, epoch)
            if avg_attn_guidance is not None:
                writer.add_scalar(f"{label}/aux/attn_guidance_loss", avg_attn_guidance, epoch)
            if attn_guidance_component_avg:
                for name, value in attn_guidance_component_avg.items():
                    writer.add_scalar(f"{label}/aux/attn_guidance_components/{name}", value, epoch)
            if avg_relational is not None:
                writer.add_scalar(f"{label}/aux/relational_loss", avg_relational, epoch)
            if relational_component_avg:
                for name, value in relational_component_avg.items():
                    writer.add_scalar(f"{label}/aux/relational_components/{name}", value, epoch)
            if avg_teacher_kd is not None:
                writer.add_scalar(f"{label}/aux/teacher_kd_loss", avg_teacher_kd, epoch)
            if avg_teacher_ce is not None:
                writer.add_scalar(f"{label}/teacher/ce_loss", avg_teacher_ce, epoch)
            if teacher_eval_acc is not None:
                writer.add_scalar(f"{label}/teacher/accuracy", teacher_eval_acc, epoch)
            for name, value in component_acc.items():
                writer.add_scalar(f"{label}/components/{name}", value, epoch)
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

        if clause_specialize and clause_usage:
            model.apply_clause_head_specialization(clause_usage, clause_specialize_strength)

    train_time = time.time() - start
    eval_model = ema_model if ema_model is not None else model
    eval_acc, eval_component_acc, eval_preds, eval_clause_usage = evaluate_transformer_components(
        eval_model,
        test_loader,
        device,
        collect_preds=True,
    )
    teacher_eval_final = None
    if teacher_model is not None:
        teacher_eval_final = evaluate_classifier(teacher_model, test_loader, device)
    holdout_acc = None
    holdout_component_acc: Optional[Dict[str, float]] = None
    holdout_clause_usage: Optional[Dict[str, List[float]]] = None
    holdout_preds: List[int] = []
    teacher_holdout_acc = None
    if holdout_loader is not None:
        holdout_acc, holdout_component_acc, holdout_preds, holdout_clause_usage = evaluate_transformer_components(
            eval_model,
            holdout_loader,
            device,
            collect_preds=True,
        )
        if teacher_model is not None:
            teacher_holdout_acc = evaluate_classifier(teacher_model, holdout_loader, device)
    if holdout_loader is not None and holdout_acc is not None:
        test_acc = holdout_acc
        final_component_acc = holdout_component_acc or {}
        final_clause_usage = holdout_clause_usage or {}
        preds = holdout_preds
    else:
        test_acc = eval_acc
        final_component_acc = eval_component_acc
        final_clause_usage = eval_clause_usage
        preds = eval_preds
    if best_test_acc is None or test_acc > best_test_acc:
        best_test_acc = test_acc
    diagnostics = {
        "per_epoch": component_history,
        "final": final_component_acc,
        "clause_usage": final_clause_usage,
        "training": {
            "mixup_schedule": mixup_schedule,
            "cutmix_schedule": cutmix_schedule,
            "self_distill_weight": self_distill_weight,
            "hint_weight": hint_distill_weight,
            "hint_temp": hint_distill_temp,
            "clause_align_weight": clause_align_weight,
            "clause_align_temp": clause_align_temp,
            "attn_guidance_weight": attn_guidance_weight,
            "attn_guidance_temp": attn_guidance_temp,
            "relational_weight": relational_weight,
            "kd_stage_epoch": kd_stage_epochs,
            "kd_stage2_scales": {
                "teacher": kd_stage2_teacher_scale,
                "self": kd_stage2_self_scale,
                "hint": kd_stage2_hint_scale,
                "clause": kd_stage2_clause_scale,
                "attn": kd_stage2_attn_scale,
                "rel": kd_stage2_rel_scale,
            },
            "kd_stage2_temp_scales": {
                "teacher": kd_stage2_teacher_temp_scale,
                "self": kd_stage2_self_temp_scale,
                "hint": kd_stage2_hint_temp_scale,
                "clause": kd_stage2_clause_temp_scale,
                "attn": kd_stage2_attn_temp_scale,
            },
            "kd_stage2_tau": kd_stage2_tau,
            "clause_init_batches": clause_init_batches,
            "clause_init_lr": clause_init_lr,
            "clause_init_temp": clause_init_temp,
            "contrastive_weight": contrastive_weight,
            "contrastive_use_teacher": contrastive_use_teacher,
            "teacher_model": distill_teacher_timm or distill_teacher_checkpoint,
            "teacher_weight": distill_teacher_weight,
            "teacher_temp": distill_teacher_temp,
            "teacher_adapter": distill_teacher_adapter,
            "teacher_adapter_lr": distill_teacher_adapter_lr if distill_teacher_adapter else None,
            "teacher_aug_path": teacher_aug_path,
            "teacher_aug_batches": teacher_aug_batches,
        },
        "eval_accuracy": eval_acc,
        "eval_component_accuracy": eval_component_acc,
        "eval_clause_usage": eval_clause_usage,
    }
    if teacher_eval_final is not None:
        diagnostics["teacher_eval_accuracy"] = teacher_eval_final
    if holdout_loader is not None and holdout_acc is not None:
        diagnostics["holdout_accuracy"] = holdout_acc
        diagnostics["holdout_component_accuracy"] = holdout_component_acc
        diagnostics["holdout_clause_usage"] = holdout_clause_usage
        if teacher_holdout_acc is not None:
            diagnostics["teacher_holdout_accuracy"] = teacher_holdout_acc

    if clause_specialize and final_clause_usage:
        model.apply_clause_head_specialization(final_clause_usage, clause_specialize_strength)

    if visualize_dir and visualize_samples > 0 and test_dataset is not None:
        overlay_dir = Path(visualize_dir)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        label_names: List[str]
        if class_names is not None:
            label_names = list(class_names)
        else:
            label_names = [str(i) for i in range(num_classes)]
        sample_total = min(visualize_samples, len(test_dataset))
        for sample_idx in range(sample_total):
            sample = test_dataset[sample_idx]
            if isinstance(sample, tuple):
                sample_img, sample_label = sample[0], int(sample[1])
            else:
                sample_img, sample_label = sample, 0
            if not isinstance(sample_img, torch.Tensor):
                continue
            generate_transformer_overlay(
                eval_model,
                sample_img,
                sample_label,
                label_names,
                architecture,
                patch_size,
                image_size,
                normalize_stats,
                overlay_dir / f"sample_{sample_idx:05d}.png",
                device=device,
                topk=visualize_topk,
            )
    if writer is not None:
        writer.add_scalar(f"{label}/final/test_accuracy", test_acc, epochs)
        for name, value in final_component_acc.items():
            writer.add_scalar(f"{label}/final/components/{name}", value, epochs)
        writer.add_text(f"{label}/diagnostics_summary", json.dumps(diagnostics, indent=2))
        writer.flush()
        writer.close()
    total_samples = len(train_loader.dataset)
    total_steps = epochs * max(1, len(train_loader))
    throughput = (total_samples * epochs) / train_time if train_time > 0 else None
    profile = {
        "num_samples": total_samples,
        "epochs": epochs,
        "total_steps": total_steps,
        "samples_per_second": throughput,
        "max_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }
    if save_path:
        checkpoint_path = Path(save_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        teacher_kd_summary = None
        if diagnostics.get("per_epoch"):
            teacher_kd_summary = diagnostics["per_epoch"][-1].get("teacher_kd_loss")
        checkpoint = {
            "model_state": model.state_dict(),
            "model_kwargs": base_model_kwargs,
            "train_args": {
                "epochs": epochs,
                "base_lr": base_lr,
                "min_lr": min_lr,
                "weight_decay": weight_decay,
                "layer_decay": layer_decay,
                "warmup_epochs": warmup_epochs,
                "lr_cycle_steps": lr_cycle_steps,
                "mixup_alpha": mixup_alpha,
                "cutmix_alpha": cutmix_alpha,
                "mixup_schedule": mixup_schedule,
                "cutmix_schedule": cutmix_schedule,
                "label_smoothing": label_smoothing,
                "aux_weight": aux_weight,
                "self_distill_weight": self_distill_weight,
                "self_distill_temp": self_distill_temp,
                "hint_distill_weight": hint_distill_weight,
                "hint_distill_temp": hint_distill_temp,
                "clause_align_weight": clause_align_weight,
                "clause_align_temp": clause_align_temp,
                "attn_guidance_weight": attn_guidance_weight,
                "attn_guidance_temp": attn_guidance_temp,
                "relational_weight": relational_weight,
                "kd_stage_epoch": kd_stage_epochs,
                "kd_stage2_teacher_scale": kd_stage2_teacher_scale,
                "kd_stage2_teacher_temp_scale": kd_stage2_teacher_temp_scale,
                "kd_stage2_self_scale": kd_stage2_self_scale,
                "kd_stage2_self_temp_scale": kd_stage2_self_temp_scale,
                "kd_stage2_hint_scale": kd_stage2_hint_scale,
                "kd_stage2_hint_temp_scale": kd_stage2_hint_temp_scale,
                "kd_stage2_clause_scale": kd_stage2_clause_scale,
                "kd_stage2_clause_temp_scale": kd_stage2_clause_temp_scale,
                "kd_stage2_attn_scale": kd_stage2_attn_scale,
                "kd_stage2_attn_temp_scale": kd_stage2_attn_temp_scale,
                "kd_stage2_rel_scale": kd_stage2_rel_scale,
                "kd_stage2_tau": kd_stage2_tau,
                "clause_init_batches": clause_init_batches,
                "clause_init_lr": clause_init_lr,
                "clause_init_temp": clause_init_temp,
                "contrastive_weight": contrastive_weight,
                "contrastive_temp": contrastive_temp,
                "contrastive_use_teacher": contrastive_use_teacher,
                "teacher_aug_path": teacher_aug_path,
                "teacher_aug_batches": teacher_aug_batches,
                "clause_specialize": clause_specialize,
                "clause_specialize_strength": clause_specialize_strength,
                "distill_teacher_timm": distill_teacher_timm,
                "distill_teacher_checkpoint": distill_teacher_checkpoint,
                "distill_teacher_weight": distill_teacher_weight,
                "distill_teacher_temp": distill_teacher_temp,
                "distill_teacher_adapter": distill_teacher_adapter,
                "distill_teacher_adapter_lr": distill_teacher_adapter_lr if distill_teacher_adapter else None,
            },
            "train_metrics": {
                "train_accuracy": last_train_acc,
                "test_accuracy": test_acc,
                "best_test_accuracy": best_test_acc,
                "teacher_kd_loss": teacher_kd_summary,
                "train_time": train_time,
            },
            "meta": {
                "architecture": architecture,
                "backend": backend,
                "timestamp": time.time(),
                "image_size": image_size,
                "num_classes": num_classes,
            },
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"  Saved transformer checkpoint to {checkpoint_path}")
    return label, last_train_acc, test_acc, train_time, preds, diagnostics, best_test_acc, profile


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

    dataset_root = args.dataset_root
    imagenet_train_dir: Optional[str] = None
    imagenet_val_dir: Optional[str] = None

    pin_memory = args.pin_memory
    if pin_memory is None:
        pin_memory = device.type != "cpu"

    persistent_workers = args.persistent_workers
    if persistent_workers is None:
        persistent_workers = args.num_workers > 0
    if args.num_workers == 0:
        persistent_workers = False

    target_channels = args.tm_target_channels if args.tm_target_channels is not None else dataset_cfg["input_channels"]
    target_size = tuple(args.tm_target_size) if args.tm_target_size is not None else tuple(dataset_cfg["image_size"])
    imagenet_mode = dataset_key == "imagenet"
    if imagenet_mode and dataset_root == DEFAULT_DATA_ROOT:
        dataset_root = DEFAULT_IMAGENET_ROOT
        args.dataset_root = dataset_root
    if imagenet_mode:
        target_size = (args.imagenet_crop, args.imagenet_crop)
        imagenet_train_dir = args.imagenet_train_dir or os.path.join(dataset_root, "train")
        imagenet_val_dir = args.imagenet_val_dir or os.path.join(dataset_root, "val")
        if args.write_bool_dataset:
            print("[WARN] Boolean dataset export disabled for ImageNet.")
            args.write_bool_dataset = False

    test_normalize_stats: Optional[Tuple[Sequence[float], Sequence[float]]] = None
    holdout_loader: Optional[DataLoader] = None
    holdout_dataset: Optional[Dataset] = None
    eval_dataset: Optional[Dataset] = None

    if imagenet_mode:
        crop_size = args.imagenet_crop
        resize_size = args.imagenet_resize
        train_steps: List[Any] = [
            transforms.RandomResizedCrop(crop_size, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
        ]
        if args.randaugment:
            train_steps.append(transforms.RandAugment(num_ops=args.randaugment_n, magnitude=args.randaugment_m))
        train_steps.extend([transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
        train_transform = transforms.Compose(train_steps)
        test_transform = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        test_normalize_stats = (IMAGENET_MEAN, IMAGENET_STD)
    else:
        base_train_transform = dataset_cfg.get("train_transform", dataset_cfg.get("transform"))
        base_test_transform = dataset_cfg.get("test_transform", dataset_cfg.get("transform"))
        resize_target = target_size if tuple(dataset_cfg["image_size"]) != target_size else None
        train_transform = _build_transform(
            base_train_transform,
            target_size=resize_target,
            apply_randaugment=args.randaugment,
            randaugment_n=args.randaugment_n,
            randaugment_m=args.randaugment_m,
            add_normalize=None,
        )
        test_transform = _build_transform(
            base_test_transform,
            target_size=resize_target,
            apply_randaugment=False,
            randaugment_n=args.randaugment_n,
            randaugment_m=args.randaugment_m,
            add_normalize=None,
        )
        test_normalize_stats = _extract_normalize(test_transform)

    with profile_block("load-dataset"):
        if imagenet_mode:
            if imagenet_train_dir is None or imagenet_val_dir is None:
                raise FileNotFoundError("ImageNet directories must be specified for training and validation.")
            if not os.path.isdir(imagenet_train_dir):
                raise FileNotFoundError(f"ImageNet train directory not found: {imagenet_train_dir}")
            if not os.path.isdir(imagenet_val_dir):
                raise FileNotFoundError(f"ImageNet validation directory not found: {imagenet_val_dir}")
            train_ds = dataset_cfg["train_class"](imagenet_train_dir, transform=train_transform)
            test_ds = dataset_cfg["test_class"](imagenet_val_dir, transform=test_transform)
        else:
            train_ds = dataset_cfg["train_class"](
                root=args.dataset_root,
                train=True,
                download=args.download,
                transform=train_transform,
            )
            test_ds = dataset_cfg["test_class"](
                root=args.dataset_root,
                train=False,
                download=args.download,
                transform=test_transform,
            )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        eval_dataset = test_ds
        holdout_fraction = max(0.0, min(1.0, float(args.test_holdout_fraction)))
        if holdout_fraction > 0.0 and len(test_ds) > 1:
            proposed_size = int(round(len(test_ds) * holdout_fraction))
            holdout_size = max(1, min(proposed_size, len(test_ds) - 1))
            eval_size = len(test_ds) - holdout_size
            split_seed = (
                args.test_holdout_seed
                if args.test_holdout_seed is not None
                else (args.seed if args.seed is not None else 0)
            )
            generator = torch.Generator()
            generator.manual_seed(split_seed)
            eval_dataset, holdout_dataset = random_split(test_ds, [eval_size, holdout_size], generator=generator)
        else:
            holdout_fraction = 0.0
        test_loader = DataLoader(
            eval_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        if holdout_dataset is not None:
            holdout_loader = DataLoader(
                holdout_dataset,
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

    if eval_dataset is None:
        eval_dataset = test_ds

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

    teacher_baseline_summary: Optional[Dict[str, Any]] = None
    if args.teacher_baseline:
        with profile_block("teacher-baseline"):
            teacher_baseline_summary = run_teacher_baseline(
                args=args,
                dataset_key=dataset_key,
                dataset_cfg=dataset_cfg,
                device=device,
                data_root=dataset_root,
                download=args.download,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                imagenet_mode=imagenet_mode,
                imagenet_train_dir=imagenet_train_dir,
                imagenet_val_dir=imagenet_val_dir,
            )
        results["teacher_baseline"] = teacher_baseline_summary

    for model_key in selected_models:
        with profile_block(f"variant-{model_key}"):
            diagnostics: Optional[Dict[str, Any]] = None
            if model_key == "tm":
                tm_variant_min_lr = tm_min_lr
                if args.min_lr is None and args.tm_impl.lower() == "stcm":
                    tm_variant_min_lr = tm_base_lr * 0.1
                label, train_acc, test_acc, train_time, preds, bundle, best_epoch_test_acc, profile = run_variant_tm(
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
                    min_lr=tm_variant_min_lr,
                    warmup_epochs=warmup_epochs,
                    weight_decay=weight_decay,
                    gradient_centralize=args.gradient_centralize,
                    label_smoothing=args.label_smoothing,
                    lr_cycle_steps=args.lr_cycle_steps,
                    tm_prior_template=args.tm_prior_template,
                    tm_clause_dropout=args.tm_clause_dropout,
                    tm_literal_dropout=args.tm_literal_dropout,
                    stcm_operator=args.stcm_operator,
                    stcm_ternary_voting=args.stcm_ternary_voting,
                    stcm_ternary_band=args.stcm_ternary_band,
                    stcm_ste_temperature=args.stcm_ste_temperature,
                )
                variant_classes = args.num_classes
            elif model_key == "stcm":
                stcm_min_lr = tm_min_lr
                if args.min_lr is None:
                    stcm_min_lr = tm_base_lr * 0.1
                label, train_acc, test_acc, train_time, preds, bundle, best_epoch_test_acc, profile = run_variant_tm(
                    tm_train_loader,
                    tm_test_loader,
                    device,
                    epochs=args.epochs,
                    report_train_acc=args.report_train_acc,
                    report_epoch_acc=args.report_epoch_acc,
                    report_epoch_test=args.report_epoch_test,
                    n_features=tm_n_features,
                    prepare_fn=tm_prepare_fn,
                    tm_impl="stcm",
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
                    min_lr=stcm_min_lr,
                    warmup_epochs=warmup_epochs,
                    weight_decay=weight_decay,
                    gradient_centralize=args.gradient_centralize,
                    label_smoothing=args.label_smoothing,
                    lr_cycle_steps=args.lr_cycle_steps,
                    tm_prior_template=args.tm_prior_template,
                    tm_clause_dropout=args.tm_clause_dropout,
                    tm_literal_dropout=args.tm_literal_dropout,
                    stcm_operator=args.stcm_operator,
                    stcm_ternary_voting=args.stcm_ternary_voting,
                    stcm_ternary_band=args.stcm_ternary_band,
                    stcm_ste_temperature=args.stcm_ste_temperature,
                )
                variant_classes = args.num_classes
            elif model_key == "deep_tm":
                label, train_acc, test_acc, train_time, preds, bundle, best_epoch_test_acc, profile = run_variant_deeptm(
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
                    label_smoothing=args.label_smoothing,
                    lr_cycle_steps=args.lr_cycle_steps,
                    tm_prior_template=args.tm_prior_template,
                    clause_dropout=args.tm_clause_dropout,
                    literal_dropout=args.tm_literal_dropout,
                    use_ste_train=False,
                    use_ste_eval=True,
                )
                variant_classes = args.num_classes
            elif model_key == "deep_stcm":
                stcm_layer_kwargs = {
                    "ternary_band": args.stcm_ternary_band,
                    "ste_temperature": args.stcm_ste_temperature,
                }
                deep_stcm_min_lr = deeptm_min_lr
                if args.min_lr is None:
                    deep_stcm_min_lr = deeptm_base_lr * 0.1
                label, train_acc, test_acc, train_time, preds, bundle, best_epoch_test_acc, profile = run_variant_deeptm(
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
                    min_lr=deep_stcm_min_lr,
                    warmup_epochs=warmup_epochs,
                    weight_decay=weight_decay,
                    gradient_centralize=args.gradient_centralize,
                    label_smoothing=args.label_smoothing,
                    lr_cycle_steps=args.lr_cycle_steps,
                    tm_prior_template=args.tm_prior_template,
                    layer_cls=FuzzyPatternTM_STCM,
                    layer_operator=args.stcm_operator,
                    layer_ternary_voting=args.stcm_ternary_voting,
                    layer_extra_kwargs=stcm_layer_kwargs,
                    label_override="Deep-STCM",
                    clause_dropout=args.tm_clause_dropout,
                    literal_dropout=args.tm_literal_dropout,
                    use_ste_train=False,
                    use_ste_eval=False,
                )
                variant_classes = args.num_classes
            elif model_key == "hybrid":
                hybrid_classes = args.hybrid_classes if args.hybrid_classes is not None else args.num_classes
                label, train_acc, test_acc, train_time, preds, bundle, best_epoch_test_acc, profile = run_variant_hybrid(
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
                    label_smoothing=args.label_smoothing,
                    tm_prior_template=args.tm_prior_template,
                    lr_cycle_steps=args.lr_cycle_steps,
                )
                variant_classes = hybrid_classes
            elif model_key == "transformer":
                label, train_acc, test_acc, train_time, preds, diagnostics, best_epoch_test_acc, profile = run_variant_transformer(
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
                    drop_path_schedule=args.transformer_drop_path_schedule,
                    dropout=args.transformer_dropout,
                    tm_clauses=transformer_tm_clauses_cfg,
                    tm_tau=args.tm_tau,
                    use_cls_token=args.transformer_use_cls,
                    pool=args.transformer_pool,
                    grad_checkpoint=args.transformer_grad_checkpoint,
                    ema_decay=args.transformer_ema_decay,
                    mixup_alpha=args.mixup_alpha,
                    cutmix_alpha=args.cutmix_alpha,
                    mixup_schedule=args.mixup_schedule,
                    cutmix_schedule=args.cutmix_schedule,
                    base_lr=transformer_base_lr,
                    min_lr=transformer_min_lr,
                    warmup_epochs=warmup_epochs,
                    weight_decay=weight_decay,
                    layer_decay=args.lr_layer_decay,
                    lr_cycle_steps=args.lr_cycle_steps,
                    gradient_centralize=args.gradient_centralize,
                    aux_weight=args.transformer_aux_weight,
                    label_smoothing=args.label_smoothing,
                    auto_clause=args.transformer_auto_clause,
                    clause_target=args.transformer_clause_target,
                    auto_clause_batches=args.transformer_auto_clause_batches,
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
                    tm_prior_template=args.tm_prior_template,
                    self_distill_weight=args.transformer_self_distill_weight,
                    self_distill_temp=args.transformer_self_distill_temp,
                    hint_distill_weight=args.transformer_hint_weight,
                    hint_distill_temp=args.transformer_hint_temp,
                    clause_align_weight=args.transformer_clause_align_weight,
                    clause_align_temp=args.transformer_clause_align_temp,
                    attn_guidance_weight=args.transformer_attn_guidance_weight,
                    attn_guidance_temp=args.transformer_attn_guidance_temp,
                    relational_weight=args.transformer_relational_weight,
                    kd_stage_epochs=args.transformer_kd_stage_epochs,
                    kd_stage2_teacher_scale=args.transformer_kd_stage2_teacher_scale,
                    kd_stage2_teacher_temp_scale=args.transformer_kd_stage2_teacher_temp_scale,
                    kd_stage2_self_scale=args.transformer_kd_stage2_self_scale,
                    kd_stage2_self_temp_scale=args.transformer_kd_stage2_self_temp_scale,
                    kd_stage2_hint_scale=args.transformer_kd_stage2_hint_scale,
                    kd_stage2_hint_temp_scale=args.transformer_kd_stage2_hint_temp_scale,
                    kd_stage2_clause_scale=args.transformer_kd_stage2_clause_scale,
                    kd_stage2_clause_temp_scale=args.transformer_kd_stage2_clause_temp_scale,
                    kd_stage2_attn_scale=args.transformer_kd_stage2_attn_scale,
                    kd_stage2_attn_temp_scale=args.transformer_kd_stage2_attn_temp_scale,
                    kd_stage2_rel_scale=args.transformer_kd_stage2_rel_scale,
                    kd_stage2_tau=args.transformer_kd_stage2_tau,
                    clause_init_batches=args.transformer_clause_init_batches,
                    clause_init_lr=args.transformer_clause_init_lr,
                    clause_init_temp=args.transformer_clause_init_temp,
                    contrastive_weight=args.transformer_contrastive_weight,
                    contrastive_temp=args.transformer_contrastive_temp,
                    contrastive_use_teacher=args.transformer_contrastive_use_teacher,
                    teacher_aug_path=args.teacher_aug_path,
                    teacher_aug_batches=args.teacher_aug_batches,
                    use_flash_attention=args.transformer_use_flash_attn,
                    use_residual_attention=args.transformer_residual_attn,
                    log_dir=args.transformer_log_dir,
                    relative_position_type=args.transformer_relative_pos,
                    visualize_dir=args.visualize_overlays,
                    visualize_samples=args.visualize_samples,
                    visualize_topk=args.visualize_topk,
                    class_names=getattr(train_ds, "classes", None),
                    test_dataset=eval_dataset,
                    holdout_loader=holdout_loader,
                    normalize_stats=test_normalize_stats,
                    clause_specialize=args.transformer_clause_specialize,
                    clause_specialize_strength=args.transformer_clause_specialize_strength,
                    distill_teacher_timm=args.distill_teacher_timm,
                    distill_teacher_checkpoint=args.distill_teacher_checkpoint,
                    distill_teacher_weight=args.distill_teacher_weight,
                    distill_teacher_temp=args.distill_teacher_temp,
                    distill_teacher_trainable=args.distill_teacher_trainable,
                    distill_teacher_lr=args.distill_teacher_lr,
                    distill_teacher_adapter=args.distill_teacher_adapter,
                    distill_teacher_adapter_lr=args.distill_teacher_adapter_lr,
                            save_path=args.transformer_save_path,
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
            "profile": profile,
        }
        if diagnostics is not None:
            results[model_key]["diagnostics"] = diagnostics
            if "eval_accuracy" in diagnostics:
                results[model_key]["eval_accuracy"] = diagnostics["eval_accuracy"]
            if "holdout_accuracy" in diagnostics:
                results[model_key]["holdout_accuracy"] = diagnostics["holdout_accuracy"]
            if "teacher_eval_accuracy" in diagnostics:
                results[model_key]["teacher_eval_accuracy"] = diagnostics["teacher_eval_accuracy"]
            if "teacher_holdout_accuracy" in diagnostics:
                results[model_key]["teacher_holdout_accuracy"] = diagnostics["teacher_holdout_accuracy"]
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
            results[model_key]["clause_usage"] = diagnostics.get("clause_usage")

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
        profile = info.get("profile", {})
        throughput = profile.get("samples_per_second")
        if throughput:
            acc_parts.append(f"throughput={throughput:.1f}/s")
        max_mem = profile.get("max_memory_bytes")
        if max_mem:
            acc_parts.append(f"max_mem={max_mem / (1024**2):.1f}MB")
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

