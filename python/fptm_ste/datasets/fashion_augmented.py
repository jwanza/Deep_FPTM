import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.transforms import functional as TF


Tensor = torch.Tensor


@dataclass(frozen=True)
class AugmentationRecipe:
    """Augmentation specification applied to datasets prior to booleanisation."""

    name: str
    fn: Callable[[Tensor], Tensor]


def _identity(img: Tensor) -> Tensor:
    return img


def _zoom(scale: float) -> Callable[[Tensor], Tensor]:
    def _apply(img: Tensor) -> Tensor:
        tensor = img.unsqueeze(0)
        transformed = TF.affine(tensor, angle=0.0, translate=[0, 0], scale=scale, shear=[0.0, 0.0])
        return transformed.squeeze(0)

    return _apply


def _rotate(deg: float) -> Callable[[Tensor], Tensor]:
    def _apply(img: Tensor) -> Tensor:
        tensor = img.unsqueeze(0)
        rotated = TF.rotate(tensor, angle=deg, fill=0.0)
        cropped = TF.center_crop(rotated, [img.size(-2), img.size(-1)])
        return cropped.squeeze(0)

    return _apply


def _flip_vertical(img: Tensor) -> Tensor:
    return TF.vflip(img)


DEFAULT_AUGMENTATIONS: Tuple[AugmentationRecipe, ...] = (
    AugmentationRecipe("identity", _identity),
    AugmentationRecipe("zoom_0.9", _zoom(0.9)),
    AugmentationRecipe("zoom_1.1", _zoom(1.1)),
    AugmentationRecipe("rotate_-6", _rotate(-6.0)),
    AugmentationRecipe("rotate_6", _rotate(6.0)),
    AugmentationRecipe("flip_y", _flip_vertical),
)


@dataclass(frozen=True)
class PreprocessConfig:
    name: str
    image_size: Tuple[int, int]
    in_channels: int
    to_grayscale: bool = True
    augmentations: Tuple[AugmentationRecipe, ...] = DEFAULT_AUGMENTATIONS


DEFAULT_PREPROCESS_CONFIGS: Dict[str, PreprocessConfig] = {
    "fashionmnist": PreprocessConfig(
        name="fashionmnist",
        image_size=(28, 28),
        in_channels=1,
        to_grayscale=True,
        augmentations=DEFAULT_AUGMENTATIONS,
    ),
    "mnist": PreprocessConfig(
        name="mnist",
        image_size=(28, 28),
        in_channels=1,
        to_grayscale=True,
        augmentations=(DEFAULT_AUGMENTATIONS[0],),
    ),
    "cifar10": PreprocessConfig(
        name="cifar10",
        image_size=(32, 32),
        in_channels=3,
        to_grayscale=False,
        augmentations=DEFAULT_AUGMENTATIONS,
    ),
}


def _dataset_to_tensor(ds: VisionDataset, config: PreprocessConfig) -> Tuple[Tensor, torch.Tensor]:
    xs: List[Tensor] = []
    ys: List[int] = []
    to_tensor = transforms.ToTensor()
    for img, label in ds:
        if isinstance(img, torch.Tensor):
            tensor_img = img.clone().detach()
        else:
            tensor_img = to_tensor(img)

        if tensor_img.dim() == 2:
            tensor_img = tensor_img.unsqueeze(0)

        tensor_img = tensor_img.float()

        channels = tensor_img.size(0)

        if config.to_grayscale and channels > 1:
            tensor_img = TF.rgb_to_grayscale(tensor_img)
            channels = 1

        if channels == 1 and config.in_channels > 1:
            tensor_img = tensor_img.repeat(config.in_channels, 1, 1)
        elif channels > 1 and config.in_channels == 1:
            tensor_img = TF.rgb_to_grayscale(tensor_img)
        elif channels != config.in_channels:
            raise ValueError(
                f"Input tensor has {channels} channels but config expects {config.in_channels}."
            )

        if tensor_img.size(-2) != config.image_size[0] or tensor_img.size(-1) != config.image_size[1]:
            tensor_img = TF.resize(tensor_img, config.image_size, antialias=True)

        xs.append(tensor_img)
        ys.append(int(label))
    stacked = torch.stack(xs, dim=0).float()
    return stacked, torch.tensor(ys, dtype=torch.int64)


def _apply_augmentations(images: Tensor, augmentations: Sequence[AugmentationRecipe]) -> Tensor:
    augmented: List[Tensor] = []
    for aug in augmentations:
        augmented.append(torch.stack([aug.fn(img) for img in images], dim=0))
    return torch.cat(augmented, dim=0)


def _rotation90(kernel: Tensor) -> Tensor:
    return torch.rot90(kernel, k=1, dims=(-2, -1))


def _prepare_kernels(device: torch.device = torch.device("cpu")) -> Dict[str, Tuple[Tensor, int]]:
    kernels: Dict[str, Tuple[Tensor, int]] = {}
    kx3 = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device)
    kx5 = torch.tensor(
        [[0, 1, 2, 3, 4], [-1, 0, 2, 3, 3], [-2, -2, 0, 2, 2], [-3, -3, -2, 0, 1], [-4, -3, -2, -1, 0]],
        dtype=torch.float32,
        device=device,
    )
    kx7 = torch.tensor(
        [
            [-3, -2, -1, 0, 1, 2, 3],
            [-4, -3, -2, 0, 2, 3, 4],
            [-5, -4, -3, 0, 3, 4, 5],
            [-6, -5, -4, 0, 4, 5, 6],
            [-5, -4, -3, 0, 3, 4, 5],
            [-4, -3, -2, 0, 2, 3, 4],
            [-3, -2, -1, 0, 1, 2, 3],
        ],
        dtype=torch.float32,
        device=device,
    )
    kx9 = torch.tensor([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=torch.float32, device=device)

    kernels["x3"] = (kx3, 1)
    kernels["y3"] = (_rotation90(kx3), 1)
    kernels["x5"] = (kx5, 2)
    kernels["y5"] = (_rotation90(kx5), 2)
    kernels["x7"] = (kx7, 3)
    kernels["y7"] = (_rotation90(kx7), 3)
    kernels["x9"] = (kx9, 1)
    kernels["y9"] = (_rotation90(kx9), 1)
    return kernels


def _full_convolution(images: Tensor, kernel: Tensor, crop: int) -> Tensor:
    batch, height, width = images.shape
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    pad = kernel.size(-1) - 1
    padded = F.pad(images.unsqueeze(1), (pad, pad, pad, pad))
    conv = F.conv2d(padded, kernel)
    if crop > 0:
        conv = conv[..., crop:-crop, crop:-crop]
    return conv.view(batch, height, width)


def _quantiles_for_tensor(values: Tensor, qs: Iterable[float]) -> Dict[float, float]:
    out: Dict[float, float] = {}
    flat = values.reshape(-1)
    for q in qs:
        if flat.numel() == 0:
            out[q] = 0.0
        else:
            out[q] = float(torch.quantile(flat, torch.tensor(q, dtype=torch.float32, device=flat.device)))
    return out


def _safe_quantile(values: Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    try:
        return float(torch.quantile(values, torch.tensor(q, dtype=torch.float32, device=values.device)))
    except RuntimeError:
        return float(np.quantile(values.detach().cpu().numpy(), q))


def _compute_thresholds(
    raw: Tensor,
    kernels: Dict[str, Tuple[Tensor, int]],
) -> Tuple[List[Dict[str, Dict[str, Dict[str, float]]]], List[Dict[str, Tensor]]]:
    thresholds_all: List[Dict[str, Dict[str, Dict[str, float]]]] = []
    caches_all: List[Dict[str, Tensor]] = []
    for channel in range(raw.size(1)):
        channel_tensor = raw[:, channel, :, :]
        raw_pos = channel_tensor[channel_tensor > 0]
        thresholds: Dict[str, Dict[str, Dict[str, float]]] = {
            "raw": {
                "pos": {
                    "0.25": _safe_quantile(raw_pos, 0.25),
                    "0.50": _safe_quantile(raw_pos, 0.50),
                    "0.75": _safe_quantile(raw_pos, 0.75),
                }
            }
        }
        cache: Dict[str, Tensor] = {}
        pos_queries = [0.25, 0.34, 0.5, 0.75]
        neg_queries = [1 - q for q in pos_queries]
        for name, (kernel, crop) in kernels.items():
            conv = _full_convolution(channel_tensor, kernel, crop)
            cache[name] = conv
            pos_vals = conv[conv > 0]
            neg_vals = conv[conv < 0]
            thresholds[name] = {
                "pos": {f"{q:.2f}": _safe_quantile(pos_vals, q) for q in pos_queries},
                "neg": {f"{q:.2f}": _safe_quantile(neg_vals, q) for q in neg_queries},
            }
        thresholds_all.append(thresholds)
        caches_all.append(cache)
    return thresholds_all, caches_all


def _booleanize_batch(
    raw: Tensor,
    kernels: Dict[str, Tuple[Tensor, int]],
    thresholds: Sequence[Dict[str, Dict[str, Dict[str, float]]]],
) -> Tensor:
    features: List[Tensor] = []
    batch = raw.size(0)
    for channel in range(raw.size(1)):
        channel_tensor = raw[:, channel, :, :]
        channel_thresholds = thresholds[channel]

        raw_flat = channel_tensor.reshape(batch, -1)
        features.append((raw_flat > 0).to(torch.uint8))
        for key in ("0.25", "0.50", "0.75"):
            thresh = channel_thresholds["raw"]["pos"][key]
            features.append((raw_flat > thresh).to(torch.uint8))

        for name_order in ("x3", "y3", "x5", "y5", "x7", "y7", "x9", "y9"):
            kernel, crop = kernels[name_order]
            conv = _full_convolution(channel_tensor, kernel, crop)
            conv_flat = conv.reshape(batch, -1)
            features.extend(
                [
                    (conv_flat > 0).to(torch.uint8),
                    (conv_flat > channel_thresholds[name_order]["pos"]["0.25"]).to(torch.uint8),
                    (conv_flat > channel_thresholds[name_order]["pos"]["0.34"]).to(torch.uint8),
                    (conv_flat > channel_thresholds[name_order]["pos"]["0.50"]).to(torch.uint8),
                    (conv_flat > channel_thresholds[name_order]["pos"]["0.75"]).to(torch.uint8),
                    (conv_flat < channel_thresholds[name_order]["neg"]["0.75"]).to(torch.uint8),
                    (conv_flat < channel_thresholds[name_order]["neg"]["0.66"]).to(torch.uint8),
                    (conv_flat < channel_thresholds[name_order]["neg"]["0.50"]).to(torch.uint8),
                    (conv_flat < channel_thresholds[name_order]["neg"]["0.25"]).to(torch.uint8),
                ]
            )
    return torch.cat(features, dim=1)


def _pack_bits(array: np.ndarray, num_bits: int) -> np.ndarray:
    packed = np.packbits(array, axis=1)
    expected_cols = (num_bits + 7) // 8
    if packed.shape[1] != expected_cols:
        raise RuntimeError(f"Packed feature width mismatch: got {packed.shape[1]}, expected {expected_cols}")
    return packed


def _save_memmap(path: Path, shape: Tuple[int, int], dtype: np.dtype) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(path, mode="w+", dtype=dtype, shape=shape)


def _write_packed_features(
    out_file: Path,
    images: Tensor,
    kernels: Dict[str, Tuple[Tensor, int]],
    thresholds: Dict[str, Dict[str, Dict[str, float]]],
    batch_size: int,
    num_bits: int,
) -> None:
    packed_cols = (num_bits + 7) // 8
    mmap = _save_memmap(out_file, (images.size(0), packed_cols), np.uint8)
    try:
        start = 0
        while start < images.size(0):
            end = min(start + batch_size, images.size(0))
            batch = images[start:end].to(torch.float32)
            bools = _booleanize_batch(batch, kernels, thresholds)
            packed = _pack_bits(bools.cpu().numpy().astype(np.uint8), num_bits)
            mmap[start:end, :] = packed
            start = end
    finally:
        mmap.flush()
        del mmap


class PackedBooleanDataset(Dataset):
    """Dataset reading bit-packed boolean features from disk."""

    def __init__(self, data_path: Path, labels: np.ndarray, num_bits: int):
        self.data_path = data_path
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.num_bits = num_bits
        packed_cols = (num_bits + 7) // 8
        self._memmap = np.memmap(data_path, mode="r", dtype=np.uint8, shape=(len(labels), packed_cols))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        packed = self._memmap[idx]
        bits = np.unpackbits(packed, axis=0)[: self.num_bits]
        features = torch.from_numpy(bits.astype(np.float32))
        label = self.labels[idx]
        return features, label

    def close(self) -> None:
        del self._memmap


@dataclass
class FashionAugmentedBundle:
    train_dataset: PackedBooleanDataset
    test_dataset: PackedBooleanDataset
    num_bits: int
    thresholds: Dict[str, Dict[str, Dict[str, float]]]
    augmentation_names: List[str]
    config_name: str


def _meta_path(cache_dir: Path) -> Path:
    return cache_dir / "fashion_augmented_meta.json"


def _load_meta(cache_dir: Path) -> Optional[Dict]:
    meta_file = _meta_path(cache_dir)
    if meta_file.exists():
        with meta_file.open("r") as fh:
            return json.load(fh)
    return None


def _save_meta(
    cache_dir: Path,
    *,
    num_bits: int,
    thresholds: Dict[str, Dict[str, Dict[str, float]]],
    augmentation_names: Sequence[str],
    config_name: str,
    train_shape: Tuple[int, int],
    test_shape: Tuple[int, int],
) -> None:
    meta_file = _meta_path(cache_dir)
    meta = {
        "num_bits": num_bits,
        "thresholds": thresholds,
        "augmentations": list(augmentation_names),
        "config": config_name,
        "train_shape": train_shape,
        "test_shape": test_shape,
    }
    with meta_file.open("w") as fh:
        json.dump(meta, fh)


def prepare_fashion_augmented_bundle(
    train_ds: VisionDataset,
    test_ds: VisionDataset,
    cache_dir: Path,
    batch_size: int = 256,
    augmentations: Sequence[AugmentationRecipe] = DEFAULT_AUGMENTATIONS,
    force: bool = False,
    apply_augmentations: bool = True,
) -> FashionAugmentedBundle:
    config = PreprocessConfig(
        name="fashionmnist",
        image_size=(28, 28),
        in_channels=1,
        to_grayscale=True,
        augmentations=tuple(augmentations),
    )
    return prepare_boolean_feature_bundle(
        train_ds,
        test_ds,
        cache_dir,
        config=config,
        batch_size=batch_size,
        force=force,
        apply_augmentations=apply_augmentations,
    )


def prepare_boolean_feature_bundle(
    train_ds: VisionDataset,
    test_ds: VisionDataset,
    cache_dir: Path,
    *,
    config: PreprocessConfig,
    batch_size: int = 256,
    force: bool = False,
    apply_augmentations: bool = True,
) -> FashionAugmentedBundle:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_file = cache_dir / "fashion_augmented_train.bin"
    test_file = cache_dir / "fashion_augmented_test.bin"
    train_labels_file = cache_dir / "fashion_augmented_train_labels.npy"
    test_labels_file = cache_dir / "fashion_augmented_test_labels.npy"

    augmentations = config.augmentations if apply_augmentations else (DEFAULT_AUGMENTATIONS[0],)
    aug_names = [aug.name for aug in augmentations]

    meta = _load_meta(cache_dir)
    if (
        not force
        and meta
        and meta.get("augmentations") == aug_names
        and meta.get("config") == config.name
        and train_file.exists()
        and test_file.exists()
        and train_labels_file.exists()
        and test_labels_file.exists()
    ):
        train_labels = np.load(train_labels_file)
        test_labels = np.load(test_labels_file)
        train_dataset = PackedBooleanDataset(train_file, train_labels, meta["num_bits"])
        test_dataset = PackedBooleanDataset(test_file, test_labels, meta["num_bits"])
        return FashionAugmentedBundle(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            num_bits=meta["num_bits"],
            thresholds=meta["thresholds"],
            augmentation_names=list(meta["augmentations"]),
            config_name=config.name,
        )

    raw_train, train_labels = _dataset_to_tensor(train_ds, config)
    raw_test, test_labels = _dataset_to_tensor(test_ds, config)

    augmented_train = _apply_augmentations(raw_train, augmentations)
    repeated_labels = train_labels.repeat(len(augmentations))

    kernels = _prepare_kernels()
    thresholds, _ = _compute_thresholds(augmented_train, kernels)

    height, width = config.image_size
    channels = augmented_train.size(1)
    num_raw_layers = 4
    num_orientation_layers = len(kernels)
    bits_per_channel = (num_raw_layers + num_orientation_layers * 9) * height * width
    bits_per_sample = channels * bits_per_channel

    np.save(train_labels_file, repeated_labels.numpy())
    np.save(test_labels_file, test_labels.numpy())

    train_count = augmented_train.size(0)
    test_count = raw_test.size(0)

    _write_packed_features(train_file, augmented_train, kernels, thresholds, batch_size, bits_per_sample)
    del augmented_train
    del raw_train
    _write_packed_features(test_file, raw_test, kernels, thresholds, batch_size, bits_per_sample)
    del raw_test

    _save_meta(
        cache_dir,
        num_bits=bits_per_sample,
        thresholds=thresholds,
        augmentation_names=aug_names,
        config_name=config.name,
        train_shape=(train_count, bits_per_sample),
        test_shape=(test_count, bits_per_sample),
    )

    train_dataset = PackedBooleanDataset(train_file, np.load(train_labels_file), bits_per_sample)
    test_dataset = PackedBooleanDataset(test_file, np.load(test_labels_file), bits_per_sample)

    return FashionAugmentedBundle(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_bits=bits_per_sample,
        thresholds=thresholds,
        augmentation_names=[aug.name for aug in augmentations],
        config_name=config.name,
    )


