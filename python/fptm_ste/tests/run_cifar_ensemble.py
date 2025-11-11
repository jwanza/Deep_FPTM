import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, Subset

try:  # Optional dependency for YAML configs
    import yaml
except ImportError:  # pragma: no cover - handled gracefully at runtime
    yaml = None

FILE_PATH = Path(__file__).resolve()
PYTHON_ROOT = FILE_PATH.parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from fptm_ste import SwinFeatureExtractor, MultiScaleTMEnsemble
from fptm_ste.trainers import train_step


DEFAULT_HEAD_CONFIGS = [
    {"stages": [0], "n_clauses": 200, "thresholds": 16, "binarizer": "dual"},
    {"stages": [1], "n_clauses": 150, "thresholds": 16, "binarizer": "dual"},
    {"stages": [2], "n_clauses": 120, "thresholds": 16, "binarizer": "dual"},
    {"stages": [3], "n_clauses": 100, "thresholds": 16, "binarizer": "dual"},
    {"stages": [1, 2], "n_clauses": 180, "thresholds": 8, "binarizer": "dual"},
    {"stages": [2, 3], "n_clauses": 160, "thresholds": 8, "binarizer": "dual"},
]

BACKBONE_ALIASES = {
    "swin_t": "tiny",
    "swin_s": "small",
    "swin_b": "base",
    "swin_l": "large",
    "swin_tiny": "tiny",
    "swin_small": "small",
    "swin_base": "base",
    "swin_large": "large",
    "tiny": "tiny",
    "small": "small",
    "base": "base",
    "large": "large",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Swin feature extractor + TM ensemble on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of fine-tuning epochs for the TM ensemble.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for both train and test loaders.")
    parser.add_argument("--train-subset", type=int, default=1000, help="Number of training samples to use (<=50000).")
    parser.add_argument("--test-subset", type=int, default=500, help="Number of test samples to use (<=10000).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for subset selection and initialization.")
    parser.add_argument("--backbone", type=str, default="swin_t", help="Backbone variant (alias or native Swin size).")
    parser.add_argument("--num-scales", type=int, default=4, help="Number of Swin scales to extract (1-4).")
    parser.add_argument("--image-size", type=int, default=224, help="Resize edge length for CIFAR inputs.")
    parser.add_argument("--heads", type=str, default=None, help="Path to YAML/JSON file describing head configs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for ensemble optimizer.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for ensemble optimizer.")
    parser.add_argument("--freeze-stages", type=int, default=4, help="Number of Swin stages to freeze (0-4).")
    parser.add_argument("--drop-path", type=float, default=0.2, help="Drop path rate applied to Swin backbone.")
    parser.add_argument("--use-fpn", action="store_true", help="Enable FPN neck on top of Swin outputs.")
    parser.add_argument("--fpn-channels", type=int, default=256, help="Channel count when FPN is enabled.")
    parser.add_argument("--data-root", type=str, default="/tmp/cifar", help="Directory for CIFAR-10 downloads/cache.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes per loader.")
    parser.add_argument("--device", type=str, default=None, help="Override device string (e.g., 'cuda:0', 'cpu').")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="Disable pretrained Swin weights.")
    parser.add_argument("--no-download", dest="allow_download", action="store_false", help="Disable dataset downloads.")
    parser.set_defaults(pretrained=True, allow_download=True)
    return parser.parse_args()


def resolve_backbone(name: str) -> str:
    key = name.lower()
    if key not in BACKBONE_ALIASES:
        raise ValueError(f"Unknown backbone '{name}'. Supported: {sorted(BACKBONE_ALIASES.keys())}")
    return BACKBONE_ALIASES[key]


def normalize_head_configs(heads: Iterable[dict]) -> List[dict]:
    heads_list = list(heads)
    normalized: List[dict] = []
    for idx, head in enumerate(heads_list):
        if not isinstance(head, dict):
            raise ValueError(f"Head config at index {idx} must be a mapping, got {type(head).__name__}.")
        if "stages" not in head:
            raise ValueError(f"Head config at index {idx} is missing 'stages'.")
        stages = head["stages"]
        if isinstance(stages, (int, float)):
            stages = [int(stages)]
        stages = [int(stage) for stage in stages]
        config = {**head, "stages": stages}
        normalized.append(config)
    return normalized


def load_head_configs(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Head config file '{path}' does not exist.")
    data = path.read_text()
    ext = path.suffix.lower()
    if ext in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("PyYAML is required to load YAML head configs. Install with `pip install pyyaml`.")
        raw = yaml.safe_load(data)
    else:
        raw = json.loads(data)
    if raw is None:
        raise ValueError(f"Head config file '{path}' contained no data.")
    if isinstance(raw, dict):
        if "heads" in raw and isinstance(raw["heads"], Iterable):
            raw = raw["heads"]
        else:
            raw = list(raw.values())
    return normalize_head_configs(raw)


def choose_indices(total: int, subset: int, seed: int) -> List[int]:
    if subset <= 0 or subset >= total:
        return list(range(total))
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total, generator=generator)
    return perm[:subset].tolist()


def build_dataloaders(
    args: argparse.Namespace,
    *,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size), interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
        ]
    )
    trainset = datasets.CIFAR10(
        root=args.data_root,
        train=True,
        download=args.allow_download,
        transform=transform,
    )
    testset = datasets.CIFAR10(
        root=args.data_root,
        train=False,
        download=args.allow_download,
        transform=transform,
    )
    train_idx = choose_indices(len(trainset), args.train_subset, args.seed)
    test_idx = choose_indices(len(testset), args.test_subset, args.seed)
    train_loader = DataLoader(
        Subset(trainset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        Subset(testset, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    return train_loader, test_loader


def validate_head_indices(heads: Sequence[dict], num_scales: int) -> None:
    for head in heads:
        invalid = [stage for stage in head["stages"] if stage < 0 or stage >= num_scales]
        if invalid:
            raise ValueError(
                f"Head stages {invalid} are outside the available range [0, {num_scales - 1}]."
            )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    train_loader, test_loader = build_dataloaders(args, pin_memory=pin_memory)

    variant = resolve_backbone(args.backbone)
    try:
        backbone = SwinFeatureExtractor(
            variant=variant,
            pretrained=args.pretrained,
            num_scales=args.num_scales,
            input_size=args.image_size,
            freeze_stages=args.freeze_stages,
            drop_path_rate=args.drop_path,
            use_fpn=args.use_fpn,
            fpn_channels=args.fpn_channels,
        ).to(device)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to initialize SwinFeatureExtractor. Install/upgrade timm to >=0.9.0 so pretrained Swin models "
            "are available (timm is required for this script)."
        ) from exc
    backbone.eval()

    with torch.no_grad():
        sample = next(iter(train_loader))[0].to(device)
        feats = backbone(sample)
    feature_dims = [feat.shape[1] for feat in feats]
    num_scales = len(feature_dims)

    if args.heads:
        try:
            head_configs = load_head_configs(Path(args.heads))
        except (FileNotFoundError, ImportError, ValueError) as exc:
            warnings.warn(f"{exc}. Falling back to built-in head configuration.")
            head_configs = list(DEFAULT_HEAD_CONFIGS)
    else:
        head_configs = list(DEFAULT_HEAD_CONFIGS)
    validate_head_indices(head_configs, num_scales)

    model = MultiScaleTMEnsemble(
        feature_dims,
        n_classes=10,
        head_configs=head_configs,
        backbone_type="swin",
        init_temperature=1.0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def evaluate(loader: DataLoader) -> float:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                feats = backbone(x)
                logits, _, _ = model(feats, use_ste=True)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / max(total, 1)

    acc_before = evaluate(test_loader)

    for epoch in range(args.epochs):
        model.train()
        last_loss = float("nan")
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                feats = backbone(x)
            optimizer.zero_grad()
            logits, _, _ = model(feats, use_ste=True)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            last_loss = loss.item()
        acc_epoch = evaluate(test_loader)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | loss={last_loss:.4f} | test_acc={acc_epoch:.4f}",
            flush=True,
        )

    acc_after = evaluate(test_loader)
    print(
        f"CIFAR10 Swin+TM ensemble accuracy: before={acc_before:.3f}, after={acc_after:.3f}"
        f" (variant={variant}, pretrained={args.pretrained})"
    )


if __name__ == "__main__":
    main()


