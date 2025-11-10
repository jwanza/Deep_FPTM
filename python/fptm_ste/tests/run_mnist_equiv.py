"""
MNIST equivalence runner with profiling and accelerated training.
- Trains STE TM, Deep TM, CNN+TM Hybrid, and TM-Transformer (5 epochs by default)
- Uses fuzzy (float) inputs for learning stability
- Skips regenerating boolean datasets if cached
- Exports compiled literals to JSON for Julia bridge
"""

import argparse
import json
import os
import resource
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fptm_ste import FuzzyPatternTM_STE, FuzzyPatternTMFPTM
from fptm_ste.binarizers import CNNSingleBinarizer
from fptm_ste.deep_tm import DeepTMNetwork
from fptm_ste.export import export_compiled_to_json
from fptm_ste.tm_transformer import UnifiedTMTransformer

DEFAULT_MODELS = ("tm", "deep_tm", "hybrid", "transformer")
EXPORT_SLUGS = {
    "tm": "tm",
    "deep_tm": "deeptm",
    "hybrid": "hybrid",
    "transformer": "transformer",
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
    parser.add_argument("--epochs", type=int, default=int_env("TM_MNIST_EPOCHS", 25), help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=int_env("TM_MNIST_BATCH", 128), help="Training batch size.")
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=int_env("TM_MNIST_TEST_BATCH", 512),
        help="Evaluation batch size.",
    )
    parser.add_argument("--num-classes", type=int, default=10, help="Number of target classes.")
    parser.add_argument(
        "--dataset-root",
        default=os.environ.get("TM_MNIST_ROOT", "/tmp/mnist"),
        help="Root directory for the MNIST dataset.",
    )
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow torchvision to download MNIST if missing.",
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
        help="Toggle writing cached boolean MNIST datasets.",
    )
    parser.add_argument(
        "--bool-threshold",
        type=float,
        default=0.5,
        help="Threshold applied when binarizing MNIST tensors.",
    )
    parser.add_argument(
        "--bool-train-path",
        default="/tmp/MNISTTrainingData.txt",
        help="File path for the boolean training dataset export.",
    )
    parser.add_argument(
        "--bool-test-path",
        default="/tmp/MNISTTestData.txt",
        help="File path for the boolean test dataset export.",
    )
    parser.add_argument(
        "--output-json",
        default="/tmp/mnist_equiv_results.json",
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
        default="tm_mnist",
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
        default=bool_env("TM_MNIST_REPORT_EPOCH", False),
        help="Include training accuracy in epoch summary lines.",
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

    # TM specific options
    parser.add_argument(
        "--tm-impl",
        choices=["ste", "fptm"],
        default=os.environ.get("TM_MNIST_TM_IMPL", "ste"),
        help="Implementation for the TM baseline.",
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

    return parser


def current_device(preferred: str = "auto") -> torch.device:
    preferred = (preferred or "auto").lower()
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda":
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            return torch.device("cuda")
        print("Requested CUDA device is unavailable; falling back to CPU.")
        return torch.device("cpu")
    if preferred == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
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


def flatten_images(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size(0), -1)


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
                   use_ste: bool) -> Tuple[float, List[int]]:
    model.eval()
    preds = []
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            xb = prepare_fn(x)
            logits, _ = model(xb, use_ste=use_ste)
            pred = logits.argmax(dim=1)
            preds.extend(pred.cpu().tolist())
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total
    return acc, preds


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
                   report_epoch_acc: bool) -> Tuple[Optional[float], float, float, List[int]]:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    start = time.time()
    last_train_acc: Optional[float] = None
    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            xb = prepare_fn(x)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits, _ = model(xb, use_ste=use_ste_train)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(opt)
            scaler.update()
            running_loss += loss.item()
            if report_train_acc:
                preds = logits.argmax(dim=1)
                epoch_correct += (preds == y).sum().item()
                epoch_total += y.size(0)
        dur = time.time() - epoch_start
        avg_loss = running_loss / batch_idx
        epoch_acc = None
        if report_train_acc and epoch_total > 0:
            epoch_acc = epoch_correct / epoch_total
            last_train_acc = epoch_acc
        log_msg = f"  {label:12s} epoch {epoch + 1:02d}/{epochs:02d} | loss {avg_loss:.4f}"
        if epoch_acc is not None and report_epoch_acc:
            log_msg += f" | train_acc {epoch_acc:.4f}"
        log_msg += f" | {dur:5.1f}s | seen {batch_idx * train_loader.batch_size}"
        print(log_msg)
    train_time = time.time() - start
    test_acc, preds = evaluate_model(model, prepare_fn, test_loader, device, use_ste=use_ste_eval)
    return last_train_acc, test_acc, train_time, preds


class CNNHybrid(nn.Module):
    """Small CNN backbone + learnable binarizer feeding a TM head."""

    def __init__(self, thresholds: int = 32, tm_clauses: int = 384, n_classes: int = 10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.binarizer = CNNSingleBinarizer(in_channels=64, num_thresholds=thresholds, init_temperature=1.0)
        fused_dim = thresholds
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fused_dim, max(64, fused_dim)), nn.GELU(),
            nn.Linear(max(64, fused_dim), max(32, fused_dim // 2)), nn.Sigmoid(),
        )
        self.tm = FuzzyPatternTM_STE(max(32, fused_dim // 2), tm_clauses, n_classes, tau=0.5)

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
                   tm_impl: str,
                   tm_tau: float,
                   tm_lf: int,
                   tm_literal_budget: Optional[int],
                   tm_vote_clamp: Optional[float],
                   tm_n_clauses: int,
                   n_classes: int = 10) -> Tuple[str, Optional[float], float, float, List[int], Dict[str, Any]]:
    tm_impl = tm_impl.lower()
    n_features = 28 * 28
    if tm_impl == "fptm":
        model = FuzzyPatternTMFPTM(
            n_features=n_features,
            n_clauses=tm_n_clauses,
            n_classes=n_classes,
            tau=tm_tau,
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
        ).to(device)
        use_ste_train = False
        label = "STE-TM"

    train_acc, test_acc, ttrain, preds = train_tm_model(
        model,
        flatten_images,
        train_loader,
        test_loader,
        device,
        label,
        use_ste_train=use_ste_train,
        use_ste_eval=True,
        epochs=epochs,
        report_train_acc=report_train_acc,
        report_epoch_acc=report_epoch_acc,
    )
    bundle = model.discretize(threshold=0.5)
    return label, train_acc, test_acc, ttrain, preds, bundle


def run_variant_deeptm(train_loader,
                       test_loader,
                       device,
                       *,
                       epochs: int,
                       report_train_acc: bool,
                       report_epoch_acc: bool,
                       hidden_dims: List[int],
                       n_clauses: int,
                       dropout: float,
                       tau: float,
                       n_classes: int = 10) -> Tuple[str, Optional[float], float, float, List[int], Dict[str, Any]]:
    model = DeepTMNetwork(
        input_dim=28 * 28,
        hidden_dims=hidden_dims,
        n_classes=n_classes,
        n_clauses=n_clauses,
        dropout=dropout,
        tau=tau,
    ).to(device)
    label = "Deep-TM"
    train_acc, test_acc, ttrain, preds = train_tm_model(
        model,
        flatten_images,
        train_loader,
        test_loader,
        device,
        label,
        use_ste_train=False,
        use_ste_eval=True,
        epochs=epochs,
        report_train_acc=report_train_acc,
        report_epoch_acc=report_epoch_acc,
    )
    bundle = model.classifier.discretize(threshold=0.5)
    return label, train_acc, test_acc, ttrain, preds, bundle


def run_variant_hybrid(train_loader,
                       test_loader,
                       device,
                       *,
                       epochs: int,
                       report_train_acc: bool,
                       report_epoch_acc: bool,
                       thresholds: int,
                       tm_clauses: int,
                       n_classes: int = 10) -> Tuple[str, Optional[float], float, float, List[int], Dict[str, Any]]:
    model = CNNHybrid(thresholds=thresholds, tm_clauses=tm_clauses, n_classes=n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    start = time.time()
    last_train_acc: Optional[float] = None
    label = "Hybrid"
    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits, _ = model(x, use_ste=False)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(opt)
            scaler.update()
            running_loss += loss.item()
            if report_train_acc:
                preds = logits.argmax(dim=1)
                epoch_correct += (preds == y).sum().item()
                epoch_total += y.size(0)
        dur = time.time() - epoch_start
        avg_loss = running_loss / batch_idx
        epoch_acc = None
        if report_train_acc and epoch_total > 0:
            epoch_acc = epoch_correct / epoch_total
            last_train_acc = epoch_acc
        log_msg = f"  {label:12s} epoch {epoch + 1:02d}/{epochs:02d} | loss {avg_loss:.4f}"
        if epoch_acc is not None and report_epoch_acc:
            log_msg += f" | train_acc {epoch_acc:.4f}"
        log_msg += f" | {dur:5.1f}s | seen {batch_idx * train_loader.batch_size}"
        print(log_msg)
    train_time = time.time() - start
    model.eval()
    preds = []
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits, _ = model(x, use_ste=True)
            pred = logits.argmax(dim=1)
            preds.extend(pred.cpu().tolist())
            correct += (pred == y).sum().item()
            total += y.size(0)
    test_acc = correct / total
    bundle = model.tm.discretize(threshold=0.5)
    return label, last_train_acc, test_acc, train_time, preds, bundle


def run_variant_transformer(train_loader,
                            test_loader,
                            device,
                            *,
                            epochs: int,
                            report_train_acc: bool,
                            report_epoch_acc: bool,
                            vocab_size: int,
                            d_model: int,
                            n_heads: int,
                            num_layers: int,
                            n_clauses: int) -> Tuple[str, Optional[float], float, float, List[int], Optional[Dict[str, Any]]]:
    model = UnifiedTMTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        max_len=28 * 28,
        n_clauses=n_clauses,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    start = time.time()
    last_train_acc: Optional[float] = None
    label = "Transformer"
    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            tokens = (x.view(x.size(0), -1) * 255).long()
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(tokens, use_ste=True)
                loss = F.cross_entropy(logits[:, -1, :], y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(opt)
            scaler.update()
            running_loss += loss.item()
            if report_train_acc:
                preds = logits[:, -1, :].argmax(dim=1)
                epoch_correct += (preds == y).sum().item()
                epoch_total += y.size(0)
        dur = time.time() - epoch_start
        avg_loss = running_loss / batch_idx
        epoch_acc = None
        if report_train_acc and epoch_total > 0:
            epoch_acc = epoch_correct / epoch_total
            last_train_acc = epoch_acc
        log_msg = f"  {label:12s} epoch {epoch + 1:02d}/{epochs:02d} | loss {avg_loss:.4f}"
        if epoch_acc is not None and report_epoch_acc:
            log_msg += f" | train_acc {epoch_acc:.4f}"
        log_msg += f" | {dur:5.1f}s | seen {batch_idx * train_loader.batch_size}"
        print(log_msg)
    train_time = time.time() - start
    model.eval()
    preds = []
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            tokens = (x.view(x.size(0), -1) * 255).long()
            logits = model(tokens, use_ste=True)
            pred = logits[:, -1, :].argmax(dim=1)
            preds.extend(pred.cpu().tolist())
            correct += (pred == y).sum().item()
            total += y.size(0)
    test_acc = correct / total
    return label, last_train_acc, test_acc, train_time, preds, None


def main(argv: Optional[List[str]] = None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    maybe_set_seed(args.seed)
    device = current_device(args.device)

    pin_memory = args.pin_memory
    if pin_memory is None:
        pin_memory = device.type != "cpu"

    persistent_workers = args.persistent_workers
    if persistent_workers is None:
        persistent_workers = args.num_workers > 0
    if args.num_workers == 0:
        persistent_workers = False

    transform = transforms.ToTensor()
    with profile_block("load-dataset"):
        train_ds = datasets.MNIST(
            root=args.dataset_root,
            train=True,
            download=args.download,
            transform=transform,
        )
        test_ds = datasets.MNIST(
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

    selected_models = tuple(args.models or DEFAULT_MODELS)
    if not selected_models:
        raise ValueError("No model variants selected.")

    hidden_dims = parse_int_list(args.deeptm_hidden_dims)
    if not hidden_dims:
        hidden_dims = [256, 128]

    results: Dict[str, Dict[str, Any]] = {}

    for model_key in selected_models:
        with profile_block(f"variant-{model_key}"):
            if model_key == "tm":
                label, train_acc, test_acc, train_time, preds, bundle = run_variant_tm(
                    train_loader,
                    test_loader,
                    device,
                    epochs=args.epochs,
                    report_train_acc=args.report_train_acc,
                    report_epoch_acc=args.report_epoch_acc,
                    tm_impl=args.tm_impl,
                    tm_tau=args.tm_tau,
                    tm_lf=args.tm_lf,
                    tm_literal_budget=args.tm_literal_budget,
                    tm_vote_clamp=args.tm_vote_clamp,
                    tm_n_clauses=args.tm_n_clauses,
                    n_classes=args.num_classes,
                )
                variant_classes = args.num_classes
            elif model_key == "deep_tm":
                label, train_acc, test_acc, train_time, preds, bundle = run_variant_deeptm(
                    train_loader,
                    test_loader,
                    device,
                    epochs=args.epochs,
                    report_train_acc=args.report_train_acc,
                    report_epoch_acc=args.report_epoch_acc,
                    hidden_dims=hidden_dims,
                    n_clauses=args.deeptm_n_clauses,
                    dropout=args.deeptm_dropout,
                    tau=args.deeptm_tau,
                    n_classes=args.num_classes,
                )
                variant_classes = args.num_classes
            elif model_key == "hybrid":
                hybrid_classes = args.hybrid_classes if args.hybrid_classes is not None else args.num_classes
                label, train_acc, test_acc, train_time, preds, bundle = run_variant_hybrid(
                    train_loader,
                    test_loader,
                    device,
                    epochs=args.epochs,
                    report_train_acc=args.report_train_acc,
                    report_epoch_acc=args.report_epoch_acc,
                    thresholds=args.hybrid_thresholds,
                    tm_clauses=args.hybrid_clauses,
                    n_classes=hybrid_classes,
                )
                variant_classes = hybrid_classes
            elif model_key == "transformer":
                label, train_acc, test_acc, train_time, preds, bundle = run_variant_transformer(
                    train_loader,
                    test_loader,
                    device,
                    epochs=args.epochs,
                    report_train_acc=args.report_train_acc,
                    report_epoch_acc=args.report_epoch_acc,
                    vocab_size=args.transformer_vocab,
                    d_model=args.transformer_d_model,
                    n_heads=args.transformer_heads,
                    num_layers=args.transformer_layers,
                    n_clauses=args.transformer_clauses,
                )
                variant_classes = args.num_classes
            else:
                raise ValueError(f"Unknown model variant '{model_key}'.")

        export_path = None
        if args.export_compiled and bundle is not None:
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
                L=16,
                LF=4,
            )

        results[model_key] = {
            "label": label,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "train_time_s": train_time,
            "json": export_path,
            "preds": preds,
        }

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


if __name__ == "__main__":
    main()

