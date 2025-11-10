"""
Baseline MNIST runner for FPTM models.

This script reproduces the single-layer TM and DeepTM baselines while logging
per-epoch metrics (including TM-specific diagnostics) using
``fptm_ste.experiments.ExperimentLogger``. Artefacts are written to the same
directory as this script.
"""

import argparse
import time
from pathlib import Path
from typing import Callable, Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fptm_ste.deep_tm import DeepTMNetwork
from fptm_ste.experiments import EpochMetrics, ExperimentLogger, collect_tm_diagnostics
from fptm_ste.tm import FuzzyPatternTM_STE


def get_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_mnist(batch_size: int, test_batch_size: int, device: torch.device) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root="/tmp/mnist", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="/tmp/mnist", train=False, download=True, transform=transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type != "cpu"),
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type != "cpu"),
        persistent_workers=True,
    )
    return train_loader, test_loader


def flatten_images(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size(0), -1)


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, use_ste: bool) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits, _ = model(flatten_images(x), use_ste=use_ste)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def train_model(
    model_name: str,
    model_builder: Callable[[], torch.nn.Module],
    params: Dict,
    device: torch.device,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    threshold: float,
    output_dir: Path,
    seed: int,
) -> None:
    model = model_builder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    logger = ExperimentLogger(
        output_dir=output_dir,
        run_name=model_name.lower(),
        metadata={
            "model_name": model_name,
            "params": params,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
            "threshold": threshold,
            "seed": seed,
            "device": device.type,
        },
    )

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        start = time.time()

        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            xb = flatten_images(x)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits, _ = model(xb, use_ste=False)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        duration = time.time() - start
        train_acc = correct / total if total else 0.0
        avg_loss = epoch_loss / batch_idx
        test_acc = evaluate(model, test_loader, device, use_ste=True)
        diagnostics = collect_tm_diagnostics(model, threshold=threshold)

        logger.log_epoch(
            EpochMetrics(
                epoch=epoch,
                train_loss=avg_loss,
                train_accuracy=train_acc,
                eval_accuracy=test_acc,
                duration_s=duration,
                diagnostics=diagnostics,
            )
        )

        print(
            f"{model_name} | epoch {epoch:02d}/{epochs:02d} | loss={avg_loss:.4f} | "
            f"train_acc={train_acc:.4f} | test_acc={test_acc:.4f}"
        )

    logger.flush_all()


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline MNIST runners for TM variants.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for clause sparsity diagnostics.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device()
    train_loader, test_loader = load_mnist(args.batch_size, args.test_batch_size, device)

    tm_params = {"n_features": 28 * 28, "n_clauses": 256, "n_classes": 10, "tau": 0.4}
    train_model(
        "FuzzyPatternTM_STE",
        lambda: FuzzyPatternTM_STE(**tm_params),
        tm_params,
        device,
        train_loader,
        test_loader,
        args.epochs,
        args.lr,
        args.weight_decay,
        args.grad_clip,
        args.threshold,
        args.output_dir,
        args.seed,
    )

    deep_params = {
        "input_dim": 28 * 28,
        "hidden_dims": [256, 128],
        "n_classes": 10,
        "n_clauses": 256,
        "dropout": 0.1,
        "tau": 0.5,
        "noise_std": 0.0,
    }
    train_model(
        "DeepTMNetwork",
        lambda: DeepTMNetwork(**deep_params),
        deep_params,
        device,
        train_loader,
        test_loader,
        args.epochs,
        args.lr,
        args.weight_decay,
        args.grad_clip,
        args.threshold,
        args.output_dir,
        args.seed,
    )


if __name__ == "__main__":
    main()
