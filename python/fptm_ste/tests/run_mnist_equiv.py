"""
MNIST equivalence runner with profiling and accelerated training.
- Trains STE TM, Deep TM, CNN+TM Hybrid, and TM-Transformer (5 epochs by default)
- Uses fuzzy (float) inputs for learning stability
- Skips regenerating boolean datasets if cached
- Exports compiled literals to JSON for Julia bridge
"""

import json
import os
import resource
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fptm_ste import FuzzyPatternTM_STE
from fptm_ste.binarizers import CNNSingleBinarizer
from fptm_ste.deep_tm import DeepTMNetwork
from fptm_ste.export import export_compiled_to_json
from fptm_ste.tm_transformer import UnifiedTMTransformer

EPOCHS = int(os.environ.get("TM_MNIST_EPOCHS", 25))
BATCH_SIZE = int(os.environ.get("TM_MNIST_BATCH", 128))
TEST_BATCH_SIZE = int(os.environ.get("TM_MNIST_TEST_BATCH", 512))
REPORT_TRAIN_ACC = os.environ.get("TM_MNIST_REPORT_TRAIN", "1") != "0"
REPORT_EPOCH_ACC = os.environ.get("TM_MNIST_REPORT_EPOCH", "0") != "0"


def current_device() -> torch.device:
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
                   use_ste_eval: bool) -> Tuple[Optional[float], float, float, List[int]]:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    start = time.time()
    last_train_acc: Optional[float] = None
    for epoch in range(EPOCHS):
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
            if REPORT_TRAIN_ACC:
                preds = logits.argmax(dim=1)
                epoch_correct += (preds == y).sum().item()
                epoch_total += y.size(0)
        dur = time.time() - epoch_start
        avg_loss = running_loss / batch_idx
        epoch_acc = None
        if REPORT_TRAIN_ACC and epoch_total > 0:
            epoch_acc = epoch_correct / epoch_total
            last_train_acc = epoch_acc
        log_msg = f"  {label:12s} epoch {epoch + 1:02d}/{EPOCHS:02d} | loss {avg_loss:.4f}"
        if epoch_acc is not None and REPORT_EPOCH_ACC:
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


def run_variant_tm(train_loader, test_loader, device):
    model = FuzzyPatternTM_STE(n_features=28 * 28, n_clauses=256, n_classes=10, tau=0.4).to(device)
    train_acc, test_acc, ttrain, preds = train_tm_model(model, flatten_images, train_loader, test_loader, device, "STE-TM", use_ste_train=False, use_ste_eval=True)
    bundle = model.discretize(threshold=0.5)
    return train_acc, test_acc, ttrain, preds, bundle


def run_variant_deeptm(train_loader, test_loader, device):
    model = DeepTMNetwork(input_dim=28 * 28, hidden_dims=[256, 128], n_classes=10, n_clauses=256, dropout=0.1, tau=0.5).to(device)
    train_acc, test_acc, ttrain, preds = train_tm_model(model, flatten_images, train_loader, test_loader, device, "Deep-TM", use_ste_train=False, use_ste_eval=True)
    bundle = model.classifier.discretize(threshold=0.5)
    return train_acc, test_acc, ttrain, preds, bundle


def run_variant_hybrid(train_loader, test_loader, device):
    model = CNNHybrid(thresholds=32, tm_clauses=384, n_classes=10).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    start = time.time()
    last_train_acc: Optional[float] = None
    for epoch in range(EPOCHS):
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
            if REPORT_TRAIN_ACC:
                preds = logits.argmax(dim=1)
                epoch_correct += (preds == y).sum().item()
                epoch_total += y.size(0)
        dur = time.time() - epoch_start
        avg_loss = running_loss / batch_idx
        epoch_acc = None
        if REPORT_TRAIN_ACC and epoch_total > 0:
            epoch_acc = epoch_correct / epoch_total
            last_train_acc = epoch_acc
        log_msg = f"  Hybrid       epoch {epoch + 1:02d}/{EPOCHS:02d} | loss {avg_loss:.4f}"
        if epoch_acc is not None and REPORT_EPOCH_ACC:
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
    return last_train_acc, test_acc, train_time, preds, bundle


def run_variant_transformer(train_loader, test_loader, device):
    vocab = 256
    model = UnifiedTMTransformer(vocab_size=vocab, d_model=96, n_heads=3, num_layers=2, max_len=28 * 28, n_clauses=192).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    start = time.time()
    last_train_acc: Optional[float] = None
    for epoch in range(EPOCHS):
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
            if REPORT_TRAIN_ACC:
                preds = logits[:, -1, :].argmax(dim=1)
                epoch_correct += (preds == y).sum().item()
                epoch_total += y.size(0)
        dur = time.time() - epoch_start
        avg_loss = running_loss / batch_idx
        epoch_acc = None
        if REPORT_TRAIN_ACC and epoch_total > 0:
            epoch_acc = epoch_correct / epoch_total
            last_train_acc = epoch_acc
        log_msg = f"  Transformer  epoch {epoch + 1:02d}/{EPOCHS:02d} | loss {avg_loss:.4f}"
        if epoch_acc is not None and REPORT_EPOCH_ACC:
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
    return last_train_acc, test_acc, train_time, preds, None


def main():
    device = current_device()
    transform = transforms.ToTensor()
    with profile_block("load-dataset"):
        train_ds = datasets.MNIST(root="/tmp/mnist", train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root="/tmp/mnist", train=False, download=True, transform=transform)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=(device.type != "cpu"), persistent_workers=True)
        test_loader = DataLoader(test_ds, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=(device.type != "cpu"), persistent_workers=True)

    out_train = "/tmp/MNISTTrainingData.txt"
    out_test = "/tmp/MNISTTestData.txt"
    with profile_block("write-boolean-datasets"):
        save_bool_dataset(train_loader, test_loader, out_train, out_test, thr=0.5)

    results: Dict[str, Dict] = {}

    with profile_block("variant-ste-tm"):
        train_acc, test_acc, ttrain, preds, bundle = run_variant_tm(train_loader, test_loader, device)
        path = export_compiled_to_json(bundle, [str(i) for i in range(10)], "/tmp/tm_mnist_tm", bundle["clauses_num"], L=16, LF=4)
        results["tm"] = {"train_accuracy": train_acc, "test_accuracy": test_acc, "train_time_s": ttrain, "json": path, "preds": preds}

    with profile_block("variant-deep-tm"):
        train_acc, test_acc, ttrain, preds, bundle = run_variant_deeptm(train_loader, test_loader, device)
        path = export_compiled_to_json(bundle, [str(i) for i in range(10)], "/tmp/tm_mnist_deeptm", bundle["clauses_num"], L=16, LF=4)
        results["deep_tm"] = {"train_accuracy": train_acc, "test_accuracy": test_acc, "train_time_s": ttrain, "json": path, "preds": preds}

    with profile_block("variant-hybrid"):
        train_acc, test_acc, ttrain, preds, bundle = run_variant_hybrid(train_loader, test_loader, device)
        path = export_compiled_to_json(bundle, [str(i) for i in range(10)], "/tmp/tm_mnist_hybrid", bundle["clauses_num"], L=16, LF=4)
        results["hybrid"] = {"train_accuracy": train_acc, "test_accuracy": test_acc, "train_time_s": ttrain, "json": path, "preds": preds}

    with profile_block("variant-transformer"):
        train_acc, test_acc, ttrain, preds, bundle = run_variant_transformer(train_loader, test_loader, device)
        json_path = None
        if bundle is not None:
            json_path = export_compiled_to_json(bundle, [str(i) for i in range(10)], "/tmp/tm_mnist_transformer", bundle["clauses_num"], L=16, LF=4)
        results["transformer"] = {"train_accuracy": train_acc, "test_accuracy": test_acc, "train_time_s": ttrain, "json": json_path, "preds": preds}

    with open("/tmp/mnist_equiv_results.json", "w") as f:
        json.dump(results, f)

    print("\n=== Summary ===")
    for key, info in results.items():
        train_acc = info.get("train_accuracy")
        test_acc = info.get("test_accuracy")
        acc_str = ""
        if train_acc is not None:
            acc_str += f"train_acc={train_acc:.4f} "
        if test_acc is not None:
            acc_str += f"test_acc={test_acc:.4f} "
        print(f"{key.upper():12s} {acc_str.strip()} train_time={info['train_time_s']:.1f}s json={info['json']}")
    print("Boolean datasets:", out_train, out_test)


if __name__ == "__main__":
    main()

