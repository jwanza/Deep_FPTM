"""
Pyramid TM experiment runner for MNIST with advanced scheduling and regularization options.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fptm_ste.experiments import EpochMetrics, ExperimentLogger, collect_tm_diagnostics
from fptm_ste.pyramid_tm import PyramidStageConfig, PyramidTM
from fptm_ste.trainers import EMAWrapper, anneal_ste_factor


def get_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_mnist(batch_size: int, test_batch_size: int, device: torch.device):
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


def evaluate(model: PyramidTM, loader: DataLoader, device: torch.device, use_ste: bool) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits, _, _ = model(flatten_images(x), use_ste=use_ste)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def linear_tau(initial: float, final: float, epoch: int, total_epochs: int) -> float:
    if total_epochs <= 1:
        return final
    progress = epoch / (total_epochs - 1)
    return initial + (final - initial) * progress


def parse_sequence(arg: str, cast):
    values = [cast(v) for v in arg.split(",") if v.strip()]
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PyramidTM on MNIST.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tau-start", type=float, default=0.6)
    parser.add_argument("--tau-end", type=float, default=0.3)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=2)
    parser.add_argument("--no-clause-attention", action="store_true")
    parser.add_argument("--no-scale-attention", action="store_true")
    parser.add_argument("--scale-entropy-weight", type=float, default=0.01)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--input-noise-std", type=float, default=0.0)
    parser.add_argument("--stage-clauses", type=str, default="")
    parser.add_argument("--stage-dropouts", type=str, default="")
    parser.add_argument("--export-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device()
    train_loader, test_loader = load_mnist(args.batch_size, args.test_batch_size, device)

    stage_configs = [
        PyramidStageConfig(pool_size=28, projection_dim=256, n_clauses=256, tau=args.tau_start, dropout=0.05),
        PyramidStageConfig(pool_size=14, projection_dim=192, n_clauses=192, tau=args.tau_start, dropout=0.05),
        PyramidStageConfig(pool_size=7, projection_dim=128, n_clauses=160, tau=args.tau_start, dropout=0.05),
        PyramidStageConfig(pool_size=1, projection_dim=96, n_clauses=128, tau=args.tau_start, dropout=0.0),
    ]

    if args.stage_clauses:
        overrides = parse_sequence(args.stage_clauses, int)
        if len(overrides) != len(stage_configs):
            raise ValueError("stage_clauses length must match number of stages")
        for cfg, value in zip(stage_configs, overrides):
            cfg.n_clauses = value

    if args.stage_dropouts:
        overrides = parse_sequence(args.stage_dropouts, float)
        if len(overrides) != len(stage_configs):
            raise ValueError("stage_dropouts length must match number of stages")
        for cfg, value in zip(stage_configs, overrides):
            cfg.dropout = value

    stage_configs = tuple(stage_configs)

    model = PyramidTM(
        stage_configs=stage_configs,
        use_clause_attention=not args.no_clause_attention,
        attention_heads=args.attention_heads,
        use_scale_attention=not args.no_scale_attention,
        scale_entropy_weight=args.scale_entropy_weight,
        input_noise_std=args.input_noise_std,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs - args.warmup_epochs),
        eta_min=args.min_lr,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    ema = EMAWrapper(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    logger = ExperimentLogger(
        output_dir=args.output_dir,
        run_name="pyramid_tm",
        metadata={
            "model_name": "PyramidTM",
            "epochs": args.epochs,
            "lr": args.lr,
            "min_lr": args.min_lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "grad_accum": args.grad_accum,
            "tau_start": args.tau_start,
            "tau_end": args.tau_end,
            "warmup_epochs": args.warmup_epochs,
            "stage_configs": [cfg.__dict__ for cfg in stage_configs],
            "seed": args.seed,
            "device": device.type,
            "attention_heads": args.attention_heads,
            "use_clause_attention": not args.no_clause_attention,
            "use_scale_attention": not args.no_scale_attention,
            "scale_entropy_weight": args.scale_entropy_weight,
            "ema_decay": args.ema_decay,
            "input_noise_std": args.input_noise_std,
            "export_path": str(args.export_path) if args.export_path else None,
            "input_noise_std": args.input_noise_std,
        },
    )

    for epoch in range(1, args.epochs + 1):
        if epoch <= args.warmup_epochs:
            warmup_lr = args.lr * epoch / max(1, args.warmup_epochs)
            for group in optimizer.param_groups:
                group["lr"] = warmup_lr
        current_lr = optimizer.param_groups[0]["lr"]

        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        start = time.time()
        accum_counter = 0
        num_batches = 0

        optimizer.zero_grad(set_to_none=True)
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            num_batches += 1
            accum_counter += 1
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            xb = flatten_images(x)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits, _, _ = model(xb, use_ste=False)
                loss = F.cross_entropy(logits, y)
                aux_loss = model.attention_entropy_loss()
                if aux_loss is not None:
                    loss = loss + aux_loss
                scaled_loss = loss / args.grad_accum
            scaler.scale(scaled_loss).backward()

            if accum_counter >= args.grad_accum:
                scaler.unscale_(optimizer)
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accum_counter = 0
                if ema is not None:
                    ema.update(model)

            epoch_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        if accum_counter > 0:
            scaler.unscale_(optimizer)
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

        duration = time.time() - start
        train_acc = correct / total if total else 0.0
        avg_loss = epoch_loss / max(1, num_batches)

        if ema is not None:
            ema.apply_shadow(model)
        test_acc = evaluate(model, test_loader, device, use_ste=True)
        if ema is not None:
            ema.restore(model)

        diagnostics = collect_tm_diagnostics(model, threshold=0.5)

        logger.log_epoch(
            EpochMetrics(
                epoch=epoch,
                train_loss=avg_loss,
                train_accuracy=train_acc,
                eval_accuracy=test_acc,
                duration_s=duration,
                diagnostics=diagnostics,
                extra={"lr": current_lr},
            )
        )

        print(
            f"PyramidTM | epoch {epoch:02d}/{args.epochs:02d} | loss={avg_loss:.4f} | "
            f"train_acc={train_acc:.4f} | test_acc={test_acc:.4f} | lr={current_lr:.5f}"
        )

        new_tau = linear_tau(args.tau_start, args.tau_end, epoch, args.epochs)
        anneal_ste_factor(model, new_tau)

        if epoch >= args.warmup_epochs:
            scheduler.step()

    logger.flush_all()

    if ema is not None:
        ema.apply_shadow(model)
    final_acc = evaluate(model, test_loader, device, use_ste=True)
    if ema is not None:
        ema.restore(model)

    if args.export_path:
        stage_exports = {f"stage_{idx}": stage.tm.discretize() for idx, stage in enumerate(model.stages)}
        attention_weights = None
        if model.last_attention_weights is not None:
            attention_weights = model.last_attention_weights.detach().cpu().tolist()
        payload = {
            "final_test_accuracy": final_acc,
            "stage_exports": stage_exports,
            "attention_weights": attention_weights,
        }
        args.export_path.parent.mkdir(parents=True, exist_ok=True)
        args.export_path.write_text(json.dumps(payload, indent=2))
        print(f"Exported discretized model to {args.export_path}")

    print(f"Final test accuracy: {final_acc:.4f}")


if __name__ == "__main__":
    main()
