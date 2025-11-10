"""TM experiment runner supporting classic pyramid and Swin-style variants."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from contextlib import closing
from pathlib import Path
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from fptm_ste.experiments import EpochMetrics, ExperimentLogger, collect_tm_diagnostics
from fptm_ste.pyramid_tm import PyramidStageConfig, PyramidTM
from fptm_ste.swin_pyramid_tm import SwinLikePyramidTM, SwinStageConfig
from fptm_ste.trainers import EMAWrapper, anneal_ste_factor


def get_device(local_rank: Optional[int] = None) -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if local_rank is not None:
            return torch.device("cuda", local_rank)
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


class NullLogger:
    def log_epoch(self, *_args, **_kwargs) -> None:
        pass

    def flush_all(self) -> None:
        pass


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def setup_distributed(rank: int, world_size: int, backend: str, init_method: str) -> None:
    if dist.is_initialized():
        return
    torch.cuda.set_device(rank % max(1, torch.cuda.device_count()))
    dist.init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=rank)


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def reduce_sum(value: float | int, device: torch.device) -> float:
    if not dist.is_available() or not dist.is_initialized():
        return float(value)
    tensor = torch.tensor([float(value)], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


def reduce_max(value: float, device: torch.device) -> float:
    if not dist.is_available() or not dist.is_initialized():
        return value
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def flatten_images(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size(0), -1)


def evaluate(model, loader: DataLoader, device: torch.device, use_ste: bool, *, distributed: bool = False) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            xb = flatten_images(x)
            logits, _, _ = model(xb, use_ste=use_ste)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    if distributed and dist.is_available() and dist.is_initialized():
        tensor = torch.tensor([correct, total], device=device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        correct = int(tensor[0].item())
        total = int(tensor[1].item())
    return correct / total if total else 0.0


def linear_tau(initial: float, final: float, epoch: int, total_epochs: int) -> float:
    if total_epochs <= 1:
        return final
    progress = epoch / (total_epochs - 1)
    return initial + (final - initial) * progress


def parse_sequence(arg: str, cast):
    values = [cast(v) for v in arg.split(",") if v.strip()]
    return values


def build_pyramid(stage_clauses: List[int], stage_dropouts: List[float], tau: float, clause_attn: bool,
                  scale_attn: bool, entropy_weight: float, input_noise: float) -> List[PyramidStageConfig]:
    configs = [
        PyramidStageConfig(pool_size=28, projection_dim=256, n_clauses=256, tau=tau, dropout=0.05),
        PyramidStageConfig(pool_size=14, projection_dim=192, n_clauses=192, tau=tau, dropout=0.05),
        PyramidStageConfig(pool_size=7, projection_dim=128, n_clauses=160, tau=tau, dropout=0.05),
        PyramidStageConfig(pool_size=1, projection_dim=96, n_clauses=128, tau=tau, dropout=0.0),
    ]
    if stage_clauses:
        if len(stage_clauses) != len(configs):
            raise ValueError("stage_clauses length must match number of stages")
        for cfg, val in zip(configs, stage_clauses):
            cfg.n_clauses = val
    if stage_dropouts:
        if len(stage_dropouts) != len(configs):
            raise ValueError("stage_dropouts length must match number of stages")
        for cfg, val in zip(configs, stage_dropouts):
            cfg.dropout = val
    return configs


def build_swin_configs(embed_dims: List[int], depths: List[int], window_size: int, shift_size: int,
                       tm_hidden: List[int], tm_clauses: List[int], head_clauses: List[int],
                       tau: float, dropouts: List[float]) -> List[SwinStageConfig]:
    configs: List[SwinStageConfig] = []
    num = len(embed_dims)
    for idx in range(num):
        configs.append(
            SwinStageConfig(
                embed_dim=embed_dims[idx],
                depth=depths[idx],
                window_size=window_size,
                shift_size=shift_size if idx % 2 == 1 else 0,
                tm_hidden_dim=tm_hidden[idx],
                tm_clauses=tm_clauses[idx],
                head_clauses=head_clauses[idx],
                tau=tau,
                dropout=dropouts[idx],
            )
        )
    return configs


def run_training(args, rank: int, world_size: int, distributed: bool, virtual_multiplier: int = 1) -> None:
    seed = args.seed + rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = get_device(rank if (distributed and torch.cuda.is_available()) else None)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root="/tmp/mnist", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="/tmp/mnist", train=False, download=True, transform=transform)

    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed
        else None
    )
    test_sampler = (
        DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if distributed
        else None
    )

    effective_train_batch = args.batch_size if distributed else args.batch_size * virtual_multiplier
    effective_test_batch = args.test_batch_size if distributed else args.test_batch_size * virtual_multiplier

    train_loader = DataLoader(
        train_ds,
        batch_size=effective_train_batch,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=(device.type != "cpu"),
        persistent_workers=True,
        drop_last=(not distributed and virtual_multiplier > 1),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=effective_test_batch,
        shuffle=False,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=(device.type != "cpu"),
        persistent_workers=True,
    )

    sample = train_ds[0][0]
    if sample.dim() == 3:
        in_channels = sample.shape[0]
        image_size = sample.shape[-2:]
    else:
        in_channels = 1
        image_size = (28, 28)

    stage_clause_overrides = parse_sequence(args.stage_clauses, int) if args.stage_clauses else []
    stage_dropout_overrides = parse_sequence(args.stage_dropouts, float) if args.stage_dropouts else []

    metadata_stage_configs: List[dict]

    if args.tm_variant == "swin":
        embed_dims = parse_sequence(args.swin_embed_dims, int) or [96, 192, 384, 768]
        depths = parse_sequence(args.swin_depths, int) or [2, 2, 4, 2]
        if len(depths) != len(embed_dims):
            raise ValueError("Length of --swin-depths must match embed dims")
        tm_hidden = parse_sequence(args.swin_tm_hidden, int) if args.swin_tm_hidden else []
        if not tm_hidden:
            tm_hidden = [256 for _ in embed_dims]
        tm_clauses = parse_sequence(args.swin_tm_clauses, int) or stage_clause_overrides or [256 for _ in embed_dims]
        head_clauses = parse_sequence(args.swin_head_clauses, int) or tm_clauses
        dropouts = stage_dropout_overrides or [0.05 for _ in embed_dims]
        for seq in (tm_hidden, tm_clauses, head_clauses, dropouts):
            if len(seq) < len(embed_dims):
                seq.extend([seq[-1]] * (len(embed_dims) - len(seq)))
        swin_configs = build_swin_configs(
            embed_dims,
            depths,
            args.swin_window_size,
            args.swin_shift_size,
            tm_hidden,
            tm_clauses,
            head_clauses,
            args.tau_start,
            dropouts,
        )
        metadata_stage_configs = [cfg.__dict__ for cfg in swin_configs]
        model_core = SwinLikePyramidTM(
            image_size=image_size,
            in_channels=in_channels,
            num_classes=10,
            stage_configs=swin_configs,
            use_scale_attention=not args.no_scale_attention,
            scale_entropy_weight=args.scale_entropy_weight,
            patch_size=args.patch_size,
        ).to(device)
    else:
        pyramid_configs = build_pyramid(
            stage_clause_overrides,
            stage_dropout_overrides,
            args.tau_start,
            not args.no_clause_attention,
            not args.no_scale_attention,
            args.scale_entropy_weight,
            args.input_noise_std,
        )
        metadata_stage_configs = [cfg.__dict__ for cfg in pyramid_configs]
        model_core = PyramidTM(
            stage_configs=pyramid_configs,
            use_clause_attention=not args.no_clause_attention,
            attention_heads=args.attention_heads,
            use_scale_attention=not args.no_scale_attention,
            scale_entropy_weight=args.scale_entropy_weight,
            input_noise_std=args.input_noise_std,
        ).to(device)

    ddp_kwargs = {}
    if distributed and device.type == "cuda":
        ddp_kwargs = {"device_ids": [device.index], "output_device": device.index, "broadcast_buffers": False}
    elif distributed:
        ddp_kwargs = {"broadcast_buffers": False}

    model = DistributedDataParallel(model_core, **ddp_kwargs) if distributed else model_core

    optimizer = torch.optim.AdamW(model_core.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs - args.warmup_epochs),
        eta_min=args.min_lr,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    ema = EMAWrapper(model_core, decay=args.ema_decay) if args.ema_decay > 0 else None

    logger = (
        ExperimentLogger(
            output_dir=args.output_dir,
            run_name=f"{args.tm_variant}_tm",
            metadata={
                "model_name": args.tm_variant,
                "epochs": args.epochs,
                "lr": args.lr,
                "min_lr": args.min_lr,
                "weight_decay": args.weight_decay,
                "grad_clip": args.grad_clip,
                "grad_accum": args.grad_accum,
                "tau_start": args.tau_start,
                "tau_end": args.tau_end,
                "warmup_epochs": args.warmup_epochs,
                "stage_configs": metadata_stage_configs,
                "seed": args.seed,
                "device": device.type,
                "attention_heads": args.attention_heads,
                "use_clause_attention": not args.no_clause_attention,
                "use_scale_attention": not args.no_scale_attention,
                "scale_entropy_weight": args.scale_entropy_weight,
                "ema_decay": args.ema_decay,
                "input_noise_std": args.input_noise_std,
                "tm_variant": args.tm_variant,
                "export_path": str(args.export_path) if args.export_path else None,
                "patch_size": args.patch_size,
                "batch_multiplier": world_size,
                "virtual_batch_multiplier": virtual_multiplier,
            },
        )
        if is_main_process(rank)
        else NullLogger()
    )

    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

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
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if not distributed and virtual_multiplier > 1:
                x_chunks = torch.chunk(x, virtual_multiplier, dim=0)
                y_chunks = torch.chunk(y, virtual_multiplier, dim=0)
            else:
                x_chunks = (x,)
                y_chunks = (y,)

            for micro_x, micro_y in zip(x_chunks, y_chunks):
                if micro_x.numel() == 0:
                    continue
                num_batches += 1
                accum_counter += 1
                xb = flatten_images(micro_x)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits, _, _ = model(xb, use_ste=False)
                    loss = F.cross_entropy(logits, micro_y)
                    aux_loss = model.attention_entropy_loss() if hasattr(model, "attention_entropy_loss") else None
                    if aux_loss is not None:
                        loss = loss + aux_loss
                    scaled_loss = loss / args.grad_accum
                scaler.scale(scaled_loss).backward()

                if accum_counter >= args.grad_accum:
                    scaler.unscale_(optimizer)
                    if args.grad_clip and args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model_core.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    accum_counter = 0
                    if ema is not None:
                        ema.update(model_core)

                epoch_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == micro_y).sum().item()
                total += micro_y.size(0)

        if accum_counter > 0:
            scaler.unscale_(optimizer)
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model_core.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model_core)

        duration = time.time() - start

        global_loss = reduce_sum(epoch_loss, device) if distributed else epoch_loss
        global_batches = reduce_sum(num_batches, device) if distributed else num_batches
        avg_loss = global_loss / max(1.0, global_batches)

        global_correct = reduce_sum(correct, device) if distributed else correct
        global_total = reduce_sum(total, device) if distributed else total
        train_acc = global_correct / max(1.0, global_total)

        duration = reduce_max(duration, device) if distributed else duration

        if ema is not None:
            ema.apply_shadow(model_core)
        test_acc = evaluate(model_core, test_loader, device, use_ste=True, distributed=distributed)
        if ema is not None:
            ema.restore(model_core)

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        diagnostics = (
            collect_tm_diagnostics(model_core, threshold=0.5) if is_main_process(rank) else {}
        )

        if is_main_process(rank):
            logger.log_epoch(
                EpochMetrics(
                    epoch=epoch,
                    train_loss=avg_loss,
                    train_accuracy=train_acc,
                    eval_accuracy=test_acc,
                    duration_s=duration,
                    diagnostics=diagnostics,
                    extra={"lr": current_lr, "best_test_acc": best_test_acc, "world_size": world_size},
                )
            )
            print(
                f"PyramidTM | epoch {epoch:02d}/{args.epochs:02d} | loss={avg_loss:.4f} | "
                f"train_acc={train_acc:.4f} | test_acc={test_acc:.4f} | best_acc={best_test_acc:.4f} | "
                f"lr={current_lr:.5f} | {duration:4.1f}s"
            )

        new_tau = linear_tau(args.tau_start, args.tau_end, epoch, args.epochs)
        anneal_ste_factor(model_core, new_tau)

        if epoch >= args.warmup_epochs:
            scheduler.step()

    if is_main_process(rank):
        logger.flush_all()

    if ema is not None:
        ema.apply_shadow(model_core)
    final_acc = evaluate(model_core, test_loader, device, use_ste=True, distributed=distributed)
    if ema is not None:
        ema.restore(model_core)

    if is_main_process(rank):
        if args.export_path:
            stage_exports = {f"stage_{idx}": stage.tm.discretize() for idx, stage in enumerate(model_core.stages)}
            attention_weights = None
            if hasattr(model_core, "last_attention_weights") and model_core.last_attention_weights is not None:
                attention_weights = model_core.last_attention_weights.detach().cpu().tolist()
            payload = {
                "final_test_accuracy": final_acc,
                "stage_exports": stage_exports,
                "attention_weights": attention_weights,
            }
            args.export_path.parent.mkdir(parents=True, exist_ok=True)
            args.export_path.write_text(json.dumps(payload, indent=2))
            print(f"Exported discretized model to {args.export_path}")

        print(f"Final test accuracy: {final_acc:.4f}")


def _distributed_worker(rank: int, args) -> None:
    world_size = args.batch_multiplier
    setup_distributed(rank, world_size, args.dist_backend, args.dist_url)
    try:
        run_training(args, rank=rank, world_size=world_size, distributed=True, virtual_multiplier=1)
    finally:
        cleanup_distributed()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TM variants on MNIST.")
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
    parser.add_argument("--tm-variant", choices=["pyramid", "swin"], default="pyramid")
    parser.add_argument("--swin-embed-dims", type=str, default="")
    parser.add_argument("--swin-depths", type=str, default="")
    parser.add_argument("--swin-window-size", type=int, default=4)
    parser.add_argument("--swin-shift-size", type=int, default=2)
    parser.add_argument("--swin-tm-hidden", type=str, default="")
    parser.add_argument("--swin-tm-clauses", type=str, default="")
    parser.add_argument("--swin-head-clauses", type=str, default="")
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--batch-multiplier", type=int, default=1, help="Number of parallel replicas (world size)")
    parser.add_argument("--dist-backend", type=str, default="nccl", help="Distributed backend to use")
    parser.add_argument(
        "--dist-url",
        type=str,
        default="",
        help="Initialization URL for distributed training (defaults to tcp://127.0.0.1:<free_port>)",
    )
    args = parser.parse_args()

    if args.batch_multiplier < 1:
        raise ValueError("--batch-multiplier must be >= 1")

    available_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if args.batch_multiplier > 1 and torch.cuda.is_available() and available_cuda >= args.batch_multiplier:
        if not args.dist_url:
            port = find_free_port()
            args.dist_url = f"tcp://127.0.0.1:{port}"
        if args.dist_backend == "nccl" and available_cuda == 0:
            raise RuntimeError("NCCL backend requires CUDA devices")
        mp.spawn(_distributed_worker, args=(args,), nprocs=args.batch_multiplier, join=True)
    else:
        virtual_multiplier = 1
        if args.batch_multiplier > 1:
            virtual_multiplier = args.batch_multiplier
            if torch.cuda.is_available() and available_cuda < args.batch_multiplier:
                print(
                    f"[Info] Only {available_cuda} CUDA device(s) detected — running virtual batch multiplier "
                    f"{virtual_multiplier} on a single process."
                )
            elif not torch.cuda.is_available():
                print(
                    f"[Info] CUDA not available — running virtual batch multiplier {virtual_multiplier} on CPU/MPS."
                )
        run_training(args, rank=0, world_size=1, distributed=False, virtual_multiplier=virtual_multiplier)


if __name__ == "__main__":
    main()
