from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Dict, List, Sequence, Tuple

import torch

from fptm_ste.tm_transformer import UnifiedTMTransformer


@dataclass
class BenchmarkModelConfig:
    name: str
    image_size: Tuple[int, int]
    in_channels: int
    builder_args: Dict[str, object]


MODEL_REGISTRY: Dict[str, BenchmarkModelConfig] = {
    "tm_vit_ste_base": BenchmarkModelConfig(
        name="TM-ViT-STE-Base",
        image_size=(32, 32),
        in_channels=3,
        builder_args=dict(
            architecture="vit",
            backend="ste",
            embed_dim=128,
            depths=4,
            num_heads=4,
            mlp_ratio=4.0,
            tm_clauses=128,
            drop_path_rate=0.1,
        ),
    ),
    "tm_vit_deeptm_base": BenchmarkModelConfig(
        name="TM-ViT-DeepTM-Base",
        image_size=(32, 32),
        in_channels=3,
        builder_args=dict(
            architecture="vit",
            backend="deeptm",
            embed_dim=128,
            depths=4,
            num_heads=4,
            mlp_ratio=4.0,
            tm_clauses=192,
            drop_path_rate=0.1,
        ),
    ),
    "tm_swin_deeptm_tiny": BenchmarkModelConfig(
        name="TM-Swin-DeepTM-Tiny",
        image_size=(32, 32),
        in_channels=3,
        builder_args=dict(
            architecture="swin",
            backend="deeptm",
            embed_dim=(48, 96),
            depths=(2, 2),
            num_heads=(3, 6),
            mlp_ratio=(4.0, 4.0),
            tm_clauses=(128, 160),
            window_size=4,
            drop_path_rate=0.1,
        ),
    ),
}


def parse_batch_sizes(text: str) -> Sequence[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def resolve_device(device_flag: str) -> torch.device:
    if device_flag == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_flag)


def benchmark_model(
    config: BenchmarkModelConfig,
    batch_sizes: Sequence[int],
    device: torch.device,
    warmup_steps: int,
    measure_steps: int,
) -> Dict[str, object]:
    results: Dict[str, object] = {"model": config.name, "device": str(device), "batches": []}

    model = UnifiedTMTransformer(
        num_classes=10,
        image_size=config.image_size,
        in_channels=config.in_channels,
        **config.builder_args,
    ).to(device)
    model.eval()

    use_cuda_amp = device.type == "cuda"

    for batch_size in batch_sizes:
        sample = torch.randn(batch_size, config.in_channels, *config.image_size, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Warmup
        for _ in range(warmup_steps):
            with torch.no_grad():
                with (torch.cuda.amp.autocast() if use_cuda_amp else nullcontext()):
                    model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()

        timings: List[float] = []
        with torch.no_grad():
            for _ in range(measure_steps):
                start = time.perf_counter()
                with (torch.cuda.amp.autocast() if use_cuda_amp else nullcontext()):
                    model(sample)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000.0
                timings.append(elapsed)

        avg_latency = sum(timings) / len(timings)
        throughput = (batch_size * 1000.0) / avg_latency
        assumed_power_w = 350.0 if device.type == "cuda" else 65.0
        energy_mj = avg_latency * assumed_power_w

        results["batches"].append(
            {
                "batch_size": batch_size,
                "latency_ms": avg_latency,
                "throughput_img_per_s": throughput,
                "energy_mj": energy_mj,
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TM transformer variants.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["tm_vit_ste_base", "tm_swin_deeptm_tiny"],
        choices=list(MODEL_REGISTRY.keys()),
        help="Model registry keys to benchmark.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,32",
        help="Comma separated batch sizes to profile (e.g. '1,32,64').",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to profile on (auto chooses CUDA if available).",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations before measurement.")
    parser.add_argument("--iters", type=int, default=20, help="Measured iterations per batch size.")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Optional JSON output file.")

    args = parser.parse_args()
    device = resolve_device(args.device)
    batch_sizes = parse_batch_sizes(args.batch_sizes)

    report: Dict[str, object] = {
        "device": str(device),
        "batch_sizes": batch_sizes,
        "warmup": args.warmup,
        "iterations": args.iters,
        "results": [],
    }

    for model_key in args.models:
        config = MODEL_REGISTRY[model_key]
        report["results"].append(
            benchmark_model(config, batch_sizes, device, warmup_steps=args.warmup, measure_steps=args.iters)
        )

    print(json.dumps(report, indent=2))
    if args.output:
        with open(args.output, "w") as fh:
            json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()

