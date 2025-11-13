from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from fptm_ste.tests.run_mnist_equiv import DEFAULT_DATA_ROOT, DEFAULT_IMAGENET_ROOT, build_arg_parser, run_experiment_with_args


def _parse_float_list(text: str) -> List[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _parse_int_list(text: str) -> List[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


@dataclass
class SweepConfig:
    patch_size: int
    drop_path: float
    clause_multiplier: float

    def slug(self) -> str:
        return f"patch{self.patch_size}_drop{self.drop_path:.2f}_mult{self.clause_multiplier:.2f}".replace(".", "p")


def generate_sweep_configs(
    patch_sizes: Sequence[int],
    drop_paths: Sequence[float],
    clause_multipliers: Sequence[float],
) -> Iterable[SweepConfig]:
    for patch in patch_sizes:
        for drop in drop_paths:
            for mult in clause_multipliers:
                yield SweepConfig(patch, drop, mult)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ImageNet clause/patch sweeps for TM-based transformers.")
    parser.add_argument("--train-dir", type=str, default=None, help="Path to ImageNet training directory.")
    parser.add_argument("--val-dir", type=str, default=None, help="Path to ImageNet validation directory.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Fallback dataset root (defaults to TM_IMAGENET_ROOT).")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run training on (cuda/cpu).")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs per sweep configuration.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--test-batch-size", type=int, default=256, help="Validation batch size.")
    parser.add_argument("--backend", choices=["ste", "deeptm"], default="deeptm", help="TM backend to benchmark.")
    parser.add_argument("--embed-dim", type=int, default=768, help="Vision transformer embedding dimension.")
    parser.add_argument("--layers", type=int, default=12, help="Number of transformer encoder layers.")
    parser.add_argument("--heads", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--mlp-ratio", type=float, default=4.0, help="Transformer MLP ratio.")
    parser.add_argument("--base-clauses", type=int, default=384, help="Base TM clause count before applying multipliers.")
    parser.add_argument("--patch-sizes", type=str, default="14,16", help="Comma-separated patch sizes to sweep.")
    parser.add_argument("--drop-paths", type=str, default="0.1,0.2,0.3", help="Comma-separated drop-path rates to sweep.")
    parser.add_argument("--clause-multipliers", type=str, default="1.0,1.5,2.0", help="Comma-separated clause multipliers.")
    parser.add_argument("--mixup-alpha", type=float, default=0.8, help="Mixup alpha coefficient.")
    parser.add_argument("--cutmix-alpha", type=float, default=1.0, help="CutMix alpha coefficient.")
    parser.add_argument("--randaugment", action=argparse.BooleanOptionalAction, default=True, help="Enable RandAugment.")
    parser.add_argument("--randaugment-n", type=int, default=2, help="RandAugment N parameter.")
    parser.add_argument("--randaugment-m", type=int, default=9, help="RandAugment magnitude parameter.")
    parser.add_argument("--use-flash-attn", action=argparse.BooleanOptionalAction, default=True, help="Enable FlashAttention kernels.")
    parser.add_argument("--residual-attn", action=argparse.BooleanOptionalAction, default=True, help="Enable residual attention connections.")
    parser.add_argument("--relative-pos", choices=["none", "learned", "rotary"], default="learned", help="Relative positional encoding mode.")
    parser.add_argument("--output-dir", type=str, default="imagenet_sweeps", help="Directory to store sweep artifacts.")
    parser.add_argument("--summary-json", type=str, default="imagenet_sweep_summary.json", help="Filename for aggregated summary JSON.")

    args = parser.parse_args()

    patch_sizes = _parse_int_list(args.patch_sizes)
    drop_paths = _parse_float_list(args.drop_paths)
    clause_multipliers = _parse_float_list(args.clause_multipliers)

    sweep_dir = Path(args.output_dir)
    log_dir = sweep_dir / "logs"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    parser_defaults = build_arg_parser()
    base_args = parser_defaults.parse_args([])
    base_args.dataset = "imagenet"
    base_args.models = ["transformer"]
    base_args.device = args.device
    base_args.epochs = args.epochs
    base_args.batch_size = args.batch_size
    base_args.test_batch_size = args.test_batch_size
    base_args.transformer_arch = "vit"
    base_args.transformer_backend = args.backend
    base_args.transformer_d_model = args.embed_dim
    base_args.transformer_layers = args.layers
    base_args.transformer_heads = args.heads
    base_args.transformer_mlp_ratio = str(args.mlp_ratio)
    base_args.transformer_use_flash_attn = args.use_flash_attn
    base_args.transformer_residual_attn = args.residual_attn
    base_args.transformer_relative_pos = args.relative_pos
    base_args.mixup_alpha = args.mixup_alpha
    base_args.cutmix_alpha = args.cutmix_alpha
    base_args.randaugment = args.randaugment
    base_args.randaugment_n = args.randaugment_n
    base_args.randaugment_m = args.randaugment_m
    base_args.label_smoothing = 0.1
    base_args.transformer_log_dir = ""
    base_args.write_bool_dataset = False
    base_args.download = False

    dataset_root = args.dataset_root or args.train_dir or DEFAULT_IMAGENET_ROOT
    base_args.dataset_root = dataset_root
    base_args.imagenet_train_dir = args.train_dir
    base_args.imagenet_val_dir = args.val_dir
    base_args.output_json = str(sweep_dir / "checkpoint.json")

    summary: Dict[str, Dict[str, object]] = {}

    for config in generate_sweep_configs(patch_sizes, drop_paths, clause_multipliers):
        run_args = argparse.Namespace(**vars(base_args))
        clause_count = int(args.base_clauses * config.clause_multiplier)
        run_args.transformer_patch = config.patch_size
        run_args.transformer_drop_path = config.drop_path
        run_args.transformer_clauses = clause_count
        run_args.seed = 42
        slug = config.slug()
        run_args.output_json = str(sweep_dir / f"{slug}.json")
        run_args.transformer_log_dir = str(log_dir / slug)

        print(f"\n=== Running sweep configuration: {slug} ===")
        results = run_experiment_with_args(run_args)
        transformer_result = results.get("transformer", {})
        summary[slug] = {
            "config": asdict(config),
            "clauses": clause_count,
            "patch_size": config.patch_size,
            "drop_path": config.drop_path,
            "metrics": transformer_result,
        }

    summary_path = sweep_dir / args.summary_json
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSweep summary written to {summary_path}")


if __name__ == "__main__":
    main()

