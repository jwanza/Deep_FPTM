#!/usr/bin/env python3
"""
Compare baseline vs. enhanced result JSON files emitted by run_validation_suite.

Example:
  python python/fptm_ste/experiments/analyze_results.py \
      --baseline results/cifar10/deep_cstcm/baseline.json \
      --enhanced results/cifar10/deep_cstcm/enhanced.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional


def _load_results(path: Path) -> Dict[str, Dict[str, float]]:
    data = json.loads(path.read_text())
    summary: Dict[str, Dict[str, float]] = {}
    for label, payload in data.items():
        summary[label] = {
            "train_accuracy": float(payload.get("train_accuracy", 0.0)),
            "test_accuracy": float(payload.get("test_accuracy", 0.0)),
            "best_test_accuracy": float(payload.get("best_epoch_test_accuracy", payload.get("test_accuracy", 0.0))),
            "train_time_s": float(payload.get("train_time_s", 0.0)),
            "throughput": float(payload.get("profile", {}).get("samples_per_second") or 0.0),
        }
    return summary


def _format_delta(new: float, old: float) -> str:
    delta = new - old
    sign = "+" if delta >= 0 else "-"
    return f"{new:.3f} ({sign}{abs(delta):.3f})"


def compare(baseline: Dict[str, Dict[str, float]], enhanced: Dict[str, Dict[str, float]], metric: str) -> None:
    keys = sorted(set(baseline.keys()) | set(enhanced.keys()))
    print(f"{'Model':20s} | {'Baseline':>16s} | {'Enhanced':>16s}")
    print("-" * 60)
    for key in keys:
        base_val = baseline.get(key, {}).get(metric)
        new_val = enhanced.get(key, {}).get(metric)
        if base_val is None or new_val is None:
            print(f"{key:20s} | {str(base_val):>16s} | {str(new_val):>16s}")
            continue
        print(f"{key:20s} | {base_val:16.3f} | {_format_delta(new_val, base_val):>16s}")


def main(argv: Optional[object] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two result JSON files.")
    parser.add_argument("--baseline", required=True, help="Path to baseline JSON output.")
    parser.add_argument("--enhanced", required=True, help="Path to enhanced JSON output.")
    parser.add_argument("--metric", default="test_accuracy", choices=["train_accuracy", "test_accuracy", "best_test_accuracy", "throughput"], help="Metric to compare.")
    args = parser.parse_args(argv)

    baseline_path = Path(args.baseline)
    enhanced_path = Path(args.enhanced)
    if not baseline_path.exists():
        parser.error(f"Baseline file '{baseline_path}' not found.")
    if not enhanced_path.exists():
        parser.error(f"Enhanced file '{enhanced_path}' not found.")

    baseline = _load_results(baseline_path)
    enhanced = _load_results(enhanced_path)
    compare(baseline, enhanced, metric=args.metric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



