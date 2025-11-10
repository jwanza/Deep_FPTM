"""
Utility to summarise PyramidTM export bundles.
"""

import argparse
import json
from pathlib import Path


def summarize_export(path: Path) -> None:
    data = json.loads(path.read_text())
    print(f"Export: {path}")
    print(f"Final test accuracy: {data.get('final_test_accuracy', 'n/a')}")

    stage_exports = data.get("stage_exports", {})
    for stage, bundle in stage_exports.items():
        pos = bundle.get("positive", [])
        neg = bundle.get("negative", [])
        pos_inv = bundle.get("positive_inv", [])
        neg_inv = bundle.get("negative_inv", [])
        clauses_num = bundle.get("clauses_num", len(pos) + len(neg))
        mean_literals = sum(len(c) for c in pos + pos_inv + neg + neg_inv) / max(1, clauses_num * 4)
        print(f"  {stage}: clauses={clauses_num} mean_literals={mean_literals:.2f}")

    attention_weights = data.get("attention_weights")
    if attention_weights is not None:
        print(f"Attention weights: {attention_weights}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise PyramidTM export JSON file.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    summarize_export(args.path)


if __name__ == "__main__":
    main()
