import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch


def _sigmoid_mask(tensor: torch.Tensor, threshold: float) -> torch.Tensor:
    probs = torch.sigmoid(tensor.detach())
    return (probs >= threshold).float()


def _mean_float(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return float(tensor.detach().mean().item())


def collect_tm_diagnostics(module: torch.nn.Module, threshold: float = 0.5) -> Dict[str, Any]:
    """Collect tau, clause sparsity, and attention statistics across submodules."""

    tau_values: Dict[str, float] = {}
    sparsity: Dict[str, Dict[str, float]] = {}
    attention: Dict[str, List[float]] = {}

    for name, submodule in module.named_modules():
        key = name or "root"

        if hasattr(submodule, "tau"):
            try:
                tau_values[key] = float(getattr(submodule, "tau"))
            except (TypeError, ValueError):
                pass

        literal_masks = []
        literal_stats: Dict[str, float] = {}
        for attr, label in (
            ("ta_pos", "positive"),
            ("ta_neg", "negative"),
            ("ta_pos_inv", "positive_inv"),
            ("ta_neg_inv", "negative_inv"),
        ):
            if hasattr(submodule, attr):
                tensor = getattr(submodule, attr)
                mask = _sigmoid_mask(tensor, threshold)
                literal_stats[label] = _mean_float(mask)
                literal_masks.append(mask)

        if literal_stats:
            if literal_masks:
                literal_stats["overall"] = _mean_float(torch.cat(literal_masks))
            sparsity[key] = literal_stats

        if hasattr(submodule, "last_attention_weights"):
            weights = getattr(submodule, "last_attention_weights")
            if weights is not None:
                if isinstance(weights, torch.Tensor):
                    attention[key] = weights.detach().cpu().tolist()
                elif isinstance(weights, Iterable):
                    attention[key] = list(weights)

    return {
        "tau": tau_values,
        "clause_sparsity": sparsity,
        "attention": attention,
    }


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    eval_accuracy: float
    duration_s: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "eval_accuracy": self.eval_accuracy,
            "duration_s": self.duration_s,
            "diagnostics": self.diagnostics,
        }
        if self.extra:
            payload["extra"] = self.extra
        return payload


@dataclass
class ExperimentLogger:
    output_dir: Path
    run_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    _records: List[EpochMetrics] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def json_path(self) -> Path:
        return self.output_dir / f"{self.run_name}.json"

    @property
    def csv_path(self) -> Path:
        return self.output_dir / f"{self.run_name}.csv"

    def log_epoch(self, metrics: EpochMetrics) -> None:
        self._records.append(metrics)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "run_name": self.run_name,
            "metadata": self.metadata,
            "epochs": [record.to_dict() for record in self._records],
        }

    def flush_json(self) -> None:
        self.json_path.write_text(json.dumps(self.to_payload(), indent=2))

    def flush_csv(self) -> None:
        if not self._records:
            return

        fieldnames = [
            "epoch",
            "train_loss",
            "train_accuracy",
            "eval_accuracy",
            "duration_s",
            "diagnostics_json",
        ]
        include_extra = any(record.extra for record in self._records)
        if include_extra:
            fieldnames.append("extra_json")

        with self.csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for record in self._records:
                row = {
                    "epoch": record.epoch,
                    "train_loss": record.train_loss,
                    "train_accuracy": record.train_accuracy,
                    "eval_accuracy": record.eval_accuracy,
                    "duration_s": record.duration_s,
                    "diagnostics_json": json.dumps(record.diagnostics),
                }
                if include_extra:
                    row["extra_json"] = json.dumps(record.extra)
                writer.writerow(row)

    def flush_all(self) -> None:
        self.flush_json()
        self.flush_csv()
