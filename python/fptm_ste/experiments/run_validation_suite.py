#!/usr/bin/env python3
"""
Utility entrypoint that orchestrates the validation strategy described in
docs/TEST_REPORT.md.  It stitches together fast unit suites, MNIST/CIFAR smoke
tests, and config-driven CIFAR-10 sweeps so we can reproduce benchmark numbers
with one command.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUN_MNIST_EQUIV = PROJECT_ROOT / "python" / "fptm_ste" / "tests" / "run_mnist_equiv.py"


@dataclass
class CommandSpec:
    label: str
    argv: List[str]
    description: str = ""
    env: Optional[Dict[str, str]] = None
    output_json: Optional[Path] = None


def _python() -> str:
    return sys.executable or "python3"


def _resolve_tokens(argv: Sequence[str]) -> List[str]:
    resolved: List[str] = []
    for token in argv:
        if token == "{python}":
            resolved.append(_python())
        elif token == "{run_mnist_equiv}":
            resolved.append(str(RUN_MNIST_EQUIV))
        else:
            resolved.append(token)
    return resolved


def _ensure_output_arg(argv: List[str], output_path: Optional[Path]) -> List[str]:
    if output_path is None:
        return argv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if "--output-json" in argv:
        return argv
    return argv + ["--output-json", str(output_path)]


def _run_command(
    spec: CommandSpec,
    log_dir: Path,
    dry_run: bool = False,
) -> Dict[str, object]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{spec.label}.log"
    argv = _ensure_output_arg(_resolve_tokens(spec.argv), spec.output_json)
    env = dict(os.environ)
    if spec.env:
        env.update(spec.env)
    quoted = " ".join(shlex.quote(part) for part in argv)
    metadata = {
        "label": spec.label,
        "description": spec.description,
        "command": quoted,
        "log_path": str(log_path),
    }
    if dry_run:
        print(f"[DRY-RUN] {quoted}")
        return {**metadata, "returncode": 0, "summary": None, "duration": 0.0}

    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        if spec.description:
            log_file.write(f"# {spec.description}\n")
        log_file.write(f"$ {quoted}\n\n")
        log_file.flush()
        result = subprocess.run(argv, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    duration = time.time() - start

    summary = None
    if spec.output_json and Path(spec.output_json).exists():
        try:
            summary = json.loads(Path(spec.output_json).read_text())
        except json.JSONDecodeError:
            summary = {"error": "Failed to parse output JSON"}

    return {
        **metadata,
        "returncode": result.returncode,
        "duration": duration,
        "summary": summary,
    }


def _suite_unit() -> List[CommandSpec]:
    return [
        CommandSpec(
            label="pytest-core-backbone",
            description="Backbone + binarizer + spatial regression tests.",
            argv=[
                "{python}",
                "-m",
                "pytest",
                "python/fptm_ste/tests/test_backbone_factory.py",
                "python/fptm_ste/tests/test_binarizers.py",
                "python/fptm_ste/tests/test_spatial_tm.py",
                "python/fptm_ste/tests/test_schedule.py",
            ],
        ),
    ]


def _suite_mnist_smoke() -> List[CommandSpec]:
    base = ["{python}", "{run_mnist_equiv}", "--batch-size", "64", "--test-batch-size", "256"]
    return [
        CommandSpec(
            label="mnist-tm-baseline",
            description="STE TM sanity check without modern tricks.",
            argv=base
            + [
                "--dataset",
                "mnist",
                "--models",
                "tm",
                "--epochs",
                "1",
                "--label-smoothing",
                "0.0",
                "--mixup-alpha",
                "0.0",
                "--cutmix-alpha",
                "0.0",
            ],
        ),
        CommandSpec(
            label="mnist-tm-enhanced",
            description="STE TM with smoothing + mixup verifying new training hooks.",
            argv=base
            + [
                "--dataset",
                "mnist",
                "--models",
                "tm",
                "--epochs",
                "2",
                "--label-smoothing",
                "0.1",
                "--mixup-alpha",
                "0.2",
                "--cutmix-alpha",
                "0.0",
            ],
        ),
        CommandSpec(
            label="mnist-deep-tm",
            description="Deep TM stack smoke test.",
            argv=base
            + [
                "--dataset",
                "mnist",
                "--models",
                "deep_tm",
                "--epochs",
                "2",
            ],
        ),
        CommandSpec(
            label="mnist-transformer",
            description="Transformer hybrid sanity run (ViT backend).",
            argv=base
            + [
                "--dataset",
                "mnist",
                "--models",
                "transformer",
                "--epochs",
                "1",
                "--transformer-arch",
                "vit",
                "--transformer-d-model",
                "64",
                "--transformer-layers",
                "2",
                "--transformer-clauses",
                "128",
            ],
        ),
    ]


def _suite_cifar_smoke(results_dir: Path) -> List[CommandSpec]:
    base = [
        "{python}",
        "{run_mnist_equiv}",
        "--dataset",
        "cifar10",
        "--batch-size",
        "96",
        "--test-batch-size",
        "256",
        "--epochs",
        "3",
    ]
    return [
        CommandSpec(
            label="cifar10-tm-smoke",
            description="CIFAR-10 STE TM baseline (short run).",
            argv=base
            + [
                "--models",
                "tm",
                "--label-smoothing",
                "0.0",
                "--mixup-alpha",
                "0.0",
            ],
            output_json=results_dir / "cifar10" / "tm" / "smoke.json",
        ),
        CommandSpec(
            label="cifar10-hybrid-smoke",
            description="CIFAR-10 CNN+TM hybrid with learnable binarizer.",
            argv=base
            + [
                "--models",
                "hybrid",
                "--hybrid-thresholds",
                "32",
                "--label-smoothing",
                "0.1",
            ],
            output_json=results_dir / "cifar10" / "hybrid" / "smoke.json",
        ),
    ]


def _load_config_specs(config_dir: Path, tags: Iterable[str], results_dir: Path) -> List[CommandSpec]:
    specs: List[CommandSpec] = []
    tag_set = set(tags)
    if not config_dir.exists():
        return specs
    for path in sorted(config_dir.glob("*.json")):
        data = json.loads(path.read_text())
        config_tags = set(data.get("tags", []))
        if tag_set and not tag_set.issubset(config_tags):
            continue
        argv = data["command"]
        output_json = data.get("output_json")
        output_path = None
        if output_json:
            output_path = Path(output_json)
            if not output_path.is_absolute():
                output_path = (results_dir / output_path).resolve()
        specs.append(
            CommandSpec(
                label=data["label"],
                description=data.get("description", ""),
                argv=argv,
                env=data.get("env"),
                output_json=output_path,
            )
        )
    return specs


SUITE_BUILDERS = {
    "unit": _suite_unit,
    "mnist-smoke": _suite_mnist_smoke,
    "cifar-smoke": _suite_cifar_smoke,
}


def build_suite(name: str, config_dir: Path, results_dir: Path) -> List[CommandSpec]:
    if name in SUITE_BUILDERS:
        specs = SUITE_BUILDERS[name](results_dir) if name == "cifar-smoke" else SUITE_BUILDERS[name]()
    elif name == "cifar-full":
        specs = _load_config_specs(config_dir, tags={"cifar-full"}, results_dir=results_dir)
    elif name == "transformer-smoke":
        specs = _load_config_specs(config_dir, tags={"transformer-smoke"}, results_dir=results_dir)
    else:
        raise ValueError(f"Unknown suite '{name}'. Available: {', '.join(sorted(SUITE_BUILDERS.keys()) + ['cifar-full', 'transformer-smoke'])}")
    return specs


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run validation suites (unit, smoke, CIFAR sweeps).")
    parser.add_argument("--suite", default="unit", help="Comma separated list of suites to run.")
    parser.add_argument("--config-dir", default="python/fptm_ste/experiments/configs", help="Directory containing JSON configs.")
    parser.add_argument("--results-dir", default="results", help="Directory root for result JSON/metrics.")
    parser.add_argument("--log-dir", default="logs/validation", help="Directory to store stdout/stderr logs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args(argv)

    config_dir = (PROJECT_ROOT / args.config_dir).resolve()
    results_dir = (PROJECT_ROOT / args.results_dir).resolve()
    log_dir = (PROJECT_ROOT / args.log_dir).resolve()

    suites = [s.strip() for s in args.suite.split(",") if s.strip()]
    if not suites:
        parser.error("At least one suite must be specified.")

    all_specs: List[CommandSpec] = []
    for suite in suites:
        all_specs.extend(build_suite(suite, config_dir, results_dir))

    if not all_specs:
        print("No commands to run (suite definitions may have been filtered out).")
        return 0

    results = []
    for spec in all_specs:
        res = _run_command(spec, log_dir, dry_run=args.dry_run)
        results.append(res)
        status = "OK" if res["returncode"] == 0 else "FAIL"
        print(f"[{status}] {spec.label} ({res['duration']:.1f}s) -> log: {res['log_path']}")

    failures = [res for res in results if res["returncode"] != 0]
    if failures:
        print("\nThe following commands failed:")
        for res in failures:
            print(f"- {res['label']} (log: {res['log_path']})")
        return 1

    print("\nAll commands completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

