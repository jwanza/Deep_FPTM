"""
Convenience wrapper that launches the PyramidTM MNIST experiment with recommended
settings aimed at high accuracy.
"""

from pathlib import Path
import subprocess
import sys

THIS_DIR = Path(__file__).parent
RUNNER = THIS_DIR / "run_pyramid.py"

DEFAULT_ARGS = [
    "--epochs", "30",
    "--batch-size", "256",
    "--test-batch-size", "512",
    "--lr", "8e-4",
    "--min-lr", "1e-4",
    "--warmup-epochs", "3",
    "--grad-accum", "2",
    "--grad-clip", "0.4",
    "--tau-start", "0.6",
    "--tau-end", "0.25",
    "--attention-heads", "2",
    "--scale-entropy-weight", "0.02",
    "--ema-decay", "0.999",
    "--input-noise-std", "0.03",
    "--stage-clauses", "384,256,192,128",
    "--stage-dropouts", "0.1,0.1,0.05,0.0",
    "--output-dir", str(THIS_DIR),
]


def main() -> None:
    cmd = [sys.executable, str(RUNNER)] + DEFAULT_ARGS
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
