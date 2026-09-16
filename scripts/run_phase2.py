"""Runs Phase 2 in full: trains AE and VAE on all 3 MVTec categories (6
training runs total), then evaluates every trained checkpoint through the
shared evaluation harness. This is just a convenience runner around
scripts/train.py and scripts/evaluate.py -- it documents exactly how the
Phase 2 results were produced, so anyone (including us later) can reproduce
them with one command.

Usage:
    python scripts/run_phase2.py
    python scripts/run_phase2.py --epochs 100
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
MODELS = ["ae", "vae"]
CATEGORIES = ["bottle", "capsule", "pill"]


def run(command: list[str]) -> None:
    print(f"\n$ {' '.join(command)}")
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    start_time = time.time()

    print("=== Training ===")
    for model in MODELS:
        for category in CATEGORIES:
            run([
                sys.executable, str(SCRIPTS_DIR / "train.py"),
                "--model", model,
                "--category", category,
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
            ])

    print("\n=== Evaluating ===")
    for model in MODELS:
        for category in CATEGORIES:
            run([
                sys.executable, str(SCRIPTS_DIR / "evaluate.py"),
                "--model", model,
                "--category", category,
            ])

    elapsed_minutes = (time.time() - start_time) / 60
    print(f"\n[done] Phase 2 finished in {elapsed_minutes:.1f} minutes. See results/summary.csv")


if __name__ == "__main__":
    main()
