"""Sweep the number of DDIM sampling steps and record how latency AND
detection quality change together. This is the whole point of DDIM: it lets
us trade reconstruction quality for speed by choosing fewer denoising steps,
and this script measures exactly where that trade-off starts to hurt.

Usage:
    python scripts/sweep_ddim_steps.py --category bottle
    python scripts/sweep_ddim_steps.py --category bottle --steps 5 10 25 50 100
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from src.data.mvtec_dataset import MVTecTestDataset  # noqa: E402
from evaluate_diffusion import build_transform, evaluate_config, load_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", required=True, choices=["bottle", "capsule", "pill"])
    parser.add_argument("--checkpoint", default=None, help="Defaults to checkpoints/ddim_<category>/model.pt")
    parser.add_argument("--t-start", type=int, default=250)
    parser.add_argument("--steps", type=int, nargs="+", default=[5, 10, 25, 50, 100])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else REPO_ROOT / "checkpoints" / f"ddim_{args.category}" / "model.pt"

    # Load the model and test data ONCE, then reuse them for every step count
    # in the sweep -- reloading per step count would waste most of the time
    # on disk I/O instead of the actual comparison we care about.
    model, diffusion, image_size = load_checkpoint(checkpoint_path, device)
    transform = build_transform(image_size)
    test_dataset = MVTecTestDataset(REPO_ROOT / "data", args.category, image_size=image_size, transform=transform)
    print(f"[setup] device={device} category={args.category} t_start={args.t_start} step_counts={args.steps}")

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    sweep_path = results_dir / "ddim_steps_sweep.csv"
    write_header = not sweep_path.exists()

    with open(sweep_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "category", "t_start", "ddim_steps", "image_auroc_max",
                "pixel_auroc", "iou", "latency_ms_mean", "latency_ms_p95", "timestamp",
            ])

        for ddim_steps in args.steps:
            with torch.no_grad():
                results = evaluate_config(model, diffusion, test_dataset, device, args.t_start, ddim_steps)

            print(f"[ddim_steps={ddim_steps:>3}] image_auroc_max={results['image_auroc_max']:.4f} "
                  f"pixel_auroc={results['pixel_auroc']:.4f} iou={results['iou']:.4f} "
                  f"latency_ms={results['latency_ms_mean']:.2f}")

            writer.writerow([
                args.category, args.t_start, ddim_steps, results["image_auroc_max"],
                results["pixel_auroc"], results["iou"], results["latency_ms_mean"], results["latency_ms_p95"], time.time(),
            ])
            f.flush()

    print(f"\n[done] appended results to {sweep_path}")


if __name__ == "__main__":
    main()
