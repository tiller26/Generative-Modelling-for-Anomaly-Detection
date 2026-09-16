"""One-off fix-up script: re-measures ONLY the latency numbers for every DDIM
row in results/summary.csv and results/ddim_steps_sweep.csv, using the fixed
measure_latency (see src/eval/metrics.py -- it now warms the GPU up for a
minimum real amount of time, not just a fixed iteration count).

Why this script exists instead of just re-running the sweep: the AUROC /
pixel AUROC / IoU numbers in those files do not depend on wall-clock timing
at all, so they were correct already -- only the latency columns were
corrupted by the warmup bug. Recomputing AUROC over the full test set is by
far the most expensive part of the original sweep, so re-measuring latency
only (and leaving the already-correct quality numbers alone) finishes in
minutes instead of hours.

Usage:
    python scripts/remeasure_ddim_latency.py
"""
import csv
import sys
from pathlib import Path

import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from src.data.mvtec_dataset import MVTecTestDataset  # noqa: E402
from src.eval.metrics import measure_latency  # noqa: E402
from evaluate_diffusion import build_transform, load_checkpoint  # noqa: E402

CATEGORIES = ["bottle", "capsule", "pill"]


def measure_one(model, diffusion, test_dataset, device, t_start: int, ddim_steps: int) -> tuple[float, float]:
    def anomaly_map_fn(x: torch.Tensor):
        return diffusion.anomaly_map(model, x, t_start=t_start, ddim_steps=ddim_steps)

    single_image = test_dataset[0]["image"].unsqueeze(0).to(device)
    sync_fn = torch.cuda.synchronize if device == "cuda" else None
    with torch.no_grad():
        result = measure_latency(anomaly_map_fn, single_image, warmup=3, n_runs=20, sync_fn=sync_fn)
    return result.mean_ms, result.p95_ms


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load each category's model + test set once, reuse for every latency
    # measurement that category needs (both the summary.csv row and every
    # step count in the sweep).
    loaded = {}
    for category in CATEGORIES:
        checkpoint_path = REPO_ROOT / "checkpoints" / f"ddim_{category}" / "model.pt"
        model, diffusion, image_size = load_checkpoint(checkpoint_path, device)
        transform = build_transform(image_size)
        test_dataset = MVTecTestDataset(REPO_ROOT / "data", category, image_size=image_size, transform=transform)
        loaded[category] = (model, diffusion, test_dataset)
        print(f"[loaded] {category}: {len(test_dataset)} test images")

    # --- Fix results/summary.csv (one row per category, ddim_steps=25) ---
    summary_path = REPO_ROOT / "results" / "summary.csv"
    with open(summary_path, newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        if row["model"] != "ddim":
            continue
        category = row["category"]
        model, diffusion, test_dataset = loaded[category]
        print(f"[summary.csv] re-measuring latency for ddim/{category} (25 steps)...")
        mean_ms, p95_ms = measure_one(model, diffusion, test_dataset, device, t_start=250, ddim_steps=25)
        print(f"  old latency_ms_mean={row['latency_ms_mean']} -> new={mean_ms:.2f}")
        row["latency_ms_mean"] = mean_ms
        row["latency_ms_p95"] = p95_ms

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # --- Fix results/ddim_steps_sweep.csv (one row per category x ddim_steps) ---
    sweep_path = REPO_ROOT / "results" / "ddim_steps_sweep.csv"
    with open(sweep_path, newline="") as f:
        sweep_rows = list(csv.DictReader(f))

    for row in sweep_rows:
        category = row["category"]
        ddim_steps = int(row["ddim_steps"])
        t_start = int(row["t_start"])
        model, diffusion, test_dataset = loaded[category]
        print(f"[sweep] re-measuring latency for {category}/{ddim_steps} steps...")
        mean_ms, p95_ms = measure_one(model, diffusion, test_dataset, device, t_start=t_start, ddim_steps=ddim_steps)
        print(f"  old latency_ms_mean={row['latency_ms_mean']} -> new={mean_ms:.2f}")
        row["latency_ms_mean"] = mean_ms
        row["latency_ms_p95"] = p95_ms

    with open(sweep_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sweep_rows[0].keys())
        writer.writeheader()
        writer.writerows(sweep_rows)

    print("\n[done] latency columns re-measured and rewritten in both CSV files.")


if __name__ == "__main__":
    main()
