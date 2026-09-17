"""Turns results/summary.csv and results/ddim_steps_sweep.csv into the
comparison plots needed for Phase 4: bar charts comparing AE/VAE/DDIM on
AUROC/IoU/latency across all 3 categories, plus the DDIM step-count
trade-off plot. Saves PNGs into figures/.

Usage:
    python scripts/plot_results.py
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MODELS = ["ae", "vae", "ddim"]
MODEL_LABELS = {"ae": "AE", "vae": "VAE", "ddim": "DDIM"}
CATEGORIES = ["bottle", "capsule", "pill"]
MODEL_COLORS = {"ae": "#4C72B0", "vae": "#DD8452", "ddim": "#55A868"}


def read_csv_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def grouped_bar_chart(rows: list[dict], value_key: str, title: str, ylabel: str, out_path: Path, log_scale: bool = False) -> None:
    """One bar chart with categories on the x-axis and one coloured bar per
    model within each category group -- the standard way to compare 3 models
    across 3 categories side by side."""
    # Look up value_by_model_category[model][category] = float value, so we
    # can plot bars in a fixed, predictable order regardless of row order.
    value_by_model_category = {model: {} for model in MODELS}
    for row in rows:
        value_by_model_category[row["model"]][row["category"]] = float(row[value_key])

    x_positions = np.arange(len(CATEGORIES))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, model in enumerate(MODELS):
        values = [value_by_model_category[model][category] for category in CATEGORIES]
        offset = (i - 1) * bar_width  # centers the 3 bars around each x tick
        ax.bar(x_positions + offset, values, width=bar_width, label=MODEL_LABELS[model], color=MODEL_COLORS[model])

    ax.set_xticks(x_positions)
    ax.set_xticklabels([c.capitalize() for c in CATEGORIES])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    if log_scale:
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {out_path}")


def ddim_step_sweep_plot(sweep_rows: list[dict], out_path: Path) -> None:
    """Two side-by-side line plots (one per category group would be too
    cluttered) showing how image AUROC and latency change together as the
    number of DDIM sampling steps increases -- this is the core trade-off
    Phase 3/4 is about."""
    fig, (ax_auroc, ax_latency) = plt.subplots(1, 2, figsize=(11, 4.5))

    for category in CATEGORIES:
        category_rows = [row for row in sweep_rows if row["category"] == category]
        category_rows.sort(key=lambda row: int(row["ddim_steps"]))
        steps = [int(row["ddim_steps"]) for row in category_rows]
        auroc_values = [float(row["image_auroc_max"]) for row in category_rows]
        latency_values = [float(row["latency_ms_mean"]) for row in category_rows]

        ax_auroc.plot(steps, auroc_values, marker="o", label=category.capitalize())
        ax_latency.plot(steps, latency_values, marker="o", label=category.capitalize())

    ax_auroc.set_xlabel("DDIM sampling steps")
    ax_auroc.set_ylabel("Image AUROC (max-pooled)")
    ax_auroc.set_title("Detection quality vs. step count")
    ax_auroc.legend()

    ax_latency.set_xlabel("DDIM sampling steps")
    ax_latency.set_ylabel("Latency (ms/image)")
    ax_latency.set_title("Latency vs. step count")
    ax_latency.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {out_path}")


def main() -> None:
    figures_dir = REPO_ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)

    summary_rows = read_csv_rows(REPO_ROOT / "results" / "summary.csv")

    grouped_bar_chart(
        summary_rows, "image_auroc_max", "Image-level AUROC (max-pooled)", "AUROC",
        figures_dir / "image_auroc.png",
    )
    grouped_bar_chart(
        summary_rows, "pixel_auroc", "Pixel-level AUROC", "AUROC",
        figures_dir / "pixel_auroc.png",
    )
    grouped_bar_chart(
        summary_rows, "iou", "Segmentation IoU (best threshold)", "IoU",
        figures_dir / "iou.png",
    )
    grouped_bar_chart(
        summary_rows, "latency_ms_mean", "Inference latency per image", "Latency (ms, log scale)",
        figures_dir / "latency.png", log_scale=True,
    )

    sweep_rows = read_csv_rows(REPO_ROOT / "results" / "ddim_steps_sweep.csv")
    ddim_step_sweep_plot(sweep_rows, figures_dir / "ddim_step_sweep.png")

    print(f"\n[done] all figures saved to {figures_dir}")


if __name__ == "__main__":
    main()
