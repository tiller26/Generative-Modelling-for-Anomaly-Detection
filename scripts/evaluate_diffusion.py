"""Evaluate a trained diffusion (DDIM) checkpoint on one MVTec AD category,
using the shared evaluation harness. Mirrors scripts/evaluate.py (used for
AE/VAE), but the diffusion model needs two extra settings that AE/VAE don't:

  --t-start:    how much noise to add before reconstructing (see
                src/models/diffusion.py's module docstring for why this
                matters -- too little and defects survive untouched, too
                much and even normal images stop reconstructing well).
  --ddim-steps: how many denoising steps to take on the way back. This is
                DDIM's key selling point: fewer steps = faster, and Phase 3
                is specifically about measuring that speed/quality trade-off
                (see scripts/sweep_ddim_steps.py).

Usage:
    python scripts/evaluate_diffusion.py --category bottle
    python scripts/evaluate_diffusion.py --category bottle --t-start 250 --ddim-steps 25
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.mvtec_dataset import MVTecTestDataset  # noqa: E402
from src.eval.metrics import best_threshold_iou, image_auroc, measure_latency, pixel_auroc  # noqa: E402
from src.models.diffusion import GaussianDiffusion  # noqa: E402
from src.models.unet import UNet  # noqa: E402


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def load_checkpoint(checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = UNet(in_channels=3, base_channels=checkpoint["base_channels"], channel_mults=(1, 2, 4))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    diffusion = GaussianDiffusion(timesteps=checkpoint["timesteps"]).to(device)

    return model, diffusion, checkpoint["image_size"]


def evaluate_config(model, diffusion, test_dataset, device, t_start: int, ddim_steps: int) -> dict:
    """Runs every test image through the model once at the given (t_start,
    ddim_steps) setting, and returns all the metrics we care about. Pulled
    out into its own function so scripts/sweep_ddim_steps.py can call this
    many times for the SAME loaded model without reloading it from disk or
    re-reading images from the dataset each time -- reloading would make a
    5-value sweep take 5x longer for no benefit."""

    def anomaly_map_fn(x: torch.Tensor):
        return diffusion.anomaly_map(model, x, t_start=t_start, ddim_steps=ddim_steps)

    all_anomaly_maps = []
    all_masks = []
    all_mean_scores = []
    all_max_scores = []
    all_labels = []

    for sample in test_dataset:
        image = sample["image"].unsqueeze(0).to(device)
        anomaly_map, _reconstruction = anomaly_map_fn(image)

        anomaly_map = anomaly_map.squeeze(0).cpu().numpy()
        all_anomaly_maps.append(anomaly_map)
        all_masks.append(sample["mask"].squeeze(0).numpy())
        all_mean_scores.append(float(anomaly_map.mean()))
        all_max_scores.append(float(anomaly_map.max()))
        all_labels.append(sample["label"])

    anomaly_maps = np.stack(all_anomaly_maps)
    masks = np.stack(all_masks)
    mean_scores = np.array(all_mean_scores)
    max_scores = np.array(all_max_scores)
    labels = np.array(all_labels)

    img_auroc_mean = image_auroc(mean_scores, labels)
    img_auroc_max = image_auroc(max_scores, labels)
    pix_auroc = pixel_auroc(anomaly_maps, masks)
    iou_result = best_threshold_iou(anomaly_maps, masks)

    single_image = test_dataset[0]["image"].unsqueeze(0).to(device)
    sync_fn = torch.cuda.synchronize if device == "cuda" else None
    latency = measure_latency(anomaly_map_fn, single_image, warmup=3, n_runs=20, sync_fn=sync_fn)

    return {
        "image_auroc_mean": img_auroc_mean,
        "image_auroc_max": img_auroc_max,
        "pixel_auroc": pix_auroc,
        "iou": iou_result["iou"],
        "latency_ms_mean": latency.mean_ms,
        "latency_ms_p95": latency.p95_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", required=True, choices=["bottle", "capsule", "pill"])
    parser.add_argument("--checkpoint", default=None, help="Defaults to checkpoints/ddim_<category>/model.pt")
    parser.add_argument("--t-start", type=int, default=250)
    parser.add_argument("--ddim-steps", type=int, default=25)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else REPO_ROOT / "checkpoints" / f"ddim_{args.category}" / "model.pt"

    model, diffusion, image_size = load_checkpoint(checkpoint_path, device)
    print(f"[setup] device={device} category={args.category} image_size={image_size} "
          f"t_start={args.t_start} ddim_steps={args.ddim_steps}")

    transform = build_transform(image_size)
    test_dataset = MVTecTestDataset(REPO_ROOT / "data", args.category, image_size=image_size, transform=transform)
    print(f"[data] {len(test_dataset)} test images")

    with torch.no_grad():
        results = evaluate_config(model, diffusion, test_dataset, device, args.t_start, args.ddim_steps)

    print(f"[results] image_auroc(mean/max)={results['image_auroc_mean']:.4f}/{results['image_auroc_max']:.4f} "
          f"pixel_auroc={results['pixel_auroc']:.4f} iou={results['iou']:.4f} "
          f"latency_ms={results['latency_ms_mean']:.2f}")

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    summary_path = results_dir / "summary.csv"
    write_header = not summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "model", "category", "image_auroc_mean", "image_auroc_max", "pixel_auroc",
                "iou", "latency_ms_mean", "latency_ms_p95", "timestamp",
            ])
        writer.writerow([
            "ddim", args.category, results["image_auroc_mean"], results["image_auroc_max"], results["pixel_auroc"],
            results["iou"], results["latency_ms_mean"], results["latency_ms_p95"], time.time(),
        ])

    print(f"[done] appended results to {summary_path}")


if __name__ == "__main__":
    main()
