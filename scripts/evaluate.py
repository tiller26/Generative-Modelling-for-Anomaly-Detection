"""Evaluate a trained AE or VAE checkpoint on one MVTec AD category, using
the shared evaluation harness (src/eval/metrics.py): image-level AUROC,
pixel-level AUROC, IoU, and per-image latency.

Usage:
    python scripts/evaluate.py --model ae --category pill
    python scripts/evaluate.py --model vae --category bottle --checkpoint checkpoints/vae_bottle/model.pt

Results are printed to the console and appended as one row to
results/summary.csv, so every model/category combination ends up in a single
table we can compare at the end.
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
from src.models.autoencoder import Autoencoder  # noqa: E402
from src.models.vae import VAE  # noqa: E402


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def load_model(checkpoint_path: Path, device: str) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if checkpoint["model_name"] == "ae":
        model = Autoencoder(latent_dim=checkpoint["latent_dim"], image_size=checkpoint["image_size"])
    elif checkpoint["model_name"] == "vae":
        model = VAE(
            latent_dim=checkpoint["latent_dim"],
            image_size=checkpoint["image_size"],
            kl_weight=checkpoint["kl_weight"],
        )
    else:
        raise ValueError(f"Unknown model_name in checkpoint: {checkpoint['model_name']}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint["image_size"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=["ae", "vae"])
    parser.add_argument("--category", required=True, choices=["bottle", "capsule", "pill"])
    parser.add_argument("--checkpoint", default=None, help="Defaults to checkpoints/<model>_<category>/model.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else REPO_ROOT / "checkpoints" / f"{args.model}_{args.category}" / "model.pt"

    model, image_size = load_model(checkpoint_path, device)
    print(f"[setup] device={device} model={args.model} category={args.category} image_size={image_size}")

    transform = build_transform(image_size)
    test_dataset = MVTecTestDataset(REPO_ROOT / "data", args.category, image_size=image_size, transform=transform)
    print(f"[data] {len(test_dataset)} test images")

    # Run every test image through the model once, collecting the anomaly
    # map, the ground-truth mask, and the image-level label (0=good, 1=defect).
    #
    # We try TWO ways of turning a per-pixel anomaly map into one image-level
    # score: the mean pixel value, and the max pixel value. Defects are small,
    # localized patches on a mostly-normal-looking part, so averaging over the
    # whole image can drown out a small-but-severe defect in a sea of
    # near-zero background error. Taking the single worst pixel (max) is
    # usually more sensitive for this reason -- we report both so the
    # difference is visible rather than hidden behind one hard-coded choice.
    all_anomaly_maps = []
    all_masks = []
    all_mean_scores = []
    all_max_scores = []
    all_labels = []

    with torch.no_grad():
        for sample in test_dataset:
            image = sample["image"].unsqueeze(0).to(device)
            anomaly_map, _reconstruction = model.anomaly_map(image)

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

    # Latency: time a single-image forward pass (encode + decode), which is
    # what matters for a real-time inspection line -- not batched throughput.
    single_image = test_dataset[0]["image"].unsqueeze(0).to(device)
    sync_fn = torch.cuda.synchronize if device == "cuda" else None
    latency = measure_latency(lambda x: model.anomaly_map(x), single_image, warmup=5, n_runs=50, sync_fn=sync_fn)

    print(f"[results] image_auroc(mean/max)={img_auroc_mean:.4f}/{img_auroc_max:.4f} pixel_auroc={pix_auroc:.4f} "
          f"iou={iou_result['iou']:.4f} latency_ms={latency.mean_ms:.2f} (+/- {latency.std_ms:.2f})")

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
            args.model, args.category, img_auroc_mean, img_auroc_max, pix_auroc,
            iou_result["iou"], latency.mean_ms, latency.p95_ms, time.time(),
        ])

    print(f"[done] appended results to {summary_path}")


if __name__ == "__main__":
    main()
