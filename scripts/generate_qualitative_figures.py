"""Generates the qualitative "figure set" for the paper: for each category,
one defective test image run through AE, VAE, and DDIM, showing the input,
each model's reconstruction, its anomaly heatmap, and the ground-truth
defect mask side by side. This is what lets a reader SEE the difference
between the three models, not just compare numbers in a table.

Usage:
    python scripts/generate_qualitative_figures.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from src.data.mvtec_dataset import MVTecTestDataset  # noqa: E402
from evaluate import build_transform as build_ae_vae_transform, load_model  # noqa: E402
from evaluate_diffusion import build_transform as build_ddim_transform, load_checkpoint  # noqa: E402

CATEGORIES = ["bottle", "capsule", "pill"]
DDIM_T_START = 250
DDIM_STEPS = 25


def to_displayable_image(image_tensor: torch.Tensor):
    """Our models all work on images scaled to [-1, 1]. matplotlib's imshow
    expects [0, 1] (or [0, 255]) and a (height, width, channels) shaped
    array instead of PyTorch's (channels, height, width), so this undoes
    both of those before we can plot anything."""
    image_0_to_1 = (image_tensor.clamp(-1, 1) + 1) / 2
    return image_0_to_1.permute(1, 2, 0).cpu().numpy()


def find_defective_sample(test_dataset: MVTecTestDataset) -> dict:
    """Picks the first defective (non-'good') test image, for a consistent,
    reproducible choice every time this script runs."""
    for index, (_path, _mask_path, label, _defect_type) in enumerate(test_dataset.samples):
        if label == 1:
            return test_dataset[index]
    raise RuntimeError("No defective test image found")


def build_row(ax_row, row_label, input_image, reconstruction, anomaly_map, mask) -> None:
    """Fills in one row of the figure: [input | reconstruction | anomaly map | ground truth]."""
    ax_row[0].imshow(to_displayable_image(input_image))
    ax_row[0].set_ylabel(row_label, fontsize=12, fontweight="bold")

    ax_row[1].imshow(to_displayable_image(reconstruction))

    # Each model's anomaly scores live on a different scale, so we normalise
    # each map to its OWN min/max just for display -- this makes the DEFECT
    # LOCATION visible for every model, but the colour intensity is not
    # comparable between rows (that comparison already lives in the AUROC
    # numbers, not in this figure).
    anomaly_map_np = anomaly_map.squeeze(0).cpu().numpy()
    ax_row[2].imshow(anomaly_map_np, cmap="inferno")

    ax_row[3].imshow(mask.squeeze(0).cpu().numpy(), cmap="gray")

    for ax in ax_row:
        ax.set_xticks([])
        ax.set_yticks([])


def generate_figure_for_category(category: str, device: str, figures_dir: Path) -> None:
    print(f"[category] {category}")
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    column_titles = ["Input", "Reconstruction", "Anomaly map", "Ground truth"]
    for ax, title in zip(axes[0], column_titles):
        ax.set_title(title)

    # --- AE ---
    ae_model, ae_image_size = load_model(REPO_ROOT / "checkpoints" / f"ae_{category}" / "model.pt", device)
    ae_transform = build_ae_vae_transform(ae_image_size)
    ae_test_dataset = MVTecTestDataset(REPO_ROOT / "data", category, image_size=ae_image_size, transform=ae_transform)
    sample = find_defective_sample(ae_test_dataset)
    print(f"  using defect_type={sample['defect_type']}")
    image = sample["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        anomaly_map, reconstruction = ae_model.anomaly_map(image)
    build_row(axes[0], "AE", sample["image"], reconstruction.squeeze(0), anomaly_map, sample["mask"])

    # --- VAE ---
    vae_model, vae_image_size = load_model(REPO_ROOT / "checkpoints" / f"vae_{category}" / "model.pt", device)
    vae_transform = build_ae_vae_transform(vae_image_size)
    vae_test_dataset = MVTecTestDataset(REPO_ROOT / "data", category, image_size=vae_image_size, transform=vae_transform)
    sample = find_defective_sample(vae_test_dataset)
    image = sample["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        anomaly_map, reconstruction = vae_model.anomaly_map(image)
    build_row(axes[1], "VAE", sample["image"], reconstruction.squeeze(0), anomaly_map, sample["mask"])

    # --- DDIM ---
    ddim_model, diffusion, ddim_image_size = load_checkpoint(REPO_ROOT / "checkpoints" / f"ddim_{category}" / "model.pt", device)
    ddim_transform = build_ddim_transform(ddim_image_size)
    ddim_test_dataset = MVTecTestDataset(REPO_ROOT / "data", category, image_size=ddim_image_size, transform=ddim_transform)
    sample = find_defective_sample(ddim_test_dataset)
    image = sample["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        anomaly_map, reconstruction = diffusion.anomaly_map(ddim_model, image, t_start=DDIM_T_START, ddim_steps=DDIM_STEPS)
    build_row(axes[2], "DDIM", sample["image"], reconstruction.squeeze(0), anomaly_map, sample["mask"])

    fig.suptitle(f"{category.capitalize()}: input / reconstruction / anomaly map / ground truth", fontsize=13)
    fig.tight_layout()
    out_path = figures_dir / f"qualitative_{category}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {out_path}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    figures_dir = REPO_ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)

    for category in CATEGORIES:
        generate_figure_for_category(category, device, figures_dir)

    print(f"\n[done] all qualitative figures saved to {figures_dir}")


if __name__ == "__main__":
    main()
