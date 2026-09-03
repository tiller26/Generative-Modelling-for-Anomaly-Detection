"""End-to-end smoke test for our from-scratch UNet + GaussianDiffusion (DDIM).

Confirms: (1) training loss backprops and decreases over a few steps on real
MVTec data, and (2) the DDIM reconstruction / anomaly-map pass runs on both a
normal and a defective test image with correct output shapes. This is a
pipeline sanity check, not a real training run -- reduced resolution/width
and a handful of steps only.

Usage:
    python scripts/smoke_test_ddim.py --category pill --steps 20
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.mvtec_dataset import MVTecTestDataset, MVTecTrainDataset  # noqa: E402
from src.models.diffusion import GaussianDiffusion  # noqa: E402
from src.models.unet import UNet  # noqa: E402


def diffusion_transform(image_size: int) -> transforms.Compose:
    """Diffusion models are conventionally trained on [-1, 1]-scaled inputs
    (matches the tanh-like dynamic range noise is added/removed around),
    unlike the ImageNet-normalized default used for discriminative models."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="pill", choices=["bottle", "capsule", "pill"])
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--ddim-t-start", type=int, default=250, help="Partial-noise starting timestep for reconstruction")
    parser.add_argument("--ddim-steps", type=int, default=25)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device} category={args.category} img_size={args.img_size}")

    transform = diffusion_transform(args.img_size)
    data_dir = REPO_ROOT / "data"
    train_ds = MVTecTrainDataset(data_dir, args.category, image_size=args.img_size, transform=transform)
    test_ds = MVTecTestDataset(data_dir, args.category, image_size=args.img_size, transform=transform)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"[data] {len(train_ds)} train images, {len(test_ds)} test images")

    model = UNet(in_channels=3, base_channels=args.base_channels, channel_mults=(1, 2, 4)).to(device)
    diffusion = GaussianDiffusion(timesteps=args.timesteps).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-4)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] UNet params: {n_params:,}")

    data_iter = iter(loader)
    losses = []
    start = time.time()
    for step in range(args.steps):
        try:
            x0 = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x0 = next(data_iter)
        x0 = x0.to(device)

        loss = diffusion.training_loss(model, x0)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        assert torch.isfinite(loss), "Loss diverged to NaN/Inf"
        losses.append(loss.item())
        print(f"[train step {step + 1}/{args.steps}] loss={loss.item():.4f}")

    elapsed = time.time() - start
    print(f"[done training] {args.steps} steps in {elapsed:.1f}s ({elapsed / args.steps * 1000:.0f} ms/step)")

    # Reconstruction / anomaly-map smoke check on one normal + one defective test image.
    model.eval()
    good_idx = next(i for i, s in enumerate(test_ds.samples) if s[2] == 0)
    defect_idx = next(i for i, s in enumerate(test_ds.samples) if s[2] == 1)

    for label, idx in [("good", good_idx), ("defect", defect_idx)]:
        sample = test_ds[idx]
        x0 = sample["image"].unsqueeze(0).to(device)
        anomaly_score_map, reconstruction = diffusion.anomaly_map(
            model, x0, t_start=args.ddim_t_start, ddim_steps=args.ddim_steps
        )
        assert anomaly_score_map.shape == (1, args.img_size, args.img_size)
        assert reconstruction.shape == x0.shape
        assert torch.isfinite(reconstruction).all()
        print(
            f"[reconstruct {label}] defect_type={sample['defect_type']} "
            f"anomaly_score(mean/max)={anomaly_score_map.mean().item():.4f}/{anomaly_score_map.max().item():.4f}"
        )

    print("\n[result] From-scratch UNet + GaussianDiffusion (DDIM) runs end-to-end on real MVTec data. OK")


if __name__ == "__main__":
    main()
