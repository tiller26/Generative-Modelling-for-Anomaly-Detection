"""End-to-end smoke test for the cloned third_party/AnoDDPM reference implementation.

Confirms the DDPM (UNet + Gaussian diffusion, simplex noise) forward/backward
pass runs on our actual MVTec AD data on GPU, without needing their CLI/arg-file
workflow (which is hardwired to a private MRI dataset). This is NOT a real
training run -- it's a handful of iterations at reduced resolution/width just
to prove the adapted pipeline is wired correctly end to end.

Usage:
    python scripts/smoke_test_anoddpm.py --category pill --steps 5
"""
import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "third_party" / "AnoDDPM"))

from dataset import MVTec  # noqa: E402  (third_party import, path patched above)
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule  # noqa: E402
from UNet import UNetModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="pill", choices=["bottle", "capsule", "pill"])
    parser.add_argument("--img-size", type=int, default=64, help="Small size to keep the smoke test fast")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Upstream AnoDDPM's simplex noise path is written for batch size 1 "
             "(every example config in third_party/AnoDDPM/test_args uses Batch_Size=1)",
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--noise", default="simplex", choices=["gauss", "simplex"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device} category={args.category} img_size={args.img_size}")

    data_dir = REPO_ROOT / "data" / args.category
    dataset = MVTec(str(data_dir), anomalous=False, img_size=(args.img_size, args.img_size), rgb=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"[data] {len(dataset)} training images found under {data_dir}/train/good")

    model = UNetModel(
        args.img_size,
        args.base_channels,
        channel_mults="",
        dropout=0,
        n_heads=2,
        n_head_channels=-1,
        in_channels=3,
    ).to(device)

    betas = get_beta_schedule(args.timesteps, "cosine")
    diffusion = GaussianDiffusionModel(
        (args.img_size, args.img_size), betas, loss_weight="none", loss_type="l2",
        noise=args.noise, img_channels=3,
    )

    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)

    diffusion_args = {
        "img_size": (args.img_size, args.img_size),
        "train_start": True,
        "sample_distance": args.timesteps,
    }
    data_iter = iter(loader)
    losses = []
    start = time.time()
    for step in range(args.steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        x = batch["image"].to(device)
        loss, _estimates = diffusion.p_loss(model, x, diffusion_args)

        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimiser.step()

        loss_val = loss.item()
        losses.append(loss_val)
        assert torch.isfinite(loss).all(), "Loss diverged to NaN/Inf"
        print(f"[step {step + 1}/{args.steps}] loss={loss_val:.4f}")

    elapsed = time.time() - start
    print(f"\n[done] {args.steps} steps in {elapsed:.1f}s ({elapsed / args.steps * 1000:.0f} ms/step) "
          f"on {device}. Losses: {[round(loss_val, 4) for loss_val in losses]}")
    print("[result] AnoDDPM UNet + GaussianDiffusion pipeline runs end-to-end on real MVTec data. OK")


if __name__ == "__main__":
    main()
