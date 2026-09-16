"""Train the from-scratch UNet + GaussianDiffusion (DDPM/DDIM) model on ONE
MVTec AD category, and save a checkpoint. Mirrors scripts/train.py (used for
the AE/VAE) so all three model families are trained the same way, on the
same data, for a fair comparison.

Usage:
    python scripts/train_diffusion.py --category bottle --epochs 300
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.mvtec_dataset import MVTecTrainDataset, train_val_split  # noqa: E402
from src.models.diffusion import GaussianDiffusion  # noqa: E402
from src.models.unet import UNet  # noqa: E402
from src.utils.tracking import ExperimentLogger  # noqa: E402


def build_transform(image_size: int) -> transforms.Compose:
    """Same [-1, 1] scaling as the AE/VAE training script -- the diffusion
    math (in particular the fixed noise schedule) assumes this range."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def run_one_epoch(model, diffusion, loader, optimiser=None) -> float:
    """One full pass over `loader`. Training when `optimiser` is given,
    validation (loss only, no weight updates) when it's None."""
    is_training = optimiser is not None
    model.train(is_training)

    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = batch.to(next(model.parameters()).device)

        if is_training:
            loss = diffusion.training_loss(model, batch)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        else:
            with torch.no_grad():
                loss = diffusion.training_loss(model, batch)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", required=True, choices=["bottle", "capsule", "pill"])
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Kept smaller than the AE/VAE's batch size (32) on purpose: the UNet's "
             "skip connections and attention keep many activations in memory at once, "
             "so batch=32 at base_channels=64/image_size=128 needs ~11.4GB -- more than "
             "an 8GB GPU has. That doesn't crash on Windows, it silently spills into slow "
             "shared system memory instead, which measured ~280x slower per step. "
             "batch=8 measured 3GB peak and was also faster per-epoch than batch=16.",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Matches the AE/VAE epoch budget (scripts/train.py) so all three model "
             "families get the same amount of training for a fair comparison.",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = f"ddim_{args.category}"
    print(f"[setup] device={device} run={run_name} image_size={args.image_size} base_channels={args.base_channels}")

    torch.manual_seed(args.seed)

    transform = build_transform(args.image_size)
    full_train_dataset = MVTecTrainDataset(
        REPO_ROOT / "data", args.category, image_size=args.image_size, transform=transform
    )
    train_subset, val_subset = train_val_split(full_train_dataset, val_fraction=args.val_fraction, seed=args.seed)
    print(f"[data] {len(train_subset)} train images, {len(val_subset)} val images")

    # num_workers>0: reading + decoding each PNG on the main thread was the
    # bottleneck for AE/VAE training (measured ~17x slower than GPU compute),
    # so we load images in parallel background processes here too.
    loader_kwargs = dict(num_workers=4, persistent_workers=True)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = UNet(in_channels=3, base_channels=args.base_channels, channel_mults=(1, 2, 4)).to(device)
    diffusion = GaussianDiffusion(timesteps=args.timesteps).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] UNet params: {n_params:,}")

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    logger = ExperimentLogger(run_name=run_name)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = run_one_epoch(model, diffusion, train_loader, optimiser)
        val_loss = run_one_epoch(model, diffusion, val_loader, optimiser=None)

        logger.log_scalar("train/loss", train_loss, step=epoch)
        logger.log_scalar("val/loss", val_loss, step=epoch)

        if epoch % 10 == 0 or epoch == args.epochs:
            elapsed = time.time() - start_time
            print(f"[epoch {epoch}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} ({elapsed:.0f}s elapsed)")

    checkpoint_dir = REPO_ROOT / "checkpoints" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "category": args.category,
            "image_size": args.image_size,
            "base_channels": args.base_channels,
            "timesteps": args.timesteps,
        },
        checkpoint_path,
    )
    print(f"[done] saved checkpoint to {checkpoint_path}")
    logger.close()


if __name__ == "__main__":
    main()
