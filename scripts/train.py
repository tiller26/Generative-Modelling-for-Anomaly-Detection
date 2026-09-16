"""Train an AE or VAE on ONE MVTec AD category, and save a checkpoint.

We train one separate model per category (an AE trained on bottles has no
reason to also understand capsules -- they look nothing alike), which is
standard practice for this benchmark and matches how the DDIM model was
smoke-tested earlier.

Usage:
    python scripts/train.py --model ae --category pill --epochs 150
    python scripts/train.py --model vae --category bottle --epochs 150
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
from src.models.autoencoder import Autoencoder  # noqa: E402
from src.models.vae import VAE  # noqa: E402
from src.utils.tracking import ExperimentLogger  # noqa: E402


def build_transform(image_size: int) -> transforms.Compose:
    """AE/VAE reconstruct images, so like the diffusion model we scale
    pixel values to [-1, 1] (matches the Tanh at the end of our Decoder)
    rather than the ImageNet-style normalization used for classifiers."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def build_model(model_name: str, latent_dim: int, image_size: int, kl_weight: float) -> torch.nn.Module:
    if model_name == "ae":
        return Autoencoder(latent_dim=latent_dim, image_size=image_size)
    if model_name == "vae":
        return VAE(latent_dim=latent_dim, image_size=image_size, kl_weight=kl_weight)
    raise ValueError(f"Unknown model '{model_name}', expected 'ae' or 'vae'")


def run_one_epoch(model, loader, optimiser=None) -> float:
    """Runs one full pass over `loader`. If `optimiser` is given we are
    training (update weights); if it's None we are just measuring loss
    (validation), so we skip backward()/step() and gradient tracking."""
    is_training = optimiser is not None
    model.train(is_training)

    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = batch.to(next(model.parameters()).device)

        if is_training:
            loss = model.training_loss(batch)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        else:
            with torch.no_grad():
                loss = model.training_loss(batch)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=["ae", "vae"])
    parser.add_argument("--category", required=True, choices=["bottle", "capsule", "pill"])
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--kl-weight", type=float, default=0.001, help="Only used when --model vae")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = f"{args.model}_{args.category}"
    print(f"[setup] device={device} run={run_name} image_size={args.image_size} latent_dim={args.latent_dim}")

    torch.manual_seed(args.seed)

    transform = build_transform(args.image_size)
    full_train_dataset = MVTecTrainDataset(
        REPO_ROOT / "data", args.category, image_size=args.image_size, transform=transform
    )
    train_subset, val_subset = train_val_split(full_train_dataset, val_fraction=args.val_fraction, seed=args.seed)
    print(f"[data] {len(train_subset)} train images, {len(val_subset)} val images")

    # num_workers>0 loads images in parallel background processes instead of
    # one at a time on the main thread. Reading + decoding each PNG is much
    # slower than a single training step on the GPU, so without this the GPU
    # would sit idle waiting for images most of the time.
    loader_kwargs = dict(num_workers=4, persistent_workers=True)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = build_model(args.model, args.latent_dim, args.image_size, args.kl_weight).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.model.upper()} params: {n_params:,}")

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    logger = ExperimentLogger(run_name=run_name)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = run_one_epoch(model, train_loader, optimiser)
        val_loss = run_one_epoch(model, val_loader, optimiser=None)

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
            "model_name": args.model,
            "category": args.category,
            "image_size": args.image_size,
            "latent_dim": args.latent_dim,
            "kl_weight": args.kl_weight,
        },
        checkpoint_path,
    )
    print(f"[done] saved checkpoint to {checkpoint_path}")
    logger.close()


if __name__ == "__main__":
    main()
