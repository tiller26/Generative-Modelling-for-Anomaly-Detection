"""A plain convolutional Variational Autoencoder (VAE) for reconstruction-based
anomaly detection. Original implementation, deliberately close in structure
to src/models/autoencoder.py so the two are easy to compare.

How this differs from the plain Autoencoder:
  A normal Autoencoder's encoder outputs ONE fixed latent vector per image.
  A VAE's encoder instead outputs the parameters of a small probability
  distribution -- a mean ("mu") and a spread ("logvar", log of the variance)
  -- and then we DRAW a random latent vector from that distribution before
  decoding. This forces the latent space to be smooth and well-organised
  (nearby points decode to similar-looking images), because the network has
  to work even when noise is added around each encoded point. In principle
  this can give more reliable anomaly scores than a plain AE, at the cost of
  slightly blurrier reconstructions.

  Training adds a second loss term, the "KL divergence", which penalises the
  encoder for drifting too far from a standard normal distribution (mean 0,
  variance 1). Without it, the encoder could cheat by making the spread tiny
  and just behaving like a plain deterministic autoencoder.
"""
from __future__ import annotations

import torch
from torch import nn

from src.models.autoencoder import Decoder


class VAEEncoder(nn.Module):
    """Same convolutional trunk as the plain Autoencoder's Encoder, but the
    final layer produces TWO vectors (mu and logvar) instead of one."""

    def __init__(self, latent_dim: int = 256, image_size: int = 128):
        super().__init__()

        self.feature_map_size = image_size // 16
        self.final_channels = 256

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, self.final_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        flattened_size = self.final_channels * self.feature_map_size * self.feature_map_size
        self.to_mu = nn.Linear(flattened_size, latent_dim)
        self.to_logvar = nn.Linear(flattened_size, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.conv_layers(x)
        features = features.flatten(start_dim=1)
        mu = self.to_mu(features)
        logvar = self.to_logvar(features)
        return mu, logvar


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Draws a random latent vector z from the distribution described by
    (mu, logvar), in a way that still allows gradients to flow back to mu
    and logvar. This is the "reparameterization trick":

        z = mu + std * epsilon

    where epsilon is plain random noise (no gradient needed) and std comes
    from mu/logvar (which DO need gradients). Because the randomness (epsilon)
    is separated out from the learned parameters (mu, std), backpropagation
    works normally through mu and std.
    """
    std = torch.exp(0.5 * logvar)
    epsilon = torch.randn_like(std)
    return mu + std * epsilon


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """How far the learned distribution (mu, logvar) is from a standard
    normal distribution (mean 0, variance 1), averaged over the batch. This
    is the closed-form KL divergence formula for two Gaussians, used because
    it turns a hard integral into simple arithmetic."""
    per_sample_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return per_sample_kl.mean()


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 256, image_size: int = 128, kl_weight: float = 1e-6):
        super().__init__()
        self.encoder = VAEEncoder(latent_dim, image_size)
        self.decoder = Decoder(latent_dim, image_size)
        self.kl_weight = kl_weight

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar

    def training_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Total loss = reconstruction error + kl_weight * KL divergence.

        IMPORTANT -- loss scale: the reconstruction term is a MEAN over every
        pixel (so it is tiny, ~1e-3 once trained) while the KL term is a SUM
        over latent dimensions (hundreds of nats). They are on very different
        scales, so kl_weight must be tiny (default 1e-6) to balance them. Our
        first run used 1e-3, which made the KL penalty larger than the whole
        reconstruction loss: the encoder collapsed to the prior (0 of 256
        latent dims active), the decoder ignored the input, and anomaly
        detection got worse than a plain AE. See README "VAE posterior
        collapse" for the diagnosis and the kl_weight sweep that fixed it.

        kl_weight is mutable on purpose: scripts/train.py ramps it up from 0
        over the first epochs ("KL annealing") so the decoder learns to use
        the latent code before the KL penalty kicks in.
        """
        reconstruction, mu, logvar = self.forward(x)
        reconstruction_loss = torch.nn.functional.mse_loss(reconstruction, x)
        kl_loss = kl_divergence(mu, logvar)
        return reconstruction_loss + self.kl_weight * kl_loss

    @torch.no_grad()
    def anomaly_map(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Same idea as Autoencoder.anomaly_map: reconstruct, then measure
        the per-pixel absolute error. We use mu directly (skip the random
        sampling) so the anomaly score is repeatable rather than noisy."""
        mu, _logvar = self.encoder(x)
        reconstruction = self.decoder(mu)
        error = (x - reconstruction).abs().mean(dim=1)
        return error, reconstruction
