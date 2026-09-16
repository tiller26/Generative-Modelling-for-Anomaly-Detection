"""A plain convolutional Autoencoder (AE) for reconstruction-based anomaly
detection. Original implementation, written simply on purpose so every step
is easy to read and explain.

How an autoencoder detects anomalies:
  We only ever train this network on GOOD (non-defective) images. The
  "Encoder" squeezes an image down into a small vector (the "bottleneck" /
  "latent code"). The "Decoder" then has to rebuild the full image from just
  that small vector. Because the bottleneck is much smaller than the image,
  the network is forced to only remember the KIND of patterns it saw a lot
  during training -- i.e. what a normal pill/bottle/capsule looks like.

  At test time, we feed in a real image (good or defective) and look at the
  reconstruction. A normal image reconstructs well, because the network has
  seen thousands of images like it. A defect is something the network never
  learned to encode, so the decoder cannot rebuild it faithfully -- it just
  redraws that patch as if it were normal. The difference between the input
  and the reconstruction is therefore large exactly where the defect is.
"""
from __future__ import annotations

import torch
from torch import nn


class Encoder(nn.Module):
    """Squeezes a 3-channel image down to a single vector of size latent_dim.

    Each conv layer below has stride=2, which halves the width and height.
    Starting from a 128x128 image, four of these layers gets us to 8x8:
        128 -> 64 -> 32 -> 16 -> 8
    """

    def __init__(self, latent_dim: int = 256, image_size: int = 128):
        super().__init__()

        # After 4 stride-2 convolutions, the spatial size has been divided by 2 four times.
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
        self.to_latent = nn.Linear(flattened_size, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_layers(x)
        features = features.flatten(start_dim=1)  # (batch, channels*h*w)
        latent_code = self.to_latent(features)
        return latent_code


class Decoder(nn.Module):
    """Rebuilds a 3-channel image from a latent code. Mirrors the Encoder's
    layers in reverse, using transposed convolutions to upsample instead of
    plain convolutions to downsample."""

    def __init__(self, latent_dim: int = 256, image_size: int = 128):
        super().__init__()

        self.feature_map_size = image_size // 16
        self.start_channels = 256

        flattened_size = self.start_channels * self.feature_map_size * self.feature_map_size
        self.from_latent = nn.Linear(latent_dim, flattened_size)

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(self.start_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # squashes output to [-1, 1], matching our normalized input images
        )

    def forward(self, latent_code: torch.Tensor) -> torch.Tensor:
        features = self.from_latent(latent_code)
        features = features.reshape(-1, self.start_channels, self.feature_map_size, self.feature_map_size)
        reconstruction = self.deconv_layers(features)
        return reconstruction


class Autoencoder(nn.Module):
    """Encoder + Decoder wired together, plus a small helper for turning a
    reconstruction into an anomaly map (used at test time, not training time).
    """

    def __init__(self, latent_dim: int = 256, image_size: int = 128):
        super().__init__()
        self.encoder = Encoder(latent_dim, image_size)
        self.decoder = Decoder(latent_dim, image_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent_code = self.encoder(x)
        reconstruction = self.decoder(latent_code)
        return reconstruction

    def training_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Plain reconstruction loss: how far off is the rebuilt image from
        the real one, pixel by pixel (mean squared error)."""
        reconstruction = self.forward(x)
        return torch.nn.functional.mse_loss(reconstruction, x)

    @torch.no_grad()
    def anomaly_map(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct the input and turn the per-pixel error into a single
        greyscale anomaly map per image (averaged over the R, G, B channels).
        Returns (anomaly_map [batch, H, W], reconstruction [batch, 3, H, W]).
        """
        reconstruction = self.forward(x)
        error = (x - reconstruction).abs().mean(dim=1)
        return error, reconstruction
