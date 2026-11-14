"""A VAE in the style of An & Cho (2015), "Variational Autoencoder based
Anomaly Detection using Reconstruction Probability". Original implementation.

How this differs from src/models/vae.py:
  1. The decoder outputs a MEAN and a LOG-VARIANCE for every pixel (a
     Gaussian likelihood p(x|z) with a learned, per-pixel variance), instead
     of just a mean.
  2. Training uses the proper ELBO: the Gaussian negative log-likelihood plus
     the KL divergence, weight 1. Because the variance is learned, the
     reconstruction term automatically gets stronger as the model improves,
     so we do not need a hand-tuned, tiny kl_weight (see vae.py's training_loss
     for the loss-scale problem that caused our first VAE to collapse).
  3. Anomalies are scored by the RECONSTRUCTION PROBABILITY: draw L latent
     samples z ~ q(z|x), decode each to a (mean, variance), and average the
     probability of the real input under those Gaussians. A low probability
     means "this input is unlikely under the model of normal images".

Findings (see README "Probabilistic scoring"): the learned variance used in the
SCORE is what helps -- a deterministic AE with the same decoder (GaussianAE
below) gains as much -- and n_samples (L) makes no measurable difference
(L=1 matched L=10). The reported results use the default L=10; the GUI uses
L=1 because it is ~7x cheaper.

score_mode lets the SAME trained model be scored three ways, so we can tell
whether any gain comes from training or from the scoring rule:
  "prob"  -- negative log reconstruction probability (the paper's score)
  "abs"   -- absolute error of the decoder mean at the mean latent (our
             original VAE's score; ignores the learned variance)
  "wsq"   -- variance-weighted squared error of the decoder mean at the mean
             latent ((x - mean)^2 / variance); uses the variance, no sampling
"""
from __future__ import annotations

import math

import torch
from torch import nn

from src.models.autoencoder import Encoder
from src.models.vae import VAEEncoder, kl_divergence, reparameterize

LOGVAR_MIN, LOGVAR_MAX = -10.0, 2.0


class GaussianDecoder(nn.Module):
    """Same layers as autoencoder.Decoder, but the last transposed conv
    produces 6 channels: 3 for the per-pixel mean (Tanh, in [-1, 1] like the
    input images) and 3 for the per-pixel log-variance."""

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
            nn.ConvTranspose2d(32, 6, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.from_latent(z)
        features = features.reshape(-1, self.start_channels, self.feature_map_size, self.feature_map_size)
        out = self.deconv_layers(features)
        mean_raw, logvar_raw = out[:, :3], out[:, 3:]
        mean = torch.tanh(mean_raw)
        # Squash the raw log-variance into [LOGVAR_MIN, LOGVAR_MAX] smoothly.
        # A hard clamp would give zero gradient at the bounds; tanh keeps it
        # trainable while still preventing exp(logvar) from exploding or
        # collapsing to 0 (which would make the likelihood infinite).
        half_range = (LOGVAR_MAX - LOGVAR_MIN) / 2
        mid = (LOGVAR_MAX + LOGVAR_MIN) / 2
        logvar = mid + half_range * torch.tanh(logvar_raw / half_range)
        return mean, logvar


def gaussian_log_prob(x: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Per-element log N(x; mean, exp(logvar))."""
    return -0.5 * (math.log(2 * math.pi) + logvar + (x - mean).pow(2) * torch.exp(-logvar))


class GaussianVAE(nn.Module):
    def __init__(self, latent_dim: int = 256, image_size: int = 128, kl_weight: float = 1.0,
                 n_samples: int = 10, score_mode: str = "prob"):
        super().__init__()
        self.encoder = VAEEncoder(latent_dim, image_size)
        self.decoder = GaussianDecoder(latent_dim, image_size)
        # kl_weight is 1.0 for the true ELBO; scripts/train.py ramps it from 0
        # (KL annealing) by assigning to this attribute each epoch.
        self.kl_weight = kl_weight
        self.n_samples = n_samples
        self.score_mode = score_mode

    def forward(self, x: torch.Tensor):
        mu, logvar_z = self.encoder(x)
        z = reparameterize(mu, logvar_z)
        mean_x, logvar_x = self.decoder(z)
        return mean_x, logvar_x, mu, logvar_z

    def training_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Negative ELBO per image, divided by the number of pixel values so
        the number is on a convenient scale (dividing by a constant does not
        change the optimum or the NLL-vs-KL balance)."""
        mean_x, logvar_x, mu, logvar_z = self.forward(x)
        nll = -gaussian_log_prob(x, mean_x, logvar_x).flatten(start_dim=1).sum(dim=1).mean()
        kl = kl_divergence(mu, logvar_z)
        n_values = x[0].numel()
        return (nll + self.kl_weight * kl) / n_values

    @torch.no_grad()
    def anomaly_map(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (anomaly map [batch, H, W], reconstruction [batch, 3, H, W]).
        The reconstruction is the decoder mean at the mean latent, so PSNR/SSIM
        are comparable with the other models."""
        mu, logvar_z = self.encoder(x)
        mean_mu, logvar_mu = self.decoder(mu)

        if self.score_mode == "abs":
            score = (x - mean_mu).abs().mean(dim=1)
        elif self.score_mode == "wsq":
            score = ((x - mean_mu).pow(2) * torch.exp(-logvar_mu)).mean(dim=1)
        elif self.score_mode == "prob":
            # Fixed-seed generator so the sampled score is repeatable.
            gen = torch.Generator(device=x.device).manual_seed(0)
            std = torch.exp(0.5 * logvar_z)
            log_probs = []
            for _ in range(self.n_samples):
                eps = torch.randn(std.shape, generator=gen, device=x.device)
                mean_x, logvar_x = self.decoder(mu + std * eps)
                log_probs.append(gaussian_log_prob(x, mean_x, logvar_x))
            # Average the PROBABILITIES over the samples (as in the paper), done
            # stably in log space: log mean_l p_l = logsumexp(log p_l) - log L.
            log_mean_prob = torch.logsumexp(torch.stack(log_probs), dim=0) - math.log(self.n_samples)
            score = (-log_mean_prob).mean(dim=1)
        else:
            raise ValueError(f"Unknown score_mode '{self.score_mode}'")
        return score, mean_mu


class GaussianAE(nn.Module):
    """The deterministic counterpart of GaussianVAE: a plain autoencoder encoder
    (one fixed latent vector per image, no sampling, no KL term) feeding the
    same GaussianDecoder (per-pixel mean + log-variance), trained with the
    Gaussian negative log-likelihood only.

    It exists to complete a 2x2 comparison with the plain AE / VAE:
        AE            deterministic latent, mean decoder,     abs score
        AE + variance deterministic latent, mean+var decoder, probability score   <- this class
        VAE           stochastic latent,    mean decoder,     abs score
        VAE + variance stochastic latent,   mean+var decoder, probability score   <- GaussianVAE
    score_mode works exactly as in GaussianVAE ("prob", "abs", "wsq"); with a
    deterministic latent the probability score needs no sampling.
    """

    def __init__(self, latent_dim: int = 256, image_size: int = 128, score_mode: str = "prob"):
        super().__init__()
        self.encoder = Encoder(latent_dim, image_size)
        self.decoder = GaussianDecoder(latent_dim, image_size)
        self.score_mode = score_mode

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decoder(self.encoder(x))

    def training_loss(self, x: torch.Tensor) -> torch.Tensor:
        mean_x, logvar_x = self.forward(x)
        nll = -gaussian_log_prob(x, mean_x, logvar_x).flatten(start_dim=1).sum(dim=1).mean()
        return nll / x[0].numel()

    @torch.no_grad()
    def anomaly_map(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean_x, logvar_x = self.forward(x)
        if self.score_mode == "abs":
            score = (x - mean_x).abs().mean(dim=1)
        elif self.score_mode == "wsq":
            score = ((x - mean_x).pow(2) * torch.exp(-logvar_x)).mean(dim=1)
        elif self.score_mode == "prob":
            score = (-gaussian_log_prob(x, mean_x, logvar_x)).mean(dim=1)
        else:
            raise ValueError(f"Unknown score_mode '{self.score_mode}'")
        return score, mean_x
