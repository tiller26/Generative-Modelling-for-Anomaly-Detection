"""Gaussian diffusion process: training objective (DDPM) + deterministic
sampling (DDIM), plus the reconstruction-based anomaly scoring built on top.

Original implementation of the standard formulations from Ho et al. 2020
(DDPM) and Song et al. 2020 (DDIM) -- not adapted from a reference codebase.

Anomaly detection recipe (partial-diffusion reconstruction, used by most
diffusion-based AD methods): noise a test image to an intermediate timestep
t* < T (not all the way to pure noise), then run the deterministic DDIM
reverse process back to t=0. A normal image survives this round-trip almost
unchanged; a defect region does not resemble anything in the learned normal
data manifold, so the model "hallucinates" a plausible normal patch there --
the difference between input and reconstruction is the anomaly map.
"""
from __future__ import annotations

import torch
from torch import nn


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal cosine schedule -- avoids the too-fast noising of a
    linear schedule near t=0, which matters here since anomaly scoring relies
    on partial (small-t) noising leaving normal structure intact."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-4, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps: int = 1000):
        super().__init__()
        self.timesteps = timesteps

        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def _extract(self, values: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        out = values.gather(0, t)
        return out.reshape(t.shape[0], *([1] * (len(shape) - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward process: sample x_t ~ q(x_t | x_0) in one shot."""
        sqrt_ac = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_1m_ac = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ac * x0 + sqrt_1m_ac * noise

    def training_loss(self, model: nn.Module, x0: torch.Tensor) -> torch.Tensor:
        """Standard DDPM simple loss: MSE between actual and predicted noise,
        for a uniformly sampled timestep per image in the batch."""
        t = torch.randint(0, self.timesteps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        predicted_noise = model(x_t, t)
        return torch.nn.functional.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def ddim_reconstruct(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        t_start: int,
        ddim_steps: int = 50,
    ) -> torch.Tensor:
        """Noise x0 to timestep t_start, then deterministically denoise back
        to t=0 with DDIM (eta=0) using a strided subsequence of timesteps.
        Returns the reconstruction, same shape as x0.
        """
        assert 0 < t_start <= self.timesteps
        device = x0.device

        t_start_batch = torch.full((x0.shape[0],), t_start - 1, device=device, dtype=torch.long)
        x_t = self.q_sample(x0, t_start_batch, torch.randn_like(x0))

        step_indices = torch.linspace(0, t_start - 1, ddim_steps, device=device).long().unique().flip(0)

        for i, t_val in enumerate(step_indices):
            t_batch = torch.full((x0.shape[0],), t_val.item(), device=device, dtype=torch.long)
            predicted_noise = model(x_t, t_batch)

            alpha_cumprod_t = self._extract(self.alphas_cumprod, t_batch, x0.shape)
            predicted_x0 = (x_t - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            predicted_x0 = predicted_x0.clamp(-1, 1)

            if i + 1 < len(step_indices):
                t_prev = step_indices[i + 1]
                alpha_cumprod_prev = self.alphas_cumprod[t_prev]
            else:
                alpha_cumprod_prev = torch.tensor(1.0, device=device)

            x_t = torch.sqrt(alpha_cumprod_prev) * predicted_x0 + torch.sqrt(1 - alpha_cumprod_prev) * predicted_noise

        return x_t

    def anomaly_map(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        t_start: int,
        ddim_steps: int = 50,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruction-based anomaly map: per-pixel absolute error between
        input and its noise-then-denoise round trip, averaged over channels.
        Returns (anomaly_map [B, H, W], reconstruction [B, C, H, W]).
        """
        reconstruction = self.ddim_reconstruct(model, x0, t_start, ddim_steps)
        error = (x0 - reconstruction).abs().mean(dim=1)
        return error, reconstruction
