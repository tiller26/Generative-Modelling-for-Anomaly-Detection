"""Sanity checks for the from-scratch UNet + GaussianDiffusion (DDIM) on tiny
random tensors -- no dataset or GPU required, kept fast for CI.
"""
import torch

from src.models.diffusion import GaussianDiffusion
from src.models.unet import UNet


def make_tiny_model():
    # img_size=16 with 2 downsample stages (channel_mults has 3 levels) -> 16x16 -> 8x8 -> 4x4
    return UNet(in_channels=3, base_channels=8, channel_mults=(1, 2, 4), num_res_blocks=1)


def test_unet_output_shape_matches_input():
    model = make_tiny_model()
    x = torch.randn(2, 3, 16, 16)
    t = torch.randint(0, 1000, (2,))
    out = model(x, t)
    assert out.shape == x.shape


def test_q_sample_shape_and_range():
    diffusion = GaussianDiffusion(timesteps=100)
    x0 = torch.randn(4, 3, 16, 16)
    t = torch.randint(0, 100, (4,))
    noise = torch.randn_like(x0)
    x_t = diffusion.q_sample(x0, t, noise)
    assert x_t.shape == x0.shape
    assert torch.isfinite(x_t).all()


def test_q_sample_at_t0_is_close_to_x0():
    diffusion = GaussianDiffusion(timesteps=1000)
    x0 = torch.randn(1, 3, 16, 16)
    t = torch.zeros(1, dtype=torch.long)
    noise = torch.randn_like(x0)
    x_t = diffusion.q_sample(x0, t, noise)
    assert torch.allclose(x_t, x0, atol=0.3)


def test_training_loss_is_finite_and_backprops():
    model = make_tiny_model()
    diffusion = GaussianDiffusion(timesteps=100)
    x0 = torch.randn(2, 3, 16, 16)

    loss = diffusion.training_loss(model, x0)
    assert torch.isfinite(loss)

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_ddim_reconstruct_shape():
    model = make_tiny_model()
    diffusion = GaussianDiffusion(timesteps=100)
    x0 = torch.randn(2, 3, 16, 16)

    with torch.no_grad():
        reconstruction = diffusion.ddim_reconstruct(model, x0, t_start=50, ddim_steps=5)
    assert reconstruction.shape == x0.shape
    assert torch.isfinite(reconstruction).all()


def test_anomaly_map_shape():
    model = make_tiny_model()
    diffusion = GaussianDiffusion(timesteps=100)
    x0 = torch.randn(2, 3, 16, 16)

    with torch.no_grad():
        anomaly_map, reconstruction = diffusion.anomaly_map(model, x0, t_start=50, ddim_steps=5)
    assert anomaly_map.shape == (2, 16, 16)
    assert reconstruction.shape == x0.shape
    assert (anomaly_map >= 0).all()
