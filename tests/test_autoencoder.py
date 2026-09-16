"""Sanity checks for the from-scratch Autoencoder on tiny random tensors --
no dataset or GPU required, kept fast for CI.
"""
import torch

from src.models.autoencoder import Autoencoder


def make_tiny_model():
    # image_size must be divisible by 16 (four stride-2 conv layers).
    return Autoencoder(latent_dim=8, image_size=32)


def test_forward_output_shape_matches_input():
    model = make_tiny_model()
    x = torch.randn(2, 3, 32, 32)
    reconstruction = model(x)
    assert reconstruction.shape == x.shape


def test_training_loss_is_finite_and_backprops():
    model = make_tiny_model()
    x = torch.randn(2, 3, 32, 32)

    loss = model.training_loss(x)
    assert torch.isfinite(loss)

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_anomaly_map_shape_and_non_negative():
    model = make_tiny_model()
    x = torch.randn(2, 3, 32, 32)

    anomaly_map, reconstruction = model.anomaly_map(x)
    assert anomaly_map.shape == (2, 32, 32)
    assert reconstruction.shape == x.shape
    assert (anomaly_map >= 0).all()


def test_reconstruction_improves_with_training():
    """Not a real training run -- just confirms gradient descent actually
    reduces the loss on a fixed tiny batch, i.e. the model is learnable."""
    torch.manual_seed(0)
    model = make_tiny_model()
    x = torch.randn(4, 3, 32, 32)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)

    first_loss = model.training_loss(x).item()
    for _ in range(50):
        optimiser.zero_grad()
        loss = model.training_loss(x)
        loss.backward()
        optimiser.step()
    last_loss = model.training_loss(x).item()

    assert last_loss < first_loss
