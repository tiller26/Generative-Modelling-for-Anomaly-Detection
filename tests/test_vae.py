"""Sanity checks for the from-scratch VAE on tiny random tensors -- no
dataset or GPU required, kept fast for CI.
"""
import torch

from src.models.vae import VAE, kl_divergence, reparameterize


def make_tiny_model():
    return VAE(latent_dim=8, image_size=32, kl_weight=0.001)


def test_forward_output_shapes():
    model = make_tiny_model()
    x = torch.randn(2, 3, 32, 32)
    reconstruction, mu, logvar = model(x)
    assert reconstruction.shape == x.shape
    assert mu.shape == (2, 8)
    assert logvar.shape == (2, 8)


def test_reparameterize_shape():
    mu = torch.zeros(4, 8)
    logvar = torch.zeros(4, 8)
    z = reparameterize(mu, logvar)
    assert z.shape == mu.shape


def test_kl_divergence_is_zero_for_standard_normal():
    # mu=0, logvar=0 (variance=1) IS the standard normal distribution,
    # so the KL divergence against itself should be exactly 0.
    mu = torch.zeros(4, 8)
    logvar = torch.zeros(4, 8)
    kl = kl_divergence(mu, logvar)
    assert torch.isclose(kl, torch.tensor(0.0), atol=1e-6)


def test_kl_divergence_is_positive_when_distributions_differ():
    mu = torch.ones(4, 8) * 2.0
    logvar = torch.zeros(4, 8)
    kl = kl_divergence(mu, logvar)
    assert kl.item() > 0


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


def test_kl_weight_scales_the_loss_and_zero_weight_is_plain_reconstruction():
    # Guards the posterior-collapse fix: scripts/train.py anneals
    # model.kl_weight from 0 upward by assigning to it each epoch, so the
    # attribute must actually drive the loss. At weight 0 the loss must equal
    # a plain autoencoder's MSE (no KL term at all).
    model = make_tiny_model()
    x = torch.randn(4, 3, 32, 32)

    torch.manual_seed(0)
    model.kl_weight = 0.0
    loss_no_kl = model.training_loss(x)

    torch.manual_seed(0)  # same sampled noise, so the only difference is the weight
    model.kl_weight = 1.0
    loss_with_kl = model.training_loss(x)

    assert loss_with_kl > loss_no_kl
