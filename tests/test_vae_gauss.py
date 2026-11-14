"""Sanity checks for the Gaussian-likelihood VAE (An & Cho style), on tiny
random tensors -- no dataset or GPU required."""
import math

import torch

from src.models.vae_gauss import LOGVAR_MAX, LOGVAR_MIN, GaussianAE, GaussianVAE, gaussian_log_prob


def make_tiny_model(**kwargs):
    return GaussianVAE(latent_dim=8, image_size=32, **kwargs)


def test_gaussian_log_prob_matches_closed_form():
    x = torch.tensor([0.5])
    mean = torch.tensor([0.0])
    logvar = torch.tensor([0.0])  # variance 1
    expected = -0.5 * (math.log(2 * math.pi) + 0.25)
    assert torch.isclose(gaussian_log_prob(x, mean, logvar), torch.tensor([expected]))


def test_decoder_outputs_and_logvar_range():
    model = make_tiny_model()
    mean, logvar = model.decoder(torch.randn(3, 8))
    assert mean.shape == (3, 3, 32, 32) and logvar.shape == (3, 3, 32, 32)
    assert mean.abs().max() <= 1.0
    assert logvar.min() >= LOGVAR_MIN and logvar.max() <= LOGVAR_MAX


def test_training_loss_finite_and_backprops():
    model = make_tiny_model()
    loss = model.training_loss(torch.randn(2, 3, 32, 32).clamp(-1, 1))
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)


def test_all_score_modes_shape_and_repeatable():
    x = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    for mode in ["prob", "abs", "wsq"]:
        model = make_tiny_model(score_mode=mode, n_samples=3)
        a1, rec = model.anomaly_map(x)
        a2, _ = model.anomaly_map(x)
        assert a1.shape == (2, 32, 32) and rec.shape == x.shape
        assert torch.isfinite(a1).all()
        assert torch.allclose(a1, a2)  # fixed-seed sampling => repeatable


def test_unlikely_input_gets_higher_prob_score_than_the_models_own_output():
    model = make_tiny_model(score_mode="prob", n_samples=5)
    x = torch.randn(1, 3, 32, 32).clamp(-1, 1)
    mu, _ = model.encoder(x)
    mean, _ = model.decoder(mu)
    own, _ = model.anomaly_map(mean.detach())
    far, _ = model.anomaly_map((mean.detach() + 0.8).clamp(-1, 1))
    assert far.mean() > own.mean()


def test_kl_weight_scales_the_loss():
    x = torch.randn(4, 3, 32, 32).clamp(-1, 1)
    model = make_tiny_model()
    torch.manual_seed(0)
    model.kl_weight = 0.0
    no_kl = model.training_loss(x)
    torch.manual_seed(0)
    model.kl_weight = 1e6  # an untrained encoder has near-zero KL, so use a big weight to see the effect
    with_kl = model.training_loss(x)
    assert with_kl > no_kl


def test_gaussian_ae_is_deterministic_and_trains():
    model = GaussianAE(latent_dim=8, image_size=32)
    x = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    model.eval()
    # no sampling anywhere: two forward passes are identical
    assert torch.equal(model(x)[0], model(x)[0])
    model.train()
    loss = model.training_loss(x)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)


def test_gaussian_ae_score_modes():
    x = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    for mode in ["prob", "abs", "wsq"]:
        model = GaussianAE(latent_dim=8, image_size=32, score_mode=mode)
        score, rec = model.anomaly_map(x)
        assert score.shape == (2, 32, 32) and rec.shape == x.shape
        assert torch.isfinite(score).all()
