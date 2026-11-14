"""Sanity checks for the eval harness using synthetic (dummy) data —
no trained model required. Run with: python -m pytest tests/test_metrics.py -v
"""
import numpy as np
import pytest

from src.eval.metrics import (
    aggregate_image_scores,
    best_threshold_iou,
    image_auroc,
    measure_latency,
    pixel_auroc,
    reconstruction_quality,
    top_fraction_mean,
)


def test_image_auroc_perfect_separation():
    scores = np.array([0.1, 0.2, 0.9, 0.8, 0.15, 0.95])
    labels = np.array([0, 0, 1, 1, 0, 1])
    assert image_auroc(scores, labels) == pytest.approx(1.0)


def test_image_auroc_random_is_near_half():
    rng = np.random.default_rng(0)
    scores = rng.random(2000)
    labels = rng.integers(0, 2, size=2000)
    auroc = image_auroc(scores, labels)
    assert 0.4 < auroc < 0.6


def test_image_auroc_requires_both_classes():
    with pytest.raises(ValueError):
        image_auroc(np.array([0.1, 0.2]), np.array([0, 0]))


def test_pixel_auroc_perfect_map():
    gt = np.zeros((2, 8, 8))
    gt[0, 2:5, 2:5] = 1
    maps = gt.copy() * 10.0 + 0.01  # anomalous pixels score strictly higher
    assert pixel_auroc(maps, gt) == pytest.approx(1.0)


def test_best_threshold_iou_recovers_exact_box():
    gt = np.zeros((1, 10, 10))
    gt[0, 3:7, 3:7] = 1
    anomaly_map = gt.copy()  # scores exactly match ground truth
    result = best_threshold_iou(anomaly_map, gt, num_thresholds=10)
    assert result["iou"] == pytest.approx(1.0)


def test_best_threshold_iou_no_overlap_gives_low_score():
    gt = np.zeros((1, 10, 10))
    gt[0, 0:3, 0:3] = 1
    anomaly_map = np.zeros((1, 10, 10))
    anomaly_map[0, 7:10, 7:10] = 1.0
    result = best_threshold_iou(anomaly_map, gt, num_thresholds=10)
    assert result["iou"] < 0.2


def test_measure_latency_dummy_function():
    calls = {"n": 0}

    def fake_infer(_inputs):
        calls["n"] += 1
        return _inputs

    # min_warmup_seconds=0 disables the time-based warmup extension (see
    # test_measure_latency_warmup_extends_for_slow_min_warmup_seconds below),
    # so warmup here is exactly the `warmup` argument, like the old behaviour.
    result = measure_latency(fake_infer, inputs=None, warmup=2, n_runs=10, min_warmup_seconds=0)
    assert result.n_frames == 10
    assert result.mean_ms >= 0
    assert calls["n"] == 12  # warmup + n_runs


def test_measure_latency_warmup_extends_until_min_time_elapsed():
    """A fast dummy function finishes its `warmup` iterations almost
    instantly, so with min_warmup_seconds set, warmup should keep calling
    infer_fn past that count until enough real time has passed. This is the
    fix for the bug where fast configs (few DDIM steps) looked artificially
    slower than slow configs (many DDIM steps) because they weren't warmed
    up for as long in real time."""
    calls = {"n": 0}

    def fake_infer(_inputs):
        calls["n"] += 1

    measure_latency(fake_infer, inputs=None, warmup=1, n_runs=1, min_warmup_seconds=0.05)
    # warmup=1 alone would mean 1 warmup call; the time-based minimum should
    # force many more since each call is near-instant.
    assert calls["n"] > 1 + 1  # +1 for the single timed n_runs call at the end


def test_measure_latency_warmup_respects_max_iters_cap():
    """Without a cap, a min_warmup_seconds target could loop forever if
    infer_fn is fast enough that real time barely advances. max_warmup_iters
    guarantees the function always returns."""
    def fake_infer(_inputs):
        pass

    result = measure_latency(
        fake_infer, inputs=None, warmup=1, n_runs=1, min_warmup_seconds=100, max_warmup_iters=50
    )
    assert result.n_frames == 1


def test_reconstruction_quality_returns_expected_keys():
    originals = np.zeros((2, 3, 8, 8), dtype=np.float32)
    reconstructions = np.zeros((2, 3, 8, 8), dtype=np.float32)
    quality = reconstruction_quality(originals, reconstructions)
    assert set(quality.keys()) == {"psnr", "ssim"}


def test_reconstruction_quality_near_identical_images_score_highly():
    rng = np.random.default_rng(0)
    originals = rng.uniform(-1, 1, size=(4, 3, 16, 16)).astype(np.float32)
    reconstructions = originals + rng.normal(0, 0.01, size=originals.shape).astype(np.float32)

    quality = reconstruction_quality(originals, reconstructions)
    assert quality["ssim"] > 0.9
    assert quality["psnr"] > 20


def test_reconstruction_quality_is_better_for_closer_images():
    rng = np.random.default_rng(0)
    originals = rng.uniform(-1, 1, size=(4, 3, 16, 16)).astype(np.float32)
    close_reconstructions = originals + rng.normal(0, 0.01, size=originals.shape).astype(np.float32)
    far_reconstructions = rng.uniform(-1, 1, size=(4, 3, 16, 16)).astype(np.float32)

    close_quality = reconstruction_quality(originals, close_reconstructions)
    far_quality = reconstruction_quality(originals, far_reconstructions)

    assert close_quality["psnr"] > far_quality["psnr"]
    assert close_quality["ssim"] > far_quality["ssim"]


def test_top_fraction_mean_picks_the_highest_pixels():
    maps = np.zeros((2, 10, 10))          # 100 pixels per image
    maps[0, 0, :5] = 1.0                  # 5 pixels at 1.0 -> top5% mean = 1.0
    maps[1, 0, 0] = 1.0                   # 1 pixel at 1.0  -> top1% mean = 1.0, top5% = 0.2
    assert np.allclose(top_fraction_mean(maps, 0.05), [1.0, 0.2])
    assert np.allclose(top_fraction_mean(maps, 0.01), [1.0, 1.0])


def test_top_fraction_mean_uses_at_least_one_pixel_and_full_fraction_is_mean():
    maps = np.random.RandomState(0).rand(3, 4, 4)
    assert np.allclose(top_fraction_mean(maps, 1e-9), maps.reshape(3, -1).max(axis=1))
    assert np.allclose(top_fraction_mean(maps, 1.0), maps.reshape(3, -1).mean(axis=1))


def test_aggregate_image_scores_has_every_aggregation_and_ordering():
    maps = np.random.RandomState(1).rand(5, 16, 16)
    scores = aggregate_image_scores(maps)
    assert set(scores) == {"max", "mean", "top0.1%", "top1%", "top5%"}
    # a smaller worst-fraction can only be larger than a bigger one, and max >= everything
    assert np.all(scores["max"] >= scores["top0.1%"] - 1e-12)
    assert np.all(scores["top0.1%"] >= scores["top1%"] - 1e-12)
    assert np.all(scores["top1%"] >= scores["top5%"] - 1e-12)
    assert np.all(scores["top5%"] >= scores["mean"] - 1e-12)
