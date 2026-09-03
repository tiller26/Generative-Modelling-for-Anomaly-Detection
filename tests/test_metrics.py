"""Sanity checks for the eval harness using synthetic (dummy) data —
no trained model required. Run with: python -m pytest tests/test_metrics.py -v
"""
import numpy as np
import pytest

from src.eval.metrics import (
    best_threshold_iou,
    image_auroc,
    measure_latency,
    pixel_auroc,
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

    result = measure_latency(fake_infer, inputs=None, warmup=2, n_runs=10)
    assert result.n_frames == 10
    assert result.mean_ms >= 0
    assert calls["n"] == 12  # warmup + n_runs
