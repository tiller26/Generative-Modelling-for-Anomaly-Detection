"""Evaluation harness shared across AE / VAE / DDIM anomaly detectors.

All functions take plain numpy arrays so they work regardless of which
framework produced the anomaly maps. An "anomaly map" is a per-pixel score
(higher = more anomalous), the same spatial size as the input image.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score


def image_auroc(image_scores: np.ndarray, image_labels: np.ndarray) -> float:
    """Image-level AUROC. `image_scores` is one scalar anomaly score per image
    (e.g. max or mean of its anomaly map); `image_labels` is 0 (normal) / 1 (anomalous).
    """
    image_scores = np.asarray(image_scores).ravel()
    image_labels = np.asarray(image_labels).ravel()
    if len(np.unique(image_labels)) < 2:
        raise ValueError("image_auroc requires both normal and anomalous samples")
    return float(roc_auc_score(image_labels, image_scores))


def pixel_auroc(anomaly_maps: np.ndarray, gt_masks: np.ndarray) -> float:
    """Pixel-level AUROC over every pixel of every test image.

    anomaly_maps: (N, H, W) float scores.
    gt_masks: (N, H, W) binary ground-truth masks (0/1).
    """
    scores = np.asarray(anomaly_maps).ravel()
    labels = (np.asarray(gt_masks).ravel() > 0.5).astype(int)
    if len(np.unique(labels)) < 2:
        raise ValueError("pixel_auroc requires both normal and anomalous pixels")
    return float(roc_auc_score(labels, scores))


def best_threshold_iou(anomaly_maps: np.ndarray, gt_masks: np.ndarray, num_thresholds: int = 50) -> dict:
    """Sweep thresholds on the anomaly map and report the best mean per-image IoU.

    Returns {"iou": float, "threshold": float}.
    """
    anomaly_maps = np.asarray(anomaly_maps)
    gt_masks = (np.asarray(gt_masks) > 0.5)

    lo, hi = float(anomaly_maps.min()), float(anomaly_maps.max())
    if lo == hi:
        raise ValueError("anomaly_maps has zero dynamic range; cannot threshold")

    best_iou, best_thresh = 0.0, lo
    for thresh in np.linspace(lo, hi, num_thresholds):
        pred = anomaly_maps > thresh
        iou = _mean_iou(pred, gt_masks)
        if iou > best_iou:
            best_iou, best_thresh = iou, float(thresh)

    return {"iou": best_iou, "threshold": best_thresh}


def _mean_iou(pred_masks: np.ndarray, gt_masks: np.ndarray, eps: float = 1e-8) -> float:
    """Mean IoU across the batch dimension, skipping images with no anomaly
    predicted and no anomaly present (both-empty -> IoU 1.0, avoids penalizing
    correctly-quiet 'good' images)."""
    axes = tuple(range(1, pred_masks.ndim))
    intersection = np.logical_and(pred_masks, gt_masks).sum(axis=axes)
    union = np.logical_or(pred_masks, gt_masks).sum(axis=axes)

    iou = np.where(union == 0, 1.0, intersection / np.maximum(union, eps))
    return float(iou.mean())


@dataclass
class LatencyResult:
    mean_ms: float
    std_ms: float
    p95_ms: float
    n_frames: int
    per_frame_ms: list[float] = field(default_factory=list)


def measure_latency(
    infer_fn, inputs, warmup: int = 5, n_runs: int = 50, sync_fn=None,
    min_warmup_seconds: float = 1.0, max_warmup_iters: int = 2000,
) -> LatencyResult:
    """Measure per-frame inference latency in milliseconds.

    infer_fn: callable that runs one forward pass, e.g. `lambda: model(batch)`.
    sync_fn: optional callable to synchronize device timing (e.g. `torch.cuda.synchronize`).
              Pass this when timing GPU code, otherwise CUDA calls return before finishing
              and latency reads as artificially low.

    Many GPUs (especially laptop GPUs) dynamically scale their clock speed:
    idle/cold means a low, power-saving clock, and it takes real wall-clock
    time under sustained load before the clock ramps up to full boost speed.
    A *fixed number* of warmup calls isn't enough to guarantee this ramp-up
    actually finished -- a config with very few, very fast calls (e.g. only
    5 DDIM steps) might still be "cold" by the time warmup ends, while a
    config with many slow calls (e.g. 100 DDIM steps) naturally gets more
    real warmup time and ends up looking artificially *faster per step*.
    We saw exactly this in practice: a 25-step DDIM run measured slower than
    a 100-step run on the same model, which is impossible if both are fully
    warmed up. So warmup here keeps going until BOTH the iteration count
    (`warmup`) AND a minimum real time (`min_warmup_seconds`) are satisfied.
    """
    warmup_start = time.perf_counter()
    iterations_done = 0
    while iterations_done < warmup or (time.perf_counter() - warmup_start) < min_warmup_seconds:
        infer_fn(inputs)
        iterations_done += 1
        if iterations_done >= max_warmup_iters:
            break
    if sync_fn:
        sync_fn()

    timings_ms = []
    for _ in range(n_runs):
        start = time.perf_counter()
        infer_fn(inputs)
        if sync_fn:
            sync_fn()
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    timings = np.array(timings_ms)
    return LatencyResult(
        mean_ms=float(timings.mean()),
        std_ms=float(timings.std()),
        p95_ms=float(np.percentile(timings, 95)),
        n_frames=n_runs,
        per_frame_ms=timings_ms,
    )
