"""Lightweight local experiment tracking: TensorBoard scalars + a flat CSV/JSON
log per run. No external accounts/services required.

    logger = ExperimentLogger(run_name="ae_bottle_2026-09-03")
    logger.log_scalar("train/loss", 0.42, step=100)
    logger.log_metrics({"auroc_image": 0.91, "iou": 0.63, "latency_ms": 12.4})
    logger.close()
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    def __init__(self, run_name: str, runs_dir: str | Path = "runs"):
        self.run_dir = Path(runs_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.run_dir))
        self._csv_path = self.run_dir / "scalars.csv"
        self._metrics_path = self.run_dir / "final_metrics.json"
        self._csv_file = open(self._csv_path, "a", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        if self._csv_path.stat().st_size == 0:
            self._csv_writer.writerow(["timestamp", "tag", "step", "value"])

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)
        self._csv_writer.writerow([time.time(), tag, step, value])
        self._csv_file.flush()

    def log_metrics(self, metrics: dict, step: int = 0) -> None:
        """Log a dict of final/summary metrics (e.g. AUROC, IoU, latency) both
        as TensorBoard scalars and as a JSON file for easy cross-run comparison."""
        for name, value in metrics.items():
            self.log_scalar(f"eval/{name}", value, step)

        existing = {}
        if self._metrics_path.exists():
            existing = json.loads(self._metrics_path.read_text())
        existing.update(metrics)
        self._metrics_path.write_text(json.dumps(existing, indent=2))

    def close(self) -> None:
        self.writer.close()
        self._csv_file.close()
