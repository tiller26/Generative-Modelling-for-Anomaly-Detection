"""PyQt5 desktop prototype for the anomaly detection benchmark -- an
inspection-station style UI.

Two tabs:
  Inspect: pick a category and a trained model, load an image (or a whole
           folder), and the app reports a verdict -- NORMAL or ANOMALY
           DETECTED -- together with the input, the model's reconstruction,
           the anomaly heat-map and an overlay of the heat-map on the part.
           Every inspection is logged to a results table that can be exported
           to CSV.
  Train:   configure hyperparameters (epochs, batch size, learning rate) and
           train a new checkpoint, watching the train/val loss curve update
           live. This is the GUI's own hyperparameter configuration +
           telemetry.

How the verdict is made
-----------------------
The models output a per-pixel anomaly map, not a yes/no answer, and raw
scores are on a different scale for every model and category. So the first
time a (model, category) pair is used the app *calibrates* it: it scores the
held-out normal validation images (the same train/val split, seed 42, that
scripts/train.py used -- images the model never trained on, and no defective
image is involved) and sets the decision threshold at

    threshold = mean + k * std        (of the image scores of those normal images)

An image whose score reaches the threshold is flagged as an anomaly. The image
score is the mean of the worst 1 % of pixels ("top-1 %"), the pooling rule
that was best on average in the paper's aggregation experiment. k is the
"Sensitivity" slider: a lower k flags more images (fewer missed defects, more
false alarms). Calibrations are cached next to each checkpoint.

The Train tab launches scripts/train.py or scripts/train_diffusion.py as a
genuine separate process (QProcess) rather than re-implementing the training
loop inside the GUI. This was a deliberate fix, not the original design: an
earlier version ran training on a QThread inside this same process, reusing
the training functions directly -- but a PyTorch DataLoader with
num_workers>0 spawned from a background QThread (rather than the process's
main thread) turned out to be unreliable on Windows, repeatedly hanging in a
worker-crash/respawn loop during testing. (Inference on a QThread, as the
Inspect tab does, uses no DataLoader workers and is fine.)

Run with:
    python app/gui.py

IMPORTANT: `import torch` must come before any `import PyQt5...` line below.
On Windows, importing PyQt5 first loads a conflicting copy of a shared DLL
that then makes torch's own DLL loading crash with a cryptic
"WinError 1114" -- confirmed by testing both import orders directly. Do not
reorder or alphabetize these imports.
"""
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch  # must be imported before PyQt5 -- see warning above
from matplotlib import cm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
from PyQt5.QtCore import QProcess, QRectF, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# This file lives in app/, but it needs to import from scripts/ (to reuse
# the exact same model-loading and image-preprocessing code the batch
# evaluation scripts use, so the GUI behaves identically to the numbers in
# the paper) and from the repo root (for src/).
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from evaluate import build_transform as build_ae_vae_transform, load_model  # noqa: E402
from evaluate_diffusion import build_transform as build_ddim_transform, load_checkpoint  # noqa: E402
from src.data.mvtec_dataset import MVTecTrainDataset, train_val_split  # noqa: E402

CATEGORIES = ["bottle", "capsule", "pill"]
DEFAULT_DDIM_T_START = 250
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

# GUI model name (lower-cased combo text) -> how to train / load / score it:
#   (value for train.py --model, checkpoint folder prefix, optional folder tag, score mode)
# "VAE" loads the TUNED VAE (kl_weight=1e-6 + KL annealing, folder tag "kl1e-6"),
# NOT the original checkpoints/vae_<category>/ run, which suffered posterior
# collapse and is kept only as the "before" result. "Prob. AE" / "Prob. VAE" are
# the models with a learned per-pixel variance, scored by -log likelihood
# (reconstruction probability) instead of absolute pixel error.
MODEL_SPECS = {
    "ae": ("ae", "ae", "", None),
    "vae": ("vae", "vae", "kl1e-6", None),
    "prob. ae": ("ae_gauss", "ae_gauss", "", "prob"),
    "prob. vae": ("vae_gauss", "vae_gauss", "", "prob"),
    "ddim": (None, "ddim", "", None),
}
MODEL_CHOICES = ["AE", "VAE", "Prob. AE", "Prob. VAE", "DDIM"]

# Decision rule (see the module docstring).
TOP_FRACTION = 0.01          # image score = mean of the worst 1 % of pixels
DEFAULT_K = 2.0              # threshold = mean + k * std of held-out normal scores
BORDERLINE_BAND = 0.10       # within +/-10 % of the threshold -> "borderline"
MAX_CALIBRATION_IMAGES = 24  # caps DDIM calibration time
CALIBRATION_VERSION = 1

PANEL_SIZE = 280  # how big each image panel looks on screen

# Matches a line like "[epoch 10/100] train_loss=0.0554 val_loss=0.0448 (99s elapsed)",
# printed by both scripts/train.py and scripts/train_diffusion.py every 10
# epochs (and on the final epoch) -- this is what feeds the live loss plot.
EPOCH_LINE_PATTERN = re.compile(r"\[epoch (\d+)/\d+\] train_loss=([\d.eE+-]+) val_loss=([\d.eE+-]+)")

# ---- visual theme ---------------------------------------------------------
BG = "#0b1220"
SURFACE = "#111a2e"
SURFACE_2 = "#17233b"
BORDER = "#24324f"
TEXT = "#e5ecf6"
MUTED = "#8b9bb8"
ACCENT = "#38bdf8"
OK_COLOR = "#22c55e"
BAD_COLOR = "#ef4444"
WARN_COLOR = "#f59e0b"
IDLE_COLOR = "#475569"

STYLESHEET = f"""
QMainWindow, QWidget#root {{ background: {BG}; }}
QWidget {{ color: {TEXT}; font-family: "Segoe UI", "Inter", sans-serif; font-size: 13px; }}
QLabel {{ background: transparent; }}
QFrame#card, QFrame#sidebar {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; }}
QFrame#header {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; }}
QLabel#h1 {{ font-size: 18px; font-weight: 600; }}
QLabel#sectionTitle {{ color: {MUTED}; font-size: 11px; font-weight: 600; letter-spacing: 1px; }}
QLabel#panelTitle {{ color: {MUTED}; font-size: 11px; font-weight: 600; letter-spacing: 1px; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#chip {{ background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 10px; padding: 3px 10px; color: {MUTED}; }}
QLabel#imagePanel {{ background: #060b16; border: 1px solid {BORDER}; border-radius: 8px; }}
QLabel#metricValue {{ font-size: 20px; font-weight: 600; }}
QLabel#metricName {{ color: {MUTED}; font-size: 11px; }}
QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 6px; padding: 5px 8px; min-height: 22px;
}}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {ACCENT}; }}
QComboBox:disabled, QSpinBox:disabled {{ color: #4b5a75; }}
QComboBox QAbstractItemView {{ background: {SURFACE_2}; selection-background-color: {ACCENT}; selection-color: #04121c; border: 1px solid {BORDER}; }}
QPushButton {{
    background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 6px; padding: 8px 14px; font-weight: 600;
}}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: #4b5a75; border-color: {SURFACE_2}; }}
QPushButton#primary {{ background: {ACCENT}; color: #04121c; border: none; }}
QPushButton#primary:hover {{ background: #7dd3fc; }}
QPushButton#primary:disabled {{ background: #1e3a4c; color: #4b6b7f; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 14px; height: 14px; margin: -6px 0; border-radius: 7px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: transparent; color: {MUTED}; padding: 9px 22px; font-weight: 600; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTableWidget {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px; gridline-color: {BORDER}; alternate-background-color: {SURFACE_2}; }}
QHeaderView::section {{ background: {SURFACE_2}; color: {MUTED}; border: none; border-bottom: 1px solid {BORDER}; padding: 6px; font-weight: 600; }}
QTableWidget::item:selected {{ background: #1e3a5f; }}
QProgressBar {{ background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 6px; text-align: center; height: 16px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 5px; }}
QScrollBar:vertical {{ background: {SURFACE}; width: 10px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; min-height: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QMessageBox {{ background: {SURFACE}; }}
"""


def checkpoint_folder(model_type: str, category: str) -> str:
    _train_model, prefix, tag, _score_mode = MODEL_SPECS[model_type]
    return f"{prefix}_{category}" + (f"_{tag}" if tag else "")


def tensor_to_rgb(image_tensor: torch.Tensor) -> np.ndarray:
    """[-1, 1]-scaled (channels, height, width) tensor -> uint8 (H, W, 3)."""
    image_0_to_1 = (image_tensor.clamp(-1, 1) + 1) / 2
    image_uint8 = (image_0_to_1.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return np.ascontiguousarray(image_uint8)


def array_to_pixmap(array: np.ndarray, size: int = PANEL_SIZE) -> QPixmap:
    """uint8 (H, W, 3) or (H, W, 4) array -> QPixmap scaled to the panel."""
    array = np.ascontiguousarray(array)
    height, width, channels = array.shape
    fmt = QImage.Format_RGB888 if channels == 3 else QImage.Format_RGBA8888
    qimage = QImage(array.data, width, height, channels * width, fmt).copy()
    return QPixmap.fromImage(qimage).scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def normalise_map(anomaly_map: torch.Tensor, vmax: float) -> np.ndarray:
    """Scales a raw anomaly map to [0, 1] against a fixed reference ceiling
    (the largest pixel score seen on normal calibration images), so a normal
    part looks dark and a defective one lights up -- rather than every image
    being stretched to its own min/max."""
    map_np = anomaly_map.squeeze().detach().cpu().numpy()
    return np.clip(map_np / max(vmax, 1e-8), 0.0, 1.0)


def heatmap_rgb(normalised: np.ndarray) -> np.ndarray:
    return (cm.inferno(normalised)[..., :3] * 255).astype(np.uint8)


def overlay_rgb(image_rgb: np.ndarray, normalised: np.ndarray) -> np.ndarray:
    """Blends the heat-map over the input; strong anomaly -> opaque heat."""
    heat = cm.inferno(normalised)[..., :3] * 255
    alpha = (np.clip(normalised, 0, 1) ** 1.5 * 0.85)[..., None]
    return (image_rgb * (1 - alpha) + heat * alpha).astype(np.uint8)


def top_fraction_score(anomaly_map: torch.Tensor, fraction: float = TOP_FRACTION) -> float:
    flat = anomaly_map.flatten()
    return torch.topk(flat, k=max(1, int(fraction * flat.numel()))).values.mean().item()


def ground_truth_from_path(path: Path) -> str:
    """MVTec layout: .../test/good/x.png is normal, .../test/<defect>/x.png is
    defective. Anything else (e.g. a customer's own image) -> unknown."""
    parts = [p.lower() for p in path.parts]
    if "test" in parts:
        return "normal" if path.parent.name.lower() == "good" else "defect"
    return "unknown"


# ---- widgets --------------------------------------------------------------
class ScoreGauge(QWidget):
    """Horizontal bar showing the image score against the decision threshold.
    The bar spans 0 .. 2x threshold; the tick marks the threshold and the
    fill turns red once the score crosses it."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(54)
        self.score = None
        self.threshold = None

    def set_values(self, score, threshold) -> None:
        self.score = score
        self.threshold = threshold
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width, height = self.width(), self.height()
        bar = QRectF(0, 26, width, 12)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(BORDER)))
        painter.drawRoundedRect(bar, 6, 6)

        if self.score is None or self.threshold is None:
            painter.end()
            return

        full_scale = 2.0 * self.threshold
        fraction = min(max(self.score / full_scale, 0.0), 1.0)
        over = self.score >= self.threshold
        near = abs(self.score - self.threshold) <= BORDERLINE_BAND * self.threshold
        colour = QColor(BAD_COLOR if over else (WARN_COLOR if near else OK_COLOR))
        painter.setBrush(QBrush(colour))
        painter.drawRoundedRect(QRectF(0, 26, max(width * fraction, 12), 12), 6, 6)

        tick_x = width / 2
        painter.setPen(QPen(QColor(TEXT), 2))
        painter.drawLine(int(tick_x), 18, int(tick_x), 46)
        painter.setPen(QColor(MUTED))
        font = QFont("Segoe UI", 8)
        painter.setFont(font)
        painter.drawText(int(tick_x) - 40, 0, 80, 16, Qt.AlignCenter, "threshold")
        painter.drawText(0, 42, 60, 12, Qt.AlignLeft, "0")
        painter.end()


class Worker(QThread):
    """Runs fn(progress_callback) off the GUI thread."""

    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        try:
            with torch.no_grad():
                result = self._fn(lambda done, total, text: self.progress.emit(done, total, text))
            self.succeeded.emit(result)
        except Exception as error:  # surfaced to the user in a dialog
            self.failed.emit(f"{type(error).__name__}: {error}")


def make_card() -> QFrame:
    card = QFrame()
    card.setObjectName("card")
    return card


def section_label(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setObjectName("sectionTitle")
    return label


class AnomalyDetectorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visual Inspection Station -- MVTec Anomaly Detector")
        self.resize(1360, 880)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_image_path = None
        # Caches loaded checkpoints by (model_type, category) so re-running
        # on the same model doesn't reload it from disk every click.
        self.loaded_models = {}
        # Calibration stats keyed by (model_type, category, ddim_steps).
        self.calibrations = {}
        self.worker = None
        self.last_result = None
        self.history_rows = []  # dicts, mirrors the history table (for CSV export)
        # Kept alive on self, not a local variable in _on_start_training_clicked:
        # if the only reference to the running QProcess were a local
        # variable, Python could garbage-collect it (and Qt would kill the
        # process) the moment that function returns, well before training
        # actually finishes.
        self.training_process = None
        self.training_model_type = None  # remembered so _on_training_process_finished
        self.training_category = None    # knows which checkpoint just finished
        # Accumulated (epoch, train_loss, val_loss) history for the live
        # plot on the Train tab -- redrawn from scratch on every new epoch.
        self.train_epoch_history = []
        self.train_loss_history = []
        self.val_loss_history = []

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(10)

        layout.addWidget(self._build_header())

        tabs = QTabWidget()
        detect_tab = QWidget()
        self._build_detect_tab(detect_tab)
        tabs.addTab(detect_tab, "Inspect")
        train_tab = QWidget()
        self._build_train_tab(train_tab)
        tabs.addTab(train_tab, "Train")
        layout.addWidget(tabs, 1)

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("header")
        row = QHBoxLayout(header)
        row.setContentsMargins(16, 10, 16, 10)

        title_box = QVBoxLayout()
        title = QLabel("Visual Inspection Station")
        title.setObjectName("h1")
        subtitle = QLabel("Unsupervised surface-defect detection: AE / VAE / Probabilistic AE+VAE / DDIM")
        subtitle.setObjectName("muted")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        row.addLayout(title_box)
        row.addStretch(1)

        return header

    def _build_detect_tab(self, tab_widget: QWidget) -> None:
        outer = QHBoxLayout(tab_widget)
        outer.setContentsMargins(0, 8, 0, 0)
        outer.setSpacing(12)
        outer.addWidget(self._build_sidebar())

        main_column = QVBoxLayout()
        main_column.setSpacing(10)
        main_column.addWidget(self._build_verdict_card())
        main_column.addLayout(self._build_images_row())
        main_column.addWidget(self._build_metrics_card())
        main_column.addWidget(self._build_history_card(), 1)
        outer.addLayout(main_column, 1)

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(270)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(6)

        layout.addWidget(section_label("Model"))
        self.category_combo = QComboBox()
        self.category_combo.addItems(CATEGORIES)
        layout.addWidget(self._labelled("Product category", self.category_combo))

        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_CHOICES)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        layout.addWidget(self._labelled("Detector", self.model_combo))

        self.ddim_steps_spinbox = QSpinBox()
        self.ddim_steps_spinbox.setRange(1, 200)
        self.ddim_steps_spinbox.setValue(25)
        self.ddim_steps_spinbox.setEnabled(False)  # only meaningful when Model = DDIM
        self.ddim_steps_spinbox.setToolTip(
            "Fewer steps = faster but noisier reconstruction (see the Phase 3 step-count sweep)."
        )
        layout.addWidget(self._labelled("DDIM steps", self.ddim_steps_spinbox))

        layout.addSpacing(8)
        layout.addWidget(section_label("Decision"))
        self.k_slider = QSlider(Qt.Horizontal)
        self.k_slider.setRange(5, 60)  # k = value / 10
        self.k_slider.setValue(int(DEFAULT_K * 10))
        self.k_slider.valueChanged.connect(self._on_sensitivity_changed)
        self.k_label = QLabel()
        self.k_label.setObjectName("muted")
        self.k_label.setWordWrap(True)
        layout.addWidget(self.k_label)
        layout.addWidget(self.k_slider)
        hint = QLabel("Lower: flags more parts  |  Higher: flags fewer")
        hint.setObjectName("muted")
        hint.setStyleSheet("font-size: 10px;")
        layout.addWidget(hint)
        self._on_sensitivity_changed()

        layout.addSpacing(8)
        layout.addWidget(section_label("Input"))
        self.load_image_button = QPushButton("Load image…")
        self.load_image_button.clicked.connect(self._on_load_image_clicked)
        layout.addWidget(self.load_image_button)

        self.run_button = QPushButton("Run inspection")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self._on_run_clicked)
        self.run_button.setEnabled(False)  # enabled once an image is loaded
        layout.addWidget(self.run_button)

        self.folder_button = QPushButton("Inspect folder…")
        self.folder_button.clicked.connect(self._on_inspect_folder_clicked)
        layout.addWidget(self.folder_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        layout.addStretch(1)
        self.status_label = QLabel(f"Running on {self.device}. Load an image to begin.")
        self.status_label.setObjectName("muted")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        return sidebar

    @staticmethod
    def _labelled(text: str, widget: QWidget) -> QWidget:
        box = QWidget()
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(0, 2, 0, 2)
        box_layout.setSpacing(3)
        label = QLabel(text)
        label.setObjectName("muted")
        box_layout.addWidget(label)
        box_layout.addWidget(widget)
        return box

    def _build_verdict_card(self) -> QFrame:
        card = make_card()
        layout = QHBoxLayout(card)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(20)

        self.verdict_badge = QLabel("NO RESULT")
        self.verdict_badge.setAlignment(Qt.AlignCenter)
        self.verdict_badge.setMinimumWidth(260)
        self.verdict_badge.setMinimumHeight(64)
        layout.addWidget(self.verdict_badge)

        detail_box = QVBoxLayout()
        self.verdict_detail = QLabel("Load an image and run an inspection.")
        self.verdict_detail.setStyleSheet("font-size: 14px;")
        self.verdict_detail.setWordWrap(True)
        self.verdict_truth = QLabel("")
        self.verdict_truth.setObjectName("muted")
        self.gauge = ScoreGauge()
        detail_box.addWidget(self.verdict_detail)
        detail_box.addWidget(self.gauge)
        detail_box.addWidget(self.verdict_truth)
        layout.addLayout(detail_box, 1)

        self._style_badge("NO RESULT", IDLE_COLOR)
        return card

    def _style_badge(self, text: str, colour: str) -> None:
        self.verdict_badge.setText(text)
        self.verdict_badge.setStyleSheet(
            f"background: {colour}; color: #ffffff; font-size: 22px; font-weight: 700; "
            f"letter-spacing: 1px; border-radius: 10px; padding: 10px 18px;"
        )

    def _build_images_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)
        self.input_image_label = self._image_panel(row, "Input")
        self.reconstruction_image_label = self._image_panel(row, "Reconstruction")
        self.anomaly_map_image_label = self._image_panel(row, "Anomaly map")
        self.overlay_image_label = self._image_panel(row, "Overlay")
        return row

    def _image_panel(self, row: QHBoxLayout, title: str) -> QLabel:
        card = make_card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        title_label = QLabel(title.upper())
        title_label.setObjectName("panelTitle")
        image_label = QLabel()
        image_label.setObjectName("imagePanel")
        image_label.setFixedSize(PANEL_SIZE, PANEL_SIZE)
        image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        layout.addWidget(image_label, 0, Qt.AlignCenter)
        row.addWidget(card)
        return image_label

    def _build_metrics_card(self) -> QFrame:
        card = make_card()
        grid = QGridLayout(card)
        grid.setContentsMargins(18, 12, 18, 12)
        self.metric_labels = {}
        names = [
            ("score", "Image score (top-1 %)"),
            ("threshold", "Decision threshold"),
            ("margin", "Score / threshold"),
            ("mean", "Mean pixel score"),
            ("max", "Max pixel score"),
            ("time", "Inference time"),
        ]
        for column, (key, name) in enumerate(names):
            value = QLabel("—")
            value.setObjectName("metricValue")
            caption = QLabel(name)
            caption.setObjectName("metricName")
            grid.addWidget(value, 0, column)
            grid.addWidget(caption, 1, column)
            self.metric_labels[key] = value
        return card

    def _build_history_card(self) -> QFrame:
        card = make_card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 10, 14, 12)

        top = QHBoxLayout()
        top.addWidget(section_label("Inspection log"))
        top.addStretch(1)
        self.summary_label = QLabel("")
        self.summary_label.setObjectName("muted")
        top.addWidget(self.summary_label)
        export_button = QPushButton("Export CSV")
        export_button.clicked.connect(self._on_export_csv_clicked)
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self._on_clear_history_clicked)
        top.addWidget(export_button)
        top.addWidget(clear_button)
        layout.addLayout(top)

        self.history_table = QTableWidget(0, 7)
        self.history_table.setHorizontalHeaderLabels(
            ["Time", "File", "Detector", "Score", "Threshold", "Verdict", "Ground truth"]
        )
        self.history_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.history_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.history_table.setMinimumHeight(170)
        self.history_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.history_table)
        return card

    def _build_train_tab(self, tab_widget: QWidget) -> None:
        main_layout = QVBoxLayout(tab_widget)
        main_layout.setContentsMargins(0, 8, 0, 0)
        main_layout.setSpacing(10)

        controls_card = make_card()
        controls_card_layout = QVBoxLayout(controls_card)
        controls_card_layout.setContentsMargins(16, 14, 16, 14)
        controls_card_layout.addLayout(self._build_train_controls_row())
        main_layout.addWidget(controls_card)

        # A matplotlib Figure embedded directly in the Qt window via
        # FigureCanvasQTAgg -- this IS the GUI's "telemetry": a live-updating
        # loss curve, the same information scripts/train.py prints to the
        # console and logs to TensorBoard, but visible inside the app itself
        # rather than requiring a separate tool.
        plot_card = make_card()
        plot_layout = QVBoxLayout(plot_card)
        plot_layout.setContentsMargins(10, 10, 10, 10)
        self.loss_figure = Figure(figsize=(6, 3), facecolor=SURFACE)
        self.loss_axes = self.loss_figure.add_subplot(111)
        self._style_loss_axes()
        self.loss_canvas = FigureCanvas(self.loss_figure)
        plot_layout.addWidget(self.loss_canvas)
        main_layout.addWidget(plot_card, 1)

        self.train_status_label = QLabel(
            f"Running on {self.device}. Configure hyperparameters above, then click Start Training."
        )
        self.train_status_label.setObjectName("muted")
        main_layout.addWidget(self.train_status_label)

    def _style_loss_axes(self) -> None:
        axes = self.loss_axes
        axes.set_facecolor(SURFACE)
        axes.set_xlabel("Epoch", color=MUTED)
        axes.set_ylabel("Loss", color=MUTED)
        axes.tick_params(colors=MUTED)
        for spine in axes.spines.values():
            spine.set_color(BORDER)
        axes.grid(True, color=BORDER, linewidth=0.6, alpha=0.7)

    def _build_train_controls_row(self) -> QHBoxLayout:
        """The Train tab's hyperparameter configuration: model family,
        category, epochs, batch size, and learning rate, all editable
        before starting a run -- this is the GUI's own "hyperparameter
        configuration" surface, independent of the CLI scripts' argparse
        flags."""
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)

        self.train_model_combo = QComboBox()
        self.train_model_combo.addItems(MODEL_CHOICES)
        self.train_model_combo.currentTextChanged.connect(self._on_train_model_changed)
        controls_layout.addWidget(self._labelled("Model", self.train_model_combo))

        self.train_category_combo = QComboBox()
        self.train_category_combo.addItems(CATEGORIES)
        controls_layout.addWidget(self._labelled("Category", self.train_category_combo))

        self.train_epochs_spinbox = QSpinBox()
        self.train_epochs_spinbox.setRange(1, 2000)
        self.train_epochs_spinbox.setValue(100)
        self.train_epochs_spinbox.setSingleStep(10)
        controls_layout.addWidget(self._labelled("Epochs", self.train_epochs_spinbox))

        self.train_batch_size_spinbox = QSpinBox()
        self.train_batch_size_spinbox.setRange(1, 128)
        self.train_batch_size_spinbox.setValue(32)
        controls_layout.addWidget(self._labelled("Batch size", self.train_batch_size_spinbox))

        self.train_lr_spinbox = QDoubleSpinBox()
        self.train_lr_spinbox.setDecimals(5)
        self.train_lr_spinbox.setRange(0.00001, 1.0)
        self.train_lr_spinbox.setSingleStep(0.0001)
        self.train_lr_spinbox.setValue(0.001)
        controls_layout.addWidget(self._labelled("Learning rate", self.train_lr_spinbox))

        controls_layout.addStretch(1)
        self.start_training_button = QPushButton("Start training")
        self.start_training_button.setObjectName("primary")
        self.start_training_button.clicked.connect(self._on_start_training_clicked)
        controls_layout.addWidget(self.start_training_button, 0, Qt.AlignBottom)

        return controls_layout

    # --------------------------------------------------------- UI callbacks
    @property
    def sensitivity_k(self) -> float:
        return self.k_slider.value() / 10.0

    def _on_sensitivity_changed(self) -> None:
        self.k_label.setText(f"Threshold = mean + {self.sensitivity_k:.1f} × std of normal parts")
        if getattr(self, "last_result", None) is not None:
            self._refresh_verdict()

    def _on_model_changed(self, model_text: str) -> None:
        self.ddim_steps_spinbox.setEnabled(model_text == "DDIM")

    def _on_train_model_changed(self, model_text: str) -> None:
        """DDIM needs a smaller batch size and lower learning rate than
        AE/VAE (see the --batch-size help text in scripts/train_diffusion.py:
        the UNet's skip connections and attention keep far more activations
        in GPU memory at once than the plain AE/VAE conv stack does).
        Switching the model here updates the defaults so a user doesn't have
        to already know that to avoid the out-of-memory slowdown we hit
        ourselves during Phase 3."""
        if model_text == "DDIM":
            self.train_batch_size_spinbox.setValue(8)
            self.train_lr_spinbox.setValue(0.0002)
        else:
            self.train_batch_size_spinbox.setValue(32)
            self.train_lr_spinbox.setValue(0.001)

    def _on_load_image_clicked(self) -> None:
        category = self.category_combo.currentText()
        default_dir = str(REPO_ROOT / "data" / category / "test")
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self, "Select an image to inspect", default_dir, "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_path:
            return

        self.current_image_path = file_path
        self.run_button.setEnabled(True)
        self.status_label.setText(f"Loaded: {Path(file_path).name}")
        self.input_image_label.setPixmap(array_to_pixmap(np.array(Image.open(file_path).convert("RGB"))))

    def _set_busy(self, busy: bool, text: str = "") -> None:
        for widget in (self.run_button, self.load_image_button, self.folder_button,
                       self.category_combo, self.model_combo):
            widget.setEnabled(not busy)
        if not busy:
            self.run_button.setEnabled(self.current_image_path is not None)
            self.ddim_steps_spinbox.setEnabled(self.model_combo.currentText() == "DDIM")
        else:
            self.ddim_steps_spinbox.setEnabled(False)
        self.progress_bar.setVisible(busy)
        self.progress_bar.setRange(0, 0)  # indeterminate until progress arrives
        if text:
            self.status_label.setText(text)

    def _on_progress(self, done: int, total: int, text: str) -> None:
        self.progress_bar.setRange(0, max(total, 1))
        self.progress_bar.setValue(done)
        self.status_label.setText(text)

    # ----------------------------------------------------------- inference
    def _get_model(self, model_type: str, category: str):
        """Loads a checkpoint from disk the first time it's asked for, then
        returns the cached copy on every later call for the same
        (model_type, category) pair."""
        cache_key = (model_type, category)
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]

        checkpoint_path = REPO_ROOT / "checkpoints" / checkpoint_folder(model_type, category) / "model.pt"
        if not checkpoint_path.exists():
            train_model, _prefix, tag, _score_mode = MODEL_SPECS[model_type]
            train_script = "train_diffusion.py" if model_type == "ddim" else "train.py"
            model_flag = "" if model_type == "ddim" else f" --model {train_model}"
            tag_flag = f" --tag {tag}" if tag else ""
            raise FileNotFoundError(
                f"No checkpoint found at:\n{checkpoint_path}\n\n"
                f"Train one first with:\npython scripts/{train_script}{model_flag} --category {category}{tag_flag}"
            )

        if model_type == "ddim":
            model, diffusion, image_size = load_checkpoint(checkpoint_path, self.device)
            loaded = (model, diffusion, image_size)
        else:
            model, image_size = load_model(checkpoint_path, self.device)
            # Prob. AE / Prob. VAE: switch the model to its probability score
            score_mode = MODEL_SPECS[model_type][3]
            if score_mode is not None:
                model.score_mode = score_mode
            if hasattr(model, "n_samples"):
                model.n_samples = 1  # L=1 matched L=10 in accuracy (README sweep) and is ~7x cheaper
            loaded = (model, image_size)

        self.loaded_models[cache_key] = loaded
        return loaded

    def _transform_for(self, model_type: str, image_size: int):
        return build_ddim_transform(image_size) if model_type == "ddim" else build_ae_vae_transform(image_size)

    def _infer(self, model_type: str, category: str, image_tensor: torch.Tensor, ddim_steps: int):
        """Returns (anomaly_map, reconstruction) for one preprocessed image."""
        if model_type == "ddim":
            model, diffusion, _size = self.loaded_models[(model_type, category)]
            return diffusion.anomaly_map(model, image_tensor, t_start=DEFAULT_DDIM_T_START, ddim_steps=ddim_steps)
        model, _size = self.loaded_models[(model_type, category)]
        return model.anomaly_map(image_tensor)

    def _calibration_path(self, model_type: str, category: str) -> Path:
        return REPO_ROOT / "checkpoints" / checkpoint_folder(model_type, category) / "gui_calibration.json"

    def _ensure_calibration(self, model_type: str, category: str, ddim_steps: int, progress) -> dict:
        """Scores held-out NORMAL validation images to learn what a normal
        part looks like to this model, and caches the result. Runs on the
        worker thread."""
        steps_key = ddim_steps if model_type == "ddim" else 0
        key = (model_type, category, steps_key)
        if key in self.calibrations:
            return self.calibrations[key]

        checkpoint_path = REPO_ROOT / "checkpoints" / checkpoint_folder(model_type, category) / "model.pt"
        stamp = checkpoint_path.stat().st_mtime
        cache_path = self._calibration_path(model_type, category)
        try:
            cached = json.loads(cache_path.read_text())
            if (cached.get("version") == CALIBRATION_VERSION and cached.get("checkpoint_mtime") == stamp
                    and cached.get("ddim_steps") == steps_key and cached.get("top_fraction") == TOP_FRACTION):
                self.calibrations[key] = cached
                return cached
        except (OSError, ValueError):
            pass

        image_size = self.loaded_models[(model_type, category)][-1]
        dataset = MVTecTrainDataset(REPO_ROOT / "data", category, image_size,
                                    transform=self._transform_for(model_type, image_size))
        _train_subset, val_subset = train_val_split(dataset, val_fraction=0.1, seed=42)
        indices = list(range(min(len(val_subset), MAX_CALIBRATION_IMAGES)))

        scores, map_maxima = [], []
        for done, index in enumerate(indices):
            progress(done, len(indices), f"Calibrating on normal reference images… {done + 1}/{len(indices)}")
            image_tensor = val_subset[index].unsqueeze(0).to(self.device)
            anomaly_map, _reconstruction = self._infer(model_type, category, image_tensor, ddim_steps)
            scores.append(top_fraction_score(anomaly_map))
            map_maxima.append(anomaly_map.max().item())

        calibration = {
            "version": CALIBRATION_VERSION,
            "checkpoint_mtime": stamp,
            "ddim_steps": steps_key,
            "top_fraction": TOP_FRACTION,
            "n_images": len(scores),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "map_vmax": float(np.max(map_maxima)),
        }
        try:
            cache_path.write_text(json.dumps(calibration, indent=2))
        except OSError:
            pass  # cache is an optimisation only
        self.calibrations[key] = calibration
        return calibration

    @staticmethod
    def _threshold(calibration: dict, k: float) -> float:
        # Floor the spread so a tiny calibration set can't give a razor-thin margin.
        spread = max(calibration["std"], 0.05 * abs(calibration["mean"]))
        return calibration["mean"] + k * spread

    def _inspect_one(self, model_type: str, category: str, path: str, ddim_steps: int, calibration: dict) -> dict:
        _model_entry_size = self.loaded_models[(model_type, category)][-1]
        transform = self._transform_for(model_type, _model_entry_size)
        image = Image.open(path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(self.device)

        start_time = time.perf_counter()
        anomaly_map, reconstruction = self._infer(model_type, category, image_tensor, ddim_steps)
        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        flat_map = anomaly_map.flatten()
        input_rgb = tensor_to_rgb(image_tensor.squeeze(0))
        normalised = normalise_map(anomaly_map, calibration["map_vmax"] * 1.5)
        return {
            "path": path,
            "model_type": model_type,
            "category": category,
            "score": top_fraction_score(anomaly_map),
            "mean_score": flat_map.mean().item(),
            "max_score": flat_map.max().item(),
            "elapsed_ms": elapsed_ms,
            "calibration": calibration,
            "input_rgb": input_rgb,
            "recon_rgb": tensor_to_rgb(reconstruction.squeeze(0)),
            "heat_rgb": heatmap_rgb(normalised),
            "overlay_rgb": overlay_rgb(input_rgb, normalised),
            "truth": ground_truth_from_path(Path(path)),
        }

    def _start_worker(self, fn, on_success, busy_text: str) -> None:
        self._set_busy(True, busy_text)
        self.worker = Worker(fn)
        self.worker.progress.connect(self._on_progress)
        self.worker.succeeded.connect(on_success)
        self.worker.failed.connect(self._on_worker_failed)
        self.worker.start()

    def _on_worker_failed(self, message: str) -> None:
        self._set_busy(False, "Inspection failed.")
        QMessageBox.critical(self, "Inspection failed", message)

    def _prepare(self):
        """Common setup for single/batch runs on the GUI thread. Returns
        (model_type, category, ddim_steps) or None if the checkpoint is missing."""
        category = self.category_combo.currentText()
        model_type = self.model_combo.currentText().lower()
        try:
            self._get_model(model_type, category)
        except FileNotFoundError as error:
            QMessageBox.warning(self, "Checkpoint missing", str(error))
            return None
        return model_type, category, self.ddim_steps_spinbox.value()

    def _on_run_clicked(self) -> None:
        prepared = self._prepare()
        if prepared is None:
            return
        model_type, category, ddim_steps = prepared
        path = self.current_image_path

        def job(progress):
            calibration = self._ensure_calibration(model_type, category, ddim_steps, progress)
            progress(0, 1, "Inspecting…")
            return self._inspect_one(model_type, category, path, ddim_steps, calibration)

        self._start_worker(job, self._on_single_result, "Preparing detector…")

    def _on_single_result(self, result: dict) -> None:
        self._set_busy(False, f"Done: {Path(result['path']).name}")
        self.last_result = result
        self.input_image_label.setPixmap(array_to_pixmap(result["input_rgb"]))
        self.reconstruction_image_label.setPixmap(array_to_pixmap(result["recon_rgb"]))
        self.anomaly_map_image_label.setPixmap(array_to_pixmap(result["heat_rgb"]))
        self.overlay_image_label.setPixmap(array_to_pixmap(result["overlay_rgb"]))
        verdict = self._refresh_verdict()
        self._append_history(result, verdict)

    def _on_inspect_folder_clicked(self) -> None:
        category = self.category_combo.currentText()
        folder = QFileDialog.getExistingDirectory(
            self, "Select a folder of images to inspect", str(REPO_ROOT / "data" / category / "test")
        )
        if not folder:
            return
        paths = sorted(str(p) for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not paths:
            QMessageBox.information(self, "No images", "That folder contains no image files.")
            return
        prepared = self._prepare()
        if prepared is None:
            return
        model_type, category, ddim_steps = prepared
        k = self.sensitivity_k

        def job(progress):
            calibration = self._ensure_calibration(model_type, category, ddim_steps, progress)
            results = []
            for done, path in enumerate(paths):
                progress(done, len(paths), f"Inspecting {done + 1}/{len(paths)}: {Path(path).name}")
                results.append(self._inspect_one(model_type, category, path, ddim_steps, calibration))
            return results

        def on_done(results):
            self._set_busy(False, f"Inspected {len(results)} images.")
            for result in results:
                self._append_history(result, self._verdict_for(result, k))
            if results:
                last = results[-1]
                self.last_result = last
                self.input_image_label.setPixmap(array_to_pixmap(last["input_rgb"]))
                self.reconstruction_image_label.setPixmap(array_to_pixmap(last["recon_rgb"]))
                self.anomaly_map_image_label.setPixmap(array_to_pixmap(last["heat_rgb"]))
                self.overlay_image_label.setPixmap(array_to_pixmap(last["overlay_rgb"]))
                self._refresh_verdict()

        self._start_worker(job, on_done, "Preparing detector…")

    # ------------------------------------------------------------- verdict
    def _verdict_for(self, result: dict, k: float) -> dict:
        threshold = self._threshold(result["calibration"], k)
        score = result["score"]
        is_anomaly = score >= threshold
        borderline = abs(score - threshold) <= BORDERLINE_BAND * threshold
        return {"threshold": threshold, "anomaly": is_anomaly, "borderline": borderline, "ratio": score / threshold}

    def _refresh_verdict(self) -> dict:
        result = self.last_result
        verdict = self._verdict_for(result, self.sensitivity_k)
        score, threshold, ratio = result["score"], verdict["threshold"], verdict["ratio"]

        if verdict["anomaly"]:
            self._style_badge("ANOMALY DETECTED", BAD_COLOR)
            detail = (f"Image score {score:.4f} is {ratio:.2f}× the decision threshold ({threshold:.4f}). "
                      "The model could not reproduce this part as a normal one.")
            if verdict["borderline"]:
                detail += " Borderline: consider a manual check."
        else:
            self._style_badge("NORMAL", OK_COLOR if not verdict["borderline"] else WARN_COLOR)
            detail = (f"Image score {score:.4f} is below the decision threshold ({threshold:.4f}); "
                      f"the part is reproduced like a normal one ({ratio:.2f}× threshold).")
            if verdict["borderline"]:
                detail += " Borderline: close to the threshold."
        self.verdict_detail.setText(detail)
        self.gauge.set_values(score, threshold)

        truth = result["truth"]
        if truth == "unknown":
            self.verdict_truth.setText("Ground truth: unknown (image not from the MVTec test folders)")
        else:
            correct = (truth == "defect") == verdict["anomaly"]
            self.verdict_truth.setText(
                f"Ground truth (from folder name): {truth}  —  "
                + ("model is CORRECT" if correct else "model is WRONG on this image")
            )

        self.metric_labels["score"].setText(f"{score:.4f}")
        self.metric_labels["threshold"].setText(f"{threshold:.4f}")
        self.metric_labels["margin"].setText(f"{ratio:.2f}×")
        self.metric_labels["mean"].setText(f"{result['mean_score']:.4f}")
        self.metric_labels["max"].setText(f"{result['max_score']:.4f}")
        self.metric_labels["time"].setText(f"{result['elapsed_ms']:.1f} ms")
        return verdict

    # ------------------------------------------------------------- history
    def _append_history(self, result: dict, verdict: dict) -> None:
        row = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "file": Path(result["path"]).name,
            "path": result["path"],
            "detector": f"{result['model_type'].upper()} / {result['category']}",
            "score": result["score"],
            "threshold": verdict["threshold"],
            "verdict": "ANOMALY" if verdict["anomaly"] else "NORMAL",
            "truth": result["truth"],
        }
        self.history_rows.append(row)

        table = self.history_table
        index = table.rowCount()
        table.insertRow(index)
        correct = None if row["truth"] == "unknown" else ((row["truth"] == "defect") == verdict["anomaly"])
        truth_text = row["truth"] if correct is None else f"{row['truth']}  {'(correct)' if correct else '(wrong)'}"
        cells = [row["time"], row["file"], row["detector"], f"{row['score']:.4f}",
                 f"{row['threshold']:.4f}", row["verdict"], truth_text]
        for column, text in enumerate(cells):
            item = QTableWidgetItem(text)
            if column == 5:
                item.setForeground(QBrush(QColor(BAD_COLOR if verdict["anomaly"] else OK_COLOR)))
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            if column == 6 and correct is not None:
                item.setForeground(QBrush(QColor(OK_COLOR if correct else BAD_COLOR)))
            table.setItem(index, column, item)
        table.scrollToBottom()
        self._update_summary()

    def _update_summary(self) -> None:
        total = len(self.history_rows)
        flagged = sum(1 for r in self.history_rows if r["verdict"] == "ANOMALY")
        text = f"{total} inspected  |  {flagged} anomalies  |  {total - flagged} normal"
        labelled = [r for r in self.history_rows if r["truth"] != "unknown"]
        if labelled:
            correct = sum(1 for r in labelled if (r["truth"] == "defect") == (r["verdict"] == "ANOMALY"))
            text += f"  |  accuracy vs folder labels {correct}/{len(labelled)}"
        self.summary_label.setText(text)

    def _on_clear_history_clicked(self) -> None:
        self.history_rows = []
        self.history_table.setRowCount(0)
        self._update_summary()

    def _on_export_csv_clicked(self) -> None:
        if not self.history_rows:
            QMessageBox.information(self, "Nothing to export", "Run an inspection first.")
            return
        file_path, _filter = QFileDialog.getSaveFileName(
            self, "Export inspection log", str(REPO_ROOT / "inspection_log.csv"), "CSV (*.csv)"
        )
        if not file_path:
            return
        with open(file_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.history_rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.history_rows)
        self.status_label.setText(f"Exported {len(self.history_rows)} rows to {Path(file_path).name}")

    # ------------------------------------------------------------ training
    def _on_start_training_clicked(self) -> None:
        model_type = self.train_model_combo.currentText().lower()
        category = self.train_category_combo.currentText()
        epochs = self.train_epochs_spinbox.value()
        batch_size = self.train_batch_size_spinbox.value()
        lr = self.train_lr_spinbox.value()

        # Build the exact command line a user would type themselves --
        # scripts/train.py for AE/VAE, scripts/train_diffusion.py for DDIM
        # (it doesn't take --model since there's only one model it trains).
        if model_type == "ddim":
            script_path = SCRIPTS_DIR / "train_diffusion.py"
            script_args = ["--category", category]
        else:
            train_model, _prefix, tag, _score_mode = MODEL_SPECS[model_type]
            script_path = SCRIPTS_DIR / "train.py"
            script_args = ["--model", train_model, "--category", category]
            if tag:
                script_args += ["--tag", tag]  # e.g. the tuned VAE saves to vae_<category>_kl1e-6
        script_args += [
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--lr", str(lr),
        ]

        # Remember which checkpoint this run will produce, so
        # _on_training_process_finished knows what to invalidate/report
        # without needing to re-read the (possibly since-changed) combo boxes.
        self.training_model_type = model_type
        self.training_category = category

        # Clear any previous run's curve before starting a fresh one.
        self.train_epoch_history = []
        self.train_loss_history = []
        self.val_loss_history = []
        self.loss_axes.clear()
        self._style_loss_axes()
        self.loss_canvas.draw()

        self.start_training_button.setEnabled(False)
        self.train_status_label.setText(
            f"Starting: {model_type.upper()} on {category}, {epochs} epochs, "
            f"batch_size={batch_size}, lr={lr}..."
        )

        # QProcess launches scripts/train*.py as a completely separate OS
        # process (using this same Python interpreter, sys.executable) and
        # integrates with Qt's own event loop -- readyReadStandardOutput
        # fires whenever the process prints something, and finished fires
        # when it exits, both delivered safely on the GUI thread with no
        # manual threading code needed at all.
        self.training_process = QProcess(self)
        self.training_process.setProgram(sys.executable)
        self.training_process.setArguments([str(script_path)] + script_args)
        self.training_process.setWorkingDirectory(str(REPO_ROOT))
        self.training_process.readyReadStandardOutput.connect(self._on_training_output_ready)
        self.training_process.finished.connect(self._on_training_process_finished)
        self.training_process.start()

    def _on_training_output_ready(self) -> None:
        """Called whenever scripts/train*.py prints a new line of output.
        We read whatever text has arrived so far, decode it from raw bytes
        to a string, and check each line for the "[epoch N/M] ..." pattern
        the training scripts print every 10 epochs."""
        output = bytes(self.training_process.readAllStandardOutput()).decode(errors="replace")
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue

            match = EPOCH_LINE_PATTERN.search(line)
            if match:
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                val_loss = float(match.group(3))
                self.train_epoch_history.append(epoch)
                self.train_loss_history.append(train_loss)
                self.val_loss_history.append(val_loss)

                self.loss_axes.clear()
                self._style_loss_axes()
                self.loss_axes.plot(self.train_epoch_history, self.train_loss_history, label="train loss", color=ACCENT)
                self.loss_axes.plot(self.train_epoch_history, self.val_loss_history, label="val loss", color=WARN_COLOR)
                legend = self.loss_axes.legend(facecolor=SURFACE_2, edgecolor=BORDER)
                for legend_text in legend.get_texts():
                    legend_text.set_color(TEXT)
                self.loss_canvas.draw()

            self.train_status_label.setText(line)

    def _on_training_process_finished(self, exit_code: int, _exit_status) -> None:
        self.start_training_button.setEnabled(True)

        if exit_code == 0:
            # The checkpoint on disk for this (model_type, category) just
            # changed. If the Inspect tab already has the OLD version cached
            # in self.loaded_models, drop it now so the next inspection
            # reloads the freshly-trained weights (and re-calibrates against
            # them) instead of silently reusing the stale in-memory model.
            self.loaded_models.pop((self.training_model_type, self.training_category), None)
            for key in [k for k in self.calibrations if k[:2] == (self.training_model_type, self.training_category)]:
                del self.calibrations[key]
            message = f"Training complete for {self.training_model_type.upper()}/{self.training_category}."
            self.train_status_label.setText(message)
            QMessageBox.information(self, "Training complete", message)
        else:
            message = f"Training process exited with code {exit_code}. See the status line above for its last output."
            self.train_status_label.setText(message)
            QMessageBox.critical(self, "Training failed", message)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    window = AnomalyDetectorWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
