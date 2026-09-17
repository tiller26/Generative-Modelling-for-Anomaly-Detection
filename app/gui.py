"""PyQt5 desktop prototype for the anomaly detection benchmark.

Pick a category and a trained model, load a test image, click "Run
Detection", and see the model's reconstruction and anomaly heatmap. This is
the interactive prototype -- separate from scripts/train*.py and
scripts/evaluate*.py, which are the batch tools that produced the numbers
and figures in the paper.

Run with:
    python app/gui.py

IMPORTANT: `import torch` must come before any `import PyQt5...` line below.
On Windows, importing PyQt5 first loads a conflicting copy of a shared DLL
that then makes torch's own DLL loading crash with a cryptic
"WinError 1114" -- confirmed by testing both import orders directly. Do not
reorder or alphabetize these imports.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch  # must be imported before PyQt5 -- see warning above
from matplotlib import cm
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
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

CATEGORIES = ["bottle", "capsule", "pill"]
DEFAULT_DDIM_T_START = 250
PANEL_SIZE = 256  # how big each image panel looks on screen


def tensor_to_pixmap(image_tensor: torch.Tensor) -> QPixmap:
    """Converts a [-1, 1]-scaled (channels, height, width) tensor -- the
    format every model in this project uses -- into a QPixmap a QLabel can
    display."""
    image_0_to_1 = (image_tensor.clamp(-1, 1) + 1) / 2
    image_uint8 = (image_0_to_1.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    image_uint8 = np.ascontiguousarray(image_uint8)
    height, width, _channels = image_uint8.shape
    qimage = QImage(image_uint8.data, width, height, 3 * width, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage).scaled(PANEL_SIZE, PANEL_SIZE, Qt.KeepAspectRatio)


def anomaly_map_to_pixmap(anomaly_map: torch.Tensor) -> QPixmap:
    """Converts a single-channel anomaly map into a QPixmap using the same
    'inferno' colour map as the paper's qualitative figures, so bright =
    more anomalous. Normalised to its own min/max, since raw anomaly scores
    are on a different scale for every model."""
    map_np = anomaly_map.squeeze().detach().cpu().numpy()
    normalized = (map_np - map_np.min()) / (map_np.max() - map_np.min() + 1e-8)
    colored = (cm.inferno(normalized) * 255).astype(np.uint8)  # (H, W, 4) RGBA
    colored = np.ascontiguousarray(colored)
    height, width, _channels = colored.shape
    qimage = QImage(colored.data, width, height, 4 * width, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimage).scaled(PANEL_SIZE, PANEL_SIZE, Qt.KeepAspectRatio)


class AnomalyDetectorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MVTec Anomaly Detector -- AE / VAE / DDIM Prototype")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_image_path = None
        # Caches loaded checkpoints by (model_type, category) so re-running
        # on the same model doesn't reload it from disk every click.
        self.loaded_models = {}

        self._build_ui()

    def _build_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        main_layout.addLayout(self._build_controls_row())
        main_layout.addLayout(self._build_images_row())

        self.status_label = QLabel(f"Running on {self.device}. Load an image to begin.")
        main_layout.addWidget(self.status_label)

    def _build_controls_row(self) -> QHBoxLayout:
        controls_layout = QHBoxLayout()

        controls_layout.addWidget(QLabel("Category:"))
        self.category_combo = QComboBox()
        self.category_combo.addItems(CATEGORIES)
        controls_layout.addWidget(self.category_combo)

        controls_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["AE", "VAE", "DDIM"])
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        controls_layout.addWidget(self.model_combo)

        controls_layout.addWidget(QLabel("DDIM steps:"))
        self.ddim_steps_spinbox = QSpinBox()
        self.ddim_steps_spinbox.setRange(1, 200)
        self.ddim_steps_spinbox.setValue(25)
        self.ddim_steps_spinbox.setEnabled(False)  # only meaningful when Model = DDIM
        self.ddim_steps_spinbox.setToolTip(
            "Fewer steps = faster but noisier reconstruction (see the Phase 3 step-count sweep)."
        )
        controls_layout.addWidget(self.ddim_steps_spinbox)

        self.load_image_button = QPushButton("Load Image...")
        self.load_image_button.clicked.connect(self._on_load_image_clicked)
        controls_layout.addWidget(self.load_image_button)

        self.run_button = QPushButton("Run Detection")
        self.run_button.clicked.connect(self._on_run_clicked)
        self.run_button.setEnabled(False)  # enabled once an image is loaded
        controls_layout.addWidget(self.run_button)

        return controls_layout

    def _build_images_row(self) -> QHBoxLayout:
        images_row = QHBoxLayout()
        input_layout, self.input_image_label = self._build_image_panel("Input")
        reconstruction_layout, self.reconstruction_image_label = self._build_image_panel("Reconstruction")
        anomaly_layout, self.anomaly_map_image_label = self._build_image_panel("Anomaly map")
        images_row.addLayout(input_layout)
        images_row.addLayout(reconstruction_layout)
        images_row.addLayout(anomaly_layout)
        return images_row

    def _build_image_panel(self, title: str):
        layout = QVBoxLayout()
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)

        image_label = QLabel()
        image_label.setFixedSize(PANEL_SIZE, PANEL_SIZE)
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setStyleSheet("border: 1px solid gray;")

        layout.addWidget(title_label)
        layout.addWidget(image_label)
        return layout, image_label

    def _on_model_changed(self, model_text: str) -> None:
        self.ddim_steps_spinbox.setEnabled(model_text == "DDIM")

    def _on_load_image_clicked(self) -> None:
        category = self.category_combo.currentText()
        default_dir = str(REPO_ROOT / "data" / category / "test")
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self, "Select a test image", default_dir, "Images (*.png *.jpg *.jpeg)"
        )
        if not file_path:
            return

        self.current_image_path = file_path
        self.run_button.setEnabled(True)
        self.status_label.setText(f"Loaded: {file_path}")

    def _get_model(self, model_type: str, category: str):
        """Loads a checkpoint from disk the first time it's asked for, then
        returns the cached copy on every later call for the same
        (model_type, category) pair."""
        cache_key = (model_type, category)
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]

        checkpoint_path = REPO_ROOT / "checkpoints" / f"{model_type}_{category}" / "model.pt"
        if not checkpoint_path.exists():
            train_script = "train_diffusion.py" if model_type == "ddim" else "train.py"
            model_flag = "" if model_type == "ddim" else f" --model {model_type}"
            raise FileNotFoundError(
                f"No checkpoint found at:\n{checkpoint_path}\n\n"
                f"Train one first with:\npython scripts/{train_script}{model_flag} --category {category}"
            )

        if model_type == "ddim":
            model, diffusion, image_size = load_checkpoint(checkpoint_path, self.device)
            loaded = (model, diffusion, image_size)
        else:
            model, image_size = load_model(checkpoint_path, self.device)
            loaded = (model, image_size)

        self.loaded_models[cache_key] = loaded
        return loaded

    def _on_run_clicked(self) -> None:
        category = self.category_combo.currentText()
        model_type = self.model_combo.currentText().lower()

        try:
            if model_type == "ddim":
                model, diffusion, image_size = self._get_model(model_type, category)
                transform = build_ddim_transform(image_size)
            else:
                model, image_size = self._get_model(model_type, category)
                transform = build_ae_vae_transform(image_size)
        except FileNotFoundError as error:
            QMessageBox.warning(self, "Checkpoint missing", str(error))
            return

        image = Image.open(self.current_image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(self.device)

        self.status_label.setText("Running... (DDIM can take several seconds per image)")
        QApplication.processEvents()  # repaint the status message before the slow call below blocks the UI

        start_time = time.perf_counter()
        with torch.no_grad():
            if model_type == "ddim":
                ddim_steps = self.ddim_steps_spinbox.value()
                anomaly_map, reconstruction = diffusion.anomaly_map(
                    model, image_tensor, t_start=DEFAULT_DDIM_T_START, ddim_steps=ddim_steps
                )
            else:
                anomaly_map, reconstruction = model.anomaly_map(image_tensor)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        self.input_image_label.setPixmap(tensor_to_pixmap(image_tensor.squeeze(0)))
        self.reconstruction_image_label.setPixmap(tensor_to_pixmap(reconstruction.squeeze(0)))
        self.anomaly_map_image_label.setPixmap(anomaly_map_to_pixmap(anomaly_map))

        mean_score = anomaly_map.mean().item()
        max_score = anomaly_map.max().item()
        self.status_label.setText(
            f"Mean anomaly score: {mean_score:.4f}    Max anomaly score: {max_score:.4f}    "
            f"Inference time: {elapsed_ms:.1f} ms"
        )


def main() -> None:
    app = QApplication(sys.argv)
    window = AnomalyDetectorWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
