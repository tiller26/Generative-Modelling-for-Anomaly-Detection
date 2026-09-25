"""PyQt5 desktop prototype for the anomaly detection benchmark.

Two tabs:
  Detect: pick a category and a trained model, load a test image, click "Run
          Detection", and see the model's reconstruction and anomaly heatmap.
  Train:  configure hyperparameters (epochs, batch size, learning rate) and
          train a new checkpoint, watching the train/val loss curve update
          live as it trains. This is the GUI's own hyperparameter
          configuration + telemetry.

The Train tab launches scripts/train.py or scripts/train_diffusion.py as a
genuine separate process (QProcess) rather than re-implementing the training
loop inside the GUI. This was a deliberate fix, not the original design: an
earlier version ran training on a QThread inside this same process, reusing
the training functions directly -- but a PyTorch DataLoader with
num_workers>0 spawned from a background QThread (rather than the process's
main thread) turned out to be unreliable on Windows, repeatedly hanging in a
worker-crash/respawn loop during testing. Running the CLI script as its own
OS process sidesteps that entirely: the script gets a completely normal
main-thread execution context, identical to a user typing the command
themselves, which is exactly the well-tested code path already used to
produce every checkpoint behind the paper's results table.

Run with:
    python app/gui.py

IMPORTANT: `import torch` must come before any `import PyQt5...` line below.
On Windows, importing PyQt5 first loads a conflicting copy of a shared DLL
that then makes torch's own DLL loading crash with a cryptic
"WinError 1114" -- confirmed by testing both import orders directly. Do not
reorder or alphabetize these imports.
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch  # must be imported before PyQt5 -- see warning above
from matplotlib import cm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
from PyQt5.QtCore import Qt, QProcess
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
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

CATEGORIES = ["bottle", "capsule", "pill"]
DEFAULT_DDIM_T_START = 250
PANEL_SIZE = 256  # how big each image panel looks on screen

# Matches a line like "[epoch 10/100] train_loss=0.0554 val_loss=0.0448 (99s elapsed)",
# printed by both scripts/train.py and scripts/train_diffusion.py every 10
# epochs (and on the final epoch) -- this is what feeds the live loss plot.
EPOCH_LINE_PATTERN = re.compile(r"\[epoch (\d+)/\d+\] train_loss=([\d.eE+-]+) val_loss=([\d.eE+-]+)")


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

    def _build_ui(self) -> None:
        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        detect_tab = QWidget()
        self._build_detect_tab(detect_tab)
        tabs.addTab(detect_tab, "Detect")

        train_tab = QWidget()
        self._build_train_tab(train_tab)
        tabs.addTab(train_tab, "Train")

    def _build_detect_tab(self, tab_widget: QWidget) -> None:
        main_layout = QVBoxLayout(tab_widget)

        main_layout.addLayout(self._build_controls_row())
        main_layout.addLayout(self._build_images_row())

        self.status_label = QLabel(f"Running on {self.device}. Load an image to begin.")
        main_layout.addWidget(self.status_label)

    def _build_train_tab(self, tab_widget: QWidget) -> None:
        main_layout = QVBoxLayout(tab_widget)
        main_layout.addLayout(self._build_train_controls_row())

        # A matplotlib Figure embedded directly in the Qt window via
        # FigureCanvasQTAgg -- this IS the GUI's "telemetry": a live-updating
        # loss curve, the same information scripts/train.py prints to the
        # console and logs to TensorBoard, but visible inside the app itself
        # rather than requiring a separate tool.
        self.loss_figure = Figure(figsize=(6, 3))
        self.loss_axes = self.loss_figure.add_subplot(111)
        self.loss_axes.set_xlabel("Epoch")
        self.loss_axes.set_ylabel("Loss")
        self.loss_canvas = FigureCanvas(self.loss_figure)
        main_layout.addWidget(self.loss_canvas)

        self.train_status_label = QLabel(
            f"Running on {self.device}. Configure hyperparameters above, then click Start Training."
        )
        main_layout.addWidget(self.train_status_label)

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

    def _build_train_controls_row(self) -> QHBoxLayout:
        """The Train tab's hyperparameter configuration: model family,
        category, epochs, batch size, and learning rate, all editable
        before starting a run -- this is the GUI's own "hyperparameter
        configuration" surface, independent of the CLI scripts' argparse
        flags."""
        controls_layout = QHBoxLayout()

        controls_layout.addWidget(QLabel("Model:"))
        self.train_model_combo = QComboBox()
        self.train_model_combo.addItems(["AE", "VAE", "DDIM"])
        self.train_model_combo.currentTextChanged.connect(self._on_train_model_changed)
        controls_layout.addWidget(self.train_model_combo)

        controls_layout.addWidget(QLabel("Category:"))
        self.train_category_combo = QComboBox()
        self.train_category_combo.addItems(CATEGORIES)
        controls_layout.addWidget(self.train_category_combo)

        controls_layout.addWidget(QLabel("Epochs:"))
        self.train_epochs_spinbox = QSpinBox()
        self.train_epochs_spinbox.setRange(1, 2000)
        self.train_epochs_spinbox.setValue(100)
        self.train_epochs_spinbox.setSingleStep(10)
        controls_layout.addWidget(self.train_epochs_spinbox)

        controls_layout.addWidget(QLabel("Batch size:"))
        self.train_batch_size_spinbox = QSpinBox()
        self.train_batch_size_spinbox.setRange(1, 128)
        self.train_batch_size_spinbox.setValue(32)
        controls_layout.addWidget(self.train_batch_size_spinbox)

        controls_layout.addWidget(QLabel("Learning rate:"))
        self.train_lr_spinbox = QDoubleSpinBox()
        self.train_lr_spinbox.setDecimals(5)
        self.train_lr_spinbox.setRange(0.00001, 1.0)
        self.train_lr_spinbox.setSingleStep(0.0001)
        self.train_lr_spinbox.setValue(0.001)
        controls_layout.addWidget(self.train_lr_spinbox)

        self.start_training_button = QPushButton("Start Training")
        self.start_training_button.clicked.connect(self._on_start_training_clicked)
        controls_layout.addWidget(self.start_training_button)

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
            script_path = SCRIPTS_DIR / "train.py"
            script_args = ["--model", model_type, "--category", category]
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
        self.loss_axes.set_xlabel("Epoch")
        self.loss_axes.set_ylabel("Loss")
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
                self.loss_axes.plot(self.train_epoch_history, self.train_loss_history, label="train loss")
                self.loss_axes.plot(self.train_epoch_history, self.val_loss_history, label="val loss")
                self.loss_axes.set_xlabel("Epoch")
                self.loss_axes.set_ylabel("Loss")
                self.loss_axes.legend()
                self.loss_canvas.draw()

            self.train_status_label.setText(line)

    def _on_training_process_finished(self, exit_code: int, _exit_status) -> None:
        self.start_training_button.setEnabled(True)

        if exit_code == 0:
            # The checkpoint on disk for this (model_type, category) just
            # changed. If the Detect tab already has the OLD version cached
            # in self.loaded_models, drop it now so the next "Run Detection"
            # click reloads the freshly-trained weights instead of silently
            # reusing the stale in-memory model.
            self.loaded_models.pop((self.training_model_type, self.training_category), None)
            message = f"Training complete for {self.training_model_type.upper()}/{self.training_category}."
            self.train_status_label.setText(message)
            QMessageBox.information(self, "Training complete", message)
        else:
            message = f"Training process exited with code {exit_code}. See the status line above for its last output."
            self.train_status_label.setText(message)
            QMessageBox.critical(self, "Training failed", message)


def main() -> None:
    app = QApplication(sys.argv)
    window = AnomalyDetectorWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
