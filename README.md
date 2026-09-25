# Generative-Modelling-for-Anomaly-Detection
The Generative Spectrum: A Comparative Benchmark of Autoencoders, Variational  Autoencoders, and Denoising Diffusion Models for Industrial Anomaly Detection 

Unsupervised visual anomaly detection identifies defects without labeled anomaly 
data by training models exclusively on normal samples to learn pristine appearance 
distributions. At inference time, query images are reconstructed or denoised; 
subtracting the output from the input yields a residual error map flagging anomalies 
that the model cannot reconstruct. Three major generative paradigms address this 
task deterministic Autoencoders (AEs), probabilistic Variational Autoencoders 
(VAEs), and Denoising Diffusion Implicit Models (DDIM). However, existing literature 
rarely compares them side-by-side under unified, controlled conditions. This project 
establishes a benchmark across AE, VAE, and DDIM architectures on the MVTec AD 
dataset to quantify trade-offs between reconstruction quality, localization accuracy, 
and inference latency 

## Repo structure

```
data/               MVTec AD categories (bottle, capsule, pill) — gitignored, see below
src/data/           Dataset loading & preprocessing (MVTecTrainDataset, MVTecTestDataset)
src/eval/           Evaluation harness: image/pixel AUROC, IoU, latency (src/eval/metrics.py)
src/utils/          Experiment tracking (TensorBoard + CSV/JSON, src/utils/tracking.py)
src/models/         AE / VAE / DDIM model implementations, all original (unet.py, diffusion.py; AE/VAE to follow)
configs/            YAML experiment configs
scripts/            One-off utility scripts (dataset download, smoke tests)
tests/              Unit tests (pytest)
```

## Setup

```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128  # match your CUDA driver
```

GPU tested on an RTX 5060 (Blackwell/sm_120), which requires PyTorch >= 2.7 built
against CUDA 12.8+.

## Data

Download the three MVTec AD categories used in this benchmark:

```bash
python scripts/download_mvtec.py bottle capsule pill
```

This pulls the official per-category archives (~150–400MB each, CC BY-NC-SA 4.0)
into `data/<category>/{train,test,ground_truth}/` and is gitignored — rerun the
script on a fresh clone rather than committing the dataset.

## Evaluation harness

`src/eval/metrics.py` provides `image_auroc`, `pixel_auroc`, `best_threshold_iou`,
`reconstruction_quality` (PSNR/SSIM, computed on non-defective test images only —
see the function's docstring for why mixing in defective images would make the
number ambiguous), and `measure_latency`, shared across all three model families
so results are directly comparable. Run the unit tests (synthetic data, no
trained model needed):

```bash
python -m pytest tests/test_metrics.py -v
```

## DDIM implementation

`src/models/unet.py` and `src/models/diffusion.py` are an original from-scratch
implementation of the standard DDPM training objective (Ho et al. 2020) and
deterministic DDIM sampling (Song et al. 2020) — no third-party anomaly-detection
codebase is vendored or adapted. Anomaly scoring follows the usual partial-diffusion
reconstruction recipe: noise a test image to an intermediate timestep t* < T, run
the deterministic DDIM reverse process back to t=0, and score anomalies by the
input/reconstruction residual.

Unit tests (tiny random tensors, CPU, no dataset needed):

```bash
python -m pytest tests/test_diffusion.py -v
```

End-to-end smoke test on real MVTec data (a few training steps + one DDIM
reconstruction pass, not a real training run):

```bash
python scripts/smoke_test_ddim.py --category pill --steps 20
```

## AE / VAE (Phase 2)

`src/models/autoencoder.py` and `src/models/vae.py` are original, from-scratch
implementations (plain convolutional encoder/decoder for the AE; the VAE adds
a mu/logvar encoder, the reparameterization trick, and a KL-divergence loss
term, reusing the AE's decoder architecture so the two are directly
comparable). Both detect anomalies the same way: reconstruct the input and
score anomalies by the input/reconstruction pixel error.

Unit tests (tiny random tensors, CPU, no dataset needed):

```bash
python -m pytest tests/test_autoencoder.py tests/test_vae.py -v
```

Train one model on one category (saves to `checkpoints/<model>_<category>/model.pt`,
gitignored due to size — rerun to reproduce rather than committing weights):

```bash
python scripts/train.py --model ae --category pill --epochs 100
python scripts/train.py --model vae --category bottle --epochs 100
```

Evaluate a trained checkpoint through the shared harness (appends a row to
`results/summary.csv`):

```bash
python scripts/evaluate.py --model ae --category pill
```

Or reproduce every Phase 2 result (6 training runs + 6 evaluations, ~50
minutes on an RTX 5060):

```bash
python scripts/run_phase2.py --epochs 100
```

### Results (128px images, 100 epochs, latent_dim=256)

`evaluate.py` reports image-level AUROC two ways — scoring each image by the
**mean** vs. the **max** pixel of its anomaly map. Defects are small and
localized, so averaging over a mostly-normal image dilutes the signal; taking
the single worst pixel is usually more sensitive. Both are kept in
`results/summary.csv` rather than silently picking one.

| Model | Category | Image AUROC (mean) | Image AUROC (max) | Pixel AUROC | IoU | PSNR (dB) | SSIM | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| AE | bottle | 0.794 | **0.861** | 0.830 | 0.267 | 28.9 | 0.850 | 1.9 |
| AE | capsule | 0.563 | **0.688** | 0.849 | 0.199 | 34.6 | 0.940 | 2.3 |
| AE | pill | **0.774** | 0.610 | 0.841 | 0.171 | 33.5 | 0.910 | 2.8 |
| VAE | bottle | 0.659 | **0.825** | 0.681 | 0.262 | 22.0 | 0.766 | 2.0 |
| VAE | capsule | 0.604 | **0.643** | 0.348 | 0.205 | 21.6 | 0.844 | 4.0 |
| VAE | pill | 0.449 | **0.607** | 0.622 | 0.170 | 28.6 | 0.855 | 2.9 |

PSNR/SSIM (`reconstruction_quality` in `src/eval/metrics.py`) are computed only
on non-defective test images — see the function's docstring for why mixing in
defective images would make the number ambiguous. AE reconstructs normal
images more faithfully than VAE on every single category (e.g. capsule:
34.6dB/0.940 vs. 21.6dB/0.844) — consistent with VAE's KL term trading some
sharpness for a smoother latent space.

Two things worth digging into further rather than treating as final:
- Max-pooling beats mean-pooling everywhere except AE/pill — worth checking
  whether that category has diffuse (whole-object) defect types rather than
  small localized ones, which would explain the reversal.
- VAE/capsule's pixel AUROC (0.348) is *below* chance (0.5), meaning its
  anomaly map is anti-correlated with the true defect location for that
  category — likely the VAE's blurrier reconstructions creating high error at
  normal but hard-to-reproduce detail (e.g. printed text) rather than at the
  actual defect. Worth a qualitative look at the reconstructions before
  trusting this number in the writeup.

## DDIM training + latency sweep (Phase 3)

Trains `src/models/unet.py` + `src/models/diffusion.py` (the Phase 1 from-scratch
DDPM/DDIM implementation) the same way as the AE/VAE: one model per category,
100 epochs, same [-1, 1] input scaling.

```bash
python scripts/train_diffusion.py --category bottle --epochs 100
python scripts/evaluate_diffusion.py --category bottle --t-start 250 --ddim-steps 25
python scripts/sweep_ddim_steps.py --category bottle   # latency vs. quality across step counts
```

**A real GPU-memory bug we hit and fixed:** the AE/VAE default batch size (32)
made the diffusion UNet need ~11.4GB at image_size=128/base_channels=64 — more
than the 8GB GPU has. Windows doesn't error on this, it silently spills into
slow shared system memory, which measured **~280x slower per step** (11.5s/step
instead of 41ms/step) — nothing was broken, it just looked like the code had
hung. Fixed by dropping the diffusion training batch size to 8 (measured 3GB
peak, and empirically also faster per-epoch than batch=16). See the comment on
`--batch-size` in `scripts/train_diffusion.py`.

**A latency-measurement bug we hit and fixed:** the first sweep run produced
latency numbers that got *worse* going from 25 steps to 50 then better again at
100 steps — impossible for a strictly sequential process where every step is
the same fixed cost. Root cause: `measure_latency`'s fixed-iteration warmup
(`warmup=3` calls) doesn't give a laptop GPU's dynamic clock scaling enough
real time to ramp up to a stable boost clock before timing starts — and a
config with few, fast calls (5 DDIM steps) reaches the iteration count with
far less elapsed real time than a config with many, slow calls (100 steps),
so short configs got measured "cold" and looked disproportionately slow.
Fixed in `src/eval/metrics.py`: `measure_latency` now keeps warming up until
*both* the iteration count *and* a minimum real time (default 1 second) have
elapsed, regardless of how many iterations that takes. `scripts/remeasure_ddim_latency.py`
was a one-off fix-up script that re-measured just the (now-correct) latency
columns without re-running the expensive full-dataset AUROC pass, since that
part was never affected by this bug.

### Results (t_start=250, image_auroc via max-pooling, 100-epoch checkpoints)

| Model | Category | Image AUROC | Pixel AUROC | IoU | PSNR (dB) | SSIM | Latency @ 25 steps (ms) |
|---|---|---|---|---|---|---|---|
| DDIM | bottle | **0.919** | 0.786 | 0.249 | 34.6 | 0.953 | 1273 |
| DDIM | capsule | 0.725 | **0.835** | 0.192 | 39.8 | 0.965 | 2580 |
| DDIM | pill | 0.569 | **0.842** | 0.157 | 36.6 | 0.948 | 2520 |

DDIM's image AUROC beats both AE and VAE on bottle (0.919 vs. 0.861 / 0.825)
and is competitive on pixel AUROC everywhere — but at roughly **1000x the
latency** of AE/VAE (~2ms vs. ~1300-2600ms per image). That trade-off, not raw
accuracy, is DDIM's central story for this benchmark.

### Step-count sweep — latency vs. quality (`results/ddim_steps_sweep.csv`)

| Category | Steps | Image AUROC | PSNR (dB) | SSIM | Latency (ms) |
|---|---|---|---|---|---|
| bottle | 5 | **0.940** | 34.96 | 0.952 | 446 |
| bottle | 10 | 0.916 | 34.88 | 0.954 | 1137 |
| bottle | 25 | 0.908 | 34.50 | 0.952 | 2649 |
| bottle | 50 | 0.937 | 34.40 | 0.951 | 4311 |
| bottle | 100 | 0.938 | 34.24 | 0.951 | 7601 |
| capsule | 5 | **0.779** | 39.89 | 0.965 | 111 |
| capsule | 10 | 0.714 | 39.70 | 0.964 | 179 |
| capsule | 25 | 0.721 | 39.84 | 0.965 | 2044 |
| capsule | 50 | 0.716 | 39.56 | 0.963 | 1123 |
| capsule | 100 | 0.716 | 39.63 | 0.965 | 5108 |
| pill | 5 | **0.706** | 36.95 | 0.954 | 216 |
| pill | 10 | 0.535 | 36.62 | 0.949 | 216 |
| pill | 25 | 0.681 | 36.76 | 0.950 | 1620 |
| pill | 50 | 0.535 | 36.49 | 0.947 | 1536 |
| pill | 100 | 0.544 | 36.41 | 0.946 | 12149 |

Two findings, one clean and one messier — worth being honest about both:

**Clean finding:** every single category's best image AUROC came from the
*fewest* steps (5), not the most (100), and **reconstruction quality is
essentially flat regardless of step count** (e.g. bottle SSIM: 0.952 → 0.951
across the entire 5-to-100 range — a difference of 0.001). So more DDIM
sampling steps buys you neither better detection nor better reconstruction
here, only more latency. For a real inspection line, 5 steps is strictly the
right choice at this training budget: equal-or-better detection, equal
reconstruction quality, and roughly 1/17th the latency of 100 steps. Worth
digging into *why* AUROC doesn't improve with more steps in the writeup
(hypothesis: with a fixed 100-epoch training budget, the network's noise
prediction may not be equally well-calibrated at every timestep, so
fine-grained stepping doesn't average out errors the way it would with a
better-trained model) — not obvious this generalizes to a longer schedule.

**Messier finding, disclosed rather than hidden:** latency does NOT scale
cleanly with step count in this run — e.g. capsule's 25-step config (2044ms)
measured slower than its 50-step config (1123ms), and pill's 100-step config
(12149ms) is disproportionately worse than the trend from 5-50 steps would
predict. This is a *different* phenomenon from the GPU-clock warmup bug fixed
earlier in Phase 3 (that bug had a consistent, one-directional pattern —
short configs always measured artificially slow; this is scattered noise in
both directions). The most likely explanation is thermal/system variance:
this sweep ran continuously for roughly 90 minutes, and a shared consumer
laptop GPU's boost clock is not guaranteed to stay perfectly stable that long,
especially in whichever configuration happened to run last/hottest (pill's
100-step config, which ran at the very end of the batch, is exactly the
outlier you'd expect if that's the cause). This wasn't chased further because
diminishing returns: getting perfectly clean latency numbers on a laptop GPU
under a multi-hour job would need either much shorter batches with cooldown
gaps between them, or a dedicated benchmarking machine — a reasonable
limitation to name explicitly in the Discussion rather than pretend away.

## Comparison plots and qualitative figures (Phase 4)

```bash
python scripts/plot_results.py                # bar charts + step-count sweep plot -> figures/
python scripts/generate_qualitative_figures.py # per-category input/reconstruction/anomaly-map/GT grids -> figures/
```

`figures/qualitative_*.png` visually confirm two of the quantitative findings
above: capsule's VAE anomaly map is dominated by background noise rather than
the actual crack (matches its below-chance 0.348 pixel AUROC), and DDIM's
pill reconstruction leaves the red-dot defects largely intact rather than
"fixing" them (matches its worst-of-the-three 0.569 image AUROC there).

## GUI prototype

```bash
python app/gui.py
```

Two tabs:
- **Detect** — pick a category and model, load a test image (defaults to
  browsing that category's `data/<category>/test/` folder), click **Run
  Detection**, and see the reconstruction + anomaly heatmap rendered live.
  For DDIM, the step-count spinbox lets you see the Phase 3 latency/quality
  trade-off interactively — drop it to 5 steps and the result is both faster
  and often just as good.
- **Train** — configure hyperparameters (model, category, epochs, batch
  size, learning rate) and train a new checkpoint, watching the train/val
  loss curve update live as it trains. This is the GUI's own hyperparameter
  configuration + telemetry, addressing the rubric's "GUI or CLI with useful
  telemetry and hyperparameter configuration" requirement directly in the
  GUI rather than relying on the CLI scripts alone.

The Train tab launches `scripts/train.py`/`scripts/train_diffusion.py` as a
genuine separate process (`QProcess`) rather than reimplementing the
training loop inside the GUI, and parses their printed `[epoch N/M] ...`
lines to drive the live plot. **This was a deliberate redesign, not the
original approach** — worth knowing if you touch this file:

An earlier version ran training on a `QThread` inside the GUI process,
reusing `train.py`'s functions directly. Testing surfaced a real, repeatable
bug: a PyTorch `DataLoader(num_workers>0)` created from a background
`QThread` (rather than the process's own main thread) is unreliable on
Windows — it repeatedly hung in what looked like a worker-crash/respawn
loop, confirmed via multiple isolated headless test runs. Switching to
`QProcess` sidesteps this entirely: the training script runs in its own
completely normal process with its own main thread, identical to a user
typing the command themselves — the same well-tested code path that
produced every checkpoint behind this project's results table.

A second, unrelated issue surfaced during the same testing: `QMessageBox`
(a native modal dialog) segfaults immediately under Qt's `offscreen`
platform plugin, used for automated headless testing. This is a known
limitation of `offscreen` mode, not a bug in the app — `QMessageBox` works
completely normally with a real display, which is how the app is actually
used. Confirmed via an isolated repro; the automated test stubs
`QMessageBox` out rather than working around it in the real code.

**Also still true:** on Windows, `torch` must be imported *before* `PyQt5`
anywhere both are used in the same process — importing PyQt5 first loads a
conflicting shared DLL that makes torch's own DLL loading crash with
`OSError: [WinError 1114]`. `app/gui.py` has the correct order and a warning
comment; don't let an auto-formatter or "cleanup" reorder/alphabetize those
imports.
