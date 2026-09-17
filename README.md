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
and `measure_latency`, shared across all three model families so results are
directly comparable. Run the unit tests (synthetic data, no trained model needed):

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

| Model | Category | Image AUROC (mean) | Image AUROC (max) | Pixel AUROC | IoU | Latency (ms) |
|---|---|---|---|---|---|---|
| AE | bottle | 0.794 | **0.861** | 0.830 | 0.267 | 1.9 |
| AE | capsule | 0.563 | **0.688** | 0.849 | 0.199 | 2.3 |
| AE | pill | **0.774** | 0.610 | 0.841 | 0.171 | 2.8 |
| VAE | bottle | 0.659 | **0.825** | 0.681 | 0.262 | 2.0 |
| VAE | capsule | 0.604 | **0.643** | 0.348 | 0.205 | 4.0 |
| VAE | pill | 0.449 | **0.607** | 0.622 | 0.170 | 2.9 |

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

| Model | Category | Image AUROC | Pixel AUROC | IoU | Latency @ 25 steps (ms) |
|---|---|---|---|---|---|
| DDIM | bottle | **0.919** | 0.786 | 0.249 | 1273 |
| DDIM | capsule | 0.725 | **0.835** | 0.192 | 2580 |
| DDIM | pill | 0.569 | **0.842** | 0.157 | 2520 |

DDIM's image AUROC beats both AE and VAE on bottle (0.919 vs. 0.861 / 0.825)
and is competitive on pixel AUROC everywhere — but at roughly **1000x the
latency** of AE/VAE (~2ms vs. ~1300-2600ms per image). That trade-off, not raw
accuracy, is DDIM's central story for this benchmark.

### Step-count sweep — latency vs. quality (`results/ddim_steps_sweep.csv`)

| Category | Steps | Image AUROC | Latency (ms) |
|---|---|---|---|
| bottle | 5 | **0.949** | 504 |
| bottle | 25 | 0.912 | 2494 |
| bottle | 100 | 0.929 | 9811 |
| capsule | 5 | **0.750** | 477 |
| capsule | 25 | 0.704 | 2465 |
| capsule | 100 | 0.625 | 7600 |
| pill | 5 | 0.651 | 366 |
| pill | 10 | **0.655** | 794 |
| pill | 100 | 0.632 | 7448 |

Latency scales almost perfectly linearly with step count (confirms the fix
above worked — e.g. bottle: 504 → 1015 → 2494 → 4900 → 9811ms for 5/10/25/50/100
steps). The more interesting result: **more steps does not mean better
detection** — every category's best image AUROC came from 5 or 10 steps, not
100. That means the practical choice for a real inspection line is the cheap
one: 5-10 steps gets equal-or-better detection at roughly 1/15th the latency
of 100 steps. Worth digging into *why* in the writeup (hypothesis: with a
fixed 100-epoch training budget, the network's noise prediction may not be
equally well-calibrated at every timestep, so very fine-grained stepping can
compound small errors rather than average them out) — not obvious that this
generalizes to a longer training schedule.

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

Pick a category and model, load a test image (defaults to browsing that
category's `data/<category>/test/` folder), click **Run Detection**, and see
the reconstruction + anomaly heatmap rendered live. For DDIM, the step-count
spinbox lets you see the Phase 3 latency/quality trade-off interactively —
drop it to 5 steps and the result is both faster and often just as good.

**A real bug worth knowing about if you touch this file:** on Windows, `torch`
must be imported *before* `PyQt5` anywhere both are used in the same process —
importing PyQt5 first loads a conflicting shared DLL that makes torch's own
DLL loading crash with `OSError: [WinError 1114]`. Confirmed by testing both
import orders directly. `app/gui.py` already has the correct order and a
warning comment; don't let an auto-formatter or "cleanup" reorder/alphabetize
those imports.
