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

## Key findings

1. **The original "AE beats VAE" result was a bug, not a property of VAEs.** The first VAE
   suffered posterior collapse (0 of 256 latent dimensions used) from a loss-scale mismatch.
   With `kl_weight=1e-6` plus KL annealing it matches the AE ([details](#vae-posterior-collapse-why-the-original-vae-lost-to-the-ae)).
2. **A single-seed gap can be an outlier.** The apparent VAE weakness on pill came from one
   unlucky seed ([seed study](#why-is-pill-weaker-diagnosis--seed-study)).
3. **Scoring with a learned per-pixel variance (the reconstruction probability of An & Cho, 2015)
   is what helps** — not the stochastic latent. On bottle it lifts image AUROC from 0.86 to 0.98
   (5/5 seeds); on pill it improves pixel AUROC by about 0.05; it does not help image-level
   detection on capsule or pill ([Phase 5](#probabilistic-scoring-and-the-2x2-comparison-phase-5)).
4. **How pixel scores are pooled into an image score matters** and is category dependent;
   top-1% was best on average (0.762 vs 0.724 for max).
5. The AE, VAE, Probabilistic AE and Probabilistic VAE are compared over **5 seeds** with
   mean ± std; the one failure mode found is that the deterministic Probabilistic AE
   sometimes fails to train (2 of 15 runs vs 0 of 45 for the other three models).

## Repo structure

```
data/               MVTec AD categories (bottle, capsule, pill) — gitignored, see below
src/data/           Dataset loading & preprocessing (MVTecTrainDataset, MVTecTestDataset)
src/eval/           Evaluation harness: image/pixel AUROC, IoU, latency (src/eval/metrics.py)
src/utils/          Experiment tracking (TensorBoard + CSV/JSON, src/utils/tracking.py)
src/models/         AE / VAE / Gaussian-likelihood AE+VAE / DDIM, all original (autoencoder.py, vae.py, vae_gauss.py, unet.py, diffusion.py)
configs/            YAML experiment configs
scripts/            Train / evaluate / sweep / plot scripts, dataset download, smoke tests
app/                PyQt5 inspection GUI (app/gui.py)
tests/              Unit tests (pytest)

Created when you run the scripts (not stored in the repository):
checkpoints/        Trained model weights
runs/               Training logs (TensorBoard)
results/            CSV results written by the evaluation and sweep scripts
figures/            Plots and qualitative figures written by the plotting scripts
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
so results are directly comparable. It also provides `aggregate_image_scores`
(max, mean and top-0.1% / 1% / 5% pooling of a per-pixel map into one image score);
`scripts/evaluate.py` now reports image AUROC for mean, max **and top-1%**
(`image_auroc_top1pct`). Run the unit tests (synthetic data, no
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

> **Note on VAE defaults.** The VAE rows in `results/summary.csv` below were
> produced with the *original* settings (`--kl-weight 1e-3 --kl-anneal-epochs 0`),
> which suffered posterior collapse (see the next section). The script defaults
> have since been changed to the fixed settings (`--kl-weight 1e-6
> --kl-anneal-epochs 30`), so re-running `run_phase2.py` now trains the
> *non-collapsed* VAE and will overwrite `checkpoints/vae_<category>/`. To
> reproduce the original collapsed run exactly, pass
> `--kl-weight 1e-3 --kl-anneal-epochs 0` to `train.py`.

### Results: original run (128px images, 100 epochs, latent_dim=256)

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
defective images would make the number ambiguous.

**In this original run the AE beat the VAE on every category** (e.g. capsule:
34.6dB/0.940 vs. 21.6dB/0.844), which is the opposite of much of the
literature. We initially suspected the VAE's KL term trading sharpness for a
smoother latent space, but that explanation turned out to be incomplete: the
VAE was not a *working* VAE at all. See the next section.

One more thing worth digging into rather than treating as final:
- Max-pooling beats mean-pooling everywhere except AE/pill — worth checking
  whether that category has diffuse (whole-object) defect types rather than
  small localized ones, which would explain the reversal.

### VAE posterior collapse: why the original VAE lost to the AE

Rather than accept "AE beats VAE" at face value, we inspected the trained VAE
checkpoints (`python scripts/check_vae_collapse.py`):

| Run | KL / sample | Active latent dims | Mean posterior std |
|---|---|---|---|
| bottle, kl_weight=1e-3 (original) | 4.3 | **0 / 256** | 0.98 |
| capsule, kl_weight=1e-3 (original) | 3.1 | **0 / 256** | 0.98 |
| pill, kl_weight=1e-3 (original) | 3.9 | **0 / 256** | 0.98 |

Zero latent dimensions varied between images and the posterior was almost
exactly the prior N(0,1) (std ≈ 1): the encoder was passing the decoder no
information about the input, so the decoder output a blurry average image.
This is **posterior collapse**.

**Cause: a loss-scale mismatch.** In `src/models/vae.py` the reconstruction
term is a *mean* over pixels (≈ 0.002 once trained) but the KL term is a
*sum* over 256 latent dimensions. With `kl_weight=1e-3` the weighted KL
(≈ 0.004) was larger than the entire reconstruction loss, so the optimiser
"won" by collapsing the posterior. The original code comment said the weight
was "small", which was true in absolute terms but wrong relative to a
mean-MSE reconstruction loss.

**A second, compounding cause: scoring a collapsed VAE with `mu`.**
`VAE.anomaly_map` reconstructs from the mean latent `mu` (so scores are
repeatable). A collapsed decoder is only ever trained on noisy samples
`z ~ N(0,1)`, never on the single point `mu`, so `decoder(mu)` is systematically
off (capsule: background colour error of -0.19). Averaging the decoder over 30
sampled latents instead (`python scripts/check_vae_scoring.py`) lifts the
*original* models' pixel AUROC from 0.348 to 0.812 (capsule), 0.681 to 0.830
(bottle) and 0.622 to 0.723 (pill). So the original below-chance capsule number
was part collapse, part this scoring mismatch. For the *tuned* VAEs the two
scoring methods agree (e.g. capsule 1e-6: 0.871 vs. 0.869), so `mu` scoring is
fine once the VAE is not collapsed.

**Fix:** a much smaller `kl_weight` plus *KL annealing*: `train.py
--kl-anneal-epochs 30` ramps the weight linearly from 0 so the decoder learns
to use the latent code before the KL penalty regularises it. We ran a sweep
(`python scripts/sweep_vae_kl.py`, 100 epochs, 30-epoch warm-up, results in
`results/vae_kl_sweep.csv`):

| Category | Run | Image AUROC (mean) | Image AUROC (max) | Pixel AUROC | IoU | PSNR (dB) | SSIM | Active dims |
|---|---|---|---|---|---|---|---|---|
| bottle | AE | 0.794 | 0.861 | 0.830 | 0.267 | 28.9 | 0.850 | n/a |
| bottle | VAE 1e-3 (orig) | 0.659 | 0.825 | 0.681 | 0.262 | 22.0 | 0.766 | 0 |
| bottle | VAE 1e-4 | 0.828 | 0.863 | 0.799 | 0.266 | 27.7 | 0.844 | 46 |
| bottle | VAE 1e-5 | 0.801 | 0.862 | 0.846 | 0.267 | 28.5 | 0.843 | 73 |
| bottle | VAE 1e-6 | 0.808 | 0.864 | **0.847** | 0.267 | 29.0 | 0.850 | 122 |
| capsule | AE | 0.563 | 0.688 | 0.849 | 0.199 | 34.6 | 0.940 | n/a |
| capsule | VAE 1e-3 (orig) | 0.604 | 0.643 | 0.348 | 0.205 | 21.6 | 0.844 | 0 |
| capsule | VAE 1e-4 | 0.533 | 0.659 | 0.510 | 0.204 | 25.0 | 0.889 | 19 |
| capsule | VAE 1e-5 | 0.572 | 0.700 | 0.846 | 0.201 | 34.8 | 0.942 | 75 |
| capsule | VAE 1e-6 | 0.652 | 0.694 | **0.871** | 0.200 | 35.4 | 0.943 | 116 |
| pill | AE | 0.774 | 0.610 | 0.841 | 0.171 | 33.5 | 0.910 | n/a |
| pill | VAE 1e-3 (orig) | 0.449 | 0.607 | 0.622 | 0.170 | 28.6 | 0.855 | 0 |
| pill | VAE 1e-4 | 0.672 | 0.619 | 0.726 | 0.181 | 32.1 | 0.901 | 0 |
| pill | VAE 1e-5 | 0.631 | 0.624 | 0.793 | 0.179 | 33.0 | 0.908 | 0 |
| pill | VAE 1e-6 | 0.744 | 0.607 | 0.682 | 0.175 | 32.2 | 0.912 | 0 |

![VAE KL-weight sweep](figures/vae_kl_sweep.png)

**Findings:**
- The original "AE beats VAE" result was a **tuning artifact of posterior
  collapse**, not evidence that VAEs are inherently worse for this task.
- With `kl_weight=1e-6` the VAE matches or beats the AE on pixel AUROC for
  bottle (0.847 vs. 0.830) and capsule (0.871 vs. 0.849), and its capsule
  PSNR rises from 21.6 to 35.4 dB (above the AE's 34.6). The below-chance
  capsule pixel AUROC (0.348) disappears.
- Image-level AUROC is roughly on par with the AE (differences are within what
  the small test sets can resolve), so we do not claim a clear image-level winner.
- Latent usage rises as the weight falls (0 → 46 → 73 → 122 active dims on
  bottle), and reconstruction quality and localisation recover with it.
- **Pill looked like a limitation in this sweep, but the seed study below shows
  that was largely a single-seed artifact.** (The per-seed investigation is in
  "Why is pill weaker?" below.) The "active dims = 0" reading on pill is a crude
  variance count on very uniform images; treat it as a rough indicator only.
- **Caveats to state in the report:** (1) the best weight differs per category
  (1e-6 on bottle/capsule, 1e-5 on pill) and weights were compared on test-set
  metrics with no separate validation split, so per-category "best" numbers
  are optimistic; the comparison charts therefore use one fixed weight (1e-6)
  for every category. (2) The sweep table above is a single run (seed 42) per configuration, and
  the test sets are small, so small AUROC differences are noise. The seed study
  below shows this matters: on pill the seed-42 VAE 1e-6 run was an outlier.

Qualitatively (see `figures/qualitative_*.png`), the original VAE's
reconstruction is a blurry average and its anomaly map lights up the entire
background; the tuned VAE reconstructs like the AE. Note that on the capsule
example shown, *none* of the models highlight the small crack (the anomaly
maps are dominated by edges and printed text), so good pixel AUROC averaged
over the test set does not mean every defect is found.

### Why is pill weaker? (diagnosis + seed study)

In the single-seed sweep the VAE trailed the AE on pill pixel AUROC (0.682 at
1e-6, 0.793 at 1e-5 vs. 0.841), and 1e-5 beat 1e-6, which is not the monotonic
trend seen on bottle and capsule. We investigated rather than accept it.

**Step 1 — where does the gap come from?** (`python scripts/diagnose_background_bias.py`,
seed-42 checkpoints):
- *Inside the pill* all models localise defects equally well: pixel AUROC
  computed over the object region only is 0.702 (AE), 0.707 (VAE 1e-5) and
  0.711 (VAE 1e-6). Defect pixels have the same mean error (~0.096) in every model.
- The gap is in the **plain background**: mean error 0.031 for VAE 1e-6 vs.
  0.011 for the AE (3x). The VAE 1e-6 reconstruction was shifted *darker* by
  about 0.025 and the shift varied per image (std 0.012 vs. 0.001 for the AE).
  After subtracting each image's constant offset, background error is 0.0098 vs.
  0.0100 (AE), i.e. the structure is reconstructed equally well; only a global
  brightness offset differs.
- By defect type, the VAE 1e-6 run was weakest on subtle defects (scratch 0.62
  vs. AE 0.79; faulty_imprint 0.66 vs. 0.82) and matched the AE on color,
  crack and contamination.
- **Metric caveat:** all defect pixels lie inside the pill, so pixel AUROC
  partly rewards a model for having a low, uniform *background* error, not only
  for localising defects. Pill image-level AUROC (max-pooled) is ~0.61 for
  *every* model including the AE, so pill is hard for all of them.

**Step 2 — is it the sampling/`mu` mismatch?** No. Scoring the tuned pill VAE
with 30 averaged samples gives the identical background bias (-0.0244), so the
decoder itself produced the darker background in that run.

**Step 3 — is it run-to-run variance?** (`python scripts/seed_study.py`; results
in `results/seed_study.csv`; seeds 1, 2, 3 plus the original seed 42; each seed
changes weight init, data order and the train/val split.) Pill pixel AUROC:

| Config | seed 42 | seed 1 | seed 2 | seed 3 | mean ± sd |
|---|---|---|---|---|---|
| AE | 0.841 | 0.840 | 0.783 | 0.837 | 0.825 ± 0.028 |
| VAE 1e-6 | **0.682** | 0.844 | 0.797 | 0.847 | 0.792 ± 0.077 |
| VAE 1e-5 | 0.793 | 0.809 | 0.823 | 0.777 | 0.801 ± 0.020 |

**Conclusion:** the seed-42 VAE 1e-6 run was an outlier. Across the other three
seeds VAE 1e-6 is on par with the AE (0.844 / 0.797 / 0.847 vs. 0.840 / 0.783 /
0.837). The darker-background offset is also not VAE-specific: AE seed 2 has a
background bias of -0.018 and its pixel AUROC drops to 0.783, and VAE 1e-5
seed 3 has a +0.022 bias. The most likely explanation is that the final-epoch
weights (constant learning rate, no decay, no checkpoint selection) sometimes
land on a slightly off global brightness; this affects both models, and the VAE
at 1e-6 had one bad draw in four. **We did not test this explanation directly**
(e.g. with a learning-rate schedule or weight averaging). With only four seeds
per config the means have wide uncertainty, so we do not claim the VAE beats
or matches the AE on pill, only that the original "VAE is weaker on pill"
reading is not supported once seeds are varied. Image-level mean-pooled AUROC is
also very seed-sensitive on pill (AE: 0.647 to 0.798), so it should not be
over-interpreted.

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
python scripts/plot_results.py                # original charts + VAE KL sweep + DDIM step-count plot + final 5-seed charts -> figures/
python scripts/generate_qualitative_figures.py # per-category input/reconstruction/anomaly-map/GT grids -> figures/
python scripts/generate_variance_figures.py    # why the probability map differs from reconstruction error -> figures/variance_*.png
```

`figures/final_*.png` are the final 5-seed charts (mean ± std with every seed shown as a dot;
DDIM is a single-seed reference). `figures/qualitative_*.png` show five rows per category
(AE, VAE original/collapsed, VAE tuned at 1e-6, Probabilistic VAE, and DDIM) and visually confirm the quantitative findings:
the original VAE's anomaly map on capsule is dominated by background noise
(matches its below-chance 0.348 pixel AUROC) while the tuned VAE's map looks
like the AE's, and DDIM's pill reconstruction leaves the red-dot defects
largely intact rather than "fixing" them (matches its worst-of-the-three 0.569
image AUROC there). `figures/vae_kl_sweep.png` plots pixel AUROC and PSNR
against `kl_weight`, and the bar charts include both the original and tuned VAE.

## Probabilistic scoring and the 2x2 comparison (Phase 5)

After fixing the VAE we compared our approach with An & Cho (2015), *Variational Autoencoder based
Anomaly Detection using Reconstruction Probability*. Their VAE has a decoder that predicts a **mean
and a variance for every pixel**, is trained with the proper ELBO, and scores anomalies by the
**probability of the input** under that Gaussian, averaged over latent samples, instead of the
absolute pixel error we used. Their claim that this beats a plain AE confounds three things, so we
separated them with a 2x2 design (`src/models/vae_gauss.py`):

| Model | Latent | Decoder | Score |
|---|---|---|---|
| AE baseline | deterministic | mean | abs error |
| **Probabilistic AE** (`GaussianAE`) | deterministic | mean + variance | probability (-log likelihood) |
| VAE baseline (tuned, kl_weight 1e-6) | stochastic | mean | abs error |
| **Probabilistic VAE** (`GaussianVAE`) | stochastic | mean + variance | probability (reconstruction probability) |

The variance models can also be scored with plain abs error (`--score-mode abs`), which separates
the effect of the **training** from the effect of the **score**. A third mode, `wsq`
(variance-weighted squared error), was unreliable (pill pixel AUROC 0.66) and is not recommended.

```bash
python scripts/run_factorial.py --categories bottle capsule pill --seeds 42 1 2 3 4   # trains what is missing, evaluates all 4 cells, resumable
python scripts/sweep_aggregation.py --seeds 42 1 2 3 4                                # image-score pooling experiment
python scripts/generate_variance_figures.py                                           # why it works
python scripts/plot_results.py                                                        # final charts
```

### Final results (5 seeds: 42, 1, 2, 3, 4; mean ± sample std; 128px, 100 epochs, latent 256)

**Pixel AUROC**

| Model | Bottle | Capsule | Pill |
|---|---|---|---|
| AE baseline | 0.843 ± 0.017 | 0.867 ± 0.030 | 0.830 ± 0.027 |
| Probabilistic AE | 0.779 ± 0.160 | 0.826 ± 0.118 | 0.824 ± 0.042 |
| VAE baseline | 0.849 ± 0.015 | 0.838 ± 0.052 | 0.800 ± 0.069 |
| Probabilistic VAE | 0.843 ± 0.008 | 0.867 ± 0.021 | 0.845 ± 0.039 |

**Image AUROC, max-pooled** (top-1% and mean in `figures/final_*.png` / `results/aggregation.csv`)

| Model | Bottle | Capsule | Pill |
|---|---|---|---|
| AE baseline | 0.862 ± 0.003 | 0.692 ± 0.005 | 0.598 ± 0.015 |
| Probabilistic AE | 0.889 ± 0.200 | 0.684 ± 0.048 | 0.563 ± 0.037 |
| VAE baseline | 0.864 ± 0.012 | 0.698 ± 0.005 | 0.602 ± 0.008 |
| Probabilistic VAE | **0.978 ± 0.005** | 0.694 ± 0.022 | 0.566 ± 0.029 |

![final pixel AUROC](figures/final_pixel_auroc.png)
![final image AUROC (max)](figures/final_image_auroc_max.png)

**Two Probabilistic-AE runs did not train** (bottle seed 4: PSNR 5.0 dB; capsule seed 42: 28.7 dB vs ~34 for
its peers), which inflates its spread. Excluding them (flagged by PSNR > 3 dB below the seed median) the
Probabilistic AE gives bottle image AUROC 0.979 ± 0.002 (n=4), bottle pixel 0.850 ± 0.003 (n=4) and
capsule pixel 0.878 ± 0.023 (n=4) — the same as the Probabilistic VAE. The charts keep the failed runs
and show every seed as a dot. Across all 60 runs only the Probabilistic AE had under-trained runs
(2 of 15 vs 0 of 45; one-sided Fisher p ≈ 0.06 — suggestive, not significant).

### What drives the gain (paired by seed; mean difference, seeds won out of 5)

| Comparison | Bottle image AUROC (max) | Capsule | Pill | Pixel AUROC: bottle / capsule / pill |
|---|---|---|---|---|
| **Score only**: prob vs abs, Probabilistic VAE | +0.129 (5/5) | −0.011 (2/5) | −0.045 (0/5) | +0.015 (5/5) / +0.028 (5/5) / +0.060 (5/5) |
| **Score only**: prob vs abs, Probabilistic AE | +0.100 (5/5) | +0.004 (4/5) | −0.039 (1/5) | −0.000 / +0.000 / +0.052 (5/5) |
| **Training only**: Prob. VAE scored with abs vs VAE baseline | −0.015 (2/5) | +0.007 (3/5) | +0.009 (4/5) | −0.021 / +0.001 / −0.014 |
| **Stochastic latent**: VAE baseline vs AE baseline | +0.002 (3/5) | +0.007 (5/5) | +0.004 (4/5) | +0.006 (4/5) / −0.029 (2/5) / −0.030 (3/5) |
| **Stochastic latent**: Prob. VAE vs Prob. AE | +0.089 (3/5) | +0.010 (2/5) | +0.002 (2/5) | +0.064 (1/5) / +0.041 (2/5) / +0.021 (4/5) |

1. **The score is the gain.** The "score only" effect is about the whole effect; training with a variance
   output adds nothing and slightly lowers the mean reconstruction (PSNR on good images, 5-seed mean:
   AE 29.0 / 34.9 / 33.4 dB vs Prob. VAE 27.4 / 33.4 / 32.5 on bottle / capsule / pill).
2. **It does not need a VAE.** A deterministic AE with a variance output gets the same bottle gain.
   The stochastic latent matters little (differences ≤ 0.03 and with split wins).
3. **It is category specific.** Big image-level gain on bottle; none on capsule; slightly negative on pill.
   Pixel AUROC improves in most runs (always on pill with the Probabilistic VAE).
4. The Probabilistic VAE is best or tied-best on pixel AUROC in all three categories.

### Why it works (`scripts/generate_variance_figures.py`, seed-42 AE vs Probabilistic AE)

![variance figure, bottle](figures/variance_bottle.png)

| Category | Spearman(σ, plain AE \|error\|) on good images | Mean σ on defect pixels vs normal pixels |
|---|---|---|
| Bottle | 0.84 | 0.061 vs 0.057 |
| Pill | 0.58 | 0.070 vs 0.050 |
| Capsule | 0.26 (the seed-42 capsule model was the under-trained one) | 0.058 vs 0.038 |

On bottle the plain AE's error map lights up every ring edge and highlight (natural reconstruction
errors); the model's predicted σ is high at exactly those places, so the probability map suppresses
them and leaves the defect. Defect pixels have the same σ as normal ones, so dividing by σ removes
false alarms without weakening the defect. On pill σ is high over the whole pill body, so defects there
are partly divided away, while tiny σ on the dark background heavily penalises small deviations
(faint speckle in the probability map), which raises the max score of good images — it cleans the
pixel map (pixel AUROC +0.06) but does not help image-level max-pooling.

### Image-score aggregation (`scripts/sweep_aggregation.py`, 5 seeds)

Pixel maps are turned into an image score by pooling; we compared max, top-0.1%, top-1%, top-5% and mean.

| | max | top 0.1% | top 1% | top 5% | mean |
|---|---|---|---|---|---|
| Bottle, VAE baseline | 0.864 | 0.878 | **0.914** | 0.880 | 0.797 |
| Bottle, Prob. VAE | 0.978 | 0.983 | **0.987** | 0.976 | 0.961 |
| Capsule, VAE baseline | **0.698** | 0.678 | 0.660 | 0.643 | 0.576 |
| Capsule, Prob. VAE | **0.694** | 0.684 | 0.663 | 0.651 | 0.617 |
| Pill, VAE baseline | 0.602 | 0.622 | 0.710 | 0.731 | **0.740** |
| Pill, Prob. VAE | 0.567 | 0.585 | 0.697 | **0.729** | 0.717 |

- The best pooling is **category dependent** (bottle: top-1%, capsule: max, pill: mean / top-5%); averaged over
  every model, category and seed top-1% is best (0.762 vs 0.724 for max, 0.719 for mean).
- Probabilistic VAE minus VAE baseline, by pooling (seeds won of 5): bottle +0.073 to +0.165 (5/5 for every
  pooling); capsule −0.004 to +0.041 (mixed); pill −0.037 to −0.003, shrinking to about zero at top-5%.
- Caveat: choosing the best pooling per category on the test set is optimistic and cannot be tuned without
  anomalies. We therefore report several and use top-1% as the default.

![aggregation](figures/final_aggregation.png)

### Ablations — what does *not* matter (Probabilistic VAE unless noted; `scripts/sweep_n_samples.py`, `scripts/run_vae_gauss.py`)

| Question | How tested | Result |
|---|---|---|
| Does averaging over more latent samples L help? | Re-scored 12 trained models with L = 1, 2, 5, 10, 30 | No: bottle image AUROC 0.978 / 0.978 / 0.978 / 0.977 / 0.977; capsule pixel 0.859 to 0.863; pill pixel 0.839 to 0.842. **L=1 is enough** (~3.5 ms instead of ~15 ms per image vs ~2 ms for abs scoring) |
| Does the KL weight matter? (ELBO weight 0.1 / 1 / 10; seeds 42, 1) | 8 new training runs | No: bottle image AUROC 0.974 / 0.977 / 0.976; pill pixel 0.853 / 0.821 / 0.839 (seed spread ±0.04); no collapse at 10 |
| Does the latent size matter? (16 / 256 / 1024; seeds 42, 1) | 8 new training runs | No: bottle image AUROC 0.978 / 0.977 / 0.975, pixel 0.853 / 0.837 / 0.835. Pill seed 42 is a bad seed at every size (0.49 to 0.78) |
| Does the KL term matter for the gain? | Control: same model, KL off (noisy latent), 7 runs | No: healthy runs 0.977 ± 0.005 vs 0.978 ± 0.006 on bottle — but 3 of 7 control runs under-trained vs 0 of 8 |
| Is proper ELBO with *fixed* variance different from our tuned VAE? | Algebra (not a separate run) | No: a Gaussian ELBO with constant variance σ² equals our loss with `kl_weight = 2σ²/49152`; our 1e-6 corresponds to σ² ≈ 0.025, and the collapsed 1e-3 was ~25× stronger than a standard ELBO |

### Limitations

- 5 seeds, three MVTec categories, 100 epochs, 128px images; weights and pooling compared on test-set metrics.
- DDIM was not re-run with multiple seeds, so it appears as a single-seed reference.
- The Probabilistic VAE's reported scores use L=10 latent samples; L=1 gave the same accuracy in the sweep above.
- Only the bottleneck size was varied for model capacity, not decoder width or depth.

### Recommendation

Use the **Probabilistic VAE** (`--model vae_gauss`, `--score-mode prob`, L=1) with **top-1%** pooling when image-level
detection of structured, small defects matters (bottle-like objects); for pill- and capsule-like categories the
plain AE/VAE with a broader pooling is as good, and the probabilistic score still improves pixel-level localisation.

## GUI prototype

```bash
python app/gui.py
```

A dark, inspection-station style window with two tabs:
- **Inspect** — pick a category and detector (AE, VAE, **Prob. AE, Prob. VAE**, DDIM; "VAE" loads the
  tuned `vae_<category>_kl1e-6` checkpoint, not the collapsed original; the Prob. models use the
  probability score with a single latent sample), then **Load image…** (defaults to
  browsing that category's `data/<category>/test/` folder) and **Run inspection**, or
  **Inspect folder…** to run a whole folder. The app gives a **verdict — NORMAL or
  ANOMALY DETECTED** — with a score-vs-threshold gauge, the input, reconstruction,
  anomaly heat-map and a heat-map overlay, plus the image score, threshold,
  score/threshold ratio, mean and max pixel score and the inference time.
  Every inspection is added to an **inspection log** (exportable to CSV); images
  under the MVTec `test/` folders are also compared with their folder-name label
  (`good` = normal, anything else = defect) so the log shows whether the model was right.
  For DDIM, the step-count spinbox lets you see the Phase 3 latency/quality
  trade-off interactively — drop it to 5 steps and the result is both faster
  and often just as good.
- **Train** — configure hyperparameters (model, category, epochs, batch
  size, learning rate) and train a new checkpoint, watching the train/val
  loss curve update live as it trains. This gives the GUI its own
  hyperparameter configuration and live telemetry, so training does not
  have to be started from the command line.

**How the verdict is made.** The models output a per-pixel anomaly map, not a
yes/no answer, and raw scores are on a different scale for every model and
category, so the first use of a (model, category) pair is *calibrated*: the app
scores the held-out **normal** validation images (the same seed-42 train/val
split `scripts/train.py` used, so the model never trained on them and no
defective image is involved; capped at 24 images) and sets

```
threshold = mean + k * std     (image scores of those normal images)
```

An image whose score reaches the threshold is flagged as an anomaly. The image
score is the **top-1 %** pooled score (mean of the worst 1 % of pixels), the
rule that was best on average in the aggregation experiment. `k` is the
**Sensitivity** slider (default 2.0): lower flags more parts (fewer missed
defects, more false alarms). The calibration is cached next to each checkpoint
as `gui_calibration.json` and re-computed if the checkpoint changes. Scores
within ±10 % of the threshold are marked *borderline*. This is a rule of thumb
from about 20 normal images per category, and it inherits each model's
limits: on bottle the Prob. VAE verdicts are about 95 % correct on the test
set, while on capsule and pill many defects pass as normal, in line with the
modest image-level AUROC reported above.

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
