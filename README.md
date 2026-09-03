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
src/models/         AE / VAE / DDIM model implementations (to be added)
configs/            YAML experiment configs
scripts/            One-off utility scripts (dataset download, smoke tests)
tests/              Unit tests (pytest)
third_party/        Cloned reference implementations (gitignored, see below)
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

## Reference DDIM/AnoDDPM implementation

`third_party/AnoDDPM` is a clone of the official implementation of
[AnoDDPM (Wyatt et al., CVPR Workshops 2022)](https://github.com/Julian-Wyatt/AnoDDPM)
— UNet + Gaussian diffusion with simplex noise — kept as reference/adaptation
material rather than vendored into `src/`. It's gitignored; reclone with:

```bash
git clone https://github.com/Julian-Wyatt/AnoDDPM.git third_party/AnoDDPM
```

Note: `third_party/AnoDDPM/simplex.py` needed a one-line fix for a numpy 2.x /
Windows int32 overflow bug (`np.random.randint` needs `dtype=np.int64`) — reapply
that if you reclone. Confirm the pipeline runs end-to-end on our real MVTec data
(a handful of training iterations at reduced resolution, not a real training run):

```bash
python scripts/smoke_test_anoddpm.py --category pill --steps 5
```
