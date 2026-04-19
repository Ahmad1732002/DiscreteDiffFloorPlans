# DiscreteDiffFloorPlans

Constrained floor plan graph generation using discrete diffusion (DiGress), conditioned on a building boundary and optional room constraints. Generated room-type adjacency graphs are rendered into full floor plan images via a frozen Graph2Plan GNN.

---

## Overview

The pipeline has two stages:

1. **DiGress (this repo)** — a discrete graph diffusion model that generates a room-type adjacency graph conditioned on:
   - A boundary TF (Turning Function) descriptor (1000-dim)
   - Optional room count and adjacency constraints (classifier-free guidance)

2. **Graph2Plan (frozen GNN)** — converts the generated graph + boundary into a rendered floor plan image.

The constrained model supports inference with or without explicit room constraints. When no constraints are given, the model generates unconditionally from the boundary alone (learned via CFG dropout during training).

---

## Pretrained Checkpoint

Download the constrained DiGress checkpoint from HuggingFace:

```
https://huggingface.co/ahmadfraij/disdif/resolve/main/last-v1.ckpt
```

---

## Repository Structure

```
DiscreteDiffFloorPlans/
├── DiGress/                   # Diffusion model (training + sampling)
│   ├── src/
│   │   ├── main.py            # Training entry point
│   │   ├── diffusion_model_discrete.py
│   │   ├── datasets/
│   │   │   ├── floorplan_dataset.py
│   │   │   └── floorplan_constrained_dataset.py
│   │   └── configs/
│   │       ├── config_constrained.yaml
│   │       └── experiment/floorplan_constrained.yaml
├── Interface/                 # Django web interface + Graph2Plan GNN
│   ├── model/                 # Graph2Plan model code and weights
│   └── static/Data/           # Test and train pkl data
├── scripts/
│   ├── inference_constrained.py   # CLI inference
│   ├── evaluate.py                # Full evaluation benchmark
│   ├── generate_tf.py             # Pre-compute TF descriptors
│   ├── run_training_constrained.sh
│   └── run_evaluation.sh
├── Dockerfile                 # Multi-mode Docker image
├── requirements.txt
└── DataPreparation/           # Data conversion utilities
```

---

## Installation

### Requirements

- Python 3.9
- CUDA 12.1 (for GPU training)
- PyTorch 2.1.0

```bash
conda create -n digress python=3.9
conda activate digress

# PyTorch + CUDA
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# PyG extensions (must match torch + cuda version exactly)
pip install torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Project dependencies
pip install -r requirements.txt
```

---

## Data Preparation

Download the RPLAN dataset (Graph2Plan format):

```bash
wget https://github.com/HanHan55/Graph2plan/releases/download/data/Data.zip
unzip Data.zip
```

Copy the required files into the raw data directory:

```bash
mkdir -p DiGress/data/floorplan/raw
cp path/to/extracted/data_train_converted.pkl DiGress/data/floorplan/raw/
```

Pre-compute the boundary TF descriptors (only needed once):

```bash
python scripts/generate_tf.py \
    --pkl DiGress/data/floorplan/raw/data_train_converted.pkl \
    --out DiGress/data/floorplan/raw/tf_train.npy
```

---

## Training

### Constrained Model

The constrained model is conditioned on boundary shape plus room count and adjacency constraints, with classifier-free guidance dropout (15% of training steps drop the constraint signal so the model also learns unconditional generation).

```bash
cd DiGress
PYTHONPATH=src python src/main.py --config-name config_constrained
```

Key config options (edit `DiGress/configs/experiment/floorplan_constrained.yaml`):

| Parameter | Default | Description |
|---|---|---|
| `train.n_epochs` | 100 | Number of training epochs |
| `train.batch_size` | 32 | Batch size |
| `train.lr` | 0.0002 | Learning rate |
| `model.diffusion_steps` | 500 | Number of diffusion timesteps |
| `model.n_layers` | 6 | Transformer layers |
| `dataset.cfg_p_uncond` | 0.15 | CFG constraint dropout rate |

Checkpoints are saved to `DiGress/checkpoints/floorplan_constrained/`.

To resume training from a checkpoint:

```bash
cd DiGress
PYTHONPATH=src python src/main.py --config-name config_constrained \
    general.resume=checkpoints/floorplan_constrained/last.ckpt
```

---

## Inference

Run constrained inference from the command line. The script accepts a boundary polygon and optional room count constraints.

```bash
python scripts/inference_constrained.py \
    --ckpt checkpoints/floorplan_constrained/last.ckpt \
    --boundary "0,0 200,0 200,150 0,150" \
    --rooms "LivingRoom:1,Kitchen:1,Bathroom:2,MasterRoom:1" \
    --num-samples 4
```

**Boundary-only (no room constraints):**

```bash
python scripts/inference_constrained.py \
    --ckpt checkpoints/floorplan_constrained/last.ckpt \
    --boundary "0,0 200,0 200,150 0,150"
```

**Arguments:**

| Argument | Description |
|---|---|
| `--ckpt` | Path to `.ckpt` checkpoint file |
| `--boundary` | Space-separated `x,y` polygon vertices |
| `--rooms` | Comma-separated `RoomType:count` pairs (optional) |
| `--num-samples` | Number of samples to generate (default: 4) |
| `--debug` | Print per-step diffusion stats |

**Available room types:** `LivingRoom`, `MasterRoom`, `Kitchen`, `Bathroom`, `DiningRoom`, `ChildRoom`, `StudyRoom`, `SecondRoom`, `GuestRoom`, `Balcony`, `Entrance`, `Storage`, `Wall`

---

## Evaluation

Runs the full benchmark comparing the DiGress model against the retrieval baseline on the held-out test split.

```bash
python scripts/evaluate.py \
    --ckpt checkpoints/floorplan_constrained/last.ckpt \
    --num_samples 500 \
    --out results/
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--ckpt` | required | Path to `.ckpt` checkpoint |
| `--num_samples` | 5 | Number of test samples (use 500+ for reliable FID) |
| `--out` | `results/` | Output directory for results |
| `--seed` | 42 | Random seed |

**Metrics reported:**

| Metric | Direction | Description |
|---|---|---|
| FID | ↓ | Fréchet Inception Distance on rendered 128×128 floor plan images |
| KID | ↓ | Kernel Inception Distance on same |
| Connectivity rate | ↑ | Fraction of generated graphs that are fully connected |
| Graph validity | ↑ | Connected + required rooms (LivingRoom, Bathroom) + no isolated nodes |
| Room-type accuracy | ↑ | Intersection-over-GT of predicted vs. GT room-type multiset |
| Room-type F1 | ↑ | Harmonic mean of room-type precision and recall |
| Node count overlap | ↑ | `min(n_pred, n_gt) / max(n_pred, n_gt)` |
| Count satisfaction | ↑ | Fraction of constrained room types with exact count match (constrained model) |
| Adj satisfaction | ↑ | Fraction of GT constrained adjacencies reproduced (constrained model) |
| Pixel mIoU | ↑ | Mean per-class pixel IoU vs. GT segmentation |

Results are saved to `results/eval_results.txt` (human-readable) and `results/eval_results.json` (raw per-sample arrays).

---

## Web Interface

The Django interface lets you draw a building boundary, optionally edit the generated room graph, and view the rendered floor plan.

### Setup

```bash
cd Interface
pip install django

# Start the server
python manage.py runserver
```

Then open `http://127.0.0.1:8000` in your browser.

### Loading a DiGress Checkpoint

The interface loads the DiGress model via the bridge at `Interface/model/digress_bridge.py`. Set the checkpoint path before starting the server:

```bash
# Set the checkpoint path via environment variable
export DIGRESS_CKPT=path/to/last-v1.ckpt
python manage.py runserver
```

Or edit `Interface/model/digress_bridge.py` and set `CKPT_PATH` directly.

---

## Docker (Northflank / Cloud)

A single Docker image supports training and evaluation via the `TRAINING_MODE` environment variable.

### Build

```bash
docker build -t floorplan .
```

### Run Evaluation (default)

```bash
docker run --gpus all \
    -v /your/storage:/mnt/storage \
    floorplan
```

The checkpoint is downloaded automatically from HuggingFace on first run and cached on the volume.

### Run Training

```bash
# Constrained model
docker run --gpus all \
    -v /your/storage:/mnt/storage \
    -e TRAINING_MODE=constrained \
    floorplan

# Baseline model (boundary only, no constraints)
docker run --gpus all \
    -v /your/storage:/mnt/storage \
    -e TRAINING_MODE=baseline \
    floorplan
```

Checkpoints and processed data are stored on the mounted volume so they persist across container restarts.

---

## Citation

This project builds on [DiGress](https://github.com/cvignac/DiGress) and [Graph2Plan](https://vcc.tech/research/2020/Graph2Plan).

```bibtex
@article{vignac2022digress,
  title={DiGress: Discrete Denoising diffusion for graph generation},
  author={Vignac, Clement and Krawczuk, Igor and Siraudin, Antoine and Wang, Bohan and Cevher, Volkan and Frossard, Pascal},
  journal={arXiv preprint arXiv:2209.14734},
  year={2022}
}

@article{hu2020graph2plan,
  title={Graph2Plan: Learning Floorplan Generation from Layout Graphs},
  author={Hu, Ruizhen and Huang, Zeyu and Tang, Yuhan and Van Kaick, Oliver and Zhang, Hao and Huang, Hui},
  journal={ACM Transactions on Graphics},
  year={2020}
}
```
