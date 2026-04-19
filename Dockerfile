FROM --platform=linux/amd64 nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04

WORKDIR /app

# System deps + Python
RUN apt-get update && apt-get install -y \
    python3.9 python3.9-dev python3-pip \
    wget unzip git libxrender1 libxext6 libgl1 \
    && ln -sf /usr/bin/python3.9 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 2.1.0 + CUDA 12.1
RUN pip install --no-cache-dir \
    torch==2.1.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy repo first (data excluded via .dockerignore)
COPY . .

# PyG C++ extensions pinned to torch 2.1.0 + cu121
RUN pip install --no-cache-dir \
    torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# torchvision (base runtime image omits it; needed for RoIAlign in Graph2Plan)
RUN pip install --no-cache-dir torchvision==0.16.0

# DiGress Python dependencies
RUN pip install --no-cache-dir -r /app/DiGress/requirements.txt

# torchmetrics image extras (FID / KID in evaluate.py)
RUN pip install --no-cache-dir "torchmetrics[image]==0.11.4"

# rdkit (needed by molecular metrics even when not used)
RUN pip install --no-cache-dir rdkit

# opencv (headless — no display needed in container)
RUN pip install --no-cache-dir opencv-python-headless

# Make all scripts executable
RUN chmod +x /app/scripts/*.sh

# ── Runtime configuration ─────────────────────────────────────────────────────
# STORAGE_PATH  : volume mount point for data + checkpoints  (default: /mnt/storage)
# TRAINING_MODE : "baseline" | "constrained" | "eval"        (default: constrained)
#
# Extra env vars for eval mode:
#   CKPT_PATH         path to .ckpt file      (default: $STORAGE_PATH/checkpoints/last.ckpt)
#   NUM_EVAL_SAMPLES  test samples to run     (default: 500)
#   EVAL_OUT          results output dir      (default: $STORAGE_PATH/eval_results)
#   EVAL_SEED         random seed             (default: 42)
#
# Example runs:
#   # train (constrained, default)
#   docker run --gpus all -v /storage:/mnt/storage <image>
#
#   # train (baseline)
#   docker run --gpus all -v /storage:/mnt/storage -e TRAINING_MODE=baseline <image>
#
#   # evaluate a checkpoint
#   docker run --gpus all -v /storage:/mnt/storage \
#     -e TRAINING_MODE=eval \
#     -e CKPT_PATH=/mnt/storage/checkpoints/epoch=99.ckpt \
#     -e NUM_EVAL_SAMPLES=500 \
#     <image>
# ─────────────────────────────────────────────────────────────────────────────

ENV STORAGE_PATH=/mnt/storage
ENV TRAINING_MODE=eval

CMD ["/bin/bash", "-c", "\
  if [ \"$TRAINING_MODE\" = 'baseline' ]; then \
    exec /app/scripts/run_training.sh; \
  elif [ \"$TRAINING_MODE\" = 'eval' ]; then \
    exec /app/scripts/run_evaluation.sh; \
  else \
    exec /app/scripts/run_training_constrained.sh; \
  fi"]
