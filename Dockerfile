FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    wget unzip git libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy repo first (data excluded via .dockerignore)
COPY . .

# PyG C++ extensions pinned to torch 2.1.0 + cu121
RUN pip install --no-cache-dir \
    torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# DiGress Python dependencies
RUN pip install --no-cache-dir -r /app/DiGress/requirements.txt

# rdkit (needed by molecular metrics even when not used)
RUN pip install --no-cache-dir rdkit

# Make all scripts executable
RUN chmod +x /app/scripts/*.sh

# ── Runtime configuration ─────────────────────────────────────────────────────
# STORAGE_PATH : where data + checkpoints are stored (mount a volume here)
#   default: /mnt/storage
# TRAINING_MODE: "baseline" | "constrained" | "small"  (default: constrained)
#
# For TRAINING_MODE=small the following env vars are also honoured:
#   N_TRAIN      number of training samples  (default 1000)
#   N_VAL        number of val samples       (default 100)
#   EPOCHS       training epochs             (default 100)
#   BATCH_SIZE                               (default 32)
#   VAL_EVERY    validate every N epochs     (default 5)
#   LR           learning rate               (default 0.0002)
#   WEIGHT_DECAY                             (default 0.0001)
#   RUN_NAME     name for checkpoints dir    (default small_test)
#
# Example runs:
#   docker run --gpus all -v /storage:/mnt/storage -e TRAINING_MODE=small <image>
#   docker run --gpus all -v /storage:/mnt/storage \
#     -e TRAINING_MODE=small -e N_TRAIN=1000 -e N_VAL=100 -e EPOCHS=100 <image>
# ─────────────────────────────────────────────────────────────────────────────

ENV STORAGE_PATH=/mnt/storage
ENV TRAINING_MODE=small

CMD ["/bin/bash", "-c", "\
  if [ \"$TRAINING_MODE\" = 'baseline' ]; then \
    exec /app/scripts/run_training.sh; \
  elif [ \"$TRAINING_MODE\" = 'small' ]; then \
    exec /app/scripts/run_training_small.sh; \
  else \
    exec /app/scripts/run_training_constrained.sh; \
  fi"]
