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
# TRAINING_MODE: "baseline" | "constrained"  (default: constrained)
#
# Example run:
#   docker run --gpus all \
#     -v /your/host/storage:/mnt/storage \
#     -e TRAINING_MODE=baseline \
#     <image>
# ─────────────────────────────────────────────────────────────────────────────

ENV STORAGE_PATH=/mnt/storage
ENV TRAINING_MODE=constrained

CMD ["/bin/bash", "-c", "\
  if [ \"$TRAINING_MODE\" = 'baseline' ]; then \
    exec /app/scripts/run_training.sh; \
  else \
    exec /app/scripts/run_training_constrained.sh; \
  fi"]
