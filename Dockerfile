FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y wget unzip git && rm -rf /var/lib/apt/lists/*

# Copy repo (data is excluded via .dockerignore)
COPY . .

# Install DiGress dependencies
RUN pip install --no-cache-dir \
    torch_geometric==2.3.1 \
    pytorch_lightning==2.0.4 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    wandb==0.15.4 \
    networkx==2.8.7 \
    torchmetrics==0.11.4 \
    pyemd==1.0.0 \
    PyGSP==0.5.1 \
    overrides==7.3.1 \
    imageio==2.31.1 \
    scipy==1.11.0 \
    numpy==1.23 \
    tqdm \
    pandas

RUN chmod +x /app/scripts/run_training.sh

CMD ["/app/scripts/run_training.sh"]
