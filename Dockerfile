FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y wget unzip git libxrender1 libxext6 && rm -rf /var/lib/apt/lists/*

# Copy repo (data is excluded via .dockerignore)
COPY . .

# Install PyG and its required C++ extensions for PyTorch 2.0.1 + CUDA 11.7
RUN pip install --no-cache-dir \
    torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.0.1+cu117.html

# Install DiGress dependencies
RUN pip install --no-cache-dir -r /app/DiGress/requirements.txt

RUN pip install --no-cache-dir rdkit

RUN chmod +x /app/scripts/run_training.sh

CMD ["/app/scripts/run_training.sh"]
