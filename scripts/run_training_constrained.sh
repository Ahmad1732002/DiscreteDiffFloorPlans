#!/bin/bash
set -e

STORAGE_PATH="${STORAGE_PATH:-/mnt/storage}"
DATA_DIR="$STORAGE_PATH/data/floorplan/raw"
DATA_URL="https://github.com/HanHan55/Graph2plan/releases/download/data/Data.zip"

export STORAGE_PATH

mkdir -p "$DATA_DIR"

# Download data only if not already on the volume
if [ ! -f "$DATA_DIR/data_train_converted.pkl" ]; then
    echo "=== Downloading Data.zip ==="
    wget -q --show-progress -O "$STORAGE_PATH/Data.zip" "$DATA_URL"

    echo "=== Extracting ==="
    unzip -q "$STORAGE_PATH/Data.zip" -d "$STORAGE_PATH/extracted"

    echo "=== Copying data files ==="
    find "$STORAGE_PATH/extracted" -name "data_train_converted.pkl" -exec cp {} "$DATA_DIR/" \;

    echo "=== Cleaning up zip ==="
    rm -rf "$STORAGE_PATH/Data.zip" "$STORAGE_PATH/extracted"
else
    echo "=== Data already on volume, skipping download ==="
fi

# Generate tf_train.npy only if not already present
if [ ! -f "$DATA_DIR/tf_train.npy" ]; then
    echo "=== Generating tf_train.npy ==="
    python /app/scripts/generate_tf.py \
        --pkl "$DATA_DIR/data_train_converted.pkl" \
        --out "$DATA_DIR/tf_train.npy"
else
    echo "=== tf_train.npy already on volume, skipping ==="
fi

# Point DiGress dataset config at the volume data
mkdir -p /app/DiGress/data/floorplan
ln -sfn "$DATA_DIR" /app/DiGress/data/floorplan/raw

# Store processed .pt files on volume so Docker rebuilds don't re-process
mkdir -p "$STORAGE_PATH/data/floorplan/processed"
ln -sfn "$STORAGE_PATH/data/floorplan/processed" /app/DiGress/data/floorplan/processed

# Delete stale processed files for constrained dataset so they are re-built
# (needed if switching from unconstrained to constrained run)
for split in train val test; do
    PROC_FILE="$STORAGE_PATH/data/floorplan/processed/floorplan_constrained_${split}.pt"
    if [ -f "$PROC_FILE" ]; then
        echo "=== Removing stale processed file: $PROC_FILE ==="
        rm -f "$PROC_FILE"
    fi
done

echo "=== Starting DiGress constrained training (CFG, 100 epochs) ==="
cd /app/DiGress
PYTHONPATH=src python src/main.py --config-name config_constrained
