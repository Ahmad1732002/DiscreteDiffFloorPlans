#!/bin/bash
set -e

STORAGE_PATH="${STORAGE_PATH:-/mnt/storage}"
DATA_DIR="$STORAGE_PATH/data/floorplan/raw"
DATA_URL="https://github.com/HanHan55/Graph2plan/releases/download/data/Data.zip"

N_TRAIN="${N_TRAIN:-1000}"
N_VAL="${N_VAL:-100}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
VAL_EVERY="${VAL_EVERY:-5}"
LR="${LR:-0.0002}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
RUN_NAME="${RUN_NAME:-small_test}"

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

# Symlink data so DiGress config finds it
mkdir -p /app/DiGress/data/floorplan
ln -sfn "$DATA_DIR" /app/DiGress/data/floorplan/raw

mkdir -p "$STORAGE_PATH/data/floorplan/processed"
ln -sfn "$STORAGE_PATH/data/floorplan/processed" /app/DiGress/data/floorplan/processed

echo "=== Starting small-subset training: ${N_TRAIN} train / ${N_VAL} val ==="
python /app/scripts/train_small.py \
    --n_train   "$N_TRAIN" \
    --n_val     "$N_VAL" \
    --epochs    "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --val_every "$VAL_EVERY" \
    --lr        "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --name      "$RUN_NAME"
