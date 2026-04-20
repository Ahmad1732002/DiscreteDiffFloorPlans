#!/bin/bash
set -e

STORAGE_PATH="${STORAGE_PATH:-/mnt/storage}"

# ── DiGress checkpoint ────────────────────────────────────────────────────────
HF_URL="https://huggingface.co/ahmadfraij/disdif/resolve/main/last-v1.ckpt"
CKPT_PATH="$STORAGE_PATH/checkpoints/last-v1.ckpt"

mkdir -p "$(dirname "$CKPT_PATH")"

if [ ! -f "$CKPT_PATH" ]; then
    echo "=== Downloading DiGress checkpoint from HuggingFace ==="
    wget -q --show-progress -O "$CKPT_PATH" "$HF_URL"
    echo "=== Checkpoint saved to $CKPT_PATH ==="
else
    echo "=== Checkpoint already present, skipping download ==="
fi

# ── Data: download from HuggingFace if not already on volume ─────────────────
G2P_MODEL="/app/Interface/model/model.pth"
TEST_DATA="$STORAGE_PATH/data/data_test_converted.pkl"
TRAIN_DATA="$STORAGE_PATH/data/data_train_converted.pkl"

mkdir -p "$STORAGE_PATH/data"

if [ ! -f "$TEST_DATA" ]; then
    echo "=== Downloading test data from HuggingFace ==="
    wget -q --show-progress -O "$TEST_DATA" \
        "https://huggingface.co/ahmadfraij/disdif/resolve/main/data_test_converted.pkl"
else
    echo "=== Test data already present, skipping download ==="
fi

if [ ! -f "$TRAIN_DATA" ]; then
    echo "=== Downloading train data from HuggingFace ==="
    wget -q --show-progress -O "$TRAIN_DATA" \
        "https://huggingface.co/ahmadfraij/disdif/resolve/main/data_train_converted.pkl"
else
    echo "=== Train data already present, skipping download ==="
fi

NUM_EVAL_SAMPLES="${NUM_EVAL_SAMPLES:-2500}"
EVAL_OUT="$STORAGE_PATH/eval_results"
EVAL_SEED=42

mkdir -p "$EVAL_OUT"

echo "=== Starting evaluation ==="
echo "    checkpoint   : $CKPT_PATH"
echo "    test data    : $TEST_DATA"
echo "    train data   : $TRAIN_DATA"
echo "    num samples  : $NUM_EVAL_SAMPLES"
echo "    output dir   : $EVAL_OUT"

python /app/scripts/evaluate.py \
    --ckpt        "$CKPT_PATH" \
    --g2p_model   "$G2P_MODEL" \
    --test_data   "$TEST_DATA" \
    --train_data  "$TRAIN_DATA" \
    --num_samples "$NUM_EVAL_SAMPLES" \
    --out         "$EVAL_OUT" \
    --seed        "$EVAL_SEED"

echo "=== Evaluation complete. Results at $EVAL_OUT ==="
