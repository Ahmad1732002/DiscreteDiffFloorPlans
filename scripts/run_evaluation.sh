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

# ── All other data is baked into the image ────────────────────────────────────
G2P_MODEL="/app/Interface/model/model.pth"
TEST_DATA="/app/Interface/static/Data/data_test_converted.pkl"
TRAIN_DATA="/app/Interface/static/Data/data_train_converted.pkl"

NUM_EVAL_SAMPLES=500
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
