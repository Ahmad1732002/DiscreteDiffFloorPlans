#!/bin/bash
#SBATCH -A iscrb_wearusfm
#SBATCH -p boost_usr_prod
#SBATCH --time 23:50:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=123500
#SBATCH --job-name=digress_floorplan
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=afraij@ethz.ch
#SBATCH --output=/leonardo/home/userexternal/afraij00/DiscreteDiffFloorPlans/logs/%x_%j.out
#SBATCH --error=/leonardo/home/userexternal/afraij00/DiscreteDiffFloorPlans/logs/%x_%j.err

DATA_DIR="/leonardo_scratch/fast/IscrB_WearUsFM/dlms/floorplan/raw"
STORAGE_PATH="/leonardo_scratch/fast/IscrB_WearUsFM/dlms/floorplan"
REPO_DIR="/leonardo/home/userexternal/afraij00/DiscreteDiffFloorPlans"

export STORAGE_PATH

mkdir -p "$REPO_DIR/logs"

module load cuda/12.2
conda activate digress

# Symlink data into DiGress expected location
mkdir -p "$REPO_DIR/DiGress/data/floorplan"
ln -sfn "$DATA_DIR" "$REPO_DIR/DiGress/data/floorplan/raw"

echo "=== Starting DiGress training ==="
cd "$REPO_DIR/DiGress"
PYTHONPATH=src python src/main.py
