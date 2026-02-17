#!/bin/bash
#SBATCH --job-name=sft_sqlstorm
#SBATCH --output=logs/slurm_sft_%j.out
#SBATCH --error=logs/slurm_sft_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#
# Submit from repo root: sbatch scripts/slurm_sft.sbatch
# Multi-GPU: set e.g. #SBATCH --gres=gpu:4
# Overrides: SFT_OUTPUT_DIR=... SFT_DATASET_PATH=... SFT_EXTRA_ARGS="--subset_size 100" sbatch scripts/slurm_sft.sbatch

# Optional: uncomment and set for your cluster
# module load cuda/12.1
# module load anaconda3

set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

# Create logs dir if missing
mkdir -p logs

# Activate venv (adjust path if you use conda instead)
if [[ -d "$PROJECT_ROOT/.venv" ]]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
elif [[ -n "$CONDA_PREFIX" ]]; then
  : # already in conda env
else
  echo "Activate your Python env (venv or conda) before sbatch, or edit this script."
  exit 1
fi

# Training args (override via env or edit here)
OUTPUT_DIR="${SFT_OUTPUT_DIR:-$PROJECT_ROOT/sft_output}"
DATASET_PATH="${SFT_DATASET_PATH:-$PROJECT_ROOT/cardinal_dataset/stackoverflow_n3000.csv}"
EXTRA_ARGS="${SFT_EXTRA_ARGS:-}"

# Use accelerate so 1-GPU and multi-GPU both work (Slurm sets CUDA_VISIBLE_DEVICES)
if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
  NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-1}
echo "Using $NUM_GPUS GPU(s)"
accelerate launch --num_processes "$NUM_GPUS" \
  "$PROJECT_ROOT/sft_finetune.py" \
  --output_dir "$OUTPUT_DIR" \
  --dataset_path "$DATASET_PATH" \
  $EXTRA_ARGS

echo "Done."
