#!/bin/bash
#SBATCH --job-name=rl_grpo_sqlstorm
#SBATCH --output=logs/slurm_rl_grpo_%j.out
#SBATCH --error=logs/slurm_rl_grpo_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#
# GRPO RL fine-tuning (3B SFT checkpoint: abharadwaj123/llama3-sql2plan).
# Submit from repo root: sbatch scripts/slurm_rl_grpo.sbatch
# Requires: .env with POSTGRES_* for reward (query execution).

set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"
mkdir -p logs

if [[ -d "$PROJECT_ROOT/.venv" ]]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
elif [[ -n "$CONDA_PREFIX" ]]; then
  :
else
  echo "Activate your Python env (venv or conda) before sbatch."
  exit 1
fi

OUTPUT_DIR="${RL_OUTPUT_DIR:-$PROJECT_ROOT/rl_grpo_output}"
DATASET_PATH="${RL_DATASET_PATH:-$PROJECT_ROOT/cardinal_dataset/stackoverflow_n3000.csv}"
EXTRA_ARGS="${RL_EXTRA_ARGS:-}"

# 3B SFT checkpoint (plan JSON output); base model required for PEFT
python "$PROJECT_ROOT/rl_finetune_grpo.py" \
  --sft_checkpoint "abharadwaj123/llama3-sql2plan" \
  --base_model "meta-llama/Llama-3.2-3B" \
  --completion_format plan \
  --dataset_path "$DATASET_PATH" \
  --output_dir "$OUTPUT_DIR" \
  $EXTRA_ARGS

echo "Done."
