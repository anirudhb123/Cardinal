# Slurm scripts

## RL GRPO fine-tuning (3B SFT checkpoint)

Run GRPO with your SFT checkpoint [abharadwaj123/llama3-sql2plan](https://huggingface.co/abharadwaj123/llama3-sql2plan) (Llama-3.2-3B + LoRA):

```bash
mkdir -p logs
sbatch scripts/slurm_rl_grpo.sbatch
```

- **Feasibility:** 3B is fine for GRPO (TRL docs use 0.5B–7B+). One GPU (16–24GB) is enough; reduce `per_device_train_batch_size` or `num_generations` if OOM.
- **Completion format:** The script uses `--completion_format plan`: the model outputs execution plan JSON (same as SFT); we convert plan → hints via `plan_to_hints`, then run the query with hints and compute reward (latency, CPU, memory).
- **PEFT:** The job loads `meta-llama/Llama-3.2-3B` then applies the LoRA adapter from `abharadwaj123/llama3-sql2plan`. You must be logged in and allowed for the gated base: `huggingface-cli login`.
- **PostgreSQL:** Reward requires a running Postgres DB and `.env` with `POSTGRES_*`. Queries from the dataset are executed (with optional hints from the plan); failures get `reward_on_error`.

**Running locally on M1 with 16GB:** If SFT worked on the same machine, GRPO can often run with small settings. Use batch 1 and 2 generations so memory stays low; gradient checkpointing is enabled by default. On **Mac** skip `--use_4bit` (bitsandbytes is CUDA-only). Example:

```bash
python rl_finetune_grpo.py \
  --sft_checkpoint abharadwaj123/llama3-sql2plan \
  --base_model meta-llama/Llama-3.2-3B \
  --completion_format plan \
  --batch_size 1 \
  --num_generations 2 \
  --subset_size 50
```

If you hit OOM, try `--subset_size 20` or run on Slurm/cloud instead.

Overrides (optional):

```bash
RL_EXTRA_ARGS="--subset_size 200 --num_generations 4" sbatch scripts/slurm_rl_grpo.sbatch
```

---

## SFT fine-tuning

Run SFT on a Slurm cluster:

```bash
# From repo root
mkdir -p logs
sbatch scripts/slurm_sft.sbatch
```

### Resource defaults (edit in `slurm_sft.sbatch`)

- 1 GPU, 8 CPUs, 32G RAM, 24h
- Partition: `gpu` (change `#SBATCH --partition` for your cluster)
- Outputs: `logs/slurm_sft_<jobid>.out`, `logs/slurm_sft_<jobid>.err`

### Multi-GPU

In `slurm_sft.sbatch` set e.g.:

```bash
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
```

The script uses `accelerate launch` and will use all allocated GPUs.

### Overriding training options

Use environment variables when submitting:

```bash
SFT_OUTPUT_DIR=/path/to/output \
SFT_DATASET_PATH=cardinal_dataset/stackoverflow_n3000.csv \
SFT_EXTRA_ARGS="--subset_size 500 --epochs 2 --no_flash_attention" \
  sbatch scripts/slurm_sft.sbatch
```

### One-time setup

If `accelerate` has never been configured, run once interactively (e.g. on a login node):

```bash
accelerate config
```

Choose defaults (single machine, multi-GPU if needed, no DeepSpeed unless you want it). Or the script will use inline `--num_processes` and typical defaults.
