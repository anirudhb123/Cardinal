# Checklist: Run GRPO RL (locally or Slurm)

## In the repo (you have these)

- [x] **`rl_finetune_grpo.py`** – GRPO training, 3-signal reward (latency, CPU, memory)
- [x] **`query_execution/`** – `run_query_with_metrics`, `plan_to_hints` for reward
- [x] **`cardinal_dataset/stackoverflow_n3000.csv`** – dataset with `sql_text` (and optional `plan_json`)
- [x] **`scripts/slurm_rl_grpo.sbatch`** – Slurm job for RL
- [x] **`requirements.txt`** – includes `trl`, `peft`, `transformers`, `datasets`, `psycopg2-binary`, `python-dotenv`

## Before you run (do these once)

### 1. Python env and deps

```bash
cd /path/to/SQLStorm
source .venv/bin/activate   # or your conda env
pip install -r requirements.txt
```

### 2. Hugging Face (for Llama-3.2-3B and your SFT checkpoint)

```bash
huggingface-cli login
```

- Accept the **Meta Llama 3.2** license on the model page if gated: https://huggingface.co/meta-llama/Llama-3.2-3B  
- Your SFT model is public: `abharadwaj123/llama3-sql2plan`

### 3. PostgreSQL and `.env`

- **PostgreSQL** must be running (local or remote).
- Create a database (e.g. `cardinal_test`) and load schema/data so the **Stack Overflow–style queries** in the dataset can run (or at least not crash; failed runs get `reward_on_error`).
- In the **repo root**, create **`.env`** (do not commit it):

```bash
POSTGRES_HOST=localhost
POSTGRES_DB=cardinal_test
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_actual_password
POSTGRES_PORT=5432
```

## Run

**Local (e.g. M1 16GB):**

```bash
python rl_finetune_grpo.py \
  --sft_checkpoint abharadwaj123/llama3-sql2plan \
  --base_model meta-llama/Llama-3.2-3B \
  --completion_format plan \
  --batch_size 1 \
  --num_generations 2 \
  --subset_size 50
```

**Slurm:**

```bash
mkdir -p logs
sbatch scripts/slurm_rl_grpo.sbatch
```

Optional overrides: `--dataset_path`, `--output_dir`, `--subset_size`, `--num_generations`, `--lr`, etc. See `python rl_finetune_grpo.py --help`.
