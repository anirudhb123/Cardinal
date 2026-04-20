# Cardinal

**Cardinal** is a research project for **learning and evaluating language models on PostgreSQL query optimization**. This repository implements the full pipeline: Hugging Face models talk to a live database—generating execution plans (and optional `pg_hint_plan`–style hints), running queries on PostgreSQL, and recording latency, CPU time, and memory. It also includes **supervised fine-tuning (SFT)** and **GRPO reinforcement learning** so models can be trained to improve those metrics on real workloads.

For **goals, system design, and evaluation context** (course group 37), see the **[project deck (PDF)](docs/group37_cardinal.pdf)**.

The Stack Overflow–style **SQLStorm** benchmark artifacts and Cardinal-style dataset export are part of the tooling; naming like `sqlstorm_data.zip` and `convert_to_cardinal.py` reflects that split.

## Pretrained models (Hugging Face)

Published checkpoints from this project:

| Stage | Model |
|--------|--------|
| **SFT** — Llama 3.2 3B + LoRA, SQL → PostgreSQL execution plan (JSON) | [abharadwaj123/llama3-sql2plan](https://huggingface.co/abharadwaj123/llama3-sql2plan) |
| **GRPO** — further trained with GRPO from the SFT adapter | [abharadwaj123/sqlstorm-grpo-plan8192](https://huggingface.co/abharadwaj123/sqlstorm-grpo-plan8192) |

Pass the model id to **`model_evaluator.py`** (`--model …`) or the Streamlit evaluator’s model field.

## What’s in this repo

| Area | Role |
|------|------|
| **`model_evaluator.py`** | CLI: CSV of SQL in → load an HF model → generate plans → run on Postgres → results CSV (times, plans, optional hints). |
| **`app.py`** | Streamlit UI for the same evaluation loop with presets, uploads, and progress. Run: `streamlit run app.py`. |
| **`query_executor.py`**, **`plan_to_hints.py`**, **`query_execution/`** | PostgreSQL execution, `EXPLAIN` plans, and conversion from plans to hint strings for RL/evaluation. |
| **`executor_cli.py`** | Interactive CLI around single-query execution and hint extraction. |
| **`batch_executor.py`** | Batch execution helpers for larger runs. |
| **`sft_finetune.py`** | LoRA/QLoRA SFT on plan/query datasets (see [SFT_README.md](SFT_README.md)). |
| **`rl_finetune_grpo.py`** | GRPO training with rewards from latency, CPU, and memory (see [RUN_RL_CHECKLIST.md](RUN_RL_CHECKLIST.md)). |
| **`convert_to_cardinal.py`** | Build Cardinal-style Parquet/CSV subsets from SQLStorm benchmark outputs under configurable subset rules. |
| **`prompts/`** | YAML prompts (e.g. Stack Overflow schema) for generation. |
| **`scripts/`** | Utilities for analysis, joins, validation, Slurm, etc. |

## Prerequisites

- **Python 3.10+** (recommended)
- **PostgreSQL** with a database that matches your workload (e.g. Stack Overflow schema in `v1.0/stackoverflow/schema.sql`)
- Optional: **NVIDIA GPU** for fast training; CPU/MPS can work but are slower for large models
- Optional: **Hugging Face token** (`HF_TOKEN` or `huggingface-cli login`) for gated models

## Installation

```bash
git clone https://github.com/anirudhb123/sql-storm-dataset-generation.git
cd sql-storm-dataset-generation   # or your clone folder name

python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Alternatively, `./setup.sh` creates a `.venv` and installs dependencies.

## Configuration

1. Copy environment variables for Postgres (example names used by the code):

   ```bash
   # .env in the repo root (do not commit secrets)
   POSTGRES_HOST=localhost
   POSTGRES_DB=stackoverflow          # or cardinal_test, etc.
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=your_password
   POSTGRES_PORT=5432
   ```

2. **`config.py`** defaults read from `.env` via `python-dotenv`. The Streamlit app (`app.py`) can also read the same values.

3. For gated models, set **`HF_TOKEN`** or run `huggingface-cli login`.

## Database setup (minimal)

Example for local development (paths may vary):

```bash
createdb stackoverflow
psql -d stackoverflow -f v1.0/stackoverflow/schema.sql
```

Load data for your benchmark as needed. Several flows expect Stack Overflow–compatible tables when using the bundled prompts and datasets.

Optional: enable **`pg_hint_plan`** if you use hint-based execution; see comments in `config.py` (`HINT_PLAN_CONFIG`).

## Data bundle

If benchmarks refer to CSVs that are not in Git, extract the bundled archive **after clone**:

```bash
unzip sqlstorm_data.zip
```

That layout is what `convert_to_cardinal.py` and some scripts expect when they look for result CSVs.

---

## Workflow: evaluate a model (CLI)

```bash
python model_evaluator.py your_queries.csv -o results.csv \
  --model meta-llama/Llama-3.2-3B \
  --query-column sql_text \
  --workers 4
```

Common flags: `--limit N`, `--use-hints`, `--iterations`, `--verbose`. See `python model_evaluator.py --help`.

## Workflow: evaluate in the browser

```bash
streamlit run app.py
```

Upload a CSV, pick a model preset or enter an HF id, and run. Connection fields can be filled from `.env`.

## Workflow: supervised fine-tuning (SFT)

Trained checkpoint: **[abharadwaj123/llama3-sql2plan](https://huggingface.co/abharadwaj123/llama3-sql2plan)** (see the model card for training data and hyperparameters).

See **[SFT_README.md](SFT_README.md)** for LoRA/QLoRA options, dataset layout (`cardinal_dataset/`, CSV columns), and how to run training locally.

Entry point: `sft_finetune.py`.

## Workflow: GRPO reinforcement learning

Trained checkpoint: **[abharadwaj123/sqlstorm-grpo-plan8192](https://huggingface.co/abharadwaj123/sqlstorm-grpo-plan8192)** (GRPO on top of the SFT adapter; model card cites [GRPO / DeepSeekMath](https://arxiv.org/abs/2402.03300)).

See **[RUN_RL_CHECKLIST.md](RUN_RL_CHECKLIST.md)** for env checks, Postgres, Hugging Face, and Slurm.

Entry point: `rl_finetune_grpo.py`.

---

## Dataset export (`convert_to_cardinal.py`)

Use this when you need **Cardinal-format Parquet/CSV** slices from SQLStorm benchmark outputs (sequential, random, or ordered by execution time).

1. Install: `pip install pandas psycopg2-binary tqdm pyarrow` (already in `requirements.txt`).
2. Point **`DB_USER`** (and DB settings) at your Postgres user—see the script’s top-level constants alongside your `.env`.
3. Adjust **`SUBSET_SIZE`**, **`SUBSET_START`**, **`SUBSET_METHOD`** (`sequential` | `random` | `by_time`), and optional **`RANDOM_SEED`** in `convert_to_cardinal.py`.
4. Run:

   ```bash
   python convert_to_cardinal.py
   ```

Outputs land under **`cardinal_dataset/`** (e.g. `stackoverflow_n100.parquet` / `.csv`). Each row typically includes `query`, `sql_text`, `plan_json`, `execution_time`, and a normalized **`reward`** score.

**Troubleshooting:** “No results CSV found” usually means `sqlstorm_data.zip` was not extracted or paths do not match the expected layout.

---

## License

This project is licensed under the MIT License—see [LICENSE](LICENSE).
