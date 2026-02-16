# Query Execution (Cardinal-style)

PostgreSQL query execution and benchmarking for **RL fine-tuning**: run SQL, collect execution plans, measure runtimes, and convert plans to `pg_hint_plan` hints.

Source: [Cath3333/Cardinal – query-execution](https://github.com/Cath3333/Cardinal/tree/main/query-execution).

## Layout

- **`single_query/`** – run one query, get plan + hints, benchmark
  - `config.py` – DB config (env: `POSTGRES_*`)
  - `query_executor.py` – `SimpleQueryExecutor`: execute, explain, hints, benchmark
  - `plan_to_hints.py` – EXPLAIN JSON → pg_hint_plan hint string
  - `executor_cli.py` – interactive or single-query CLI
- **`batch_query/`** – run many queries from a CSV (e.g. for reward computation)
  - `batch_executor.py` – parallel batch over CSV with optional `plan_json` hints

## Setup

1. **Env** (optional): create `.env` in repo root or set:

   ```bash
   POSTGRES_HOST=localhost
   POSTGRES_DB=cardinal_test
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=your_password
   POSTGRES_PORT=5432
   ```

2. **Run single-query CLI** (from `query_execution/single_query`):

   ```bash
   cd query_execution/single_query
   python executor_cli.py                    # interactive
   python executor_cli.py -q "SELECT 1"       # one query
   python executor_cli.py -q "SELECT ..." --hints "/*+ SeqScan(t) */" -i 5
   ```

3. **Run batch executor** (from repo root):

   ```bash
   python query_execution/batch_query/batch_executor.py cardinal_dataset/stackoverflow_n3000.csv -o out.csv
   python query_execution/batch_query/batch_executor.py my.csv --use-hints --iterations 3 --workers 4
   ```

   CSV must have a **`query`** or **`sql_text`** column; optional **`plan_json`** for hint-based execution. Output adds **`execution_time`** (ms).

## Use in RL fine-tuning

- **Reward**: e.g. negative execution time from `SimpleQueryExecutor.execute_query()` or batch CSV `execution_time`.
- **Plans**: `get_execution_plan(query, analyze=True)` and `get_plan_and_hints()` for plan + hint string.
- **Batch**: use `batch_executor.py` to fill execution times for a dataset CSV, then use that column as reward (or part of it).

## Optional: pg_hint_plan

For hint-guided execution, install and enable `pg_hint_plan` in your PostgreSQL DB. Without it, execution still works; hint application will be skipped or fail gracefully.
