#!/usr/bin/env python3
"""
Build a small Cardinal-compatible dataset from SQLStorm.

1. Load benchmark CSV results (runtime data)
2. Match queries to SQL text
3. Generate EXPLAIN (FORMAT JSON) plans in PostgreSQL
4. Compute rewards and save as Parquet
"""

import os
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# Optional PostgreSQL import
try:
    import psycopg2
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False
    print("⚠️  psycopg2 not found. Will skip plan generation step.")

# === CONFIG ===
DATASET = "stackoverflow"       # dataset to use
VERSION = "v1.0"
DB_NAME = os.getenv("POSTGRES_DB", "stackoverflow")
DB_USER = os.getenv("POSTGRES_USER", "cardinal-2026")

# Subset parameters
SUBSET_SIZE = 3000               # number of queries to sample
SUBSET_START = 0                 # starting index (0 = from beginning)
SUBSET_METHOD = "sequential"     # "sequential", "random", or "by_time"
RANDOM_SEED = 42                 # for reproducible random sampling

OUTPUT_DIR = Path("cardinal_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

# === STEP 1: Load results CSV (runtime info) ===
print("🔹 Loading benchmark results...")

# Check multiple possible directories for results (prefer analysis folder for timing data)
possible_dirs = [
    Path("analysis/features/v1.0"),
    Path("results/v1.0"),
    Path("results")
]

results_files = []
for d in possible_dirs:
    results_files += list(d.rglob(f"{DATASET}*.csv*"))

if not results_files:
    print("❌ No results CSV found!")
    print("\n🔧 TROUBLESHOOTING:")
    print("1. Make sure you're running from the SQLStorm root directory")
    print("2. If you cloned from GitHub, the data files are missing due to Git LFS issues.")
    print("   Download and extract: sqlstorm_data.zip")
    print("   This will create the missing analysis/features/v1.0/ directory with data.")
    print("3. Check that these directories exist:")
    for d in possible_dirs:
        exists = "✅" if d.exists() else "❌"
        print(f"   {exists} {d}")
    print(f"\n4. Looking for files matching: {DATASET}*.csv*")
    raise FileNotFoundError("No results CSV found. See troubleshooting steps above.")

# Prefer analysis folder files (they have proper timing data), but avoid expressions/operators files
analysis_files = [f for f in results_files if "analysis" in str(f) and "expressions" not in str(f) and "operators" not in str(f)]
uncompressed = [f for f in results_files if f.suffix == '.csv']
results_path = analysis_files[0] if analysis_files else (uncompressed[0] if uncompressed else results_files[0])
print(f"Found results file: {results_path}")

df = pd.read_csv(results_path)
# Filter for successful queries (state column instead of status)
df = df[df["state"].str.lower() == "success"].copy()
print(f"✅ Loaded {len(df)} successful results")

# === STEP 2: Load SQL query text ===
print("🔹 Loading SQL queries...")
query_dir = Path(VERSION) / DATASET / "queries"
queries = {p.stem: p.read_text() for p in query_dir.glob("*.sql")}
# Strip .sql extension from query names for matching
df["query_stem"] = df["query"].str.replace(".sql", "", regex=False)
df["sql_text"] = df["query_stem"].map(queries)
print(f"✅ Attached SQL text for {df['sql_text'].notna().sum()} / {len(df)} queries")

# --- Select subset of queries ---
print(f"🔹 Selecting subset from {len(df)} total queries...")
print(f"   Method: {SUBSET_METHOD}, Size: {SUBSET_SIZE}, Start: {SUBSET_START}")

if SUBSET_METHOD == "sequential":
    subset_df = df.iloc[SUBSET_START:SUBSET_START + SUBSET_SIZE].copy()
elif SUBSET_METHOD == "random":
    import numpy as np
    np.random.seed(RANDOM_SEED)
    indices = np.random.choice(len(df), size=min(SUBSET_SIZE, len(df)), replace=False)
    subset_df = df.iloc[indices].copy()
elif SUBSET_METHOD == "by_time":
    # Sort by execution time and take a range
    df_sorted = df.sort_values("time" if "time" in df.columns else "execution")
    subset_df = df_sorted.iloc[SUBSET_START:SUBSET_START + SUBSET_SIZE].copy()
else:
    raise ValueError(f"Unknown SUBSET_METHOD: {SUBSET_METHOD}")

print(f"✅ Selected {len(subset_df)} queries for plan extraction")

# === STEP 3: Generate EXPLAIN JSON plans from PostgreSQL ===
if HAS_PSYCOPG2:
    print("🔹 Generating query plans from PostgreSQL...")
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname=DB_NAME,
            user=DB_USER,
            password=os.getenv("POSTGRES_PASSWORD", ""),
        )
        cur = conn.cursor()

        plans = {}
        failed = []

        for qid, sql in tqdm(subset_df.set_index("query")["sql_text"].items(), total=len(subset_df)):
            try:
                cur.execute(f"EXPLAIN (FORMAT JSON) {sql}")
                plan = cur.fetchall()[0][0]  # PostgreSQL returns JSON as list
                plans[qid] = plan
            except Exception as e:
                # Rollback transaction to prevent cascade failures
                conn.rollback()
                plans[qid] = None
                failed.append((qid, str(e)))

        conn.close()
        print(f"✅ Generated plans for {len(plans) - len(failed)} / {len(plans)} queries")
        if failed:
            print(f"⚠️ {len(failed)} queries failed during EXPLAIN (showing first 3):")
            for qid, err in failed[:3]:
                print(f"   {qid}: {err[:120]}")

        subset_df["plan_json"] = subset_df["query"].map(plans)
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        print("🔹 Continuing without plans...")
        subset_df["plan_json"] = None
else:
    print("🔹 Skipping plan generation (psycopg2 not available)")
    subset_df["plan_json"] = None

# === STEP 4: Compute reward and save dataset ===
print("🔹 Computing rewards and saving dataset...")
# Use 'time' column if available (from analysis folder), otherwise 'execution'
time_col = "time" if "time" in subset_df.columns else "execution"
subset_df = subset_df.dropna(subset=[time_col])
subset_df["reward"] = -subset_df[time_col] / subset_df[time_col].max()

df_final = subset_df[["query", "sql_text", "plan_json", time_col, "reward"]]
df_final = df_final.rename(columns={time_col: "execution_time"})

# Generate output filename with parameters
method_suffix = f"_{SUBSET_METHOD}" if SUBSET_METHOD != "sequential" else ""
start_suffix = f"_start{SUBSET_START}" if SUBSET_START != 0 else ""
filename_base = f"{DATASET}_n{SUBSET_SIZE}{method_suffix}{start_suffix}"

# Save Parquet
parquet_file = OUTPUT_DIR / f"{filename_base}.parquet"
df_final.to_parquet(parquet_file)

# Save CSV (convert plan_json to string)
import json
import numpy as np

def convert_plan_to_string(plan):
    if plan is None:
        return None
    try:
        if isinstance(plan, np.ndarray):
            plan = plan.tolist()
        return json.dumps(plan, default=str)
    except:
        return str(plan)

df_csv = df_final.copy()
df_csv['plan_json'] = df_csv['plan_json'].apply(convert_plan_to_string)
csv_file = OUTPUT_DIR / f"{filename_base}.csv"
df_csv.to_csv(csv_file, index=False)

print(f"\n✅ Saved {len(df_final)} queries to:")
print(f"   📊 Parquet: {parquet_file}")
print(f"   📄 CSV: {csv_file}")
print(f"\nDataset summary:")
print(f"   Queries with plans: {df_final['plan_json'].notna().sum()}")
print(f"   Execution time range: {df_final['execution_time'].min():.2f} - {df_final['execution_time'].max():.2f} ms")
print(f"   Reward range: {df_final['reward'].min():.3f} - {df_final['reward'].max():.3f}")
print(f"\nFirst 3 rows:")
print(df_final[['query', 'execution_time', 'reward']].head(3))
