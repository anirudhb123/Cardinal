#!/usr/bin/env python3
"""
Model Evaluator: End-to-end pipeline for LLM query plan generation and execution benchmarking.

Input:  CSV of SQL queries + model name
Output: CSV with queries, generated plans, execution time, CPU time, and peak memory usage
"""

import argparse
import json
import os
import resource
import sys
import time
import tracemalloc
from pathlib import Path

# Hugging Face: project-local cache and disable Xet to avoid 416 / permission errors
_project_root = Path(__file__).resolve().parent
_hf_base = _project_root / ".cache" / "huggingface"
_hf_base.mkdir(parents=True, exist_ok=True)
if "HUGGINGFACE_HUB_CACHE" not in os.environ:
    _hf_cache = _hf_base / "hub"
    _hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(_hf_cache)
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(_hf_base)
# Disable Xet backend (avoids HTTP 416 Range Not Satisfiable and ~/.cache permission errors)
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Tuple

import pandas as pd
from tqdm import tqdm

from query_executor import SimpleQueryExecutor


# ---------------------------------------------------------------------------
# 1. LLM Plan Generation
# ---------------------------------------------------------------------------

def load_llm_model(model_name):
    """Loads the LLM model using the transformers library."""
    from huggingface_hub import login
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN")
    if token:
        try:
            login(token=token)
        except Exception:
            # Invalid or expired token; continue without login (public models work without auth)
            pass
    print(f"Loading model {model_name}...")
    print("  Step 1/3: Downloading/loading tokenizer (may take a moment on first run)...")
    sys.stdout.flush()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("  Step 2/3: Downloading/loading model weights (may be several GB on first run)...")
    sys.stdout.flush()
    # Prefer bfloat16; GPT-2 and other small/older models often use float16 or float32
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
            )
            break
        except (TypeError, ValueError):
            continue
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    print("  Step 3/3: Model loaded successfully.")
    sys.stdout.flush()
    return model, tokenizer


def generate_query_plan(query, model, tokenizer):
    """Use an LLM to generate a PostgreSQL JSON execution plan for a SQL query."""
    example_sql = """SELECT u.DisplayName, COUNT(p.Id) AS PostCount
FROM Users u
JOIN Posts p ON u.Id = p.OwnerUserId
WHERE u.Reputation > 1000
GROUP BY u.DisplayName
ORDER BY PostCount DESC
LIMIT 10;"""

    example_plan = """[
  {
    "Plan": {
      "Node Type": "Limit",
      "Startup Cost": 51.64,
      "Total Cost": 51.67,
      "Plan Rows": 10,
      "Plans": [
        {
          "Node Type": "Sort",
          "Parent Relationship": "Outer",
          "Sort Key": ["(count(p.id)) DESC"],
          "Plans": [
            {
              "Node Type": "Aggregate",
              "Strategy": "Sorted",
              "Parent Relationship": "Outer",
              "Group Key": ["u.displayname"],
              "Plans": [
                {
                  "Node Type": "Merge Join",
                  "Parent Relationship": "Outer",
                  "Join Type": "Inner",
                  "Merge Cond": "(u.id = p.owneruserid)",
                  "Plans": [
                    {
                      "Node Type": "Sort",
                      "Parent Relationship": "Outer",
                      "Sort Key": ["u.id"],
                      "Plans": [
                        {
                          "Node Type": "Seq Scan",
                          "Parent Relationship": "Outer",
                          "Relation Name": "users",
                          "Alias": "u",
                          "Filter": "(reputation > 1000)"
                        }
                      ]
                    },
                    {
                      "Node Type": "Sort",
                      "Parent Relationship": "Inner",
                      "Sort Key": ["p.owneruserid"],
                      "Plans": [
                        {
                          "Node Type": "Seq Scan",
                          "Parent Relationship": "Outer",
                          "Relation Name": "posts",
                          "Alias": "p"
                        }
                      ]
                    }
                  ]
                }
              ]
            }
          ]
        }
      ]
    }
  }
]"""

    prompt = f"""You are a PostgreSQL query optimizer. Given a SQL query, output ONLY the valid JSON execution plan.

Example:
Input SQL:
{example_sql}

Output JSON:
{example_plan}

Now your turn.
Input SQL:
{query}

Output JSON:
"""

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        # Stop as soon as we have a complete plan (top-level ]); saves time vs max_new_tokens
        from transformers import StoppingCriteria
        import torch
        class StopWhenPlanComplete(StoppingCriteria):
            def __init__(self, tokenizer, prompt_str, device):
                self.tokenizer = tokenizer
                self.prompt_str = prompt_str
                self.device = device
            def __call__(self, input_ids, scores, **kwargs):
                text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                completion = text[len(self.prompt_str):].strip()
                done = completion.endswith("}]") or completion.endswith("]")
                return torch.tensor([[done]], device=self.device)
        stop_criteria = StopWhenPlanComplete(tokenizer, prompt, model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=[stop_criteria],
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = generated_text[len(prompt):].strip()

        start_idx = completion.find('[')
        end_idx = completion.rfind(']')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return completion[start_idx:end_idx + 1]
        else:
            return "Error: Model did not generate valid JSON list."
    except Exception as e:
        return f"Error generating plan: {e}"


def generate_plans_for_df(df, query_column, model_name):
    """Generate query plans for all rows in a DataFrame. Returns df with 'plan_json' column."""
    model, tokenizer = load_llm_model(model_name)

    plans = []
    total = len(df)
    print(f"Generating query plans for {total} queries...")
    sys.stdout.flush()
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=total), start=1):
        plan = generate_query_plan(row[query_column], model, tokenizer)
        plans.append(plan)
        print(f"Generated {i}/{total} plans", flush=True)

    df = df.copy()
    df["plan_json"] = plans
    return df


# ---------------------------------------------------------------------------
# 2. Plan-to-Hint Conversion
# ---------------------------------------------------------------------------

def extract_tables_and_joins(plan: Any) -> Tuple[set, list]:
    """Traverse plan JSON and collect scanned table names and join info."""
    scans = set()
    joins = []

    def safe_relation_name(node):
        return node.get("Relation Name") or node.get("Relation") or node.get("Alias")

    def gather(node):
        if not isinstance(node, dict):
            return
        node_type = node.get("Node Type", "")

        if "Scan" in node_type:
            rn = safe_relation_name(node)
            if rn:
                scans.add(rn)

        if "Join" in node_type:
            left = right = None
            if node.get("Plans") and len(node["Plans"]) >= 2:
                def find_rel(n):
                    if not isinstance(n, dict):
                        return None
                    rn = safe_relation_name(n)
                    if rn:
                        return rn
                    for c in n.get("Plans", []):
                        res = find_rel(c)
                        if res:
                            return res
                    return None
                left = find_rel(node["Plans"][0])
                right = find_rel(node["Plans"][1])
            joins.append((node_type.replace(" ", ""), left or "?", right or "?"))

        for child in node.get("Plans", []) or []:
            gather(child)

    if isinstance(plan, list):
        for entry in plan:
            if isinstance(entry, dict) and "Plan" in entry:
                gather(entry["Plan"])
    elif isinstance(plan, dict):
        if "Plan" in plan:
            gather(plan["Plan"])
        else:
            gather(plan)

    return scans, joins


def plan_json_to_pg_hint(plan_json_str: str) -> str:
    """Convert a JSON plan string into a pg_hint_plan-style hint."""
    try:
        parsed = json.loads(plan_json_str)
    except Exception:
        return ""

    scans, joins = extract_tables_and_joins(parsed)
    tokens = [f"SeqScan({s})" for s in scans]

    for node_type, left, right in joins[:6]:
        if "Hash" in node_type:
            name = "HashJoin"
        elif "Merge" in node_type:
            name = "MergeJoin"
        elif "Nested" in node_type:
            name = "NestLoop"
        else:
            name = node_type
        tokens.append(f"{name}({left} {right})")

    return f"/*+ {' '.join(tokens)} */" if tokens else ""


# ---------------------------------------------------------------------------
# 3. Query Execution with Resource Measurement
# ---------------------------------------------------------------------------

def worker_execute(query, plan_json, use_hints, iterations, timeout_ms=30000, verbose=False):
    """Run one query in a separate process and return timing, CPU, and memory metrics."""
    executor = SimpleQueryExecutor(options=f"-c statement_timeout={timeout_ms}")
    res = None

    try:
        hints = plan_json_to_pg_hint(plan_json) if (use_hints and plan_json) else ""

        # Start resource tracking
        tracemalloc.start()
        cpu_start = time.process_time_ns()
        rusage_before = resource.getrusage(resource.RUSAGE_SELF)

        if iterations > 1:
            if hints:
                times = []
                for _ in range(iterations):
                    res = executor.execute_with_hints(query, hints)
                    if res.get("error"):
                        tracemalloc.stop()
                        return {"error": res["error"]}
                    t = (
                        res.get("actual_total_time")
                        or res.get("execution_time")
                        or res.get("execution_time_ms")
                    )
                    if t is None:
                        tracemalloc.stop()
                        return {"error": "No timing value found", "raw_result": res}
                    times.append(float(t))
                t = sum(times) / len(times)
            else:
                res = executor.benchmark_query(query, iterations=iterations)
                if res.get("error"):
                    tracemalloc.stop()
                    return {"error": res["error"]}
                t = res.get("avg_time_ms")
                if t is None:
                    tracemalloc.stop()
                    return {"error": "No timing value found", "raw_result": res}
        else:
            if hints:
                res = executor.execute_with_hints(query, hints)
            else:
                res = executor.execute_query(query)

            if res.get("error"):
                tracemalloc.stop()
                return {"error": res["error"]}

            t = (
                res.get("actual_total_time")
                or res.get("execution_time")
                or res.get("execution_time_ms")
            )
            if t is None:
                tracemalloc.stop()
                return {"error": "No timing value found", "raw_result": res}

        # Collect resource metrics
        cpu_end = time.process_time_ns()
        rusage_after = resource.getrusage(resource.RUSAGE_SELF)
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        cpu_time_ms = (cpu_end - cpu_start) / 1_000_000
        # ru_maxrss is in bytes on Linux, kilobytes on macOS
        peak_rss_kb = rusage_after.ru_maxrss - rusage_before.ru_maxrss
        if sys.platform == "darwin":
            peak_rss_kb = peak_rss_kb // 1024  # macOS reports bytes

        return {
            "execution_time_ms": float(t),
            "cpu_time_ms": round(cpu_time_ms, 3),
            "peak_memory_kb": round(peak_memory_bytes / 1024, 2),
        }

    except Exception as e:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        return {"error": str(e), "raw_result": res}


# ---------------------------------------------------------------------------
# 4. Batch Execution
# ---------------------------------------------------------------------------

def execute_plans(df, query_column, use_hints, iterations, workers, verbose, timeout_ms=30000):
    """Execute queries from a DataFrame and return it with execution metrics columns."""
    if "plan_json" not in df.columns:
        df["plan_json"] = None

    df = df.copy()
    df["execution_time_ms"] = pd.NA
    df["cpu_time_ms"] = pd.NA
    df["peak_memory_kb"] = pd.NA

    total = len(df)
    print(f"Executing {total} queries with {workers} workers (use_hints={use_hints}, iterations={iterations}, timeout={timeout_ms}ms)")

    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {
            exe.submit(
                worker_execute,
                row[query_column],
                row.get("plan_json"),
                use_hints,
                iterations,
                timeout_ms,
                verbose,
            ): idx
            for idx, row in df.iterrows()
        }

        for i, fut in enumerate(as_completed(futures), start=1):
            idx = futures[fut]
            try:
                res = fut.result()
                if res.get("error"):
                    if verbose:
                        print(f"[{idx}] ERROR: {res['error']}")
                else:
                    df.at[idx, "execution_time_ms"] = res.get("execution_time_ms")
                    df.at[idx, "cpu_time_ms"] = res.get("cpu_time_ms")
                    df.at[idx, "peak_memory_kb"] = res.get("peak_memory_kb")
                    if verbose:
                        print(
                            f"[{idx}] OK {res.get('execution_time_ms')} ms | "
                            f"CPU {res.get('cpu_time_ms')} ms | "
                            f"Mem {res.get('peak_memory_kb')} KB"
                        )
            except Exception as e:
                print(f"[{idx}] Worker crashed: {e}")

            if i % 100 == 0 or i == total:
                print(f"Completed {i}/{total}")

    return df


# ---------------------------------------------------------------------------
# 5. End-to-End Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(input_csv, output_csv, query_column, model_name,
                 use_hints=True, iterations=1, workers=4, limit=None, verbose=False):
    """
    Full pipeline: read queries -> generate plans -> execute with metrics -> save CSV.
    """
    if not os.path.exists(input_csv):
        print(f"Error: Input file '{input_csv}' not found.")
        return

    print(f"Reading input CSV from {input_csv}...")
    df = pd.read_csv(input_csv)

    if query_column not in df.columns:
        print(f"Error: Column '{query_column}' not found. Available: {list(df.columns)}")
        return

    if limit:
        print(f"Limiting to first {limit} rows.")
        df = df.head(limit)

    # Step 1: Generate plans via LLM
    df = generate_plans_for_df(df, query_column, model_name)

    # Step 2: Execute queries and measure resources
    df = execute_plans(df, query_column, use_hints, iterations, workers, verbose)

    # Step 3: Save results
    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end LLM query plan generation and execution benchmarking"
    )
    parser.add_argument("input_csv", help="Path to CSV file containing SQL queries")
    parser.add_argument("--output", "-o", required=True, help="Output CSV path")
    parser.add_argument("--query-column", "-q", default="sql_text",
                        help="Name of the SQL query column (default: sql_text)")
    parser.add_argument("--model", "-m", required=True,
                        help="HuggingFace model name for plan generation")
    parser.add_argument("--use-hints", action="store_true",
                        help="Use pg_hint_plan hints derived from generated plans")
    parser.add_argument("--iterations", "-i", type=int, default=1,
                        help="Number of benchmark iterations per query")
    parser.add_argument("--workers", "-w", type=int, default=4,
                        help="Number of parallel execution workers")
    parser.add_argument("--limit", "-l", type=int, default=None,
                        help="Limit number of rows to process")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    run_pipeline(
        input_csv=args.input_csv,
        output_csv=args.output,
        query_column=args.query_column,
        model_name=args.model,
        use_hints=args.use_hints,
        iterations=args.iterations,
        workers=args.workers,
        limit=args.limit,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
