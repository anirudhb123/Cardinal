#!/usr/bin/env python3
"""
Parallel batch executor for CSV of SQL queries.

Reads a CSV with:
 - 'query' column (SQL)
 - optional 'plan_json' column (execution plan)

Writes back a CSV with an 'execution_time_ms' column.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Optional, Tuple

import pandas as pd

# --- Add the single-query directory to sys.path ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SINGLE_QUERY_DIR = os.path.join(PROJECT_ROOT, "single-query")
if SINGLE_QUERY_DIR not in sys.path:
    sys.path.insert(0, SINGLE_QUERY_DIR)

# --- Import the existing executor and config ---
from query_executor import SimpleQueryExecutor  # noqa: E402


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


def worker_execute(query, plan_json, use_hints, iterations, verbose=False):
    """Run one query in a separate process and return timing result."""
    executor = SimpleQueryExecutor()
    res = None  # make sure it's always defined

    try:
        hints = plan_json_to_pg_hint(plan_json) if (use_hints and plan_json) else ""

        # --- case 1: multiple iterations ---
        if iterations > 1:
            if hints:
                times = []
                for _ in range(iterations):
                    res = executor.execute_with_hints(query, hints)
                    if res.get("error"):
                        return {"error": res["error"]}
                    t = (
                        res.get("actual_total_time")
                        or res.get("execution_time")
                        or res.get("execution_time_ms")
                    )
                    if t is None:
                        return {"error": "No timing value found", "raw_result": res}
                    times.append(float(t))
                avg = sum(times) / len(times)
                t = avg  # use avg as the execution time for multiple iterations
            else:
                res = executor.benchmark_query(query, iterations=iterations)
                if res.get("error"):
                    return {"error": res["error"]}
                t = res.get("avg_time_ms")
                if t is None:
                    return {"error": "No timing value found", "raw_result": res}

        # --- case 2: single run ---
        else:
            if hints:
                res = executor.execute_with_hints(query, hints)
            else:
                res = executor.execute_query(query)

            if res.get("error"):
                return {"error": res["error"]}

            t = (
                res.get("actual_total_time")
                or res.get("execution_time")
                or res.get("execution_time_ms")
            )
            if t is None:
                return {"error": "No timing value found", "raw_result": res}

        # --- normalize timing and return ---
        return {"execution_time_ms": float(t)}

    except Exception as e:
        return {"error": str(e), "raw_result": res}


def process_csv(csv_path, output_path, use_hints, iterations, workers, verbose):
    df = pd.read_csv(csv_path)
    if "query" not in df.columns:
        raise ValueError("CSV must have a 'query' column")
    if "plan_json" not in df.columns:
        df["plan_json"] = None

    out_col = "execution_time"
    df[out_col] = pd.NA
    total = len(df)
    print(f"Processing {total} queries with {workers} workers (use_hints={use_hints}, iterations={iterations})")

    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = {
            exe.submit(worker_execute, row["sql_text"], row.get("plan_json"), use_hints, iterations, verbose): idx
            for idx, row in df.iterrows()
        }

        for i, fut in enumerate(as_completed(futures), start=1):
            idx = futures[fut]
            try:
                res = fut.result()
                if res.get("error"):
                    if verbose:
                        print(f"[{idx}] ERROR: {res['error']}")
                    df.at[idx, out_col] = pd.NA
                else:
                    df.at[idx, out_col] = res.get("execution_time_ms")
                    if verbose:
                        print(f"[{idx}] OK {res.get('execution_time_ms')} ms")
            except Exception as e:
                df.at[idx, out_col] = pd.NA
                print(f"[{idx}] Worker crashed: {e}")

            if i % 100 == 0 or i == total:
                print(f"Completed {i}/{total}")
            if verbose and i % 500 == 0:
                print(f"[{idx}] result: {res}")

    if not output_path:
        base, ext = os.path.splitext(csv_path)
        output_path = base + "_with_times" + (ext or ".csv")

    df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Parallel batch query executor")
    parser.add_argument("csv", help="Path to CSV file containing queries")
    parser.add_argument("--output", "-o", help="Output CSV path")
    parser.add_argument("--use-hints", action="store_true", help="Use pg_hint_plan hints")
    parser.add_argument("--iterations", "-i", type=int, default=1, help="Number of benchmark iterations")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Parallel workers")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose mode")
    args = parser.parse_args()

    process_csv(args.csv, args.output, args.use_hints, args.iterations, args.workers, args.verbose)


if __name__ == "__main__":
    main()
