#!/usr/bin/env python3
"""
Run a single query in an isolated process and return latency, CPU time, and peak memory.
Used by RL reward computation so metrics are per-query and not polluted by the trainer process.
"""

import os
import sys
import time
import resource
from typing import Any, Dict, Optional

# Subprocess worker: must be importable and runnable with minimal deps in child
def _worker_run_query(query: str, hints: Optional[str], db_config: Dict[str, Any]) -> Dict[str, Any]:
    """Run in subprocess: execute query (with optional hints), return latency_ms, cpu_time_s, max_rss_kb."""
    import psycopg2

    out = {
        "latency_ms": None,
        "cpu_time_s": None,
        "max_rss_kb": None,
        "error": None,
        "success": False,
    }

    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        sql = f"{hints}\n{query}" if hints and hints.strip() else query

        # CPU time (user + system) and RSS before
        ru0 = resource.getrusage(resource.RUSAGE_SELF)
        t0 = time.perf_counter()

        cursor.execute(sql)
        cursor.fetchall()

        t1 = time.perf_counter()
        ru1 = resource.getrusage(resource.RUSAGE_SELF)

        cursor.close()
        conn.close()

        latency_ms = (t1 - t0) * 1000.0
        cpu_time_s = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)
        max_rss = ru1.ru_maxrss
        # Linux: ru_maxrss is in KB; macOS: in bytes (see getrusage(2))
        if sys.platform == "darwin":
            max_rss = max_rss / 1024.0  # bytes -> KB
        max_rss_kb = max_rss

        out["latency_ms"] = round(latency_ms, 4)
        out["cpu_time_s"] = round(cpu_time_s, 4)
        out["max_rss_kb"] = round(max_rss_kb, 2)
        out["success"] = True

    except Exception as e:
        out["error"] = str(e)

    return out


def run_query_with_metrics(
    query: str,
    hints: Optional[str] = None,
    db_config: Optional[Dict[str, Any]] = None,
    timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run query in a subprocess and return latency (ms), CPU time (s), and peak RSS (KB).

    Args:
        query: SQL query string.
        hints: Optional pg_hint_plan hint string (e.g. "/*+ HashJoin(a b) */").
        db_config: DB connection dict (host, database, user, password, port).
                   If None, uses env (POSTGRES_*) from the current process.
        timeout_s: If set, worker is allowed up to this many seconds (not enforced in worker).

    Returns:
        Dict with latency_ms, cpu_time_s, max_rss_kb, success, and optional error.
    """
    if db_config is None:
        from dotenv import load_dotenv
        load_dotenv()
        db_config = {
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "database": os.getenv("POSTGRES_DB", "cardinal_test"),
            "user": os.getenv("POSTGRES_USER", "postgres"),
            "password": os.getenv("POSTGRES_PASSWORD", "your_password"),
            "port": int(os.getenv("POSTGRES_PORT", 5432)),
        }

    try:
        from multiprocessing import Process, Queue
    except ImportError:
        # Fallback: run in-process (metrics then reflect current process, not isolated)
        cfg = db_config or {
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "database": os.getenv("POSTGRES_DB", "cardinal_test"),
            "user": os.getenv("POSTGRES_USER", "postgres"),
            "password": os.getenv("POSTGRES_PASSWORD", "your_password"),
            "port": int(os.getenv("POSTGRES_PORT", 5432)),
        }
        return _worker_run_query(query, hints, cfg)

    q = Queue()
    def target():
        q.put(_worker_run_query(query, hints, db_config))

    p = Process(target=target)
    p.start()
    p.join(timeout=timeout_s or 60.0)
    if p.is_alive():
        p.terminate()
        p.join(timeout=2.0)
        if p.is_alive():
            p.kill()
        return {
            "latency_ms": None,
            "cpu_time_s": None,
            "max_rss_kb": None,
            "success": False,
            "error": "query timeout",
        }

    try:
        return q.get_nowait()
    except Exception:
        return {
            "latency_ms": None,
            "cpu_time_s": None,
            "max_rss_kb": None,
            "success": False,
            "error": "failed to get worker result",
        }
