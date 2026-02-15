#!/usr/bin/env python3
"""
Simplified query executor for Cardinal project initial setup.
Run PostgreSQL queries, collect execution plans, and support plan-to-hints for RL.
"""

import json
import time
from typing import Dict, List, Tuple, Any

import psycopg2

from config import DB_CONFIG
from plan_to_hints import plan_to_hints, plan_to_hints_verbose


class SimpleQueryExecutor:
    def __init__(self, **kwargs):
        """Initialize with database configuration. Override with kwargs if needed."""
        self.connection_params = DB_CONFIG.copy()
        self.connection_params.update(kwargs)

    def get_connection(self):
        """Create a database connection"""
        return psycopg2.connect(**self.connection_params)

    def get_execution_plan(self, query: str, analyze: bool = False) -> Dict[str, Any]:
        """
        Get execution plan for a query.

        Args:
            query: SQL query string
            analyze: If True, actually execute query and get real stats

        Returns:
            Dictionary containing execution plan and metadata
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            if analyze:
                explain_cmd = "EXPLAIN (ANALYZE true, BUFFERS true, FORMAT JSON) "
            else:
                explain_cmd = "EXPLAIN (FORMAT JSON) "

            explain_query = explain_cmd + query
            start_time = time.time()
            cursor.execute(explain_query)
            explain_result = cursor.fetchone()[0]
            explain_time = time.time() - start_time

            plan = explain_result[0]["Plan"]
            result = {
                "query": query,
                "execution_plan": explain_result[0],
                "explain_time_ms": round(explain_time * 1000, 2),
                "analyzed": analyze,
            }

            if analyze:
                result.update({
                    "actual_total_time": plan.get("Actual Total Time", 0),
                    "actual_rows": plan.get("Actual Rows", 0),
                    "planning_time": explain_result[0].get("Planning Time", 0),
                    "execution_time": explain_result[0].get("Execution Time", 0),
                })
            else:
                result.update({
                    "estimated_cost": plan.get("Total Cost", 0),
                    "estimated_rows": plan.get("Plan Rows", 0),
                    "planning_time": explain_result[0].get("Planning Time", 0),
                })

            return result

        except Exception as e:
            return {"query": query, "error": str(e), "execution_plan": None}
        finally:
            cursor.close()
            conn.close()

    def execute_query(self, query: str) -> Dict[str, Any]:
        """Execute query and measure performance."""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            start_time = time.time()
            cursor.execute(query)
            results = cursor.fetchall()
            execution_time = time.time() - start_time

            return {
                "query": query,
                "execution_time_ms": round(execution_time * 1000, 2),
                "row_count": len(results),
                "results": results[:5],
                "success": True,
            }

        except Exception as e:
            return {"query": query, "error": str(e), "success": False}
        finally:
            cursor.close()
            conn.close()

    def execute_with_hints(self, query: str, hints: str) -> Dict[str, Any]:
        """
        Execute query with PostgreSQL hints (requires pg_hint_plan extension).

        Args:
            query: SQL query
            hints: Hint string (e.g., "/*+ HashJoin(a b) SeqScan(c) */")
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT 1")
            hinted_query = f"{hints}\n{query}"
            return self.get_execution_plan(hinted_query, analyze=True)

        except Exception as e:
            return {
                "query": query,
                "hints": hints,
                "error": f"Failed to execute with hints: {str(e)}",
            }
        finally:
            cursor.close()
            conn.close()

    def compare_execution_strategies(
        self, query: str, hint_variations: List[str]
    ) -> Dict[str, Any]:
        """Compare different execution strategies for the same query."""
        results = []

        default_result = self.get_execution_plan(query, analyze=True)
        if not default_result.get("error"):
            results.append({
                "strategy": "default",
                "hints": None,
                "execution_time": default_result.get("actual_total_time", 0),
                "rows": default_result.get("actual_rows", 0),
            })

        for i, hints in enumerate(hint_variations):
            hint_result = self.execute_with_hints(query, hints)
            if not hint_result.get("error"):
                results.append({
                    "strategy": f"hint_{i+1}",
                    "hints": hints,
                    "execution_time": hint_result.get("actual_total_time", 0),
                    "rows": hint_result.get("actual_rows", 0),
                })

        return {
            "query": query,
            "strategies_tested": len(results),
            "results": results,
            "best_strategy": min(results, key=lambda x: x["execution_time"]) if results else None,
        }

    def benchmark_query(self, query: str, iterations: int = 5) -> Dict[str, Any]:
        """Run query multiple times and collect performance statistics."""
        results = []

        for i in range(iterations):
            result = self.execute_query(query)
            if result["success"]:
                results.append(result["execution_time_ms"])
            else:
                return {"error": f"Query failed on iteration {i+1}: {result['error']}"}

        if results:
            avg_time = sum(results) / len(results)
            return {
                "query": query,
                "iterations": iterations,
                "avg_time_ms": round(avg_time, 2),
                "min_time_ms": min(results),
                "max_time_ms": max(results),
                "all_times": results,
            }
        return {"error": "No successful executions"}

    def extract_hints_from_plan(self, plan_json: Any, verbose: bool = False) -> Dict[str, Any]:
        """Extract pg_hint_plan hints from an execution plan JSON."""
        if verbose:
            return plan_to_hints_verbose(plan_json)
        return {"hint_string": plan_to_hints(plan_json)}

    def get_plan_and_hints(self, query: str, analyze: bool = False) -> Dict[str, Any]:
        """Get execution plan and extract equivalent pg_hint_plan hints."""
        plan_result = self.get_execution_plan(query, analyze=analyze)

        if plan_result.get("error"):
            return plan_result

        if plan_result.get("execution_plan"):
            hint_info = self.extract_hints_from_plan(plan_result["execution_plan"], verbose=True)
            plan_result["extracted_hints"] = hint_info["hint_string"]
            plan_result["hint_details"] = {
                "scan_hints": hint_info.get("scan_hints", []),
                "join_hints": hint_info.get("join_hints", []),
                "index_hints": hint_info.get("index_hints", []),
            }

        return plan_result

    def execute_with_extracted_hints(self, query: str, plan_json: Any) -> Dict[str, Any]:
        """Execute a query using hints extracted from a provided execution plan."""
        hints = plan_to_hints(plan_json)

        if not hints:
            return {
                "query": query,
                "error": "Could not extract any hints from the provided plan",
                "plan_provided": plan_json,
            }

        result = self.execute_with_hints(query, hints)
        result["extracted_hints"] = hints
        return result

    def compare_plan_with_hints(
        self, query: str, plan_json: Any, iterations: int = 3
    ) -> Dict[str, Any]:
        """Compare executing a query with default optimizer vs extracted hints."""
        hints = plan_to_hints(plan_json)

        results = {
            "query": query,
            "extracted_hints": hints,
            "default_execution": None,
            "hinted_execution": None,
            "comparison": None,
        }

        default_result = self.get_execution_plan(query, analyze=True)
        if not default_result.get("error"):
            default_benchmark = self.benchmark_query(query, iterations)
            results["default_execution"] = {
                "plan": default_result.get("execution_plan"),
                "execution_time": default_result.get("actual_total_time", 0),
                "benchmark": default_benchmark if not default_benchmark.get("error") else None,
            }

        if hints:
            hinted_result = self.execute_with_hints(query, hints)
            if not hinted_result.get("error"):
                hinted_query = f"{hints}\n{query}"
                hinted_benchmark = self.benchmark_query(hinted_query, iterations)
                results["hinted_execution"] = {
                    "plan": hinted_result.get("execution_plan"),
                    "execution_time": hinted_result.get("actual_total_time", 0),
                    "benchmark": hinted_benchmark if not hinted_benchmark.get("error") else None,
                }

        if results["default_execution"] and results["hinted_execution"]:
            default_time = results["default_execution"]["execution_time"]
            hinted_time = results["hinted_execution"]["execution_time"]
            speedup = (default_time - hinted_time) / default_time * 100 if default_time > 0 else 0
            results["comparison"] = {
                "default_time_ms": default_time,
                "hinted_time_ms": hinted_time,
                "difference_ms": round(hinted_time - default_time, 2),
                "speedup_percent": round(speedup, 2),
                "winner": "hints" if hinted_time < default_time else "default",
            }

        return results


def main():
    """Test the simplified executor with plan-to-hints functionality."""
    executor = SimpleQueryExecutor()

    test_queries = [
        "SELECT * FROM customers WHERE country = 'USA'",
        """
        SELECT c.name, COUNT(o.order_id) as order_count
        FROM customers c
        LEFT JOIN orders o ON c.customer_id = o.customer_id
        GROUP BY c.customer_id, c.name
        ORDER BY order_count DESC
        """,
    ]

    print("=== Simple Query Executor Test ===\n")
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query.strip()}")
        plan_result = executor.get_execution_plan(query, analyze=False)
        if plan_result.get("execution_plan"):
            print(f"  Estimated Cost: {plan_result.get('estimated_cost')}")
            print(f"  Estimated Rows: {plan_result.get('estimated_rows')}")
        analyze_result = executor.get_execution_plan(query, analyze=True)
        if not analyze_result.get("error"):
            print(f"  Actual Time: {analyze_result.get('actual_total_time')} ms")
            print(f"  Actual Rows: {analyze_result.get('actual_rows')}")
        benchmark = executor.benchmark_query(query, iterations=3)
        if not benchmark.get("error"):
            print(f"  Avg Execution: {benchmark['avg_time_ms']} ms")
        print()


if __name__ == "__main__":
    main()
