#!/usr/bin/env python3
"""
Interactive command-line interface for PostgreSQL query execution and benchmarking.
"""

import json
import sys
import argparse

from config import DB_CONFIG
from plan_to_hints import plan_to_hints, plan_to_hints_verbose
from query_executor import SimpleQueryExecutor


def print_results(result_dict, title="Results"):
    """Pretty print results dictionary."""
    print(f"\n=== {title} ===")
    for key, value in result_dict.items():
        if key == "execution_plan":
            print(f"{key}: [JSON execution plan - use --verbose to see full plan]")
        elif key == "results":
            print(f"{key}: {value} (showing first 5 rows)")
        elif key == "extracted_hints":
            print(f"{key}: {value}")
        elif key == "hint_details":
            print(f"{key}:")
            for hint_type, hints in value.items():
                if hints:
                    print(f"  {hint_type}: {hints}")
        elif isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    print()


def print_hints_from_plan(plan_json, verbose=False):
    """Pretty print extracted hints from an execution plan."""
    print("\n=== Extracted pg_hint_plan Hints ===")
    if verbose:
        result = plan_to_hints_verbose(plan_json)
        print(f"Hint string: {result['hint_string']}")
        print(f"\nBreakdown:")
        print(f"  Scan hints: {result.get('scan_hints', [])}")
        print(f"  Join hints: {result.get('join_hints', [])}")
        print(f"  Index hints: {result.get('index_hints', [])}")
    else:
        print(f"Hints: {plan_to_hints(plan_json)}")
    print()


def print_execution_plan(plan, indent=0):
    """Recursively print execution plan in a readable format."""
    spacing = " " * indent
    node_type = plan.get("Node Type", "Unknown")
    if "Actual Total Time" in plan:
        time_info = f" (Time: {plan['Actual Total Time']:.2f}ms, Rows: {plan.get('Actual Rows', 0)})"
    else:
        time_info = f" (Est Cost: {plan.get('Total Cost', 0):.2f}, Est Rows: {plan.get('Plan Rows', 0)})"
    print(f"{spacing}{node_type}{time_info}")
    if "Relation Name" in plan:
        print(f"{spacing}  Table: {plan['Relation Name']}")
    if "Hash Cond" in plan:
        print(f"{spacing}  Join Condition: {plan['Hash Cond']}")
    elif "Merge Cond" in plan:
        print(f"{spacing}  Join Condition: {plan['Merge Cond']}")
    if "Plans" in plan:
        for child_plan in plan["Plans"]:
            print_execution_plan(child_plan, indent + 1)


def interactive_mode():
    """Run interactive query execution loop."""
    executor = SimpleQueryExecutor()
    print("=== Interactive PostgreSQL Query Executor ===")
    print("Enter SQL queries to execute and benchmark.")
    print("Type 'quit' or 'exit' to end. Type 'help' for commands.\n")
    last_plan = None

    while True:
        try:
            query = input("Enter your SQL query (or command):\n> ").strip()
            if not query:
                continue
            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            if query.lower() in ["help", "h"]:
                print("\nAvailable: help, quit, clear, hints (from last plan)\n")
                continue
            if query.lower() in ["clear", "cls"]:
                import os
                os.system("cls" if os.name == "nt" else "clear")
                continue
            if query.lower() == "hints":
                if last_plan:
                    print_hints_from_plan(last_plan, verbose=True)
                else:
                    print("No execution plan available. Run a query first.")
                continue

            hints_input = input("PostgreSQL hints (optional, Enter to skip): ").strip()
            hints = None
            if hints_input.lower() == "auto" and last_plan:
                hints = plan_to_hints(last_plan)
                print(f"Auto-extracted hints: {hints}")
            elif hints_input and hints_input.lower() != "auto":
                hints = hints_input

            iterations_input = input("Number of benchmark iterations (default 3): ").strip()
            try:
                iterations = int(iterations_input) if iterations_input else 3
                iterations = max(1, min(iterations, 20))
            except ValueError:
                iterations = 3

            if hints:
                plan_result = executor.execute_with_hints(query, hints)
            else:
                plan_result = executor.get_execution_plan(query, analyze=True)

            if plan_result.get("error"):
                print(f"Error: {plan_result['error']}")
                continue

            if plan_result.get("execution_plan"):
                last_plan = plan_result["execution_plan"]
                plan_result["extracted_hints"] = plan_to_hints(last_plan)

            print_results(plan_result, "Execution Plan Analysis")
            if plan_result.get("execution_plan"):
                print("=== Execution Plan Tree ===")
                print_execution_plan(plan_result["execution_plan"]["Plan"])
                print("\n=== Extracted pg_hint_plan Hints ===")
                print(f"To replay: {plan_result.get('extracted_hints', 'N/A')}\n")

            benchmark_result = executor.benchmark_query(query, iterations=iterations)
            if benchmark_result.get("error"):
                print(f"Benchmark error: {benchmark_result['error']}")
            else:
                print_results(benchmark_result, f"Benchmark Results ({iterations} iterations)")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")


def single_query_mode(query, hints=None, iterations=3, verbose=False, plan_json=None):
    """Execute a single query and return results."""
    executor = SimpleQueryExecutor()

    if plan_json:
        print("Extracting hints from provided execution plan...")
        extracted = plan_to_hints(plan_json)
        print(f"Extracted hints: {extracted}")
        if not hints:
            hints = extracted
        print()

    print(f"Executing query: {query}")
    if hints:
        print(f"With hints: {hints}")

    if hints:
        result = executor.execute_with_hints(query, hints)
    else:
        result = executor.get_execution_plan(query, analyze=True)

    if result.get("error"):
        print(f"Error: {result['error']}")
        return

    if result.get("execution_plan"):
        result["extracted_hints"] = plan_to_hints(result["execution_plan"])

    print_results(result, "Execution Plan Analysis")

    if verbose and result.get("execution_plan"):
        print("=== Detailed Execution Plan ===")
        print(json.dumps(result["execution_plan"], indent=2))
        print()

    if result.get("execution_plan"):
        print("=== Execution Plan Tree ===")
        print_execution_plan(result["execution_plan"]["Plan"])
        print("=== Extracted pg_hint_plan Hints ===")
        print_hints_from_plan(result["execution_plan"], verbose=verbose)

    if iterations > 1:
        benchmark_result = executor.benchmark_query(query, iterations=iterations)
        if not benchmark_result.get("error"):
            print_results(benchmark_result, f"Benchmark Results ({iterations} iterations)")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive PostgreSQL Query Executor and Benchmarker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-q", "--query", help="SQL query to execute (else interactive mode)")
    parser.add_argument("--hints", help='PostgreSQL hints (e.g. "/*+ HashJoin(a b) */")')
    parser.add_argument("-i", "--iterations", type=int, default=3, help="Benchmark iterations (1-20)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show full plan JSON")
    parser.add_argument("--plan-to-hints", dest="plan_json", help="Extract hints from plan JSON string")
    parser.add_argument("--plan-file", help="Extract hints from plan JSON file")

    args = parser.parse_args()

    if args.iterations < 1 or args.iterations > 20:
        print("Iterations must be between 1 and 20")
        return 1

    plan_json = None
    if args.plan_file:
        try:
            with open(args.plan_file, "r") as f:
                plan_json = json.load(f)
        except FileNotFoundError:
            print(f"Error: Plan file not found: {args.plan_file}")
            return 1
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in plan file: {e}")
            return 1
    elif args.plan_json:
        try:
            plan_json = json.loads(args.plan_json)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON: {e}")
            return 1

    if plan_json and not args.query:
        print("=== Plan-to-Hints Conversion ===\n")
        print_hints_from_plan(plan_json, verbose=args.verbose)
        return 0

    if args.query:
        single_query_mode(args.query, args.hints, args.iterations, args.verbose, plan_json)
    else:
        interactive_mode()

    return 0


if __name__ == "__main__":
    sys.exit(main())
