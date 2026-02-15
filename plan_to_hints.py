#!/usr/bin/env python3
"""
Module to parse PostgreSQL execution plan JSON into pg_hint_plan hints.
Converts the optimizer's chosen plan into explicit hints that can reproduce it.
"""

import json
from typing import Dict, List, Any, Optional, Tuple


class PlanToHintConverter:
    """Converts PostgreSQL EXPLAIN JSON plans to pg_hint_plan syntax"""

    # Mapping of plan node types to hint names
    SCAN_HINTS = {
        "Seq Scan": "SeqScan",
        "Index Scan": "IndexScan",
        "Index Only Scan": "IndexOnlyScan",
        "Bitmap Heap Scan": "BitmapScan",
        "Bitmap Index Scan": "BitmapScan",
        "Tid Scan": "TidScan",
        "Tid Range Scan": "TidRangeScan",
    }

    JOIN_HINTS = {
        "Nested Loop": "NestLoop",
        "Hash Join": "HashJoin",
        "Merge Join": "MergeJoin",
    }

    # Leading hint for join order
    JOIN_ORDER_HINT = "Leading"

    def __init__(self):
        self.tables: List[str] = []
        self.scan_hints: List[str] = []
        self.join_hints: List[str] = []
        self.join_order: List[str] = []
        self.index_hints: List[Tuple[str, str]] = []  # (table, index_name)

    def reset(self):
        """Reset internal state for a new conversion"""
        self.tables = []
        self.scan_hints = []
        self.join_hints = []
        self.join_order = []
        self.index_hints = []

    def parse_plan(self, plan_json: Any) -> str:
        """
        Parse execution plan JSON and return pg_hint_plan hint string.

        Args:
            plan_json: Either a dict (the plan), a list containing the plan,
                      or a JSON string to parse

        Returns:
            pg_hint_plan hint string like "/*+ HashJoin(a b) SeqScan(a) */"
        """
        self.reset()

        # Handle different input formats
        if isinstance(plan_json, str):
            plan_json = json.loads(plan_json)

        # Handle list format from EXPLAIN (FORMAT JSON)
        if isinstance(plan_json, list):
            plan_json = plan_json[0]

        # Extract the Plan node
        if "Plan" in plan_json:
            plan = plan_json["Plan"]
        else:
            plan = plan_json

        # Recursively traverse the plan tree
        self._traverse_plan(plan)

        # Build the hint string
        return self._build_hint_string()

    def _get_table_alias(self, node: Dict[str, Any]) -> Optional[str]:
        """Extract table alias from a plan node"""
        # Prefer Alias over Relation Name for accuracy
        return node.get("Alias") or node.get("Relation Name")

    def _traverse_plan(self, node: Dict[str, Any], depth: int = 0) -> List[str]:
        """
        Recursively traverse the plan tree and collect hints.

        Returns list of table aliases involved in this subtree (for join hints)
        """
        if not node:
            return []

        node_type = node.get("Node Type", "")
        tables_in_subtree = []

        # Process child nodes first (bottom-up for join order)
        child_tables_list = []
        if "Plans" in node:
            for child in node["Plans"]:
                child_tables = self._traverse_plan(child, depth + 1)
                child_tables_list.append(child_tables)
                tables_in_subtree.extend(child_tables)

        # Handle scan nodes
        if node_type in self.SCAN_HINTS:
            table_alias = self._get_table_alias(node)
            if table_alias:
                hint_name = self.SCAN_HINTS[node_type]
                self.scan_hints.append(f"{hint_name}({table_alias})")
                tables_in_subtree.append(table_alias)

                # Track index usage
                if "Index Name" in node:
                    self.index_hints.append((table_alias, node["Index Name"]))

        # Handle join nodes
        elif node_type in self.JOIN_HINTS:
            hint_name = self.JOIN_HINTS[node_type]

            # Get tables from both sides of the join
            if len(child_tables_list) >= 2:
                # Flatten all tables from children for the join hint
                all_join_tables = []
                for child_tables in child_tables_list:
                    all_join_tables.extend(child_tables)

                if len(all_join_tables) >= 2:
                    # Create join hint with all involved tables
                    tables_str = " ".join(all_join_tables)
                    self.join_hints.append(f"{hint_name}({tables_str})")

        return tables_in_subtree

    def _extract_join_order(self, node: Dict[str, Any]) -> List[str]:
        """
        Extract the join order from the plan tree.
        Returns a nested structure representing join order.
        """
        if not node:
            return []

        node_type = node.get("Node Type", "")

        # Base case: scan node
        if node_type in self.SCAN_HINTS:
            table_alias = self._get_table_alias(node)
            return [table_alias] if table_alias else []

        # Recursive case: join or other nodes
        result = []
        if "Plans" in node:
            for child in node["Plans"]:
                child_order = self._extract_join_order(child)
                result.extend(child_order)

        return result

    def _build_hint_string(self) -> str:
        """Build the final pg_hint_plan hint string"""
        hints = []

        # Add scan hints
        hints.extend(self.scan_hints)

        # Add join hints (deduplicate)
        seen_joins = set()
        for join_hint in self.join_hints:
            if join_hint not in seen_joins:
                hints.append(join_hint)
                seen_joins.add(join_hint)

        # Add index hints
        for table, index_name in self.index_hints:
            hints.append(f"IndexScan({table} {index_name})")

        if not hints:
            return ""

        return "/*+ " + " ".join(hints) + " */"

    def parse_plan_verbose(self, plan_json: Any) -> Dict[str, Any]:
        """
        Parse execution plan and return detailed hint information.

        Returns a dict with:
        - hint_string: The complete pg_hint_plan string
        - scan_hints: List of scan hints
        - join_hints: List of join hints
        - index_hints: List of index hints
        - tables: List of tables involved
        """
        hint_string = self.parse_plan(plan_json)

        return {
            "hint_string": hint_string,
            "scan_hints": self.scan_hints.copy(),
            "join_hints": list(set(self.join_hints)),
            "index_hints": self.index_hints.copy(),
            "tables": list(set(t for t in self.tables if t)),
        }


def plan_to_hints(plan_json: Any) -> str:
    """
    Convenience function to convert a plan to hints.

    Args:
        plan_json: Execution plan JSON (dict, list, or JSON string)

    Returns:
        pg_hint_plan hint string
    """
    converter = PlanToHintConverter()
    return converter.parse_plan(plan_json)


def plan_to_hints_verbose(plan_json: Any) -> Dict[str, Any]:
    """
    Convenience function to get verbose hint information.

    Args:
        plan_json: Execution plan JSON (dict, list, or JSON string)

    Returns:
        Dict with hint_string and component hints
    """
    converter = PlanToHintConverter()
    return converter.parse_plan_verbose(plan_json)


# Example usage and testing
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert PostgreSQL execution plan JSON to pg_hint_plan hints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plan_to_hints.py '[{"Plan": {...}}]'
  python plan_to_hints.py --file plan.json
  python plan_to_hints.py --file plan.json -v
  echo '[{"Plan": {...}}]' | python plan_to_hints.py --stdin
        """,
    )

    parser.add_argument(
        "plan_json",
        nargs="?",
        help="Execution plan JSON string",
    )
    parser.add_argument(
        "-f",
        "--file",
        help="Read execution plan from JSON file",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read execution plan from stdin",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed hint breakdown",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run built-in test with example plan",
    )

    args = parser.parse_args()

    # Run test mode
    if args.test or (not args.plan_json and not args.file and not args.stdin):
        # Test with the provided execution plan
        test_plan = [{"Plan": {"Node Type": "Limit", "Parallel Aware": False, "Async Capable": False, "Startup Cost": 51.64, "Total Cost": 51.67, "Plan Rows": 10, "Plan Width": 114, "Plans": [{"Node Type": "Sort", "Parent Relationship": "Outer", "Parallel Aware": False, "Async Capable": False, "Startup Cost": 51.64, "Total Cost": 51.68, "Plan Rows": 17, "Plan Width": 114, "Sort Key": ["(count(p.id)) DESC"], "Plans": [{"Node Type": "Aggregate", "Strategy": "Sorted", "Partial Mode": "Simple", "Parent Relationship": "Outer", "Parallel Aware": False, "Async Capable": False, "Startup Cost": 50.95, "Total Cost": 51.29, "Plan Rows": 17, "Plan Width": 114, "Group Key": ["u.displayname"], "Plans": [{"Node Type": "Sort", "Parent Relationship": "Outer", "Parallel Aware": False, "Async Capable": False, "Startup Cost": 50.95, "Total Cost": 51.0, "Plan Rows": 17, "Plan Width": 106, "Sort Key": ["u.displayname"], "Plans": [{"Node Type": "Hash Join", "Parent Relationship": "Outer", "Parallel Aware": False, "Async Capable": False, "Join Type": "Right", "Startup Cost": 21.69, "Total Cost": 50.61, "Plan Rows": 17, "Plan Width": 106, "Inner Unique": False, "Hash Cond": "(v.postid = p.id)", "Plans": [{"Node Type": "Seq Scan", "Parent Relationship": "Outer", "Parallel Aware": False, "Async Capable": False, "Relation Name": "votes", "Alias": "v", "Startup Cost": 0.0, "Total Cost": 28.88, "Plan Rows": 8, "Plan Width": 8, "Filter": "(votetypeid = 8)"}, {"Node Type": "Hash", "Parent Relationship": "Inner", "Parallel Aware": False, "Async Capable": False, "Startup Cost": 21.48, "Total Cost": 21.48, "Plan Rows": 17, "Plan Width": 102, "Plans": [{"Node Type": "Hash Join", "Parent Relationship": "Outer", "Parallel Aware": False, "Async Capable": False, "Join Type": "Inner", "Startup Cost": 10.84, "Total Cost": 21.48, "Plan Rows": 17, "Plan Width": 102, "Inner Unique": True, "Hash Cond": "(p.owneruserid = u.id)", "Plans": [{"Node Type": "Seq Scan", "Parent Relationship": "Outer", "Parallel Aware": False, "Async Capable": False, "Relation Name": "posts", "Alias": "p", "Startup Cost": 0.0, "Total Cost": 10.5, "Plan Rows": 50, "Plan Width": 8}, {"Node Type": "Hash", "Parent Relationship": "Inner", "Parallel Aware": False, "Async Capable": False, "Startup Cost": 10.62, "Total Cost": 10.62, "Plan Rows": 17, "Plan Width": 102, "Plans": [{"Node Type": "Seq Scan", "Parent Relationship": "Outer", "Parallel Aware": False, "Async Capable": False, "Relation Name": "users", "Alias": "u", "Startup Cost": 0.0, "Total Cost": 10.62, "Plan Rows": 17, "Plan Width": 102, "Filter": "(reputation > 1000)"}]}]}]}]}]}]}]}]}}]

        print("=== Plan to Hints Converter Test ===\n")

        # Test basic conversion
        hints = plan_to_hints(test_plan)
        print(f"Generated hints: {hints}\n")

        # Test verbose conversion
        verbose_result = plan_to_hints_verbose(test_plan)
        print("Verbose result:")
        print(f"  Hint string: {verbose_result['hint_string']}")
        print(f"  Scan hints: {verbose_result['scan_hints']}")
        print(f"  Join hints: {verbose_result['join_hints']}")
        print(f"  Index hints: {verbose_result['index_hints']}")
        sys.exit(0)

    # Get plan JSON from source
    plan_json = None
    if args.stdin:
        plan_json_str = sys.stdin.read()
        try:
            plan_json = json.loads(plan_json_str)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON from stdin: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.file:
        try:
            with open(args.file, "r") as f:
                plan_json = json.load(f)
        except FileNotFoundError:
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in file: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.plan_json:
        try:
            plan_json = json.loads(args.plan_json)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    # Convert and output
    if args.verbose:
        result = plan_to_hints_verbose(plan_json)
        print(f"Hint string: {result['hint_string']}")
        print(f"\nBreakdown:")
        print(f"  Scan hints: {result.get('scan_hints', [])}")
        print(f"  Join hints: {result.get('join_hints', [])}")
        print(f"  Index hints: {result.get('index_hints', [])}")
    else:
        hints = plan_to_hints(plan_json)
        print(hints)
