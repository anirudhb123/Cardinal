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
                all_join_tables = []
                for child_tables in child_tables_list:
                    all_join_tables.extend(child_tables)

                if len(all_join_tables) >= 2:
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

        if node_type in self.SCAN_HINTS:
            table_alias = self._get_table_alias(node)
            return [table_alias] if table_alias else []

        result = []
        if "Plans" in node:
            for child in node["Plans"]:
                child_order = self._extract_join_order(child)
                result.extend(child_order)

        return result

    def _build_hint_string(self) -> str:
        """Build the final pg_hint_plan hint string"""
        hints = []
        hints.extend(self.scan_hints)

        seen_joins = set()
        for join_hint in self.join_hints:
            if join_hint not in seen_joins:
                hints.append(join_hint)
                seen_joins.add(join_hint)

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


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert PostgreSQL execution plan JSON to pg_hint_plan hints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("plan_json", nargs="?", help="Execution plan JSON string")
    parser.add_argument("-f", "--file", help="Read execution plan from JSON file")
    parser.add_argument("--stdin", action="store_true", help="Read execution plan from stdin")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed hint breakdown")
    parser.add_argument("--test", action="store_true", help="Run built-in test with example plan")

    args = parser.parse_args()

    if args.test or (not args.plan_json and not args.file and not args.stdin):
        test_plan = [{"Plan": {"Node Type": "Limit", "Plans": [{"Node Type": "Seq Scan", "Relation Name": "users", "Alias": "u"}]}}]
        print("=== Plan to Hints Converter Test ===\n")
        hints = plan_to_hints(test_plan)
        print(f"Generated hints: {hints}\n")
        sys.exit(0)

    plan_json = None
    if args.stdin:
        plan_json = json.loads(sys.stdin.read())
    elif args.file:
        with open(args.file, "r") as f:
            plan_json = json.load(f)
    elif args.plan_json:
        plan_json = json.loads(args.plan_json)
    else:
        parser.print_help()
        sys.exit(1)

    if args.verbose:
        result = plan_to_hints_verbose(plan_json)
        print(f"Hint string: {result['hint_string']}")
        print(f"\nBreakdown:\n  Scan hints: {result.get('scan_hints', [])}\n  Join hints: {result.get('join_hints', [])}\n  Index hints: {result.get('index_hints', [])}")
    else:
        print(plan_to_hints(plan_json))
