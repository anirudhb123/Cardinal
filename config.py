#!/usr/bin/env python3
"""
Configuration file for Cardinal query optimization project
"""

import os

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# Database Configuration
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "database": os.getenv("POSTGRES_DB", "cardinal_test"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "your_password"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
}

# Experiment Configuration
EXPERIMENT_CONFIG = {
    "benchmark_iterations": 5,
    "timeout_seconds": 30,
    "max_query_length": 10000,
    "sample_result_limit": 10,
}

# File Paths
PATHS = {
    "data_dir": "data/",
    "results_dir": "results/",
    "queries_dir": "queries/",
    "logs_dir": "logs/",
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "logs/cardinal.log",
}

# PostgreSQL Hint Plan Settings (if using pg_hint_plan extension)
HINT_PLAN_CONFIG = {
    "enabled": False,  # Set to True if pg_hint_plan is installed
    "debug_level": 1,
}

# Query Categories for Testing
QUERY_CATEGORIES = {
    "simple_select": [
        "SELECT * FROM customers WHERE country = 'USA'",
        "SELECT name, email FROM customers ORDER BY name",
    ],
    "joins": [
        """SELECT c.name, COUNT(o.order_id) as order_count 
           FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id 
           GROUP BY c.customer_id, c.name""",
    ],
    "complex": [
        """SELECT c.name, oi.product_name, SUM(oi.quantity) as total_quantity
           FROM customers c
           JOIN orders o ON c.customer_id = o.customer_id
           JOIN order_items oi ON o.order_id = oi.order_id
           WHERE o.status = 'completed'
           GROUP BY c.name, oi.product_name
           ORDER BY total_quantity DESC"""
    ],
}
