#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
import sys
import time
from tempfile import TemporaryDirectory

from openai import OpenAI

from log import Log
from prompt import write_gpt_queries
from util import smart_open, sort_query_list

log = Log()


def find_queries_with_errors(csv_path, success_threshold, subset=None):
    """
    Find queries with errors based on a success threshold.

    Args:
        csv_path (str): Path to the CSV file containing query results.
        success_threshold (int): Minimum number of successful DBMS runs required to keep a query.
        subset (list, optional): Subset of queries to consider. Defaults to None.

    Returns:
        tuple: Lists of queries to delete and queries to keep.
    """
    # Dictionary to track errors for each query
    queries = {}
    umbra = []

    # Convert subset to a set for faster lookups
    if subset is not None:
        subset = set(subset)

    # Open and read the CSV file
    with smart_open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            dbms = row['dbms'].strip()  # Remove surrounding whitespace
            query = row['query']

            if 'query' in row and 'error' in row:
                message = row['error'].strip()  # Remove surrounding whitespace
                error = bool(message)
            else:
                assert 'state' in row and 'message' in row
                error = row['state'] != 'success'
                message = row['message'].strip()

            if subset is not None and query not in subset:
                continue

            if query not in queries:
                queries[query] = []

            # If there is an error (i.e., error is not empty)
            if not error:
                queries[query].append(dbms)
            else:
                if dbms == "umbradev" or dbms == "umbra":
                    umbra.append((query, message))

    # Return queries that have errors in at least two DBMS
    queries_to_delete = [query for query, systems in queries.items() if len(systems) < success_threshold]
    queries_to_keep = [query for query, systems in queries.items() if len(systems) >= success_threshold]

    for (query, error) in umbra:
        if len(queries[query]) > 0:
            log.info_verbose(f"Umbra failed for query {query} but {','.join(queries[query])} succeeded (error: '{error}')")

    for query in queries.keys():
        if len(queries[query]) == 1 and ("umbradev" in queries[query] or "umbra" in queries[query]):
            log.info_verbose(f"Umbra succeeded for query {query} but the other systems failed")

    # Print the number of successful queries
    successful_queries = {}
    for query in queries.keys():
        if len(queries[query]) > 0:
            s = ','.join(queries[query])
            if s not in successful_queries:
                successful_queries[s] = 0
            successful_queries[s] += 1

    for systems in sorted(successful_queries, key=len, reverse=True):
        count = successful_queries[systems]
        log.info(f"{count:5} queries succeeded in {systems}")
    log.info()

    return queries_to_delete, queries_to_keep


def delete_queries(queries_to_delete, query_dir):
    """
    Delete queries from the specified directory.

    Args:
        queries_to_delete (list): List of queries to delete.
        query_dir (str): Directory containing the queries.
    """
    # Iterate over the queries and delete corresponding files
    for query in queries_to_delete:
        query_file_path = os.path.join(query_dir, query)

        if os.path.exists(query_file_path):
            os.remove(query_file_path)


def copy_queries(src_dir, dst_dir, postfix):
    """
    Copy queries from the source directory to the destination directory.

    Args:
        src_dir (str): Source directory containing the queries.
        dst_dir (str): Destination directory to copy the queries to.
        postfix (list): List of postfixes to append to query filenames.

    Returns:
        int: Number of queries copied.
    """
    # Remove all files in the destination directory
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)

    # Iterate over files in the source directory
    count = 0
    files = [file for file in sort_query_list(os.listdir(src_dir)) if file.endswith('.sql') and os.path.isfile(os.path.join(src_dir, file))]

    log.info(f"Copying {len(files)} queries to `{dst_dir}`")
    with log.progress("Copying queries", total=len(files)) as progress:
        for filename in files:
            progress.advance()
            progress.description(filename)

            file_path = os.path.join(src_dir, filename)

            src_file_path = file_path
            for p in postfix:
                if os.path.exists(file_path + p):
                    src_file_path = file_path + p

            dst_file_path = os.path.join(dst_dir, filename)
            shutil.copy2(src_file_path, dst_file_path)
            count += 1

    return count


def find_duplicated_queries(query_dir):
    """
    Find duplicated queries in the specified directory.

    Args:
        query_dir (str): Directory containing the queries.

    Returns:
        list: List of duplicated queries.
    """
    # Dictionary to map file content to file names
    queries = []
    duplicates = []

    files = [file for file in sort_query_list(os.listdir(query_dir)) if file.endswith('.sql') and os.path.isfile(os.path.join(query_dir, file))]
    log.info(f"Checking {len(files):5} queries for duplicates")

    with log.progress("Checking queries", total=len(files)) as progress:
        for filename in files:
            progress.advance()
            progress.description(filename)

            with open(os.path.join(query_dir, filename), 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if content == "":
                    duplicates.append(filename)
                    log.info_verbose(f"Empty query found: {filename}")

                elif content in queries:
                    duplicates.append(filename)
                    log.info_verbose(f"Duplicate query found: {filename}")

                else:
                    queries.append(content)

    return duplicates


def replace_sql(content, modified, old, new):
    """
    Replace text in SQL queries.

    Args:
        content (str): SQL query content.
        modified (bool): Flag indicating if the content was modified.
        old (str): Text to replace.
        new (str): Replacement text.

    Returns:
        tuple: Modified content and modified flag.
    """
    # Replace text in sql queries
    index = 0
    while True:
        index = content.lower().find(old, index)
        if index == -1:
            break

        if content[:index].count("'") % 2 == 0:
            content = content[:index] + new + content[index + len(old):]
            modified = True
        index += 1

    return content, modified


def replace_year(content, modified, old, new):
    """
    Replace year in SQL queries.

    Args:
        content (str): SQL query content.
        modified (bool): Flag indicating if the content was modified.
        old (int): Year to replace.
        new (int): Replacement year.

    Returns:
        tuple: Modified content and modified flag.
    """
    # Replace year in sql queries
    index = 0
    while True:
        index = content.find(str(old), index)
        if index == -1:
            break

        # Check if that the character before and after the year is not alphanumeric
        front = index == 0 or not content[index - 1].isalnum()
        back = index + len(str(old)) >= len(content) or not content[index + len(str(old))].isalnum()
        if back and front:
            content = content[:index] + str(new) + content[index + len(str(old)):]
            modified = True
        index += 1

    return content, modified


def rewrite_queries(query_dir):
    """
    Rewrite queries in the specified directory.

    Args:
        query_dir (str): Directory containing the queries.

    Returns:
        int: Number of queries rewritten.
    """
    # Find all SQL files in the directory
    files = [file for file in sort_query_list(os.listdir(query_dir)) if file.endswith('.sql') and os.path.isfile(os.path.join(query_dir, file))]

    queries_to_rewrite = []
    log.info(f"Rewriting {len(files)} queries")
    with log.progress("Rewriting queries", total=len(files)) as progress:
        for filename in files:
            progress.advance()
            progress.description(filename)

            file_path = os.path.join(query_dir, filename)

            # Only process files, skip directories
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()

            modified = False

            # Remove all SQL comments
            while True:
                comment_start = content.find('--')
                if comment_start == -1:
                    break

                comment_end = content.find('\n', comment_start)
                if comment_end == -1:
                    content = content[:comment_start]
                else:
                    content = content[:comment_start] + content[comment_end:]

                modified = True

            # Find the position of the first occurrence of SELECT or WITH (case-insensitive)
            select_pos = content.lower().find('select')
            with_pos = content.lower().find('with')
            first_pos = min(select_pos, with_pos) if select_pos != -1 and with_pos != -1 else max(select_pos, with_pos)

            # Check if 'with' or 'select are not the first character
            if first_pos > 0:
                # Trim the content before 'with' or 'select'
                content = content[first_pos:]
                modified = True

            # Find the first semicolon
            semicolon_index = 0
            while semicolon_index != -1:
                semicolon_index = content.find(';', semicolon_index)
                if semicolon_index == -1 or content[:semicolon_index].count("'") % 2 == 0:
                    break

                semicolon_index += 1

            # Check if the semicolon is not the last character
            if semicolon_index != -1 and semicolon_index < len(content) - 1:
                # Trim content after the first semicolon
                content = content[:semicolon_index + 1]  # Include the semicolon itself
                modified = True

            # Replace all occurrences of current_date, current_timestamp, current_time and now() with a fixed date
            content, modified = replace_sql(content, modified, 'current_date', "cast('2024-10-01' as date)")
            content, modified = replace_sql(content, modified, 'current_timestamp', "cast('2024-10-01 12:34:56' as timestamp)")
            content, modified = replace_sql(content, modified, 'current_time', "cast('12:34:56' as time)")
            content, modified = replace_sql(content, modified, 'now()', "cast('2024-10-01 12:34:56' as timestamp)")

            # Map dates to the correct range
            year_mapping = {
                "tpch": [(2024, 1998), (2023, 1997), (2022, 1996), (2021, 1995), (2020, 1994), (2019, 1993), (2018, 1992)],
                "tpcds": [(2025, 2003), (2024, 2002), (2023, 2001), (2022, 2000), (2021, 1999), (2020, 1998), (2019, 1998), (2018, 1998), (2017, 1998), (2016, 1998), (2015, 1998)]
            }
            for benchmark, years in year_mapping.items():
                if benchmark in os.path.abspath(query_dir):
                    for old_year, new_year in years:
                        content, modified = replace_year(content, modified, str(old_year), str(new_year))

            if modified:
                content = content.strip()

                # Check if the query has already been rewritten
                if os.path.exists(file_path + "_rewritten"):
                    with open(file_path + "_rewritten", 'r', encoding='utf-8') as file:
                        old_content = file.read().strip()
                        if old_content != content:
                            queries_to_rewrite.append((filename, content))

    exists_compatible = 0
    for filename, _ in queries_to_rewrite:
        if os.path.exists(os.path.join(query_dir, filename + "_compatible")):
            exists_compatible += 1

    log.info(f"Found {len(queries_to_rewrite)} queries to rewrite")
    if exists_compatible > 0:
        log.warn(f"Deleting {exists_compatible} compatible queries that are outdated")
        if not log.confirm("Do you want to continue?"):
            sys.exit(0)

    # Rewrite the queries
    with log.progress("Rewriting queries", total=len(queries_to_rewrite)) as progress:
        for filename, content in queries_to_rewrite:
            progress.advance()
            progress.description(os.path.basename(filename))

            # Delete an outdated compatible query
            if os.path.exists(os.path.join(query_dir, filename + "_compatible")):
                log.info_verbose(f"Deleting {filename} compatible query")
                os.remove(os.path.join(query_dir, filename + "_compatible"))

            # Rewrite the file
            with open(os.path.join(query_dir, filename + "_rewritten"), 'w', encoding='utf-8') as file:
                log.info_verbose(f"Rewriting {filename} query")
                file.write(content)
    return len(queries_to_rewrite)


def make_queries_compatible(query_dir, queries):
    """
    Make queries compatible with different SQL dialects using GPT-4o-mini.

    Args:
        query_dir (str): Directory containing the queries.
        queries (list): List of queries to make compatible.

    Returns:
        int: Number of queries regenerated.
    """
    tempdir = TemporaryDirectory()
    input_file = os.path.join(tempdir.name, "input.jsonl")

    batch_json = {
        "custom_id": "",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "temperature": 1.0,
            "max_tokens": 1000,
            "messages": [
                {"role": "system", "content": "You are a SQL generator that cannot speak."},
                {"role": "user", "content": ""}
            ]
        }
    }

    prompt = "Make the following SQL query more compatible with different SQL dialects. The query might contain \'::\' casts, rewrite them to standard SQL. The query might contain some errors, fix them if you can find any mistakes. Remember to put all ungrouped columns and columns that appear in windows into the group by clause. Do not explain the query, only output the converted query."

    count = 0
    with open(input_file, 'w') as f:
        for query in queries:
            query_file_path = os.path.join(query_dir, query)

            # Skip if the query is already compatible
            if os.path.exists(query_file_path + "_compatible"):
                continue

            # Check if a rewritten version exists
            if os.path.exists(query_file_path + "_rewritten"):
                query_file_path += "_rewritten"

            # Create the request JSON
            with open(query_file_path, 'r', encoding='utf-8') as file:
                sql = file.read().strip()
                batch_json["custom_id"] = query
                batch_json["body"]["messages"][1]["content"] = prompt + "\n```sql\n" + sql + "\n```"
                f.write(json.dumps(batch_json) + "\n")

            count += 1

    if count == 0:
        log.info("All queries are already compatible")
        return 0

    log.warn(f"Regenerating {count} queries with GPT-4o-mini will incur charges to your OpenAI account.")
    if not log.confirm("Do you want to continue?"):
        sys.exit(0)

    log.info(f"Starting batch generation with GPT-4o-mini ...")
    start = time.time()

    # Create the batch
    client = OpenAI()
    batch_input_file = client.files.create(file=open(input_file, "rb"), purpose="batch")
    batch = client.batches.create(input_file_id=batch_input_file.id, endpoint="/v1/chat/completions", completion_window="24h", metadata={"description": "1k"})
    log.info(f"GPT-4o-mini batch: {batch.id}")
    time.sleep(10)

    # Wait for the batch to complete
    result = client.batches.retrieve(batch.id)
    with log.progress(result.status.capitalize(), count) as progress:
        while result.status != 'completed':
            if result.status == 'failed':
                raise Exception(f"Batch {batch.id} failed: {result}")

            time.sleep(30)
            try:
                result = client.batches.retrieve(batch.id)
                progress.description(result.status.capitalize())
                if result.request_counts:
                    progress.completed(result.request_counts.completed)
            except Exception as e:
                log.warn(str(e))

    log.info(f"Batch {batch.id} completed in {int((time.time() - start) / 60)} minutes")

    # Download the output file
    file_response = client.files.content(result.output_file_id)

    # Write the output to a file
    write_gpt_queries(query_dir, file_response.text.split('\n'), postfix="_compatible")

    return count


def write_sql_queries_file(query_dir, query_list):
    """
    Write SQL queries to a file.

    Args:
        query_dir (str): Directory containing the queries.
        query_list (list): List of queries to write.
    """
    dest_file_path = os.path.join(query_dir, '..', 'queries.sql')
    with open(dest_file_path, 'w', encoding='utf-8') as dest_file:
        dest_file.write("\\o -\n")
        for filename in sort_query_list(query_list):
            file_path = os.path.join(query_dir, filename)
            dest_file.write(f"\\echo {filename}\n")
            dest_file.write(f"\\i {file_path}\n")


def main():
    """
    Main function to generate SQL queries based on the provided prompt and dataset.
    """
    log.header("Select SQL Queries")

    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("dataset", help="Dataset of the benchmark")
    argparser.add_argument("version", help="Version of the benchmark")
    argparser.add_argument("-c", "--compatible", help="The compatible queries")
    argparser.add_argument("-p", "--parse", help="The parseable queries")
    argparser.add_argument("-e", "--execution", help="The executable queries")
    args = argparser.parse_args()

    query_src_dir = os.path.join(args.version, args.dataset, "queries_generated")
    query_dest_dir = os.path.join(args.version, args.dataset, "queries")

    for result_file in [args.compatible, args.parse, args.execution]:
        if result_file and not os.path.exists(result_file):
            raise Exception(f"Result file {result_file} does not exist")
        elif result_file and args.dataset not in result_file:
            log.warn(f"Result file {result_file} was not created for dataset {args.dataset}")
            if not log.confirm("Do you want to continue?"):
                sys.exit(0)

    # Rewrite the queries
    log.header2("Rewrite queries")
    rewritten_queries = rewrite_queries(query_src_dir)

    postfix = ["_rewritten"]
    not_compatible, compatible = [], []
    regenerated_queries = 0

    # Make queries compatible
    if args.compatible:
        log.header2("Make queries compatible")

        not_compatible, compatible = find_queries_with_errors(args.compatible, 3)

        log.info(f"Found {len(not_compatible):5} incompatible queries")

        # Find queries with '::'
        incompatible_casts = 0
        for query in compatible:
            query_file_path = os.path.join(query_src_dir, query)
            if os.path.exists(query_file_path + "_rewritten"):
                query_file_path += "_rewritten"

            with open(query_file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if content.find('::') != -1 and query not in not_compatible:
                    not_compatible.append(query)
                    incompatible_casts += 1
        log.info(f"Found {incompatible_casts:5} queries with '::' casts")

        regenerated_queries = make_queries_compatible(query_src_dir, not_compatible)
        postfix.append("_compatible")

    # Copy the rewritten/compatible queries to the destination directory
    log.header2("Copy queries")
    query_count = copy_queries(query_src_dir, query_dest_dir, postfix)

    # Find duplicated queries in the directory
    log.header2("Remove duplicates")
    duplicates = find_duplicated_queries(query_src_dir)
    log.info(f"Deleting {len(duplicates)} duplicated queries")
    delete_queries(duplicates, query_dest_dir)

    # Find queries with errors in at least two DBMS
    not_parseable, parseable = [], []
    if args.parse:
        log.header2("Select parseable queries")
        not_parseable, parseable = find_queries_with_errors(args.parse, 2)
        log.info(f"Deleting {len(not_parseable)} not parseable queries")
        delete_queries(not_parseable, query_dest_dir)

    # Restore queries that were parseable before making them compatible
    restored_queries = 0
    if args.compatible and len(not_parseable) > 0:
        log.header2("Restore parseable queries")

        # Finda all queries that were parseable by two DBMS before rewritting
        _, parseable_rewritten = find_queries_with_errors(args.compatible, 2, subset=not_parseable)
        with log.progress("Restoring queries", len(parseable_rewritten)) as progress:
            for query in parseable_rewritten:
                progress.advance()
                progress.description(query)

                src_file_path = os.path.join(query_src_dir, query)
                if os.path.exists(src_file_path + "_rewritten"):
                    src_file_path += "_rewritten"

                dst_file_path = os.path.join(query_dest_dir, query)
                shutil.copy2(src_file_path, dst_file_path)
                restored_queries += 1

        log.info(f"Restored {restored_queries:5} parseable queries")

    # Find queries with errors in at least two DBMS
    not_executable, executable = [], []
    if args.execution:
        log.header2("Select executable queries")
        not_executable, executable = find_queries_with_errors(args.execution, 1)
        log.info(f"Deleting {len(not_executable)} not executable queries")
        delete_queries(not_executable, query_dest_dir)

    # Write the queries.sql file
    write_sql_queries_file(query_dest_dir, executable)

    log.header2("Summary")
    log.info(f"{rewritten_queries:5} queries rewritten")
    log.info(f"{regenerated_queries:5} queries regenerated ({len(not_compatible)} not compatible, {len(compatible)} compatible)")
    log.info(f"{query_count:5} queries copied")
    log.info(f"{len(duplicates):5} duplicated queries deleted ({query_count - len(duplicates)} distinct queries)")
    log.info(f"{len(not_parseable):5} not parseable queries deleted ({len(parseable)} parseable queries)")
    log.info(f"{restored_queries:5} queries restored that were parseable before making them compatible")
    log.info(f"{len(not_executable):5} not executable queries deleted ({len(executable)} executable queries)")
    log.info(f"{query_count - len(duplicates) - len(not_parseable) + restored_queries - len(not_executable):5} queries remaining")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(e)
        raise e
