
import os
os.environ["WANDB_SILENT"] = "true"

import time
import tqdm
import wandb
import json
import argparse
import concurrent.futures
import polars as pl
import multiprocessing as mp

from functools import partial
from wandb.sdk.internal.internal_api import gql
from typing import List, Dict
from src.constant import PROJECT_NAME
from src.utils.config import fetch_run_metadata


def process_chunk(chunk: List[str], threads_per_process: int = 16) -> List[Dict]:
    """Process a chunk of runs using a shared API object and thread pool."""
    api = wandb.Api(timeout=180)
    
    records = []
    errors = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads_per_process) as executor:
        futures = {executor.submit(fetch_run_metadata, api, run_id): run_id for run_id in chunk}
        
        for future in concurrent.futures.as_completed(futures):
            run_id = futures[future]
            try:
                record = future.result()
                records.append(record)
            except Exception as e:
                errors.append((run_id, str(e)))
    
    return {"records": records, "errors": errors}


def chunk_list(data: List, chunk_size: int = 128) -> List[List]:
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunks.append(data[i:i + chunk_size])
    return chunks


def main(
    model: str = "matrix_factorization",
    process_count: int = 8,
    threads_per_process: int = 32,
    ensure_available_locally: bool = False,
    output_path: str = "wandb/summary.parquet",
    max_retries: int = 3,
):
    api = wandb.Api()
    print(f"Max retries for failed runs: {max_retries}")

    query = """
        query Runs($project: String!, $entity: String!, $cursor: String, $filters: JSONString) {
            project(name: $project, entityName: $entity) {
                runs(first: 256, after: $cursor, filters: $filters) {
                    edges {
                        node {
                            id
                            name
                        }
                        cursor
                    }
                    pageInfo {
                        hasNextPage
                        endCursor
                    }
                }
            }
        }
    """
    query = gql(query)

    all_runs = []
    cursor = None
    filters = json.dumps({
        "config.model": model,
        "state": "finished",
    })

    while True:
        variables = {
            "project": PROJECT_NAME,
            "entity": api.default_entity,
            "cursor": cursor,
            "filters": filters
        }
        
        result = api.client.execute(query, variables)
        runs_data = result["project"]["runs"]
        
        for edge in runs_data["edges"]:
            all_runs.append(edge["node"]["name"])
        
        if not runs_data["pageInfo"]["hasNextPage"]:
            print(f"Fetched {len(all_runs)} runs...")
            break

        cursor = runs_data["pageInfo"]["endCursor"]
        print(f"Fetched {len(all_runs)} runs...")

    records = []
    errors = []
    ctx = mp.get_context('fork')

    attempt = 0
    is_retry = False
    while attempt <= max_retries and (attempt == 0 or len(errors) > 0):
        chunks = chunk_list(all_runs)
        print(f"Split {len(all_runs)} runs into {len(chunks)} chunks of sizes: {[len(c) for c in chunks]}")
        errors = []

        label = f"Retry {attempt}/{max_retries}" if is_retry else "Processing chunks"
        with concurrent.futures.ProcessPoolExecutor(max_workers=process_count, mp_context=ctx) as executor:
            process_kernel = partial(process_chunk, threads_per_process=threads_per_process)
            futures = {executor.submit(process_kernel, chunk): i for i, chunk in enumerate(chunks)}

            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=label):
                result = future.result()
                records.extend(result["records"])
                errors.extend(result["errors"])

        if errors:
            print(f"{'Remaining' if is_retry else ''} Errors: {len(errors)}")
            time.sleep(30)
            is_retry = True
            all_runs = [run_id for run_id, _ in errors]
        attempt += 1

    print(f"\nTotal processed: {len(records)} runs successfully")
    if errors:
        print(f"Total unrecoverable errors after {max_retries} retries: {len(errors)}")
        for run_id, err in errors:
            print(f"  - {run_id}: {err}")

    experiment_runs = pl.DataFrame(records, infer_schema_length=None)
    experiment_runs = experiment_runs.with_columns(
        pl.col("created_at").str.to_datetime("%Y-%m-%dT%H:%M:%SZ")
    )

    if ensure_available_locally:
        experiment_runs = experiment_runs.filter(pl.col("available_locally") == True)

    experiment_runs = experiment_runs.sort("_timestamp", descending=False)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    experiment_runs.write_parquet(output_path)
    print(f"Experiment summary saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="matrix_factorization", help="Model name to filter runs")
    parser.add_argument("--ensure_available_locally", action="store_true", help="Filter runs to only those available locally")
    parser.add_argument("--process_count", type=int, default=8, help="Number of parallel processes to use")
    parser.add_argument("--threads_per_process", type=int, default=32, help="Number of threads per process")
    parser.add_argument("--output_path", type=str, default="wandb/summary.parquet", help="Path to save the output parquet file")
    parser.add_argument("--max_retries", type=int, default=3, help="Number of times to retry failed runs")

    args = parser.parse_args()

    main(
        model=args.model, 
        process_count=args.process_count,
        threads_per_process=args.threads_per_process,
        ensure_available_locally=args.ensure_available_locally,
        output_path=args.output_path,
        max_retries=args.max_retries,
    )
