import json
import time
import wandb
import tqdm
import argparse
import concurrent.futures
import multiprocessing as mp

from functools import partial
from wandb.apis.public import Run
from wandb.sdk.internal.internal_api import gql
from typing import List, Dict, Tuple

from src.constant import PROJECT_NAME

def fetch_run_names(api: wandb.Api, model: str) -> List[str]:
    """Fetch all finished run names for a given model using GraphQL (much faster than api.runs)."""
    query = gql("""
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
    """)

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
            "filters": filters,
        }

        result = api.client.execute(query, variables)
        runs_data = result["project"]["runs"]

        for edge in runs_data["edges"]:
            all_runs.append(edge["node"]["name"])

        if not runs_data["pageInfo"]["hasNextPage"]:
            break

        cursor = runs_data["pageInfo"]["endCursor"]
        print(f"Fetched {len(all_runs)} runs...")

    print(f"Fetched {len(all_runs)} runs...")
    return all_runs

def adjust_run_config(api: wandb.Api, run_name: str) -> Tuple[str, bool, str]:
    """
    Fetch a single run and update its config if needed.
    Returns (run_name, was_updated, error_or_empty).
    """
    try:
        run: Run = api.run(f"{api.default_entity}/{PROJECT_NAME}/{run_name}")

        updated = False

        # ── Add default-value adjustments here ──────────────────────
        run.config["tracker"] = "wandb"
        updated = True
        # ────────────────────────────────────────────────────────────

        if updated:
            run.update()

        return (run_name, updated, "")
    except Exception as e:
        return (run_name, False, str(e))


def process_chunk(
    chunk: List[str],
    threads_per_process: int = 16,
) -> Dict:
    """Process a chunk of runs: each process creates its own API and fans out threads."""
    api = wandb.Api(timeout=180)

    updated: List[str] = []
    skipped: List[str] = []
    errors: List[Tuple[str, str]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads_per_process) as executor:
        futures = {
            executor.submit(adjust_run_config, api, run_name): run_name
            for run_name in chunk
        }

        for future in concurrent.futures.as_completed(futures):
            run_name, was_updated, err = future.result()
            if err:
                errors.append((run_name, err))
            elif was_updated:
                updated.append(run_name)
            else:
                skipped.append(run_name)

    return {"updated": updated, "skipped": skipped, "errors": errors}


def chunk_list(data: List, chunk_size: int = 128) -> List[List]:
    """Split a list into chunks of specified size."""
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


def main(
    model: str = "matrix_factorization",
    process_count: int = 8,
    threads_per_process: int = 32,
    max_retries: int = 3,
):
    api = wandb.Api()

    # ── 1. Fast GraphQL fetch of run names ──────────────────────────
    all_runs = fetch_run_names(api, model)

    # ── 2. Process with multiprocessing + multithreading ────────────
    total_updated = 0
    total_skipped = 0
    errors: List[Tuple[str, str]] = []
    ctx = mp.get_context("fork")

    attempt = 0
    is_retry = False
    while attempt <= max_retries and (attempt == 0 or len(errors) > 0):
        chunks = chunk_list(all_runs)
        print(f"Split {len(all_runs)} runs into {len(chunks)} chunks of sizes: {[len(c) for c in chunks]}")
        errors = []

        label = f"Retry {attempt}/{max_retries}" if is_retry else "Adjusting configs"
        with concurrent.futures.ProcessPoolExecutor(max_workers=process_count, mp_context=ctx) as executor:
            kernel = partial(process_chunk, threads_per_process=threads_per_process)
            futures = {executor.submit(kernel, chunk): i for i, chunk in enumerate(chunks)}

            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=label):
                result = future.result()
                total_updated += len(result["updated"])
                total_skipped += len(result["skipped"])
                errors.extend(result["errors"])

        if errors:
            print(f"{'Remaining ' if is_retry else ''}Errors: {len(errors)}")
            time.sleep(30)
            is_retry = True
            all_runs = [run_name for run_name, _ in errors]

        attempt += 1

    # ── 3. Summary ──────────────────────────────────────────────────
    print(f"\nTotal runs updated : {total_updated}")
    print(f"Total runs skipped : {total_skipped}")
    if errors:
        print(f"Unrecoverable errors after {max_retries} retries: {len(errors)}")
        for run_name, err in errors:
            print(f"  - {run_name}: {err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adjust W&B run configs with default values")
    parser.add_argument("--model", type=str, default="matrix_factorization", help="Model name to filter runs")
    parser.add_argument("--process_count", type=int, default=4, help="Number of parallel processes")
    parser.add_argument("--threads_per_process", type=int, default=8, help="Threads per process")
    parser.add_argument("--max_retries", type=int, default=3, help="Retries for failed runs")

    args = parser.parse_args()

    main(
        model=args.model,
        process_count=args.process_count,
        threads_per_process=args.threads_per_process,
        max_retries=args.max_retries,
    )
