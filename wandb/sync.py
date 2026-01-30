
import os
import tqdm
import wandb
import json
import pprint
import warnings
import argparse
import concurrent.futures
import numpy as np
import polars as pl
import pandas as pd
import multiprocessing as mp

from functools import partial
from wandb.apis.public import Run
from wandb.sdk.internal.internal_api import gql
from typing import Union, List, Dict
from src.constant import PROJECT_NAME

pd.set_option('future.no_silent_downcasting', True)
warnings.filterwarnings('ignore', category=FutureWarning, message='.*Downcasting behavior in `replace`.*')

def parse_metric_criterion(criterion_list: List[str]) -> Union[str, Dict[str, float]]:
    criterion_dict = {}
    for item in criterion_list:
        if ":" in item:
            metric, weight = item.split(":")
            criterion_dict[metric] = float(weight)
        else:
            criterion_dict[item] = 1.0

    # If only one metric with weight 1.0, return as string
    if len(criterion_dict) == 1 and list(criterion_dict.values())[0] == 1.0:
        return list(criterion_dict.keys())[0]
    
    return criterion_dict

def fetch_run_metadata(api: wandb.Api, run_id: str, considered_metrics: Union[str, Dict[str, float]] = "epoch/epoch") -> Dict:
    """Fetch full metadata for a single run by ID."""
    run: Run = api.run(f"{api.default_entity}/{PROJECT_NAME}/{run_id}")
    
    run_config = {}
    for key, value in run.config.items():
        if isinstance(value, (list, dict)):
            run_config[key] = str(value)
        else:
            run_config[key] = value

    run_history = run.history()
    run_history = run_history.replace({"Infinity": np.inf, "NaN": np.nan})

    if isinstance(considered_metrics, str):
        run_history["score"] = run_history[considered_metrics]
    elif isinstance(considered_metrics, dict):
        run_history["score"] = sum(
            run_history[metric] * weight for metric, weight in considered_metrics.items()
        )
    else:
        raise ValueError("considered_metrics must be either a string or a dictionary")
    
    best_summary = run_history.iloc[run_history["score"].argmax()]
    best_summary = {f"best:{key}": val for key, val in best_summary.items()}
    
    return {
        "run_id": run.id,
        "run_name": run.name,
        "sweep_id": run.sweep.id if run.sweep else None,
        "model": run.config.get("model"),
        "created_at": run.created_at,
        **run_config,
        **{metric: run_history[metric].to_list() for metric in run_history},
        **best_summary,
        "gpu_type": run.metadata.get("gpu"),
        "cpu_count": run.metadata.get("cpu_count"),
    }

def process_chunk(chunk: List[str], considered_metrics: Union[str, Dict[str, float]], threads_per_process: int = 16) -> List[Dict]:
    """Process a chunk of runs using a shared API object and thread pool."""
    # Each process creates its own API instance
    api = wandb.Api(timeout=60)
    
    records = []
    errors = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads_per_process) as executor:
        futures = {executor.submit(fetch_run_metadata, api, run_id, considered_metrics): run_id for run_id in chunk}
        
        for future in concurrent.futures.as_completed(futures):
            run_id = futures[future]
            try:
                record = future.result()
                records.append(record)
            except Exception as e:
                errors.append((run_id, str(e)))
    
    return {"records": records, "errors": errors}

def chunk_list(data: List, chunk_size: int = 128) -> List[List]:
    """Split a list into chunks of specified size. Last chunk may be smaller."""
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunks.append(data[i:i + chunk_size])
    return chunks

def main(
    model: str = "matrix_factorization",
    process_count: int = 8,
    threads_per_process: int = 32,
    ensure_available_locally: bool = False,
    sorting_criterion: List[str] = ["epoch/epoch"],
    output_path: str = "wandb/summary.parquet",
):
    api = wandb.Api() # Initialize Weights & Biases API, used for fetching run data

    sorting_criterion = parse_metric_criterion(sorting_criterion)
    print(f"Using sorting criterion: ")
    pprint.pprint(sorting_criterion)

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

    # Split all runs into chunks
    chunks = chunk_list(all_runs)
    print(f"Split {len(all_runs)} runs into {len(chunks)} chunks of sizes: {[len(c) for c in chunks]}")

    # Process chunks in parallel using multiprocessing where each process uses multi-threading
    records = []
    errors = []
    ctx = mp.get_context('fork')
    with concurrent.futures.ProcessPoolExecutor(max_workers=process_count, mp_context=ctx) as executor:
        process_kernel = partial(process_chunk, considered_metrics=sorting_criterion, threads_per_process=threads_per_process)
        
        futures = {executor.submit(process_kernel, chunk): i for i, chunk in enumerate(chunks)}
        
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing chunks"):
            result = future.result()
            records.extend(result["records"])
            errors.extend(result["errors"])

    print(f"\nProcessed {len(records)} runs successfully")
    if errors:
        print(f"Errors: {len(errors)}")
        for run_id, err in errors[:5]:
            print(f"  - {run_id}: {err}")

    # Create a Polars DataFrame from the records
    experiment_runs = pl.DataFrame(records, infer_schema_length=None)
    experiment_runs = experiment_runs.with_columns(
        pl.col("created_at").str.to_datetime("%Y-%m-%dT%H:%M:%SZ")
    )
        
    # Tag run as available locally if the model files exist
    local_run_ids = []
    local_sweep_ids = os.listdir(f"./models/{model}/")
    for sweep_id in local_sweep_ids:
        local_run_ids.extend([run_id for run_id in os.listdir(f"./models/{model}/{sweep_id}/")])
        
    experiment_runs = experiment_runs.with_columns(
        available_locally=pl.col("run_id").is_in(local_run_ids)
    )

    if ensure_available_locally:
        experiment_runs = experiment_runs.filter(pl.col("available_locally") == True)

    experiment_runs = experiment_runs.sort("_timestamp", descending=False)
    experiment_runs = experiment_runs.with_columns(
        run_duration_second=pl.col("_runtime").list.max(),
        run_duration_minute=(pl.col("_runtime").list.max() / 60)
    )
    experiment_runs.select(
        pl.col("run_id"),
        pl.col("run_name"),
        pl.col("sweep_id"),
        pl.col("model"),
        pl.col("embedding_dimension"),
        pl.col("shuffle"),
        pl.col("best:epoch/epoch"),
        pl.col("best:epoch/train_loss"),
        pl.col("best:epoch/test_loss"),
        pl.col("best:epoch/test_recall@10"),
        pl.col("best:epoch/test_ndcg@10"),
    )

    # Save the DataFrame to a Parquet file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    experiment_runs.write_parquet(output_path)
    print(f"Experiment summary saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="matrix_factorization", help="Model name to filter runs")
    parser.add_argument("--ensure_available_locally", action="store_true", help="Filter runs to only those available locally")
    parser.add_argument("--sorting_criterion", type=str, nargs="+", default=["epoch/epoch"], help="Metric to sort runs by")
    parser.add_argument("--process_count", type=int, default=8, help="Number of parallel processes to use")
    parser.add_argument("--threads_per_process", type=int, default=32, help="Number of threads per process")
    parser.add_argument("--output_path", type=str, default="wandb/summary.parquet", help="Path to save the output CSV file")

    args = parser.parse_args()

    main(
        model=args.model, 
        process_count=args.process_count,
        threads_per_process=args.threads_per_process,
        ensure_available_locally=args.ensure_available_locally,
        sorting_criterion=args.sorting_criterion,
        output_path=args.output_path
    )