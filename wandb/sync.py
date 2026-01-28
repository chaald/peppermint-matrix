
import os
import tqdm
import wandb
import pprint
import warnings
import argparse
import concurrent.futures
import numpy as np
import polars as pl

from wandb.apis.public import Run
from typing import Union, List, Dict

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

def main(
    model: str = "matrix_factorization", 
    ensure_available_locally: bool = False,
    sorting_criterion: List[str] = ["epoch/epoch"],
    output_path: str = "wandb/summary.parquet",
):
    api = wandb.Api() # Initialize Weights & Biases API, used for fetching run data

    sorting_criterion = parse_metric_criterion(sorting_criterion)
    print(f"Using sorting criterion: ")
    pprint.pprint(sorting_criterion)

    def fetch_run_metadata(run: Run, considered_metrics: Union[str, Dict[str, float]] = "epoch/epoch") -> Dict:
        run_config = {}
        for key, value in run.config.items():
            # Convert lists and dicts to strings
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

    batch_size = 16
    records = []
    futures = {}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=batch_size)
    runs:List[Run] = api.runs("feedr/peppermint-matrix", per_page=2*batch_size-1, filters={"config.model": model})
    run_iterator = iter(runs)
    with tqdm.tqdm(total=len(runs), ncols=128) as pbar:
        while len(records) < len(runs):
            # submit new tasks if we empty slots in the batch
            while len(futures) < batch_size and len(records) + len(futures) < len(runs):
                current_runs = next(run_iterator)
                current_future = executor.submit(fetch_run_metadata, current_runs, sorting_criterion)
                futures[current_future] = current_runs

            # check for completed tasks
            finished_futures, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED, timeout=0.1)
            for finished_future in finished_futures:
                finished_run = futures.pop(finished_future)
                records.append(finished_future.result())
                pbar.update(1)

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
    parser.add_argument("--output_path", type=str, default="wandb/summary.parquet", help="Path to save the output CSV file")

    args = parser.parse_args()

    main(
        model=args.model, 
        ensure_available_locally=args.ensure_available_locally,
        sorting_criterion=args.sorting_criterion,
        output_path=args.output_path
    )