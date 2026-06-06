import os

import warnings
warnings.filterwarnings("ignore", message=".*np\\.object.*")

import argparse
import multiprocessing as mp
import pprint
import wandb
import time

from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from typing import Dict
from functools import partial

from src.constant import PROJECT_NAME
from src.utils import load_yaml
from src.utils.config import write_decision_log
from main import main, compile_config

# ==================================
# Start Agent Utility Function
# ==================================
def start_agent(sweep_id: str | None = None, nruns: int = 1):
    assert sweep_id is not None

    wandb.agent(
        sweep_id=sweep_id,
        function=main,
        project=PROJECT_NAME,
        count=nruns
    )

def local_agent(args: argparse.Namespace, nruns: int = 1):
    for _ in range(nruns):
        run_config, decision_metadata = compile_config(args)
        start_time = datetime.now()
        report = main(**run_config)
        end_time = datetime.now()
        if decision_metadata:
            ordered_decision_metadata = {
                "run_id": report["run_id"],
                "run_name": report["run_name"],
                "sweep_id": report["sweep_id"],
                "start_time": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
                "end_time": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            ordered_decision_metadata.update(decision_metadata)
            log_path = run_config.get("log_path", "hyperparameter_search.log.csv")
            write_decision_log(ordered_decision_metadata, log_path)

# ==================================
# Script Entry Point
# ==================================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--nworker", type=int, default=1, help="Number of agents to run in parallel")
    parser.add_argument("--nruns", type=int, default=1, help="Number of runs for each workers")
    parser.add_argument("--method", type=str, default="wandb", help="Sweep method: wandb, random, exhaustive, model_based")
    parser.add_argument("--sweep_id", type=str, default=None, help="W&B sweep ID. If not provided, create a new sweep.")
    parser.add_argument("--config", type=str, default=None, help="Config path in yaml format.")
    parser.add_argument("--model_based_beta", type=float, default=None, help="Exploration weight for UCB formula. Only for --method=model_based.")
    parser.add_argument("--model_based_target", type=str, default=None, help="Target metric key in parquet. Supports weighted composite. Only for --method=model_based.")
    parser.add_argument("--model_based_estimator_count", type=int, default=None, help="Number of trees in Random Forest surrogate. Only for --method=model_based.")
    parser.add_argument("--model_based_virtual_sample_count", type=int, default=None, help="Number of optimistic virtual samples to inject. Only for --method=model_based.")
    parser.add_argument("--model_based_virtual_lambda", type=float, default=None, help="Lambda multiplier for virtual sample target = max(best, mean + lambda * std). Only for --method=model_based.")
    parser.add_argument("--model_based_log_path", type=str, default=None, help="Path for the hyperparameter search decision log. Only for --method=model_based.")

    args = parser.parse_args()

    # Validation
    if args.method == "wandb" and args.config is None and args.sweep_id is None:
        raise ValueError("Either --config or --sweep_id must be provided for wandb sweeps.")
    elif args.method in ["random", "exhaustive", "model_based"]:
        if args.sweep_id is not None:
            raise ValueError("--sweep_id is only valid with --method=wandb.")
        if args.config is None:
            raise ValueError(f"--config must be provided for {args.method} sweeps.")
    elif args.method not in ["wandb", "random", "exhaustive", "model_based"]:
        raise NotImplementedError(f"Sweep method {args.method} not implemented yet.")
    
    if args.method == "wandb":
        # Initialize or retrieve sweep ID
        if args.sweep_id is None:
            # Load sweep configuration from YAML file
            sweep_configuration = load_yaml(args.config)
            
            sweep_id = wandb.sweep(sweep_configuration, project=PROJECT_NAME)
            
            print(f"Started new sweep {sweep_id} with params;")
            pprint.pprint(sweep_configuration)
        else:
            sweep_id = args.sweep_id
    else:
        sweep_id = None

    with ProcessPoolExecutor(max_workers=args.nworker) as executor:
        if args.method == "wandb":
            kernel_function = partial(start_agent, sweep_id, args.nruns)
        else:
            kernel_function = partial(local_agent, args, args.nruns)

        futures = []
        for i in range(args.nworker):
            futures.append(
                executor.submit(kernel_function)
            )
            print(f"Started worker {i+1}/{args.nworker}")
            # add a small delay between starting workers to avoid race conditions during exhaustive search computation
            time.sleep(60) # 1 minute delay, an exhaustive search take ~40 seconds to start and record parameter

        for future in futures:
            future.result() # This will raise any exceptions encountered in the worker processes