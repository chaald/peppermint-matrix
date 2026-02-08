import argparse
import multiprocessing as mp
import pprint
import wandb

from concurrent.futures import ProcessPoolExecutor
from typing import Dict
from functools import partial

from src.constant import PROJECT_NAME
from src.utils import load_yaml
from main import main, compile_config

# ==================================
# Start Agent Utility Function
# ==================================
def start_agent(
        sweep_id: str = None, 
        nruns: int = 1,
    ):
    wandb.agent(
        sweep_id=sweep_id,
        function=main,
        project=PROJECT_NAME,
        count=nruns
    )

def local_agent(
        args: argparse.Namespace,
        nruns: int = 1,
    ):
    for _ in range(nruns):
        run_config = compile_config(args)
        main(**run_config)

# ==================================
# Script Entry Point
# ==================================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--nworker", type=int, default=1, help="Number of agents to run in parallel")
    parser.add_argument("--nruns", type=int, default=1, help="Number of runs for each workers")
    parser.add_argument("--method", type=str, default="wandb", help="Sweep method: wandb, random, exhaustive")
    parser.add_argument("--sweep_id", type=str, default=None, help="W&B sweep ID. If not provided, create a new sweep.")
    parser.add_argument("--config", type=str, default=None, help="Config path in yaml format.")

    args = parser.parse_args()

    # Validation
    if args.method == "wandb" and args.config is None and args.sweep_id is None:
        raise ValueError("Either --config or --sweep_id must be provided for wandb sweeps.")
    elif args.method in ["random", "exhaustive"] and args.config is None:
        raise ValueError(f"--config must be provided for {args.method} sweeps.")
    elif args.method not in ["wandb", "random", "exhaustive"]:
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

        futures = [executor.submit(kernel_function) for _ in range(args.nworker)]
        for future in futures:
            future.result() # This will raise any exceptions encountered in the worker processes