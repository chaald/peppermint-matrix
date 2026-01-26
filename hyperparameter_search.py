import argparse
import multiprocessing as mp
import pprint
import wandb

from concurrent.futures import ProcessPoolExecutor

from src.constant import PROJECT_NAME
from src.utils import load_yaml
from main import main

# ==================================
# Start Agent Utility Function
# ==================================
def start_agent(sweep_id, nruns):
    wandb.agent(
        sweep_id=sweep_id,
        function=main,
        project=PROJECT_NAME,
        count=nruns
    )

# ==================================
# Script Entry Point
# ==================================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--nworker", type=int, default=1, help="Number of agents to run in parallel")
    parser.add_argument("--nruns", type=int, default=1, help="Number of runs for each workers")
    parser.add_argument("--sweep_id", type=str, default=None, help="W&B sweep ID. If not provided, create a new sweep.")
    parser.add_argument("--config", type=str, default=None, help="Config path in yaml format.")

    args = parser.parse_args()

    if args.sweep_id is None:
        # Define sweep config programmatically or load from YAML
        sweep_configuration = load_yaml(args.config)
        
        print(f"Started new sweep with params;")
        pprint.pprint(sweep_configuration)

        sweep_id = wandb.sweep(sweep_configuration, project=PROJECT_NAME)
    else:
        sweep_id = args.sweep_id

    with ProcessPoolExecutor(max_workers=args.nworker) as executor:
        futures = [executor.submit(start_agent, sweep_id, args.nruns) for _ in range(args.nworker)]
        for future in futures:
            future.result() # This will raise any exceptions encountered in the worker processes