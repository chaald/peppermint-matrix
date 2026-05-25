import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import polars as pl
import joblib

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.simulation_worker import run_trajectory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_directory", type=str, default="wandb/trajectories")
    parser.add_argument("--n_trajectories", type=int, default=5)
    parser.add_argument("--n_runs", type=int, default=1000)
    parser.add_argument("--probe_interval", type=int, default=10)
    args = parser.parse_args()

    output_directory = args.output_directory

    oracle = joblib.load(os.path.join(output_directory, "oracle.pkl"))
    parameters_config = joblib.load(os.path.join(output_directory, "parameters_config.pkl"))
    feature_names = joblib.load(os.path.join(output_directory, "feature_names.pkl"))
    target_spec = joblib.load(os.path.join(output_directory, "target_spec.pkl"))
    global_best = joblib.load(os.path.join(output_directory, "best_score.pkl"))

    print(f"State loaded from {output_directory}/")
    print(f"Running {args.n_trajectories} UCB + {args.n_trajectories} random trajectories ({args.n_runs} runs each)")
    print()

    all_history = []

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {}
        for seed in range(args.n_trajectories):
            for strategy in ("ucb", "random"):
                key = (strategy, seed)
                futures[key] = executor.submit(
                    run_trajectory,
                    strategy=strategy, seed=seed,
                    parameters_config=parameters_config,
                    oracle_pipeline=oracle,
                    feature_names=feature_names,
                    target_spec=target_spec,
                    global_best=global_best,
                    n_runs=args.n_runs, beta=1.0,
                    estimator_count=128, virtual_sample_count=100, virtual_lambda=2.0,
                    probe_interval=args.probe_interval,
                    output_directory=output_directory,
                    verbose=True,
                )

        for (strategy, seed), future in futures.items():
            history = future.result()
            history.write_parquet(
                os.path.join(output_directory, f"history_{strategy}_{seed}.parquet")
            )

    print("All trajectories complete.")
