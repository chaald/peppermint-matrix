import os
import sys
import io
import contextlib
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import polars as pl
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import model_based_parse_parameters, parse_parameters


def run_trajectory(
    strategy,
    parameters_config,
    oracle_pipeline,
    feature_names,
    target_spec,
    global_best,
    seed=0,
    beta=1.0,
    n_runs=500,
    estimator_count=128,
    virtual_sample_count=100,
    virtual_lambda=3.0,
    probe_interval=10,
    output_directory="wandb/trajectories",
    verbose=True,
):
    os.makedirs(output_directory, exist_ok=True)
    rng = np.random.RandomState(seed)

    simulated_parquet = os.path.join(output_directory, f"trajectory_{strategy}_{seed}.parquet")
    log_path = os.path.join(output_directory, f"trajectory_{strategy}_{seed}.log.csv")

    log_transformer = oracle_pipeline.named_steps["log_reg"]
    random_forest = oracle_pipeline.named_steps["rf"]

    simulated_records = []
    history_records = []

    best_found = -np.inf
    cumulative_regret = 0.0

    for run_idx in range(n_runs):
        if strategy == "ucb":
            suppress_output = io.StringIO()
            with contextlib.redirect_stdout(suppress_output):
                config_dict, meta = model_based_parse_parameters(
                    parameters_config,
                    beta=beta,
                    target=target_spec,
                    estimator_count=estimator_count,
                    summary_path=simulated_parquet,
                    log_path=log_path,
                    virtual_sample_count=virtual_sample_count,
                    virtual_lambda=virtual_lambda,
                )
            config_tuple = tuple(config_dict[col] for col in feature_names)
        else:
            config_dict = parse_parameters(parameters_config)
            config_tuple = tuple(config_dict[col] for col in feature_names)

        config_vector = np.array([[config_tuple[i] for i, col in enumerate(feature_names)]])
        transformed = log_transformer.transform(config_vector)
        tree_preds = np.stack([tree.predict(transformed) for tree in random_forest.estimators_], axis=1)
        oracle_mu = tree_preds.mean()
        true_score = float(oracle_mu)

        if true_score > best_found:
            best_found = true_score

        simple_regret = global_best - best_found
        run_regret = global_best - true_score
        cumulative_regret += run_regret

        selected_config = {col: config_tuple[i] for i, col in enumerate(feature_names)}
        selected_config[target_spec] = [true_score]
        selected_config["model"] = "matrix_factorization"
        simulated_records.append(selected_config)

        simulated_dataframe = pl.DataFrame(simulated_records)
        simulated_dataframe.write_parquet(simulated_parquet)

        history_record = {
            "run": run_idx,
            "strategy": strategy,
            "seed": seed,
            "score": round(true_score, 6),
            "best_found": round(best_found, 6),
            "simple_regret": round(simple_regret, 6),
            "run_regret": round(run_regret, 6),
            "cumulative_regret": round(cumulative_regret, 6),
        }
        for i, col in enumerate(feature_names):
            history_record[col] = config_tuple[i]

        if strategy == "ucb":
            history_record["esm"] = meta.get("esm")
            history_record["coverage_75"] = meta.get("coverage@75")
        elif run_idx % probe_interval == 0:
            suppress_output = io.StringIO()
            with contextlib.redirect_stdout(suppress_output):
                _, probe_meta = model_based_parse_parameters(
                    parameters_config,
                    beta=beta,
                    target=target_spec,
                    estimator_count=estimator_count,
                    summary_path=simulated_parquet,
                    log_path=log_path + ".probe",
                    virtual_sample_count=virtual_sample_count,
                    virtual_lambda=virtual_lambda,
                )
            history_record["esm"] = probe_meta.get("esm")
            history_record["coverage_75"] = probe_meta.get("coverage@75")
        else:
            history_record["esm"] = None
            history_record["coverage_75"] = None

        history_records.append(history_record)

        if verbose and (run_idx + 1) % 20 == 0:
            print(f"  [{strategy.upper()}] seed={seed:2d}  run {run_idx+1:4d}/{n_runs}  best={best_found:.6f}  regret={simple_regret:.6f}")

    if verbose:
        print(f"  ✓ {strategy} seed={seed}: {len(history_records)} runs")

    return pl.DataFrame(history_records)


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
