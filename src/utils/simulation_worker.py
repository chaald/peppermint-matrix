import os
import io
import contextlib
import re
import warnings
from collections import deque

import numpy as np
import polars as pl
import sklearn

from src.utils.config import model_based_parse_parameters, parse_parameters

warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")


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
    max_samples=0.1,
    min_samples_leaf=3,
    probe_interval=10,
    output_directory="wandb/trajectories",
    label=None,
    verbose=True,
):
    os.makedirs(output_directory, exist_ok=True)
    rng = np.random.RandomState(seed)

    safe_label = re.sub(r'[^a-zA-Z0-9_-]+', '_', (label or strategy)).strip('_')

    simulated_parquet = os.path.join(output_directory, f"trajectory_{safe_label}_{seed}.parquet")
    log_path = os.path.join(output_directory, f"trajectory_{safe_label}_{seed}.log.csv")

    preprocessor = oracle_pipeline.named_steps["preprocess"]
    random_forest = oracle_pipeline.named_steps["random_forest"]

    simulated_records = []
    history_records = []
    recent_scores = deque(maxlen=20)
    trajectory_name = label or f"{strategy}-{seed}"

    best_found = -np.inf
    cumulative_regret = 0.0

    for run_idx in range(n_runs):
        if strategy == "ucb":
            suppress_output = io.StringIO()
            with contextlib.redirect_stdout(suppress_output):
                config_dict, decision_metadata = model_based_parse_parameters(
                    parameters_config,
                    beta=beta,
                    target=target_spec,
                    estimator_count=estimator_count,
                    summary_path=simulated_parquet,
                    log_path=log_path,
                    virtual_sample_count=virtual_sample_count,
                    virtual_lambda=virtual_lambda,
                    max_samples=max_samples,
                    min_samples_leaf=min_samples_leaf,
                )
            config_tuple = tuple(config_dict[col] for col in feature_names)
        else:
            config_dict = parse_parameters(parameters_config)
            config_tuple = tuple(config_dict[col] for col in feature_names)

        config_df = pl.DataFrame({col: [config_tuple[i]] for i, col in enumerate(feature_names)})
        with sklearn.config_context(transform_output="pandas"):
            transformed = preprocessor.transform(config_df)
        tree_preds = np.stack([tree.predict(transformed.to_numpy()) for tree in random_forest.estimators_], axis=1)
        oracle_mu = tree_preds.mean()
        true_score = float(oracle_mu)
        recent_scores.append(true_score)

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
            "beta": beta,
            "estimator_count": estimator_count,
            "max_samples": max_samples,
            "virtual_lambda": virtual_lambda,
            "virtual_sample_count": virtual_sample_count,
            "min_samples_leaf": min_samples_leaf,
            "score": round(true_score, 6),
            "best_found": round(best_found, 6),
            "simple_regret": round(simple_regret, 6),
            "run_regret": round(run_regret, 6),
            "cumulative_regret": round(cumulative_regret, 6),
        }
        for i, col in enumerate(feature_names):
            history_record[col] = config_tuple[i]

        if strategy == "ucb":
            history_record["esm"] = decision_metadata.get("esm")
            history_record["coverage_75"] = decision_metadata.get("coverage@75")
            history_record["surprise_rate"] = decision_metadata.get("surprise_rate")
        elif run_idx % probe_interval == 0:
            suppress_output = io.StringIO()
            with contextlib.redirect_stdout(suppress_output):
                _, probe_metadata = model_based_parse_parameters(
                    parameters_config,
                    beta=beta,
                    target=target_spec,
                    estimator_count=estimator_count,
                    summary_path=simulated_parquet,
                    log_path=log_path + ".probe",
                    virtual_sample_count=virtual_sample_count,
                    virtual_lambda=virtual_lambda,
                    max_samples=max_samples,
                    min_samples_leaf=min_samples_leaf,
                )
            history_record["esm"] = probe_metadata.get("esm")
            history_record["coverage_75"] = probe_metadata.get("coverage@75")
            history_record["surprise_rate"] = probe_metadata.get("surprise_rate")
        else:
            history_record["esm"] = None
            history_record["coverage_75"] = None
            history_record["surprise_rate"] = None

        history_records.append(history_record)

        if verbose and (run_idx + 1) % 20 == 0:
            ma20 = sum(recent_scores) / len(recent_scores)
            print(f"[{trajectory_name}] {run_idx+1:4d}/{n_runs}  best={best_found:.6f}  regret={simple_regret:.6f}  ma20={ma20:.6f}", flush=True)

    if verbose:
        print(f"  ✓ {trajectory_name}: {len(history_records)} runs", flush=True)

    return pl.DataFrame(history_records)
