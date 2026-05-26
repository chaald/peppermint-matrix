import os
import csv
import sys
import random
import wandb
import argparse
import numpy as np
import polars as pl
import tensorflow as tf
import keras
import pprint

from wandb.integration.keras import WandbMetricsLogger

from src.constant import PROJECT_NAME
from src.utils import filter_vocabulary, load_yaml, store_yaml, load_config
from src.utils.config import fetch_run_metadata, write_decision_log
from src.preprocessing import construct_features_meta
from src.preprocessing.data_loader import load_data
from src.sampler import BayesianSampler
from src.losses.bayesian_personalized_ranking import BayesianPersonalizedRankingLoss
from src.models.matrix_factorization import MatrixFactorization

# Set CuDNN library path for TensorFlow
# LD_LIBRARY_PATH=$(uv run python -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])")/lib:$LD_LIBRARY_PATH

def main(**config):
    # Initialize Run Configurations
    if config and "log_freq" in config:
        config["log_freq"] = int(config["log_freq"]) if str(config["log_freq"]).isdigit() else config["log_freq"]
    
    # Initialize Trackers
    if config["tracker"] != "disabled":
        wandb.init(project=PROJECT_NAME, config=config if config else None)
        config = dict(wandb.config)

        sweep_id = wandb.run.sweep_id if wandb.run.sweep_id else "single_runs"
        run_id = wandb.run.id
        run_name = wandb.run.name
    else:
        sweep_id = "single_runs"
        run_id = "local"
        run_name = "local"

    # Print the final config for this run
    print(f"{'='*10} Run Configs {'='*25}")
    pprint.pprint(config)
    print(f"{'='*48}")

    # Set random seeds for reproducibility
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    tf.random.set_seed(config["random_seed"])    

    # A. Load and preprocess data
    train_user_interaction = load_data("dataset/yelp2018/train.txt")
    train_features_meta = construct_features_meta(train_user_interaction)
    test_user_interaction = load_data("dataset/yelp2018/test.txt")
    test_features_meta = construct_features_meta(test_user_interaction)

    user_items = train_user_interaction.groupby("user_id")["item_id"].apply(set).to_dict()
    item_users = train_user_interaction.groupby("item_id")["user_id"].apply(set).to_dict()

    train_dataset = tf.data.Dataset.from_tensor_slices(
    {
            "user_id": train_user_interaction["user_id"].values,
            "item_id": train_user_interaction["item_id"].values
        }
    )
    test_dataset = tf.data.Dataset.from_tensor_slices(
        {
            "user_id": test_user_interaction["user_id"].values,
            "item_id": test_user_interaction["item_id"].values
        }
    )
    print(f"{'='*10} Dataset Summary {'='*21}")
    print(f"Training Dataset: {len(train_dataset)}")
    print(f"Test Dataset: {len(test_dataset)}")
    pprint.pprint(filter_vocabulary(train_features_meta))
    print(f"{'='*48}")

    # B. Model Initialization
    sampler = BayesianSampler(item_set=train_features_meta["item_id"]["vocabulary"], user_items=user_items)

    if config["model"] == "matrix_factorization":
        model = MatrixFactorization(
            train_features_meta, 
            embedding_dimension_count=config["embedding_dimension"],
            l1_regularization=config["l1_regularization"],
            l2_regularization=config["l2_regularization"],
            embedding_dropout_rate=config["embedding_dropout_rate"],
            evaluation_cutoffs=config["evaluation_cutoffs"]
        )
    else:
        raise ValueError(f"Unknown model type: {config['model']}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss_functions=[
            BayesianPersonalizedRankingLoss()
        ],
        sampler=sampler,
    )

    # C. Model Training
    callbacks = []
    if config["early_stopping"]:
        callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor=config["early_stopping_monitor"],
                    mode=config["early_stopping_mode"],
                    patience=config["early_stopping_patience"],
                    restore_best_weights=True,
                    verbose=1
            )
        )
    if config["tracker"] != "disabled":
        callbacks.append(
            WandbMetricsLogger(
                log_freq=config["log_freq"],
            )
        )

    results = model.fit(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        nepochs=config["max_epoch"],
        shuffle=config["shuffle"],
        batch_size=config["batch_size"],
        callbacks=callbacks
    )

    if config["tracker"] != "disabled":
        wandb.finish()

        # Fetch full run metadata from W&B and append to summary parquet
        tracker_api = wandb.Api()
        try:
            current_run = fetch_run_metadata(tracker_api, run_id, max_retries=3)
        except Exception:
            print("WARNING: could not fetch run from W&B — skipping parquet append")
            current_run = None

        if current_run is not None and os.path.exists(config["summary_path"]):
            existing = pl.read_parquet(config["summary_path"])
            current_run = pl.DataFrame([current_run])
            for col in current_run.columns:
                if col in existing.schema:
                    current_run = current_run.with_columns(
                        pl.col(col).cast(existing.schema[col])
                    )
            combined = pl.concat([existing, current_run], how="diagonal")
            combined.write_parquet(config["summary_path"])

    print(f"{'='*10} Final Results {'='*24}")
    pprint.pprint(results)
    print(f"{'='*48}")

    # Save the model
    if config["store_model"]:
        base_path = f"models/{config['model']}/{sweep_id}/{run_id}"
        os.makedirs(base_path, exist_ok=True)

        model.save(os.path.join(base_path, "model.keras"))
        store_yaml(config, os.path.join(base_path, "config.yaml"))
        print(f"Model saved to {os.path.join(base_path, 'model.keras')}")

    return {
        "run_id": run_id,
        "run_name": run_name,
        "sweep_id": sweep_id,
        "config": config,
        "final_metrics": results,
    }

def compile_config(args):
    # Load Default Config, priority 3
    config, _ = load_config("configs/default.yaml")

    # Extract model-based kwargs if applicable
    model_based_kwargs = {}
    if args.method == "model_based":
        for key, value in vars(args).items():
            if not key.startswith("model_based_"):
                continue
            if value is not None:
                model_based_kwargs[key.removeprefix("model_based_")] = value
                
        summary_path = getattr(args, "summary_path", None)
        if summary_path is not None:
            model_based_kwargs["summary_path"] = summary_path

    # Load Config File, priority 2
    if args.config is not None:
        loaded_config, decision_metadata = load_config(args.config, method=args.method, **model_based_kwargs)
        config.update(loaded_config)
    else:
        decision_metadata = {}

    # Override with CLI Arguments, priority 1
    for key, value in vars(args).items():
        if key.startswith("model_based_") or key in ["nworker", "nruns", "sweep_id", "method"]:
            continue

        if (value is not None and not isinstance(value, bool)) or (isinstance(value, bool) and value == True):
            config[key] = value

    return config, decision_metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Config file
    parser.add_argument("--config", type=str, default=None, help="Path to the YAML configuration file.")
    parser.add_argument("--method", type=str, default="random", help="How to collapse the config into single run config, Literal[random, exhaustive, model_based].")
    # Model configuration
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--embedding_dimension", type=int, default=None)
    # Training configurations
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--l1_regularization", type=float, default=None)
    parser.add_argument("--l2_regularization", type=float, default=None)
    parser.add_argument("--embedding_dropout_rate", type=float, default=None)
    # Tracking configurations
    parser.add_argument("--log_freq", type=str, default=None)
    parser.add_argument("--evaluation_cutoffs", type=int, nargs="+", default=None)
    # Early stopping configurations
    parser.add_argument("--early_stopping", action="store_true", default=False)
    parser.add_argument("--early_stopping_monitor", type=str, default=None)
    parser.add_argument("--early_stopping_mode", type=str, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    # Utilities
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--store_model", action="store_true", default=False)
    parser.add_argument("--tracker", type=str, default=None, help="Tracking backend. Use 'disabled' to skip wandb entirely.")
    parser.add_argument("--summary_path", type=str, default=None, help="Path to the summary parquet file for surrogate training. Overrides configs/default.yaml.")
    parser.add_argument("--model_based_beta", type=float, default=None, help="Exploration weight for UCB formula. Overrides configs/default.yaml.")
    parser.add_argument("--model_based_target", type=str, default=None, help="Target metric key in parquet. Overrides configs/default.yaml.")
    parser.add_argument("--model_based_estimator_count", type=int, default=None, help="Number of trees in Random Forest surrogate. Overrides configs/default.yaml.")
    parser.add_argument("--model_based_virtual_sample_count", type=int, default=None, help="Number of optimistic virtual samples to inject. Only for --method=model_based.")
    parser.add_argument("--model_based_virtual_lambda", type=float, default=None, help="Lambda multiplier for virtual sample target = max(best, mean + lambda * std). Only for --method=model_based.")
    parser.add_argument("--model_based_log_path", type=str, default=None, help="Path for the hyperparameter search decision log. Only for --method=model_based.")

    args = parser.parse_args()
    config, decision_metadata = compile_config(args)

    report = main(**config)

    if decision_metadata:
        ordered_decision_metadata = {
            "run_id": report["run_id"],
            "run_name": report["run_name"],
            "sweep_id": report["sweep_id"],
        }
        ordered_decision_metadata.update(decision_metadata)
        log_path = config.get("log_path", "hyperparameter_search.log.csv")
        write_decision_log(ordered_decision_metadata, log_path)