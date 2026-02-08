import os
import sys
import random
import wandb
import argparse
import numpy as np
import tensorflow as tf
import keras
import pprint

from wandb.integration.keras import WandbMetricsLogger

from src.constant import PROJECT_NAME
from src.utils import filter_vocabulary, load_yaml, store_yaml, load_config
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
    wandb.init(project=PROJECT_NAME, config=config if config else None)
    config = dict(wandb.config)
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

    wandb.finish()

    print(f"{'='*10} Final Results {'='*24}")
    pprint.pprint(results)
    print(f"{'='*48}")

    # Save the model
    if config["store_model"]:
        sweep_id = wandb.run.sweep_id if wandb.run.sweep_id else "single_runs"
        base_path = f"models/{config['model']}/{sweep_id}/{wandb.run.id}"
        os.makedirs(base_path, exist_ok=True)

        model.save(os.path.join(base_path, "model.keras"))
        store_yaml(config, os.path.join(base_path, "config.yaml"))
        print(f"Model saved to {os.path.join(base_path, 'model.keras')}")

def compile_config(args):
    # Load Default Config, priority 3
    config = load_config("configs/default.yaml")

    # Load Config File, priority 2
    loaded_config = load_config(args.config, method=args.method) if args.config is not None else {}
    config.update(loaded_config)

    # Override with CLI Arguments, priority 1
    for key, value in vars(args).items():
        if key in ["nworker", "nruns", "sweep_id", "method"]:
            continue

        if (value is not None and not isinstance(value, bool)) or (isinstance(value, bool) and value == True):
            config[key] = value

    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Config file
    parser.add_argument("--config", type=str, default=None, help="Path to the YAML configuration file.")
    parser.add_argument("--method", type=str, default="random", help="How to collapse the config into single run config, Literal[random, exhaustive].")
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

    args = parser.parse_args()
    config = compile_config(args)

    main(**config)