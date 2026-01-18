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
from src.utils import filter_vocabulary
from src.preprocessing import construct_features_meta
from src.preprocessing.data_loader import load_data
from src.sampler import BayesianSampler
from src.losses.bayesian_personalized_ranking import BayesianPersonalizedRankingLoss
from src.models.matrix_factorization import MatrixFactorization

def main(**config):
    # Initialize Run Configurations
    wandb.init(project=PROJECT_NAME, config=config if config else None)
    config = dict(wandb.config)
    print(f"{'='*10} Run Configs {'='*20}")
    pprint.pprint(config)
    print(f"{'='*48}")

    # Set random seeds for reproducibility
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    tf.random.set_seed(config["random_seed"])

    # Create base folder for this run
    sweep_id = wandb.run.sweep_id if wandb.run.sweep_id else "single_runs"
    base_path = f"models/{config['model']}/{sweep_id}/{wandb.run.id}"
    os.makedirs(base_path, exist_ok=True)

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
    print(f"{'='*10} Dataset Summary {'='*20}")
    print(f"Training Dataset: {len(train_dataset)}")
    print(f"Test Dataset: {len(test_dataset)}")
    pprint.pprint(filter_vocabulary(train_features_meta))
    print(f"{'='*48}")


    # B. Model Initialization
    sampler = BayesianSampler(item_set=train_features_meta["item_id"]["vocabulary"], user_items=user_items)

    if config["model"] == "matrix_factorization":
        model = MatrixFactorization(
            train_features_meta, 
            embedding_dimension_count=64,
            l2_regularization=1e-6
        )
    else:
        raise ValueError(f"Unknown model type: {config['model']}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss_functions=[
            BayesianPersonalizedRankingLoss()
        ],
        sampler=sampler,
        evaluation_cutoffs=[2, 10, 50]
    )

    # C. Model Training
    callbacks = []
    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor="test_recall@10",
            mode="max",
            patience=0,
            restore_best_weights=True,
            verbose=1
        )
    )
    callbacks.append(
        WandbMetricsLogger(
            log_freq="epoch",
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

    print(f"{'='*10} Final Results {'='*10}")
    pprint.pprint(results)
    print(f"{'='*30}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="matrix_factorization")
    parser.add_argument("--max_epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--random_seed", type=int, default=42)


    args = parser.parse_args()
    config = vars(args)

    main(**config)