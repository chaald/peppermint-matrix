import os
import pathlib
import pytest
import tempfile
import numpy as np
import pandas as pd
import tensorflow as tf

from src.preprocessing import construct_features_meta, FeatureMeta
from src.sampler import BayesianSampler
from src.models.matrix_factorization import MatrixFactorization


@pytest.fixture
def sample_interaction_file(tmp_path: pathlib.Path) -> str:
    """Create a minimal interaction file in the project's data format.

    Format: each line is ``user_id item1 item2 ...``
    """
    content = (
        "0 10 11 12\n"
        "1 11 13\n"
        "2 10 14 15\n"
        "3 16 17\n"
        "4 18 19 20\n"
        "5 12 21\n"
        "6 22 23\n"
        "7 24 25\n"
        "8 13 26\n"
        "9 27 28\n"
    )
    filepath = tmp_path / "interactions.txt"
    filepath.write_text(content)
    return str(filepath)

@pytest.fixture
def sample_interaction_file_empty_user(tmp_path: pathlib.Path) -> str:
    """Interaction file where one user has no items."""
    content = (
        "0 10 11\n"
        "1\n"
        "2 12\n"
        "3 13 14\n"
        "4 15 16\n"
        "5 17 18\n"
        "6 19 20\n"
        "7 21 22\n"
        "8 23 24\n"
        "9 25 26\n"
    )
    filepath = tmp_path / "interactions_empty.txt"
    filepath.write_text(content)
    return str(filepath)

@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """A user-item interaction DataFrame with >=8 unique users and items."""
    return pd.DataFrame({
        "user_id": [0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9],
        "item_id": [10, 11, 12, 11, 13, 10, 14, 15, 16, 17, 18, 19, 20, 12, 21, 22, 23, 24, 25, 13, 26, 27, 28],
    })

@pytest.fixture
def sample_features_meta(sample_dataframe: pd.DataFrame) -> FeatureMeta:
    """Features meta built from ``sample_dataframe``."""
    return construct_features_meta(sample_dataframe)

@pytest.fixture
def sample_user_items() -> dict[int, set[int]]:
    """User→positive-items mapping."""
    return {
        0: {10, 11, 12},
        1: {11, 13},
        2: {10, 14, 15},
        3: {16, 17},
        4: {18, 19, 20},
        5: {12, 21},
        6: {22, 23},
        7: {24, 25},
        8: {13, 26},
        9: {27, 28},
    }

@pytest.fixture
def sample_sampler(sample_features_meta: FeatureMeta, sample_user_items: dict[int, set[int]]) -> BayesianSampler:
    return BayesianSampler(
        item_set=sample_features_meta["item_id"]["vocabulary"],
        user_items=sample_user_items,
    )

@pytest.fixture
def sample_model(sample_features_meta: FeatureMeta) -> MatrixFactorization:
    """A small MatrixFactorization model for unit tests."""
    from src.losses.bayesian_personalized_ranking import BayesianPersonalizedRankingLoss

    model = MatrixFactorization(
        features_meta=sample_features_meta,
        embedding_dimension_count=8,
        l1_regularization=0.0,
        l2_regularization=0.0,
        embedding_dropout_rate=0.0,
        evaluation_cutoffs=[2, 5],
    )
    return model

@pytest.fixture
def compiled_model(sample_model: MatrixFactorization, sample_sampler: BayesianSampler) -> MatrixFactorization:
    """A compiled MatrixFactorization model (with optimizer, loss, sampler)."""
    import keras
    from src.losses.bayesian_personalized_ranking import BayesianPersonalizedRankingLoss

    sample_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss_functions=[BayesianPersonalizedRankingLoss()],
        sampler=sample_sampler,
    )
    return sample_model

@pytest.fixture
def sample_train_dataset(sample_dataframe: pd.DataFrame) -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices({
        "user_id": sample_dataframe["user_id"].values,
        "item_id": sample_dataframe["item_id"].values,
    })

@pytest.fixture
def sample_yaml_config(tmp_path: pathlib.Path) -> str:
    """Create a minimal YAML config file and return its path."""
    import yaml

    config = {
        "model": "matrix_factorization",
        "embedding_dimension": 16,
        "max_epoch": 2,
        "batch_size": 4,
        "learning_rate": 0.01,
        "random_seed": 42,
    }
    filepath = tmp_path / "test_config.yaml"
    with open(filepath, "w") as f:
        yaml.dump(config, f)
    return str(filepath)

@pytest.fixture
def sample_yaml_parameters_config(tmp_path: pathlib.Path) -> str:
    """Create a YAML config with ``parameters`` block (hyperparameter search format)."""
    import yaml

    config = {
        "parameters": {
            "model": {"value": "matrix_factorization"},
            "embedding_dimension": {"distribution": "constant", "value": 16},
            "learning_rate": {"distribution": "categorical", "values": [0.01, 0.001]},
            "batch_size": {"distribution": "int_uniform", "min": 32, "max": 64},
            "l2_regularization": {"distribution": "uniform", "min": 0.0, "max": 0.1},
        }
    }
    filepath = tmp_path / "test_params_config.yaml"
    with open(filepath, "w") as f:
        yaml.dump(config, f)
    return str(filepath)