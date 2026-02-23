import argparse
import pytest
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from typing import Any
from pprint import pprint
from wandb.apis.public import Run

from main import compile_config, main
from src.constant import PROJECT_NAME
from src.preprocessing import FeatureMeta
from src.models.matrix_factorization import MatrixFactorization

class TestConfigCompilation:
    """Tests for ``compile_config`` in ``main.py``."""

    def args(self, **overrides: Any) -> argparse.Namespace:
        """Build a minimal argparse.Namespace mimicking CLI parsing."""

        defaults: dict[str, Any] = dict(
            config=None,
            method="random",
            model=None,
            embedding_dimension=None,
            max_epoch=None,
            batch_size=None,
            shuffle=False,
            learning_rate=None,
            l1_regularization=None,
            l2_regularization=None,
            embedding_dropout_rate=None,
            log_freq=None,
            evaluation_cutoffs=None,
            early_stopping=False,
            early_stopping_monitor=None,
            early_stopping_mode=None,
            early_stopping_patience=None,
            random_seed=None,
            store_model=False,
            tracker=None,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_default_config_loaded(self) -> None:
        """With no CLI overrides, compile_config returns the default config."""
        args: argparse.Namespace = self.args()
        config: dict[str, Any] = compile_config(args)
        assert "model" in config
        assert "random_seed" in config
        # Values should come from configs/default.yaml
        assert config["model"] == "matrix_factorization"

    def test_cli_override_takes_priority(self) -> None:
        """CLI arguments override default config values."""
        args: argparse.Namespace = self.args(learning_rate=0.05, max_epoch=100)
        config: dict[str, Any] = compile_config(args)
        assert config["learning_rate"] == 0.05
        assert config["max_epoch"] == 100

    def test_boolean_flag_override(self) -> None:
        """Boolean flags like --shuffle override when True."""
        args: argparse.Namespace = self.args(shuffle=True)
        config: dict[str, Any] = compile_config(args)
        assert config["shuffle"] is True

    def test_boolean_flag_false_does_not_override(self) -> None:
        """Boolean flags with False should not override the default config value."""
        args: argparse.Namespace = self.args(shuffle=False)
        config: dict[str, Any] = compile_config(args)
        # Should use default from YAML (which is false)
        assert config["shuffle"] is False or config["shuffle"] == False

    def test_none_values_do_not_override(self) -> None:
        """None CLI values should not override defaults."""
        args: argparse.Namespace = self.args(model=None)
        config: dict[str, Any] = compile_config(args)
        assert config["model"] == "matrix_factorization"

    def test_config_file_override(self, sample_yaml_config: str) -> None:
        """Config file values override default config."""
        args: argparse.Namespace = self.args(config=sample_yaml_config)
        config: dict[str, Any] = compile_config(args)
        assert config["embedding_dimension"] == 16
        assert config["max_epoch"] == 2

    def test_cli_overrides_config_file(self, sample_yaml_config: str) -> None:
        """CLI has highest priority, overriding both default and config file."""
        args: argparse.Namespace = self.args(config=sample_yaml_config, max_epoch=999)
        config: dict[str, Any] = compile_config(args)
        assert config["max_epoch"] == 999

    def test_evaluation_cutoffs_list(self) -> None:
        """List-type CLI arguments like --evaluation_cutoffs are handled."""
        args: argparse.Namespace = self.args(evaluation_cutoffs=[5, 20])
        config: dict[str, Any] = compile_config(args)
        assert config["evaluation_cutoffs"] == [5, 20]

    def test_random_seed_override(self) -> None:
        args: argparse.Namespace = self.args(random_seed=123)
        config: dict[str, Any] = compile_config(args)
        assert config["random_seed"] == 123

    def test_store_model_flag(self) -> None:
        args: argparse.Namespace = self.args(store_model=True)
        config: dict[str, Any] = compile_config(args)
        assert config["store_model"] is True

class TestTrainingDeterminism:
    """Tests to ensure that the model behaves deterministically with a fixed random seed and parameters."""

    def config(self, **overrides: Any) -> dict[str, Any]:
        """Helper to build a config dict with defaults and overrides."""

        defaults: dict[str, Any] = dict(
            config=None,
            method="random",
            model=None,
            embedding_dimension=None,
            max_epoch=None,
            batch_size=None,
            shuffle=False,
            learning_rate=None,
            l1_regularization=None,
            l2_regularization=None,
            embedding_dropout_rate=None,
            log_freq=None,
            evaluation_cutoffs=None,
            early_stopping=False,
            early_stopping_monitor=None,
            early_stopping_mode=None,
            early_stopping_patience=None,
            random_seed=None,
            store_model=False,
            tracker=None,
        )
        defaults.update(overrides)
        args = argparse.Namespace(**defaults)

        return compile_config(args)
    
    def test_model_user_initialization(self, sample_features_meta: FeatureMeta) -> None:
        """With the same random seed and parameters, model initialization should be the same."""
        config1: dict[str, Any] = self.config(random_seed=42, embedding_dimension=8)
        config2: dict[str, Any] = self.config(random_seed=42, embedding_dimension=8)

        # Create two models with the same config
        random.seed(config1["random_seed"])
        np.random.seed(config1["random_seed"])
        tf.random.set_seed(config1["random_seed"])   
        model1: MatrixFactorization = MatrixFactorization(features_meta=sample_features_meta, embedding_dimension_count=config1["embedding_dimension"])
        random.seed(config2["random_seed"])
        np.random.seed(config2["random_seed"])
        tf.random.set_seed(config2["random_seed"])   
        model2: MatrixFactorization = MatrixFactorization(features_meta=sample_features_meta, embedding_dimension_count=config2["embedding_dimension"])

        # Check that the initial embeddings are the same
        vocab: list[int] = sample_features_meta["user_id"]["vocabulary"]
        user_embedding1: tf.Tensor = model1.user_embedding(tf.constant(vocab[:3]))
        user_embedding2: tf.Tensor = model2.user_embedding(tf.constant(vocab[:3]))

        assert np.allclose(np.array(user_embedding1), np.array(user_embedding2))

    def test_model_item_initialization(self, sample_features_meta: FeatureMeta) -> None:
        """With the same random seed and parameters, item embeddings should be the same."""
        config1: dict[str, Any] = self.config(random_seed=42, embedding_dimension=8)
        config2: dict[str, Any] = self.config(random_seed=42, embedding_dimension=8)

        # Create two models with the same config
        random.seed(config1["random_seed"])
        np.random.seed(config1["random_seed"])
        tf.random.set_seed(config1["random_seed"])
        model1: MatrixFactorization = MatrixFactorization(features_meta=sample_features_meta, embedding_dimension_count=config1["embedding_dimension"])
        random.seed(config2["random_seed"])
        np.random.seed(config2["random_seed"])
        tf.random.set_seed(config2["random_seed"])
        model2: MatrixFactorization = MatrixFactorization(features_meta=sample_features_meta, embedding_dimension_count=config2["embedding_dimension"])

        # Check that the initial item embeddings are the same
        vocab: list[int] = sample_features_meta["item_id"]["vocabulary"]
        item_embedding1: tf.Tensor = model1.item_embedding(tf.constant(vocab[:3]))
        item_embedding2: tf.Tensor = model2.item_embedding(tf.constant(vocab[:3]))

        assert np.allclose(np.array(item_embedding1), np.array(item_embedding2))

    def test_different_seeds(self, sample_features_meta: FeatureMeta) -> None:
        """Different random seeds should lead to different initializations."""
        config1: dict[str, Any] = self.config(random_seed=42, embedding_dimension=8)
        config2: dict[str, Any] = self.config(random_seed=43, embedding_dimension=8)

        random.seed(config1["random_seed"])
        np.random.seed(config1["random_seed"])
        tf.random.set_seed(config1["random_seed"])
        model1: MatrixFactorization = MatrixFactorization(features_meta=sample_features_meta, embedding_dimension_count=config1["embedding_dimension"])
        random.seed(config2["random_seed"])
        np.random.seed(config2["random_seed"])
        tf.random.set_seed(config2["random_seed"])
        model2: MatrixFactorization = MatrixFactorization(features_meta=sample_features_meta, embedding_dimension_count=config2["embedding_dimension"])

        vocab: list[int] = sample_features_meta["user_id"]["vocabulary"]
        user_embedding1: tf.Tensor = model1.user_embedding(tf.constant(vocab[:3]))
        user_embedding2: tf.Tensor = model2.user_embedding(tf.constant(vocab[:3]))

        assert not np.allclose(np.array(user_embedding1), np.array(user_embedding2))

        vocab: list[int] = sample_features_meta["item_id"]["vocabulary"]
        item_embedding1: tf.Tensor = model1.item_embedding(tf.constant(vocab[:3]))
        item_embedding2: tf.Tensor = model2.item_embedding(tf.constant(vocab[:3]))

        assert not np.allclose(np.array(item_embedding1), np.array(item_embedding2))

    def test_entire_run_determinism(self):
        """
        With the same config and random seed, the entire training run should be deterministic.
        This is a more complex test that would involve running the main function twice with the same config
        and checking that the final model weights or evaluation metrics are the same.
        """
        
        config = self.config(
            embedding_dimension=64, 
            max_epoch=8, 
            learning_rate=0.01, 
            tracker="disabled",
            random_seed=42, 
            store_model=False, 
        )

        results1 = main(**config)
        results2 = main(**config)

        assert results1["final_metrics"] == pytest.approx(results2["final_metrics"], rel=1e-4)

    def test_different_runs_different_results(self):
        """Different parameters lead to different final metrics."""
        config1 = self.config(
            embedding_dimension=64, 
            max_epoch=8, 
            learning_rate=0.01,
            tracker="disabled",
            random_seed=42, 
            store_model=False, 
        )
        config2 = self.config(
            embedding_dimension=64, 
            max_epoch=8, 
            learning_rate=0.01, 
            l1_regularization=1e-7,
            l2_regularization=1e-7,
            tracker="disabled",
            random_seed=42, 
            store_model=False, 
        )

        results1 = main(**config1)
        results2 = main(**config2)

        assert results1["final_metrics"] != pytest.approx(results2["final_metrics"], rel=1e-4)

class TestTrainingReplication:
    """
    Given a run name, we fetch the config and final metrics from wandb and check that running with that config locally reproduces the same metrics (within some tolerance).
    """
    ignore_fields = ["_runtime", "_timestamp", "_step", "_wandb"]
    tested_columns = [
        "epoch/epoch", "epoch/train_loss", "epoch/test_loss",
        "epoch/train_recall@10", "epoch/test_recall@10",
        "epoch/train_recall@20", "epoch/test_recall@20",
        "epoch/train_recall@50", "epoch/test_recall@50",
        "epoch/train_precision@10", "epoch/test_precision@10",
        "epoch/train_precision@20", "epoch/test_precision@20",
        "epoch/train_precision@50", "epoch/test_precision@50",
        "epoch/train_mrr@10", "epoch/test_mrr@10",
        "epoch/train_mrr@20", "epoch/test_mrr@20",
        "epoch/train_mrr@50", "epoch/test_mrr@50",
        "epoch/train_map@10", "epoch/test_map@10",
        "epoch/train_map@20", "epoch/test_map@20",
        "epoch/train_map@50", "epoch/test_map@50",
        "epoch/train_ndcg@10", "epoch/test_ndcg@10",
        "epoch/train_ndcg@20", "epoch/test_ndcg@20",
        "epoch/train_ndcg@50", "epoch/test_ndcg@50",
    ]

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.target_run_id: str = "px3wtuxs"

    def test_replicate_run(self) -> None:
        import wandb
        
        api = wandb.Api()
        target_run: Run = api.run(f"{api.default_entity}/{PROJECT_NAME}/{self.target_run_id}")
        target_config = target_run.config
        target_summary = {key: value for key, value in target_run.summary.items() if key not in self.ignore_fields}

        # Run the main function with the fetched config
        results = main(**target_config)
        result_run_id = results["run_id"]
        result_run: Run = api.run(f"{api.default_entity}/{PROJECT_NAME}/{result_run_id}")
        result_summary = {key: value for key, value in result_run.summary.items() if key not in self.ignore_fields}

        pprint(result_summary)

        # Check that the final metrics match (within some tolerance)
        assert target_summary == pytest.approx(result_summary, rel=5e-3)
        pd.testing.assert_frame_equal(
            target_run.history()[self.tested_columns], result_run.history()[self.tested_columns], rtol=5e-3
        )

class TestTrainingReplicationCase1(TestTrainingReplication):
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.target_run_id: str = "ppnmplwp"
