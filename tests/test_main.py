import argparse
import pytest

from typing import Any

from main import compile_config


class TestCompileConfig:
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
