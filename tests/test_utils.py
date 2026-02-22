import os
import json
import math
import pytest
import pathlib
import tempfile
import numpy as np

from typing import Union
from src.preprocessing import FeatureMeta

from src.utils import (
    filter_vocabulary,
    preprocess_metric_aggregate,
    load_yaml,
    store_yaml,
)
from src.utils.config import (
    store_json,
    load_json,
    parse_scientific_notation,
    parse_parameters,
    load_config,
)

class TestParseScientificNotation:
    """Tests for ``parse_scientific_notation``."""

    def test_valid_scientific_string(self) -> None:
        assert parse_scientific_notation("2.5E+2") == pytest.approx(250.0)

    def test_valid_negative_exponent(self) -> None:
        assert parse_scientific_notation("1e-3") == pytest.approx(0.001)

    def test_plain_string_unchanged(self) -> None:
        assert parse_scientific_notation("hello") == "hello"

    def test_plain_number_string_unchanged(self) -> None:
        # "42" does not match scientific notation pattern — returned as-is
        assert parse_scientific_notation("42") == "42"

    def test_non_string_passthrough(self) -> None:
        assert parse_scientific_notation(3.14) == 3.14

    def test_list_of_mixed_values(self) -> None:
        result: list[Union[float, str]] = parse_scientific_notation(["1e-2", "foo", "3.0e1"])
        assert result[0] == pytest.approx(0.01)
        assert result[1] == "foo"
        assert result[2] == pytest.approx(30.0)

class TestParseParameters:
    """Tests for ``parse_parameters``."""

    def test_plain_value(self) -> None:
        config: dict[str, object] = {"model": "mf"}
        result: dict[str, object] = parse_parameters(config)
        assert result["model"] == "mf"

    def test_value_key(self) -> None:
        config: dict[str, dict[str, int]] = {"embedding_dimension": {"value": 16}}
        result: dict[str, object] = parse_parameters(config)
        assert result["embedding_dimension"] == 16

    def test_constant_distribution(self) -> None:
        config: dict[str, dict[str, object]] = {"lr": {"distribution": "constant", "value": 0.01}}
        result: dict[str, object] = parse_parameters(config)
        assert result["lr"] == pytest.approx(0.01)

    def test_categorical_distribution_returns_valid_choice(self) -> None:
        choices: list[float] = [0.01, 0.001, 0.0001]
        config: dict[str, dict[str, object]] = {"lr": {"distribution": "categorical", "values": choices}}
        result: dict[str, object] = parse_parameters(config)
        assert result["lr"] in choices

    def test_int_uniform_distribution_within_range(self) -> None:
        config: dict[str, dict[str, object]] = {"batch_size": {"distribution": "int_uniform", "min": 16, "max": 64}}
        result: dict[str, object] = parse_parameters(config)
        assert 16 <= result["batch_size"] <= 64
        assert isinstance(result["batch_size"], int)

    def test_uniform_distribution_within_range(self) -> None:
        config: dict[str, dict[str, object]] = {"lr": {"distribution": "uniform", "min": 0.0, "max": 1.0}}
        result: dict[str, object] = parse_parameters(config)
        assert 0.0 <= result["lr"] <= 1.0

    def test_log_uniform_distribution_positive(self) -> None:
        config: dict[str, dict[str, object]] = {"lr": {"distribution": "log_uniform", "min": -5, "max": -1}}
        result: dict[str, object] = parse_parameters(config)
        assert result["lr"] > 0
        assert math.exp(-5) <= result["lr"] <= math.exp(-1)

    def test_unsupported_distribution_raises(self) -> None:
        config: dict[str, dict[str, object]] = {"lr": {"distribution": "beta", "alpha": 1, "beta": 1}}
        with pytest.raises(ValueError, match="Unsupported distribution"):
            parse_parameters(config)

    def test_invalid_config_raises(self) -> None:
        config: dict[str, dict[str, int]] = {"lr": {"min": 0, "max": 1}}  # missing "distribution" and "value"
        with pytest.raises(ValueError, match="Invalid parameter configuration"):
            parse_parameters(config)

class TestJsonIO:
    """Tests for ``store_json`` / ``load_json`` roundtrip."""

    def test_roundtrip(self, tmp_path: pathlib.Path) -> None:
        data: dict[str, object] = {"a": 1, "b": [2, 3], "c": "hello"}
        filepath: str = str(tmp_path / "test.json")
        store_json(data, filepath)
        loaded: dict[str, object] = load_json(filepath)
        assert loaded == data

    def test_nested_data(self, tmp_path: pathlib.Path) -> None:
        data: dict[str, dict[str, int]] = {"outer": {"inner": 42}}
        filepath: str = str(tmp_path / "nested.json")
        store_json(data, filepath)
        loaded: dict[str, object] = load_json(filepath)
        assert loaded["outer"]["inner"] == 42

class TestYamlIO:
    """Tests for ``store_yaml`` / ``load_yaml`` roundtrip."""

    def test_roundtrip(self, tmp_path: pathlib.Path) -> None:
        data: dict[str, object] = {"model": "mf", "lr": 0.01, "epochs": 10}
        filepath: str = str(tmp_path / "test.yaml")
        store_yaml(data, filepath)
        loaded: dict[str, object] = load_yaml(filepath)
        assert loaded == data

    def test_list_values(self, tmp_path: pathlib.Path) -> None:
        data: dict[str, list[int]] = {"cutoffs": [2, 10, 50]}
        filepath: str = str(tmp_path / "list.yaml")
        store_yaml(data, filepath)
        loaded: dict[str, object] = load_yaml(filepath)
        assert loaded["cutoffs"] == [2, 10, 50]

class TestFilterVocabulary:
    """Tests for ``filter_vocabulary``."""

    def test_removes_vocabulary_key(self, sample_features_meta: FeatureMeta) -> None:
        filtered: dict[str, dict[str, object]] = filter_vocabulary(sample_features_meta)
        for feature_name, meta in filtered.items():
            assert "vocabulary" not in meta

    def test_preserves_other_keys(self, sample_features_meta: FeatureMeta) -> None:
        filtered: dict[str, dict[str, object]] = filter_vocabulary(sample_features_meta)
        for feature_name in sample_features_meta:
            assert "dtype" in filtered[feature_name]
            assert "unique_count" in filtered[feature_name]

class TestPreprocessMetricAggregate:
    """Tests for ``preprocess_metric_aggregate``."""

    def test_filters_known_keys(self) -> None:
        metrics: dict[str, float] = {
            "loss": 0.123456,
            "test_loss": 0.654321,
            "recall@10": 0.5,
            "test_recall@10": 0.4,
            "ndcg@10": 0.3,  # should be excluded
        }
        result: dict[str, str] = preprocess_metric_aggregate(metrics)
        assert "loss" in result
        assert "test_loss" in result
        assert "recall@10" in result
        assert "test_recall@10" in result
        assert "ndcg@10" not in result

    def test_formats_to_four_decimals(self) -> None:
        metrics: dict[str, float] = {"loss": 0.123456789}
        result: dict[str, str] = preprocess_metric_aggregate(metrics)
        assert result["loss"] == "0.1235"

    def test_empty_input(self) -> None:
        assert preprocess_metric_aggregate({}) == {}

class TestLoadConfig:
    """Tests for ``load_config`` with plain YAML (no parameters block)."""

    def test_loads_plain_yaml(self, sample_yaml_config: str) -> None:
        config: dict[str, object] = load_config(sample_yaml_config)
        assert config["model"] == "matrix_factorization"
        assert config["embedding_dimension"] == 16

    def test_loads_parameters_yaml_random(self, sample_yaml_parameters_config: str) -> None:
        config: dict[str, object] = load_config(sample_yaml_parameters_config, method="random")
        assert config["model"] == "matrix_factorization"
        assert config["embedding_dimension"] == 16
        assert config["learning_rate"] in [0.01, 0.001]
        assert 32 <= config["batch_size"] <= 64
        assert 0.0 <= config["l2_regularization"] <= 0.1
