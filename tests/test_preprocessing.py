import pytest
import pandas as pd

from src.preprocessing import construct_features_meta, FeatureMeta
from src.preprocessing.data_loader import load_data

class TestLoadData:
    """Tests for ``load_data``."""

    def test_returns_dataframe(self, sample_interaction_file: str) -> None:
        interaction_data: pd.DataFrame = load_data(sample_interaction_file)
        assert isinstance(interaction_data, pd.DataFrame)

    def test_correct_columns(self, sample_interaction_file: str) -> None:
        interaction_data: pd.DataFrame = load_data(sample_interaction_file)
        assert list(interaction_data.columns) == ["user_id", "item_id"]

    def test_correct_row_count(self, sample_interaction_file: str) -> None:
        """File has 3+2+3+2+3+2+2+2+2+2 = 23 interactions."""
        interaction_data: pd.DataFrame = load_data(sample_interaction_file)
        assert len(interaction_data) == 23

    def test_correct_user_ids(self, sample_interaction_file: str) -> None:
        interaction_data: pd.DataFrame = load_data(sample_interaction_file)
        assert set(interaction_data["user_id"].unique()) == set(range(10))

    def test_correct_item_ids(self, sample_interaction_file: str) -> None:
        interaction_data: pd.DataFrame = load_data(sample_interaction_file)
        expected: set[int] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28}
        assert set(interaction_data["item_id"].unique()) == expected

    def test_user_item_pairs(self, sample_interaction_file: str) -> None:
        interaction_data: pd.DataFrame = load_data(sample_interaction_file)
        user_0_items: set[int] = set(interaction_data[interaction_data["user_id"] == 0]["item_id"])
        assert user_0_items == {10, 11, 12}

    def test_empty_user_excluded(self, sample_interaction_file_empty_user: str) -> None:
        """User 1 has no items → no rows for user 1."""
        interaction_data: pd.DataFrame = load_data(sample_interaction_file_empty_user)
        assert 1 not in interaction_data["user_id"].values
        # Other users should still be present
        assert 0 in interaction_data["user_id"].values
        assert 2 in interaction_data["user_id"].values

class TestConstructFeaturesMeta:
    """Tests for ``construct_features_meta``."""

    def test_returns_dict(self, sample_dataframe: pd.DataFrame) -> None:
        meta: FeatureMeta = construct_features_meta(sample_dataframe)
        assert isinstance(meta, dict)

    def test_has_user_and_item_keys(self, sample_dataframe: pd.DataFrame) -> None:
        meta: FeatureMeta = construct_features_meta(sample_dataframe)
        assert "user_id" in meta
        assert "item_id" in meta

    def test_dtype_field(self, sample_dataframe: pd.DataFrame) -> None:
        meta: FeatureMeta = construct_features_meta(sample_dataframe)
        assert meta["user_id"]["dtype"] == "int64"
        assert meta["item_id"]["dtype"] == "int64"

    def test_unique_count(self, sample_dataframe: pd.DataFrame) -> None:
        meta: FeatureMeta = construct_features_meta(sample_dataframe)
        assert meta["user_id"]["unique_count"] == 10  # users 0-9
        assert meta["item_id"]["unique_count"] == 19  # 19 unique items

    def test_vocabulary_contents(self, sample_dataframe: pd.DataFrame) -> None:
        meta: FeatureMeta = construct_features_meta(sample_dataframe)
        assert set(meta["user_id"]["vocabulary"]) == set(range(10))
        expected_items: set[int] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28}
        assert set(meta["item_id"]["vocabulary"]) == expected_items

    def test_vocabulary_length_matches_unique_count(self, sample_dataframe: pd.DataFrame) -> None:
        meta: FeatureMeta = construct_features_meta(sample_dataframe)
        for feature in ["user_id", "item_id"]:
            assert len(meta[feature]["vocabulary"]) == meta[feature]["unique_count"]
