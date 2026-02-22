import pytest
import numpy as np
import tensorflow as tf

from src.sampler import BayesianSampler
from src.preprocessing import FeatureMeta


class TestBayesianSampler:
    """Tests for ``BayesianSampler``."""

    def test_output_shape_matches_input(self, sample_sampler: BayesianSampler) -> None:
        user_ids: tf.Tensor = tf.constant([0, 1, 2])
        result: tf.Tensor = sample_sampler.sample(user_ids)
        assert result.shape == (3,)

    def test_sampled_items_are_from_item_set(self, sample_sampler: BayesianSampler, sample_features_meta: FeatureMeta) -> None:
        user_ids: tf.Tensor = tf.constant([0, 0, 1, 1, 2, 2])
        result: tf.Tensor = sample_sampler.sample(user_ids)
        valid_items: set[int] = set(sample_features_meta["item_id"]["vocabulary"])
        for item in result.numpy():
            assert item in valid_items

    def test_negative_samples_not_in_positive_set(self, sample_sampler: BayesianSampler, sample_user_items: dict[int, set[int]]) -> None:
        """Sampled negatives must not be in the user's positive interaction set."""
        user_ids: tf.Tensor = tf.constant([0, 0, 1, 1, 2, 2, 3, 4, 5])
        # Run multiple times to increase confidence
        for _ in range(10):
            result: tf.Tensor = sample_sampler.sample(user_ids)
            for uid_tensor, neg_item in zip(user_ids, result.numpy()):
                uid: int = int(uid_tensor.numpy())
                assert neg_item not in sample_user_items[uid], (
                    f"Negative item {neg_item} found in positives for user {uid}"
                )

    def test_returns_tensor(self, sample_sampler: BayesianSampler) -> None:
        user_ids: tf.Tensor = tf.constant([0, 1])
        result: tf.Tensor = sample_sampler.sample(user_ids)
        assert isinstance(result, tf.Tensor)

    def test_single_user(self, sample_sampler: BayesianSampler, sample_user_items: dict[int, set[int]]) -> None:
        user_ids: tf.Tensor = tf.constant([0])
        result: tf.Tensor = sample_sampler.sample(user_ids)
        assert result.shape == (1,)
        assert int(result.numpy()[0]) not in sample_user_items[0]

    def test_large_batch(self, sample_sampler: BayesianSampler, sample_user_items: dict[int, set[int]]) -> None:
        """Sampling works with a larger batch without errors."""
        user_ids: tf.Tensor = tf.constant([0] * 50 + [1] * 50 + [2] * 50 + [3] * 50 + [4] * 50 + [5] * 50)
        result: tf.Tensor = sample_sampler.sample(user_ids)
        assert result.shape == (300,)
        for uid_tensor, neg_item in zip(user_ids, result.numpy()):
            uid: int = int(uid_tensor.numpy())
            assert neg_item not in sample_user_items[uid]
