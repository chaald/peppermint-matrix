import math
import pytest
import numpy as np
import tensorflow as tf

from src.losses.bayesian_personalized_ranking import BayesianPersonalizedRankingLoss


class TestBayesianPersonalizedRankingLoss:
    """Tests for ``BayesianPersonalizedRankingLoss``."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.loss_function: BayesianPersonalizedRankingLoss = BayesianPersonalizedRankingLoss()

    def test_identical_scores_returns_ln2(self) -> None:
        """When pos == neg, sigmoid(0) = 0.5, so loss = -log(0.5) = ln(2)."""
        user_embedding: tf.Tensor = tf.constant([[1.0, 0.0]])
        positive_embedding: tf.Tensor = tf.constant([[1.0, 0.0]])
        negative_embedding: tf.Tensor = tf.constant([[1.0, 0.0]])
        loss: tf.Tensor = self.loss_function(user_embedding, positive_embedding, negative_embedding)
        assert loss.numpy() == pytest.approx(math.log(2), abs=1e-5)

    def test_positive_much_larger_loss_near_zero(self) -> None:
        """When positive score >> negative score, loss → 0."""
        user_embedding: tf.Tensor = tf.constant([[1.0, 1.0]])
        positive_embedding: tf.Tensor = tf.constant([[10.0, 10.0]])  # dot = 20
        negative_embedding: tf.Tensor = tf.constant([[0.0, 0.0]])  # dot = 0
        loss: tf.Tensor = self.loss_function(user_embedding, positive_embedding, negative_embedding)
        assert loss.numpy() < 0.01

    def test_negative_larger_loss_large(self) -> None:
        """When negative score >> positive score, loss is large."""
        user_embedding: tf.Tensor = tf.constant([[1.0, 1.0]])
        positive_embedding: tf.Tensor = tf.constant([[0.0, 0.0]])  # dot = 0
        negative_embedding: tf.Tensor = tf.constant([[10.0, 10.0]])  # dot = 20
        loss: tf.Tensor = self.loss_function(user_embedding, positive_embedding, negative_embedding)
        assert loss.numpy() > 10.0

    def test_batch_dimension(self) -> None:
        """Loss supports batched inputs and returns a scalar."""
        batch_size: int = 16
        dim: int = 4
        user_embedding: tf.Tensor = tf.random.normal([batch_size, dim])
        positive_embedding: tf.Tensor = tf.random.normal([batch_size, dim])
        negative_embedding: tf.Tensor = tf.random.normal([batch_size, dim])
        loss: tf.Tensor = self.loss_function(user_embedding, positive_embedding, negative_embedding)
        assert loss.shape == ()  # scalar

    def test_loss_is_non_negative(self) -> None:
        """BPR loss should always be non-negative (since -log(sigmoid(x)) >= 0)."""
        user_embedding: tf.Tensor = tf.random.normal([32, 8])
        positive_embedding: tf.Tensor = tf.random.normal([32, 8])
        negative_embedding: tf.Tensor = tf.random.normal([32, 8])
        loss: tf.Tensor = self.loss_function(user_embedding, positive_embedding, negative_embedding)
        assert loss.numpy() >= 0.0

    def test_symmetry_swap_increases_loss(self) -> None:
        """Swapping positive and negative items should increase the loss when pos > neg."""
        user_embedding: tf.Tensor = tf.constant([[1.0, 1.0]])
        positive_embedding: tf.Tensor = tf.constant([[5.0, 5.0]])
        negative_embedding: tf.Tensor = tf.constant([[0.1, 0.1]])
        loss_normal: tf.Tensor = self.loss_function(user_embedding, positive_embedding, negative_embedding)
        loss_swapped: tf.Tensor = self.loss_function(user_embedding, negative_embedding, positive_embedding)
        assert loss_swapped.numpy() > loss_normal.numpy()
