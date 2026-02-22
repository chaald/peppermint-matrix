import pytest
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import keras

from src.models.matrix_factorization import MatrixFactorization
from src.losses.bayesian_personalized_ranking import BayesianPersonalizedRankingLoss
from src.preprocessing import FeatureMeta

class TestMatrixFactorizationInit:
    """Tests for model initialization and layer shapes."""

    def test_model_is_keras_model(self, sample_model: MatrixFactorization) -> None:
        assert isinstance(sample_model, keras.Model)

    def test_user_embedding_layer_input_dim(self, sample_model: MatrixFactorization, sample_features_meta: FeatureMeta) -> None:
        expected_input_dim: int = sample_features_meta["user_id"]["unique_count"] + 1
        weights: npt.NDArray[np.float32] = sample_model.user_embedding_layer.get_weights()[0]
        assert weights.shape[0] == expected_input_dim
        assert weights.shape[1] == 8

    def test_item_embedding_layer_input_dim(self, sample_model: MatrixFactorization, sample_features_meta: FeatureMeta) -> None:
        expected_input_dim: int = sample_features_meta["item_id"]["unique_count"] + 1
        weights: npt.NDArray[np.float32] = sample_model.item_embedding_layer.get_weights()[0]
        assert weights.shape[0] == expected_input_dim
        assert weights.shape[1] == 8

    def test_evaluation_cutoffs_stored(self, sample_model: MatrixFactorization) -> None:
        assert sample_model.evaluation_cutoffs == [2, 5]

    def test_metric_trackers_initialized(self, sample_model: MatrixFactorization) -> None:
        for k in [2, 5]:
            assert k in sample_model.train_recall_tracker
            assert k in sample_model.test_recall_tracker
            assert k in sample_model.train_ndcg_tracker
            assert k in sample_model.test_ndcg_tracker

class TestMatrixFactorizationForward:
    """Tests for forward pass and embedding methods."""

    def test_call_output_shape(self, sample_model: MatrixFactorization, sample_features_meta: FeatureMeta) -> None:
        n: int = 4
        user_ids: tf.Tensor = tf.constant(sample_features_meta["user_id"]["vocabulary"][:n])
        item_ids: tf.Tensor = tf.constant(sample_features_meta["item_id"]["vocabulary"][:n])
        output: tf.Tensor = sample_model(user_ids, item_ids)
        assert output.shape == (n,)

    def test_call_output_dtype(self, sample_model: MatrixFactorization, sample_features_meta: FeatureMeta) -> None:
        n: int = 3
        user_ids: tf.Tensor = tf.constant(sample_features_meta["user_id"]["vocabulary"][:n])
        item_ids: tf.Tensor = tf.constant(sample_features_meta["item_id"]["vocabulary"][:n])
        output: tf.Tensor = sample_model(user_ids, item_ids)
        assert output.dtype == tf.float32

    def test_user_embedding_output_shape(self, sample_model: MatrixFactorization, sample_features_meta: FeatureMeta) -> None:
        user_ids: tf.Tensor = tf.constant(sample_features_meta["user_id"]["vocabulary"][:4])
        emb: tf.Tensor = sample_model.user_embedding(user_ids)
        assert emb.shape == (4, 8)

    def test_item_embedding_output_shape(self, sample_model: MatrixFactorization, sample_features_meta: FeatureMeta) -> None:
        item_ids: tf.Tensor = tf.constant(sample_features_meta["item_id"]["vocabulary"][:5])
        emb: tf.Tensor = sample_model.item_embedding(item_ids)
        assert emb.shape == (5, 8)

    def test_different_inputs_different_outputs(self, sample_model: MatrixFactorization, sample_features_meta: FeatureMeta) -> None:
        vocab: list[int] = sample_features_meta["user_id"]["vocabulary"]
        emb1: tf.Tensor = sample_model.user_embedding(tf.constant([vocab[0]]))
        emb2: tf.Tensor = sample_model.user_embedding(tf.constant([vocab[1]]))
        assert not np.allclose(emb1.numpy(), emb2.numpy())

class TestMatrixFactorizationCalculateLoss:
    """Tests for ``calculate_loss``."""

    def test_returns_scalar(self, compiled_model: MatrixFactorization) -> None:
        user_embedding: tf.Tensor = tf.random.normal([4, 8])
        positive_embedding: tf.Tensor = tf.random.normal([4, 8])
        negative_embedding: tf.Tensor = tf.random.normal([4, 8])
        loss: tf.Tensor = compiled_model.calculate_loss(user_embedding, positive_embedding, negative_embedding)
        assert loss.shape == ()

    def test_loss_is_non_negative(self, compiled_model: MatrixFactorization) -> None:
        user_embedding: tf.Tensor = tf.random.normal([8, 8])
        positive_embedding: tf.Tensor = tf.random.normal([8, 8])
        negative_embedding: tf.Tensor = tf.random.normal([8, 8])
        loss: tf.Tensor = compiled_model.calculate_loss(user_embedding, positive_embedding, negative_embedding)
        assert loss.numpy() >= 0.0

class TestMatrixFactorizationInteractionMatrix:
    """Tests for ``construct_interaction_matrix``."""

    def test_returns_sparse_tensor(self, compiled_model: MatrixFactorization) -> None:
        history: tf.Tensor = tf.constant([[0, 1], [0, 2], [1, 3]], dtype=tf.int64)
        matrix: tf.SparseTensor = compiled_model.construct_interaction_matrix(history)
        assert isinstance(matrix, tf.SparseTensor)

    def test_correct_values(self, compiled_model: MatrixFactorization) -> None:
        history: tf.Tensor = tf.constant([[0, 1], [0, 2], [1, 3]], dtype=tf.int64)
        matrix: tf.SparseTensor = compiled_model.construct_interaction_matrix(history)
        dense: npt.NDArray[np.float32] = tf.sparse.to_dense(matrix).numpy()
        assert dense[0, 1] == 1.0
        assert dense[0, 2] == 1.0
        assert dense[1, 3] == 1.0
        assert dense[1, 1] == 0.0  # no interaction

class TestMatrixFactorizationGetConfig:
    """Tests for model serialization config."""

    def test_get_config_keys(self, sample_model: MatrixFactorization) -> None:
        config: dict[str, object] = sample_model.get_config()
        assert "embedding_dimension_count" in config
        assert "l1_regularization" in config
        assert "l2_regularization" in config
        assert "embedding_dropout_rate" in config
        assert "evaluation_cutoffs" in config
        assert "features_meta" in config

    def test_get_config_values(self, sample_model: MatrixFactorization) -> None:
        config: dict[str, object] = sample_model.get_config()
        assert config["embedding_dimension_count"] == 8
        assert config["evaluation_cutoffs"] == [2, 5]

class TestMatrixFactorizationCompile:
    """Tests for model compilation validation."""

    def test_compile_sets_optimizer(self, compiled_model: MatrixFactorization) -> None:
        assert compiled_model.optimizer is not None
        assert isinstance(compiled_model.optimizer, keras.optimizers.Adam)

    def test_compile_sets_loss_functions(self, compiled_model: MatrixFactorization) -> None:
        assert len(compiled_model.loss_functions) == 1
        assert isinstance(compiled_model.loss_functions[0], BayesianPersonalizedRankingLoss)

    def test_compile_sets_sampler(self, compiled_model: MatrixFactorization) -> None:
        from src.sampler import BayesianSampler
        assert isinstance(compiled_model.sampler, BayesianSampler)
