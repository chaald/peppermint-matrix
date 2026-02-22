import pytest
import numpy as np
import numpy.typing as npt
import tensorflow as tf

from src.baremetal import gather_dense

class TestGatherDense:
    """Tests for ``gather_dense`` — sparse tensor row extraction."""

    @pytest.fixture
    def sparse_matrix(self) -> tf.SparseTensor:
        """A 4x5 sparse matrix with known values.

        Row 0: [1, 0, 1, 0, 0]
        Row 1: [0, 1, 0, 1, 0]
        Row 2: [1, 1, 0, 0, 1]
        Row 3: [0, 0, 0, 1, 1]
        """
        indices: tf.Tensor = tf.constant([
            [0, 0], [0, 2],
            [1, 1], [1, 3],
            [2, 0], [2, 1], [2, 4],
            [3, 3], [3, 4],
        ], dtype=tf.int64)
        values: tf.Tensor = tf.ones(9, dtype=tf.float32)
        return tf.sparse.reorder(
            tf.SparseTensor(indices=indices, values=values, dense_shape=(4, 5))
        )

    def test_single_row_extraction(self, sparse_matrix: tf.SparseTensor) -> None:
        row_indices: tf.Tensor = tf.constant([0], dtype=tf.int64)
        result: tf.Tensor = gather_dense(sparse_matrix, row_indices)
        expected: npt.NDArray[np.float32] = np.array([[1, 0, 1, 0, 0]], dtype=np.float32)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_multiple_rows(self, sparse_matrix: tf.SparseTensor) -> None:
        row_indices: tf.Tensor = tf.constant([0, 2], dtype=tf.int64)
        result: tf.Tensor = gather_dense(sparse_matrix, row_indices)
        expected: npt.NDArray[np.float32] = np.array([
            [1, 0, 1, 0, 0],
            [1, 1, 0, 0, 1],
        ], dtype=np.float32)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_output_shape(self, sparse_matrix: tf.SparseTensor) -> None:
        row_indices: tf.Tensor = tf.constant([1, 3], dtype=tf.int64)
        result: tf.Tensor = gather_dense(sparse_matrix, row_indices)
        assert result.shape == (2, 5)

    def test_all_rows(self, sparse_matrix: tf.SparseTensor) -> None:
        row_indices: tf.Tensor = tf.constant([0, 1, 2, 3], dtype=tf.int64)
        result: tf.Tensor = gather_dense(sparse_matrix, row_indices)
        expected: npt.NDArray[np.float32] = np.array([
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 1, 0, 0, 1],
            [0, 0, 0, 1, 1],
        ], dtype=np.float32)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_reversed_row_order(self, sparse_matrix: tf.SparseTensor) -> None:
        row_indices: tf.Tensor = tf.constant([3, 1], dtype=tf.int64)
        result: tf.Tensor = gather_dense(sparse_matrix, row_indices)
        expected: npt.NDArray[np.float32] = np.array([
            [0, 0, 0, 1, 1],
            [0, 1, 0, 1, 0],
        ], dtype=np.float32)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_single_element_row(self, sparse_matrix: tf.SparseTensor) -> None:
        """Row 3 has entries only at columns 3 and 4."""
        row_indices: tf.Tensor = tf.constant([3], dtype=tf.int64)
        result: tf.Tensor = gather_dense(sparse_matrix, row_indices)
        expected: npt.NDArray[np.float32] = np.array([[0, 0, 0, 1, 1]], dtype=np.float32)
        np.testing.assert_array_equal(result.numpy(), expected)
