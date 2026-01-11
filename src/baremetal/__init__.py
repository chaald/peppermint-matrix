import numpy as np
import tensorflow as tf

def gather_dense(sparse_tensor: tf.sparse.SparseTensor, row_indices:tf.Tensor) -> tf.Tensor:
    """Take only the indices specified in the row indices and then convert to dense."""
    old_indices = sparse_tensor.indices[:, 0]
    old_indices_hit_coordinates = tf.equal(old_indices[:, tf.newaxis], row_indices[tf.newaxis, :])
    old_indices_hit_mask = tf.reduce_any(old_indices_hit_coordinates, axis=-1)

    new_row_indices = tf.reduce_max(tf.range(len(row_indices), dtype=tf.int64) * tf.cast(old_indices_hit_coordinates[old_indices_hit_mask], tf.int64), axis=-1)
    new_column_indices = sparse_tensor.indices[:, 1][old_indices_hit_mask]

    interaction_matrix_slice = tf.zeros(shape=(len(row_indices), sparse_tensor.shape[1]), dtype=sparse_tensor.dtype)
    interaction_matrix_slice = tf.tensor_scatter_nd_update(
        interaction_matrix_slice,
        tf.stack([new_row_indices, new_column_indices], axis=-1),
        sparse_tensor.values[old_indices_hit_mask]
    )
    return interaction_matrix_slice
