import numpy as np
import tensorflow as tf
import keras

class MatrixFactorizationModel(keras.Model):
    def __init__(self, user_count: int, item_count: int, embedding_dimension_count: int, l1_regularization: float = 0.00, l2_regularization: float = 0.00):
        super(MatrixFactorizationModel, self).__init__()
        self.user_embedding = keras.layers.Embedding(
            input_dim=user_count,
            output_dim=embedding_dimension_count,
            embeddings_initializer='uniform',
            embeddings_regularizer=keras.regularizers.l2(l2_regularization) + keras.regularizers.l1(l1_regularization)
        )
        self.item_embedding = keras.layers.Embedding(
            input_dim=item_count,
            output_dim=embedding_dimension_count,
            embeddings_initializer='uniform',
            embeddings_regularizer=keras.regularizers.l2(l2_regularization) + keras.regularizers.l1(l1_regularization)
        )

    def call(self, user_ids: tf.Tensor, item_ids: tf.Tensor) -> tf.Tensor:
        user_embedding = self.user_embedding(user_ids)
        item_embedding = self.item_embedding(item_ids)
        predicted_interaction_probability = tf.reduce_sum(user_embedding * item_embedding, axis=1)
        return predicted_interaction_probability

