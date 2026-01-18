import numpy as np
import tensorflow as tf
import keras

class BayesianPersonalizedRankingLoss(keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def __call__(self, user_embeddings: tf.Tensor, positive_item_embeddings: tf.Tensor, negative_item_embeddings: tf.Tensor) -> tf.Tensor:
        positive_prediction = tf.reduce_sum(user_embeddings * positive_item_embeddings, axis=-1)
        negative_prediction = tf.reduce_sum(user_embeddings * negative_item_embeddings, axis=-1)

        return tf.reduce_mean(-tf.math.log(tf.sigmoid(positive_prediction - negative_prediction)))