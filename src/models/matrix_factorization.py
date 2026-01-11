import numpy as np
import tensorflow as tf
import keras
from tqdm import tqdm

from src.baremetal import gather_dense
from src.sampler import SimpleSampler
from src.preprocessing import FeatureMeta
from typing import Union, Iterable

class MatrixFactorization(keras.Model):
    """
    Matrix Factorization Model for Recommendation Systems with minimal assumptions.
    We implement several features to ensure the model works on wide variety of datasets:
    - User and Item Lookup Layers to handle arbitrary user and item IDs.
    
    """

    def __init__(
        self, 
        features_meta: FeatureMeta,
        embedding_dimension_count: int, 
        l1_regularization: float = 0.0, 
        l2_regularization: float = 0.0
    ):
        super(MatrixFactorization, self).__init__()

        # Lookup Layers
        self.user_lookup_layer = keras.layers.IntegerLookup(
            vocabulary=features_meta["user_id"]["vocabulary"],
        )
        self.item_lookup_layer = keras.layers.IntegerLookup(
            vocabulary=features_meta["item_id"]["vocabulary"],
        )

        # Embedding Layers
        self.user_embedding_layer = keras.layers.Embedding(
            input_dim=features_meta["user_id"]["unique_count"] + 1,
            output_dim=embedding_dimension_count,
            embeddings_initializer='uniform',
            embeddings_regularizer=keras.regularizers.l1_l2(l1=l1_regularization, l2=l2_regularization)
        )
        self.item_embedding_layer = keras.layers.Embedding(
            input_dim=features_meta["item_id"]["unique_count"] + 1,
            output_dim=embedding_dimension_count,
            embeddings_initializer='uniform',
            embeddings_regularizer=keras.regularizers.l1_l2(l1=l1_regularization, l2=l2_regularization)
        )

        # Loss & Metrics Tracker
        self.train_loss_tracker = keras.metrics.Mean(name="train_loss")
        self.train_hit_rate_tracker = keras.metrics.Mean(name="train_hit_rate")
        self.train_recall_tracker = keras.metrics.Mean(name="train_recall")
        self.train_precision_tracker = keras.metrics.Mean(name="train_precision")
        self.test_loss_tracker = keras.metrics.Mean(name="test_loss")

        # Loss & Metrics History
        self.train_loss_history = []
        self.train_hit_rate_history = []
        self.train_recall_history = []
        self.train_precision_history = []
        self.test_loss_history = []

    def compile(
        self, 
        optimizer: keras.optimizers.Optimizer,
        loss_functions: Union[keras.losses.Loss, Iterable[keras.losses.Loss]],
        sampler: SimpleSampler,
        **kwargs
    ):
        self.optimizer = optimizer
        self.loss_functions = loss_functions
        self.sampler = sampler

    def call(self, user_ids: tf.Tensor, item_ids: tf.Tensor) -> tf.Tensor:
        user_embedding = self.user_embedding(user_ids)
        item_embedding = self.item_embedding(item_ids)
        predicted_interaction_probability = tf.reduce_sum(user_embedding * item_embedding, axis=1)
        return predicted_interaction_probability
    
    def user_embedding(self, user_ids: tf.Tensor) -> tf.Tensor:
        user_indices = self.user_lookup_layer(user_ids)
        user_embedding = self.user_embedding_layer(user_indices)
        return user_embedding
    
    def item_embedding(self, item_ids: tf.Tensor) -> tf.Tensor:
        item_indices = self.item_lookup_layer(item_ids)
        item_embedding = self.item_embedding_layer(item_indices)
        return item_embedding
    
    def calculate_loss(
        self,
        user_embeddings: tf.Tensor,
        positive_item_embeddings: tf.Tensor,
        negative_item_embeddings: tf.Tensor
    ) -> tf.Tensor:
        loss_values = []
        for loss_function in self.loss_functions:
            loss_value = loss_function(
                user_embeddings=user_embeddings,
                positive_item_embeddings=positive_item_embeddings,
                negative_item_embeddings=negative_item_embeddings
            )
            loss_values.append(loss_value)
        total_loss = tf.reduce_sum(loss_values)

        return total_loss
    
    def fit(
        self, 
        train_dataset: tf.data.Dataset,
        test_dataset: tf.data.Dataset = None,
        nepochs: int = 1,
        shuffle: bool = True,
        batch_size: int = 16384,
        **kwargs
    ):
        # Validation Check for Loss Functions
        if not self.loss_functions:
            raise ValueError("Loss functions must be provided before training. Please compile the model with appropriate loss functions.")
        
        # Validation Check for Optimizer
        if not self.optimizer:
            raise ValueError("Optimizer must be provided before training. Please compile the model with an appropriate optimizer.")

        # Dataset Preparation
        train_dataset_length = len(train_dataset)
        test_dataset_length = len(test_dataset) if test_dataset is not None else 0
        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=4*batch_size, reshuffle_each_iteration=True)
        if batch_size is not None and batch_size > 1:
            train_dataset = train_dataset.batch(batch_size)
            test_dataset = test_dataset.batch(batch_size) if test_dataset is not None else None
        
        for epoch in range(nepochs):
            self.reset_metrics()
            self.train_interaction_history = []
            self.test_interaction_history = []

            # Training Loop
            with tqdm(total=train_dataset_length, ncols=100, desc=f"[{epoch+1}/{nepochs}]") as pbar:
                for step, training_batch in enumerate(train_dataset):
                    user_ids = training_batch["user_id"]
                    item_ids = training_batch["item_id"]

                    # random negative sample
                    random_negatives = self.sampler.sample(user_ids)
                    
                    # forward pass
                    with tf.GradientTape() as tape:
                        user_embedding = self.user_embedding(user_ids)
                        item_embedding = self.item_embedding(item_ids)
                        negative_embedding = self.item_embedding(random_negatives)

                        loss_value = self.calculate_loss(
                            user_embeddings=user_embedding,
                            positive_item_embeddings=item_embedding,
                            negative_item_embeddings=negative_embedding
                        )

                    # calculate gradient
                    user_gradient, item_gradient = tape.gradient(loss_value, self.user_embedding_layer.trainable_variables + self.item_embedding_layer.trainable_variables)

                    # back propagation
                    self.optimizer.apply_gradients(zip([user_gradient, item_gradient], self.user_embedding_layer.trainable_variables + self.item_embedding_layer.trainable_variables))

                    # update training loss
                    self.train_loss_tracker.update_state(loss_value)

                    # record interaction history
                    user_indices = self.user_lookup_layer(user_ids)
                    item_indices = self.item_lookup_layer(item_ids)
                    self.train_interaction_history.append(tf.stack([user_indices, item_indices], axis=-1))

                    pbar.update(len(user_ids))
                    pbar.set_postfix({
                        "loss": float(self.train_loss_tracker.result())
                    })

                # finalize training interaction history and loss
                self.train_interaction_history = tf.concat(self.train_interaction_history, axis=0)
                self.train_loss_history.append(float(self.train_loss_tracker.result()))
                
            # Training Offline Evaluation
            train_interaction_matrix = self.construct_interaction_matrix(self.train_interaction_history)
            user_candidates = tf.range(self.user_lookup_layer.vocabulary_size(), dtype=tf.int64)
            item_candidates = tf.range(self.item_lookup_layer.vocabulary_size(), dtype=tf.int64)
            user_dataset = tf.data.Dataset.from_tensor_slices(user_candidates)
            user_dataset = user_dataset.batch(128)

            k = 10
            with tqdm(total=self.user_lookup_layer.vocabulary_size(), ncols=100, desc=f"[{epoch+1}/{nepochs}]") as pbar:
                for step, user_batch in enumerate(user_dataset):
                    user_embedding = self.user_embedding(user_batch)
                    candidate_item_embedding = self.item_embedding(item_candidates)
                    predicted_scores = tf.matmul(user_embedding, tf.transpose(candidate_item_embedding))
                    predicted_rankings = tf.argsort(tf.argsort(predicted_scores, direction='DESCENDING', axis=-1), axis=-1) + 1 # argsort twice gets you rankings of each item

                    user_indices
                    ground_truth_interaction_matrix = gather_dense(train_interaction_matrix, user_batch)

                    true_positives = tf.cast((predicted_rankings <= k) * ground_truth_interaction_matrix, tf.int32)
                    true_positive_count = tf.reduce_sum(true_positives, axis=-1) # true_positives per user
                    actual_positive_count = tf.reduce_sum(ground_truth_interaction_matrix, axis=-1) # actual positives per user

                    # update metrics
                    hit_rate = tf.cast(true_positive_count > 0, tf.float32)
                    recall = tf.math.divide_no_nan(true_positive_count, actual_positive_count)
                    precision = tf.math.divide_no_nan(true_positive_count, k)

                    self.train_hit_rate_tracker.update_state(hit_rate)
                    self.train_recall_tracker.update_state(recall)
                    self.train_precision_tracker.update_state(precision)
                    pbar.update(len(user_batch))

            # finalize training metrics
            self.train_hit_rate_history.append(float(self.train_hit_rate_tracker.result()))
            self.train_recall_history.append(float(self.train_recall_tracker.result()))
            self.train_precision_history.append(float(self.train_precision_tracker.result()))
            pbar.set_postfix({
                "recall": float(self.train_recall_tracker.result())
            })

            # Test Loop
            if test_dataset is not None:
                for step, evaluation_batch in enumerate(test_dataset):
                    user_ids = evaluation_batch["user_id"]
                    item_ids = evaluation_batch["item_id"]

                    # random negative sample
                    random_negatives = self.sampler.sample(user_ids)

                    # forward pass
                    user_embedding = self.user_embedding(user_ids)
                    item_embedding = self.item_embedding(item_ids)
                    negative_embedding = self.item_embedding(random_negatives)

                    loss_value = self.calculate_loss(
                        user_embeddings=user_embedding,
                        positive_item_embeddings=item_embedding,
                        negative_item_embeddings=negative_embedding
                    )

                    # update test loss
                    self.test_loss_tracker.update_state(loss_value)

                    # record interaction history
                    user_indices = self.user_lookup_layer(user_ids)
                    item_indices = self.item_lookup_layer(item_ids)
                    self.test_interaction_history.append(tf.stack([user_indices, item_indices], axis=-1))
                
                # finalize test interaction history and loss
                self.test_interaction_history = tf.concat(self.test_interaction_history, axis=0)
                self.test_loss_history.append(float(self.test_loss_tracker.result()))

                pbar.set_postfix({
                    "test_loss": float(self.test_loss_tracker.result())
                })

    def evaluate(
        self,
        dataset: tf.data.Dataset,
        batch_size: int = 16384,
        **kwargs
    ):
        if batch_size is not None and batch_size > 1:
            dataset = dataset.batch(batch_size)

        for step, evaluation_batch in enumerate(dataset):
            break

    def construct_interaction_matrix(
        self,
        interaction_history: tf.Tensor,
    ) -> tf.sparse.SparseTensor:
        interaction_matrix = tf.sparse.SparseTensor(
            indices=interaction_history,
            values=tf.ones(shape=(len(interaction_history),), dtype=tf.int32),
            dense_shape=(self.user_lookup_layer.vocabulary_size(), self.item_lookup_layer.vocabulary_size())
        )

        return tf.sparse.reorder(interaction_matrix)
            
        