import numpy as np
import tensorflow as tf
import keras
from tqdm import tqdm

from src.sampler import SimpleSampler
from typing import Union, Iterable

class MatrixFactorization(keras.Model):
    def __init__(self, user_count: int, item_count: int, embedding_dimension_count: int, l1_regularization: float = 0.00, l2_regularization: float = 0.00):
        super(MatrixFactorization, self).__init__()
        self.user_embedding_layer = keras.layers.Embedding(
            input_dim=user_count,
            output_dim=embedding_dimension_count,
            embeddings_initializer='uniform',
            embeddings_regularizer=keras.regularizers.l1_l2(l1=l1_regularization, l2=l2_regularization)
        )
        self.item_embedding_layer = keras.layers.Embedding(
            input_dim=item_count,
            output_dim=embedding_dimension_count,
            embeddings_initializer='uniform',
            embeddings_regularizer=keras.regularizers.l1_l2(l1=l1_regularization, l2=l2_regularization)
        )

        # Loss & Metrics Tracker
        self.train_loss_tracker = keras.metrics.Mean(name="train_loss")
        self.test_loss_tracker = keras.metrics.Mean(name="test_loss")

        # Loss & Metrics History
        self.train_loss_history = []
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
        user_embedding = self.user_embedding_layer(user_ids)
        item_embedding = self.item_embedding_layer(item_ids)
        predicted_interaction_probability = tf.reduce_sum(user_embedding * item_embedding, axis=1)
        return predicted_interaction_probability
    
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
        
        for epoch in range(nepochs):
            self.reset_metrics()

            # Training Loop
            with tqdm(total=train_dataset_length, ncols=100, desc=f"[{epoch+1}/{nepochs}]") as pbar:
                for step, training_batch in enumerate(train_dataset):
                    user_ids = training_batch["user_id"]
                    item_ids = training_batch["item_id"]

                    # random negative sample
                    random_negatives = self.sampler.sample(user_ids)
                    
                    # forward pass
                    with tf.GradientTape() as tape:
                        user_embedding = self.user_embedding_layer(user_ids)
                        item_embedding = self.item_embedding_layer(item_ids)
                        negative_embedding = self.item_embedding_layer(random_negatives)

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

                    pbar.update(len(user_ids))
                    pbar.set_postfix({"loss": float(self.train_loss_tracker.result())})
            self.train_loss_history.append(float(self.train_loss_tracker.result()))

            # Test Loop
            if test_dataset is not None:
                self.evaluate(test_dataset, batch_size=batch_size)
                pbar.set_postfix({
                    "loss": float(self.train_loss_tracker.result()), 
                    "test_loss": float(self.test_loss_tracker.result())
                })

    def evaluate(
        self,
        dataset: tf.data.Dataset,
        batch_size: int = 16384,
        **kwargs
    ):
        dataset_length = len(dataset)
        if batch_size is not None and batch_size > 1:
            dataset = dataset.batch(batch_size)
        
        for step, evaluation_batch in enumerate(dataset):
            user_ids = evaluation_batch["user_id"]
            item_ids = evaluation_batch["item_id"]

            # random negative sample
            random_negatives = self.sampler.sample(user_ids)

            # forward pass
            user_embedding = self.user_embedding_layer(user_ids)
            item_embedding = self.item_embedding_layer(item_ids)
            negative_embedding = self.item_embedding_layer(random_negatives)

            loss_value = self.calculate_loss(
                user_embeddings=user_embedding,
                positive_item_embeddings=item_embedding,
                negative_item_embeddings=negative_embedding
            )

            # update test loss
            self.test_loss_tracker.update_state(loss_value)
        self.test_loss_history.append(float(self.test_loss_tracker.result()))