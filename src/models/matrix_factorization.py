import numpy as np
import tensorflow as tf
import keras
from tqdm import tqdm

from src.baremetal import gather_dense
from src.utils import preprocess_metric_aggregate
from src.sampler import BayesianSampler
from src.preprocessing import FeatureMeta
from typing import Union, Iterable, Dict, List

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
        l2_regularization: float = 0.0,
        evaluation_cutoffs: list = [2, 10, 50],
        **kwargs
    ):
        super(MatrixFactorization, self).__init__(**kwargs)
        self.features_meta = features_meta
        self.embedding_dimension_count = embedding_dimension_count
        self.l1_regularization = l1_regularization
        self.l2_regularization = l2_regularization
        self.evaluation_cutoffs = evaluation_cutoffs

        # Lookup Layers
        self.user_lookup_layer = keras.layers.IntegerLookup(
            vocabulary=features_meta["user_id"]["vocabulary"],
            name="user_lookup_layer"
        )
        self.item_lookup_layer = keras.layers.IntegerLookup(
            vocabulary=features_meta["item_id"]["vocabulary"],
            name="item_lookup_layer"
        )

        # Embedding Layers
        self.user_embedding_layer = keras.layers.Embedding(
            input_dim=features_meta["user_id"]["unique_count"] + 1,
            output_dim=embedding_dimension_count,
            embeddings_initializer='uniform',
            embeddings_regularizer=keras.regularizers.l1_l2(l1=l1_regularization, l2=l2_regularization),
            name="user_embedding_layer"
        )
        self.item_embedding_layer = keras.layers.Embedding(
            input_dim=features_meta["item_id"]["unique_count"] + 1,
            output_dim=embedding_dimension_count,
            embeddings_initializer='uniform',
            embeddings_regularizer=keras.regularizers.l1_l2(l1=l1_regularization, l2=l2_regularization),
            name="item_embedding_layer"
        )

        # Loss & Metrics Tracker
        self.train_loss_tracker = keras.metrics.Mean(name="train_loss")
        self.test_loss_tracker = keras.metrics.Mean(name="test_loss")
        self.train_hitrate_tracker: Dict[int, keras.metrics.Mean] = dict()
        self.train_recall_tracker: Dict[int, keras.metrics.Mean] = dict()
        self.train_precision_tracker: Dict[int, keras.metrics.Mean] = dict()
        self.train_map_tracker: Dict[int, keras.metrics.Mean] = dict()
        self.train_ndcg_tracker: Dict[int, keras.metrics.Mean] = dict()
        self.train_mrr_tracker: Dict[int, keras.metrics.Mean] = dict()
        self.test_hitrate_tracker: Dict[int, keras.metrics.Mean] = dict()
        self.test_recall_tracker: Dict[int, keras.metrics.Mean] = dict()
        self.test_precision_tracker: Dict[int, keras.metrics.Mean] = dict()
        self.test_map_tracker: Dict[int, keras.metrics.Mean] = dict()
        self.test_ndcg_tracker: Dict[int, keras.metrics.Mean] = dict()
        self.test_mrr_tracker: Dict[int, keras.metrics.Mean] = dict()
        for k in evaluation_cutoffs:
            self.train_hitrate_tracker[k] = keras.metrics.Mean(name=f"train_hitrate@{k}")
            self.train_recall_tracker[k] = keras.metrics.Mean(name=f"train_recall@{k}")
            self.train_precision_tracker[k] = keras.metrics.Mean(name=f"train_precision@{k}")
            self.train_map_tracker[k] = keras.metrics.Mean(name=f"train_map@{k}")
            self.train_ndcg_tracker[k] = keras.metrics.Mean(name=f"train_ndcg@{k}")
            self.train_mrr_tracker[k] = keras.metrics.Mean(name=f"train_mrr@{k}")
            self.test_hitrate_tracker[k] = keras.metrics.Mean(name=f"test_hitrate@{k}")
            self.test_recall_tracker[k] = keras.metrics.Mean(name=f"test_recall@{k}")
            self.test_precision_tracker[k] = keras.metrics.Mean(name=f"test_precision@{k}")
            self.test_map_tracker[k] = keras.metrics.Mean(name=f"test_map@{k}")
            self.test_ndcg_tracker[k] = keras.metrics.Mean(name=f"test_ndcg@{k}")
            self.test_mrr_tracker[k] = keras.metrics.Mean(name=f"test_mrr@{k}")

        # Loss & Metrics History
        self.train_loss_history = []
        self.test_loss_history = []
        self.train_hitrate_history: Dict[int, List[float]] = dict()
        self.train_recall_history: Dict[int, List[float]] = dict()
        self.train_precision_history: Dict[int, List[float]] = dict()
        self.train_map_history: Dict[int, List[float]] = dict()
        self.train_ndcg_history: Dict[int, List[float]] = dict()
        self.train_mrr_history: Dict[int, List[float]] = dict()
        self.test_hitrate_history: Dict[int, List[float]] = dict()
        self.test_recall_history: Dict[int, List[float]] = dict()
        self.test_precision_history: Dict[int, List[float]] = dict()
        self.test_map_history: Dict[int, List[float]] = dict()
        self.test_ndcg_history: Dict[int, List[float]] = dict()
        self.test_mrr_history: Dict[int, List[float]] = dict()
        for k in evaluation_cutoffs:
            self.train_hitrate_history[k] = []
            self.train_recall_history[k] = []
            self.train_precision_history[k] = []
            self.train_map_history[k] = []
            self.train_ndcg_history[k] = []
            self.train_mrr_history[k] = []
            self.test_hitrate_history[k] = []
            self.test_recall_history[k] = []
            self.test_precision_history[k] = []
            self.test_map_history[k] = []
            self.test_ndcg_history[k] = []
            self.test_mrr_history[k] = []

        # Build the model, if not already built
        user_ids = tf.constant(self.features_meta["user_id"]["vocabulary"][:8])
        item_ids = tf.constant(self.features_meta["item_id"]["vocabulary"][:8])
        self(user_ids, item_ids)

    def compile(
        self, 
        optimizer: keras.optimizers.Optimizer,
        loss_functions: Union[keras.losses.Loss, Iterable[keras.losses.Loss]],
        sampler: BayesianSampler,
        **kwargs
    ):
        self.optimizer = optimizer
        self.loss_functions = loss_functions
        self.sampler = sampler

        # Interaction History
        self.train_interaction_history = []
        self.test_interaction_history = []

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

        regularization_loss = tf.reduce_sum(self.losses)
        total_loss = total_loss + regularization_loss

        return total_loss
    
    def fit(
        self, 
        train_dataset: tf.data.Dataset,
        test_dataset: tf.data.Dataset = None,
        nepochs: int = 1,
        shuffle: bool = True,
        batch_size: int = 16384,
        callbacks: list = [],
        **kwargs
    ):
        # Validation Check for Loss Functions
        if not self.loss_functions:
            raise ValueError("Loss functions must be provided before training. Please compile the model with appropriate loss functions.")
        
        # Validation Check for Optimizer
        if not self.optimizer:
            raise ValueError("Optimizer must be provided before training. Please compile the model with an appropriate optimizer.")

        # Initialize Callbacks
        self.stop_training = False
        if not isinstance(callbacks, keras.callbacks.CallbackList):
            callbacks = keras.callbacks.CallbackList(
                callbacks,
                model=self,
            )

        # Dataset Preparation
        train_dataset_length = len(train_dataset)
        test_dataset_length = len(test_dataset) if test_dataset is not None else 0
        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=4*batch_size, reshuffle_each_iteration=True)
        if batch_size is not None and batch_size > 1:
            train_dataset = train_dataset.batch(batch_size)
            test_dataset = test_dataset.batch(batch_size) if test_dataset is not None else None
        
        callbacks.on_train_begin()
        for epoch in range(nepochs):
            self.reset_metrics()
            self.train_interaction_history = []
            self.test_interaction_history = []
            callbacks.on_epoch_begin(epoch)

            # Training Loop
            metrics_aggregate = {}
            with tqdm(total=train_dataset_length + test_dataset_length, ncols=176, desc=f"TL [{epoch+1}/{nepochs}]") as pbar:
                for step, training_batch in enumerate(train_dataset):
                    callbacks.on_train_batch_begin(step)
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
                    metrics_aggregate.update({"train_loss": float(self.train_loss_tracker.result()), "step_delta": len(user_ids)})

                    callbacks.on_train_batch_end(step, metrics_aggregate)
                    pbar.update(len(user_ids))
                    pbar.set_postfix(preprocess_metric_aggregate(metrics_aggregate))

                # finalize training interaction history and loss
                self.train_interaction_history = tf.concat(self.train_interaction_history, axis=0)
                self.train_loss_history.append(float(self.train_loss_tracker.result()))
            
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
                        metrics_aggregate.update({"test_loss": float(self.test_loss_tracker.result())})

                        pbar.update(len(user_ids))
                        pbar.set_postfix(preprocess_metric_aggregate(metrics_aggregate))

                    # finalize test interaction history and loss
                    self.test_interaction_history = tf.concat(self.test_interaction_history, axis=0)
                    self.test_loss_history.append(float(self.test_loss_tracker.result()))

            # Offline Evaluation
            evaluation_result = self.evaluate(
                describe=f"OE [{epoch+1}/{nepochs}]"
            )
            metrics_aggregate.update(evaluation_result)

            callbacks.on_epoch_end(epoch, metrics_aggregate)

            # Early Stopping Check
            if self.stop_training:
                break

        callbacks.on_train_end(metrics_aggregate)
        return metrics_aggregate
            
    def evaluate(
        self,
        train_dataset: tf.data.Dataset = None,
        test_dataset: tf.data.Dataset = None,
        batch_size: int = 128,
        describe: str = "",
    ):
        ## Prepare Interaction Matrices
        # This is the R matrix in MF literature
        # We need this to compute metrics in user level efficiently
        # train_interaction_matrix | [U, I] sparse tensor
        # test_interaction_matrix  | [U, I] sparse tensor
        if train_dataset is not None:
            train_interaction_matrix = self.construct_interaction_matrix(self.construct_interaction_history(train_dataset))
        else:
            train_interaction_matrix = self.construct_interaction_matrix(self.train_interaction_history)

        if test_dataset is not None:
            test_interaction_matrix = self.construct_interaction_matrix(self.construct_interaction_history(test_dataset))
        elif len(self.test_interaction_history) > 0:
            test_interaction_matrix = self.construct_interaction_matrix(self.test_interaction_history)
        else:
            test_interaction_matrix = None
        
        # Prepare Candidates and User Dataset
        user_candidates = tf.constant(self.user_lookup_layer.get_vocabulary()[1:], dtype=tf.int64) # [U]
        item_candidates = tf.constant(self.item_lookup_layer.get_vocabulary(), dtype=tf.int64) # [I]
        user_dataset = tf.data.Dataset.from_tensor_slices(user_candidates)
        user_dataset = user_dataset.batch(batch_size)

        metrics_aggregate = {}
        maxk = max(self.evaluation_cutoffs)
        with tqdm(total=len(user_candidates), ncols=176, desc=describe) as pbar:
            for step, user_batch in enumerate(user_dataset):
                # Get BxI predicted scores for all candidate items
                # where B = batch size, I = candidate item count
                user_indices = self.user_lookup_layer(user_batch) # [B]
                user_embedding = self.user_embedding(user_batch) # [B, D]
                candidate_item_embedding = self.item_embedding(item_candidates) # [I, D]
                predicted_scores = tf.matmul(user_embedding, tf.transpose(candidate_item_embedding)) # [B, I]

                self.evaluate_kernel(
                    predicted_scores = predicted_scores,
                    user_indices = user_indices,
                    train_interaction_matrix = train_interaction_matrix,
                    test_interaction_matrix = test_interaction_matrix,
                    maxk = maxk
                )
                pbar.update(len(user_batch))

        # finalize metrics
        for metrics in self.metrics:
            metrics_aggregate.update({metrics.name: float(metrics.result())})

        for k in self.evaluation_cutoffs:
            self.train_hitrate_history[k].append(float(self.train_hitrate_tracker[k].result()))
            self.train_recall_history[k].append(float(self.train_recall_tracker[k].result()))
            self.train_precision_history[k].append(float(self.train_precision_tracker[k].result()))
            self.train_map_history[k].append(float(self.train_map_tracker[k].result()))
            self.train_ndcg_history[k].append(float(self.train_ndcg_tracker[k].result()))
            self.train_mrr_history[k].append(float(self.train_mrr_tracker[k].result()))
            if len(self.test_interaction_history):
                self.test_hitrate_history[k].append(float(self.test_hitrate_tracker[k].result()))
                self.test_recall_history[k].append(float(self.test_recall_tracker[k].result()))
                self.test_precision_history[k].append(float(self.test_precision_tracker[k].result()))

        return metrics_aggregate

    @tf.function(jit_compile=True) # jit compile speed up evaluation from 32s to 2s per epoch, a 16x speedup!
    def evaluate_kernel(
        self, 
        user_indices: tf.Tensor,
        predicted_scores: tf.Tensor,
        train_interaction_matrix: tf.SparseTensor,
        test_interaction_matrix: tf.SparseTensor,
        maxk: int
    ):
        # Get Top-K recommended items and scores
        # predicted_train_rankings = tf.argsort(tf.argsort(predicted_scores, direction='DESCENDING', axis=-1), axis=-1) + 1 # argsort twice gets you rankings of each item | B*Ilog(I)
        sorted_train_scores, predicted_train_indices = tf.math.top_k(predicted_scores, k=maxk) # partition + partial sort gives you faster result  | B*(I + klog(k)), k << I | [B, K]

        # Train Metrics
        train_ground_truth = gather_dense(train_interaction_matrix, user_indices) # gather([U, I], [B]) -> [B, I]
        train_actual_positive_count = tf.reduce_sum(train_ground_truth, axis=-1) # actual positives per user (TP + FN) | [B] 
        for k in sorted(self.evaluation_cutoffs, reverse=True):
            positions = tf.range(k, dtype=tf.float32) # [K]
            predicted_train_indices_k = predicted_train_indices[:, :k] # [B, K]
            # train_true_positives = tf.cast((predicted_train_rankings <= k) * train_ground_truth, tf.int32)
            # train_true_positive_count = tf.reduce_sum(train_true_positives, axis=-1) # true_positives per user
            train_true_positives = tf.gather(train_ground_truth, predicted_train_indices_k, batch_dims=-1) # this is equivalent to the commented lines above | gather([B, I], [B, K]) -> [B, K]
            train_true_positive_count = tf.reduce_sum(train_true_positives, axis=-1) # true positives per user (TP) | [B]

            ## Hit Rate@k
            # 1 if TP_u > 0 else 0
            train_hit = tf.cast(train_true_positive_count > 0, tf.float32) # [B]

            ## Recall@k
            # TP / (TP + FN) for u in U
            train_recall = tf.math.divide_no_nan(train_true_positive_count, train_actual_positive_count) # [B]

            ## Precision@k
            # TP / (TP + FP) for u in U
            train_precision = tf.math.divide_no_nan(train_true_positive_count, k) # [B]

            ## MAP@k
            # Precision(k)
            train_precision_at_each_position = tf.cumsum(train_true_positives, axis=-1) / (positions + 1) # [B, K]
            # Precision(k) * rel(k)
            train_precision_at_each_recall_position = train_precision_at_each_position * train_true_positives # [B, K]
            # Σ Precision(k) * rel(k) / total_recalled_items | total_recalled_items = TP
            train_average_precision = tf.math.divide_no_nan(tf.reduce_sum(train_precision_at_each_recall_position, axis=-1), train_true_positive_count) # [B]
            
            ## NDCG@k
            # log(k + 1, base=2)
            log_discount = tf.math.log(positions + 2) / tf.math.log(2.0) # [K]
            # Σ rel_k / log2(k + 1)
            train_dcg = tf.reduce_sum(train_true_positives / log_discount, axis=-1) # [B]
            # ideal_rel_k = 1 if k <= total_recalled_items else 0
            train_ideal_true_positives = tf.cast(positions[tf.newaxis, :] < train_true_positive_count[:, tf.newaxis], tf.float32) # [B, K]
            # Σ ideal_rel_k / log2(k + 1)
            train_idcg = tf.reduce_sum(train_ideal_true_positives / log_discount, axis=-1) # [B]
            # dcg / idcg
            train_ndcg = tf.math.divide_no_nan(train_dcg, train_idcg) # [B]
            
            ## MRR@k
            # Set non-positive positions to inf so reduce_min finds first positive
            train_first_positive_positions = tf.where(train_true_positives==1, train_true_positives * (positions + 1), float('inf')) # [B, K]
            train_reciprocal_ranks = 1 / tf.reduce_min(train_first_positive_positions, axis=-1) # [B]

            # Update Metrics Tracker
            self.train_hitrate_tracker[k].update_state(train_hit)
            self.train_recall_tracker[k].update_state(train_recall)
            self.train_precision_tracker[k].update_state(train_precision)
            self.train_map_tracker[k].update_state(train_average_precision)
            self.train_ndcg_tracker[k].update_state(train_ndcg)
            self.train_mrr_tracker[k].update_state(train_reciprocal_ranks)

            # Test Metrics
            if test_interaction_matrix is not None:
                ## Train Interactions Mask
                # Following the original BPR paper, we only evaluate on items not seen in training set during test evaluation
                # Please see Section 6.2 of "BPR: Bayesian Personalized Ranking from Implicit Feedback" by Rendle et al., 2009
                ranking_mask = tf.where(train_ground_truth==1, float('inf'), 0.0)
                
                # Get Top-K recommended items and scores after applying train interaction mask
                # predicted_test_rankings = tf.argsort(tf.argsort((predicted_scores - ranking_mask), direction='DESCENDING', axis=-1), axis=-1) + 1
                sorted_test_scores, predicted_test_indices = tf.math.top_k(predicted_scores - ranking_mask, k=maxk) # [B, K]

                test_ground_truth = gather_dense(test_interaction_matrix, user_indices) # gather([U, I], [B]) -> [B, I]
                test_actual_positive_count = tf.reduce_sum(test_ground_truth, axis=-1) # actual positives per user | [B]
                for k in sorted(self.evaluation_cutoffs, reverse=True):
                    positions = tf.range(k, dtype=tf.float32) # [K]
                    predicted_test_indices_k = predicted_test_indices[:, :k] # [B, K]
                    # test_true_positives = tf.cast((predicted_test_rankings <= k) * test_ground_truth, tf.int32)
                    # test_true_positive_count = tf.reduce_sum(test_true_positives, axis=-1) # true_positives per user
                    test_true_positives = tf.gather(test_ground_truth, predicted_test_indices_k, batch_dims=-1) # this is equivalent to the commented lines above | | gather([B, I], [B, K]) -> [B, K]
                    test_true_positive_count = tf.reduce_sum(test_true_positives, axis=-1) # true positives per user | [B]

                    ## Hit Rate@k
                    # 1 if TP_u > 0 else 0
                    test_hit = tf.cast(test_true_positive_count > 0, tf.float32) # [B]
                    
                    ## Recall@k
                    # TP / (TP + FN) for u in U
                    test_recall = tf.math.divide_no_nan(test_true_positive_count, test_actual_positive_count) # [B]
                    
                    ## Precision@k
                    # TP / (TP + FP) for u in U
                    test_precision = tf.math.divide_no_nan(test_true_positive_count, k) # [B]
                    
                    ## MAP@k
                    # Precision(k)
                    test_precision_at_each_position = tf.cumsum(test_true_positives, axis=-1) / (positions + 1) # [B, K]
                    # Precision(k) * rel(k)
                    test_precision_at_each_recall_position = test_precision_at_each_position * test_true_positives # [B, K]
                    # Σ Precision(k) * rel(k) / total_recalled_items | total_recalled_items = TP
                    test_average_precision = tf.math.divide_no_nan(tf.reduce_sum(test_precision_at_each_recall_position, axis=-1), test_true_positive_count) # [B]
                    
                    ## NDCG@k
                    # log(k + 1, base=2)
                    log_discount = tf.math.log(positions + 2) / tf.math.log(2.0) # [K]
                    # Σ rel_k / log2(k + 1)
                    test_dcg = tf.reduce_sum(test_true_positives / log_discount, axis=-1) # [B]
                    # ideal_rel_k = 1 if k <= total_recalled_items else 0
                    test_ideal_true_positives = tf.cast(positions[tf.newaxis, :] < test_true_positive_count[:, tf.newaxis], tf.float32) # [B, K]
                    # Σ ideal_rel_k / log2(k + 1)
                    test_idcg = tf.reduce_sum(test_ideal_true_positives / log_discount, axis=-1) # [B]
                    # dcg / idcg
                    test_ndcg = tf.math.divide_no_nan(test_dcg, test_idcg) # [B]
                    
                    ## MRR@k
                    # Set non-positive positions to inf so reduce_min finds first positive
                    test_first_positive_positions = tf.where(test_true_positives==1, test_true_positives * (positions + 1), float('inf')) # [B, K]
                    test_reciprocal_ranks = 1 / tf.reduce_min(test_first_positive_positions, axis=-1) # [B]

                    self.test_hitrate_tracker[k].update_state(test_hit)
                    self.test_recall_tracker[k].update_state(test_recall)
                    self.test_precision_tracker[k].update_state(test_precision)
                    self.test_map_tracker[k].update_state(test_average_precision)
                    self.test_ndcg_tracker[k].update_state(test_ndcg)
                    self.test_mrr_tracker[k].update_state(test_reciprocal_ranks)
    
    def construct_interaction_history(
        self,
        dataset: tf.data.Dataset,
    ):
        interaction_history = []
        for step, batch in enumerate(dataset):
            user_ids = batch["user_id"]
            item_ids = batch["item_id"]
            user_indices = self.user_lookup_layer(user_ids)
            item_indices = self.item_lookup_layer(item_ids)
            interaction_history.append(tf.stack([user_indices, item_indices], axis=-1))
        
        interaction_history = tf.concat(interaction_history, axis=0)
        return interaction_history

    def construct_interaction_matrix(
        self,
        interaction_history: tf.Tensor,
        dtype: tf.dtypes.DType = tf.float32
    ) -> tf.sparse.SparseTensor:
        interaction_matrix = tf.sparse.SparseTensor(
            indices=interaction_history,
            values=tf.ones(shape=(len(interaction_history),), dtype=dtype),
            dense_shape=(self.user_lookup_layer.vocabulary_size(), self.item_lookup_layer.vocabulary_size())
        )

        return tf.sparse.reorder(interaction_matrix)
    
    def get_config(self):
        config: Dict[str, object] = super(MatrixFactorization, self).get_config()

        config.update({
            "features_meta": self.features_meta,
            "embedding_dimension_count": self.embedding_dimension_count,
            "l1_regularization": self.l1_regularization,
            "l2_regularization": self.l2_regularization,
            "evaluation_cutoffs": self.evaluation_cutoffs
        })

        return config
        