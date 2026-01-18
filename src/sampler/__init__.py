import numpy as np
import tensorflow as tf

from typing import Iterable, Dict

class BayesianSampler:
    """
    Samples negative items for users ensuring that sampled items are not in the user's positive interactions.
    Ensuring the training and test sample are disjoint as described in the original paper [1].

    References:
    [1] Steffen Rendle, Christoph Freudenthaler, Zeno Gantner and Lars Schmidt-Thieme.
        "BPR: Bayesian Personalized Ranking from Implicit Feedback"
        https://arxiv.org/abs/1205.2618
    """

    def __init__(self, item_set: Iterable[int], user_items: Dict[int, Iterable[int]]):
        self.item_set = np.array(item_set) # stored as numpy array for efficient sampling
        self.user_items = user_items

    def sample(self, user_ids: Iterable[int]) -> tf.Tensor:
        random_negatives = np.random.choice(self.item_set, size=len(user_ids))
        for i, uid in enumerate(np.array(user_ids)):
            while random_negatives[i] in self.user_items[uid]:
                random_negatives[i] = np.random.choice(self.item_set)
        random_negatives = tf.constant(random_negatives)
        return random_negatives