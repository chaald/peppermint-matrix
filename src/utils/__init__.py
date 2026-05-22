import os
from typing import Dict

from .config import load_yaml, store_yaml, load_config
from .core import get_kernel_id, is_notebook
    
def filter_vocabulary(features_meta: Dict[str, Dict]) -> Dict[str, Dict]:
    return {feature_name: {k: v for k, v in meta.items() if k not in ["vocabulary"]} for feature_name, meta in features_meta.items()}

def preprocess_metric_aggregate(metrics_aggregate: Dict[str, float]) -> Dict[str, str]:
    return {key: f"{value:.4f}" for key, value in metrics_aggregate.items() if key in ["loss", "test_loss", "recall@10", "test_recall@10"]}