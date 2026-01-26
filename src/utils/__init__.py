import os
from typing import Dict

from .config import load_yaml, store_yaml, load_config

def is_notebook() -> bool:
    "Check if the code is running in a Jupyter notebook environment."
    
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
    
def filter_vocabulary(features_meta: Dict[str, Dict]) -> Dict[str, Dict]:
    return {feature_name: {k: v for k, v in meta.items() if k not in ["vocabulary"]} for feature_name, meta in features_meta.items()}

def preprocess_metric_aggregate(metrics_aggregate: Dict[str, float]) -> Dict[str, str]:
    return {key: f"{value:.4f}" for key, value in metrics_aggregate.items() if key in ["loss", "test_loss", "recall@10", "test_recall@10"]}