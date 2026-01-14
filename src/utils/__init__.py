import os

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
    
def preprocess_metric_aggregate(metrics_aggregate):
    return {key: f"{value:.4f}" for key, value in metrics_aggregate.items() if key in ["loss", "test_loss", "recall@10", "test_recall@10"]}