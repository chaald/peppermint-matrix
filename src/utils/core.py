import os
from typing import Dict

def get_kernel_id() -> str:
    "Get the current Jupyter notebook kernel ID."

    import ipykernel
    connection_file = ipykernel.get_connection_file()
    return os.path.basename(connection_file).split("-", 1)[1].split(".")[0]


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