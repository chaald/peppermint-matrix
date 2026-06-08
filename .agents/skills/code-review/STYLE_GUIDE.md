# Coding Style Guide

This guide defines the coding style for this project. The code review subagent checks for these rules.

## Imports

Grouped: standard library → third-party → local, with blank lines between groups.

```python
# Good
import os
import itertools
from datetime import datetime

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor

from src.utils.config import parse_score_metric
from src.visualization import set_themes

# Bad — mixed groups, no blank lines
import os
import polars as pl
from src.utils.config import parse_score_metric
import numpy as np
```

## Function Signatures

Multi-line function signatures must use **one parameter per line** (not hanging indent).

```python
# Good — one per line
def format_value(
    value: float,
    metric: str,
    currency_metrics: Optional[Set[str]] = None,
) -> str:

# Bad — hanging indent
def format_value(value: float, metric: str,
                 currency_metrics: Optional[Set[str]] = None) -> str:
```

## DataFrame Naming

Do **not** use a `df_` prefix. Use descriptive names instead.

```python
# Good
live_runs = fetch_experiment_runs(...)
parquet_runs = pl.read_parquet(...)
train_dataframe = ...
full_dataframe = ...

# Bad
df_live_runs = ...
df_parquet = ...
```

## NumPy Array Naming

Use `_vector` suffix for NumPy 1-D arrays derived from DataFrames, not `_np`.

```python
# Good
explored_mu_vector = explored_mu.to_numpy()

# Bad
explored_mu_np = explored_mu.to_numpy()
```

## Inline Comments

No inline comments (`# ...`) unless the code genuinely needs explanation that can't be expressed through clearer naming or structure. Docstrings are required for new functions, methods, and classes. Existing legacy comments are left as-is unless the code is being actively modified.

```python
# Bad — obvious code commented
# Increment counter
count += 1

# Good — no comment needed
count += 1
```

## Type Annotations

Required for function parameters and return types. Use `Optional[Type]` for nullable parameters, not `Union[Type, None]` in new code.

```python
# Good
def fetch_runs(
    model: str,
    embedding_dimension: Optional[int] = None,
) -> polars.DataFrame:

# Bad — missing annotations
def fetch_runs(model, embedding_dimension = None):
```

## Falsy Guards

Use `is None` / `is not None` for nullable numeric values, not truthiness checks. Truthiness treats `0`, `0.0`, `False`, and empty collections as falsy, which silently drops legitimate values.

```python
# Good
"score": round(score, 4) if score is not None else None,

# Bad — drops 0.0
"score": round(score, 4) if score else None,
```

## Dictionary Ordering

Use ordered dicts (Python 3.7+ insertion-order preservation) for CSV column sequences. Group related keys together and order them to tell a coherent story.

```python
# Good
decision_metadata = {
    "run_id": run_id,
    "start_time": start_time,
    "beta": beta,
    "selected_config": config,
    "score": score,
    "coverage@75": coverage_75,
}

# Bad — arbitrary order
decision_metadata = {
    "score": score,
    "run_id": run_id,
    "beta": beta,
}
```
