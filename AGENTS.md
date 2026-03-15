# AGENTS.md

This file contains guidance for AI agents working in this repository.

---

## Fetching Previous Runs and Configurations from W&B

### Primary method — `fetch_experiment_runs` in `src/utils/config.py`

Use this whenever you need to programmatically check which runs have already been completed and what configurations they used.

```python
from src.utils.config import fetch_experiment_runs

# Get all runs for a given model
runs = fetch_experiment_runs({"model": "matrix_factorization"})

# Narrow down by any combination of fixed config values
runs = fetch_experiment_runs({
    "model": "matrix_factorization",
    "embedding_dimension": 64,
    "l2_regularization": 1e-6,
})

# runs is a polars.DataFrame — one row per matching run
print(runs.columns)           # all config fields + run_id, run_name
print(runs.shape)             # (n_runs, n_columns)
runs.select("run_id", "run_name", "embedding_dimension", "l2_regularization")
```

**What it does:**
- Queries W&B via GraphQL, filtering by `config.<key> = value` for each key in the `filters` dict
- Returns a `polars.DataFrame` with one row per matching run and all config parameters as columns
- Paginates automatically (256 runs per page)
- Does **not** fetch metric history — configs only (fast)

**When to use this:**
- Before proposing a new experiment — check if the configuration already exists
- To understand what hyperparameter values have been tried
- To count how many runs used a specific parameter combination

---

### Direct W&B API access

For ad-hoc querying, use the W&B public API directly:

```python
import wandb
from src.constant import PROJECT_NAME

api = wandb.Api()

# List all runs for the project
runs = api.runs(f"{api.default_entity}/{PROJECT_NAME}")

for run in runs:
    print(run.id, run.name, run.state)
    print(run.config)         # dict of hyperparameters
    print(run.summary)        # final/summary metrics
    df = run.history()        # per-step metric history as a DataFrame
```

Useful filters:
```python
# Filter by state and config value
runs = api.runs(
    f"{api.default_entity}/{PROJECT_NAME}",
    filters={
        "state": "finished",
        "config.model": "matrix_factorization",
        "config.embedding_dimension": 64,
    }
)
```
