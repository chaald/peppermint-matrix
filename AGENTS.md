# AGENTS.md

This file contains guidance for AI agents working in this repository.

---

## Documenting New Features

Whenever a new feature is being developed, create a documentation file for it in `docs/features/`.

### Steps

1. Create `docs/features/<feature-name>.md` using the template in [`docs/features/README.md`](docs/features/README.md).
2. Add a corresponding row to the **Feature Index** table in [`docs/features/README.md`](docs/features/README.md).

### Feature Index (in `docs/features/README.md`)

Keep this table up to date as features are added:

| Feature | File | Date Added | Status |
|---------|------|------------|--------|
| Example | [example.md](docs/features/example.md) | 2026-03-15 | In progress |

---

## Fetching Previous Runs and Configurations from W&B

### Primary method — `fetch_experiment_runs` in `src/utils/config.py`

Use this whenever you need to programmatically check which runs have already been completed and what configurations they used.

```python
from src.utils.config import fetch_experiment_runs

# Get all runs for a given model
runs = fetch_experiment_runs({"config.model": "matrix_factorization"})

# Narrow down by any combination of fixed config values
runs = fetch_experiment_runs({
    "config.model": "matrix_factorization",
    "config.embedding_dimension": 64,
    "config.l2_regularization": 1e-6,
})

# Filter by run state (top-level W&B field, no config. prefix)
runs = fetch_experiment_runs({
    "config.model": "matrix_factorization",
    "state": "finished",
})

# runs is a polars.DataFrame — one row per matching run
print(runs.columns)           # all config fields + run_id, run_name
print(runs.shape)             # (n_runs, n_columns)
runs.select("run_id", "run_name", "embedding_dimension", "l2_regularization")
```

**What it does:**
- Queries W&B via GraphQL. Filter keys are passed through as-is — the caller is responsible for the correct prefix (`config.` for hyperparameter fields, none for top-level W&B fields like `state`)
- Returns a `polars.DataFrame` with one row per matching run and all config parameters as columns
- Paginates automatically (256 runs per page)
- Does **not** fetch metric history — configs only (fast)

**When to use this:**
- Before proposing a new experiment — check if the configuration already exists
- To understand what hyperparameter values have been tried
- To count how many runs used a specific parameter combination

---

---

## Coding Style

### DataFrame naming

Do **not** use a `df_` prefix for DataFrame variables. Use descriptive names instead.

```python
# Good
live_runs = fetch_experiment_runs(...)
parquet_runs = pl.read_parquet(...)
full_dataframe = ...
train_dataframe = ...

# Bad
df_live_runs = fetch_experiment_runs(...)
df_parquet = pl.read_parquet(...)
```

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
