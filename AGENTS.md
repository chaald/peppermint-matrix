# AGENTS.md

## Repo Facts
- Python 3.12+; use `uv` for env/dependency management (`uv sync`, then `uv run ...`).
- `main.py` is the single-run trainer.
- `hyperparameter_search.py` launches parallel sweep workers.
- `main.py` currently always loads `dataset/yelp2018/train.txt` and `dataset/yelp2018/test.txt`; dataset choice is hardcoded there, not driven by config.

## Config And Runs
- Config precedence is `configs/default.yaml` -> `--config` YAML -> CLI flags.
- All default values live in `configs/default.yaml`. CLI args (`main.py` parser) are only for final overrides — they have `default=None` so they're skipped when not explicitly passed.
- `--method=random|exhaustive` only matters when the YAML has a `parameters` block.
- `exhaustive` search uses W&B history through `src.utils.config.fetch_experiment_runs(...)` to pick the least-explored categorical config.
- Use `--tracker=disabled` for local runs that should not touch W&B.

## Commands
- `uv run python main.py --help`
- `uv run python main.py --tracker=disabled --model=matrix_factorization --max_epoch=5`
- `uv run python hyperparameter_search.py --method=random --config=<path>`
- `uv run pytest`
- `uv run pytest tests/test_main.py` or `uv run pytest tests/test_main.py -k <pattern>` for focused checks

## Workflow
- Load `.env` before running any command: `set -a; source .env; set +a` (sets `LD_LIBRARY_PATH` for GPU/cuDNN + Jupyter vars).
- Check prior runs with `src.utils.config.fetch_experiment_runs(filters)` before starting a new experiment.
- `hyperparameter_search.py` staggers worker startup by 60 seconds for `exhaustive`; keep that unless you verify the race is gone.
- Saved models go under `models/{model}/{sweep_id}/{run_id}/`; `models/` and `wandb/` are ignored by git.
- When working on a new feature, consult `docs/features/README.md`, create `docs/features/<feature-name>.md`, and update the Feature Index.
- Start JupyterLab from the repo `.venv` with `.venv/bin/jupyter lab --no-browser --port=5601 --ip=127.0.0.1 --IdentityProvider.token="$JUPYTER_TOKEN"`.
- Required env vars for the local setup are `JUPYTER_URL`, `JUPYTER_TOKEN`, and `MCP_TOKEN`.
- Committed package versions for MCP-backed notebook editing are `jupyterlab==4.5.6`, `notebook==7.5.5`, `jupyter-collaboration==4.3.0`, `jupyter-mcp-tools>=0.1.4`, and `datalayer-pycrdt==0.12.17`.
- If notebook cell insert or edit calls fail on `/api/collaboration/session/...`, restart JupyterLab from the repo `.venv` and reconnect the notebook through MCP.
- After the collaboration stack is healthy, Jupyter MCP can create notebooks, insert cells, and execute them without manually opening the notebook in the JupyterLab web UI.
- To create a new notebook, add a valid `.ipynb` file under `notebooks/`, then call `use_notebook` with `mode=create` and a live kernel id (usually `python3`).
- For notebook work, prefer `insert_execute_code_cell`, `read_notebook`, `read_cell`, and `overwrite_cell_source` over manual browser edits.

## Feature Docs

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

### Function signatures

Multi-line function signatures must use **one parameter per line** (not hanging indent):

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

Imports must be grouped: standard library → third-party → local, with blank lines between groups.

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
>>>>>>> origin/hafidh/scafolding
