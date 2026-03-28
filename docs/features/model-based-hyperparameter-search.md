# Model-Based Hyperparameter Search
**Date:** 2026-03-15

## Overview

Replace the current exhaustive/random selection in `hyperparameter_search.py` with a **UCB (Upper Confidence Bound)** acquisition function backed by the surrogate model. Instead of picking the least-explored combination or sampling randomly, each new run is the configuration with the highest UCB score across all possible configurations.

## Motivation

The current search methods have complementary weaknesses:
- **Exhaustive** — uniform coverage, ignores observed performance signal entirely
- **Random** — no exploitation of what has already been learned, high variance in quality of selected configs

After 3382+ runs, there is substantial signal in the observed data. UCB uses this signal to balance:
- **Exploitation** — run configs predicted to score well (high $\hat{\mu}$)
- **Exploration** — run configs with high uncertainty (high $\hat{\sigma}$), since the surrogate may be wrong

This is warm-started SMAC (Sequential Model-based Algorithm Configuration), using Random Forest instead of a Gaussian Process.

## Design

### UCB Score

For each unexplored configuration $x$:

$$\text{UCB}(x) = \hat{\mu}(x) + \beta \cdot \hat{\sigma}(x)$$

Where:
- $\hat{\mu}(x)$ — mean prediction across all surrogate trees
- $\hat{\sigma}(x)$ — std across surrogate trees (proxy for uncertainty)
- $\beta$ — exploration weight, tunable (higher = more exploration)

The next configuration to run is $\arg\max_x \text{UCB}(x)$ over all unexplored cells.

### Integration Point

A new `--method=model_based` option in `hyperparameter_search.py`, alongside the existing `random`, `exhaustive`, and `wandb` methods.

The corresponding resolver in `src/utils/config.py` (`model_based_parse_params`) follows the same interface as `exhaustive_parse_parameters`:
- Takes a `parameters_config` dict
- Returns a single resolved config dict
- Can be called once per worker, just like the existing methods

### Workflow per Worker Call

1. Fetch all **finished** runs for the fixed parameters via `fetch_experiment_runs` (GraphQL with `summaryMetrics` + `state: "finished"` filter)
2. Extract the target metric value from each run's `summaryMetrics` JSON; aggregate duplicate configs with `.mean()`
3. If < 20 runs have valid target metric values → fall back to random sampling with a warning
4. Fit the Random Forest surrogate pipeline (`RegularizationLogTransformer` → `RandomForestRegressor`) on `(config → target_metric)`
5. Enumerate all cells in the categorical parameter space
6. Compute UCB score for **every cell** (explored and unexplored) using per-tree predictions for $\hat{\mu}$ and $\hat{\sigma}$
7. Compute and log ESM + Coverage@75 to stdout
8. Append the selected config's decision metadata to `hyperparameter_search.log.csv`
9. Return the config with the highest UCB score across the full grid (random tie-breaking)

Steps 1–9 replace the current W&B round-trip + argmin logic in `exhaustive_parse_parameters`. The ~40 second startup cost is similar.

Scoring the full grid (rather than filtering to unexplored cells first) means the surrogate can re-run a config if it genuinely has the highest UCB — e.g. a config with observed high variance, or one where the surrogate's uncertainty is elevated due to few repeated runs.

### New CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--method=model_based` | — | Enable model-based search |
| `--model_based_beta` | `1.0` | Exploration weight $\beta$ in the UCB formula |
| `--model_based_target` | `test_recall@10` | Target metric column to optimise (must exist in completed run data) |
| `--model_based_estimator_count` | `1024` | Number of trees in the Random Forest surrogate |

### Handling Edge Cases

- **No completed runs yet:** fall back to random sampling (surrogate cannot be fit)
- **Too few runs to fit reliably (e.g. < 20):** fall back to random sampling with a warning
- **Tie-breaking:** if multiple cells have equal UCB scores, pick randomly among them

## Implementation Details

### Data Source — Live W&B API with Metric Fetching

Each call to `model_based_parse_params` queries the W&B API **directly** (not the local `wandb/summary.parquet`). This ensures the surrogate sees newly completed runs in real time, which is critical because `hyperparameter_search.py` runs multiple workers concurrently and each worker's decision should reflect the latest state.

**Approach:** Extend the existing `fetch_experiment_runs` GraphQL query to also request the `summaryMetrics` field on each run node. This field is a JSON string containing the run's summary metrics (including per-epoch metrics logged by `WandbMetricsLogger`). The target metric value is extracted from this JSON per run.

The GraphQL query also adds a `state: "finished"` filter so only completed runs are returned. This avoids training the surrogate on partial/crashed runs.

```graphql
query Runs(...) {
    project(name: $project, entityName: $entity) {
        runs(first: 256, after: $cursor, filters: $filters) {
            edges {
                node {
                    id
                    name
                    config
                    summaryMetrics   # ← new field
                }
            }
            ...
        }
    }
}
```

**Performance:** This is the same single paginated GraphQL query used by `fetch_experiment_runs` today, with one additional field per run. No extra API calls. The ~40 second latency matches the existing `exhaustive_parse_parameters` cost.

**Best-epoch vs. last-epoch:** `summaryMetrics` contains W&B's summary values (typically the last logged value per metric). This is problematic for runs where the loss exploded — the last logged metric values may be `NaN` or severely degraded, while the actual best performance occurred at an earlier epoch. Since the current `main.py` does not enforce early stopping or checkpointing by default, this is a real concern.

For the initial implementation, `summaryMetrics` is used for speed. A **validation task** is included in the status checklist to compare `summaryMetrics` values against `history()`-derived best-epoch values (from `wandb/summary.parquet`) across all finished runs. The outcome determines whether:
- `summaryMetrics` is reliable enough (most runs don't explode, or NaN rows are few enough to filter out)
- The approach needs to be revised to fetch `history()` per run, or to use the local parquet cache as a fallback for metric values

This validation should be done **before** the surrogate pipeline is ported, since the quality of the training signal directly affects UCB decisions.

**Target metric naming:** The `--model_based_target` CLI argument specifies the metric key as it appears in `summaryMetrics` (e.g. `epoch/test_recall@10`). This differs from the `best:epoch/test_recall@10` naming used in `wandb/summary.parquet` — the `best:` prefix is a `sync.py` convention, not a W&B native key.

### Interface — Extending `load_config`

Extend `load_config` in `src/utils/config.py` to accept `**kwargs` that are forwarded to the method-specific resolver:

```python
def load_config(config_path: str, method: Literal["random", "exhaustive", "model_based"] = "random", **kwargs) -> Dict:
    ...
    if method == "model_based":
        current_run_config.update(model_based_parse_params(config["parameters"], **kwargs))
    ...
```

`model_based_parse_params` follows the same interface as `exhaustive_parse_parameters`:
- **Input:** `parameters_config` dict (the `parameters:` block from the YAML config) + keyword arguments (`beta`, `target_metric`, `estimator_count`)
- **Output:** a single resolved config dict (one value per parameter)

The caller chain: `compile_config(args)` → `load_config(config_path, method, **model_based_kwargs)` → `model_based_parse_params(parameters_config, ...)`.

`compile_config` in `main.py` extracts the model-based CLI args and passes them as kwargs to `load_config`.

### Surrogate Pipeline — Ported from Notebook

The `RegularizationLogTransformer` and sklearn `Pipeline` are ported from `notebooks/parameter_analysis/surrogate_model.ipynb` into `src/utils/config.py` (co-located with `model_based_parse_params`).

Key components:
- **`RegularizationLogTransformer`** — `BaseEstimator` + `TransformerMixin` that log₁₀-transforms `l1_regularization` and `l2_regularization` columns (zero → sentinel `-15.0`, else `round(log10(x), 1)`)
- **`Pipeline`** — `[("log_reg", RegularizationLogTransformer(...)), ("rf", RandomForestRegressor(...))]`
- **RF hyperparameters:** `n_estimators` from `--model_based_estimator_count` (default 1024), `max_features="sqrt"`, `max_samples=0.1`, `min_samples_leaf=3`, `n_jobs=-1`, `random_state=42`
- **Feature names:** derived from the categorical parameters in the config (same as `exhaustive_parse_parameters` does for its `free_categorical_parameters`)
- **Duplicate-run aggregation:** multiple runs with the same categorical config are `.mean()`-aggregated before fitting. This is consistent with the notebook approach

### State Filtering in `fetch_experiment_runs`

`state` is treated as a top-level W&B filter key (not a `config.*` field) and can be passed directly in the `filters` dict. Any key in `filters` that matches a known top-level W&B field (`state`) is passed through without the `config.` prefix; all other keys are prefixed as before. This keeps the interface simple — callers own the full filter dict:

```python
# As used by model_based_parse_params
fetch_experiment_runs(
    {**fixed_parameters, "state": "finished"},
    include_summary_metrics=True,
)`
```

Existing callers that don't pass `state` are unaffected.

### Decision Log — Repo Root

`hyperparameter_search.log.csv` is written to the repository root directory. The file is append-only with one row per `model_based_parse_params` call. Concurrent workers append independently (file-level append atomicity is sufficient on Linux for single-line CSV writes).

### ESM Logging

At the start of each `model_based_parse_params` call, after fitting the surrogate and computing the full-grid UCB scores, the ESM and Coverage@75 metrics are computed and:
- **Printed to stdout** — provides a running trace of exploration saturation as workers execute
- **Written to `hyperparameter_search.log.csv`** — the `esm` and `coverage_75` columns on each row capture the global surrogate state at the moment the config was selected, enabling post-hoc analysis of saturation progression over time

## Relationship to Surrogate Model Feature

This feature **depends on** the surrogate model feature ([surrogate-model.md](surrogate-model.md)):
- Reuses the same Random Forest fitting logic (ported from `notebooks/parameter_analysis/surrogate_model.ipynb`)
- Reuses `fetch_experiment_runs` as the data source (extended with `summaryMetrics` and state filtering)
- The ESM metric from the surrogate feature is logged at the start of each model-based worker call to track exploration progress over time

## Usage

```bash
# Model-based search with default beta
python hyperparameter_search.py \
    --method=model_based \
    --config=configs/hyperparameter_search/mf:elasticnet.yaml \
    --nworker=4 \
    --nruns=8 \
    --model_based_target=test_recall@10

# More exploratory (higher beta)
python hyperparameter_search.py \
    --method=model_based \
    --config=configs/hyperparameter_search/mf:elasticnet.yaml \
    --nworker=4 \
    --nruns=8 \
    --model_based_beta=2.0 \
    --model_based_target=test_recall@10
```

## Decision Log — `hyperparameter_search.log.csv`

Each call to `model_based_parse_params` appends one row to `hyperparameter_search.log.csv`, capturing both the surrogate's state and the decision made at that moment.

| Column | Type | Description |
|---|---|---|
| `start_time` | datetime | UTC timestamp when the worker call started |
| `<param>` | varies | One column per hyperparameter in the search space (e.g. `embedding_dimension`, `l1_regularization`, …) |
| `explored` | bool | Whether this config had at least one completed run at the time of the call |
| `explored_mu` | float \| null | Mean observed score across completed runs for this config (`null` if unexplored) |
| `explored_sigma` | float \| null | Std of observed scores across completed runs for this config (`null` if unexplored or only one run) |
| `predicted_mu` | float | Surrogate mean prediction $\hat{\mu}$ for this config |
| `predicted_sigma` | float | Surrogate uncertainty $\hat{\sigma}$ (std across trees) for this config |
| `ucb` | float | Predicted UCB score: $\hat{\mu} + \beta \cdot \hat{\sigma}$ |
| `esm` | float | Exploration Saturation Metric (%) at the time of the call |
| `coverage_75` | float | Coverage at the 75th percentile (%) at the time of the call |

**Notes:**
- The file is append-only; one row per `model_based_parse_params` call regardless of which worker produced it.
- `explored_mu` / `explored_sigma` reflect the state at call time — subsequent runs may change these values.
- The log enables post-hoc analysis of how the surrogate's beliefs evolved and whether explored configs were revisited.

## Open Questions

- **$\beta$ tuning:** Should $\beta$ decay over time (more exploitation as space fills up)? Or remain fixed?
- **Multi-worker race conditions:** Multiple workers will independently pick the same top UCB cell if run in parallel. Need a locking mechanism or a "batch UCB" strategy (pick top-$k$ diverse cells for $k$ workers). The current 60-second stagger partially mitigates this but doesn't solve it.
- **Target metric:** Should UCB optimise a single metric or a weighted composite (same format as `--sorting_criterion` in `sync.py`)?

## Task Status

> **Workflow:** Tasks are implemented one at a time in order. Each task is submitted for review before the next one begins. Do not proceed to the next task until the current one is reviewed and approved.

- [x] Planned
- [x] Implementation details documented
- [x] Extend `fetch_experiment_runs` — add `summaryMetrics` to GraphQL query + optional `state` filter
- [ ] Validate `summaryMetrics` reliability — notebook in `notebooks/parameter_analysis/` comparing `summaryMetrics` values against best-epoch values from `wandb/summary.parquet` across all finished runs; quantify how many runs have NaN or degraded last-epoch scores due to loss explosion; decide whether to proceed with `summaryMetrics` or revise the data source
- [ ] CLI arg wiring — add `--model_based_beta`, `--model_based_target`, `--model_based_estimator_count` to `hyperparameter_search.py`; extend validation; pass through `compile_config` → `load_config`
- [ ] `model_based_parse_params` skeleton — parameter parsing (fixed/categorical/random split), extend `load_config` with `**kwargs` dispatch
- [ ] Port surrogate pipeline — `RegularizationLogTransformer` + `RandomForestRegressor` Pipeline from notebook into `src/utils/config.py`
- [ ] Full-grid UCB scoring — enumerate categorical space, compute per-tree $\hat{\mu}$/$\hat{\sigma}$, argmax with random tie-breaking
- [ ] Fallback: cold start (no runs → random)
- [ ] Fallback: sparse data (< 20 runs → random with warning)
- [ ] ESM + Coverage@75 logging to stdout
- [ ] Decision log append (`hyperparameter_search.log.csv` at repo root)
- [ ] Multi-worker deduplication strategy
- [ ] Integration tests
