# Model-Based Hyperparameter Search
**Date:** 2026-03-15 (Updated 2026-05-16)

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

Each call to `model_based_parse_params` fits a **fresh surrogate** on the latest parquet data before picking a config. This is critical because workers run concurrently — by the time a worker picks its next config, other workers may have finished runs and appended them to the parquet. The surrogate must reflect that new signal to make an informed decision.

1. **Read `wandb/summary.parquet`** directly (local file, instant) — no `fetch_experiment_runs` needed for metric data. The parquet contains per-epoch history as list columns plus config fields.
2. **Compute target metric** from the list columns on the fly:
   - Single target: `target_run = max(run["epoch/test_recall@20"])` — true max across all epochs
   - Weighted composite: `composite = [sum(w * m[i] for m, w in target) for i in range(n_epochs)]; target = max(composite)`
3. **Aggregate duplicate configs** (same categorical params) with `.mean()` of the target
4. If < 20 runs have valid target values → fall back to random sampling with a warning
5. Fit the Random Forest surrogate pipeline on `(config → target)`
6. Enumerate all cells in the categorical parameter space
7. Compute UCB score for **every cell** using per-tree predictions for $\hat{\mu}$ and $\hat{\sigma}$
8. Compute and log ESM + Coverage@75 to stdout
9. Append the selected config's decision metadata to `hyperparameter_search.log.csv`
10. Return the config with the highest UCB score (random tie-breaking)

Scoring the full grid (rather than filtering to unexplored cells first) means the surrogate can re-run a config if it genuinely has the highest UCB — e.g. a config with observed high variance, or one where the surrogate's uncertainty is elevated due to few repeated runs.

**Note on param types:** All searchable params use `categorical` in practice (even `embedding_dimension` is enumerated as discrete values). Non-search utility params like `random_seed` may use `int_uniform` but they're sampled randomly and don't affect the surrogate — the grid is purely categorical.

### Data Source — `wandb/summary.parquet` with On-the-Fly Target Computation

**No live W&B API calls for metric data.** The parquet is the single source of truth for surrogate training. This avoids both:
- **Last-epoch bias** (93.6% of runs degrade at last epoch — validated in `notebooks/sandbox/summary_metrics_validation.ipynb`)
- **Sync.py latency** (regenerating parquet from W&B history is slow)

The parquet stores every epoch's metrics as **list columns** (e.g., `epoch/test_recall@20 = [0.01, 0.02, 0.045, 0.044, ...]`). The surrogate reads these lists and computes the true per-run max for whatever target is specified — no `sync.py` re-run needed when the target changes.

**Target is specified as the raw metric key** as it appears in the parquet, e.g., `epoch/test_recall@20`. Weighted composites follow the same format as `sync.py`'s `--sorting_criterion`:

```
--model_based_target "epoch/test_recall@20:0.7 epoch/test_ndcg@20:0.3"
```

### Keeping the Parquet Up to Date — Append from `main.py`

`hyperparameter_search.py` runs multiple workers concurrently. Each worker's decision should reflect the latest state — otherwise a worker might pick a config that another worker just finished evaluating. This is the same constraint that motivated the original live-API approach, but we solve it differently: every run appends its result to the local parquet on completion, so subsequent workers on the same machine always see the freshest data.

`wandb/sync.py` was run once (historically) to populate the parquet with ~3390 runs. Going forward, every run appends its own result on completion:

The surrogate only needs config fields (to identify the categorical cell) and `epoch/*` list columns (to compute the target max). Everything else (`_step`, `score`, `best:*`, `gpu_type`, etc.) is ignored by the surrogate. The append writes just what's needed:

```python
# In main(), after model.fit() and before returning:
import polars as pl

row = {
    "run_id": run_id,
    "run_name": run_name,
    "sweep_id": sweep_id,
    "model": config["model"],
    **{k: v for k, v in config.items() if k in CONFIG_FIELDS},
    "train_loss": model.train_loss_history,
    "test_loss": model.test_loss_history,
}
for k in config["evaluation_cutoffs"]:
    row[f"epoch/test_hitrate@{k}"] = model.test_hitrate_history[k]
    row[f"epoch/test_recall@{k}"] = model.test_recall_history[k]
    row[f"epoch/test_precision@{k}"] = model.test_precision_history[k]
    row[f"epoch/test_map@{k}"] = model.test_map_history[k]
    row[f"epoch/test_ndcg@{k}"] = model.test_ndcg_history[k]
    row[f"epoch/test_mrr@{k}"] = model.test_mrr_history[k]
    row[f"epoch/train_hitrate@{k}"] = model.train_hitrate_history[k]
    row[f"epoch/train_recall@{k}"] = model.train_recall_history[k]
    row[f"epoch/train_precision@{k}"] = model.train_precision_history[k]
    row[f"epoch/train_map@{k}"] = model.train_map_history[k]
    row[f"epoch/train_ndcg@{k}"] = model.train_ndcg_history[k]
    row[f"epoch/train_mrr@{k}"] = model.train_mrr_history[k]

existing = pl.read_parquet("wandb/summary.parquet")
combined = pl.concat([existing, pl.DataFrame([row])], how="diagonal")
combined.write_parquet("wandb/summary.parquet")
```

`how="diagonal"` aligns columns by name — any column not in `row` (e.g. `_step`, `score`, `best:*`, `gpu_type`) becomes null. These are safe to leave null because (a) the surrogate doesn't read them, and (b) `sync.py` replaces the full parquet with complete data the next time it runs.

### Test MAP/NDCG/MRR Histories on Model Object

The model's `evaluate_kernel` already computes MAP, NDCG, and MRR. The history lists were initialized but never appended — now fixed with the 3 missing appends in `matrix_factorization.py:evaluate()`.

### Distributed Worker Staleness

Workers on different machines won't see each other's newly appended rows in the local parquet. Accepted as minor staleness: new runs from other workers will appear in the parquet's next append on the local machine. The surrogate may be slightly behind the global state, but this decays to zero as all workers converge.

## New CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--method=model_based` | — | Enable model-based search |
| `--model_based_beta` | `1.0` | Exploration weight $\beta$ in the UCB formula |
| `--model_based_target` | `epoch/test_recall@20` | Target metric key in the parquet. Supports weighted composite: `"epoch/test_recall@20:0.7 epoch/test_ndcg@20:0.3"` |
| `--model_based_estimator_count` | `1024` | Number of trees in the Random Forest surrogate |

## Handling Edge Cases

- **No completed runs yet:** fall back to random sampling (surrogate cannot be fit)
- **Too few runs to fit reliably (e.g. < 20):** fall back to random sampling with a warning
- **Tie-breaking:** if multiple cells have equal UCB scores, pick randomly among them

## Implementation Details

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
- **Feature names:** derived from the categorical parameters in the config
- **Duplicate-run aggregation:** multiple runs with the same categorical config are `.mean()`-aggregated before fitting

### Decision Log — Repo Root

`hyperparameter_search.log.csv` is written to the repository root directory. The file is append-only with one row per `model_based_parse_params` call. Concurrent workers append independently (file-level append atomicity is sufficient on Linux for single-line CSV writes).

### ESM Logging

At the start of each `model_based_parse_params` call, after fitting the surrogate and computing the full-grid UCB scores, the ESM and Coverage@75 metrics are computed and:
- **Printed to stdout** — provides a running trace of exploration saturation as workers execute
- **Written to `hyperparameter_search.log.csv`** — the `esm` and `coverage_75` columns on each row capture the global surrogate state at the moment the config was selected, enabling post-hoc analysis of saturation progression over time

## Relationship to Surrogate Model Feature

This feature **depends on** the surrogate model feature ([surrogate-model.md](surrogate-model.md)):
- Reuses the same Random Forest fitting logic (ported from `notebooks/parameter_analysis/surrogate_model.ipynb`)
- The ESM metric from the surrogate feature is logged at the start of each model-based worker call to track exploration progress over time

## Usage

```bash
# Model-based search with default beta, target test_recall@20
python hyperparameter_search.py \
    --method=model_based \
    --config=configs/hyperparameter_search/mf:elasticnet.yaml \
    --nworker=4 \
    --nruns=8 \
    --model_based_target="epoch/test_recall@20"

# Weighted composite target
python hyperparameter_search.py \
    --method=model_based \
    --config=configs/hyperparameter_search/mf:elasticnet.yaml \
    --nworker=4 \
    --nruns=8 \
    --model_based_target="epoch/test_recall@20:0.7 epoch/test_ndcg@20:0.3"

# More exploratory (higher beta)
python hyperparameter_search.py \
    --method=model_based \
    --config=configs/hyperparameter_search/mf:elasticnet.yaml \
    --nworker=4 \
    --nruns=8 \
    --model_based_beta=2.0
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

## Task Status

> **Workflow:** Tasks are implemented one at a time in order. Each task is submitted for review before the next one begins. Do not proceed to the next task until the current one is reviewed and approved.

### Foundation — Data Pipeline

- [x] Planned
- [x] Implementation details documented
- [x] Validate `summaryMetrics` reliability — notebook in `notebooks/sandbox/summary_metrics_validation.ipynb`. **Verdict: use parquet list columns with on-the-fly max computation.**
- [x] `wandb/sync.py` run once — parquet populated with full epoch history for ~3390 historical runs
- [x] **Fix model history gap** — added missing `test_map_history`, `test_ndcg_history`, `test_mrr_history` appends
- [ ] **Parquet append in `main.py`** — after `model.fit()`, collect epoch histories from model object, build parquet-compatible row, append to `wandb/summary.parquet`
- [ ] **Verify parquet append** — run one training step with `--tracker=disabled --max_epoch=1 --store_model=false` and confirm the new row appears in `wandb/summary.parquet` without corrupting the file

### Implementation — Surrogate & UCB

- [ ] CLI arg wiring — add `--model_based_beta`, `--model_based_target`, `--model_based_estimator_count` to `hyperparameter_search.py`; extend validation; pass through `compile_config` → `load_config`
- [ ] `model_based_parse_params` skeleton — parameter parsing (fixed/categorical/random split), extend `load_config` with `**kwargs` dispatch, read parquet + compute target from list columns
- [ ] Port surrogate pipeline — `RegularizationLogTransformer` + `RandomForestRegressor` Pipeline from notebook into `src/utils/config.py`
- [ ] Full-grid UCB scoring — enumerate categorical space, compute per-tree $\hat{\mu}$/$\hat{\sigma}$, argmax with random tie-breaking
- [ ] Fallback: cold start (no runs → random)
- [ ] Fallback: sparse data (< 20 runs → random with warning)
- [ ] ESM + Coverage@75 logging to stdout

### Logging & Polish

- [ ] Decision log append (`hyperparameter_search.log.csv` at repo root)
- [ ] Multi-worker deduplication strategy
- [ ] Integration tests
