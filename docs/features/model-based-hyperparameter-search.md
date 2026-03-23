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

1. Fetch all completed runs for the fixed parameters via `fetch_experiment_runs`
2. Fit the Random Forest surrogate on `(config → target_metric)`
3. Enumerate all cells in the categorical parameter space
4. Compute UCB score for **every cell** (explored and unexplored)
5. Return the config with the highest UCB score across the full grid

Steps 1–4 replace the current W&B round-trip + argmin logic in `exhaustive_parse_parameters`. The ~40 second startup cost is similar.

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

## Relationship to Surrogate Model Feature

This feature **depends on** the surrogate model feature ([surrogate-model.md](surrogate-model.md)):
- Reuses the same Random Forest fitting logic
- Reuses `fetch_experiment_runs` as the data source
- The ERG metric from the surrogate feature can be logged at the start of each model-based worker call to track exploration progress over time

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

**Notes:**
- The file is append-only; one row per `model_based_parse_params` call regardless of which worker produced it.
- `explored_mu` / `explored_sigma` reflect the state at call time — subsequent runs may change these values.
- The log enables post-hoc analysis of how the surrogate's beliefs evolved and whether explored configs were revisited.

## Open Questions

- **$\beta$ tuning:** Should $\beta$ decay over time (more exploitation as space fills up)? Or remain fixed?
- **Multi-worker race conditions:** Multiple workers will independently pick the same top UCB cell if run in parallel. Need a locking mechanism or a "batch UCB" strategy (pick top-$k$ diverse cells for $k$ workers). The current 60-second stagger partially mitigates this but doesn't solve it.
- **Target metric:** Should UCB optimise a single metric or a weighted composite (same format as `--sorting_criterion` in `sync.py`)?

## Status
- [x] Planned
- [ ] `model_based_parse_params` skeleton + arg wiring (`--model_based_beta`, `--model_based_target`, `--model_based_estimator_count`)
- [ ] Surrogate pipeline fit (`RegularizationLogTransformer` + `RandomForestRegressor`)
- [ ] Full-grid UCB scoring and argmax selection
- [ ] Fallback: cold start (no runs → random)
- [ ] Fallback: sparse data (< 20 runs → random with warning)
- [ ] Decision log append (`hyperparameter_search.log.csv`)
- [ ] Multi-worker deduplication strategy
- [ ] Integration tests
