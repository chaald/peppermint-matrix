# UCB-Guided Hyperparameter Search
**Date:** 2026-03-15

## Overview

Replace the current exhaustive/random selection in `hyperparameter_search.py` with a **UCB (Upper Confidence Bound)** acquisition function backed by the surrogate model. Instead of picking the least-explored combination or sampling randomly, each new run is the configuration with the highest UCB score across unexplored cells.

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

A new `--method=ucb` option in `hyperparameter_search.py`, alongside the existing `random`, `exhaustive`, and `wandb` methods.

The corresponding resolver in `src/utils/config.py` (`ucb_parse_parameters`) follows the same interface as `exhaustive_parse_parameters`:
- Takes a `parameters_config` dict
- Returns a single resolved config dict
- Can be called once per worker, just like the existing methods

### Workflow per Worker Call

1. Fetch all completed runs for the fixed parameters via `fetch_experiment_runs`
2. Fit the Random Forest surrogate on `(config → target_metric)`
3. Enumerate all cells in the categorical parameter space
4. Identify unexplored cells (not in completed runs)
5. Compute UCB score for each unexplored cell
6. Return the config with the highest UCB score

Steps 1–5 replace the current W&B round-trip + argmin logic in `exhaustive_parse_parameters`. The ~40 second startup cost is similar.

### New CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--method=ucb` | — | Enable UCB-guided search |
| `--ucb_beta` | `1.0` | Exploration weight $\beta$ in the UCB formula |
| `--ucb_target` | `test_recall@10` | Target metric column to optimise (must exist in completed run data) |
| `--ucb_n_estimators` | `200` | Number of trees in the Random Forest surrogate |

### Handling Edge Cases

- **No completed runs yet:** fall back to random sampling (surrogate cannot be fit)
- **Too few runs to fit reliably (e.g. < 20):** fall back to random sampling with a warning
- **All cells explored:** fall back to the cell with lowest run count (same as exhaustive)
- **Tie-breaking:** if multiple cells have equal UCB scores, pick randomly among them

## Relationship to Surrogate Model Feature

This feature **depends on** the surrogate model feature ([surrogate-model.md](surrogate-model.md)):
- Reuses the same Random Forest fitting logic
- Reuses `fetch_experiment_runs` as the data source
- The ERG metric from the surrogate feature can be logged at the start of each UCB worker call to track exploration progress over time

## Usage

```bash
# UCB-guided search with default beta
python hyperparameter_search.py \
    --method=ucb \
    --config=configs/hyperparameter_search/mf:elasticnet.yaml \
    --nworker=4 \
    --nruns=8 \
    --ucb_target=test_recall@10

# More exploratory (higher beta)
python hyperparameter_search.py \
    --method=ucb \
    --config=configs/hyperparameter_search/mf:elasticnet.yaml \
    --nworker=4 \
    --nruns=8 \
    --ucb_beta=2.0 \
    --ucb_target=test_recall@10
```

## Open Questions

- **$\beta$ tuning:** Should $\beta$ decay over time (more exploitation as space fills up)? Or remain fixed?
- **Multi-worker race conditions:** Multiple workers will independently pick the same top UCB cell if run in parallel. Need a locking mechanism or a "batch UCB" strategy (pick top-$k$ diverse cells for $k$ workers). The current 60-second stagger partially mitigates this but doesn't solve it.
- **Target metric:** Should UCB optimise a single metric or a weighted composite (same format as `--sorting_criterion` in `sync.py`)?

## Status
- [ ] Planned
- [ ] `ucb_parse_parameters` in `src/utils/config.py`
- [ ] `--method=ucb` and related args in `hyperparameter_search.py`
- [ ] Fallback logic for cold start / all-explored cases
- [ ] Multi-worker deduplication strategy
- [ ] Integration test
