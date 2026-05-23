# Convergence Simulation — UCB vs Random Search from Cold Start
**Date:** 2026-05-22

## Overview

Simulate the UCB acquisition strategy from a cold start (zero historical runs) using an accuracy-oriented Random Forest oracle trained on all 3,400 existing yelp2018 MF runs. Compare convergence rates with random search baseline to estimate how many runs are needed to reach ESM 99% (2 nines).

## Motivation

The existing hyperparameter search space (`mf:unified.yaml`, 20,000 cells) is well-explored enough that UCB correctly exploits known-good regions near the l1=1e-7, l2=1e-7 sweet spot. This tells us the code works, but not whether the strategy converges faster from zero. Running a real experiment on a new dataset (gowalla) would take ~4-6 GPU-hours. A simulation costs nothing and gives direct answers.

**Question:** From a cold start on a 20K-cell categorical space with ~5 dimensions, how many UCB-guided runs to reach ESM 99% vs. random search?

## Design

### Two RF Models

The acquisition surrogate used in `model_based_parse_parameters` is deliberately configured for high tree diversity (`max_samples=0.1`, `max_features="sqrt"`, `min_samples_leaf=3`, 1024 trees) so that σ̂ is wide in unexplored space → UCB favors exploration. Using this same RF as the simulation oracle would create circular reasoning: the oracle's blind spots match the acquisition model's, so the simulation would underestimate convergence.

Instead, train two separate RF models:

| Model | Purpose | Hyperparameters | Rationale |
|-------|---------|----------------|-----------|
| **Oracle** | Simulated ground truth — predicts "true" test_recall@20 given a config | `max_features=None` (all features), `max_samples=1.0` (full bootstrap), deeper trees, 512 estimators | Maximize predictive accuracy (R²). No subsampling — each tree sees all data, minimizing variance. |
| **Acquisition** | UCB σ̂ estimation during simulated search | Current params: `max_samples=0.1`, `max_features="sqrt"`, `min_samples_leaf=3`, 1024 trees | Maximize tree diversity for meaningful uncertainty signal. Same config as production `model_based_parse_parameters`. |

Both models share the same `Log10Transformer` for regularization features and the same training data (3,400 completed runs with `best:epoch/test_recall@20` target).

### Monte Carlo Simulation Loop

For each trajectory (repeat N=10 with different random seeds):

1. **Initialize:** Empty explored set. Start with pure random sampling (acquisition surrogate needs ≥20 samples before fitting).

2. **Warm-up (runs 1–20):** Random sampling. Acquisition surrogate is not fit yet (fewer than 20 samples triggers the random fallback in `model_based_parse_parameters`).

3. **UCB phase (runs 21+):**
   - Fit acquisition surrogate on all explored configs
   - Enumerate full 20,000-cell grid
   - Compute UCB = μ̂ + β·σ̂ for each unexplored cell
   - Pick argmax (random tie-breaking)
   - Query oracle for the "true" score: oracle_μ̂ + ε where ε ~ N(0, oracle_σ̂)
   - Mark config as explored
   - Record ESM, Coverage@75, regret

4. **Random baseline (separate trajectories):**
   - Same warm-up (runs 1–20)
   - Runs 21+: random uniform sampling instead of UCB
   - Track same metrics

### Metrics Tracked Per Run

- **ESM** (Exploration Saturation Metric) — fraction of grid cells where acquisition σ̂ < threshold
- **Coverage@75** — fraction of top-25% predicted configs that have been explored
- **Simple regret** — difference between oracle's best score and best score found so far
- **Cumulative regret** — sum of regret over all runs

### Output

- Mean ± std ESM vs. runs curve for UCB and random (shaded ribbon for ±1σ across trajectories)
- Number of runs to reach ESM 99% and ESM 99.9% for each strategy
- Simple regret vs. runs curve
- Table summary at key milestones (runs = 50, 100, 200, 500, 1000)

## Implementation

### Files

- `notebooks/parameter_analysis/surrogate_model.ipynb` — trains the **acquisition RF** (existing, rerun to confirm metrics)
- `notebooks/parameter_analysis/oracle_model.ipynb` — validates **oracle RF** hyperparams (accuracy benchmark, no persistence)
- `notebooks/parameter_analysis/convergence_simulation.ipynb` — retrains both models from scratch using validated hyperparams, runs simulation

### Dependencies

All existing: `polars`, `numpy`, `sklearn`, `matplotlib`, `seaborn`, `scipy`. No GPU needed.

### Estimated Runtime

~30 minutes CPU time for 10 trajectories × 2 strategies × ~1000 simulated runs each.

## Task Status

> **Workflow:** Tasks are implemented one at a time in order. Each task is submitted for review before the next one begins. Do not proceed to the next task until the current one is reviewed and approved.

### Notebook 1 — `surrogate_model.ipynb` (Acquisition RF — verify baseline)

Existing notebook. Rerun all cells to confirm the current acquisition RF metrics still hold. No persistence — the final simulation notebook retrains both models from scratch.

- [x] Rerun all cells, confirm metrics match (OOB R² = 0.9946, cliff holdout R² = 0.9209)
- [x] Add validation cell: compute σ̂ for explored vs unexplored cells across the full grid
  - Metric: mean σ̂ on unexplored = **1.70×** mean σ̂ on explored (target: ≥ 1.5×) ✓
- [x] Document final acquisition hyperparams for the simulation notebook to use
  - `n_estimators=1024`, `max_features="sqrt"`, `max_samples=0.1`, `min_samples_leaf=1`

### Notebook 2 — `oracle_model.ipynb` (Oracle RF Validation)

New notebook to find and validate the right accuracy-oriented hyperparams. No persistence — the final hyperparams are retrained in the simulation notebook.

- [x] Create `notebooks/parameter_analysis/oracle_model.ipynb`
- [x] Load parquet data, filter MF runs, prepare training matrix X, y (same loading code as surrogate_model)
- [x] Train candidate **oracle RF** with accuracy-oriented hyperparams and iterate via hyperparameter sweep:
  - Final config: `n_estimators=512`, `max_features=None`, `max_samples=None`, `min_samples_leaf=3`, `oob_score=True`
  - OOB R² = 0.9989, cliff holdout R² = 0.9136, Spearman ρ = 0.9473
- [x] Add rank-based metrics on cliff holdout (Spearman ρ, NDCG@k) alongside R²/MAE
- [x] Validate per-tree σ̂ is tight across the full grid (oracle should be confident everywhere it has data)
  - Explored σ̂ = 0.000284, Unexplored σ̂ = 0.000310, ratio = 1.09x ✓
- [x] Document final hyperparams in title cell for the simulation notebook to use
  - `n_estimators=512`, `max_features=None`, `max_samples=None`, `min_samples_leaf=3`

### Notebook 3 — `convergence_simulation.ipynb` (Simulation)

Loads data, retrains oracle, then runs Monte Carlo simulation. Uses **real `model_based_parse_parameters`** for UCB config selection (reads/writes a simulated parquet). Computes ESM manually in-notebook for uniform tracking across both strategies.

- [x] Create `notebooks/parameter_analysis/convergence_simulation.ipynb`
- [x] Load parquet data, prepare training matrix (same loading code as notebook 2)
- [x] Build full 20,000-cell grid with `itertools.product`
- [x] Compute oracle's global best score (argmax over full grid) for regret calculation
  - Global optimum: `[512, 0.0, 1e-07, 1e-07, 0.0]` → score = 0.050739 (explored: YES)
- [ ] **Modify `model_based_parse_parameters`** to include `"esm"`, `"coverage_75"`, `"explored_percentage"`, `"predicted_mu"`, `"predicted_sigma"`, and `"ucb"` in its returned dict (backward-compatible `"meta"` sub-dict)
- [ ] Implement `run_trajectory(strategy, beta, n_runs, seed, output_dir)`:
  - Maintains its own **simulated parquet** under `output_dir/trajectory_{strategy}_{seed}.parquet` mirroring the real parquet schema (config columns + list column for target metric + `model` column)
  - **Warm-up** (runs 1–20): pick random config, query oracle, append to simulated parquet
  - **For UCB strategy** (runs 21+): call `model_based_parse_parameters(config, summary_path=...)` → returns config + ESM/Coverage in meta → query oracle for score → append to simulated parquet
  - **For random baseline** (runs 21+): pick random config instead, query oracle, append to simulated parquet
  - **ESM tracking** (both strategies): for UCB, read from return value; for random, call `model_based_parse_parameters` in "probe" mode every N steps to compute ESM without using its config pick
  - Saves per-run history to `output_dir/history_{strategy}_{seed}.parquet` with columns: `(run, esm, coverage_75, simple_regret, cumulative_regret, config_params...)`
  - Returns nothing — everything persisted to disk
- [ ] Parameterize β (default 1.0) to allow β-sweep later
- [ ] Run 5 trajectories for UCB (β=1.0) across seeds 0–4
- [ ] Run 5 trajectories for random baseline across seeds 0–4 (paired)
- [ ] Verify results can be loaded from disk and aggregated into a single DataFrame

**Modular design:** Each trajectory is fully self-contained. Adding more trajectories later is just calling `run_trajectory(...)` with new seeds. Trajectories can be run sequentially or in parallel without conflicts (each writes to a distinct path).

### Analysis & Visualization

- [ ] ESM vs runs curve: mean ± 1σ ribbon for UCB and random, with key thresholds marked (99%, 99.9%)
- [ ] Simple regret vs runs curve: best-found score vs oracle optimum
- [ ] Summary table at key milestones (runs = 50, 100, 200, 500, 1000): ESM mean ± std for each strategy
- [ ] Number of runs required to reach ESM 99% and ESM 99.9% for each strategy
- [ ] β-sweep comparison plot (if Phase 5 sweep was run)

### Review & Document

- [ ] Review results and draw conclusions
- [ ] Update feature index in `docs/features/README.md` with status
- [ ] Commit all notebook and doc changes
