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

- `notebooks/parameter_analysis/surrogate_model.ipynb` — trains the **acquisition RF** (existing, just rerun to confirm outputs match)
- `notebooks/parameter_analysis/oracle_model.ipynb` — validates **oracle RF** hyperparams (accuracy benchmark, no persistence needed)
- `notebooks/parameter_analysis/convergence_simulation.ipynb` — retrains both models from scratch using validated hyperparams, runs simulation

### Dependencies

All existing: `polars`, `numpy`, `sklearn`, `matplotlib`, `seaborn`, `scipy`, `joblib`. No GPU needed.

### Estimated Runtime

~30 minutes CPU time for 10 trajectories × 2 strategies × ~1000 simulated runs each.

## Task Status

> **Workflow:** Tasks are implemented one at a time in order. Each task is submitted for review before the next one begins. Do not proceed to the next task until the current one is reviewed and approved.

### Notebook 1 — `surrogate_model.ipynb` (Acquisition RF)

Existing notebook. Rerun to confirm outputs match (OOB R² ≥ 0.98, cliff holdout R² ≥ 0.91).

- [ ] Rerun all cells, confirm metrics match
- [ ] Add persistence: save trained pipeline to `models/acquisition_rf.joblib` via `joblib.dump`

### Notebook 2 — `oracle_model.ipynb` (Oracle RF Validation)

New notebook to find and validate the right accuracy-oriented hyperparams. No persistence — the final hyperparams are retrained in the simulation notebook.

- [ ] Create `notebooks/parameter_analysis/oracle_model.ipynb`
- [ ] Load parquet data, filter MF runs, prepare training matrix X, y (same loading code as surrogate_model)
- [ ] Train candidate **oracle RF** with accuracy-oriented hyperparams:
  - `n_estimators=512`, `max_features=None`, `max_samples=None` (default bootstrap), `min_samples_leaf=1`, `oob_score=True`
- [ ] Evaluate on cliff holdout: report OOB R² and held-out R² (target: match or exceed 0.986 / 0.920)
- [ ] If accuracy is insufficient, iterate: try `min_samples_leaf=1`, more estimators, or different `max_features` values
- [ ] Validate per-tree σ̂ is tight across the full grid (oracle should be confident everywhere it has data)
- [ ] Document final hyperparams in a clear cell at the top for the simulation notebook to use

### Notebook 3 — `convergence_simulation.ipynb` (Simulation)

Focused notebook — loads data, retrains both models with validated hyperparams, runs Monte Carlo simulation.

- [ ] Create `notebooks/parameter_analysis/convergence_simulation.ipynb`
- [ ] Load parquet data, prepare training matrix (same loading code as the two model notebooks)
- [ ] Train **oracle RF** using hyperparams validated in Notebook 2
- [ ] Train **acquisition RF** using hyperparams from `surrogate_model.ipynb` (standard production config)
- [ ] Build full 20,000-cell grid with `itertools.product`
- [ ] Compute oracle's global best score (argmax over full grid) for regret calculation
- [ ] Implement `simulate_trajectory(strategy, beta, n_runs, random_seed)`:
  - **Warm-up** (runs 1–20): pure random (acquisition model not fit yet)
  - **For UCB strategy** (runs 21+): fit acquisition RF on explored set → enumerate full grid → compute UCB = μ̂ + β·σ̂ → pick argmax (tie-breaking) → query oracle (oracle_μ̂ + ε, ε ~ N(0, oracle_σ̂)) → mark explored → record ESM, Coverage@75, regret
  - **For random baseline** (runs 21+): same loop, but pick randomly instead of UCB; still fit acquisition RF to compute ESM for fair comparison
- [ ] Return history: per-run `(esm, coverage_75, simple_regret, cumulative_regret, config_chosen)`
- [ ] Parameterize β (default 1.0) to allow β-sweep later
- [ ] Run 10 trajectories for UCB (β=1.0) across different random seeds
- [ ] Run 10 trajectories for random baseline (same seeds for paired comparison)
- [ ] Run β-sweep (optional): trajectories for β ∈ {0.25, 0.5, 1.0, 2.0, 4.0} at 5 seeds each
- [ ] Persist results to parquet for safe checkpointing

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
