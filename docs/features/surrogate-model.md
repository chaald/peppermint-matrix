# Surrogate Model & Exploration Saturation Metric
**Date:** 2026-03-15

## Overview

A Random Forest surrogate model trained on completed W&B runs that serves two purposes:
1. **Predict** the expected evaluation score (e.g. `test_recall@10`) for any unexplored parameter combination
2. **Quantify exploration saturation** — a single metric that signals when a parameter sub-space has been sufficiently explored and it is time to move on

## Motivation

With 3382+ completed runs covering only ~7.58% of the joint hyperparameter space, there is no principled way to answer:
- Which unexplored configurations are most likely to outperform the current best?
- Has the current sub-space been explored enough, or is there still expected gain to be found?
- When should exploration shift to a different sub-space (e.g. a different model architecture or feature set)?

Without these answers, experiment decisions are driven by intuition, and compute is spent uniformly across the space rather than where expected value is highest.

## Design

### Surrogate Model

A **Random Forest Regressor** is fit on all completed runs:

- **Inputs (X):** hyperparameter config columns (e.g. `embedding_dimension`, `l1_regularization`, `l2_regularization`, `embedding_dropout_rate`, `shuffle`)
- **Target (y):** best checkpoint metric per run (e.g. `best:epoch/test_recall@10`), selected using the same scoring criterion as `wandb/sync.py`

The Random Forest naturally provides:
- $\hat{\mu}$ — mean prediction across all trees (expected score)
- $\hat{\sigma}$ — std across trees (epistemic uncertainty / how well-explored this region is)

No encoding tricks are needed — the tree ensemble handles discrete and boolean inputs natively.

### Exploration Saturation Metric (ESM)

A single scalar that answers: *"how close am I to the theoretical maximum of this sub-space?"*

Expressed as a percentage — **99.99%** means 4 nines, essentially saturated. The number of nines after the decimal point serves as an intuitive stop threshold.

$$\text{ESM} = \frac{\mu_{\text{best, observed}}}{\hat{\mu}_{\text{max}} + \beta \cdot \hat{\sigma}_{\text{max}}} \times 100\%$$

Where:
- $\mu_{\text{best, observed}}$ — the best score actually observed in completed runs
- $\hat{\mu}_{\text{max}}$ — the highest mean prediction across all unexplored cells
- $\hat{\sigma}_{\text{max}}$ — the surrogate uncertainty at that cell
- $\beta$ — same exploration weight as used in UCB (shared hyperparameter)

Using UCB as the denominator makes ESM **conservative**: it won't claim near-saturation until the surrogate is also confident there is nothing better left. As more top-predicted cells are explored and fail to beat the observed best, $\hat{\mu}_{\text{max}} + \beta \hat{\sigma}_{\text{max}}$ converges downward toward $\mu_{\text{best, observed}}$, driving ESM toward 100%.

**Behaviour:**
- Starts low (e.g. ~60–80%) when the space is largely unexplored
- Increases monotonically on average as promising configurations are tried
- Approaches 100% as the surrogate becomes confident the current best is near-optimal

**Intuitive stop thresholds:**

| ESM | Nines | Interpretation |
|---|---|---|
| 90% | 1 | Substantial gain likely remaining |
| 99% | 2 | Modest gain remaining |
| 99.9% | 3 | Marginal gain remaining |
| 99.99% | 4 | Essentially saturated — move on |

**Open question:** What defines the theoretical maximum?

The UCB-based denominator ($\hat{\mu}_{\text{max}} + \beta \hat{\sigma}_{\text{max}}$) is the current proposed definition, but alternatives remain under consideration:
1. **Surrogate mean only:** $\hat{\mu}_{\text{max}}$ — optimistic, may claim saturation too early
2. **Extrapolation from explored space:** fit a trend line (e.g. performance vs. log-scale regularization) and extrapolate — more conservative, does not require a surrogate
3. **UCB upper bound (current proposal):** $\hat{\mu}_{\text{max}} + \beta \hat{\sigma}_{\text{max}}$ — conservative, naturally accounts for surrogate uncertainty

The chosen definition should satisfy:
- **Monotonically increasing on average** as more of the best-predicted cells are explored
- **Bounded in [0%, 100%]**
- **Unitless and comparable** across different metrics, datasets, and model families

### Exploration Progress Metric

Complementary to ERG, a metric tracking how much of the **useful** space has been covered:

$$\text{Coverage}_{p} = \frac{|\{x \in \text{explored} : \hat{\mu}(x) \geq p\text{-th percentile of } \hat{\mu}_{\text{all}}\}|}{|\{x \in \text{all} : \hat{\mu}(x) \geq p\text{-th percentile of } \hat{\mu}_{\text{all}}\}|}$$

Using e.g. $p=75$ — "what fraction of the top-25% predicted configurations have been tried?" A high value here means the promising region is well-sampled even if overall coverage is low.

### When to Stop Exploring a Sub-Space

Stop when both conditions hold:
- ESM exceeds the target nines threshold (e.g. ≥ 99.9% = 3 nines)
- Coverage at the 75th percentile exceeds a defined threshold (e.g. > 80%)

## Known Limitations

### RF Extrapolation Bias at Performance Cliffs

The hyperparameter landscape contains sharp performance cliffs — e.g. `l1_regularization` collapses from ~0.035 mean score at `1e-7` to ~0.0008 at `1e-6`, a 40× drop across a single grid step. Random Forests extrapolate by averaging leaf values, so a config just outside the training boundary (e.g. the good side of a cliff that was held out) gets predicted as the average of its two nearest observed neighbours. This causes systematic underprediction for high-performing cliff-adjacent configs.

In practice this means UCB will assign those configs lower `μ̂` than their true value — but since the surrogate is trained on *all* data including both sides of both cliffs, the effect is small. It becomes significant only when the surrogate is used to score a config class it has genuinely never seen.

### σ̂ Collapses to Zero on Unseen Cliff Values

When an input value (e.g. `l1=-7.0`) is absent from the training set, all 256 trees route it to the same nearest leaf — every tree agrees, so σ̂ = 0. This means the UCB bonus is zeroed out precisely on the configs where exploration is most needed.

**Leave-cliff-out evaluation results** (`l1 ∈ {-7, -6}` or `l2 ∈ {-6, -5}` held out):
- Held-out R² = 0.917 — strong on average, but underprediction cluster visible for good-side cliff configs
- σ̂ distribution: test > train as expected, but with a spike at 0 for configs at exact unseen values

### Mitigation: Distance-to-Nearest-Explored Bonus

To compensate for σ̂ collapsing on truly unseen configs, the UCB score can be augmented with a novelty term based on the minimum Chebyshev distance to the nearest explored config in the discrete grid:

$$\text{UCB}_{\text{aug}}(x) = \hat{\mu}(x) + \beta \cdot \hat{\sigma}(x) + \gamma \cdot d_{\min}(x)$$

Where:
- $d_{\min}(x) = \min_{x' \in \text{explored}} \|x - x'\|_\infty$ — the furthest any single feature dimension is from a seen config
- $\gamma$ — a small weight (e.g. 0.001–0.01) that adds a tie-breaking bonus for novelty without dominating `μ̂`

This ensures configs at cliff boundaries that are equidistant in `μ̂` from safer interpolated options get a small boost, preventing the surrogate from perpetually avoiding unexplored grid cells with σ̂ = 0.

The distance is computed over the discrete `parameter_space` grid, so it is fast and does not require any model calls. Regularization features use their log-scale values, so a step from `-7` to `-6` counts as distance 1 regardless of the original linear-scale ratio.

This mitigation is **not yet implemented** — it is noted here as the recommended next step if the surrogate's exploration coverage is found to be insufficient in practice.

## Downstream Usage

The surrogate model built here is a **shared dependency** for the next planned feature:

- **[UCB-Guided Hyperparameter Search](ucb-hyperparameter-search.md)** — reuses the same Random Forest fit and `fetch_experiment_runs` data source to compute UCB scores for unexplored configurations. The ESM metric can also be logged at the start of each UCB worker call to track exploration progress over time.

The surrogate model feature should be implemented and validated first before the UCB search feature is built on top of it.

## Status
- [x] Planned
- [ ] Surrogate model training + prediction
- [ ] ESM metric definition (theoretical max approach TBD)
- [ ] Coverage at percentile metric
- [ ] Stop condition thresholds
- [ ] Integration with `wandb/sync.py` output as data source
