# peppermint-matrix

A research toolkit for training and evaluating collaborative filtering recommendation system models, with integrated hyperparameter search and experiment tracking via Weights & Biases.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Datasets](#datasets)
- [Models](#models)
- [Training Pipeline](#training-pipeline)
- [Configuration](#configuration)
- [Training a Model](#training-a-model)
- [Hyperparameter Search](#hyperparameter-search)
- [Evaluation Metrics](#evaluation-metrics)
- [Experiment Tracking](#experiment-tracking)

---

## Overview

**peppermint-matrix** implements collaborative filtering models for top-K item recommendation. Given a user's past interaction history, the models learn to rank items by predicted preference using implicit feedback. The training objective is **Bayesian Personalized Ranking (BPR)** with negative sampling.

Key capabilities:
- Matrix Factorization (MF) with L1/L2 regularization and embedding dropout
- BPR loss with guaranteed-disjoint negative sampling
- Configurable ranking evaluation at multiple cutoffs (Hit Rate, Recall, Precision, MAP, NDCG, MRR)
- Three-tier configuration system (defaults → YAML file → CLI args)
- Parallel hyperparameter search via W&B sweeps, random search, or exhaustive grid search
- Experiment tracking and run summarization via W&B

---

## Project Structure

```
peppermint-matrix/
├── main.py                      # Training entry point
├── hyperparameter_search.py     # Parallel hyperparameter search entry point
├── pyproject.toml               # Project dependencies (managed via uv)
│
├── configs/
│   ├── default.yaml             # Base default configuration
│   ├── hyperparameter_search/   # Sweep configs (W&B / exhaustive / random)
│   │   ├── mf:baseline.yaml
│   │   ├── mf:elasticnet.yaml
│   │   ├── mf:elasticnet_dropout.yaml
│   │   ├── mf:embedding_dropout.yaml
│   │   └── mf:lasso.yaml
│   ├── single_runs/             # Pre-configured single-run configs
│   │   └── mf:baseline.yaml
│   └── training_queue/          # Batch run configs
│
├── dataset/
│   ├── yelp2018/                # Yelp 2018 benchmark dataset
│   ├── amazon-book/             # Amazon Book benchmark dataset
│   └── gowalla/                 # Gowalla benchmark dataset
│
├── src/
│   ├── constant.py              # Project-wide constants (PROJECT_NAME)
│   ├── models/
│   │   ├── matrix_factorization.py   # MF model implementation
│   │   └── lightgcn.py               # LightGCN (in development)
│   ├── losses/
│   │   └── bayesian_personalized_ranking.py  # BPR loss
│   ├── sampler/                 # BayesianSampler for negative sampling
│   ├── preprocessing/           # Data loading and feature metadata
│   ├── baremetal/               # Low-level TF ops (gather_dense)
│   ├── utils/                   # Config loading, YAML/JSON I/O, helpers
│   ├── tracker/                 # Experiment tracking utilities
│   └── visualization/           # Visualization helpers
│
├── models/                      # Saved model artifacts (created at runtime)
├── notebooks/                   # Exploratory Jupyter notebooks
├── tests/                       # pytest test suite
└── wandb/
    ├── sync.py                  # Fetch and summarize W&B run results
    └── adjust.py                # W&B run adjustment utilities
```

---

## Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management and requires Python 3.12+.

```bash
# Install dependencies
uv sync

# Run any script via uv
uv run python main.py --help

# Activate environtment for interactive use
source .venv/bin/activate
```

**GPU note:** On Linux, CUDA and cuDNN are included as dependencies (`nvidia-cuda-runtime-cu12`, `nvidia-cudnn-cu12`). Set the library path before running if needed:

```bash
export LD_LIBRARY_PATH=$(uv run python -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])")/lib:$LD_LIBRARY_PATH
```

---

## Datasets

Datasets are stored as text files where each line represents one user and their interacted items:

```
<user_id> <item_id_1> <item_id_2> ...
```

Supported benchmark datasets (place under `dataset/`):

| Dataset      | Path                  |
|--------------|-----------------------|
| Yelp 2018    | `dataset/yelp2018/`   |
| Amazon Book  | `dataset/amazon-book/`|
| Gowalla      | `dataset/gowalla/`    |

Each dataset directory should contain `train.txt` and `test.txt`. The dataset used is currently hardcoded to `yelp2018` in `main.py`.

---

## Models

### Matrix Factorization (`matrix_factorization`)

Classic collaborative filtering model that learns separate user and item embedding vectors. Predictions are computed as the dot product of a user embedding and an item embedding. Features:

- `IntegerLookup` layers to handle arbitrary user/item ID spaces
- L1 and/or L2 embedding regularization (ElasticNet, Lasso, Ridge)
- Embedding dropout for regularization
- Configurable embedding dimension

### LightGCN *(in development)*

Graph-based collaborative filtering model. Implementation is not yet complete.

---

## Training Pipeline

Each training run follows these steps:

1. **Config resolution** — merge `default.yaml`, optional YAML config file, and CLI arguments (CLI takes highest priority)
2. **Tracker initialization** — initialize W&B run (or skip if `--tracker=disabled`)
3. **Data loading** — load `train.txt` and `test.txt` into `tf.data.Dataset`
4. **Feature metadata** — build vocabulary and count unique users/items from training data
5. **Model initialization** — instantiate model with resolved config
6. **Model compilation** — Adam optimizer + BPR loss + BayesianSampler
7. **Training loop** — fit over epochs with optional early stopping and W&B metric logging
8. **Model saving** — optionally save to `models/{model}/{sweep_id}/{run_id}/` (`.keras` + `config.yaml`)

---

## Configuration

Configuration is resolved with three priority levels (highest → lowest):

1. **CLI arguments** — override everything
2. **YAML config file** (`--config`) — overrides defaults
3. **`configs/default.yaml`** — base fallback values

### Full Configuration Reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | str | `matrix_factorization` | Model architecture |
| `embedding_dimension` | int | `16` | Size of user/item embedding vectors |
| `max_epoch` | int | `64` | Maximum training epochs |
| `batch_size` | int | `16384` | Training batch size |
| `shuffle` | bool | `false` | Shuffle training data each epoch |
| `learning_rate` | float | `0.01` | Adam optimizer learning rate |
| `l1_regularization` | float | `0.0` | L1 regularization on embeddings |
| `l2_regularization` | float | `0.0` | L2 regularization on embeddings |
| `embedding_dropout_rate` | float | `0.0` | Dropout rate on embedding layers |
| `log_freq` | str/int | `epoch` | W&B metric logging frequency |
| `evaluation_cutoffs` | list[int] | `[2, 10, 50]` | Top-K cutoffs for ranking metrics |
| `early_stopping` | bool | `false` | Enable early stopping |
| `early_stopping_monitor` | str | `test_recall@10` | Metric to monitor |
| `early_stopping_mode` | str | `max` | `max` or `min` |
| `early_stopping_patience` | int | `0` | Epochs to wait before stopping |
| `random_seed` | int | `42` | Seed for reproducibility |
| `store_model` | bool | `false` | Save model weights after training |
| `tracker` | str | `wandb` | Tracking backend (`wandb` or `disabled`) |

---

## Training a Model

### Manual parameters via CLI

```bash
python main.py \
    --model=matrix_factorization \
    --embedding_dimension=64 \
    --max_epoch=10 \
    --batch_size=16384 \
    --shuffle \
    --learning_rate=0.01 \
    --l2_regularization=1e-6 \
    --log_freq=epoch \
    --evaluation_cutoffs 2 10 50 \
    --random_seed=171
```

### From a pre-configured YAML file

```bash
python main.py \
    --method=exhaustive \
    --config=configs/hyperparameter_search/mf:elasticnet.yaml \
    --embedding_dimension=1024
```

The `--method` argument controls how the YAML config is resolved into a single run config:
- `random` — sample randomly from defined distributions (default)
- `exhaustive` — take the next unexplored combination from the grid

### Skip W&B tracking

```bash
python main.py --tracker=disabled --model=matrix_factorization --max_epoch=5
```

---

## Hyperparameter Search

`hyperparameter_search.py` launches parallel workers to run multiple training jobs across a sweep. Each worker runs independently.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--nworker` | `1` | Number of parallel worker processes |
| `--nruns` | `1` | Number of runs per worker |
| `--method` | `wandb` | Search method: `wandb`, `random`, `exhaustive` |
| `--sweep_id` | `None` | Resume an existing W&B sweep by ID |
| `--config` | `None` | Path to YAML sweep config |

### Start a new W&B sweep

```bash
python hyperparameter_search.py \
    --method=wandb \
    --config=configs/hyperparameter_search/mf:elasticnet.yaml \
    --nworker=4 \
    --nruns=16
```

### Resume an existing W&B sweep

```bash
python hyperparameter_search.py \
    --method=wandb \
    --sweep_id=<sweep_id> \
    --nworker=4 \
    --nruns=16
```

### Exhaustive grid search (no W&B resolution)

```bash
python hyperparameter_search.py \
    --method=exhaustive \
    --config=configs/hyperparameter_search/mf:elasticnet.yaml \
    --nworker=4 \
    --nruns=8
```

**Note:** Workers are staggered with a 60-second delay between starts to avoid race conditions during exhaustive search state recording.

### Available sweep configs

| Config | Description |
|---|---|
| `mf:baseline.yaml` | Fixed hyperparameters baseline |
| `mf:elasticnet.yaml` | Search over L1 + L2 regularization and embedding size |
| `mf:lasso.yaml` | Search over L1 regularization only |
| `mf:embedding_dropout.yaml` | Search over embedding dropout rate |
| `mf:elasticnet_dropout.yaml` | Search over ElasticNet + dropout combined |

---

## Evaluation Metrics

All metrics are computed at each configured cutoff `K` (default: `[2, 10, 50]`) on both train and test sets after every epoch.

| Metric | Name pattern | Description |
|---|---|---|
| Hit Rate | `{split}_hitrate@K` | Fraction of users with at least one relevant item in top-K |
| Recall | `{split}_recall@K` | Average fraction of relevant items retrieved in top-K |
| Precision | `{split}_precision@K` | Average precision of the top-K list |
| MAP | `{split}_map@K` | Mean Average Precision at K |
| NDCG | `{split}_ndcg@K` | Normalized Discounted Cumulative Gain at K |
| MRR | `{split}_mrr@K` | Mean Reciprocal Rank at K |

`split` is either `train` or `test`. Example: `test_ndcg@10`, `train_recall@50`.

---

## Experiment Tracking

### W&B Integration

All runs are logged to the W&B project `peppermint-matrix` by default. Each run records:
- Full config
- Per-epoch train/test metrics
- Sweep membership (if part of a sweep)

Disable tracking with `--tracker=disabled`.

### Syncing and summarizing W&B results

`wandb/sync.py` fetches run history from the W&B API and exports a summary as a Parquet file, with configurable weighted scoring to identify the best checkpoint per run.

```bash
python wandb/sync.py \
    --model=matrix_factorization \
    --process_count=8 \
    --threads_per_process=32 \
    --sorting_criterion epoch/test_hitrate@20:0.5 epoch/test_ndcg@20:0.25 \
    --output_path=wandb/summary.parquet \
    --max_retries=3
```

`--sorting_criterion` accepts one or more `metric:weight` pairs to define a weighted composite score. For each run, the epoch with the highest composite score is selected as the best checkpoint, and its metrics are used to represent that run in the summary.