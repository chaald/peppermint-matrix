import os
import re
import csv
import json
import time
import yaml
import math
import random
import itertools
from datetime import datetime, timezone
import wandb
import pprint
import numpy as np
import polars as pl

from wandb.sdk.internal.internal_api import gql
from wandb.apis.public import Run
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from typing import Literal, Union, List, Dict, Tuple
from src.constant import PROJECT_NAME

def store_json(data, filepath):
    with open(filepath, "w") as file:
        json.dump(data, file, indent=4)

def load_json(filepath) -> Dict:
    with open(filepath, 'r') as file:
        data = json.load(file)
    
    return data

def store_yaml(data, filepath):
    with open(filepath, "w") as file:
        yaml.dump(data, file)

def load_yaml(filepath) -> Dict:
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
    
    return data

def parse_scientific_notation(value: Union[str, List[str]]) -> float:
    """
    Parse a string in scientific notation to a float if applicable.
    """

    scientific_notation_pattern = re.compile(r'^-?\d+\.?\d*[eE][+-]?\d+$')
    if isinstance(value, list):
        return [float(v) if isinstance(v, str) and scientific_notation_pattern.match(v) else v for v in value]
    else:
        return float(value) if isinstance(value, str) and scientific_notation_pattern.match(value) else value

def parse_parameters(parameters_config: Dict) -> Dict:
    parsed_parameters = {}
    for parameter, parameter_config in parameters_config.items():
        if not isinstance(parameter_config, dict):
            parsed_parameters[parameter] = parameter_config
        elif "value" in parameter_config:
            parsed_parameters[parameter] = parameter_config["value"]
        elif "distribution" in parameter_config:
            distribution = parameter_config["distribution"]
            if distribution == "constant":
                parsed_parameters[parameter] = parameter_config["value"]
            elif distribution == "categorical":
                parsed_parameters[parameter] = random.choice(parameter_config["values"])
            elif distribution == "int_uniform":
                parsed_parameters[parameter] = random.randint(parameter_config["min"], parameter_config["max"])
            elif distribution == "uniform": # float uniform
                parsed_parameters[parameter] = random.uniform(parameter_config["min"], parameter_config["max"])
            elif distribution == "log_uniform":
                log_min = parameter_config["min"]
                log_max = parameter_config["max"]
                sampled_log_value = random.uniform(log_min, log_max)
                parsed_parameters[parameter] = math.exp(sampled_log_value)
            else:
                raise ValueError(f"Unsupported distribution type: {distribution} for parameter: {parameter}")
        else:
            raise ValueError(f"Invalid parameter configuration for {parameter}: {parameter_config}")

    return parsed_parameters

def parse_config(config_str: str) -> Dict:
    run_config = json.loads(config_str)
    run_config = parse_parameters({k: v for k, v in run_config.items() if k != "_wandb"})
    return run_config

def fetch_experiment_runs(
    filters: Dict[str, Union[int, float, str]],
    include_summary_metrics: bool = False,
) -> pl.DataFrame:
    """Query the Weights & Biases API for experiment runs that match the
    provided *filters*, parse each run's configuration, and return the
    results as a ``polars.DataFrame`` — one row per run.

    How filters work
    ----------------
    The *filters* dictionary is passed directly to the W&B GraphQL API.
    Two kinds of keys are supported:

    - **Config (hyperparameter) fields** — prefix the key with ``config.``.
      For example, ``{"config.model": "matrix_factorization"}`` only returns
      runs whose ``model`` config key equals ``"matrix_factorization"``.
    - **Top-level W&B fields** — no prefix. For example, ``{"state": "finished"}``
      filters runs that completed successfully. Other useful top-level fields
      include ``sweep`` (sweep ID) and ``tags``.

    Multiple filters are combined with AND logic. Numerical fields support
    comparison operators via W&B's query syntax (e.g.
    ``{"config.learning_rate": {"$gte": 0.001}}``).

    Pagination
    ----------
    The W&B GraphQL API returns a maximum of 256 runs per page. This function
    handles pagination automatically — it follows ``hasNextPage`` / ``endCursor``
    tokens in a loop until all matching runs have been collected. You don't
    need to manage cursors yourself.

    What is fetched
    ---------------
    By default only **config** values are retrieved (run ID, display name,
    and every key in the run's ``config`` dictionary). Config values that are
    lists or dicts are JSON-serialised to strings so they fit in a flat
    DataFrame column.

    Per-step metric **history** is **not** fetched — this keeps the query
    fast even for thousands of runs. If you need the best-epoch logic (as
    ``wandb/sync.py`` does), use ``sync.py`` to produce ``summary.parquet``
    instead.

    Summary metrics (optional)
    --------------------------
    Set ``include_summary_metrics=True`` to also pull each run's **final**
    metric values (the last value logged for each metric key, e.g. loss,
    recall, NDCG). These are stored in W&B's ``summaryMetrics`` field.
    Internal keys starting with ``_wandb`` are stripped. The ``_runtime``,
    ``_step``, and ``_timestamp`` keys are renamed to ``runtime``, ``step``,
    and ``timestamp`` respectively for convenience.

    Use cases
    ---------
    - Checking whether a given hyperparameter combination has already been
      tried before launching a new experiment.
    - Counting how many runs exist for a given model or sweep.
    - Building a quick overview table of completed runs and their final
      metrics for analysis or reporting.

    Parameters
    ----------
    filters:
        Dictionary of filter criteria passed directly to the W&B GraphQL API.
    include_summary_metrics:
        If ``True``, include each run's summary metrics (final values) as
        columns in the returned DataFrame.

    Returns
    -------
    pl.DataFrame
        One row per matching run with columns for run_id, run_name, node_id,
        every config parameter, and optionally summary metrics.

    Example
    -------
    >>> runs = fetch_experiment_runs({
    ...     "config.model": "matrix_factorization",
    ...     "config.embedding_dimension": 64,
    ...     "state": "finished",
    ... })
    >>> runs.shape
    (42, 12)
    """
    api = wandb.Api() # Initialize Weights & Biases API, used for fetching run data

    summary_metrics_field = "summaryMetrics" if include_summary_metrics else ""
    query = f"""
        query Runs($project: String!, $entity: String!, $cursor: String, $filters: JSONString) {{
            project(name: $project, entityName: $entity) {{
                runs(first: 256, after: $cursor, filters: $filters) {{
                    edges {{
                        node {{
                            id
                            name
                            displayName
                            config
                            {summary_metrics_field}
                        }}
                        cursor
                    }}
                    pageInfo {{
                        hasNextPage
                        endCursor
                    }}
                }}
            }}
        }}
    """
    query = gql(query)

    experiment_runs = []
    cursor = None
    filters = json.dumps(filters)

    while True:
        variables = {
            "project": PROJECT_NAME,
            "entity": api.default_entity,
            "cursor": cursor,
            "filters": filters
        }
        
        result = api.client.execute(query, variables)
        runs_data = result["project"]["runs"]
        
        for edge in runs_data["edges"]:
            current_run = edge["node"]

            run_record = {
                "node_id": current_run["id"],
                "run_id": current_run["name"],
                "run_name": current_run["displayName"],
                **parse_config(current_run['config']),
            }

            if include_summary_metrics and current_run.get("summaryMetrics"):
                summary = json.loads(current_run["summaryMetrics"])
                rename_map = {"_runtime": "runtime", "_step": "step", "_timestamp": "timestamp"}
                for key, value in summary.items():
                    if key == "_wandb":
                        continue
                    run_record[rename_map.get(key, key)] = value

            experiment_runs.append(run_record)
        
        if not runs_data["pageInfo"]["hasNextPage"]:
            print(f"Fetched {len(experiment_runs)} runs...")
            break

        cursor = runs_data["pageInfo"]["endCursor"]
        print(f"Fetched {len(experiment_runs)} runs...")

    return pl.DataFrame(experiment_runs, infer_schema_length=None)

def parse_parameter_categories(parameters_config: Dict) -> Tuple[
    Dict[str, Union[int, float, str]],
    Dict[str, List[Union[int, float, str]]],
    Dict[str, Dict[str, Union[int, float, str]]],
    Dict[str, type],
]:
    fixed_parameters: Dict[str, Union[int, float, str]] = {}
    free_categorical_parameters: Dict[str, List[Union[int, float, str]]] = {}
    free_random_parameters: Dict[str, Dict[str, Union[int, float, str]]] = {}
    categorical_dtypes: Dict[str, type] = {}
    for parameter, parameter_config in parameters_config.items():
        if not isinstance(parameter_config, dict):
            fixed_parameters[parameter] = parse_scientific_notation(parameter_config)
        elif "value" in parameter_config:
            fixed_parameters[parameter] = parse_scientific_notation(parameter_config["value"])
        elif "distribution" in parameter_config:
            distribution = parameter_config["distribution"]
            if distribution == "constant":
                fixed_parameters[parameter] = parse_scientific_notation(parameter_config["value"])
            elif distribution == "categorical":
                values = parse_scientific_notation(parameter_config["values"])
                free_categorical_parameters[parameter] = values
                categorical_dtypes[parameter] = type(values[0])
            elif distribution in ["int_uniform", "uniform", "log_uniform"]:
                free_random_parameters[parameter] = parameter_config
            else:
                raise ValueError(f"Unsupported distribution type: {distribution} for parameter: {parameter}")
        else:
            raise ValueError(f"Invalid parameter configuration for {parameter}: {parameter_config}")

    # Sort categorical parameters by number of values (descending)
    free_categorical_parameters = dict(sorted(free_categorical_parameters.items(), key=lambda x: len(x[1]), reverse=True))
    
    return fixed_parameters, free_categorical_parameters, free_random_parameters, categorical_dtypes


def exhaustive_parse_parameters(parameters_config: Dict) -> Dict:
    fixed_parameters, free_categorical_parameters, free_random_parameters, _ = parse_parameter_categories(parameters_config)
    experiment_runs = fetch_experiment_runs({f"config.{k}": v for k, v in fixed_parameters.items()})

    # Pick the least explored categorical configuration
    categorical_parameter_names = list(free_categorical_parameters.keys())
    parameter_space_dimension = [len(values) for parameter, values in free_categorical_parameters.items()]
    parameter_space = np.zeros(shape=parameter_space_dimension, dtype=int)

    for i, rows in enumerate(experiment_runs.select(categorical_parameter_names).iter_rows()):
        parameter_index = []
        for j, value in enumerate(rows):
            parameter_name = categorical_parameter_names[j]

            # Skip if the value is not in the current parameter space
            if value not in free_categorical_parameters[parameter_name]:
                break
            
            value_index = free_categorical_parameters[parameter_name].index(value)
            parameter_index.append(value_index)

        # Only consider complete indices
        if len(parameter_index) != len(categorical_parameter_names):
            continue

        parameter_space[*parameter_index] += 1

    non_zero_mask = (parameter_space > 0).astype(np.int32)
    print(f"Parameter space shape: {parameter_space.shape}")
    print(f"Existing configurations in the parameter space: {np.sum(non_zero_mask)} / {parameter_space.size}")
    print(f"Parameter space are {np.sum(non_zero_mask) / parameter_space.size * 100:.2f}% occupied.")
    print(f"Average runs per configuration: {np.sum(parameter_space) / np.sum(non_zero_mask) if np.sum(non_zero_mask) > 0 else 0:.2f}")

    ## Find the flat index of the minimum value
    min_flat_index = np.argmin(parameter_space)

    ## Convert flat index to multi-dimensional indices
    min_indices = np.unravel_index(min_flat_index, parameter_space.shape)

    ## Get the actual parameter values
    least_explored_categorical_config = {}
    for i, parameter_name in enumerate(categorical_parameter_names):
        parameter_value_index = min_indices[i]
        least_explored_categorical_config[parameter_name] = free_categorical_parameters[parameter_name][parameter_value_index]
    
    return {
        **fixed_parameters,
        **least_explored_categorical_config,
        **parse_parameters(free_random_parameters)
    }

def parse_score_metric(spec: str) -> Dict[str, float]:
    """Parse a score metric spec into a weighted composite dict.
    
    Accepts formats:
        "epoch/test_recall@20"                             → {"epoch/test_recall@20": 1.0}
        "epoch/test_recall@20:0.7 epoch/test_ndcg@20:0.3"  → {"epoch/test_recall@20": 0.7, ...}
    """
    items = spec.split()
    result = {}
    for item in items:
        if ":" in item:
            metric, weight = item.split(":")
            result[metric] = float(weight)
        else:
            result[item] = 1.0
    return result


def fetch_run_metadata(api: wandb.Api, run_id: str, max_retries: int = 0) -> Dict:
    """Fetch full metadata for a single run by ID. Retries with exponential backoff if max_retries > 0."""
    for attempt in range(max_retries + 1):
        try:
            run: Run = api.run(f"{api.default_entity}/{PROJECT_NAME}/{run_id}")
            break
        except Exception:
            if attempt < max_retries:
                time.sleep(3 * (2 ** attempt))
            else:
                raise

    run_config = {}
    for key, value in run.config.items():
        if isinstance(value, (list, dict)):
            run_config[key] = str(value)
        else:
            run_config[key] = value

    run_history = run.history()
    run_history = run_history.replace({"Infinity": np.inf, "NaN": np.nan})

    result = {
        "run_id": run.id,
        "run_name": run.name,
        "sweep_id": run.sweep.id if run.sweep else None,
        "model": run.config.get("model"),
        "created_at": run.created_at,
        **run_config,
        **{metric: run_history[metric].to_list() for metric in run_history},
        "gpu_type": run.metadata.get("gpu"),
        "cpu_count": run.metadata.get("cpu_count"),
    }

    # Compute derived fields
    runtime_list = result.get("_runtime")
    if runtime_list and len(runtime_list) > 0:
        runtime_max = max(runtime_list)
        result["run_duration_second"] = runtime_max
        result["run_duration_minute"] = runtime_max / 60

    model_name = result.get("model")
    sweep = result.get("sweep_id") or "single_runs"
    local_path = f"./models/{model_name}/{sweep}/{result['run_id']}"
    result["available_locally"] = os.path.isdir(local_path)

    return result


class Log10Transformer(BaseEstimator, TransformerMixin):
    """Log10-transform specified columns. Zero maps to a sentinel of -15."""

    LOG_SENTINEL = -15.0

    def __init__(self, feature_names: List[str], log_columns: List[str] = None):
        self.feature_names = feature_names
        self.log_columns = log_columns or ["l1_regularization", "l2_regularization"]

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "Log10Transformer":
        self.column_indices_ = [
            self.feature_names.index(c) for c in self.log_columns if c in self.feature_names
        ]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        for i in self.column_indices_:
            col = X[:, i]
            X[:, i] = np.where(col > 0, np.log10(np.clip(col, 1e-300, None)).round(1), self.LOG_SENTINEL)
        return X


def model_based_parse_parameters(
    parameters_config: Dict, 
    beta: float = 1.0, 
    target: str = "epoch/test_recall@20", 
    estimator_count: int = 1024, 
    summary_path: str = "wandb/summary.parquet", 
    log_path: str = "hyperparameter_search.log.csv", 
    virtual_sample_count: int = 100, 
    virtual_lambda: float = 1.0,
    n_jobs: int = -1,
    max_samples: float = 0.1,
) -> Dict:
    """
    Resolve hyperparameters using a surrogate model with UCB acquisition.
    Falls back to random sampling when data is insufficient.
    """
    start_time = datetime.now(timezone.utc)
    fixed_parameters, free_categorical_parameters, free_random_parameters, categorical_dtypes = parse_parameter_categories(parameters_config)

    feature_names = list(free_categorical_parameters)
    random_config = {
        **fixed_parameters,
        **{k: random.choice(v) for k, v in free_categorical_parameters.items()},
        **parse_parameters(free_random_parameters),
    }

    if not feature_names:
        return random_config, {}

    if not os.path.exists(summary_path):
        print("WARNING: summary parquet not found — falling back to random")
        return random_config, {}

    runs = pl.read_parquet(summary_path)
    model_filter = fixed_parameters.get("model", "matrix_factorization")
    runs = runs.filter(pl.col("model") == model_filter)
    for key, value in fixed_parameters.items():
        if key not in runs.columns or isinstance(value, list):
            continue
        runs = runs.filter(pl.col(key) == value)

    parsed_target = parse_score_metric(target)
    missing_metrics = [m for m in parsed_target if m not in runs.columns]
    if missing_metrics:
        raise KeyError(
            f"Target metric(s) {missing_metrics} not found in summary parquet columns. "
            f"Available epoch metrics: {[c for c in runs.columns if c.startswith('epoch/')]}"
        )

    # Compute target: best weighted composite score per run, then mean per unique config
    score_expression = sum(pl.col(m) * w for m, w in parsed_target.items())
    runs = runs.with_columns(score_expression.list.max().alias("target"))

    aggregated = runs.group_by(feature_names).agg([
        pl.col("target").mean().alias("target"),
        pl.col("target").std().alias("explored_sigma"),
        pl.col("target").count().alias("nruns"),
    ])
    aggregated = aggregated.drop_nulls(subset=["target"])
    if "shuffle" in feature_names:
        aggregated = aggregated.with_columns(pl.col("shuffle").cast(pl.Float64))

    if len(aggregated) < 20:
        print(f"WARNING: only {len(aggregated)} runs available (< 20) — falling back to random")
        return random_config, {}

    # Build full grid (shared between virtual samples and grid scoring)
    parameter_space = {col: free_categorical_parameters[col] for col in feature_names}
    all_combinations = list(itertools.product(*parameter_space.values()))

    # Inject virtual samples to bias the surrogate toward optimism in unexplored space
    if virtual_sample_count > 0:
        best_observed = aggregated["target"].max()
        mean_observed = aggregated["target"].mean()
        std_observed = aggregated["target"].std()
        virtual_target = max(best_observed, mean_observed + virtual_lambda * std_observed)

        selected_cells = random.sample(
            all_combinations,
            min(virtual_sample_count, len(all_combinations)),
        )

        virtual_records = [
            dict(zip(feature_names, cell)) | {"target": virtual_target}
            for cell in selected_cells
        ]
        virtual_samples_dataframe = pl.DataFrame(virtual_records)
        if "shuffle" in feature_names:
            virtual_samples_dataframe = virtual_samples_dataframe.with_columns(
                pl.col("shuffle").cast(pl.Float64)
            )

        training_data = pl.concat([aggregated, virtual_samples_dataframe], how="diagonal")
        train_features = training_data.select(feature_names).to_numpy()
        train_target = training_data["target"].to_numpy()
    else:
        train_features = aggregated.select(feature_names).to_numpy()
        train_target = aggregated["target"].to_numpy()

    surrogate = Pipeline([
        ("log_reg", Log10Transformer(feature_names)),
        ("rf", RandomForestRegressor(
            n_estimators=estimator_count,
            max_features="sqrt",
            max_samples=max_samples,
            min_samples_leaf=3,
            oob_score=True,
            n_jobs=n_jobs,
            random_state=42,
        )),
    ])
    surrogate.fit(train_features, train_target)

    # Build full grid DataFrame
    full_grid = pl.DataFrame(
        {col: [row[i] for row in all_combinations] for i, col in enumerate(feature_names)}
    )
    if "shuffle" in feature_names:
        full_grid = full_grid.with_columns(pl.col("shuffle").cast(pl.Float64))

    # Mark explored cells
    explored_keys = set(tuple(row) for row in aggregated.select(feature_names).to_numpy().tolist())
    full_grid = full_grid.with_columns(
        pl.struct(feature_names)
        .map_elements(lambda s: tuple(s[c] for c in feature_names) in explored_keys, return_dtype=pl.Boolean)
        .alias("explored")
    )

    # Join observed metrics for explored cells (for decision log)
    observed_agg = aggregated.rename({"target": "explored_mu"}).select([*feature_names, "explored_mu", "explored_sigma", "nruns"])
    full_grid = full_grid.join(observed_agg, on=feature_names, how="left")

    # Per-tree predictions for μ̂ and σ̂
    grid_features = full_grid.select(feature_names).to_numpy()
    grid_features_transformed = surrogate.named_steps["log_reg"].transform(grid_features)
    random_forest = surrogate.named_steps["rf"]
    tree_predictions = np.stack([tree.predict(grid_features_transformed) for tree in random_forest.estimators_], axis=1)
    mu_hat = tree_predictions.mean(axis=1)
    sigma_hat = tree_predictions.std(axis=1)

    full_grid = full_grid.with_columns([
        pl.Series("mu_hat", mu_hat),
        pl.Series("sigma_hat", sigma_hat),
        pl.Series("ucb", mu_hat + beta * sigma_hat),
    ])

    # ESM
    mu_best_observed = aggregated["target"].max()
    best_ucb = full_grid["ucb"].max()
    esm = (mu_best_observed / best_ucb) * 100 if best_ucb > 0 else 0.0

    # Coverage at multiple percentiles
    mu_hat_vector = full_grid["mu_hat"].to_numpy()
    coverage = {}
    for p in [75, 90, 95, 99]:
        threshold = np.percentile(mu_hat_vector, p)
        top_cells = full_grid.filter(pl.col("mu_hat") >= threshold)
        coverage[p] = 100.0 * top_cells["explored"].sum() / len(top_cells) if len(top_cells) > 0 else 0.0

    nines = 0
    while (esm / 100) >= 9 * (10 ** (-nines - 1)) + (1 - 10 ** (-nines)):
        nines += 1
        if nines > 10:
            break

    explored_percentage = 100.0 * full_grid["explored"].sum() / len(full_grid)

    # Pick best config (argmax UCB across all cells, with random tie-breaking)
    candidates = full_grid.filter(pl.col("ucb") == best_ucb)
    selected_candidate = candidates.to_dicts()[random.randint(0, len(candidates) - 1)]

    print(f"{'='*55}")
    print(f"  ESM:           {esm:.4f}%  ({nines} nines)")
    print(f"  Coverage@75:   {coverage[75]:.2f}%")
    print(f"  Coverage@90:   {coverage[90]:.2f}%")
    print(f"  Coverage@95:   {coverage[95]:.2f}%")
    print(f"  Coverage@99:   {coverage[99]:.2f}%")
    print(f"  Explored:      {explored_percentage:.2f}%  ({full_grid['explored'].sum():,} / {len(full_grid):,} cells)")
    print(f"  Best observed: {mu_best_observed:.6f}")
    print(f"  Best UCB:      μ̂={selected_candidate['mu_hat']:.6f}  σ̂={selected_candidate['sigma_hat']:.6f}  UCB={best_ucb:.6f}")
    print(f"{'='*55}")

    selected_config = random_config.copy()
    for col in feature_names:
        selected_config[col] = categorical_dtypes[col](selected_candidate[col])

    # Append to decision log
    decision_metadata = {"start_time": start_time.strftime("%Y-%m-%dT%H:%M:%S")}
    decision_metadata.update({params: value for params, value in selected_config.items()})
    decision_metadata.update({
        "selected_explored": selected_candidate.get("explored", False),
        "selected_nruns": selected_candidate.get("nruns"),
        "explored_percentage": round(explored_percentage, 2),
        "selected_observed_mu": selected_candidate.get("explored_mu"),
        "selected_observed_sigma": selected_candidate.get("explored_sigma"),
        "selected_predicted_mu": selected_candidate.get("mu_hat"),
        "selected_predicted_sigma": selected_candidate.get("sigma_hat"),
        "selected_ucb": selected_candidate.get("ucb"),
        "esm": round(esm, 4) if esm else None,
        "coverage@75": round(coverage[75], 2) if coverage[75] else None,
        "coverage@90": round(coverage[90], 2) if coverage[90] else None,
        "coverage@95": round(coverage[95], 2) if coverage[95] else None,
        "coverage@99": round(coverage[99], 2) if coverage[99] else None,
    })

    return selected_config, decision_metadata


def write_decision_log(decision_metadata: Dict, log_path: str) -> None:
    if not decision_metadata:
        return
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(decision_metadata))
        if write_header:
            writer.writeheader()
        writer.writerow(decision_metadata)


def load_config(config_path: str, method: Literal["random", "exhaustive", "model_based"] = "random", **kwargs) -> Tuple[Dict, Dict]:
    """
    Load configuration from a YAML file.
    Will also sample hyperparameters if the config file is for hyperparameter search.

    Args:
        config_path (str): Path to the YAML configuration file.
        method (Literal["random", "exhaustive", "model_based"]): Method for hyperparameter search.
    Returns:
        tuple: (dict of configuration parameters, dict of decision metadata)
    """
    current_run_config = {}
    decision_metadata = {}

    config = load_yaml(config_path)
    if "parameters" in config:
        if method == "random":
            current_run_config.update(parse_parameters(config["parameters"]))
        elif method == "exhaustive":
            current_run_config.update(exhaustive_parse_parameters(config["parameters"]))
        elif method == "model_based":
            config_result, decision_metadata = model_based_parse_parameters(config["parameters"], **kwargs)
            current_run_config.update(config_result)
        else:
            raise ValueError(f"Unsupported hyperparameter search method: {method}")
    else:
        current_run_config.update(config)

    # Convert string scientific notation to floating point numbers
    for key, value in current_run_config.items():
        current_run_config[key] = parse_scientific_notation(value)

    return current_run_config, decision_metadata