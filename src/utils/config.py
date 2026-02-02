import re
import json
import yaml
import math
import random
import wandb
import numpy as np
import polars as pl

from wandb.sdk.internal.internal_api import gql
from typing import Literal, Union, List, Dict
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

def fetch_experiment_runs(filters: Dict[str, Union[int, float, str]]) -> pl.DataFrame:
    api = wandb.Api() # Initialize Weights & Biases API, used for fetching run data

    query = """
        query Runs($project: String!, $entity: String!, $cursor: String, $filters: JSONString) {
            project(name: $project, entityName: $entity) {
                runs(first: 256, after: $cursor, filters: $filters) {
                    edges {
                        node {
                            id
                            name
                            config
                        }
                        cursor
                    }
                    pageInfo {
                        hasNextPage
                        endCursor
                    }
                }
            }
        }
    """
    query = gql(query)

    experiment_runs = []
    cursor = None
    filters = json.dumps({
        f"config.{key}": value for key, value in filters.items()
    })

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

            experiment_runs.append({
                "id": current_run["id"],
                "name": current_run["name"],
                **parse_config(current_run['config']),
            })
        
        if not runs_data["pageInfo"]["hasNextPage"]:
            print(f"Fetched {len(experiment_runs)} runs...")
            break

        cursor = runs_data["pageInfo"]["endCursor"]
        print(f"Fetched {len(experiment_runs)} runs...")

    return pl.DataFrame(experiment_runs, infer_schema_length=None)

def exhaustive_parse_parameters(parameters_config: Dict) -> Dict:
    fixed_parameters: Dict[str, Union[int, float, str]] = {}
    free_categorical_parameters: Dict[str, List[Union[int, float, str]]] = {}
    free_random_parameters: Dict[str, Dict[str, Union[int, float, str]]] = {}
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
                free_categorical_parameters[parameter] = parse_scientific_notation(parameter_config["values"])
            elif distribution in ["int_uniform", "uniform", "log_uniform"]:
                free_random_parameters[parameter] = parameter_config
            else:
                raise ValueError(f"Unsupported distribution type: {distribution} for parameter: {parameter}")
        else:
            raise ValueError(f"Invalid parameter configuration for {parameter}: {parameter_config}")
        
    # Sort categorical parameters by number of values (descending)
    free_categorical_parameters = dict(sorted(free_categorical_parameters.items(), key=lambda x: len(x[1]), reverse=True))

    # Get latest experiment runs to count existing configurations
    experiment_runs = fetch_experiment_runs(fixed_parameters)

    # Pick the least explored categorical configuration
    categorical_parameter_names = list(free_categorical_parameters.keys())
    parameter_space_dimension = [len(values) for parameter, values in free_categorical_parameters.items()]
    parameter_space = np.zeros(shape=parameter_space_dimension, dtype=int)

    for i, rows in enumerate(experiment_runs.select(categorical_parameter_names).iter_rows()):
        parameter_index = []
        for j, value in enumerate(rows):
            parameter_name = categorical_parameter_names[j]
            value_index = free_categorical_parameters[parameter_name].index(value)
            parameter_index.append(value_index)

        parameter_space[*parameter_index] += 1

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

def load_config(config_path: str, method: Literal["random", "exhaustive"] = "random") -> Dict:
    """
    Load configuration from a YAML file.
    Will also sample hyperparameters if the config file is for hyperparameter search.

    Args:
        config_path (str): Path to the YAML configuration file.
        method (Literal["random", "exhaustive"]): Method for hyperparameter search. Options are "random" or "exhaustive".
    Returns:
        dict: Configuration parameters as a dictionary.
    """
    current_run_config = {}
    
    config = load_yaml(config_path)
    if "parameters" in config:
        if method == "random":
            current_run_config.update(parse_parameters(config["parameters"]))
        elif method == "exhaustive":
            current_run_config.update(exhaustive_parse_parameters(config["parameters"]))
        else:
            raise ValueError(f"Unsupported hyperparameter search method: {method}")
    else:
        current_run_config.update(config)

    # Convert string scientific notation to floating point numbers
    for key, value in current_run_config.items():
        current_run_config[key] = parse_scientific_notation(value)

    return current_run_config