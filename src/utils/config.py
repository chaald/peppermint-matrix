import re
import json
import yaml
import math
import random
from typing import Union, List, Dict

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

def load_config(config_path: str) -> Dict:
    """
    Load configuration from a YAML file.
    Will also sample hyperparameters if the config file is for hyperparameter search.

    Args:
        config_path (str): Path to the YAML configuration file.
    Returns:
        dict: Configuration parameters as a dictionary.
    """
    single_run_config = {}
    
    config = load_yaml(config_path)
    if "parameters" in config:
        single_run_config.update(parse_parameters(config["parameters"]))
    else:
        single_run_config.update(config)

    # Convert string scientific notation to floating point numbers
    for key, value in single_run_config.items():
        single_run_config[key] = parse_scientific_notation(value)

    return single_run_config