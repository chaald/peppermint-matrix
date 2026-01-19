import json
import yaml
import random
import math
from typing import Dict

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
        for parameter, parameter_config in config["parameters"].items():

            if not isinstance(parameter_config, dict):
                single_run_config[parameter] = parameter_config
            elif "value" in parameter_config:
                single_run_config[parameter] = parameter_config["value"]
            elif "distribution" in parameter_config:
                distribution = parameter_config["distribution"]
                if distribution == "constant":
                    single_run_config[parameter] = parameter_config["value"]
                elif distribution == "categorical":
                    single_run_config[parameter] = random.choice(parameter_config["values"])
                elif distribution == "int_uniform":
                    single_run_config[parameter] = random.randint(parameter_config["min"], parameter_config["max"])
                elif distribution == "uniform": # float uniform
                    single_run_config[parameter] = random.uniform(parameter_config["min"], parameter_config["max"])
                elif distribution == "log_uniform":
                    log_min = parameter_config["min"]
                    log_max = parameter_config["max"]
                    sampled_log_value = random.uniform(log_min, log_max)
                    single_run_config[parameter] = math.exp(sampled_log_value)
                else:
                    raise ValueError(f"Unsupported distribution type: {distribution} for parameter: {parameter}")
            else:
                raise ValueError(f"Invalid parameter configuration for {parameter}: {parameter_config}")
    else:
        single_run_config.update(config)

    return single_run_config