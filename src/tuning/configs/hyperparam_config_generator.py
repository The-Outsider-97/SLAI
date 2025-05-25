
import os
import json, yaml
import numpy as np

from logs.logger import get_logger

logger = get_logger("Hyperparam Config Generator")

class HyperparamConfigGenerator:
    """
    Generates hyperparameter configuration JSON files for both
    Grid Search and Bayesian Optimization pipelines.
    """

    def __init__(self, output_dir='hyperparam_tuning/configs'):
        """
        Initialize the generator.

        Args:
            output_dir (str): Directory where config files will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def load_from_yaml(self, yaml_path: str) -> dict:
        """
        Load and parse hyperparameter definitions from a YAML config.

        Args:
            yaml_path (str): Path to the hyperparameter YAML file.

        Returns:
            dict: Parsed configuration dictionary.
        """
        with open(yaml_path, 'r') as f:
            raw = yaml.safe_load(f)

        return {"hyperparameters": raw.get("hyperparameters", [])}

    def create_default_config(self):
        return []

    def transform_for_grid(self, config: dict) -> dict:
        """
        Transform parsed YAML config to grid search format.

        Args:
            config (dict): Raw config.

        Returns:
            dict: Grid-compatible config with 'values' field.
        """
        grid_config = {"hyperparameters": []}
        for param in config['hyperparameters']:
            transformed = {
                "name": param["name"],
                "type": param["type"]
            }

            if param["type"] in ["int", "float"] and "steps" in param:
                transformed["values"] = list(np.linspace(
                    param["min"],
                    param["max"],
                    int(param["steps"])
                ))
            elif param["type"] in ["int", "float"] and "step" in param:
                transformed["values"] = list(range(
                    param["min"],
                    param["max"] + 1,
                    param["step"]
                ))
            elif param["type"] == "categorical":
                transformed["values"] = param["values"]

            # Preserve optional metadata
            for meta in ("prior", "prior_research", "hardware_constraints", "physiological_basis"):
                if meta in param:
                    transformed[meta] = param[meta]

            grid_config["hyperparameters"].append(transformed)

        return grid_config

    def transform_for_bayesian(self, config: dict) -> dict:
        """
        Transform parsed YAML config to Bayesian optimization format.

        Args:
            config (dict): Raw config.

        Returns:
            dict: Bayesian-compatible config.
        """
        bayesian_config = {"hyperparameters": []}
        for param in config["hyperparameters"]:
            transformed = {"name": param["name"], "type": param["type"]}
    
            if param["type"] in ["int", "float"]:
                # Extract min/max from values if not explicitly defined
                if "min" not in param or "max" not in param:
                    if "values" in param:
                        param["min"] = min(param["values"])
                        param["max"] = max(param["values"])
                    else:
                        raise KeyError(f"Parameter {param['name']} missing 'values' or 'min'/'max'.")
                
                transformed["min"] = param["min"]
                transformed["max"] = param["max"]
                
                if "prior" in param:
                    transformed["prior"] = param["prior"]
            elif param["type"] == "categorical":
                transformed["choices"] = param["values"]
    
            bayesian_config["hyperparameters"].append(transformed)
        return bayesian_config

    def generate_from_yaml(self, yaml_path: str, name_prefix="hyperparam_config"):
        """
        Full pipeline: parse YAML and generate both grid and Bayesian config files.

        Args:
            yaml_path (str): Path to unified hyperparam YAML file.
            name_prefix (str): Output filename prefix.
        """
        raw_config = self.load_from_yaml(yaml_path)
        grid_config = self.transform_for_grid(raw_config)
        bayesian_config = self.transform_for_bayesian(raw_config)

        self.save_config(grid_config, f"{name_prefix}_grid.json")
        self.save_config(bayesian_config, f"{name_prefix}_bayesian.json")

    def save_config(self, config, filename):
        """
        Saves a hyperparameter configuration dictionary to a JSON file.

        Args:
            config (dict): Hyperparameter config to save.
            filename (str): Filename to save the config as.
        """
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info("Config saved to %s", output_path)

    def generate_and_save(self, name_prefix='hyperparam_config'):
        """
        Generates and saves both Grid Search and Bayesian Optimization compatible configs.

        Args:
            name_prefix (str): Prefix for generated config files.
        """
        config = self.create_default_config()

        bayesian_config = self._strip_steps_for_bayesian(config)
        grid_config = config

        bayesian_filename = f'{name_prefix}_bayesian.json'
        grid_filename = f'{name_prefix}_grid.json'

        self.save_config(bayesian_config, bayesian_filename)
        self.save_config(grid_config, grid_filename)

    def _strip_steps_for_bayesian(self, config):
        """
        Prepares a config for Bayesian search by removing grid-specific keys.

        Args:
            config (dict): Original hyperparameter config.

        Returns:
            dict: Cleaned config for Bayesian search.
        """
        bayesian_config = {"hyperparameters": []}

        for param in config['hyperparameters']:
            clean_param = param.copy()
            if 'steps' in clean_param:
                del clean_param['steps']
            if 'step' in clean_param:
                del clean_param['step']
            bayesian_config['hyperparameters'].append(clean_param)

        return bayesian_config

if __name__ == '__main__':
    generator = HyperparamConfigGenerator(output_dir='src/tuning/configs')
    generator.generate_from_yaml('src/tuning/configs/hyperparam.yaml', name_prefix='agent')
    print("✅ Hyperparameter configuration files generated from YAML!")
