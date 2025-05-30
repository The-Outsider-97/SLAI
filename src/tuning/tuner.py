import os, sys
import logging
import json
import yaml

from src.tuning.grid_search import GridSearch
from src.tuning.bayesian_search import BayesianSearch
from src.tuning.configs.hyperparam_config_generator import HyperparamConfigGenerator
from logs.logger import get_logger

logger = get_logger("Hyperparameter Tuner")

CONFIG_PATH = "src/tuning/configs/hyperparam.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_merged_config(user_config=None):
    base_config = load_config()
    if user_config:
        base_config.update(user_config)
    return base_config

class HyperparamTuner:
    """
    HyperparamTuner orchestrates hyperparameter optimization strategies.
    Supports Grid Search and Bayesian Search, dynamically selecting the strategy
    based on configuration or runtime input.
    """

    def __init__(self,
                 config_path=None,
                 evaluation_function=None,
                 strategy='bayesian',
                 n_calls=20,
                 n_random_starts=5,
                 allow_generate=True,
                 config_format=None):
        """
        Initializes the tuner.

        Args:
            config_path (str): Path to the hyperparameter config file.
            evaluation_function (callable): Function that evaluates model performance.
            strategy (str): The optimization strategy ('bayesian' or 'grid').
            n_calls (int): Number of iterations for Bayesian optimization.
            n_random_starts (int): Random starts for Bayesian optimization.
            allow_generate (bool): Whether to generate config if not provided.
            config_format (str): Format of the config file ('json' or 'yaml').
        """
        self.evaluation_function = evaluation_function
        self.strategy = strategy.lower()
        self.n_calls = n_calls
        self.n_random_starts = n_random_starts

        if not self.evaluation_function:
            raise ValueError("An evaluation_function is required.")

        # Handle config path
        if config_path and os.path.exists(config_path):
            self.config_path = config_path
            logger.info("Using provided config: %s", self.config_path)
        elif allow_generate:
            generator = HyperparamConfigGenerator(output_dir='src/tuning/configs')
            fmt = config_format or 'json'
            # Generate from YAML instead of using create_default_config()
            generator.generate_from_yaml(
                yaml_path=CONFIG_PATH,
                name_prefix='generated_config'
            )
            self.config_path = os.path.join(
                generator.output_dir,
                f'generated_config_{self.strategy}.{fmt}'
            )
            logger.info("Generated config at: %s", self.config_path)
        else:
            raise FileNotFoundError("No valid config provided.")

        # Load config data from the resolved path
        self.config_data = self._load_config()

        # Update Bayesian parameters
        if self.strategy == 'bayesian':
            self.config_data.setdefault('bayesian_search', {})
            self.config_data['bayesian_search']['n_calls'] = self.n_calls
            self.config_data['bayesian_search']['n_initial_points'] = self.n_random_starts

        # Initialize optimizer
        if self.strategy == 'bayesian':
            self.optimizer = BayesianSearch(
                config=self.config_data,
                evaluation_function=self.evaluation_function,
                output_dir_name="bayesian_search"
            )
        elif self.strategy == 'grid':
            self.optimizer = GridSearch(
                config=self.config_data,
                evaluation_function=self.evaluation_function
            )
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

    def _load_config(self):
        """Load YAML/JSON config from the resolved path."""
        if self.config_path.endswith(('.yaml', '.yml')):
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        elif self.config_path.endswith('.json'):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {self.config_path}")

    def run_tuning_pipeline(self, X_data=None, y_data=None):
        """Execute the tuning pipeline."""
        logger.info("Starting %s tuning...", self.strategy)
        if self.strategy == 'grid':
            if X_data is None or y_data is None:
                raise ValueError("X_data and y_data are required for GridSearch.")
            best_params = self.optimizer.run_search(X_data, y_data)
        else:
            best_params, _, _ = self.optimizer.run_search()
        
        logger.info("Best parameters: %s", best_params)
        return best_params

# ====================== Usage Example ======================
if __name__ == "__main__":
    import numpy as np
    print("\n=== Running Hyperparameter Tuner ===\n")
    config = load_config()

    print(f"\n* * * * * Phase 1 * * * * *\n")
    tuner = HyperparamTuner(
        evaluation_function=lambda params: np.random.rand(),
        strategy='bayesian'
    )
    tuner.run_tuning_pipeline()

    # For Grid Search
    # tuner = HyperparamTuner(
    #     evaluation_function=lambda params, X_train, y_train, X_val, y_val: np.random.rand(),
    #    strategy='grid'
    # )
    # X, y = load_your_data()
    # best_params = tuner.run_tuning_pipeline(X, y)
    print(f"\n* * * * * Phase 3 * * * * *\n")

    print("\n=== Successfully Ran Hyperparameter Tuner ===\n")
