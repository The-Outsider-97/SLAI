import os, sys
import logging
import json
import yaml

# from src.tuning.configs.hyperparam_config_generator import HyperparamConfigGenerator
from src.tuning.utils.config_loader import load_global_config, get_config_section
from src.tuning.grid_search import GridSearch
from src.tuning.bayesian_search import BayesianSearch
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Hyperparameter Tuner")
printer = PrettyPrinter

class HyperparamTuner:
    """
    HyperparamTuner orchestrates hyperparameter optimization strategies.
    Supports Grid Search and Bayesian Search, dynamically selecting the strategy
    based on configuration or runtime input.
    """

    def __init__(self, model_type=None, evaluation_function=None):
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
        self.config = load_global_config()
        self.tuning_config = get_config_section('tuning')
        self.strategy = self.tuning_config.get('strategy')
        self.n_calls = self.tuning_config.get('n_calls')
        self.n_random_starts = self.tuning_config.get('n_random_starts')
        self.allow_generate = self.tuning_config.get('allow_generate')

        self.evaluation_function = evaluation_function
        if model_type is None:
            model_type = self.config.get('model_type', 'GradientBoosting')
        if not self.evaluation_function:
            raise ValueError("An evaluation_function is required.")

        # Update Bayesian parameters
        if self.strategy == 'bayesian':
            self.bayesian_config = get_config_section('bayesian_search')
            self.n_calls = self.bayesian_config.get('n_calls')
            self.n_initial_points = self.bayesian_config.get('n_initial_points')
            self.random_state = self.bayesian_config.get('random_state')

        # Update Grid parameters
        if self.strategy == 'grid':
            self.grid_config = get_config_section('grid_search')
            self.n_calls = self.grid_config.get('n_calls')
            self.cross_val_folds = self.grid_config.get('cross_val_folds')
            self.random_state = self.grid_config.get('random_state')

        # Initialize optimizer
        if self.strategy == 'bayesian':
            self.optimizer = BayesianSearch(evaluation_function=self.evaluation_function, model_type=model_type)
        elif self.strategy == 'grid':
            self.optimizer = GridSearch(evaluation_function=self.evaluation_function)
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

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

    print(f"\n* * * * * Phase 1 * * * * *\n")
    tuner = HyperparamTuner(
        evaluation_function=lambda params: np.random.rand()
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
