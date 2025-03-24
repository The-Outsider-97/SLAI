import logging
import json
import torch
import numpy as np
import itertools
from hyperparam_tuning.grid_search import GridSearch
from hyperparam_tuning.bayesian_search import BayesianSearch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class HyperParamTuner:
    """
    HyperParamTuner orchestrates hyperparameter optimization strategies.
    Supports Grid Search and Bayesian Search, dynamically selecting the strategy
    based on configuration or runtime input.
    """

    def __init__(self, config_path, evaluation_function=None, strategy='bayesian', n_calls=20, n_random_starts=5):
        """
        Initializes the tuner.

        Args:
            config_path (str): Path to the hyperparameter config JSON.
            evaluation_function (callable): Function that evaluates model performance.
            strategy (str): The optimization strategy ('bayesian' or 'grid').
            n_calls (int): Number of iterations for Bayesian optimization.
            n_random_starts (int): Random starts for Bayesian optimization.
        """
        self.config_path = config_path
        self.evaluation_function = evaluation_function
        self.strategy = strategy.lower()
        self.n_calls = n_calls
        self.n_random_starts = n_random_starts

        if not evaluation_function:
            raise ValueError("An evaluation_function must be provided to the tuner.")

        # Strategy Selection
        if self.strategy == 'bayesian':
            self.optimizer = BayesianSearch(
                config_file=self.config_path,
                evaluation_function=self.evaluation_function,
                n_calls=self.n_calls,
                n_random_starts=self.n_random_starts
            )
        elif self.strategy == 'grid':
            self.optimizer = GridSearch(
                config_file=self.config_path,
                evaluation_function=self.evaluation_function
            )
        else:
            raise ValueError("Unsupported strategy '{}'. Choose 'bayesian' or 'grid'.".format(self.strategy))

        logger.info("HyperParamTuner initialized with strategy: %s", self.strategy)

    def run_tuning_pipeline(self):
        """
        Runs the selected hyperparameter optimization strategy.

        Returns:
            dict: Best hyperparameters found.
        """
        logger.info("Running hyperparameter tuning pipeline using %s strategy...", self.strategy)
        best_params = self.optimizer.run_search()

        logger.info("Hyperparameter tuning completed. Best parameters: %s", best_params)
        return best_params

if __name__ == "__main__":
    # Dummy evaluation function for demonstration
    def dummy_evaluation(params):
        """
        Dummy evaluation function that simulates model performance.
        Higher score is better.
        """
        score = -((params['learning_rate'] - 0.01) ** 2 + (params['num_layers'] - 3) ** 2)
        return score

    # Example usage with Bayesian Search
    bayesian_tuner = HyperParamTuner(
        config_path='hyperparam_tuning/example_config.json',
        evaluation_function=dummy_evaluation,
        strategy='bayesian',
        n_calls=10,
        n_random_starts=2
    )
    bayesian_best_params = bayesian_tuner.run_tuning_pipeline()
    print("\nBest Params from Bayesian Search:", bayesian_best_params)

    # Example usage with Grid Search
    grid_tuner = HyperParamTuner(
        config_path='hyperparam_tuning/example_grid_config.json',
        evaluation_function=dummy_evaluation,
        strategy='grid'
    )
    grid_best_params = grid_tuner.run_tuning_pipeline()
    print("\nBest Params from Grid Search:", grid_best_params)
