import os, sys
import logging
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from src.tuning.grid_search import GridSearch
from src.tuning.bayesian_search import BayesianSearch
from src.tuning.configs.hyperparam_config_generator import HyperparamConfigGenerator

from src.agents.evaluation_agent import EvaluationAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
            config_path (str): Path to the hyperparameter config JSON.
            evaluation_function (callable): Function that evaluates model performance.
            strategy (str): The optimization strategy ('bayesian' or 'grid').
            n_calls (int): Number of iterations for Bayesian optimization.
            n_random_starts (int): Random starts for Bayesian optimization.
        """

        self.evaluation_function = evaluation_function
        self.strategy = strategy.lower()
        self.n_calls = n_calls
        self.n_random_starts = n_random_starts

        if not self.evaluation_function:
            raise ValueError("An evaluation_function must be provided to the tuner.")

        # Determine config path or dynamically generate
        if config_path and os.path.exists(config_path):
            self.config_path = config_path
            logger.info("Using provided config file: %s", self.config_path)
        elif allow_generate:
            generator = HyperparamConfigGenerator()
            fmt = config_format or 'json'
            self.config_path = generator.generate_configs(strategy=self.strategy, fmt=fmt, auto_load=True)
            logger.info("Generated config at: %s", self.config_path)
        else:
            raise FileNotFoundError("No valid config file provided and generation not allowed.")

        # Load config format (YAML/JSON)
        self.config_data = self._load_config()

        # Strategy selection
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
            raise ValueError(f"Unsupported strategy '{self.strategy}'. Choose 'bayesian' or 'grid'.")

    def _load_config(self):
        if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        elif self.config_path.endswith('.json'):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {self.config_path}")

    def run_tuning_pipeline(self):
        logger.info("Running hyperparameter tuning pipeline using %s strategy...", self.strategy)
        best_params = self.optimizer.run_search()
        logger.info("Hyperparameter tuning completed. Best parameters: %s", best_params)
        return best_params


if __name__ == "__main__":
    args = parser.parse_args()
    logging.info("Starting tuner with strategy: %s", args.strategy)

    try:
        tuner = HyperparamTuner(
            config_path=None,  # Leave None to auto-generate
            evaluation_function=EvaluationAgent,
            strategy='grid',
            config_format='yaml'
          )  # Choose output format if generating
        grid_search = GridSearch(
            config_file="src/tuning/configs/grid_config.json",
            evaluation_function=rl_agent_evaluate,
            reasoning_agent=ReasoningAgent(),
            n_jobs=8,
            cross_val_folds=5
        )

        best_params = grid_search.run_search()
        best = tuner.run_tuning_pipeline()
        print("Best parameters:", best)
    except Exception as e:
        logging.error("Tuning failed: %s", str(e))
        sys.exit(1)
