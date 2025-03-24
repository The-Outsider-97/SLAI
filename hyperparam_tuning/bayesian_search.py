import logging
import json
import torch
import numpy as np
import itertools
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class BayesianSearch:
    """
    BayesianSearch performs hyperparameter optimization using Gaussian Process-based Bayesian optimization.
    """

    def __init__(self, config_file, evaluation_function, n_calls=20, n_random_starts=5):
        """
        Initializes the BayesianSearch instance.

        Args:
            config_file (str): Path to the hyperparameter config JSON.
            evaluation_function (callable): Function to evaluate model performance with given hyperparameters.
            n_calls (int): Total number of optimization calls.
            n_random_starts (int): Number of initial random search steps.
        """
        self.config_file = config_file
        self.evaluation_function = evaluation_function
        self.n_calls = n_calls
        self.n_random_starts = n_random_starts
        self.hyperparam_space, self.dimensions = self._load_search_space()

    def _load_search_space(self):
        """
        Loads hyperparameter search space from the config file.

        Returns:
            tuple: Search space as a list of dictionaries, and skopt dimensions.
        """
        logger.info("Loading hyperparameter search space from: %s", self.config_file)
        with open(self.config_file, 'r') as f:
            config = json.load(f)

        space = []
        dimensions = []

        for param in config['hyperparameters']:
            name = param['name']
            param_type = param['type']

            if param_type == 'int':
                dimensions.append(Integer(param['min'], param['max'], name=name))
            elif param_type == 'float':
                dimensions.append(Real(param['min'], param['max'], prior=param.get('prior', 'uniform'), name=name))
            elif param_type == 'categorical':
                dimensions.append(Categorical(param['choices'], name=name))
            else:
                raise ValueError(f"Unsupported parameter type: {param_type}")

            space.append({
                'name': name,
                'type': param_type,
                'specs': param
            })

        logger.info("Loaded search space for %d hyperparameters.", len(space))
        return space, dimensions

    def run_search(self):
        """
        Executes Bayesian optimization to find the best hyperparameters.

        Returns:
            dict: The best hyperparameter combination found.
        """
        logger.info("Starting Bayesian hyperparameter optimization...")

        @use_named_args(self.dimensions)
        def objective(**params):
            logger.info("Evaluating parameters: %s", params)
            score = self.evaluation_function(params)
            logger.info("Score achieved: %.4f", score)
            return -score  # Because gp_minimize minimizes the function

        result = gp_minimize(
            func=objective,
            dimensions=self.dimensions,
            n_calls=self.n_calls,
            n_random_starts=self.n_random_starts,
            random_state=42
        )

        best_params = {dim.name: val for dim, val in zip(self.dimensions, result.x)}
        best_score = -result.fun
        logger.info("Best parameters found: %s", best_params)
        logger.info("Best score achieved: %.4f", best_score)

        self._save_best_params(best_params, best_score)
        return best_params

    def _save_best_params(self, params, score):
        """
        Saves the best hyperparameters to a JSON file.
        """
        output = {
            'best_hyperparameters': params,
            'best_score': score
        }

        output_file = self.config_file.replace('.json', '_best.json')
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=4)

        logger.info("Best hyperparameters saved to: %s", output_file)


if __name__ == "__main__":

    # Example evaluation function
    def rl_agent_evaluation(params):
        """
        Evaluates an RL agent's policy using hyperparameters.

        Args:
            params (dict): Dictionary of hyperparameters.

        Returns:
            float: Average cumulative reward across validation episodes.
        """
        # Extract parameters
        learning_rate = params['learning_rate']
        num_layers = params['num_layers']
        activation_function = params['activation']

        # Initialize and train the RL agent (pseudo-code)
        agent = RLAgent(
            learning_rate=learning_rate,
            num_layers=num_layers,
            activation=activation_function
        )
        agent.train(episodes=100)

        # Evaluate the agent on validation episodes
        rewards = []
        for _ in range(10):
            cumulative_reward = agent.evaluate()
            rewards.append(cumulative_reward)

        avg_reward = np.mean(rewards)
        return avg_reward

    # Example hyperparameter config
    example_config = {
        "hyperparameters": [
            {
                "name": "learning_rate",
                "type": "float",
                "min": 0.0001,
                "max": 0.1,
                "prior": "log-uniform"
            },
            {
                "name": "num_layers",
                "type": "int",
                "min": 1,
                "max": 10
            },
            {
                "name": "batch_size",
                "type": "int",
                "min": 16,
                "max": 256
            },
            {
                "name": "optimizer",
                "type": "categorical",
                "choices": ["adam", "sgd", "rmsprop"]
            }
        ]
    }

    # Save the example config for demonstration
    with open('hyperparam_tuning/example_config.json', 'w') as f:
        json.dump(example_config, f, indent=4)

    # Dummy evaluation function for demonstration purposes
    def dummy_evaluation(params):
        """
        Dummy evaluation function that simulates model performance.
        Higher is better.
        """
        score = -((params['learning_rate'] - 0.01) ** 2 + (params['num_layers'] - 3) ** 2)
        return score

    # Run the BayesianSearch
    bayes_opt = BayesianSearch(
        config_file='hyperparam_tuning/example_config.json',
        evaluation_function=dummy_evaluation,
        n_calls=10,
        n_random_starts=2
    )

    best_params = bayes_opt.run_search()
    print("\nBest Hyperparameters:", best_params)
