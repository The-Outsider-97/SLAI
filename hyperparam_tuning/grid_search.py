import logging
import json
import itertools
import torch
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class GridSearch:
    """
    GridSearch performs hyperparameter optimization using an exhaustive search
    over all possible combinations of provided hyperparameter values.
    """

    def __init__(self, config_file, evaluation_function):
        """
        Initializes the GridSearch instance.
        
        Args:
            config_file (str): Path to the hyperparameter config JSON.
            evaluation_function (callable): Function to evaluate model performance with given hyperparameters.
        """
        self.config_file = config_file
        self.evaluation_function = evaluation_function
        self.hyperparam_space, self.param_names = self._load_search_space()

    def _load_search_space(self):
        """
        Loads hyperparameter search space from the config file.

        Returns:
            tuple: A list of hyperparameter value lists and a list of parameter names.
        """
        logger.info("Loading hyperparameter search space from: %s", self.config_file)
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        
        param_names = []
        param_values = []

        for param in config['hyperparameters']:
            name = param['name']
            param_type = param['type']
            param_names.append(name)

            if param_type == 'int':
                values = list(range(param['min'], param['max'] + 1, param.get('step', 1)))
            elif param_type == 'float':
                steps = param.get('steps', 10)
                values = [param['min'] + x * (param['max'] - param['min']) / (steps - 1) for x in range(steps)]
            elif param_type == 'categorical':
                values = param['choices']
            else:
                raise ValueError(f"Unsupported parameter type: {param_type}")

            param_values.append(values)
            logger.info("Loaded %d values for hyperparameter: %s", len(values), name)

        return param_values, param_names

    def run_search(self):
        """
        Executes grid search to find the best hyperparameters.

        Returns:
            dict: The best hyperparameter combination found.
        """
        logger.info("Starting exhaustive grid search hyperparameter optimization...")

        best_params = None
        best_score = float('-inf')

        combinations = list(itertools.product(*self.hyperparam_space))
        logger.info("Total combinations to evaluate: %d", len(combinations))

        for combo in combinations:
            params = {name: val for name, val in zip(self.param_names, combo)}
            logger.info("Evaluating combination: %s", params)

            score = self.evaluation_function(params)
            logger.info("Score achieved: %.4f", score)

            if score > best_score:
                best_score = score
                best_params = params
                logger.info("New best score %.4f with params: %s", best_score, best_params)

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

        output_file = self.config_file.replace('.json', '_grid_best.json')
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
    with open('hyperparam_tuning/example_grid_config.json', 'w') as f:
        json.dump(example_config, f, indent=4)

    # Run the GridSearch
    grid_search = GridSearch(
        config_file='hyperparam_tuning/example_grid_config.json',
        evaluation_function=dummy_evaluation
    )

    best_params = grid_search.run_search()
    print("\nBest Hyperparameters:", best_params)
