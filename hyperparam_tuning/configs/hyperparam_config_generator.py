import json
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

    def create_default_config(self):
        """
        Generates a default hyperparameter configuration with commonly tuned parameters.

        Returns:
            dict: Hyperparameter configuration dictionary.
        """
        config = {
            "hyperparameters": [
                {
                    "name": "learning_rate",
                    "type": "float",
                    "min": 0.0001,
                    "max": 0.1,
                    "prior": "log-uniform",
                    "steps": 10  # for grid search only
                },
                {
                    "name": "num_layers",
                    "type": "int",
                    "min": 1,
                    "max": 10,
                    "step": 1  # for grid search only
                },
                {
                    "name": "batch_size",
                    "type": "int",
                    "min": 16,
                    "max": 256,
                    "step": 16  # for grid search only
                },
                {
                    "name": "optimizer",
                    "type": "categorical",
                    "choices": ["adam", "sgd", "rmsprop"]
                },
                {
                    "name": "activation",
                    "type": "categorical",
                    "choices": ["relu", "tanh", "sigmoid"]
                },
                {
                    "name": "dropout_rate",
                    "type": "float",
                    "min": 0.0,
                    "max": 0.5,
                    "steps": 6  # for grid search only
                },
                {
                    "name": "gamma",
                    "type": "float",
                    "min": 0.8,
                    "max": 0.999,
                    "steps": 5  # for grid search only
                }
            ]
        }
        return config

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
    generator = HyperparamConfigGenerator(output_dir='hyperparam_tuning/configs')
    generator.generate_and_save(name_prefix='agent')

    print(" Hyperparameter configuration files generated successfully!")
