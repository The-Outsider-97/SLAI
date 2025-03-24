import unittest
import torch
import os
from hyperparam_tuning.tuner import HyperParamTuner

def dummy_eval(params):
    return 1.0

class TestHyperParamTuner(unittest.TestCase):
    def test_bayesian_strategy(self):
        tuner = HyperParamTuner(
            config_path='hyperparam_tuning/example_bayesian_config.json',
            evaluation_function=dummy_eval,
            strategy='bayesian',
            n_calls=3,
            n_random_starts=1
        )
        best_params = tuner.run_tuning_pipeline()
        self.assertIsInstance(best_params, dict)

    def test_grid_strategy(self):
        tuner = HyperParamTuner(
            config_path='hyperparam_tuning/example_grid_config.json',
            evaluation_function=dummy_eval,
            strategy='grid'
        )
        best_params = tuner.run_tuning_pipeline()
        self.assertIsInstance(best_params, dict)

if __name__ == '__main__':
    unittest.main()
