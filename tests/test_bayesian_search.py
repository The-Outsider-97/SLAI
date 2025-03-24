import unittest
import torch
import os
from hyperparam_tuning.bayesian_search import BayesianSearch

def dummy_eval(params):
    return -((params["learning_rate"] - 0.01) ** 2 + (params["num_layers"] - 3) ** 2)

class TestBayesianSearch(unittest.TestCase):

    def setUp(self):
        self.search = BayesianSearch(
            config_file="hyperparam_tuning/example_bayesian_config.json",
            evaluation_function=dummy_eval,
            n_calls=5,
            n_random_starts=2
        )

    def test_run_search(self):
        best_params = self.search.run_search()
        self.assertIn("learning_rate", best_params)
        self.assertIn("num_layers", best_params)

if __name__ == "__main__":
    unittest.main()
