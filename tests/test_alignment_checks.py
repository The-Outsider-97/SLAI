import unittest
import torch
import os
from alignment_checks.bias_detection import BiasDetection
from alignment_checks.ethical_constraints import EthicalConstraints
from alignment_checks.fairness_evaluator import FairnessEvaluator
import pandas as pd

class TestAlignmentChecks(unittest.TestCase):

    def setUp(self):
        self.bias_detector = BiasDetection(
            sensitive_attrs=["gender"],
            privileged_groups={"gender": ["Male"]},
            unprivileged_groups={"gender": ["Female"]}
        )
        self.constraints = EthicalConstraints()
        self.fairness = FairnessEvaluator(
            sensitive_attr="gender",
            privileged_groups=["Male"],
            unprivileged_groups=["Female"]
        )

    def test_bias_report(self):
        data = pd.DataFrame({"gender": ["Male", "Female", "Female", "Male"]})
        predictions = pd.Series([1, 0, 1, 1])
        probabilities = pd.Series([0.9, 0.4, 0.8, 0.95])
        labels = pd.Series([1, 0, 1, 1])

        report = self.bias_detector.run_bias_report(data, predictions, probabilities, labels)
        self.assertIn("statistical_parity", report)

    def test_ethical_constraints_pass(self):
        action_context = {
            "predicted_outcome": {
                "harm": False,
                "bias_detected": False,
                "discrimination_detected": False
            }
        }
        self.assertTrue(self.constraints.enforce(action_context))

    def test_ethical_constraints_fail(self):
        action_context = {
            "predicted_outcome": {
                "harm": True
            }
        }
        self.assertFalse(self.constraints.enforce(action_context))

if __name__ == "__main__":
    unittest.main()
