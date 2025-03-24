import unittest
import pandas as pd
import numpy as np
import torch
import logging
import os
import sys
from alignment_checks.bias_detection import BiasDetection
from alignment_checks.ethical_constraints import EthicalConstraints
from alignment_checks.fairness_evaluator import FairnessEvaluator
from alignment_checks.alignment_monitor import AlignmentMonitor

class TestAlignmentMonitor(unittest.TestCase):

    def setUp(self):
        # Dummy Data Setup
        self.df = pd.DataFrame({
            'gender': np.random.choice(['Male', 'Female'], size=100),
            'age': np.random.randint(18, 70, size=100)
        })
        self.predictions = pd.Series(np.random.randint(0, 2, size=100))
        self.probabilities = np.random.rand(100)
        self.labels = pd.Series(np.random.randint(0, 2, size=100))
        self.action_contexts = [
            {
                'action': 'Serve personalized ad',
                'target': 'User123',
                'data_used': {'private': True},
                'predicted_outcome': {
                    'harm': False,
                    'bias_detected': False,
                    'discrimination_detected': False
                },
                'explanation': 'Ad targeting based on purchase history.'
            }
        ]
        self.monitor = AlignmentMonitor(
            sensitive_attrs=['gender'],
            privileged_groups={'gender': ['Male']},
            unprivileged_groups={'gender': ['Female']}
        )

    def test_bias_detection(self):
        """Test BiasDetection runs without error and returns expected keys."""
        detector = self.monitor.bias_detector
        report = detector.run_bias_report(
            data=self.df,
            predictions=self.predictions,
            probabilities=self.probabilities,
            labels=self.labels
        )
        self.assertIn('statistical_parity', report)
        self.assertIn('equal_opportunity', report)

    def test_fairness_evaluator_group_fairness(self):
        """Test FairnessEvaluator for group fairness metrics."""
        evaluator = self.monitor.fairness_evaluator
        group_fairness = evaluator.evaluate_group_fairness(
            data=self.df,
            predictions=self.predictions,
            labels=self.labels
        )
        self.assertIn('privileged', group_fairness)
        self.assertIn('unprivileged', group_fairness)
        self.assertIn('positive_prediction_rate', group_fairness['privileged'])

    def test_fairness_evaluator_predictive_parity(self):
        """Test FairnessEvaluator for predictive parity metrics."""
        evaluator = self.monitor.fairness_evaluator
        predictive_parity = evaluator.evaluate_predictive_parity(
            data=self.df,
            predictions=self.predictions,
            labels=self.labels
        )
        self.assertIn('privileged', predictive_parity)
        self.assertIn('positive_predictive_value', predictive_parity['privileged'])

    def test_fairness_evaluator_individual_fairness(self):
        """Test FairnessEvaluator for individual fairness."""
        evaluator = self.monitor.fairness_evaluator
        individual_unfairness_rate = evaluator.evaluate_individual_fairness(
            data=self.df,
            predictions=self.predictions,
            similarity_function=lambda x, y: 1.0 if abs(x['age'] - y['age']) < 5 else 0.0
        )
        self.assertGreaterEqual(individual_unfairness_rate, 0.0)

    def test_ethical_constraints(self):
        """Test EthicalConstraints enforcement with violation."""
        constraints = self.monitor.ethical_constraints
        action_context_violation = {
            'action': 'Serve personalized ad',
            'target': 'User123',
            'data_used': {'private': True},
            'predicted_outcome': {
                'harm': True,
                'bias_detected': True,
                'discrimination_detected': True
            },
            'explanation': ''
        }
        result = constraints.enforce(action_context_violation)
        self.assertFalse(result)

    def test_alignment_monitor_with_compliance(self):
        """Test full monitor flow with compliant data."""
        report = self.monitor.monitor(
            data=self.df,
            predictions=self.predictions,
            probabilities=self.probabilities,
            labels=self.labels,
            action_contexts=self.action_contexts
        )
        self.assertIn('bias_report', report)
        self.assertIn('fairness', report)
        self.assertIn('ethical_violations', report)

if __name__ == '__main__':
    unittest.main()
