import pandas as pd
import numpy as np
import logging
from alignment_checks.bias_detection import BiasDetection
from alignment_checks.ethical_constraints import EthicalConstraints
from alignment_checks.fairness_evaluator import FairnessEvaluator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AlignmentMonitor:
    """
    Real-time alignment and ethical compliance monitor.
    Integrates bias detection, fairness evaluation, and ethical constraint enforcement.
    """

    def __init__(self, sensitive_attrs, privileged_groups, unprivileged_groups, fairness_thresholds=None, bias_threshold=0.1):
        """
        Args:
            sensitive_attrs (list): Sensitive attributes for bias/fairness checks.
            privileged_groups (dict): Privileged values for each sensitive attribute.
            unprivileged_groups (dict): Unprivileged values for each sensitive attribute.
            fairness_thresholds (dict): Thresholds for fairness disparity checks.
            bias_threshold (float): Threshold for acceptable bias.
        """
        self.bias_detector = BiasDetection(sensitive_attrs, privileged_groups, unprivileged_groups)
        self.ethical_constraints = EthicalConstraints()
        self.fairness_evaluator = FairnessEvaluator(
            sensitive_attr=sensitive_attrs[0],
            privileged_groups=privileged_groups[sensitive_attrs[0]],
            unprivileged_groups=unprivileged_groups[sensitive_attrs[0]]
        )
        self.fairness_thresholds = fairness_thresholds or {
            'statistical_parity': 0.1,
            'predictive_parity': 0.1,
            'individual_unfairness_rate': 0.1
        }
        self.bias_threshold = bias_threshold

    def monitor(self, data, predictions, probabilities, labels, action_contexts):
        """
        Perform a full alignment check on predictions and actions.

        Args:
            data (pd.DataFrame): Input data including sensitive attributes.
            predictions (np.ndarray): Model predictions.
            probabilities (np.ndarray): Predicted probabilities (if applicable).
            labels (np.ndarray): True labels.
            action_contexts (list): List of action contexts for ethical enforcement.

        Returns:
            dict: Comprehensive report of alignment status.
        """
        logger.info("Running alignment monitor...")

        # Step 1: Bias Detection
        bias_report = self.bias_detector.run_bias_report(data, predictions, probabilities, labels)
        parity_diff = abs(bias_report['statistical_parity']['parity_difference'])
        tpr_diff = abs(bias_report['equal_opportunity']['tpr_difference'])

        # Step 2: Fairness Evaluation
        group_fairness = self.fairness_evaluator.evaluate_group_fairness(data, predictions, labels)
        predictive_parity = self.fairness_evaluator.evaluate_predictive_parity(data, predictions, labels)
        individual_unfairness_rate = self.fairness_evaluator.evaluate_individual_fairness(
            data,
            predictions,
            similarity_function=lambda x, y: 1.0 if abs(x['age'] - y['age']) < 5 else 0.0
        )

        fairness_violations = []
        # Check group fairness (statistical parity)
        disparity = abs(
            group_fairness['privileged']['positive_prediction_rate'] - group_fairness['unprivileged']['positive_prediction_rate']
        )
        if disparity > self.fairness_thresholds['statistical_parity']:
            fairness_violations.append('statistical_parity')

        # Check predictive parity
        ppv_disparity = abs(
            predictive_parity['privileged']['positive_predictive_value'] - predictive_parity['unprivileged']['positive_predictive_value']
        )
        if ppv_disparity > self.fairness_thresholds['predictive_parity']:
            fairness_violations.append('predictive_parity')

        # Check individual fairness
        if individual_unfairness_rate > self.fairness_thresholds['individual_unfairness_rate']:
            fairness_violations.append('individual_fairness')

        # Step 3: Ethical Constraints Enforcement
        ethical_violations = []
        for action_context in action_contexts:
            if not self.ethical_constraints.enforce(action_context):
                ethical_violations.append(action_context.get('action'))

        # Step 4: Generate Report
        report = {
            'bias_report': bias_report,
            'fairness': {
                'group_fairness': group_fairness,
                'predictive_parity': predictive_parity,
                'individual_unfairness_rate': individual_unfairness_rate,
                'violations': fairness_violations
            },
            'ethical_violations': ethical_violations,
            'bias_violations': parity_diff > self.bias_threshold or tpr_diff > self.bias_threshold
        }

        logger.info("Alignment monitoring complete. Report generated.")
        logger.info("Fairness violations: %s", fairness_violations)
        logger.info("Ethical violations: %s", ethical_violations)

        return report

if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    # Example dummy data
    df = pd.DataFrame({
        'gender': np.random.choice(['Male', 'Female'], size=100),
        'age': np.random.randint(18, 70, size=100)
    })
    predictions = pd.Series(np.random.randint(0, 2, size=100))
    probabilities = np.random.rand(100)
    labels = pd.Series(np.random.randint(0, 2, size=100))

    action_contexts = [
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

    monitor = AlignmentMonitor(
        sensitive_attrs=['gender'],
        privileged_groups={'gender': ['Male']},
        unprivileged_groups={'gender': ['Female']}
    )

    report = monitor.monitor(
        data=df,
        predictions=predictions,
        probabilities=probabilities,
        labels=labels,
        action_contexts=action_contexts
    )

    print("\nAlignment Monitor Report:")
    print(report)
