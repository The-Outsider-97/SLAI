from alignment_checks.bias_detection import BiasDetection
import pandas as pd
import numpy as np

class AlignmentMonitor:
    """
    Real-time alignment and bias monitoring.
    """

    def __init__(self, bias_threshold=0.1, evaluation_interval=100):
        self.bias_threshold = bias_threshold
        self.evaluation_interval = evaluation_interval
        self.step_counter = 0

        # Example sensitive attributes config
        sensitive_attrs = ['gender', 'race']
        privileged = {'gender': ['Male'], 'race': ['White']}
        unprivileged = {'gender': ['Female'], 'race': ['Black', 'Asian', 'Hispanic']}

        self.bias_detector = BiasDetection(sensitive_attrs, privileged, unprivileged)

    def monitor(self, data: pd.DataFrame, predictions: np.ndarray, probabilities: np.ndarray, labels: np.ndarray):
        """
        Monitor bias during training or inference.
        """
        self.step_counter += 1

        if self.step_counter % self.evaluation_interval != 0:
            return  # Skip this step

        print(f"\n--- Running Alignment Monitor at Step {self.step_counter} ---\n")

        report = self.bias_detector.run_bias_report(data, predictions, probabilities, labels)

        # Example threshold action:
        parity_diff = max(abs(report['statistical_parity']['parity_difference']),
                          abs(report['equal_opportunity']['tpr_difference']))

        if parity_diff > self.bias_threshold:
            print(f"⚠️ Bias threshold breached! Parity Diff = {parity_diff:.3f}")
            self.handle_violation()

    def handle_violation(self):
        """
        Trigger actions: log, stop, rollback, or retrain.
        """
        print("Triggering rollback / retraining...")
        # Placeholders for rollback or retraining logic
        # e.g., rollback_handler.rollback_model()
        # e.g., retrain_model_with_constraints()
