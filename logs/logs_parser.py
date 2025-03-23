import json
import os
import numpy as np
import pandas as pd

class LogsParser:
    """
    Parses logs for metrics, biases, and alignment breaches.
    Triggers retraining or rollback when alignment thresholds are violated.
    """

    def __init__(self, log_dir='logs/', bias_threshold=0.1, reward_threshold=70.0, hyperparam_tuner=None, rollback_handler=None):
        """
        Args:
            log_dir (str): Directory containing log files.
            bias_threshold (float): Maximum allowed parity or TPR difference.
            reward_threshold (float): Minimum acceptable reward performance.
            hyperparam_tuner (object): Optional tuner object to handle retraining.
            rollback_handler (object): Optional rollback handler to revert to previous models.
        """
        self.log_dir = log_dir
        self.bias_threshold = bias_threshold
        self.reward_threshold = reward_threshold
        self.hyperparam_tuner = hyperparam_tuner
        self.rollback_handler = rollback_handler

    def parse_logs(self):
        """
        Parse all logs and evaluate bias metrics and reward performance.
        """
        print(f"\n=== Parsing logs from {self.log_dir} ===\n")
        log_files = [file for file in os.listdir(self.log_dir) if file.endswith('.json')]

        if not log_files:
            print("No log files found.")
            return

        for log_file in log_files:
            print(f"\n--- Evaluating {log_file} ---")
            log_path = os.path.join(self.log_dir, log_file)
            with open(log_path, 'r') as f:
                try:
                    log_data = json.load(f)
                    self.evaluate_log(log_data)
                except json.JSONDecodeError:
                    print(f"Failed to parse {log_file}: Invalid JSON.")

    def evaluate_log(self, log_data):
        """
        Evaluate individual log for bias breaches and reward thresholds.
        """
        # Extract relevant metrics
        parity_diff = abs(log_data.get('statistical_parity', {}).get('parity_difference', 0.0))
        tpr_diff = abs(log_data.get('equal_opportunity', {}).get('tpr_difference', 0.0))
        reward_score = log_data.get('performance', {}).get('best_reward', 0.0)

        # Evaluate Bias
        if parity_diff > self.bias_threshold or tpr_diff > self.bias_threshold:
            print(f"⚠️ Detected bias breach! Parity Diff = {parity_diff:.3f}, TPR Diff = {tpr_diff:.3f}")
            self.trigger_action(reason="Bias breach")

        # Evaluate Performance
        if reward_score < self.reward_threshold:
            print(f"⚠️ Poor performance detected! Best reward = {reward_score:.2f}")
            self.trigger_action(reason="Low reward performance")

    def trigger_action(self, reason):
        """
        Decide on retraining or rollback depending on issue and availability of components.
        """
        print(f"Initiating action due to: {reason}")

        if self.rollback_handler:
            print("Triggering rollback to previous stable model...")
            self.rollback_handler.rollback_model()

        if self.hyperparam_tuner:
            print("Triggering hyperparameter tuning and retraining...")
            self.hyperparam_tuner.run_tuning_pipeline()

        if not self.rollback_handler and not self.hyperparam_tuner:
            print("No automated systems available! Manual intervention required.")

# Example placeholders for integration
class DummyTuner:
    def run_tuning_pipeline(self):
        print("[DummyTuner] Running hyperparameter tuning pipeline...")

class DummyRollbackHandler:
    def rollback_model(self):
        print("[DummyRollbackHandler] Rolling back to previous stable model...")

if __name__ == "__main__":
    # Example usage
    parser = LogsParser(
        log_dir='logs/',
        bias_threshold=0.1,
        reward_threshold=70.0,
        hyperparam_tuner=DummyTuner(),
        rollback_handler=DummyRollbackHandler()
    )

    parser.parse_logs()
