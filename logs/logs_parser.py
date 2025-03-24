import json
import os
import shutil
import torch
import itertools
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

class RollbackHandler:
    """
    Handles rollback to a previous stable model state.
    Manages model versions, archives, and restoration.
    """

    def __init__(self, models_dir='models/', backup_dir='models/backups/'): 
        """
        Args:
            models_dir (str): Directory where current models are stored.
            backup_dir (str): Directory containing backup versions of models.
        """
        self.models_dir = models_dir
        self.backup_dir = backup_dir

        # Ensure backup directory exists
        os.makedirs(self.backup_dir, exist_ok=True)

    def create_backup(self, version_name):
        """
        Creates a backup of the current model directory.
        """
        backup_path = os.path.join(self.backup_dir, version_name)
        if os.path.exists(backup_path):
            print(f"Backup '{version_name}' already exists.")
            return

        shutil.copytree(self.models_dir, backup_path)
        print(f"Backup '{version_name}' created at {backup_path}.")

    def rollback_model(self, version_name=None):
        """
        Rolls back to a specified backup version or the latest available.
        """
        print("Attempting rollback...")

        backups = sorted(os.listdir(self.backup_dir), reverse=True)

        if not backups:
            print("❌ No backups available to rollback!")
            return

        target_version = version_name if version_name in backups else backups[0]
        target_path = os.path.join(self.backup_dir, target_version)

        if not os.path.exists(target_path):
            print(f"❌ Backup version '{target_version}' not found!")
            return

        # Clear current models
        if os.path.exists(self.models_dir):
            shutil.rmtree(self.models_dir)

        # Restore backup
        shutil.copytree(target_path, self.models_dir)
        print(f"✅ Rollback complete. Restored model version '{target_version}' to {self.models_dir}.")

    def list_backups(self):
        """
        Lists available backup versions.
        """
        backups = sorted(os.listdir(self.backup_dir))
        print("Available backups:")
        for version in backups:
            print(f"- {version}")

# Example placeholders for integration
class DummyTuner:
    def run_tuning_pipeline(self):
        print("[DummyTuner] Running hyperparameter tuning pipeline...")

if __name__ == "__main__":
    # Example usage of RollbackHandler
    rollback_handler = RollbackHandler(models_dir='models/', backup_dir='models/backups/')

    # Create backup before testing rollback
    rollback_handler.create_backup('v1.0')

    # Rollback to a version
    rollback_handler.rollback_model('v1.0')

    # List all available backups
    rollback_handler.list_backups()

    # Example usage of LogsParser with rollback handler
    parser = LogsParser(
        log_dir='logs/',
        bias_threshold=0.1,
        reward_threshold=70.0,
        hyperparam_tuner=DummyTuner(),
        rollback_handler=rollback_handler
    )

    parser.parse_logs()
