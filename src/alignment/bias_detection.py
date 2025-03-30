import logging
import queue
import torch
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.calibration import calibration_curve

# === Logger Setup ===
from logs.logger import get_logger, get_log_queue
from utils.logger import setup_logger
logger = setup_logger("SLAI", level=logging.DEBUG)

log_queue = get_log_queue()
metric_queue = queue.Queue()

# ============================================
# Main Interface
# ============================================
class BiasDetection:
    """
    Comprehensive bias detection with intersectional analysis, statistical parity,
    equal opportunity, and calibration.
    """

    def __init__(self, sensitive_attrs: list, privileged_groups: dict, unprivileged_groups: dict):
        """
        Args:
            sensitive_attrs (list): Column names of sensitive attributes (e.g., ['gender', 'race']).
            privileged_groups (dict): Privileged values for each attribute.
            unprivileged_groups (dict): Unprivileged values for each attribute.
        """
        self.sensitive_attrs = sensitive_attrs
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups

    # ============================================
    # Check if your dataset is imbalanced across sensitive attributes
    # ============================================
    def check_data_balance(self, data: pd.DataFrame):
        """
        Analyze the distribution of sensitive attributes in the dataset.
        """
        print("\nData Balance Check:")
        for attr in self.sensitive_attrs:
            distribution = data[attr].value_counts(normalize=True)
            print(f"\nDistribution for {attr}:")
            print(distribution)
            sns.barplot(x=distribution.index, y=distribution.values)
            plt.title(f'Distribution of {attr}')
            plt.ylabel('Proportion')
            plt.show()

    # ============================================
    # Probability of positive outcomes shouldn't depend on sensitive attributes
    # ============================================
    def check_statistical_parity(self, data: pd.DataFrame, predictions: np.ndarray):
        """
        Check for statistical parity across all intersectional groups.
        """
        df = data.copy()
        df['predictions'] = predictions
        print("\nStatistical Parity for Intersectional Groups:")

        for group_combination in self._get_intersectional_groups():
            group_df = self._filter_group(df, group_combination)
            if len(group_df) == 0:
                continue
            pos_rate = group_df['predictions'].mean()
            print(f"Group {group_combination}: Positive prediction rate = {pos_rate:.3f}")

    # ============================================
    # True positive rates should be equal for all groups
    # ============================================
    def check_equal_opportunity(self, data: pd.DataFrame, predictions: np.ndarray, labels: np.ndarray):
        """
        Check for equal opportunity (TPR parity) across intersectional groups.
        """
        df = data.copy()
        df['predictions'] = predictions
        df['labels'] = labels

        print("\nEqual Opportunity (TPR) for Intersectional Groups:")
        for group_combination in self._get_intersectional_groups():
            group_df = self._filter_group(df, group_combination)
            if len(group_df) == 0:
                continue
            tp = ((group_df['predictions'] == 1) & (group_df['labels'] == 1)).sum()
            pos = (group_df['labels'] == 1).sum()
            tpr = tp / pos if pos > 0 else 0.0
            print(f"Group {group_combination}: TPR = {tpr:.3f}")

    # ============================================
    # Check if probabilities are calibrated for different groups
    # ============================================
    def check_calibration(self, data: pd.DataFrame, probabilities: np.ndarray):
        """
        Calibration curve check for intersectional groups.
        """
        df = data.copy()
        df['probabilities'] = probabilities

        plt.figure(figsize=(10, 8))
        for group_combination in self._get_intersectional_groups():
            group_df = self._filter_group(df, group_combination)
            if len(group_df) == 0:
                continue
            prob_true, prob_pred = calibration_curve(group_df['labels'], group_df['probabilities'], n_bins=10)
            label = f"{'-'.join([f'{k}:{v}' for k, v in group_combination.items()])}"
            plt.plot(prob_pred, prob_true, marker='o', label=label)

        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curves by Intersectional Group')
        plt.legend()
        plt.show()

    def _get_intersectional_groups(self):
        """
        Generate all combinations of privileged and unprivileged groups.
        """
        keys = self.sensitive_attrs
        values = [self.privileged_groups[attr] + self.unprivileged_groups[attr] for attr in keys]
        for combination in product(*values):
            yield dict(zip(keys, combination))

    def _filter_group(self, df: pd.DataFrame, group_combination: dict):
        """
        Filter dataframe for rows matching the group combination.
        """
        mask = pd.Series([True] * len(df))
        for attr, value in group_combination.items():
            mask &= (df[attr] == value)
        return df[mask]

    # ============================================
    # Combine everything into a comprehensive report
    # ============================================
    def run_bias_report(self, data: pd.DataFrame, predictions: np.ndarray, probabilities: np.ndarray, labels: np.ndarray):
        """
        Run all bias checks and print a comprehensive report.
        """
        print("\n========= Running Intersectional Bias Report =========\n")
        self.check_data_balance(data)
        self.check_statistical_parity(data, predictions)
        self.check_equal_opportunity(data, predictions, labels)

        if probabilities is not None:
            self.check_calibration(data, probabilities)

        print("\n========= End of Bias Report =========\n")
