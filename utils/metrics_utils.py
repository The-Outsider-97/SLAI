import logging
import os
import json
import yaml
import csv
import pickle
import torch
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ============================================
# Threshold Evaluators
# ============================================

def check_statistical_parity(parity_difference: float, threshold: float) -> Tuple[bool, str]:
    """
    Checks whether the statistical parity difference breaches the defined threshold.
    
    Args:
        parity_difference (float): The calculated parity difference.
        threshold (float): The acceptable threshold limit.

    Returns:
        Tuple[bool, str]: (True if violation exists, message)
    """
    logger.info(f"Checking statistical parity: diff={parity_difference}, threshold={threshold}")
    
    if abs(parity_difference) > threshold:
        message = f"Statistical Parity Violation: {parity_difference:.4f} exceeds threshold {threshold:.4f}"
        logger.warning(message)
        return True, message
    
    return False, "Statistical parity within acceptable range."


def check_equal_opportunity(tpr_difference: float, threshold: float) -> Tuple[bool, str]:
    """
    Checks whether the equal opportunity difference breaches the threshold.
    
    Args:
        tpr_difference (float): The true positive rate difference between groups.
        threshold (float): The maximum allowable difference.
    
    Returns:
        Tuple[bool, str]: (True if violation exists, message)
    """
    logger.info(f"Checking equal opportunity: diff={tpr_difference}, threshold={threshold}")
    
    if abs(tpr_difference) > threshold:
        message = f"Equal Opportunity Violation: {tpr_difference:.4f} exceeds threshold {threshold:.4f}"
        logger.warning(message)
        return True, message

    return False, "Equal opportunity within acceptable range."


def check_predictive_parity(ppv_difference: float, threshold: float) -> Tuple[bool, str]:
    """
    Checks whether the predictive parity difference breaches the threshold.

    Args:
        ppv_difference (float): The positive predictive value difference between groups.
        threshold (float): Maximum allowable difference.

    Returns:
        Tuple[bool, str]: (True if violation exists, message)
    """
    logger.info(f"Checking predictive parity: diff={ppv_difference}, threshold={threshold}")
    
    if abs(ppv_difference) > threshold:
        message = f"Predictive Parity Violation: {ppv_difference:.4f} exceeds threshold {threshold:.4f}"
        logger.warning(message)
        return True, message
    
    return False, "Predictive parity within acceptable range."

def check_individual_fairness(unfairness_rate: float, threshold: float) -> Tuple[bool, str]:
    """
    Checks whether individual fairness breaches threshold.

    Args:
        unfairness_rate (float): Proportion of similar individuals treated differently.
        threshold (float): Maximum acceptable rate.

    Returns:
        Tuple[bool, str]: (True if violation exists, message)
    """
    logger.info(f"Checking individual fairness: rate={unfairness_rate}, threshold={threshold}")

    if unfairness_rate > threshold:
        message = f"Individual Fairness Violation: rate {unfairness_rate:.4f} exceeds threshold {threshold:.4f}"
        logger.warning(message)
        return True, message

    return False, "Individual fairness within acceptable range."

# ============================================
# Performance Evaluators
# ============================================

def evaluate_performance(performance: Dict[str, Any], reward_threshold: float) -> Tuple[bool, str]:
    """
    Evaluate performance metrics from parsed logs.
    
    Args:
        performance (dict): Contains keys like 'best_reward', 'average_reward'.
        reward_threshold (float): Minimum acceptable reward threshold.
    
    Returns:
        Tuple[bool, str]: (True if performance is insufficient, message)
    """
    best_reward = performance.get("best_reward", 0.0)
    logger.info(f"Evaluating performance: best_reward={best_reward}, threshold={reward_threshold}")

    if best_reward < reward_threshold:
        message = f"Performance Violation: best_reward {best_reward:.4f} is below threshold {reward_threshold:.4f}"
        logger.warning(message)
        return True, message

    return False, "Performance within acceptable range."

# ============================================
# Summary and Report Generators
# ============================================

def summarize_violations(*violation_messages: Tuple[bool, str]) -> Dict[str, Any]:
    """
    Collects all violations and summarizes them.
    
    Args:
        *violation_messages (Tuple[bool, str]): Results from threshold checks.
    
    Returns:
        dict: Summary of violations detected.
    """
    logger.info("Summarizing violations...")

    summary = {
        "violations_detected": False,
        "details": []
    }

    for violation, message in violation_messages:
        if violation:
            summary["violations_detected"] = True
            summary["details"].append(message)

    if not summary["violations_detected"]:
        summary["details"].append("No violations detected.")

    logger.info(f"Summary: {summary}")
    return summary

# ============================================
# Example Usage (optional)
# ============================================

if __name__ == "__main__":
    # Example learning curve plot
    example_rewards = [50, 60, 65, 70, 80, 85, 90]
    plot_learning_curve(example_rewards, title="Example Learning Curve", save_path="plots/learning_curve.png")

    # Example bias metrics plot
    plot_bias_metrics(0.14, 0.13, 0.04, save_path="plots/bias_metrics.png")

    # Example time series analysis
    metrics = [
        {"timestamp": "2024-03-24T10:00:00", "best_reward": 60},
        {"timestamp": "2024-03-25T10:00:00", "best_reward": 70},
        {"timestamp": "2024-03-26T10:00:00", "best_reward": 80}
    ]
    summary = time_series_analysis(metrics, "best_reward")
    print("Time-Series Analysis Summary:", summary)
