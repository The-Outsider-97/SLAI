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

# ==========================================
# Actual Usage in Main AutoTune Pipeline
# ==========================================
if __name__ == "__main__":
    import json
    from utils.metrics_utils_extension import (
        plot_learning_curve,
        plot_bias_metrics,
        time_series_analysis
    )

    # Simulated parsed_metrics.json loading (after logs_parser.parse_logs() execution)
    with open("logs/parsed_metrics.json", "r") as f:
        parsed_metrics = json.load(f)

    # === Plot Learning Curve ===
    learning_curve_data = parsed_metrics.get("performance", {}).get("learning_curve", [])
    if learning_curve_data:
        plot_learning_curve(
            rewards=learning_curve_data,
            title=f"Learning Curve - Model {parsed_metrics.get('model_version', 'N/A')}",
            save_path=f"plots/{parsed_metrics.get('run_id', 'run')}_learning_curve.png"
        )

    # === Plot Bias Metrics ===
    parity_diff = parsed_metrics.get("statistical_parity", {}).get("parity_difference", 0.0)
    tpr_diff = parsed_metrics.get("equal_opportunity", {}).get("tpr_difference", 0.0)
    ppv_diff = parsed_metrics.get("predictive_parity", {}).get("ppv_difference", 0.0)

    plot_bias_metrics(
        parity_diff=parity_diff,
        tpr_diff=tpr_diff,
        ppv_diff=ppv_diff,
        save_path=f"plots/{parsed_metrics.get('run_id', 'run')}_bias_metrics.png"
    )

    # === Time-Series Analysis Across Multiple Runs ===
    import glob

    all_metrics = []
    for file_path in glob.glob("logs/history/parsed_metrics_run_*.json"):
        with open(file_path, "r") as f:
            all_metrics.append(json.load(f))

    if all_metrics:
        trend_summary = time_series_analysis(
            metrics=all_metrics,
            metric_key="performance.best_reward"
        )
        print("\nTime-Series Trend Summary:", trend_summary)

    print("✅ Metrics visualization and time-series analysis complete.")
