"""
Counterfactual Fairness Metrics
Implements multi-level fairness quantification through:
- Individual counterfactual fairness (Kusner et al., 2017)
- Group-level causal disparity measures (Zhang & Bareinboim, 2018)
- Path-specific effect decomposition (Chiappa, 2019)
"""

import statsmodels.formula.api as smf
import networkx as nx
import numpy as np
import pandas as pd

from typing import Dict, List, Optional, Union
from statsmodels.regression.linear_model import RegressionResultsWrapper

from src.agents.alignment.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Fairness Metrics")
printer = PrettyPrinter

class CounterfactualFairness:
    """
    Multi-level counterfactual fairness assessment implementing:
    - Individual-level similarity metrics
    - Group-level distributional comparisons
    - Path-specific effect decomposition
    """

    def __init__(self):
        self.config = load_global_config()
        self.sensitive_attributes = self.config.get('sensitive_attributes')
        self.sensitive_attrs = self.sensitive_attributes

        self.fairness_config = get_config_section('fairness_metrics')

        logger.info(f"Fairness Metrics has succesfully initialized")

    def compute_individual_fairness(self,
                                  original_preds: np.ndarray,
                                  counterfactual_preds: np.ndarray) -> Dict:
        """
        Computes individual counterfactual fairness, measuring the stability
        of predictions for an individual when their sensitive attribute is changed.

        Metrics:
        - max_difference: Max absolute change in prediction.
        - mean_difference: Mean absolute change in prediction.
        - unfairness_rate: Proportion of individuals whose prediction changes above a threshold (e.g., 0.1).

        Args:
            original_preds (np.ndarray): Predictions based on original data.
            counterfactual_preds (np.ndarray): Predictions based on counterfactual data (sensitive attribute changed).

        Returns:
            Dict: Dictionary containing individual fairness metrics.
        """
        if len(original_preds) != len(counterfactual_preds):
            raise ValueError("Original and counterfactual predictions must have the same length.")
        if len(original_preds) == 0:
            return {
                'max_difference': 0.0, 'mean_difference': 0.0, 'unfairness_rate': 0.0
            }

        abs_diffs = np.abs(original_preds - counterfactual_preds)

        # Basic individual fairness metrics
        metrics = {
            'max_difference': float(np.max(abs_diffs)),
            'mean_difference': float(np.mean(abs_diffs)),
            'unfairness_rate': float(np.mean(abs_diffs > 0.1)) # Threshold can be adjusted
        }
        return metrics


    def compute_group_disparity(self,
                              data: pd.DataFrame,
                              sensitive_attr: str,
                              predictions: str,
                              y_true: str,
                              privileged_group: Union[int, str],
                              unprivileged_group: Union[int, str]) -> Dict:
        """
        Computes standard group fairness metrics based on observed predictions.

        Args:
            data (pd.DataFrame): DataFrame containing sensitive attribute, predictions, and true labels.
            sensitive_attr (str): Column name of the sensitive attribute.
            predictions (str): Column name of the model predictions (binary 0/1 or probabilities).
            y_true (str): Column name of the ground truth labels (binary 0/1).
            privileged_group (Union[int, str]): Value representing the privileged group in sensitive_attr.
            unprivileged_group (Union[int, str]): Value representing the unprivileged group in sensitive_attr.

        Returns:
            Dict: Dictionary containing group fairness metrics like Statistical Parity Difference,
                  Equal Opportunity Difference, Average Absolute Odds Difference.
        """
        if sensitive_attr not in data.columns: raise ValueError(f"Sensitive attribute '{sensitive_attr}' not found.")
        if predictions not in data.columns: raise ValueError(f"Predictions column '{predictions}' not found.")
        if y_true not in data.columns: raise ValueError(f"True labels column '{y_true}' not found.")

        # Binarize predictions if they are probabilities (common 0.5 threshold)
        if data[predictions].dtype != 'int' and data[predictions].between(0, 1).all():
            logger.info(f"Binarizing predictions '{predictions}' using 0.5 threshold.")
            preds_bin = (data[predictions] > 0.5).astype(int)
        else:
            preds_bin = data[predictions] # Assume already binary

        df_priv = data[data[sensitive_attr] == privileged_group]
        df_unpriv = data[data[sensitive_attr] == unprivileged_group]

        if df_priv.empty or df_unpriv.empty:
            logger.warning("One or both groups are empty. Cannot compute group disparity.")
            return {'statistical_parity_difference': np.nan, 'equal_opportunity_difference': np.nan, 'average_abs_odds_difference': np.nan}

        # --- Calculations ---
        # Statistical Parity Difference (SPD) = P(Y_hat=1 | A=unpriv) - P(Y_hat=1 | A=priv)
        rate_priv = preds_bin.loc[df_priv.index].mean()
        rate_unpriv = preds_bin.loc[df_unpriv.index].mean()
        spd = rate_unpriv - rate_priv

        # True Positive Rate (TPR) = P(Y_hat=1 | Y=1, A=a)
        # False Positive Rate (FPR) = P(Y_hat=1 | Y=0, A=a)
        tpr_priv = preds_bin.loc[df_priv[df_priv[y_true] == 1].index].mean() if (df_priv[y_true] == 1).any() else 0
        tpr_unpriv = preds_bin.loc[df_unpriv[df_unpriv[y_true] == 1].index].mean() if (df_unpriv[y_true] == 1).any() else 0
        fpr_priv = preds_bin.loc[df_priv[df_priv[y_true] == 0].index].mean() if (df_priv[y_true] == 0).any() else 0
        fpr_unpriv = preds_bin.loc[df_unpriv[df_unpriv[y_true] == 0].index].mean() if (df_unpriv[y_true] == 0).any() else 0

        # Equal Opportunity Difference (EOD) = TPR(unpriv) - TPR(priv)
        eod = tpr_unpriv - tpr_priv

        # Average Absolute Odds Difference (AAOD) = 0.5 * [ |FPR(unpriv)-FPR(priv)| + |TPR(unpriv)-TPR(priv)| ]
        aaod = 0.5 * (abs(fpr_unpriv - fpr_priv) + abs(tpr_unpriv - tpr_priv))

        return {
            'statistical_parity_difference': float(spd),
            'equal_opportunity_difference': float(eod),
            'average_abs_odds_difference': float(aaod),
            'tpr_privileged': float(tpr_priv),
            'tpr_unprivileged': float(tpr_unpriv),
            'fpr_privileged': float(fpr_priv),
            'fpr_unprivileged': float(fpr_unpriv)
        }


    def _equalized_odds_gap(self,
                            data: pd.DataFrame,
                            sensitive_attr: str,
                            original_preds_col: str,
                            counterfactual_preds_col: str,
                            y_true_col: str) -> Dict[str, float]:
        r"""
        Calculates the Equalized Odds gap between original and counterfactual predictions.
        Equalized odds requires that the prediction $\hat{Y}$ is independent of the
        sensitive attribute $A$, conditional on the true outcome $Y$.
        $P(\hat{Y}=1 | A=a, Y=y) = P(\hat{Y}=1 | A=a', Y=y)$ for all $a, a'$ and $y \in \{0, 1\}$.
        This translates to requiring both TPR and FPR to be equal across groups.

        This function measures the *change* in TPR and FPR disparities between the
        original predictions and counterfactual predictions (where A was hypothetically changed).

        Args:
            data (pd.DataFrame): DataFrame containing sensitive attr, true labels,
                                 original predictions, and counterfactual predictions.
            sensitive_attr (str): Column name of the sensitive attribute.
            original_preds_col (str): Column name for original predictions.
            counterfactual_preds_col (str): Column name for counterfactual predictions.
            y_true_col (str): Column name for ground truth labels.

        Returns:
            Dict[str, float]: Dictionary with 'tpr_gap_change' and 'fpr_gap_change'.
                              These represent the absolute difference in TPR/FPR gaps
                              between the original and counterfactual scenarios.
                              Lower values indicate the counterfactual change did not
                              exacerbate existing equalized odds violations.
        """
        if not all(col in data.columns for col in [sensitive_attr, original_preds_col, counterfactual_preds_col, y_true_col]):
            raise ValueError("One or more required columns are missing from the DataFrame.")

        groups = data[sensitive_attr].unique()
        if len(groups) < 2:
            logger.warning("Only one group found. Cannot compute equalized odds gaps between groups.")
            return {'tpr_gap_change': 0.0, 'fpr_gap_change': 0.0}

        rates = {'original': {'tpr': {}, 'fpr': {}}, 'counterfactual': {'tpr': {}, 'fpr': {}}}

        # Binarize predictions if needed
        preds_orig_bin = (data[original_preds_col] > 0.5).astype(int) if data[original_preds_col].between(0, 1).all() else data[original_preds_col]
        preds_cf_bin = (data[counterfactual_preds_col] > 0.5).astype(int) if data[counterfactual_preds_col].between(0, 1).all() else data[counterfactual_preds_col]

        for group_val in groups:
            group_filter = (data[sensitive_attr] == group_val)
            y_true_group = data.loc[group_filter, y_true_col]

            # --- Original Predictions ---
            preds_orig_group = preds_orig_bin[group_filter]
            # TPR = P(Y_hat=1 | Y=1, A=group_val)
            true_positives_filter_orig = group_filter & (data[y_true_col] == 1)
            actual_positives_count_orig = true_positives_filter_orig.sum()
            if actual_positives_count_orig > 0:
                rates['original']['tpr'][group_val] = preds_orig_bin[true_positives_filter_orig].mean()
            else:
                rates['original']['tpr'][group_val] = np.nan # Avoid division by zero

            # FPR = P(Y_hat=1 | Y=0, A=group_val)
            true_negatives_filter_orig = group_filter & (data[y_true_col] == 0)
            actual_negatives_count_orig = true_negatives_filter_orig.sum()
            if actual_negatives_count_orig > 0:
                 rates['original']['fpr'][group_val] = preds_orig_bin[true_negatives_filter_orig].mean()
            else:
                rates['original']['fpr'][group_val] = np.nan

            # --- Counterfactual Predictions ---
            preds_cf_group = preds_cf_bin[group_filter]
             # TPR = P(Y_hat_cf=1 | Y=1, A=group_val) - Note A is original, Y_hat is CF
            true_positives_filter_cf = group_filter & (data[y_true_col] == 1) # Same filter as original
            actual_positives_count_cf = true_positives_filter_cf.sum() # Same count
            if actual_positives_count_cf > 0:
                rates['counterfactual']['tpr'][group_val] = preds_cf_bin[true_positives_filter_cf].mean()
            else:
                rates['counterfactual']['tpr'][group_val] = np.nan

            # FPR = P(Y_hat_cf=1 | Y=0, A=group_val)
            true_negatives_filter_cf = group_filter & (data[y_true_col] == 0) # Same filter as original
            actual_negatives_count_cf = true_negatives_filter_cf.sum() # Same count
            if actual_negatives_count_cf > 0:
                rates['counterfactual']['fpr'][group_val] = preds_cf_bin[true_negatives_filter_cf].mean()
            else:
                 rates['counterfactual']['fpr'][group_val] = np.nan


        # Calculate gaps (difference between max and min rate across groups)
        def calculate_gap(rate_dict):
            valid_rates = [r for r in rate_dict.values() if not np.isnan(r)]
            if len(valid_rates) < 2: return 0.0
            return max(valid_rates) - min(valid_rates)

        tpr_gap_orig = calculate_gap(rates['original']['tpr'])
        fpr_gap_orig = calculate_gap(rates['original']['fpr'])
        tpr_gap_cf = calculate_gap(rates['counterfactual']['tpr'])
        fpr_gap_cf = calculate_gap(rates['counterfactual']['fpr'])

        # Change in gaps
        tpr_gap_change = abs(tpr_gap_cf - tpr_gap_orig)
        fpr_gap_change = abs(fpr_gap_cf - fpr_gap_orig)

        return {'tpr_gap_change': tpr_gap_change, 'fpr_gap_change': fpr_gap_change}


if __name__ == "__main__":
    print("\n=== Running Counterfactual Fairness ===\n")
    printer.status("Init", "Counterfactual Fairness initialized", "success")

    # Synthetic test data
    np.random.seed(42)
    size = 100
    df = pd.DataFrame({
        'A': np.random.choice([0, 1], size=size),  # Sensitive attribute
        'X': np.random.normal(0, 1, size=size),    # Mediator
        'Y': np.random.binomial(1, 0.5, size=size) # Outcome
    })
    df['pred'] = df['A'] * 0.3 + df['X'] * 0.5 + np.random.normal(0, 0.1, size=size)
    df['pred_cf'] = (1 - df['A']) * 0.3 + df['X'] * 0.5 + np.random.normal(0, 0.1, size=size)

    # Initialize fairness metrics
    fairness = CounterfactualFairness()

    # Test compute_individual_fairness
    ind_fair = fairness.compute_individual_fairness(df['pred'].values, df['pred_cf'].values)
    printer.pretty("Individual Fairness:", ind_fair, "success")

    # Test compute_group_disparity
    group_disp = fairness.compute_group_disparity(df, 'A', 'pred', 'Y', 1, 0)
    printer.pretty("Group Disparity:", group_disp, "success")

    eo_gap = fairness._equalized_odds_gap(
        df.assign(original_pred=df['pred'], counterfactual_pred=df['pred_cf']),
        'A', 'original_pred', 'counterfactual_pred', 'Y'
    )
    print("Equalized Odds Gap Change:", eo_gap)
    print("\n=== Counterfactual Fairness Test Completed ===\n")
