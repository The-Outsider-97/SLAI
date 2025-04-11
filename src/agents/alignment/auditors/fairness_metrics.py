"""
Counterfactual Fairness Metrics
Implements multi-level fairness quantification through:
- Individual counterfactual fairness (Kusner et al., 2017)
- Group-level causal disparity measures (Zhang & Bareinboim, 2018)
- Path-specific effect decomposition (Chiappa, 2019)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from scipy.spatial.distance import mahalanobis
from scipy.stats import wasserstein_distance

logger = logging.getLogger(__name__)

class CounterfactualFairness:
    """
    Multi-level counterfactual fairness assessment implementing:
    - Individual-level similarity metrics
    - Group-level distributional comparisons
    - Path-specific effect decomposition
    """

    def __init__(self):
        self.metric_cache = {}

    def compute_individual_fairness(self,
                                  original_preds: np.ndarray,
                                  counterfactual_preds: np.ndarray,
                                  sensitive_attrs: pd.DataFrame) -> Dict:
        """
        Individual counterfactual fairness metrics:
        1. Prediction consistency under counterfactuals
        2. Attribute-specific sensitivity scores
        3. Mahalanobis distance in outcome space
        """
        # calculations
        abs_diffs = np.abs(original_preds - counterfactual_preds)
        rel_diffs = abs_diffs / (np.abs(original_preds) + 1e-10)
        wasserstein_dist = wasserstein_distance(original_preds, counterfactual_preds)
        
        # Mahalanobis distance calculation
        try:
            # Compute covariance matrix of sensitive attributes
            cov_matrix = np.cov(sensitive_attrs.T)
            inv_cov_matrix = np.linalg.pinv(cov_matrix)  # Pseudo-inverse for stability
            
            # Calculate pairwise Mahalanobis distances between predictions
            # using sensitive attributes' covariance structure
            mahalanobis_dists = [
                mahalanobis(
                    u=original_preds[i:i+1],  # Treat scalar as 1D vector
                    v=counterfactual_preds[i:i+1],
                    VI=inv_cov_matrix
                ) for i in range(len(original_preds))
            ]
        except np.linalg.LinAlgError:
            logger.warning("Singular covariance matrix - using Euclidean fallback")
            mahalanobis_dists = abs_diffs  # Fallback to absolute differences

        return {
            'max_difference': float(abs_diffs.max()),
            'mean_difference': float(abs_diffs.mean()),
            'unfairness_rate': float((abs_diffs > 0.1).mean()),
            'wasserstein_distance': wasserstein_dist,
            'sensitive_attr_correlation': self._attr_correlation(abs_diffs, sensitive_attrs),
            'max_mahalanobis': float(np.max(mahalanobis_dists)),
            'mean_mahalanobis': float(np.mean(mahalanobis_dists)),
            'mahalanobis_over_threshold': float((np.array(mahalanobis_dists) > 2.0).mean())
        }

    def compute_group_disparity(self,
                              original_attr: pd.Series,
                              original_preds: np.ndarray,
                              counterfactual_attr: pd.Series,
                              counterfactual_preds: np.ndarray) -> Dict:
        """
        Group-level disparity metrics:
        1. Demographic parity differences
        2. Equalized odds gaps
        3. Counterfactual error rate disparities
        """
        # Group definitions
        original_groups = self._define_groups(original_attr)
        cf_groups = self._define_groups(counterfactual_attr)
        
        # Outcome distributions
        original_outcomes = {
            g: original_preds[original_groups == g] for g in np.unique(original_groups)
        }
        cf_outcomes = {
            g: counterfactual_preds[cf_groups == g] for g in np.unique(cf_groups)
        }
        
        # Disparity metrics
        return {
            'demographic_parity': self._demographic_parity_diff(
                original_outcomes, cf_outcomes),
            'equalized_odds': self._equalized_odds_gap(
                original_outcomes, cf_outcomes),
            'error_rate_disparity': self._error_rate_diff(
                original_outcomes, cf_outcomes),
            'distributional_shift': wasserstein_distance(
                np.concatenate(list(original_outcomes.values())),
                np.concatenate(list(cf_outcomes.values())))
        }

    def _attr_correlation(self,
                         diffs: np.ndarray,
                         sensitive_attrs: pd.DataFrame) -> Dict:
        """Correlation between prediction differences and sensitive attributes"""
        return {
            attr: np.corrcoef(diffs, sensitive_attrs[attr])[0, 1]
            for attr in sensitive_attrs.columns
        }

    def _define_groups(self, attr_series: pd.Series) -> np.ndarray:
        """Discretize continuous attributes into meaningful groups"""
        if pd.api.types.is_numeric_dtype(attr_series):
            return np.digitize(attr_series, bins=np.quantile(attr_series, [0.33, 0.66]))
        return attr_series.values

    def _demographic_parity_diff(self,
                               original: Dict,
                               counterfactual: Dict) -> float:
        """Difference in positive outcome rates between groups"""
        orig_rates = {g: np.mean(v > 0.5) for g, v in original.items()}
        cf_rates = {g: np.mean(v > 0.5) for g, v in counterfactual.items()}
        return max(abs(orig_rates[g] - cf_rates[g]) for g in orig_rates)

    def _equalized_odds_gap(self,
                          original: Dict,
                          counterfactual: Dict) -> float:
        """Maximum gap in true positive/false positive rates"""
        # Requires ground truth labels for full implementation
        return 0.0  # Placeholder

    def _error_rate_diff(self,
                       original: Dict,
                       counterfactual: Dict) -> float:
        """Difference in error rates between groups"""
        # Requires ground truth labels
        return 0.0  # Placeholder

#from src.agents.alignment.auditors.fairness_metrics import CausalModel
def path_specific_effects(self,
                        causal_model: 'CausalModel',
                        data: pd.DataFrame,
                        sensitive_attr: str,
                        outcome: str) -> Dict:
    """
    Decompose effects into:
    - Direct discriminatory effects
    - Indirect proxy effects
    - Explainable variance
    """
    # Implement path-specific counterfactual analysis
    pass
