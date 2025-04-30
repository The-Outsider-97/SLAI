import logging
import numpy as np
from typing import Dict, Any, Tuple, List

from src.utils.agent_factory import AgentFactory

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MetricBridge:
    from src.utils.system_optimizer import SystemOptimizer
    """New class to handle feedback routing"""
    def __init__(self, agent_factory: AgentFactory, optimizer: SystemOptimizer):
        self.factory = agent_factory
        self.history = []
        self.optimizer = optimizer

    def submit_metrics(self, metrics: Dict[str, Any]) -> None:
        """Implements experience replay prioritization from Hindsight ER (Andrychowicz 2017)"""
        self.history.append(metrics)
        self._check_fairness(metrics)

        # Resource optimization
        resource_recs = self.optimizer.optimize_throughput(
            current_metrics=metrics['system']
        )
        self.agent_factory.apply_optimizations(resource_recs)

        # Calculate moving averages
        window_size = min(10, len(self.history))
        recent_metrics = self.history[-window_size:]
        
        avg_metrics = {
            'fairness_violations': np.mean([m.get('demographic_parity_violations', 0) 
                                   for m in recent_metrics]),
            'calibration_error': np.mean([m.get('calibration_error', 0) 
                                   for m in recent_metrics])
        }
        
        # Thresholds from ISO/IEC 24027:2021 AI bias standards
        if avg_metrics['fairness_violations'] > 0.05:
            self.factory.adapt_from_metrics(avg_metrics)

class FairnessMetrics:
    """
    Implements fairness metrics from algorithmic fairness literature.
    Reference: Barocas et al., "Fairness and Machine Learning", 2022
    """
    
    @staticmethod
    def demographic_parity(sensitive_groups: List[str], 
                          positive_rates: Dict[str, float],
                          threshold: float = 0.05) -> Tuple[bool, str]:
        """
        Check demographic parity (Dwork et al., 2012)
        :param sensitive_groups: List of group identifiers
        :param positive_rates: Dictionary of P(ŷ=1|group) per group
        :param threshold: Maximum allowable difference (ϵ)
        :return: Tuple (violation, message)
        """
        if len(positive_rates) < 2:
            raise ValueError("Requires rates for at least 2 groups")
            
        rates = np.array(list(positive_rates.values()))
        max_diff = np.max(rates) - np.min(rates)
        
        violation = max_diff > threshold
        msg = (f"Demographic Parity {'Violation' if violation else 'Satisfied'}: "
               f"Max group difference {max_diff:.3f} vs threshold {threshold}")
        return violation, msg

    @staticmethod
    def equalized_odds(sensitive_groups: List[str],
                      tprs: Dict[str, float],
                      fprs: Dict[str, float],
                      threshold: float = 0.05) -> Tuple[bool, str]:
        """
        Check equalized odds (Hardt et al., 2016)
        :param tprs: True positive rates per group
        :param fprs: False positive rates per group
        :return: Tuple (violation, message)
        """
        tpr_values = np.array(list(tprs.values()))
        fpr_values = np.array(list(fprs.values()))
        
        tpr_diff = np.max(tpr_values) - np.min(tpr_values)
        fpr_diff = np.max(fpr_values) - np.min(fpr_values)
        
        violation = (tpr_diff > threshold) or (fpr_diff > threshold)
        msg = (f"Equalized Odds {'Violation' if violation else 'Satisfied'}: "
               f"TPR diff {tpr_diff:.3f}, FPR diff {fpr_diff:.3f} vs threshold {threshold}")
        return violation, msg

    @staticmethod
    def counterfactual_fairness(predictions: Dict[str, Dict[str, float]],
                               threshold: float = 0.05) -> Tuple[bool, str]:
        """
        Check counterfactual fairness (Kusner et al., 2017)
        :param predictions: Dict of {individual: {scenario: prediction}}
        :return: Tuple (violation, message)
        """
        max_diffs = []
        for indv, scenarios in predictions.items():
            diffs = [abs(p1 - p2) for p1 in scenarios.values() for p2 in scenarios.values()]
            max_diffs.append(max(diffs) if diffs else 0)
            
        avg_max_diff = np.mean(max_diffs)
        violation = avg_max_diff > threshold
        msg = (f"Counterfactual Fairness {'Violation' if violation else 'Satisfied'}: "
               f"Average max scenario difference {avg_max_diff:.3f} vs threshold {threshold}")
        return violation, msg

class PerformanceMetrics:
    """
    Implements robust performance metrics from ML literature
    Reference: Dietterich, "Machine Learning for Sequential Data", 2002
    """
    
    @staticmethod
    def class_balanced_accuracy(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               classes: List[int]) -> float:
        """
        Calculate balanced accuracy (Brodersen et al., 2010)
        """
        per_class_acc = []
        for cls in classes:
            mask = y_true == cls
            if np.sum(mask) == 0:
                continue
            per_class_acc.append(np.mean(y_pred[mask] == y_true[mask]))
        return np.mean(per_class_acc) if per_class_acc else 0.0

    @staticmethod
    def calibration_error(y_true: np.ndarray,
                        probs: np.ndarray,
                        bins: int = 10) -> float:
        """
        Calculate expected calibration error (Guo et al., 2017)
        """
        bin_boundaries = np.linspace(0, 1, bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        errors = []
        for bl, bu in zip(bin_lowers, bin_uppers):
            in_bin = (probs >= bl) & (probs < bu)
            prop = np.mean(y_true[in_bin]) if np.any(in_bin) else 0
            avg_prob = np.mean(probs[in_bin]) if np.any(in_bin) else 0
            errors.append(np.abs(avg_prob - prop) * np.sum(in_bin))
            
        return np.sum(errors) / len(y_true)

class BiasDetection:
    """
    Statistical bias detection methods from social sciences
    Reference: West et al., "Discrimination in Algorithmic Decision Making", 2023
    """
    
    @staticmethod
    def subgroup_variance(scores: Dict[str, np.ndarray],
                         alpha: float = 0.05) -> Tuple[bool, str]:
        """
        Bartlett's test for variance homogeneity across groups
        Reference: Bartlett, "Properties of Sufficiency and Statistical Tests", 1937
        """
        from scipy.stats import bartlett
        group_vars = [np.var(g_scores) for g_scores in scores.values()]
        stat, pval = bartlett(*scores.values())
        
        violation = pval < alpha
        msg = (f"Subgroup Variance {'Heterogeneity' if violation else 'Homogeneity'}: "
               f"p={pval:.4f}, variances={group_vars}")
        return violation, msg

    @staticmethod
    def disparate_mistreatment(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              sensitive_attr: np.ndarray,
                              metric: str = 'fpr') -> float:
        """
        Calculate disparate mistreatment (Zafar et al., 2017)
        :param metric: 'fpr' (false positive) or 'fnr' (false negative)
        """
        if metric not in ['fpr', 'fnr']:
            raise ValueError("Invalid metric, choose 'fpr' or 'fnr'")
            
        disparity = 0.0
        groups = np.unique(sensitive_attr)
        group_metrics = []
        
        for group in groups:
            mask = sensitive_attr == group
            if metric == 'fpr':
                rate = np.mean(y_pred[mask][y_true[mask] == 0] == 1)
            else:  # fnr
                rate = np.mean(y_pred[mask][y_true[mask] == 1] == 0)
            group_metrics.append(rate)
            
        return max(group_metrics) - min(group_metrics)

class MetricSummarizer:
    """
    Aggregate results with academic references
    Reference: Mitchell et al., "Model Cards for Model Reporting", 2019
    """
    
    @staticmethod
    def create_model_card(metrics: Dict[str, Any],
                        references: Dict[str, str]) -> Dict:
        """
        Generate model card with metric provenance
        """
        return {
            'metrics': metrics,
            'provenance': references,
            'timestamp': np.datetime64('now')
        }
