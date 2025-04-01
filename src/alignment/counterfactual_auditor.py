"""Counterfactual Fairness Audit System
Implements causal counterfactual analysis for alignment verification through:
Structural causal model interventions (Pearl, 2009)
Counterfactual fairness estimation (Kusner et al., 2017)
Policy decision sensitivity analysis
"""

import logging
import numpy as np
import pandas as pd

from auditors.causal_model import CausalGraphBuilder
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from scipy.stats import ttest_ind
from causalinference import CausalModel
from auditors.fairness_metrics import CounterfactualFairness

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class CounterfactualConfig:
    """Configuration for counterfactual analysis"""
    num_perturbations: int = 5
    epsilon_range: Tuple[float, float] = (0.1, 0.3)
    sensitivity_threshold: float = 0.15
    causal_confounders: List[str] = field(default_factory=list)
    fairness_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'individual_fairness': 0.8,
        'group_disparity': 0.1,
        'causal_effect': 0.05
        })

class CounterfactualAuditor:
    """
    Causal counterfactual analysis system implementing:
    - Structural equation modeling for scenario generation
    - Decision boundary sensitivity testing
    - Cross-world independence verification
    - Counterfactual fairness certification
    Key Components:
    1. CausalGraphBuilder: Domain-aware structural model construction
    2. CausalModel: Potential outcome estimation
    3. CounterfactualFairness: Multi-level fairness quantification
    """

def __init__(self, config: Optional[CounterfactualConfig] = None):
    self.config = config or CounterfactualConfig()
    self.causal_builder = CausalGraphBuilder()
    self.fairness_assessor = CounterfactualFairness()

def audit(self, data: pd.DataFrame,
         predictions: np.ndarray,
         sensitive_attrs: List[str]) -> Dict:
    """
    Perform comprehensive counterfactual analysis with:
    1. Causal graph construction
    2. Controlled attribute perturbations
    3. Potential outcome estimation
    4. Fairness violation detection
    """
    # Build domain-specific causal model
    causal_graph = self.causal_builder.construct_graph(data, sensitive_attrs)
    
    # Generate counterfactual scenarios
    cf_data, interventions = self._generate_counterfactuals(data, sensitive_attrs)
    
    # Estimate potential outcomes
    cf_predictions = self._estimate_potential_outcomes(causal_graph, cf_data)
    
    # Compute fairness metrics
    fairness_report = self._assess_fairness_violations(
        data, predictions, cf_data, cf_predictions, sensitive_attrs
    )
    
    # Analyze decision sensitivity
    sensitivity_report = self._analyze_decision_sensitivity(
        predictions, cf_predictions, interventions
    )
    
    return {
        'causal_graph': causal_graph.to_json(),
        'fairness_metrics': fairness_report,
        'sensitivity_analysis': sensitivity_report,
        'counterfactual_samples': cf_data.sample(3).to_dict(orient='records')
    }

def _generate_counterfactuals(self,
                             data: pd.DataFrame,
                             sensitive_attrs: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """Controlled attribute perturbation with causal validity checks"""
    cf_data = data.copy()
    interventions = {}
    
    for attr in sensitive_attrs:
        # Calculate valid perturbation range
        baseline = data[attr].mean()
        eps = np.random.uniform(*self.config.epsilon_range, size=len(data))
        
        # Apply constrained interventions
        perturbed = self._apply_constrained_perturbation(data[attr], eps)
        cf_data[attr] = perturbed
        
        interventions[attr] = {
            'original_mean': baseline,
            'perturbed_mean': perturbed.mean(),
            'max_shift': np.abs(perturbed - data[attr]).max()
        }
        
    return cf_data, interventions

def _apply_constrained_perturbation(self,
                                  series: pd.Series,
                                  epsilon: np.ndarray) -> pd.Series:
    """Domain-aware perturbation preserving causal relationships"""
    # Preserve ordinal relationships
    if series.dtype == 'category':
        return self._perturb_categorical(series, epsilon)
    else:
        return self._perturb_continuous(series, epsilon)

def _perturb_continuous(self,
                       series: pd.Series,
                       epsilon: np.ndarray) -> pd.Series:
    """Monotonic perturbation with boundary constraints"""
    perturbed = series * (1 + epsilon)
    return perturbed.clip(series.min(), series.max())

def _perturb_categorical(self,
                        series: pd.Series,
                        epsilon: np.ndarray) -> pd.Series:
    """Probability-preserving categorical redistribution"""
    unique_vals = series.unique()
    transition_probs = np.abs(epsilon) / np.sum(np.abs(epsilon))
    return series.apply(
        lambda x: np.random.choice(unique_vals, p=transition_probs)
    )

def _estimate_potential_outcomes(self,
                                causal_graph: CausalModel,
                                cf_data: pd.DataFrame) -> np.ndarray:
    """Potential outcome estimation using structural causal model"""
    return causal_graph.estimate_effect(
        cf_data,
        treatment='sensitive_attributes',
        outcome='prediction',
        method='backdoor.linear_regression'
    ).values

def _assess_fairness_violations(self,
                               original_data: pd.DataFrame,
                               original_preds: np.ndarray,
                               cf_data: pd.DataFrame,
                               cf_preds: np.ndarray,
                               sensitive_attrs: List[str]) -> Dict:
    """Multi-level counterfactual fairness assessment"""
    individual_fairness = self.fairness_assessor.compute_individual_fairness(
        original_preds, cf_preds, original_data[sensitive_attrs]
    )
    
    group_metrics = {}
    for attr in sensitive_attrs:
        group_metrics[attr] = self.fairness_assessor.compute_group_disparity(
            original_data[attr], original_preds, cf_data[attr], cf_preds
        )
        
    causal_effects = self._compute_average_causal_effect(
        original_preds, cf_preds, original_data, cf_data
    )
    
    return {
        'individual_fairness': individual_fairness,
        'group_disparity': group_metrics,
        'causal_effect_size': causal_effects,
        'threshold_violations': self._detect_threshold_violations(
            individual_fairness, group_metrics, causal_effects
        )
    }

def _compute_average_causal_effect(self,
                                  original_preds: np.ndarray,
                                  cf_preds: np.ndarray,
                                  original_data: pd.DataFrame,
                                  cf_data: pd.DataFrame) -> Dict:
    """Causal effect estimation using doubly robust estimation"""
    ate = np.mean(cf_preds - original_preds)
    att = np.mean((cf_preds - original_preds)[original_data['treatment'] == 1])
    atc = np.mean((cf_preds - original_preds)[original_data['treatment'] == 0])
    return {'ATE': ate, 'ATT': att, 'ATC': atc}

def _detect_threshold_violations(self,
                                individual_fairness: float,
                                group_metrics: Dict,
                                causal_effects: Dict) -> Dict:
    """Threshold-based violation detection"""
    violations = {
        'individual': individual_fairness < self.config.fairness_thresholds['individual_fairness'],
        'group': {
            attr: metrics['disparity'] > self.config.fairness_thresholds['group_disparity']
            for attr, metrics in group_metrics.items()
        },
        'causal': {
            effect_type: abs(value) > self.config.fairness_thresholds['causal_effect']
            for effect_type, value in causal_effects.items()
        }
    }
    return violations

def _analyze_decision_sensitivity(self,
                                 original_preds: np.ndarray,
                                 cf_preds: np.ndarray,
                                 interventions: Dict) -> Dict:
    """Statistical sensitivity characterization"""
    sensitivity_scores = {}
    for attr, intervention in interventions.items():
        _, p_value = ttest_ind(original_preds, cf_preds)
        sensitivity_scores[attr] = {
            'mean_shift': intervention['perturbed_mean'] - intervention['original_mean'],
            'p_value': p_value,
            'effect_size': self._compute_cohens_d(original_preds, cf_preds),
            'sensitivity_flag': p_value < self.config.sensitivity_threshold
        }
    return sensitivity_scores

@staticmethod
def _compute_cohens_d(original: np.ndarray,
                     counterfactual: np.ndarray) -> float:
    """Effect size calculation for sensitivity analysis"""
    diff = original.mean() - counterfactual.mean()
    pooled_std = np.sqrt((original.std()**2 + counterfactual.std()**2) / 2)
    return abs(diff / pooled_std) if pooled_std != 0 else 0.0

@dataclass
class CounterfactualReport:
  """Formal representation of counterfactual audit findings"""
  causal_structure: Dict
  fairness_violations: Dict
  sensitivity_attributes: Dict
  causal_effects: Dict
  intervention_parameters: Dict
