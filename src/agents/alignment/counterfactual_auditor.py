"""Counterfactual Fairness Audit System
Implements causal counterfactual analysis for alignment verification through:
Structural causal model interventions (Pearl, 2009)
Counterfactual fairness estimation (Kusner et al., 2017)
Policy decision sensitivity analysis
"""

import numpy as np
import pandas as pd
import networkx as nx
import hashlib

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from scipy.stats import ttest_ind, wasserstein_distance

from src.agents.alignment.auditors.causal_model import CausalGraphBuilder, CausalModel
from src.agents.alignment.auditors.fairness_metrics import CounterfactualFairness
from logs.logger import get_logger

logger = get_logger("Countefactual Auditor")

@dataclass
class CounterfactualConfig:
    """Configuration for counterfactual analysis"""
    perturbation_strategy: str = 'flip' # Options: 'flip', 'sample_distribution', 'fixed_delta'
    perturbation_magnitude: float = 0.1 # Used for 'fixed_delta' or scaling
    num_counterfactual_samples: int = 1 # Samples per instance for stochastic methods
    sensitivity_alpha: float = 0.05 # Significance level for sensitivity tests
    fairness_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'individual_fairness_max_diff': 0.1, # Max allowed prediction change
        'individual_fairness_mean_diff': 0.05, # Mean allowed prediction change
        'group_disparity_stat_parity': 0.1,
        'group_disparity_equal_opp': 0.1,
        'group_disparity_avg_odds': 0.1,
        'causal_effect_ate': 0.05 # Max allowed average treatment effect of sensitive attr
        })

class CounterfactualAuditor:
    """
    Causal counterfactual analysis system implementing:
    - Generation of counterfactual scenarios based on interventions.
    - Estimation of model predictions under these scenarios using a CausalModel.
    - Assessment of counterfactual fairness metrics.
    - Analysis of decision sensitivity to sensitive attribute changes.

    Requires pre-built CausalModel and a model prediction function.
    """

    def __init__(self,
                 causal_model: CausalModel,
                 model_predict_func: callable, # Function: predict(data) -> np.ndarray
                 config: Optional[CounterfactualConfig] = None):
        """
        Initialize the auditor.

        Args:
            causal_model (CausalModel): A pre-constructed and validated CausalModel instance.
            model_predict_func (callable): A function that takes a DataFrame compatible
                                           with the causal model's data and returns model predictions.
            config (Optional[CounterfactualConfig]): Configuration settings.
        """
        if not isinstance(causal_model, CausalModel):
             raise TypeError("causal_model must be an instance of CausalModel.")
        if not callable(model_predict_func):
             raise TypeError("model_predict_func must be callable.")

        self.causal_model = causal_model
        self.model_predict_func = model_predict_func
        self.config = config or CounterfactualConfig()
        self.fairness_assessor = CounterfactualFairness() # Assumes default init is ok

    def audit(self,
              data: pd.DataFrame,
              sensitive_attrs: List[str],
              y_true_col: Optional[str] = None # Needed for some group metrics
             ) -> Dict[str, Any]:
        """
        Perform comprehensive counterfactual fairness audit.

        Args:
            data (pd.DataFrame): The original dataset used for the causal model.
                                 Must contain sensitive attributes and features needed by the model.
            sensitive_attrs (List[str]): List of sensitive attribute column names.
            y_true_col (Optional[str]): Name of the ground truth label column, if available.

        Returns:
            Dict[str, Any]: A report containing fairness metrics, sensitivity analysis, etc.
        """
        if not all(attr in data.columns for attr in sensitive_attrs):
            missing = [attr for attr in sensitive_attrs if attr not in data.columns]
            raise ValueError(f"Sensitive attributes {missing} not found in data columns.")

        # 1. Get Original Predictions
        logger.info("Getting original model predictions...")
        original_preds = self._get_predictions(data)

        # 2. Generate and Evaluate Counterfactuals for each Sensitive Attribute
        cf_results = {}
        all_interventions = {}
        for attr in sensitive_attrs:
            logger.info(f"Generating counterfactuals by intervening on '{attr}'...")
            interventions = self._generate_interventions(data, attr)
            all_interventions[attr] = interventions # Store for reporting

            logger.info(f"Estimating potential outcomes for '{attr}' interventions...")
            # Estimate outcomes under these specific interventions using the CausalModel
            # Store results keyed by intervention for clarity
            cf_predictions_map = {}
            for cf_value, intervention_dict in interventions.items():
                 # Use CausalModel to compute the state of the system under intervention
                 # We assume compute_counterfactual returns the full DataFrame under intervention
                 cf_intervened_data = self.causal_model.compute_counterfactual(intervention=intervention_dict)
                 # Then use the ML model's predict function on this counterfactual data
                 cf_predictions_map[cf_value] = self._get_predictions(cf_intervened_data)

            cf_results[attr] = cf_predictions_map

        # 3. Assess Fairness Violations
        logger.info("Assessing fairness violations...")
        fairness_report = self._assess_fairness_violations(
            original_data=data,
            original_preds=original_preds,
            cf_results=cf_results, # Pass the dict of counterfactual predictions
            sensitive_attrs=sensitive_attrs,
            y_true_col=y_true_col
        )

        # 4. Analyze Decision Sensitivity
        logger.info("Analyzing decision sensitivity...")
        sensitivity_report = self._analyze_decision_sensitivity(
            original_preds=original_preds,
            cf_results=cf_results # Pass the dict of counterfactual predictions
        )

        # 5. Generate Final Report
        logger.info("Generating audit report...")
        final_report = {
            'audit_config': self.config.__dict__,
            'causal_graph_info': {
                'nodes': list(self.causal_model.graph.nodes()),
                'edges': list(self.causal_model.graph.edges())
                # Avoid returning full graph object unless needed & serializable
            },
            'fairness_metrics': fairness_report,
            'sensitivity_analysis': sensitivity_report,
            'interventions_applied': {
                attr: {cf_val: interv for cf_val, interv in interv_map.items()}
                for attr, interv_map in all_interventions.items()
            }
            # Optionally add samples of counterfactual data if generated and useful
        }
        return final_report


    def _get_predictions(self, data: pd.DataFrame) -> np.ndarray:
        """Helper to get predictions, handling potential errors."""
        try:
             # Ensure data passed to predict function has columns expected by the model
             # This might involve selecting specific columns based on model training features
             # Assuming predict_func handles necessary column selection/preprocessing
             preds = self.model_predict_func(data)
             if not isinstance(preds, np.ndarray):
                 preds = np.array(preds)
             return preds
        except Exception as e:
             logger.error(f"Model prediction function failed: {e}", exc_info=True)
             raise


    def _generate_interventions(self, data: pd.DataFrame, sensitive_attr: str) -> Dict[Any, Dict[str, Any]]:
        """
        Generates intervention dictionaries for a single sensitive attribute.
        An intervention dictionary is {variable: value}.

        Returns:
             Dict[Any, Dict[str, Any]]: Map from counterfactual value to intervention dict.
                                       e.g., {0: {'gender': 0}, 1: {'gender': 1}}
        """
        interventions = {}
        unique_values = data[sensitive_attr].unique()

        if self.config.perturbation_strategy == 'flip':
             # Assumes binary or allows flipping between existing unique values
             if len(unique_values) == 2:
                  val0, val1 = unique_values[0], unique_values[1]
                  interventions[val0] = {sensitive_attr: val0}
                  interventions[val1] = {sensitive_attr: val1}
             elif len(unique_values) == 1:
                  logger.warning(f"Attribute '{sensitive_attr}' has only one value. Cannot generate 'flip' intervention.")
                  val = unique_values[0]
                  interventions[val] = {sensitive_attr: val} # Trivial intervention
             else:
                  logger.warning(f"Attribute '{sensitive_attr}' has >2 unique values. 'flip' strategy may not be well-defined. Using all unique values.")
                  for val in unique_values:
                       interventions[val] = {sensitive_attr: val}

        # TODO: Implement other strategies like 'sample_distribution', 'fixed_delta' if needed.
        # Example for fixed_delta (continuous only):
        elif self.config.perturbation_strategy == 'fixed_delta':
            if pd.api.types.is_numeric_dtype(data[sensitive_attr]):
                mean_val = data[sensitive_attr].mean()
                delta = self.config.perturbation_magnitude * data[sensitive_attr].std()
                interventions[mean_val + delta] = {sensitive_attr: mean_val + delta}
                interventions[mean_val - delta] = {sensitive_attr: mean_val - delta}
            else:
                logger.warning("fixed_delta requires numeric attribute. Skipping intervention.")

        else:
            # Default: Use existing unique values as intervention points
             logger.warning(f"Unknown perturbation strategy '{self.config.perturbation_strategy}'. Using unique values.")
             for val in unique_values:
                  interventions[val] = {sensitive_attr: val}

        if not interventions:
             raise ValueError(f"Could not generate interventions for attribute '{sensitive_attr}' with strategy '{self.config.perturbation_strategy}'.")

        return interventions

    def _assess_fairness_violations(self,
                                    original_data: pd.DataFrame,
                                    original_preds: np.ndarray,
                                    cf_results: Dict[str, Dict[Any, np.ndarray]],
                                    sensitive_attrs: List[str],
                                    y_true_col: Optional[str]) -> Dict[str, Any]:
        """
        Assess fairness using CounterfactualFairness, comparing original predictions
        with predictions under interventions on each sensitive attribute.
        """
        fairness_report = {
            'individual_fairness': {},
            'group_disparity': {},
            'path_specific_effects': {}, # Placeholder, requires specific implementation path
            'overall_violations': {}
        }

        all_individual_metrics = []
        all_group_metrics = {}
        all_causal_effects = {} # Placeholder for path-specific effects

        # Iterate through each sensitive attribute that was intervened on
        for attr in sensitive_attrs: # Iterate over the list of sensitive attributes provided
            if attr not in cf_results:
                logger.warning(f"No counterfactual results found for attribute '{attr}'. Skipping fairness assessment for it.")
                continue

            cf_preds_map = cf_results[attr]
            violations_attr = {} # Store violations specific to this attribute

            # --- Individual Fairness ---
            # This part depends on the structure of cf_preds_map, often requires comparing specific pairs
            if self.config.perturbation_strategy == 'flip' and len(cf_preds_map) == 2:
                cf_val0, cf_val1 = list(cf_preds_map.keys())
                preds_a0 = cf_preds_map[cf_val0]
                preds_a1 = cf_preds_map[cf_val1]

                # Identify original predictions corresponding to A=0 and A=1
                orig_a0_mask = (original_data[attr] == cf_val0)
                orig_a1_mask = (original_data[attr] == cf_val1)

                # Ensure indices align if lengths differ due to NaNs etc. in original data
                common_idx_0 = original_data.index[orig_a0_mask]
                common_idx_1 = original_data.index[orig_a1_mask]

                # Check if lengths match for comparison - use indices from original data
                # The counterfactual predictions should have the same index as original_data
                if len(common_idx_0) > 0:
                     diffs_0_to_1 = original_preds[orig_a0_mask] - preds_a1[orig_a0_mask]
                else:
                     diffs_0_to_1 = np.array([])

                if len(common_idx_1) > 0:
                     diffs_1_to_0 = original_preds[orig_a1_mask] - preds_a0[orig_a1_mask]
                else:
                     diffs_1_to_0 = np.array([])

                all_diffs = np.concatenate([diffs_0_to_1, diffs_1_to_0])

                if len(all_diffs) > 0:
                    # Calculate Wasserstein distance between original and *relevant* counterfactuals
                    # Combine the counterfactuals that correspond to the flipped individuals
                    cf_preds_for_wdist = np.concatenate([
                        preds_a1[orig_a0_mask],
                        preds_a0[orig_a1_mask]
                    ]) if len(all_diffs) == len(original_preds) else np.array([]) # Ensure size match if needed

                    indiv_metrics = {
                        'max_difference': float(np.max(np.abs(all_diffs))),
                        'mean_difference': float(np.mean(np.abs(all_diffs))),
                        'wasserstein_distance': wasserstein_distance(original_preds, cf_preds_for_wdist) if len(cf_preds_for_wdist) == len(original_preds) else np.nan
                    }
                else:
                    indiv_metrics = {'max_difference': 0.0, 'mean_difference': 0.0, 'wasserstein_distance': 0.0}

                fairness_report['individual_fairness'][attr] = indiv_metrics
                all_individual_metrics.append(indiv_metrics)
                violations_attr['individual_mean_diff'] = indiv_metrics['mean_difference'] > self.config.fairness_thresholds['individual_fairness_mean_diff']
                violations_attr['individual_max_diff'] = indiv_metrics['max_difference'] > self.config.fairness_thresholds['individual_fairness_max_diff']

            else:
                 logger.warning(f"Individual fairness assessment for attribute '{attr}' skipped or limited. Requires 'flip' strategy with 2 outcomes currently.")
                 fairness_report['individual_fairness'][attr] = {'message': "Skipped or limited due to strategy/values."}


            # --- Group Disparity ---
            if y_true_col and self.config.perturbation_strategy == 'flip' and len(cf_preds_map) == 2:
                cf_val0, cf_val1 = list(cf_preds_map.keys()) # These are now defined in scope

                # Assuming cf_val1 is privileged, cf_val0 is unprivileged - needs careful check in practice
                priv_group = cf_val1
                unpriv_group = cf_val0

                # Original group disparity
                orig_group_disp = self.fairness_assessor.compute_group_disparity(
                    data=original_data.assign(_preds=original_preds),
                    sensitive_attr=attr,
                    predictions='_preds',
                    y_true=y_true_col,
                    privileged_group=priv_group,
                    unprivileged_group=unpriv_group
                )
                all_group_metrics[attr] = orig_group_disp # Collect original metrics

                # Counterfactual group disparities (Example: on data under do(A=1))
                # Create counterfactual dataset with predictions under do(A=privileged)
                cf_data_priv = self.causal_model.compute_counterfactual(
                    intervention={attr: priv_group}
                ).assign(_preds=cf_preds_map[priv_group])

                cf_group_disp_priv = self.fairness_assessor.compute_group_disparity(
                    data=cf_data_priv,
                    sensitive_attr=attr, # Note: sensitive attr here is now fixed to priv_group
                    predictions='_preds',
                    y_true=y_true_col,
                    privileged_group=priv_group,
                    unprivileged_group=unpriv_group
                )
                # Repeat for do(A=unprivileged) if desired

                # Calculate max absolute disparities observed (original vs counterfactual)
                # We compare the *original* disparity against the threshold for now
                stat_parity_diff = orig_group_disp.get('statistical_parity_difference', np.nan)
                eod_diff = orig_group_disp.get('equal_opportunity_difference', np.nan)
                aaod_diff = orig_group_disp.get('average_abs_odds_difference', np.nan)

                fairness_report['group_disparity'][attr] = {
                    'original': orig_group_disp,
                    'counterfactual_under_do_privileged': cf_group_disp_priv, # Example CF metric
                    'max_stat_parity_diff_observed': abs(stat_parity_diff),
                    'max_eod_diff_observed': abs(eod_diff),
                    'max_aaod_diff_observed': abs(aaod_diff)
                }

                # Check original disparities against thresholds
                violations_attr['group_stat_parity'] = abs(stat_parity_diff) > self.config.fairness_thresholds['group_disparity_stat_parity']
                violations_attr['group_equal_opp'] = abs(eod_diff) > self.config.fairness_thresholds['group_disparity_equal_opp']
                violations_attr['group_avg_odds'] = abs(aaod_diff) > self.config.fairness_thresholds['group_disparity_avg_odds']

            elif y_true_col:
                logger.warning(f"Group disparity assessment for attribute '{attr}' skipped or limited. Requires 'flip' strategy with 2 outcomes currently.")
                fairness_report['group_disparity'][attr] = {'message': 'Skipped or limited due to strategy/values.'}
            else:
                logger.warning(f"Ground truth column '{y_true_col}' not provided. Skipping group fairness metrics requiring labels for attr '{attr}'.")
                fairness_report['group_disparity'][attr] = {'message': f"Skipped for {attr} due to missing y_true"}


            # --- Path-Specific Effects ---
            # (Requires flip strategy & 2 values currently for NDE/NIE)
            if self.config.perturbation_strategy == 'flip' and len(cf_preds_map) == 2:
                 cf_val0, cf_val1 = list(cf_preds_map.keys())
                 try:
                     # Define a suitable outcome variable for path analysis.
                     # Using the model prediction *function* as the outcome mechanism within the causal graph context.
                     # This is complex. A simplification: use the *name* of the prediction column if added to data.
                     # Let's assume we need to compute effects on the *original* prediction column if it exists,
                     # or potentially requires modifying the CausalModel to handle the prediction function directly.
                     # Placeholder: Using a dummy outcome name '_preds_for_pse'. Need to align CausalModel SEMs.
                     outcome_for_pse = '_preds' # Assumes predictions added to data used by CausalModel

                     # Add predictions column temporarily if needed by CausalModel's SEM estimation
                     temp_data_for_pse = original_data.assign(_preds=original_preds)

                     if hasattr(self.fairness_assessor, 'path_specific_effects') and outcome_for_pse in temp_data_for_pse.columns:
                          # Ensure CausalModel has SEM for '_preds' (might need re-estimation or specific handling)
                          if outcome_for_pse not in self.causal_model._get_structural_equations():
                              logger.warning(f"Outcome '{outcome_for_pse}' not found in CausalModel SEMs. Skipping Path Specific Effects for {attr}.")
                              fairness_report['path_specific_effects'][attr] = {'message': f"Skipped, '{outcome_for_pse}' SEM missing."}
                          else:
                              pse_results = self.fairness_assessor.path_specific_effects(
                                   causal_model=self.causal_model, # Assumes SEMs include the prediction step
                                   data=temp_data_for_pse,
                                   sensitive_attr=attr,
                                   outcome=outcome_for_pse, # Use prediction as outcome
                                   sensitive_val_1=cf_val1,
                                   sensitive_val_0=cf_val0
                              )
                              fairness_report['path_specific_effects'][attr] = pse_results
                              all_causal_effects[attr] = pse_results
                              violations_attr['causal_nde'] = abs(pse_results.get('NDE', 0)) > self.config.fairness_thresholds['causal_effect_ate']
                              violations_attr['causal_nie'] = abs(pse_results.get('NIE', 0)) > self.config.fairness_thresholds['causal_effect_ate']
                     else:
                          fairness_report['path_specific_effects'][attr] = {'message': 'Method or outcome column not available'}
                 except Exception as e:
                     logger.error(f"Path specific effects calculation failed for {attr}: {e}", exc_info=True)
                     fairness_report['path_specific_effects'][attr] = {'error': str(e)}
            else:
                logger.warning(f"Path Specific Effects calculation for attribute '{attr}' skipped. Requires 'flip' strategy with 2 outcomes currently.")
                fairness_report['path_specific_effects'][attr] = {'message': "Skipped due to strategy/values."}


            # Consolidate violations for this attribute
            fairness_report['overall_violations'][attr] = violations_attr


        # Aggregate overall violations (simple approach: any violation for any attribute)
        overall_violations_summary = {}
        violation_keys = set(key for viol_dict in fairness_report['overall_violations'].values() for key in viol_dict)
        for key in violation_keys:
            overall_violations_summary[key] = any(
                viol_dict.get(key, False) for viol_dict in fairness_report['overall_violations'].values()
            )
        fairness_report['overall_violations']['summary'] = overall_violations_summary


        return fairness_report


    def _analyze_decision_sensitivity(self,
                                    original_preds: np.ndarray,
                                    cf_results: Dict[str, Dict[Any, np.ndarray]]
                                    ) -> Dict[str, Any]:
        """
        Analyze how much predictions change on average when sensitive attributes are intervened upon.
        Uses statistical tests (t-test) and effect sizes (Cohen's d).
        """
        sensitivity_scores = {}

        for attr, cf_preds_map in cf_results.items():
            # Compare original distribution with distribution under each counterfactual intervention
            attr_sensitivities = {}
            for cf_value, cf_preds_array in cf_preds_map.items():
                if len(original_preds) != len(cf_preds_array):
                     logger.warning(f"Original ({len(original_preds)}) and counterfactual ({len(cf_preds_array)}) prediction arrays differ in length for {attr}={cf_value}. Skipping t-test.")
                     stat, p_value = np.nan, np.nan
                elif np.var(original_preds) < 1e-9 or np.var(cf_preds_array) < 1e-9:
                     logger.warning(f"Zero variance in predictions for {attr}={cf_value}. Skipping t-test.")
                     stat, p_value = np.nan, np.nan
                else:
                    stat, p_value = ttest_ind(original_preds, cf_preds_array, equal_var=False, nan_policy='omit') # Welch's t-test

                effect_size = self._compute_cohens_d(original_preds, cf_preds_array)
                mean_shift = np.nanmean(cf_preds_array) - np.nanmean(original_preds)

                attr_sensitivities[str(cf_value)] = {
                    'mean_shift': float(mean_shift),
                    'cohens_d': float(effect_size),
                    't_statistic': float(stat),
                    'p_value': float(p_value),
                    'is_sensitive': p_value < self.config.sensitivity_alpha if not np.isnan(p_value) else None
                }
            sensitivity_scores[attr] = attr_sensitivities

        return sensitivity_scores

    @staticmethod
    def _compute_cohens_d(original: np.ndarray,
                          counterfactual: np.ndarray) -> float:
        """Effect size calculation (Cohen's d) for sensitivity analysis."""
        # Filter NaNs before calculating stats
        original_valid = original[~np.isnan(original)]
        counterfactual_valid = counterfactual[~np.isnan(counterfactual)]

        if len(original_valid) < 2 or len(counterfactual_valid) < 2:
             return 0.0 # Cannot compute with insufficient data

        diff = np.mean(original_valid) - np.mean(counterfactual_valid)
        n1, n2 = len(original_valid), len(counterfactual_valid)
        var1, var2 = np.var(original_valid, ddof=1), np.var(counterfactual_valid, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        return abs(diff / pooled_std) if pooled_std > 1e-9 else 0.0


# Note: The CounterfactualReport dataclass seems redundant if the audit method returns a dict.
# Removed unless explicitly needed for structuring output further.

# Example Usage Placeholder (requires setting up CausalModel, predict_func, data)
if __name__ == '__main__':
    # Synthetic data generation
    np.random.seed(42)
    n_samples = 2000
    data = pd.DataFrame({
        'age': np.random.normal(35, 10, n_samples).clip(18, 80),
        'gender': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),  # Sensitive attribute
        'education': np.random.choice([0, 1, 2], n_samples),
        'income': np.random.lognormal(3, 0.3, n_samples)
    })
    # Synthetic outcome (loan approval) with gender bias
    data['loan_approval'] = np.where(
        (data['income'] > 45) & (data['education'] > 0),
        1,
        np.where(data['gender'] == 1, 1, 0)  # Bias: 85% approval for gender=1 when borderline
    ).astype(int)

    # Simulate ML model predictions with some bias
    def ml_predict_func(df):
        # Simple logistic regression-like predictions
        return 1 / (1 + np.exp(-(
            0.8 * (df['income']/50) + 
            0.5 * df['education'] -
            0.2 * (df['age']/100) -
            0.3 * df['gender']  # Biased term
        )))

    # Build causal graph
    builder = CausalGraphBuilder()
    builder.config.required_edges = [('gender', 'loan_approval')]  # Force known relationship
    causal_model = builder.construct_graph(data, sensitive_attrs=['gender'])

    # Configure and run auditor
    config = CounterfactualConfig(
        perturbation_strategy='flip',
        fairness_thresholds={
            'individual_fairness_max_diff': 0.15,
            'individual_fairness_mean_diff': 0.07,
            'group_disparity_stat_parity': 0.1,
            'group_disparity_equal_opp': 0.15,
            'group_disparity_avg_odds': 0.1,
            'causal_effect_ate': 0.05
        }
    )
    
    auditor = CounterfactualAuditor(
        causal_model=causal_model,
        model_predict_func=ml_predict_func,
        config=config
    )
    
    report = auditor.audit(
        data=data,
        sensitive_attrs=['gender'],
        y_true_col='loan_approval'  # Using synthetic ground truth
    )

    # Print key results
    print("\n=== Audit Summary ===")
    print(f"Individual Fairness Violations: {report['fairness_metrics']['overall_violations']}")
    print(f"Group Disparity Violations: {report['fairness_metrics']['group_disparity']}")
    print(f"Sensitivity Findings: {report['sensitivity_analysis']}")
