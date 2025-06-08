"""Counterfactual Fairness Audit System
Implements causal counterfactual analysis for alignment verification through:
Structural causal model interventions (Pearl, 2009)
Counterfactual fairness estimation (Kusner et al., 2017)
Policy decision sensitivity analysis
"""

import numpy as np
import pandas as pd
import networkx as nx

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from scipy.stats import ttest_ind, wasserstein_distance

from src.agents.alignment.utils.config_loader import load_global_config, get_config_section
from src.agents.alignment.auditors.causal_model import CausalGraphBuilder, CausalModel
from src.agents.alignment.auditors.fairness_metrics import CounterfactualFairness
from src.agents.alignment.alignment_memory import AlignmentMemory

from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Countefactual Auditor")
printer = PrettyPrinter

class CounterfactualAuditor:
    """
    Causal counterfactual analysis system implementing:
    - Generation of counterfactual scenarios based on interventions.
    - Estimation of model predictions under these scenarios using a CausalModel.
    - Assessment of counterfactual fairness metrics.
    - Analysis of decision sensitivity to sensitive attribute changes.

    Requires pre-built CausalModel and a model prediction function.
    """

    def __init__(self):
        """
        Initialize the auditor.

        Args:
            causal_model (CausalModel): A pre-constructed and validated CausalModel instance.
            model_predict_func (callable): A function that takes a DataFrame compatible
                                           with the causal model's data and returns model predictions.
            config (Optional[CounterfactualConfig]): Configuration settings.
        """
        self.config = load_global_config()
        self.sensitive_attributes = self.config.get('sensitive_attributes')

        self.auditor_config = get_config_section('counterfactual_auditor')
        self.perturbation_strategy = self.auditor_config.get('perturbation_strategy')
        self.perturbation_magnitude = self.auditor_config.get('perturbation_magnitude')
        self.num_counterfactual_samples = self.auditor_config.get('num_counterfactual_samples')
        self.sensitivity_alpha = self.auditor_config.get('sensitivity_alpha')
        self.required_edges = self.auditor_config.get('required_edges')

        self.sensitive_attrs = self.sensitive_attributes
        graph = nx.DiGraph()
        data = pd.DataFrame()

        # model_predict_func=callable()
        self.model_predict_func = None
        self.alignment_memory = AlignmentMemory()
        self.causal_model = CausalModel(graph=graph, data=data)
        self.fairness_assessor = CounterfactualFairness()

        logger.info(f"Counterfactual Auditor has succesfully initialized")

    def set_model_predict_func(self, predict_func):
        """Set the model prediction function dynamically"""
        self.model_predict_func = predict_func

    def audit(self,
              data: pd.DataFrame,
              sensitive_attrs: List[str],
              y_true_col: Optional[str] = None) -> Dict[str, Any]:
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
        if sensitive_attrs is None:
            sensitive_attrs = self.sensitive_attrs  # Use auditor's own config
        
        # Add fallback if still None
        if sensitive_attrs is None:
            logger.warning("No sensitive attributes provided - using default configuration")
            sensitive_attrs = self.auditor_config.get('default_sensitive_attributes', [])
        
        if not sensitive_attrs:  # Check if list is empty
            raise ValueError("No sensitive attributes defined for audit")
        
        # Original validation
        if not all(attr in data.columns for attr in sensitive_attrs):
            missing = [attr for attr in sensitive_attrs if attr not in data.columns]
            raise ValueError(f"Sensitive attributes {missing} not found in data columns.")
        
        # Generate unique audit ID
        audit_id = f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        domain = data.attrs.get('domain', 'unknown')  # Get from DataFrame attributes
        task_type = self.config.get('task_type', 'classification')  # Get from global config

        # 1. Get Original Predictions & Log audit start
        self.alignment_memory.log_evaluation(
            metric="audit_started",
            value=1.0,
            threshold=0.5,
            context={
                "audit_id": audit_id,
                "samples": len(data),
                "sensitive_attrs": sensitive_attrs,
                "domain": domain,
                "task_type": task_type
            }
        )
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
                 cf_intervened_data = self.causal_model.compute_counterfactual(intervention=intervention_dict)
                 # Then use the ML model's predict function on this counterfactual data
                 cf_predictions_map[cf_value] = self._get_predictions(cf_intervened_data)

            cf_results[attr] = cf_predictions_map

            # Log attribute-level metrics
            self.alignment_memory.log_evaluation(
                metric=f"{attr}_fairness_analysis",
                value=len(interventions),
                threshold=0,
                context={
                    "audit_id": audit_id,
                    "attribute": attr,
                    "strategy": self.perturbation_strategy
                }
            )

        # 3. Assess & Log Fairness Violations
        logger.info("Assessing fairness violations...")
        fairness_report = self._assess_fairness_violations(
            original_data=data,
            original_preds=original_preds,
            cf_results=cf_results,
            sensitive_attrs=sensitive_attrs,
            y_true_col=y_true_col
        )

        for attr, metrics in fairness_report['individual_fairness'].items():
            if not isinstance(metrics, dict):
                continue

            self.alignment_memory.log_evaluation(
                metric=f"{attr}_mean_diff",
                value=metrics.get('mean_difference', 0),
                threshold=self.auditor_config['fairness_thresholds']['individual_fairness_mean_diff'],
                context={"audit_id": audit_id}
            )

        # 4. Analyze Decision Sensitivity with Log concept drift status
        logger.info("Analyzing decision sensitivity...")
        drift_detected = self.alignment_memory.detect_drift()
        self.alignment_memory.log_evaluation(
            metric="concept_drift",
            value=float(drift_detected),
            threshold=0.5,
            context={"audit_id": audit_id}
        )

        sensitivity_report = self._analyze_decision_sensitivity(
            original_preds=original_preds,
            cf_results=cf_results
        )

        # 5. Generate Final Report
        logger.info("Generating audit report...")
        self.alignment_memory.record_outcome(
            context={"audit_id": audit_id},
            outcome={
                "bias_rate": self._calculate_overall_bias(fairness_report),
                "ethics_violations": self._count_violations(fairness_report)
            }
        )

        final_report = {
            'audit_config': self.config.__dir__,
            'causal_graph_info': {
                'nodes': list(self.causal_model.graph.nodes()),
                'edges': list(self.causal_model.graph.edges())
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

    def _calculate_overall_bias(self, report: Dict) -> float:
        """Calculate aggregate bias score from fairness report"""
        total_diff = 0
        count = 0
        for attr, metrics in report['individual_fairness'].items():
            if isinstance(metrics, dict):
                total_diff += metrics.get('mean_difference', 0)
                count += 1
        return total_diff / count if count else 0
    
    def _count_violations(self, report: Dict) -> int:
        """Count total fairness violations"""
        return sum(
            1 for v in report['overall_violations']['summary'].values() 
            if v is True
        )

    def _get_predictions(self, data: pd.DataFrame) -> np.ndarray:
        """Helper to get predictions, handling potential errors."""
        printer.status("Init", "Auditor predictor initialized", "info")

        if self.model_predict_func is None:
            raise ValueError("model_predict_func is not set. Please assign it before calling audit().")
        try:
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

        if pd.api.types.is_numeric_dtype(data[sensitive_attr]):
            # Handle continuous values
            if self.perturbation_strategy == 'fixed_delta':
                interventions = self.fixed_delta(data, sensitive_attr)
            elif self.perturbation_strategy == 'sample_distribution':
                interventions = self.sample_distribution(data, sensitive_attr)

        if self.perturbation_strategy == 'flip':
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

        elif self.perturbation_strategy == 'fixed_delta':
            if pd.api.types.is_numeric_dtype(data[sensitive_attr]):
                mean_val = data[sensitive_attr].mean()
                delta = self.perturbation_magnitude * data[sensitive_attr].std()
                interventions[mean_val + delta] = {sensitive_attr: mean_val + delta}
                interventions[mean_val - delta] = {sensitive_attr: mean_val - delta}
            else:
                logger.warning("fixed_delta requires numeric attribute. Skipping intervention.")

        else:
            # Default: Use existing unique values as intervention points
             logger.warning(f"Unknown perturbation strategy '{self.perturbation_strategy}'. Using unique values.")
             for val in unique_values:
                  interventions[val] = {sensitive_attr: val}

        if not interventions:
             raise ValueError(f"Could not generate interventions for attribute '{sensitive_attr}' with strategy '{self.perturbation_strategy}'.")

        return interventions
    
    def sample_distribution(self, data: pd.DataFrame, sensitive_attr: str) -> Dict[str, Dict[str, Any]]:
        """
        Creates interventions by sampling from the empirical distribution of the sensitive attribute.
    
        Args:
            data (pd.DataFrame): The original dataset.
            sensitive_attr (str): Sensitive attribute to sample from.
    
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of sampled interventions.
        """
        interventions = {}
        values = data[sensitive_attr].dropna().values
        if len(values) == 0:
            logger.warning(f"No values available to sample for {sensitive_attr}")
            return interventions
    
        n_samples = min(self.num_counterfactual_samples, len(values))
        sampled_values = np.random.choice(values, size=n_samples, replace=False)
    
        for i, val in enumerate(sampled_values):
            interventions[f'sample_{i}'] = {sensitive_attr: val}

        return interventions

    def fixed_delta(self, data: pd.DataFrame, sensitive_attr: str) -> Dict[str, Dict[str, Any]]:
        """
        Applies a fixed delta perturbation strategy to numeric sensitive attribute.
    
        Args:
            data (pd.DataFrame): Input dataset.
            sensitive_attr (str): The attribute to perturb.
    
        Returns:
            Dict[str, Dict[str, Any]]: Interventions with increased/decreased values.
        """
        interventions = {}
        if pd.api.types.is_numeric_dtype(data[sensitive_attr]):
            std_dev = data[sensitive_attr].std()
            if std_dev > 0:
                mean_val = data[sensitive_attr].mean()
                delta = self.perturbation_magnitude * std_dev
                interventions['increase'] = {sensitive_attr: mean_val + delta}
                interventions['decrease'] = {sensitive_attr: mean_val - delta}
            else:
                logger.warning(f"No variation in {sensitive_attr}; using mean only.")
                interventions['mean'] = {sensitive_attr: data[sensitive_attr].mean()}
        else:
            logger.warning(f"Fixed delta not applicable for non-numeric attribute: {sensitive_attr}")
        return interventions

    def _assess_fairness_violations(self,
                                    original_data: pd.DataFrame,
                                    original_preds: np.ndarray,
                                    cf_results: Dict[str, Dict[Any, np.ndarray]],
                                    sensitive_attrs: List[str],
                                    y_true_col: Optional[str]) -> Dict[str, Any]:
        """Robust fairness assessment with proper scoping and error handling."""
        thresholds = self.auditor_config['fairness_thresholds']

        fairness_report = {
            'individual_fairness': {},
            'group_disparity': {},
            'path_specific_effects': {},
            'overall_violations': {}
        }

        # Process each sensitive attribute independently
        for attr in sensitive_attrs:
            if attr not in cf_results:
                logger.warning(f"Skipping fairness assessment for '{attr}' - no counterfactual results")
                continue

            cf_preds_map = cf_results[attr]
            violations_attr = {}

            # Individual fairness assessment
            individual_metrics = self._assess_individual_fairness(
                original_data, original_preds, cf_preds_map, attr, thresholds
            )
            if individual_metrics:
                fairness_report['individual_fairness'][attr] = individual_metrics
                violations_attr.update({
                    'individual_mean_diff': individual_metrics.get('mean_difference', 0) > thresholds['individual_fairness_mean_diff'],
                    'individual_max_diff': individual_metrics.get('max_difference', 0) > thresholds['individual_fairness_max_diff']
                })
            else:
                fairness_report['individual_fairness'][attr] = {'message': "Assessment skipped"}

            # Group disparity assessment
            if y_true_col:
                group_metrics = self._assess_group_disparity(
                    original_data, original_preds, cf_preds_map, attr, 
                    y_true_col, thresholds
                )
                if group_metrics:
                    fairness_report['group_disparity'][attr] = group_metrics
                    violations_attr.update({
                        'group_stat_parity': group_metrics.get('stat_parity_violation', False),
                        'group_equal_opp': group_metrics.get('equal_opp_violation', False),
                        'group_avg_odds': group_metrics.get('avg_odds_violation', False)
                    })
                else:
                    fairness_report['group_disparity'][attr] = {'message': "Assessment skipped"}
            else:
                logger.warning(f"Skipping group disparity for '{attr}' - no ground truth provided")
                fairness_report['group_disparity'][attr] = {'message': "Missing y_true"}

            # Store violations for this attribute
            fairness_report['overall_violations'][attr] = violations_attr

        # Aggregate overall violations
        self._aggregate_violations(fairness_report)
        return fairness_report
    
    def _assess_individual_fairness(self, original_data, original_preds, 
                                   cf_preds_map, attr, thresholds):
        """Compute individual fairness metrics with proper error handling."""
        if self.perturbation_strategy != 'flip' or len(cf_preds_map) != 2:
            logger.warning(f"Individual fairness skipped for '{attr}' - requires flip strategy with 2 values")
            return None
    
        try:
            cf_val0, cf_val1 = list(cf_preds_map.keys())
            preds_a0 = cf_preds_map[cf_val0]
            preds_a1 = cf_preds_map[cf_val1]
            
            # Get masks for original groups
            orig_a0_mask = (original_data[attr] == cf_val0)
            orig_a1_mask = (original_data[attr] == cf_val1)
            
            # Calculate differences
            diffs = []
            if np.any(orig_a0_mask):
                diffs.extend(original_preds[orig_a0_mask] - preds_a1[orig_a0_mask])
            if np.any(orig_a1_mask):
                diffs.extend(original_preds[orig_a1_mask] - preds_a0[orig_a1_mask])
            
            if not diffs:
                return {'max_difference': 0.0, 'mean_difference': 0.0, 'wasserstein_distance': 0.0}
            
            diffs = np.array(diffs)
            abs_diffs = np.abs(diffs)
            
            # Calculate Wasserstein distance
            cf_combined = np.concatenate([
                preds_a1[orig_a0_mask] if np.any(orig_a0_mask) else np.array([]),
                preds_a0[orig_a1_mask] if np.any(orig_a1_mask) else np.array([])
            ])
            
            w_dist = wasserstein_distance(original_preds, cf_combined) if len(cf_combined) > 0 else np.nan
            
            return {
                'max_difference': float(np.max(abs_diffs)),
                'mean_difference': float(np.mean(abs_diffs)),
                'wasserstein_distance': w_dist
            }
        except Exception as e:
            logger.error(f"Individual fairness assessment failed for {attr}: {e}")
            return None
    
    def _assess_group_disparity(self, original_data, original_preds, 
                               cf_preds_map, attr, y_true_col, thresholds):
        """Compute group disparity metrics with proper error handling."""
        if self.perturbation_strategy != 'flip' or len(cf_preds_map) != 2:
            logger.warning(f"Group disparity skipped for '{attr}' - requires flip strategy with 2 values")
            return None
    
        try:
            cf_val0, cf_val1 = list(cf_preds_map.keys())
            priv_group = cf_val1  # Convention: last value is privileged
            unpriv_group = cf_val0
            
            # Prepare data with predictions
            data_with_preds = original_data.assign(_preds=original_preds)
            
            # Compute original group disparity
            orig_group_disp = self.fairness_assessor.compute_group_disparity(
                data=data_with_preds,
                predictions='_preds',
                y_true=y_true_col,
                privileged_group=priv_group,
                unprivileged_group=unpriv_group,
                sensitive_attr=attr
            )
            
            # Compute counterfactual group disparity (privileged intervention)
            cf_data_priv = self.causal_model.compute_counterfactual(
                intervention={attr: priv_group}
            ).assign(_preds=cf_preds_map[priv_group])
            
            cf_group_disp_priv = self.fairness_assessor.compute_group_disparity(
                data=cf_data_priv,
                sensitive_attr=attr,
                predictions='_preds',
                y_true=y_true_col,
                privileged_group=priv_group,
                unprivileged_group=unpriv_group
            )
            
            # Extract key metrics
            stat_parity_diff = orig_group_disp.get('statistical_parity_difference', np.nan)
            eod_diff = orig_group_disp.get('equal_opportunity_difference', np.nan)
            aaod_diff = orig_group_disp.get('average_abs_odds_difference', np.nan)
            
            return {
                'original': orig_group_disp,
                'counterfactual_under_do_privileged': cf_group_disp_priv,
                'max_stat_parity_diff_observed': abs(stat_parity_diff),
                'max_eod_diff_observed': abs(eod_diff),
                'max_aaod_diff_observed': abs(aaod_diff),
                'stat_parity_violation': abs(stat_parity_diff) > thresholds['group_disparity_stat_parity'],
                'equal_opp_violation': abs(eod_diff) > thresholds['group_disparity_equal_opp'],
                'avg_odds_violation': abs(aaod_diff) > thresholds['group_disparity_avg_odds']
            }
        except Exception as e:
            logger.error(f"Group disparity assessment failed for {attr}: {e}")
            return None

    def _aggregate_violations(self, fairness_report):
        """Aggregate violations across all attributes."""
        overall_violations_summary = {}
        violation_keys = set()
        
        # Collect all violation keys
        for attr, viol_dict in fairness_report['overall_violations'].items():
            violation_keys.update(viol_dict.keys())
        
        # Aggregate violations
        for key in violation_keys:
            overall_violations_summary[key] = any(
                viol_dict.get(key, False) 
                for viol_dict in fairness_report['overall_violations'].values()
            )
        
        fairness_report['overall_violations']['summary'] = overall_violations_summary

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
                    'is_sensitive': p_value < self.sensitivity_alpha if not np.isnan(p_value) else None
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

if __name__ == '__main__':
    print("\n=== Running Counterfactual Auditor ===\n")
    printer.status("Init", "Counterfactual Auditor initialized", "success")

    np.random.seed(42)
    n_samples = 2000
    data = pd.DataFrame({
        'age': np.random.normal(35, 10, n_samples).clip(18, 80),
        'gender': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),  # Sensitive attribute
        'education': np.random.choice([0, 1, 2], n_samples),
        'income': np.random.lognormal(3, 0.3, n_samples)
    })
    
    # Synthetic outcome with bias
    data['loan_approval'] = np.where(
        (data['income'] > 45) & (data['education'] > 0),
        1,
        np.where(data['gender'] == 1, 1, 0)
    ).astype(int)

    # Prediction function with bias
    def ml_predict_func(df):
        return 1 / (1 + np.exp(-(
            0.8 * (df['income']/50) + 
            0.5 * df['education'] -
            0.2 * (df['age']/100) -
            0.3 * df['gender']  # Biased term
        )))

    # Build causal model properly    
    builder = CausalGraphBuilder()
    builder.required_edges = [('gender', 'loan_approval')]
    causal_model = builder.construct_graph(data, sensitive_attrs=['gender'])

    # Initialize auditor with required parameters
    auditor = CounterfactualAuditor()
    auditor.model_predict_func = ml_predict_func
    auditor.causal_model = causal_model  

    # Run audit
    report = auditor.audit(
        data=data,
        sensitive_attrs=['gender'],
        y_true_col='loan_approval'
    )

    # Print key results
    print("\n=== Audit Summary ===")
    print("Individual Fairness Violations:")
    print(report['fairness_metrics']['individual_fairness'])
    print("\nGroup Disparity Findings:")
    print(report['fairness_metrics']['group_disparity'])
    print("\nSensitivity Analysis:")
    print(report['sensitivity_analysis']['gender'])
    print("\n=== Counterfactual Auditor Test Completed ===\n")
