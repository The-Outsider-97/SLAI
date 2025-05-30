# --- START OF FILE alignment_monitor.py ---

import torch
import hashlib
import yaml, json
import numpy as np
import pandas as pd
import networkx as nx # Import kept as requested, used in DummyCausalModel

from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from scipy.stats import wasserstein_distance # Kept as requested

# Internal imports
# Assuming SLAILM exists and can be imported

from src.agents.alignment.alignment_memory import AlignmentMemory
from src.agents.alignment.bias_detection import BiasDetector
from src.agents.alignment.fairness_evaluator import FairnessEvaluator
from src.agents.alignment.ethical_constraints import EthicalConstraints
from src.agents.alignment.counterfactual_auditor import CounterfactualAuditor, CausalModel
from src.agents.alignment.value_embedding_model import ValueEmbeddingModel
from logs.logger import get_logger

logger = get_logger("Alignment Monitor")

try:
    from models.slai_lm import get_shared_slailm, SLAILM
except ImportError:
    logger.warning("SLAILM could not be imported. ValueEmbeddingModel functionality will be limited.")
    SLAILM = type(None) # Define as NoneType if import fails
    get_shared_slailm = lambda: None

UDHR_JSON_PATH = "src/agents/alignment/templates/un_human_rights.json"

try:
    with open(UDHR_JSON_PATH, "r", encoding="utf-8") as f:
        udhr_data = json.load(f)
except FileNotFoundError:
    logger.error(f"UDHR JSON file not found at: {UDHR_JSON_PATH}")
    udhr_data = {"articles": []}

CONFIG_PATH = "src/agents/alignment/configs/alignment_config.yaml"

def dict_to_namespace(d):
    """Recursively convert dicts to SimpleNamespace for dot-access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    return d

def get_config_section(section: str, config_file_path: str) -> Dict:
    """Return raw dictionary instead of SimpleNamespace"""
    try:
        with open(config_file_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if section not in config:
            raise KeyError(f"Section '{section}' not found in config file: {config_file_path}")
        # Ensure the returned value is a dictionary, even if loaded as None or other type
        config_data = config[section]
        return config_data if isinstance(config_data, dict) else {}

    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {config_file_path}")
        raise
    except KeyError as e:
        logger.error(str(e))
        raise
    except Exception as e:
        logger.error(f"Error loading configuration section '{section}' from {config_file_path}: {e}", exc_info=True)
        raise

class AlignmentMonitor:
    """
    Alignment Monitoring System implementing real-time alignment auditing through:
    - Multi-dimensional fairness verification (Hardt et al., 2016)
    - Ethical constraint satisfaction checking (Floridi et al., 2018)
    - Longitudinal value drift detection (Liang et al., 2022)
    - Counterfactual fairness analysis (Kusner et al., 2017)

    Real-time alignment verification system implementing:
    - Continuous fairness validation
    - Ethical constraint satisfaction
    - Value trajectory monitoring
    - Automated counterfactual auditing

    Key Components:
    1. BiasDetection: Statistical parity analysis
    2. FairnessEvaluator: Group/individual fairness metrics
    3. EthicalConstraints: Deontological rule checking
    4. CounterfactualAuditor: What-if scenario analysis
    5. ValueEmbeddingModel: Human value alignment scoring
    6. AlignmentMemory: Persistent storage and analysis
    """

    def __init__(self,
                 sensitive_attributes: List[str],
                 model_predict_func: Callable[[pd.DataFrame], np.ndarray],
                 causal_model: CausalModel, # Requires a pre-built CausalModel instance
                 slai_lm: Optional[SLAILM] = None,
                 config_file_path: str = CONFIG_PATH):
        """
        Initializes the AlignmentMonitor.

        Args:
            sensitive_attributes (List[str]): List of column names identifying sensitive attributes in data.
            model_predict_func (Callable): The function of the model being monitored, taking data (pd.DataFrame)
                                            and returning predictions (np.ndarray).
            causal_model (CausalModel): A pre-constructed CausalModel instance for counterfactual analysis.
            slai_lm (Optional[SLAILM]): Shared language model instance, required if ValueEmbeddingModel is used.
            config_file_path (str): Path to the main configuration YAML file.
        """
        logger.info("Initializing Alignment Monitor...")
        self.config_path = config_file_path
        self.monitor_config = get_config_section("alignment_monitor", self.config_path) # Renamed monitor's own config
        self.sensitive_attrs = sensitive_attributes
        self.model_predict_func = model_predict_func # Crucial for counterfactual auditor

        if causal_model is None:
             logger.error("CausalModel instance is required for CounterfactualAuditor but was not provided.")
             raise ValueError("CausalModel instance cannot be None.")
        self.causal_model = causal_model             # Crucial for counterfactual auditor

        # Instantiate components
        try:
            self.alignment_memory = AlignmentMemory(
                config_section_name="alignment_memory",
                config_file_path=self.config_path
            )
            # Instantiate BiasDetector first
            self.bias_detector = BiasDetector(
                sensitive_attributes=self.sensitive_attrs,
                config_section_name="bias_detector",
                config_file_path=self.config_path
            )
            # Patch BiasDetector's config to be a dictionary if it was loaded as Namespace from its own init
            # This addresses the 'types.SimpleNamespace' object is not subscriptable error
            if not isinstance(self.bias_detector.config, dict):
                 logger.warning("Patching BiasDetector config from SimpleNamespace to dict.")
                 self.bias_detector.config = get_config_section("bias_detector", self.config_path)


            self.fairness_evaluator = FairnessEvaluator(
                sensitive_attributes=self.sensitive_attrs,
                config_section_name="fairness_evaluator",
                config_file_path=self.config_path
            )
            self.ethical_constraints = EthicalConstraints(
                config_section_name="ethical_constraints",
                config_file_path=self.config_path
            )
            self.counterfactual_auditor = CounterfactualAuditor(
                causal_model=self.causal_model,
                model_predict_func=self.model_predict_func,
                config_section_name="counterfactual_auditor",
                config_file_path=self.config_path
            )

            # Value Embedding Model is optional, might depend on SLAI_LM availability
            self.value_embedding_model = None
            vem_config = get_config_section("value_embedding", self.config_path)
            if vem_config: # Check if config section exists and is not empty
                if slai_lm is None:
                    logger.warning("SLAI LM instance not provided, attempting to get shared instance for ValueEmbeddingModel.")
                    slai_lm = get_shared_slailm() # Attempt to get globally shared instance
                if slai_lm:
                     self.value_embedding_model = ValueEmbeddingModel(
                         config_section_name="value_embedding",
                         config_file_path=self.config_path,
                         slai_lm=slai_lm
                     )
                else:
                     logger.error("Failed to initialize ValueEmbeddingModel: SLAI LM instance is required but not available.")
            else:
                logger.info("ValueEmbeddingModel configuration not found or empty. Skipping initialization.")


        except KeyError as e:
            logger.error(f"Configuration error during initialization: Missing section {e} in {self.config_path}")
            raise
        except Exception as e:
            logger.error(f"Error initializing alignment components: {e}", exc_info=True)
            raise

        self.audit_counter = 0
        self.last_audit_time = datetime.min
        logger.info("Alignment Monitor initialized successfully.")

    def monitor(self,
                data: pd.DataFrame,
                predictions: np.ndarray,
                action_context: Dict[str, Any],
                policy_params: Optional[torch.Tensor] = None, # e.g., flattened model weights or features
                labels: Optional[np.ndarray] = None,
                cultural_context_vector: Optional[torch.Tensor] = None, # For ValueEmbeddingModel
                ethical_texts: Optional[List[str]] = None, # For ValueEmbeddingModel
                perform_counterfactual_audit: bool = False # Allow forcing audit
               ) -> Dict[str, Any]:
        """
        Perform comprehensive alignment check.

        Args:
            data (pd.DataFrame): Input data used for predictions. Must contain sensitive attributes.
            predictions (np.ndarray): Model's predictions corresponding to the data. Can be probability scores or binary.
            action_context (Dict[str, Any]): Contextual information about the action/decision being evaluated.
                                              Used by EthicalConstraints.
            policy_params (Optional[torch.Tensor]): Representation of the current policy/model parameters.
                                                     Used by ValueEmbeddingModel. Expected shape [batch_size or 1, param_dim].
            labels (Optional[np.ndarray]): Ground truth labels, if available.
            cultural_context_vector (Optional[torch.Tensor]): Cultural context embedding. Used by ValueEmbeddingModel.
                                                               Expected shape [batch_size or 1, num_cultural_dimensions].
            ethical_texts (Optional[List[str]]): List of relevant ethical texts/principles. Used by ValueEmbeddingModel.
            perform_counterfactual_audit (bool): Force counterfactual audit regardless of frequency.

        Returns:
            Dict[str, Any]: A comprehensive report of the alignment status.
        """
        logger.debug("Starting alignment monitoring cycle...")
        report = {
            'timestamp': datetime.now().isoformat(),
            'bias_report': None,
            'group_fairness_report': None,
            'individual_fairness_report': None,
            'ethical_compliance_report': None,
            'counterfactual_audit_report': None,
            'value_alignment_report': None,
            'overall_status': 'PASS', # Default to PASS, change if issues detected
            'violations_detected': []
        }
        violation_flag = False

        # Ensure predictions is a numpy array for downstream processing
        if isinstance(predictions, pd.Series):
            predictions_np = predictions.to_numpy()
        elif not isinstance(predictions, np.ndarray):
            predictions_np = np.array(predictions)
        else:
            predictions_np = predictions

        if labels is not None:
             if isinstance(labels, pd.Series):
                 labels_np = labels.to_numpy()
             elif not isinstance(labels, np.ndarray):
                 labels_np = np.array(labels)
             else:
                  labels_np = labels
        else:
             labels_np = None


        # --- 1. Bias Detection ---
        try:
            logger.debug("Running Bias Detector...")
            # Bias detector might need binary predictions for some metrics
            # Assuming predictions_np could be probabilities, binarize if needed
            # Let's assume bias_detector handles prediction types internally for now
            report['bias_report'] = self.bias_detector.compute_metrics(data, predictions_np, labels_np)
            for metric, groups in report['bias_report'].items():
                 if isinstance(groups, dict):
                     for group_id, result in groups.items():
                         if isinstance(result, dict) and result.get('significant', False):
                             violation_flag = True
                             value = result.get('value', float('nan'))
                             pval = result.get('adj_p_value', float('nan'))
                             violation_details = f"Bias Violation: Metric={metric}, Group={group_id}, Value={value:.4f}, p-value={pval:.4f}"
                             logger.warning(violation_details)
                             report['violations_detected'].append(violation_details)
        except Exception as e:
            logger.error(f"Bias detection failed: {e}", exc_info=True)
            report['bias_report'] = {'error': str(e)}
            violation_flag = True
            report['violations_detected'].append("Bias Detection Error")

        # --- 2. Group Fairness Evaluation ---
        if labels_np is not None:
            try:
                logger.debug("Running Group Fairness Evaluator...")
                # Group fairness typically needs binary predictions and labels
                # Assuming predictions are appropriate (e.g., binary if required by evaluator's metrics)
                report['group_fairness_report'] = self.fairness_evaluator.evaluate_group_fairness(data, predictions_np, labels_np)
                for attr, metrics in report['group_fairness_report'].items():
                    if isinstance(metrics, dict):
                        for metric, result in metrics.items():
                            if isinstance(result, dict) and result.get('significant', False):
                                violation_flag = True
                                value = result.get('value', float('nan'))
                                pval = result.get('p_value', float('nan'))
                                violation_details = f"Group Fairness Violation: Attribute={attr}, Metric={metric}, Value={value:.4f}, p-value={pval:.4f}"
                                logger.warning(violation_details)
                                report['violations_detected'].append(violation_details)
            except Exception as e:
                logger.error(f"Group fairness evaluation failed: {e}", exc_info=True)
                report['group_fairness_report'] = {'error': str(e)}
                violation_flag = True
                report['violations_detected'].append("Group Fairness Error")
        else:
            logger.info("Skipping group fairness evaluation: requires labels.")
            report['group_fairness_report'] = {'status': 'skipped', 'reason': 'Labels not provided'}

        # --- 3. Individual Fairness Evaluation ---
        try:
            logger.debug("Running Individual Fairness Evaluator...")
            features_for_individual = data.drop(columns=self.sensitive_attrs, errors='ignore')
            numeric_cols = features_for_individual.select_dtypes(include=np.number).columns
            if not numeric_cols.empty:
                 features_numeric = features_for_individual[numeric_cols]
                 # Use probability predictions if meaningful, otherwise binary
                 # Assuming predictions_np holds the appropriate format
                 report['individual_fairness_report'] = self.fairness_evaluator.evaluate_individual_fairness(
                     features_numeric,
                     predictions_np # Pass the numpy array
                 )
                 indiv_report = report['individual_fairness_report']
                 if isinstance(indiv_report, dict):
                     violation_rate = indiv_report.get('fairness_violations', {}).get('violation_rate', 0.0)
                     lipschitz_const = indiv_report.get('lipschitz_constant', 0.0)
                     # Define thresholds (should be in config ideally)
                     if violation_rate > 0.05 or lipschitz_const > 1.5: # Example thresholds
                         violation_flag = True
                         violation_details = f"Individual Fairness Violation: Violation Rate={violation_rate:.4f}, Lipschitz={lipschitz_const:.4f}"
                         logger.warning(violation_details)
                         report['violations_detected'].append(violation_details)
            else:
                 logger.warning("Skipping individual fairness evaluation: No numeric features found after excluding sensitive attributes.")
                 report['individual_fairness_report'] = {'status': 'skipped', 'reason': 'No numeric features for similarity calculation.'}


        except Exception as e:
            logger.error(f"Individual fairness evaluation failed: {e}", exc_info=True)
            report['individual_fairness_report'] = {'error': str(e)}
            violation_flag = True
            report['violations_detected'].append("Individual Fairness Error")

        # --- 4. Ethical Constraint Checking ---
        try:
            logger.debug("Running Ethical Constraints Check...")
            report['ethical_compliance_report'] = self.ethical_constraints.enforce(action_context)
            if not report['ethical_compliance_report'].get('approved', True):
                violation_flag = True
                violations = report['ethical_compliance_report'].get('violations', [])
                logger.warning(f"Ethical Constraint Violation(s): {violations}")
                report['violations_detected'].extend([f"Ethical Violation: {v}" for v in violations])
                corrections = report['ethical_compliance_report'].get('corrective_actions', [])
                if corrections:
                     logger.info(f"Ethical corrections applied: {corrections}")
                     # Future: self.alignment_memory.apply_correction(correction_details, effect_details)

        except Exception as e:
            logger.error(f"Ethical constraints enforcement failed: {e}", exc_info=True)
            report['ethical_compliance_report'] = {'error': str(e)}
            violation_flag = True
            report['violations_detected'].append("Ethical Constraints Error")


        # --- 5. Counterfactual Fairness Audit (Conditional) ---
        self.audit_counter += 1
        audit_frequency = self.monitor_config.get('audit_frequency', 1000) # Default if not in config
        # Check frequency based on counter
        should_audit = perform_counterfactual_audit or (self.audit_counter % audit_frequency == 0)

        if should_audit:
            logger.info(f"Running Counterfactual Audit (Triggered by count: {self.audit_counter} or forced)...")
            try:
                 # Determine label column name if labels were provided and exist in data
                 label_col_name = None
                 if labels_np is not None:
                     # Check if 'label' is a column name in the original df
                     if 'label' in data.columns:
                          # Potentially verify if data['label'] matches labels_np? Risky. Assume 'label' is it.
                          label_col_name = 'label'
                     else:
                          # If labels provided but not a column, maybe add temporarily? Complex.
                          logger.warning("Labels provided but 'label' column not found in data for counterfactual audit.")
                          # Or just don't pass y_true_col if unsure. Group metrics won't run.

                 report['counterfactual_audit_report'] = self.counterfactual_auditor.audit(
                     data=data.copy(), # Pass a copy to avoid modification issues if auditor adds cols
                     sensitive_attrs=self.sensitive_attrs,
                     y_true_col= label_col_name
                 )
                 self.last_audit_time = datetime.now()
                 cf_report = report['counterfactual_audit_report']
                 if isinstance(cf_report, dict) and 'fairness_metrics' in cf_report:
                      violations = cf_report['fairness_metrics'].get('overall_violations', {}).get('summary', {})
                      if any(violations.values()):
                           violation_flag = True
                           detected_cf_violations = [k for k, v in violations.items() if v]
                           logger.warning(f"Counterfactual Fairness Violation(s) Detected: {detected_cf_violations}")
                           report['violations_detected'].extend([f"Counterfactual Violation: {v}" for v in detected_cf_violations])

            except Exception as e:
                logger.error(f"Counterfactual fairness audit failed: {e}", exc_info=True)
                report['counterfactual_audit_report'] = {'error': str(e)}
                violation_flag = True
                report['violations_detected'].append("Counterfactual Audit Error")
        else:
            report['counterfactual_audit_report'] = {'status': 'skipped', 'reason': f'Audit frequency ({audit_frequency}) not met'}

        # --- 6. Value Trajectory / Alignment Scoring ---
        if self.value_embedding_model and policy_params is not None and cultural_context_vector is not None and ethical_texts:
            try:
                logger.debug("Running Value Embedding Model Analysis...")
                if policy_params.ndim == 1: policy_params = policy_params.unsqueeze(0)
                if cultural_context_vector.ndim == 1: cultural_context_vector = cultural_context_vector.unsqueeze(0)
                batch_size = policy_params.size(0)

                # Ensure ethical_texts matches batch size
                if len(ethical_texts) == 1 and batch_size > 1:
                    ethical_texts = ethical_texts * batch_size
                elif len(ethical_texts) != batch_size:
                     raise ValueError(f"Mismatch between policy_params batch size ({batch_size}) and ethical_texts length ({len(ethical_texts)})")

                value_inputs = {
                    "value_text": ethical_texts,
                    "cultural_context": cultural_context_vector.float(),
                    "policy_params": policy_params.float()
                }
                with torch.no_grad():
                    self.value_embedding_model.eval()
                    value_outputs = self.value_embedding_model(value_inputs)

                report['value_alignment_report'] = {
                    'alignment_score': value_outputs['alignment_score'].mean().item(),
                    'preference_score': value_outputs['preference_score'].mean().item(),
                }
                alignment_threshold = self.monitor_config.get('value_alignment_threshold', 0.5) # Default if not in config
                current_score = report['value_alignment_report']['alignment_score']
                if current_score < alignment_threshold:
                     violation_flag = True
                     violation_details = f"Value Alignment Violation: Score={current_score:.4f} < Threshold={alignment_threshold}"
                     logger.warning(violation_details)
                     report['violations_detected'].append(violation_details)

            except Exception as e:
                logger.error(f"Value alignment scoring failed: {e}", exc_info=True)
                report['value_alignment_report'] = {'error': str(e)}
                violation_flag = True
                report['violations_detected'].append("Value Alignment Error")
        else:
             missing_components = []
             if not self.value_embedding_model: missing_components.append("ValueEmbeddingModel")
             if policy_params is None: missing_components.append("policy_params")
             if cultural_context_vector is None: missing_components.append("cultural_context_vector")
             if not ethical_texts: missing_components.append("ethical_texts")
             if missing_components:
                report['value_alignment_report'] = {'status': 'skipped', 'reason': f'Missing components/inputs: {", ".join(missing_components)}'}


        # --- 7. Update Alignment Memory ---
        try:
             primary_metric_name = "alignment_metric" # Default name
             primary_metric_value = np.nan
             threshold = np.nan
             is_violation = False # Default

             # Prioritize logging fairness metrics if available and significant
             if report['group_fairness_report'] and 'error' not in report['group_fairness_report']:
                 max_disp = 0
                 worst_metric = "N/A"
                 for attr, metrics in report['group_fairness_report'].items():
                     stat_parity_res = metrics.get('statistical_parity', {})
                     if stat_parity_res:
                         value = abs(stat_parity_res.get('value', 0))
                         if value > max_disp:
                              max_disp = value
                              worst_metric = f"stat_parity_{attr}"

                 primary_metric_value = max_disp
                 primary_metric_name = "max_abs_statistical_parity_diff"
                 threshold = self.monitor_config.get('fairness_thresholds', {}).get('group_disparity_stat_parity', 0.1) # Get threshold from config
                 is_violation = primary_metric_value > threshold

             # Fallback to bias detection metrics
             elif report['bias_report'] and 'error' not in report['bias_report']:
                  di_report = report['bias_report'].get('disparate_impact', {})
                  if isinstance(di_report, dict) and 'global_ratio' in di_report:
                      primary_metric_value = di_report['global_ratio']
                      primary_metric_name = "min_disparate_impact_ratio"
                      threshold = 0.8 # Standard threshold
                      is_violation = primary_metric_value < threshold # Violation if value is LOW

             # Fallback to value alignment score
             elif report['value_alignment_report'] and 'error' not in report['value_alignment_report']:
                 primary_metric_value = report['value_alignment_report']['alignment_score']
                 primary_metric_name = "value_alignment_score"
                 threshold = self.monitor_config.get('value_alignment_threshold', 0.5)
                 is_violation = primary_metric_value < threshold # Violation if value is LOW

             # Only log if a valid metric was found
             if not np.isnan(primary_metric_value) and not np.isnan(threshold):
                 # Create context hash carefully
                 try:
                     context_hash = hashlib.sha256(pd.util.hash_pandas_object(data, index=True).values).hexdigest()
                 except Exception: # Handle potential hashing errors
                     context_hash = "hashing_error"
                 log_context = {'action_type': action_context.get('type', 'unknown'), 'data_hash': context_hash}

                 self.alignment_memory.log_evaluation(
                     metric=primary_metric_name,
                     value=primary_metric_value,
                     threshold=threshold, # Pass threshold used for violation check
                     # is_violation flag is determined internally in log_evaluation now
                     context=log_context
                 )
             else:
                  logger.debug("No primary metric extracted for AlignmentMemory logging.")

        except Exception as e:
             logger.error(f"Failed to update Alignment Memory: {e}", exc_info=True)
             # Optionally mark as violation:
             # violation_flag = True
             # report['violations_detected'].append("Alignment Memory Update Error")


        # --- Finalize Report ---
        if violation_flag:
            report['overall_status'] = 'FAIL'
            logger.warning(f"Alignment Check Failed. Violations: {report['violations_detected']}")
        else:
            report['overall_status'] = 'PASS'
            logger.info("Alignment Check Passed.")

        logger.debug("Alignment monitoring cycle finished.")
        return report

# --- JSON serialization helper ---
def json_default_serializer(obj):
    """ Handle non-serializable types for JSON dump """
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8,
                        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32,
                          np.float64)):
        # Handle NaN and Inf/-Inf
        if np.isnan(obj):
            return None # Or 'NaN' as string
        elif np.isinf(obj):
            return 'Infinity' if obj > 0 else '-Infinity'
        return float(obj)
    elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
        return {'real': obj.real, 'imag': obj.imag}
    elif isinstance(obj, (np.ndarray,)): # Handle arrays
        return obj.tolist()
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.void)):
        return None
    elif isinstance(obj, (datetime)): # Handle datetime objects
         return obj.isoformat()
    elif isinstance(obj, SimpleNamespace): # Handle SimpleNamespace
         return vars(obj)
    try:
         # Try a generic string representation if specific types fail
         return str(obj)
    except Exception:
         # Fallback if str() fails
         return f"<non-serializable: {type(obj).__name__}>"


# Example Usage (requires setting up dependencies like causal_model, slai_lm, predict_func)
if __name__ == '__main__':
    # --- Full Architecture Test ---
    from datetime import datetime, timedelta
    import numpy as np
    import pandas as pd
    import torch
    from faker import Faker

    # 1. Initialize Core Components -------------------------------------------------
    logger.info("Initializing Full Alignment Architecture...")
    
    # 1.1 Create synthetic causal model
    class DummyCausalModel:
        def compute_counterfactual(self, intervention):
            # Simple implementation for testing
            cf_data = pd.DataFrame({
                'gender': [intervention.get('gender', 1)]*100,
                'age': np.random.normal(40, 15, 100),
                'income': np.random.lognormal(3.5, 0.3, 100)
            })
            return cf_data

    causal_model = DummyCausalModel()

    # 1.2 Mock SLAI language model
    class MockSLM:
        def process_input(self, prompt, text):
            return {"tokens": ["mock"] * 20}
    
    # 1.3 Initialize all components
    sensitive_attrs = ['gender', 'age_group']
    
    # Prediction function for a loan approval model
    def ml_predict_func(df):
        return 1 / (1 + np.exp(-(
            0.7 * (df['income']/50) + 
            0.4 * df['education'] -
            0.3 * (df['age']/100) -
            0.2 * df['gender']
        )))

    # Create alignment monitor with all dependencies
    monitor = AlignmentMonitor(
        sensitive_attributes=sensitive_attrs,
        model_predict_func=ml_predict_func,
        causal_model=causal_model,
        slai_lm=MockSLM(),
        config_file_path=CONFIG_PATH
    )

    # 2. Generate Test Data ---------------------------------------------------------
    logger.info("Generating Synthetic Test Data...")
    fake = Faker()
    num_samples = 5000

    # Demographic data
    test_data = pd.DataFrame({
        'gender': np.random.choice([0, 1], num_samples),
        'age': np.random.normal(45, 15, num_samples).clip(18, 80),
        'education': np.random.choice([1, 2, 3, 4], num_samples),
        'income': np.random.lognormal(3.5, 0.3, num_samples)
    })
    
    # Add derived features
    test_data['age_group'] = pd.cut(test_data['age'], 
                                   bins=[18, 30, 45, 60, 100],
                                   labels=['18-30', '31-45', '46-60', '60+'])
    
    # Generate synthetic labels with bias
    test_data['label'] = np.where(
        (test_data['income'] > 50) & (test_data['education'] > 2),
        1,
        np.where(test_data['gender'] == 1, 1, 0)
    )

    # 3. Simulate Operational Monitoring --------------------------------------------
    logger.info("Starting Monitoring Simulation...")
    
    # Create policy parameters and cultural context
    policy_params = torch.randn(1, 4096)  # Matching policy_encoder input size
    cultural_context = torch.tensor([[0.8, 0.2, 0.5, 0.7, 0.3, 0.4]])  # 6 cultural dimensions
    
    # Ethical guidelines from UDHR
    ethical_guidelines = [
        "All human beings are born free and equal in dignity and rights.",
        "Everyone has the right to life, liberty and security of person."
    ]

    # Run monitoring cycles
    for cycle in range(3):
        logger.info(f"\n=== Monitoring Cycle {cycle+1} ===")
        
        # Generate fresh predictions
        predictions = ml_predict_func(test_data)
        
        # Create action context
        action_context = {
            'decision_engine': {'is_active': True},
            'affected_people': [{'id': i} for i in range(10)],
            'action_parameters': {'kinetic_energy': 40}
        }

        # Run full monitoring pipeline
        report = monitor.monitor(
            data=test_data,
            predictions=predictions,
            action_context=action_context,
            policy_params=policy_params,
            labels=test_data['label'].values,
            cultural_context_vector=cultural_context,
            ethical_texts=ethical_guidelines,
            perform_counterfactual_audit=(cycle == 0)
        )

        # Print summary
        print(f"\nCycle {cycle+1} Report Summary:")
        print(f"Overall Status: {report['overall_status']}")
        print(f"Violations Detected: {len(report['violations_detected'])}")
        print("Key Metrics:")
        print(f"- Value Alignment Score: {report.get('value_alignment_report', {}).get('alignment_score', 'N/A'):.2f}")
        print(f"- Group Fairness Violations: {len(report['group_fairness_report'] or [])}")
        print(f"- Ethical Constraint Violations: {len(report['ethical_compliance_report']['violations'])}")

    # 4. Generate Final Reports -----------------------------------------------------
    logger.info("\n=== Final System Reports ===")
    
    # Alignment Memory Report
    print("\nAlignment Memory Analysis:")
    memory_report = monitor.alignment_memory.get_memory_report()
    print(f"- Concept Drift Detected: {memory_report['drift_status']}")
    print(f"- Mean Alignment Score: {memory_report['temporal_summary']['value']['mean']:.2f}")

    # Ethical Constraints State
    print("\nEthical Constraints State:")
    print(f"Current Weights: {monitor.ethical_constraints.constraint_weights}")

    # Value Embedding Analysis
    if monitor.value_embedding_model:
        print("\nValue Embedding Space Analysis:")
        test_emb = monitor.value_embedding_model.encode_value(
            ["Test value"], cultural_context
        )
        print(f"Embedding Shape: {test_emb.shape}")

    logger.info("Full Architecture Test Completed Successfully!")
