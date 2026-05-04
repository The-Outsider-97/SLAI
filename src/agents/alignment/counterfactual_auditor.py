"""
Counterfactual Fairness Audit System

Implements causal counterfactual analysis for alignment verification through:
Structural causal model interventions (Pearl, 2009)
Counterfactual fairness estimation (Kusner et al., 2017)
Policy decision sensitivity analysis
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import networkx as nx

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from scipy.stats import ttest_ind, wasserstein_distance

from .utils.config_loader import load_global_config, get_config_section
from .utils.alignment_errors import *
from .utils.alignment_helpers import *
from .auditors.causal_model import *
from .auditors.fairness_metrics import CounterfactualFairness
from .alignment_memory import AlignmentMemory
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Countefactual Auditor")
printer = PrettyPrinter

@dataclass(frozen=True)
class CounterfactualScenario:
    """Canonical representation of a single counterfactual intervention scenario."""

    sensitive_attribute: str
    scenario_id: str
    intervention_value: Any
    intervention: Dict[str, Any]
    strategy: str
    sample_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensitive_attribute": self.sensitive_attribute,
            "scenario_id": self.scenario_id,
            "intervention_value": json_safe(self.intervention_value),
            "intervention": json_safe(self.intervention),
            "strategy": self.strategy,
            "sample_size": self.sample_size,
            "metadata": json_safe(self.metadata),
        }


class CounterfactualAuditor:
    """
    Causal counterfactual analysis system implementing:
    - Generation of counterfactual scenarios based on interventions.
    - Estimation of model predictions under these scenarios using a CausalModel.
    - Assessment of counterfactual fairness metrics.
    - Analysis of decision sensitivity to sensitive attribute changes.

    Requires a compatible CausalModel and a model prediction function, but can
    also build a causal model from audit data when configured to do so.
    """

    DEFAULT_REQUIRED_CONFIG_KEYS = (
        "perturbation_strategy",
        "perturbation_magnitude",
        "num_counterfactual_samples",
        "sensitivity_alpha",
        "fairness_thresholds",
    )

    def __init__(
        self,
        causal_model: Optional[CausalModel] = None,
        model_predict_func: Optional[Callable[[pd.DataFrame], Union[np.ndarray, Sequence[float]]]] = None,
        *,
        config_section_name: str = "counterfactual_auditor",
        config_file_path: Optional[str] = None,
        alignment_memory: Optional[AlignmentMemory] = None,
        fairness_assessor: Optional[CounterfactualFairness] = None,
    ):
        self.config = load_global_config()
        self.config_section_name = ensure_non_empty_string(
            config_section_name,
            "config_section_name",
            error_cls=ConfigurationError,
        )
        self.config_file_path = config_file_path
        self.auditor_config = get_config_section(self.config_section_name)
        self._validate_config()

        configured_sensitive = self.auditor_config.get(
            "default_sensitive_attributes",
            self.config.get("sensitive_attributes", []),
        )
        self.sensitive_attributes = list(
            normalize_sensitive_attributes(configured_sensitive, lowercase=False, allow_empty=True)
        )
        self.sensitive_attrs = self.sensitive_attributes

        self.perturbation_strategy = ensure_non_empty_string(
            self.auditor_config.get("perturbation_strategy", "flip"),
            "perturbation_strategy",
            error_cls=ConfigurationError,
        ).lower()
        self.perturbation_magnitude = coerce_float(
            self.auditor_config.get("perturbation_magnitude", 0.1),
            field_name="perturbation_magnitude",
            minimum=0.0,
        )
        self.num_counterfactual_samples = coerce_positive_int(
            self.auditor_config.get("num_counterfactual_samples", 2),
            field_name="num_counterfactual_samples",
        )
        self.sensitivity_alpha = coerce_probability(
            self.auditor_config.get("sensitivity_alpha", 0.05),
            field_name="sensitivity_alpha",
        )
        self.random_seed = coerce_int(
            self.auditor_config.get("random_seed", 42),
            field_name="random_seed",
        )
        self.default_prediction_threshold = coerce_probability(
            self.auditor_config.get("prediction_threshold", 0.5),
            field_name="prediction_threshold",
        )
        self.max_scenarios_per_attribute = coerce_positive_int(
            self.auditor_config.get("max_scenarios_per_attribute", max(2, self.num_counterfactual_samples)),
            field_name="max_scenarios_per_attribute",
        )
        self.counterfactual_predict_exogenous_strategy = ensure_non_empty_string(
            self.auditor_config.get("counterfactual_predict_exogenous_strategy", "retain_observed"),
            "counterfactual_predict_exogenous_strategy",
            error_cls=ConfigurationError,
        ).lower()
        self.auto_build_causal_model = coerce_bool(
            self.auditor_config.get("auto_build_causal_model", True),
            field_name="auto_build_causal_model",
        )
        self.enable_memory_logging = coerce_bool(
            self.auditor_config.get("enable_memory_logging", True),
            field_name="enable_memory_logging",
        )
        self.strict_memory_integration = coerce_bool(
            self.auditor_config.get("strict_memory_integration", False),
            field_name="strict_memory_integration",
        )
        self.include_counterfactual_samples = coerce_bool(
            self.auditor_config.get("include_counterfactual_samples", False),
            field_name="include_counterfactual_samples",
        )
        self.counterfactual_sample_cap = coerce_positive_int(
            self.auditor_config.get("counterfactual_sample_cap", 25),
            field_name="counterfactual_sample_cap",
        )
        self.report_distribution_samples = coerce_bool(
            self.auditor_config.get("report_distribution_samples", False),
            field_name="report_distribution_samples",
        )
        self.log_path_specific_effects = coerce_bool(
            self.auditor_config.get("log_path_specific_effects", True),
            field_name="log_path_specific_effects",
        )
        self.correction_levels = list(self.auditor_config.get("corrections", {}).get("levels", []))
        self.fairness_thresholds = self._normalize_fairness_thresholds(
            self.auditor_config.get("fairness_thresholds", {})
        )
        self.attribute_mediators = self._normalize_attribute_mediators(
            self.auditor_config.get("attribute_mediators", {})
        )
        self.required_edges = self._normalize_edge_constraints(
            self.auditor_config.get("required_edges", [])
        )
        self.quantile_grid = self._normalize_quantile_grid(
            self.auditor_config.get("quantile_grid", [0.1, 0.5, 0.9])
        )

        self.model_predict_func = model_predict_func
        self.alignment_memory = alignment_memory or AlignmentMemory()
        self.fairness_assessor = fairness_assessor or CounterfactualFairness(memory=self.alignment_memory)
        self.causal_model = causal_model
        self.random_state = np.random.default_rng(self.random_seed)

        if config_file_path:
            logger.debug(
                "CounterfactualAuditor received config_file_path=%s but retained global config loader handling.",
                config_file_path,
            )

        logger.info(
            "CounterfactualAuditor initialized | strategy=%s samples=%s auto_build=%s",
            self.perturbation_strategy,
            self.num_counterfactual_samples,
            self.auto_build_causal_model,
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def _validate_config(self) -> None:
        try:
            ensure_mapping(
                self.auditor_config,
                self.config_section_name,
                allow_empty=False,
                error_cls=ConfigurationError,
            )
            for key in self.DEFAULT_REQUIRED_CONFIG_KEYS:
                if key not in self.auditor_config:
                    raise ConfigurationError(
                        f"Missing required counterfactual auditor configuration key: '{key}'.",
                        context={
                            "config_section": self.config_section_name,
                            "config_path": self.config.get("__config_path__"),
                        },
                    )
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=ConfigurationError,
                message="CounterfactualAuditor configuration validation failed.",
                context={
                    "config_section": self.config_section_name,
                    "config_path": self.config.get("__config_path__"),
                },
            ) from exc

    def _normalize_fairness_thresholds(self, thresholds: Mapping[str, Any]) -> Dict[str, float]:
        normalized = {
            "individual_fairness_mean_diff": 0.1,
            "individual_fairness_max_diff": 0.3,
            "group_disparity_stat_parity": 0.1,
            "group_disparity_equal_opp": 0.1,
            "group_disparity_avg_odds": 0.1,
            "equalized_odds_gap_change": 0.1,
            "causal_effect_ate": 0.05,
            "sensitivity_mean_shift": 0.05,
            "sensitivity_cohens_d": 0.2,
            "overall_bias": 0.1,
        }
        if thresholds:
            source = ensure_mapping(thresholds, "fairness_thresholds", allow_empty=True, error_cls=ConfigurationError)
            for key, value in source.items():
                if isinstance(value, Mapping):
                    continue
                normalized[str(key)] = coerce_float(value, field_name=f"fairness_thresholds.{key}", minimum=0.0)
        return normalized

    def _normalize_edge_constraints(self, edges: Optional[Sequence[Any]]) -> List[Tuple[str, str]]:
        if edges is None:
            return []
        normalized: List[Tuple[str, str]] = []
        for edge in ensure_sequence(edges, "required_edges", allow_empty=True, error_cls=ConfigurationError):
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                raise ConfigurationError(
                    "required_edges must contain [source, target] pairs.",
                    context={"edge": edge},
                )
            normalized.append(
                (
                    ensure_non_empty_string(edge[0], "edge_source", error_cls=ConfigurationError),
                    ensure_non_empty_string(edge[1], "edge_target", error_cls=ConfigurationError),
                )
            )
        return normalized

    def _normalize_attribute_mediators(self, mapping: Mapping[str, Any]) -> Dict[str, List[str]]:
        normalized: Dict[str, List[str]] = {}
        source = ensure_mapping(mapping, "attribute_mediators", allow_empty=True, error_cls=ConfigurationError)
        for key, value in source.items():
            mediators = [
                ensure_non_empty_string(item, f"attribute_mediators.{key}", error_cls=ConfigurationError)
                for item in ensure_sequence(value, f"attribute_mediators.{key}", allow_empty=True, error_cls=ConfigurationError)
            ]
            normalized[str(key)] = mediators
        return normalized

    def _normalize_quantile_grid(self, values: Sequence[Any]) -> Tuple[float, ...]:
        grid = [
            coerce_probability(value, field_name="quantile_grid_value")
            for value in ensure_sequence(values, "quantile_grid", allow_empty=False, error_cls=ConfigurationError)
        ]
        return tuple(sorted(set(grid)))

    # ------------------------------------------------------------------
    # Public mutators
    # ------------------------------------------------------------------
    def set_model_predict_func(self, predict_func: Callable[[pd.DataFrame], Union[np.ndarray, Sequence[float]]]) -> None:
        """Set the model prediction function dynamically."""
        if not callable(predict_func):
            raise TypeMismatchError(
                "model_predict_func must be callable.",
                context={"actual_type": type(predict_func).__name__},
            )
        self.model_predict_func = predict_func

    def set_causal_model(self, causal_model: CausalModel) -> None:
        """Set the causal model dynamically."""
        if not isinstance(causal_model, CausalModel):
            raise TypeMismatchError(
                "causal_model must be an instance of CausalModel.",
                context={"actual_type": type(causal_model).__name__},
            )
        self.causal_model = causal_model

    # ------------------------------------------------------------------
    # Core audit flow
    # ------------------------------------------------------------------
    def audit(
        self,
        data: pd.DataFrame,
        sensitive_attrs: Optional[Sequence[str]] = None,
        y_true_col: Optional[str] = None,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform a comprehensive counterfactual fairness audit.

        The audit:
        1. validates and prepares the dataset,
        2. resolves or builds a causal model,
        3. generates attribute-specific interventions,
        4. scores model predictions under counterfactual scenarios,
        5. computes fairness and sensitivity diagnostics,
        6. logs longitudinal evidence into alignment memory,
        7. returns a machine-readable audit package.
        """
        try:
            audit_data, resolved_sensitive_attrs, resolved_y_true_col = self._validate_audit_inputs(
                data,
                sensitive_attrs=sensitive_attrs,
                y_true_col=y_true_col,
            )
            audit_id = generate_audit_id()
            trace_id = generate_trace_id()
            audit_context = merge_mappings(
                context,
                {
                    "audit_id": audit_id,
                    "trace_id": trace_id,
                    "samples": len(audit_data),
                    "sensitive_attrs": list(resolved_sensitive_attrs),
                    "task_type": self.config.get("task_type", "classification"),
                    "domain": audit_data.attrs.get("domain", "unknown"),
                },
            )
            audit_metadata = normalize_metadata(metadata)

            self._ensure_runtime_dependencies(audit_data, resolved_sensitive_attrs)
            original_preds = self._get_predictions(audit_data)
            self._log_metric(
                metric="counterfactual_audit_started",
                value=1.0,
                threshold=0.5,
                context=audit_context,
                source="counterfactual_auditor",
                tags=["audit", "start"],
                metadata=audit_metadata,
            )

            counterfactual_results, interventions_applied = self._run_counterfactual_scenarios(
                audit_data,
                resolved_sensitive_attrs,
                audit_context=audit_context,
            )
            fairness_report = self._assess_fairness_violations(
                original_data=audit_data,
                original_preds=original_preds,
                cf_results=counterfactual_results,
                sensitive_attrs=list(resolved_sensitive_attrs),
                y_true_col=resolved_y_true_col,
                audit_context=audit_context,
                metadata=audit_metadata,
            )
            sensitivity_report = self._analyze_decision_sensitivity(
                original_preds=original_preds,
                cf_results=counterfactual_results,
                metadata=audit_metadata,
                context=audit_context,
            )

            causal_effects = self._estimate_sensitive_attribute_effects(
                audit_data,
                resolved_sensitive_attrs,
                resolved_y_true_col,
                audit_context=audit_context,
                metadata=audit_metadata,
            )
            recommended_action = self._recommend_correction_level(fairness_report, sensitivity_report)
            overall_bias = self._calculate_overall_bias(fairness_report)
            total_violations = self._count_violations(fairness_report)
            drift_status = self._detect_drift(audit_context)

            self._record_outcome(
                context=audit_context,
                outcome={
                    "bias_rate": overall_bias,
                    "ethics_violations": total_violations,
                    "alignment_score": max(0.0, 1.0 - overall_bias),
                    "violation": total_violations > 0,
                },
                source="counterfactual_auditor",
                tags=["audit", "outcome"],
                metadata=merge_mappings(audit_metadata, {"recommended_action": recommended_action}),
            )

            report = {
                "audit_id": audit_id,
                "trace_id": trace_id,
                "audit_timestamp": utc_timestamp(),
                "audit_context": normalize_context(audit_context, drop_none=False),
                "audit_config": {
                    "config_section": self.config_section_name,
                    "perturbation_strategy": self.perturbation_strategy,
                    "perturbation_magnitude": self.perturbation_magnitude,
                    "num_counterfactual_samples": self.num_counterfactual_samples,
                    "sensitivity_alpha": self.sensitivity_alpha,
                    "prediction_threshold": self.default_prediction_threshold,
                },
                "causal_graph_info": self._causal_graph_info(),
                "fairness_metrics": fairness_report,
                "sensitivity_analysis": sensitivity_report,
                "causal_effect_estimates": causal_effects,
                "interventions_applied": interventions_applied,
                "drift_status": drift_status,
                "overall_bias": overall_bias,
                "violation_count": total_violations,
                "recommended_action": recommended_action,
                "metadata": audit_metadata,
            }

            self._log_metric(
                metric="counterfactual_audit_completed",
                value=float(overall_bias),
                threshold=self.fairness_thresholds["overall_bias"],
                context=audit_context,
                source="counterfactual_auditor",
                tags=["audit", "complete"],
                metadata=merge_mappings(audit_metadata, {"violations": total_violations}),
            )
            return report
        except AlignmentError:
            raise
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=CounterfactualAuditError,
                message="Counterfactual audit execution failed.",
                context={
                    "config_section": self.config_section_name,
                    "sensitive_attrs": list(sensitive_attrs or []),
                    "y_true_col": y_true_col,
                },
                metadata=normalize_metadata(metadata),
            ) from exc

    # ------------------------------------------------------------------
    # Validation and dependency resolution
    # ------------------------------------------------------------------
    def _validate_audit_inputs(
        self,
        data: pd.DataFrame,
        *,
        sensitive_attrs: Optional[Sequence[str]],
        y_true_col: Optional[str],
    ) -> Tuple[pd.DataFrame, Tuple[str, ...], Optional[str]]:
        ensure_instance(data, pd.DataFrame, "data", error_cls=DataValidationError)
        if data.empty:
            raise DataValidationError("Audit data must not be empty.")

        resolved_sensitive = normalize_sensitive_attributes(
            sensitive_attrs if sensitive_attrs is not None else self.sensitive_attrs,
            lowercase=False,
            allow_empty=False,
        )
        validate_sensitive_attributes(data, resolved_sensitive, field_name="data", error_cls=SensitiveAttributeError)
        resolved_y_true = None
        if y_true_col is not None:
            resolved_y_true = ensure_non_empty_string(y_true_col, "y_true_col", error_cls=DataValidationError)
            ensure_columns_present(data, [resolved_y_true], field_name="data", error_cls=DataValidationError)
        return data.copy(), resolved_sensitive, resolved_y_true

    def _ensure_runtime_dependencies(self, data: pd.DataFrame, sensitive_attrs: Sequence[str]) -> None:
        if self.model_predict_func is None:
            raise InitializationError(
                "model_predict_func is not set. Please assign it before calling audit().",
                context={"operation": "audit"},
            )
        if self.causal_model is None:
            if not self.auto_build_causal_model:
                raise InitializationError(
                    "causal_model is not set and auto_build_causal_model is disabled.",
                    context={"operation": "audit"},
                )
            builder = CausalGraphBuilder(alignment_memory=self.alignment_memory)
            if self.required_edges:
                builder.required_edges = list(self.required_edges)
            self.causal_model = builder.construct_graph(data, sensitive_attrs=sensitive_attrs)

        if not isinstance(self.causal_model, CausalModel):
            raise TypeMismatchError(
                "causal_model must be an instance of CausalModel.",
                context={"actual_type": type(self.causal_model).__name__},
            )

    # ------------------------------------------------------------------
    # Prediction and scenario execution
    # ------------------------------------------------------------------
    def _get_predictions(self, data: pd.DataFrame) -> np.ndarray:
        printer.status("Init", "Auditor predictor initialized", "info")
        try:
            raw_preds = self.model_predict_func(data)  # type: ignore[misc]
            preds = np.asarray(raw_preds, dtype=float).reshape(-1)
            if preds.shape[0] != len(data):
                raise DataValidationError(
                    "Prediction function returned an array with incorrect length.",
                    context={"expected": len(data), "actual": int(preds.shape[0])},
                )
            return preds
        except AlignmentError:
            raise
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=CounterfactualAuditError,
                message="Model prediction function failed during counterfactual audit.",
                context={"operation": "predict"},
            ) from exc

    def _run_counterfactual_scenarios(
        self,
        data: pd.DataFrame,
        sensitive_attrs: Sequence[str],
        *,
        audit_context: Mapping[str, Any],
    ) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Dict[str, Dict[str, Any]]]]:
        results: Dict[str, Dict[str, Dict[str, Any]]] = {}
        interventions: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for attr in sensitive_attrs:
            scenarios = self._generate_interventions(data, attr)
            interventions[attr] = {scenario.scenario_id: scenario.to_dict() for scenario in scenarios}
            results[attr] = {}
            self._log_metric(
                metric=f"counterfactual_scenarios_{normalize_metric_name(attr)}",
                value=float(len(scenarios)),
                threshold=float(max(1, len(scenarios))),
                context=merge_mappings(audit_context, {"attribute": attr}),
                source="counterfactual_auditor",
                tags=["audit", "intervention", attr],
                metadata={"strategy": self.perturbation_strategy},
            )

            for scenario in scenarios:
                cf_data = self.causal_model.compute_counterfactual(intervention=scenario.intervention)
                cf_preds = self._get_predictions(cf_data)
                scenario_result: Dict[str, Any] = {
                    "scenario": scenario.to_dict(),
                    "counterfactual_predictions": cf_preds,
                    "counterfactual_data": cf_data if self.include_counterfactual_samples else None,
                    "prediction_summary": self._prediction_summary(cf_preds),
                }
                if self.include_counterfactual_samples:
                    scenario_result["counterfactual_sample"] = cf_data.head(self.counterfactual_sample_cap).to_dict(orient="records")
                results[attr][scenario.scenario_id] = scenario_result
        return results, interventions

    def _generate_interventions(self, data: pd.DataFrame, sensitive_attr: str) -> List[CounterfactualScenario]:
        values = data[sensitive_attr].dropna()
        if values.empty:
            raise DataValidationError(
                "Cannot generate counterfactual interventions for an attribute with no observed values.",
                context={"sensitive_attr": sensitive_attr},
            )

        strategy = self.perturbation_strategy
        scenarios: List[CounterfactualScenario] = []

        if strategy == "flip":
            unique_values = list(pd.unique(values))
            if len(unique_values) == 1:
                unique_values = [unique_values[0]]
            if len(unique_values) > self.max_scenarios_per_attribute:
                unique_values = unique_values[: self.max_scenarios_per_attribute]
            for idx, value in enumerate(unique_values):
                scenarios.append(
                    CounterfactualScenario(
                        sensitive_attribute=sensitive_attr,
                        scenario_id=f"{normalize_identifier_prefix(sensitive_attr)}_flip_{idx}",
                        intervention_value=value,
                        intervention={sensitive_attr: value},
                        strategy=strategy,
                        sample_size=len(data),
                    )
                )

        elif strategy == "fixed_delta":
            if not pd.api.types.is_numeric_dtype(values):
                raise DataValidationError(
                    "fixed_delta perturbation requires a numeric sensitive attribute.",
                    context={"sensitive_attr": sensitive_attr},
                )
            series = pd.to_numeric(values, errors="coerce").dropna()
            mean_val = float(series.mean())
            std_val = float(series.std(ddof=0))
            delta = self.perturbation_magnitude * (std_val if std_val > 0 else max(abs(mean_val), 1.0))
            candidate_values = [mean_val - delta, mean_val + delta]
            for idx, value in enumerate(candidate_values[: self.max_scenarios_per_attribute]):
                scenarios.append(
                    CounterfactualScenario(
                        sensitive_attribute=sensitive_attr,
                        scenario_id=f"{normalize_identifier_prefix(sensitive_attr)}_delta_{idx}",
                        intervention_value=float(value),
                        intervention={sensitive_attr: float(value)},
                        strategy=strategy,
                        sample_size=len(data),
                        metadata={"delta": float(delta)},
                    )
                )

        elif strategy == "sample_distribution":
            sample_count = min(self.num_counterfactual_samples, len(values), self.max_scenarios_per_attribute)
            sampled_values = self.random_state.choice(values.to_numpy(), size=sample_count, replace=False)
            for idx, value in enumerate(sampled_values):
                scenarios.append(
                    CounterfactualScenario(
                        sensitive_attribute=sensitive_attr,
                        scenario_id=f"{normalize_identifier_prefix(sensitive_attr)}_sample_{idx}",
                        intervention_value=value,
                        intervention={sensitive_attr: json_safe(value)},
                        strategy=strategy,
                        sample_size=len(data),
                    )
                )

        elif strategy == "quantile_grid":
            if not pd.api.types.is_numeric_dtype(values):
                raise DataValidationError(
                    "quantile_grid perturbation requires a numeric sensitive attribute.",
                    context={"sensitive_attr": sensitive_attr},
                )
            numeric_values = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
            quantile_values = [float(np.quantile(numeric_values, q)) for q in self.quantile_grid]
            for idx, value in enumerate(quantile_values[: self.max_scenarios_per_attribute]):
                scenarios.append(
                    CounterfactualScenario(
                        sensitive_attribute=sensitive_attr,
                        scenario_id=f"{normalize_identifier_prefix(sensitive_attr)}_quantile_{idx}",
                        intervention_value=value,
                        intervention={sensitive_attr: value},
                        strategy=strategy,
                        sample_size=len(data),
                        metadata={"quantile": self.quantile_grid[idx] if idx < len(self.quantile_grid) else None},
                    )
                )

        else:
            raise ConfigurationError(
                "Unsupported perturbation_strategy configured for counterfactual auditor.",
                context={"perturbation_strategy": strategy},
            )

        if not scenarios:
            raise CounterfactualAuditError(
                "No interventions could be generated for the requested sensitive attribute.",
                context={"sensitive_attr": sensitive_attr, "strategy": strategy},
            )
        return scenarios

    def sample_distribution(self, data: pd.DataFrame, sensitive_attr: str) -> Dict[str, Dict[str, Any]]:
        return {
            scenario.scenario_id: scenario.intervention
            for scenario in self._generate_interventions(data, sensitive_attr)
            if scenario.strategy == "sample_distribution"
        }

    def fixed_delta(self, data: pd.DataFrame, sensitive_attr: str) -> Dict[str, Dict[str, Any]]:
        return {
            scenario.scenario_id: scenario.intervention
            for scenario in self._generate_interventions(data, sensitive_attr)
            if scenario.strategy == "fixed_delta"
        }

    # ------------------------------------------------------------------
    # Fairness assessment
    # ------------------------------------------------------------------
    def _assess_fairness_violations(
        self,
        original_data: pd.DataFrame,
        original_preds: np.ndarray,
        cf_results: Dict[str, Dict[str, Dict[str, Any]]],
        sensitive_attrs: List[str],
        y_true_col: Optional[str],
        *,
        audit_context: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> Dict[str, Any]:
        fairness_report: Dict[str, Any] = {
            "individual_fairness": {},
            "group_disparity": {},
            "equalized_odds_gap": {},
            "path_specific_effects": {},
            "overall_violations": {},
        }

        for attr in sensitive_attrs:
            attr_results = cf_results.get(attr, {})
            if not attr_results:
                fairness_report["individual_fairness"][attr] = {"message": "No counterfactual scenarios available."}
                fairness_report["group_disparity"][attr] = {"message": "No counterfactual scenarios available."}
                fairness_report["equalized_odds_gap"][attr] = {"message": "No counterfactual scenarios available."}
                fairness_report["path_specific_effects"][attr] = {"message": "No counterfactual scenarios available."}
                fairness_report["overall_violations"][attr] = {}
                continue

            individual_metrics = self._assess_individual_fairness(
                original_preds=original_preds,
                scenario_results=attr_results,
                attr=attr,
                audit_context=audit_context,
                metadata=metadata,
            )
            fairness_report["individual_fairness"][attr] = individual_metrics

            if y_true_col is not None:
                group_metrics = self._assess_group_disparity(
                    original_data=original_data,
                    original_preds=original_preds,
                    scenario_results=attr_results,
                    attr=attr,
                    y_true_col=y_true_col,
                    audit_context=audit_context,
                    metadata=metadata,
                )
                eq_odds_metrics = self._assess_equalized_odds_gap(
                    original_data=original_data,
                    original_preds=original_preds,
                    scenario_results=attr_results,
                    attr=attr,
                    y_true_col=y_true_col,
                    audit_context=audit_context,
                    metadata=metadata,
                )
            else:
                group_metrics = {"message": "Missing y_true_col; group disparity assessment skipped."}
                eq_odds_metrics = {"message": "Missing y_true_col; equalized-odds assessment skipped."}

            path_metrics = self._assess_path_specific_effects(
                original_data=original_data,
                attr=attr,
                y_true_col=y_true_col,
                audit_context=audit_context,
                metadata=metadata,
            )

            fairness_report["group_disparity"][attr] = group_metrics
            fairness_report["equalized_odds_gap"][attr] = eq_odds_metrics
            fairness_report["path_specific_effects"][attr] = path_metrics
            fairness_report["overall_violations"][attr] = {
                "individual_mean_diff": bool(individual_metrics.get("violation", False)),
                "individual_max_diff": bool(individual_metrics.get("max_difference", 0.0) > self.fairness_thresholds["individual_fairness_max_diff"]),
                "group_stat_parity": bool(group_metrics.get("stat_parity_violation", False)) if isinstance(group_metrics, dict) else False,
                "group_equal_opp": bool(group_metrics.get("equal_opp_violation", False)) if isinstance(group_metrics, dict) else False,
                "group_avg_odds": bool(group_metrics.get("avg_odds_violation", False)) if isinstance(group_metrics, dict) else False,
                "equalized_odds_gap": bool(eq_odds_metrics.get("violation", False)) if isinstance(eq_odds_metrics, dict) else False,
                "causal_effect_ate": bool(path_metrics.get("violation", False)) if isinstance(path_metrics, dict) else False,
            }

        self._aggregate_violations(fairness_report)
        return fairness_report

    def _assess_individual_fairness(
        self,
        *,
        original_preds: np.ndarray,
        scenario_results: Mapping[str, Dict[str, Any]],
        attr: str,
        audit_context: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> Dict[str, Any]:
        scenario_metrics: Dict[str, Dict[str, Any]] = {}
        max_difference = 0.0
        mean_differences: List[float] = []
        wasserstein_values: List[float] = []

        for scenario_id, result in scenario_results.items():
            cf_preds = np.asarray(result["counterfactual_predictions"], dtype=float).reshape(-1)
            metrics = self.fairness_assessor.compute_individual_fairness(
                original_preds,
                cf_preds,
                threshold=self.fairness_thresholds["individual_fairness_mean_diff"],
                metadata=metadata,
                context=merge_mappings(audit_context, {"attribute": attr, "scenario_id": scenario_id}),
            )
            scenario_metrics[scenario_id] = metrics
            max_difference = max(max_difference, float(metrics.get("max_difference", 0.0)))
            mean_differences.append(float(metrics.get("mean_difference", 0.0)))
            wasserstein_values.append(float(metrics.get("wasserstein_distance", 0.0)))

        aggregate = {
            "mean_difference": float(np.mean(mean_differences)) if mean_differences else 0.0,
            "max_difference": float(max_difference),
            "mean_wasserstein_distance": float(np.mean(wasserstein_values)) if wasserstein_values else 0.0,
            "scenario_count": len(scenario_metrics),
            "by_scenario": scenario_metrics,
        }
        aggregate["violation"] = bool(
            aggregate["mean_difference"] > self.fairness_thresholds["individual_fairness_mean_diff"]
            or aggregate["max_difference"] > self.fairness_thresholds["individual_fairness_max_diff"]
        )
        return aggregate

    def _assess_group_disparity(
        self,
        *,
        original_data: pd.DataFrame,
        original_preds: np.ndarray,
        scenario_results: Mapping[str, Dict[str, Any]],
        attr: str,
        y_true_col: str,
        audit_context: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> Dict[str, Any]:
        groups = list(pd.unique(original_data[attr].dropna()))
        if len(groups) < 2:
            return {"message": "At least two observed groups are required for group disparity analysis."}

        reference_groups = self._resolve_reference_groups(groups)
        original_with_preds = original_data.assign(_pred=original_preds)
        original_group = self.fairness_assessor.compute_group_disparity(
            original_with_preds,
            sensitive_attr=attr,
            predictions="_pred",
            y_true=y_true_col,
            privileged_group=reference_groups["privileged_group"],
            unprivileged_group=reference_groups["unprivileged_group"],
            prediction_threshold=self.default_prediction_threshold,
            metadata=metadata,
            context=merge_mappings(audit_context, {"attribute": attr, "scenario": "original"}),
        )

        by_scenario: Dict[str, Any] = {}
        max_stat_parity = abs(float(original_group.get("statistical_parity_difference", 0.0)))
        max_equal_opp = abs(float(original_group.get("equal_opportunity_difference", 0.0)))
        max_avg_odds = abs(float(original_group.get("average_abs_odds_difference", 0.0)))

        for scenario_id, result in scenario_results.items():
            cf_data = result.get("counterfactual_data")
            if not isinstance(cf_data, pd.DataFrame):
                cf_data = self.causal_model.compute_counterfactual(intervention=result["scenario"]["intervention"])

            # Check if counterfactual data has at least two distinct groups
            unique_groups = cf_data[attr].dropna().unique()
            if len(unique_groups) < 2:
                logger.warning(
                    "Skipping group disparity for scenario '%s' because counterfactual data has only %d unique group(s) for attribute '%s'.",
                    scenario_id,
                    len(unique_groups),
                    attr,
                )
                by_scenario[scenario_id] = {
                    "error": "Insufficient groups in counterfactual data",
                    "unique_groups": list(map(str, unique_groups)),
                }
                continue

            try:
                cf_group = self.fairness_assessor.compute_group_disparity(
                    cf_data.assign(_pred=result["counterfactual_predictions"]),
                    sensitive_attr=attr,
                    predictions="_pred",
                    y_true=y_true_col,
                    privileged_group=reference_groups["privileged_group"],
                    unprivileged_group=reference_groups["unprivileged_group"],
                    prediction_threshold=self.default_prediction_threshold,
                    metadata=metadata,
                    context=merge_mappings(audit_context, {"attribute": attr, "scenario_id": scenario_id}),
                )
            except FairnessEvaluationError as exc:
                if "At least two sensitive groups" in str(exc):
                    logger.warning(
                        "Group disparity skipped for scenario '%s' due to insufficient groups: %s",
                        scenario_id,
                        exc,
                    )
                    by_scenario[scenario_id] = {
                        "error": "Insufficient groups for disparity calculation",
                        "details": str(exc),
                    }
                    continue
                raise

            by_scenario[scenario_id] = cf_group
            max_stat_parity = max(max_stat_parity, abs(float(cf_group.get("statistical_parity_difference", 0.0))))
            max_equal_opp = max(max_equal_opp, abs(float(cf_group.get("equal_opportunity_difference", 0.0))))
            max_avg_odds = max(max_avg_odds, abs(float(cf_group.get("average_abs_odds_difference", 0.0))))

        return {
            "original": original_group,
            "by_scenario": by_scenario,
            "max_stat_parity_diff_observed": float(max_stat_parity),
            "max_eod_diff_observed": float(max_equal_opp),
            "max_aaod_diff_observed": float(max_avg_odds),
            "stat_parity_violation": bool(max_stat_parity > self.fairness_thresholds["group_disparity_stat_parity"]),
            "equal_opp_violation": bool(max_equal_opp > self.fairness_thresholds["group_disparity_equal_opp"]),
            "avg_odds_violation": bool(max_avg_odds > self.fairness_thresholds["group_disparity_avg_odds"]),
        }

    def _assess_equalized_odds_gap(
        self,
        *,
        original_data: pd.DataFrame,
        original_preds: np.ndarray,
        scenario_results: Mapping[str, Dict[str, Any]],
        attr: str,
        y_true_col: str,
        audit_context: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> Dict[str, Any]:
        by_scenario: Dict[str, Any] = {}
        max_gap_change = 0.0
        for scenario_id, result in scenario_results.items():
            frame = original_data.copy()
            frame["_original_pred"] = original_preds
            frame["_counterfactual_pred"] = result["counterfactual_predictions"]
            metrics = self.fairness_assessor.compute_equalized_odds_gap(
                frame,
                sensitive_attr=attr,
                original_preds_col="_original_pred",
                counterfactual_preds_col="_counterfactual_pred",
                y_true_col=y_true_col,
                prediction_threshold=self.default_prediction_threshold,
                metadata=metadata,
                context=merge_mappings(audit_context, {"attribute": attr, "scenario_id": scenario_id}),
            )
            by_scenario[scenario_id] = metrics
            max_gap_change = max(
                max_gap_change,
                float(max(metrics.get("tpr_gap_change", 0.0), metrics.get("fpr_gap_change", 0.0))),
            )
        return {
            "by_scenario": by_scenario,
            "max_gap_change": float(max_gap_change),
            "threshold": self.fairness_thresholds["equalized_odds_gap_change"],
            "violation": bool(max_gap_change > self.fairness_thresholds["equalized_odds_gap_change"]),
        }

    def _assess_path_specific_effects(
        self,
        *,
        original_data: pd.DataFrame,
        attr: str,
        y_true_col: Optional[str],
        audit_context: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if y_true_col is None:
            return {"message": "Missing y_true_col; path-specific effects skipped."}

        mediators = self.attribute_mediators.get(attr)
        if mediators is None and self.causal_model is not None:
            mediators = [node for node in self.causal_model.graph.successors(attr) if node != y_true_col]
        mediators = [m for m in (mediators or []) if m in original_data.columns and m != y_true_col]
        if not mediators:
            return {"message": "No mediators available for path-specific effect estimation."}

        metrics = self.fairness_assessor.compute_path_specific_effect(
            original_data,
            treatment=attr,
            outcome=y_true_col,
            mediators=mediators,
            metadata=metadata,
            context=merge_mappings(audit_context, {"attribute": attr}),
        )
        metrics["violation"] = bool(abs(float(metrics.get("total_effect", 0.0))) > self.fairness_thresholds["causal_effect_ate"])
        return metrics

    def _aggregate_violations(self, fairness_report: Dict[str, Any]) -> None:
        summary: Dict[str, bool] = {}
        for attr, violations in fairness_report["overall_violations"].items():
            if not isinstance(violations, Mapping):
                continue
            for key, value in violations.items():
                summary[key] = bool(summary.get(key, False) or bool(value))
        fairness_report["overall_violations"]["summary"] = summary

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------
    def _analyze_decision_sensitivity(
        self,
        *,
        original_preds: np.ndarray,
        cf_results: Dict[str, Dict[str, Dict[str, Any]]],
        metadata: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        sensitivity_scores: Dict[str, Any] = {}

        for attr, scenario_results in cf_results.items():
            attr_scores: Dict[str, Any] = {}
            max_abs_shift = 0.0
            max_effect_size = 0.0
            for scenario_id, result in scenario_results.items():
                cf_preds = np.asarray(result["counterfactual_predictions"], dtype=float).reshape(-1)
                stat, p_value = self._welch_t_test(original_preds, cf_preds)
                effect_size = self._compute_cohens_d(original_preds, cf_preds)
                mean_shift = float(np.nanmean(cf_preds) - np.nanmean(original_preds))
                shift_distance = float(wasserstein_distance(original_preds, cf_preds)) if len(cf_preds) else 0.0
                max_abs_shift = max(max_abs_shift, abs(mean_shift))
                max_effect_size = max(max_effect_size, abs(effect_size))
                attr_scores[scenario_id] = {
                    "mean_shift": mean_shift,
                    "absolute_mean_shift": abs(mean_shift),
                    "cohens_d": float(effect_size),
                    "t_statistic": float(stat) if not np.isnan(stat) else np.nan,
                    "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
                    "wasserstein_distance": shift_distance,
                    "is_sensitive": bool(p_value < self.sensitivity_alpha) if not np.isnan(p_value) else False,
                }
            sensitivity_scores[attr] = {
                "by_scenario": attr_scores,
                "max_absolute_mean_shift": float(max_abs_shift),
                "max_absolute_cohens_d": float(max_effect_size),
                "violation": bool(
                    max_abs_shift > self.fairness_thresholds["sensitivity_mean_shift"]
                    or max_effect_size > self.fairness_thresholds["sensitivity_cohens_d"]
                ),
            }
            self._log_metric(
                metric=f"counterfactual_sensitivity_{normalize_metric_name(attr)}",
                value=float(max_abs_shift),
                threshold=self.fairness_thresholds["sensitivity_mean_shift"],
                context=merge_mappings(context, {"attribute": attr}),
                source="counterfactual_auditor",
                tags=["audit", "sensitivity", attr],
                metadata=merge_mappings(metadata, {"max_absolute_cohens_d": max_effect_size}),
            )
        return sensitivity_scores

    @staticmethod
    def _welch_t_test(original: np.ndarray, counterfactual: np.ndarray) -> Tuple[float, float]:
        original_valid = np.asarray(original, dtype=float)
        counterfactual_valid = np.asarray(counterfactual, dtype=float)
        if len(original_valid) < 2 or len(counterfactual_valid) < 2:
            return np.nan, np.nan
        if np.nanvar(original_valid) < 1e-12 and np.nanvar(counterfactual_valid) < 1e-12:
            return np.nan, np.nan
        stat, p_value = ttest_ind(original_valid, counterfactual_valid, equal_var=False, nan_policy="omit")
        return float(stat), float(p_value)

    @staticmethod
    def _compute_cohens_d(original: np.ndarray, counterfactual: np.ndarray) -> float:
        original_valid = np.asarray(original, dtype=float)
        counterfactual_valid = np.asarray(counterfactual, dtype=float)
        if len(original_valid) < 2 or len(counterfactual_valid) < 2:
            return 0.0
        diff = float(np.nanmean(original_valid) - np.nanmean(counterfactual_valid))
        n1 = len(original_valid)
        n2 = len(counterfactual_valid)
        var1 = float(np.nanvar(original_valid, ddof=1))
        var2 = float(np.nanvar(counterfactual_valid, ddof=1))
        pooled_std = math.sqrt(max((((n1 - 1) * var1) + ((n2 - 1) * var2)) / max(n1 + n2 - 2, 1), 0.0))
        return abs(diff / pooled_std) if pooled_std > 1e-12 else 0.0

    # ------------------------------------------------------------------
    # Causal effects, scoring, and summaries
    # ------------------------------------------------------------------
    def _estimate_sensitive_attribute_effects(
        self,
        data: pd.DataFrame,
        sensitive_attrs: Sequence[str],
        y_true_col: Optional[str],
        *,
        audit_context: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if y_true_col is None or self.causal_model is None:
            return {}
        estimates: Dict[str, Any] = {}
        for attr in sensitive_attrs:
            try:
                estimate = self.causal_model.estimate_effect(
                    treatment=attr,
                    outcome=y_true_col,
                    data=data,
                    method=getattr(self.causal_model, "default_effect_method", None) or self.config.get("causal_model", {}).get("default_effect_method", "backdoor.linear_regression"),
                )
                if hasattr(estimate, "to_dict"):
                    estimate_dict = estimate.to_dict()
                elif isinstance(estimate, pd.Series):
                    estimate_dict = estimate.to_dict()
                else:
                    estimate_dict = {"effect": float(estimate)} if estimate is not None else {}
                estimates[attr] = estimate_dict
                effect_value = float(estimate_dict.get("effect", 0.0)) if estimate_dict else 0.0
                self._log_metric(
                    metric=f"counterfactual_ate_{normalize_metric_name(attr)}",
                    value=abs(effect_value),
                    threshold=self.fairness_thresholds["causal_effect_ate"],
                    context=merge_mappings(audit_context, {"attribute": attr}),
                    source="counterfactual_auditor",
                    tags=["audit", "causal_effect", attr],
                    metadata=metadata,
                )
            except Exception as exc:
                estimates[attr] = {
                    "error": str(wrap_alignment_exception(
                        exc,
                        target_cls=CounterfactualAuditError,
                        message="Failed to estimate sensitive-attribute causal effect.",
                        context={"attribute": attr, "outcome": y_true_col},
                    ))
                }
        return estimates

    def _recommend_correction_level(self, fairness_report: Mapping[str, Any], sensitivity_report: Mapping[str, Any]) -> Dict[str, Any]:
        overall_bias = self._calculate_overall_bias(fairness_report)
        any_sensitivity_violation = any(
            bool(attr_report.get("violation", False))
            for attr_report in sensitivity_report.values()
            if isinstance(attr_report, Mapping)
        )
        action = "alert_only"
        triggered_threshold = None
        for level in sorted(self.correction_levels, key=lambda item: float(item.get("threshold", 0.0)), reverse=True):
            threshold = coerce_float(level.get("threshold", 0.0), field_name="correction_threshold", minimum=0.0)
            if overall_bias >= threshold or (any_sensitivity_violation and threshold <= 0.5):
                action = str(level.get("action", "alert_only"))
                triggered_threshold = threshold
                break
        return {
            "action": action,
            "triggered_threshold": triggered_threshold,
            "overall_bias": overall_bias,
            "sensitivity_violation": any_sensitivity_violation,
        }

    def _calculate_overall_bias(self, report: Mapping[str, Any]) -> float:
        total = 0.0
        count = 0
        for attr_report in report.get("individual_fairness", {}).values():
            if not isinstance(attr_report, Mapping):
                continue
            total += float(attr_report.get("mean_difference", 0.0))
            count += 1
        for attr_report in report.get("group_disparity", {}).values():
            if not isinstance(attr_report, Mapping):
                continue
            total += float(attr_report.get("max_stat_parity_diff_observed", 0.0))
            count += 1
        return float(total / count) if count else 0.0

    def _count_violations(self, report: Mapping[str, Any]) -> int:
        summary = report.get("overall_violations", {}).get("summary", {})
        if not isinstance(summary, Mapping):
            return 0
        return int(sum(bool(value) for value in summary.values()))

    def _detect_drift(self, context: Mapping[str, Any]) -> Dict[str, Any]:
        try:
            detected = self.alignment_memory.detect_drift(return_details=True)
            if isinstance(detected, dict):
                return detected
            return {"detected": bool(detected)}
        except Exception as exc:
            if self.strict_memory_integration:
                raise
            logger.warning("Drift detection failed during counterfactual audit: %s", exc)
            return {"detected": False, "status": "memory_unavailable"}

    def _prediction_summary(self, predictions: np.ndarray) -> Dict[str, Any]:
        preds = np.asarray(predictions, dtype=float).reshape(-1)
        summary = {
            "count": int(preds.size),
            "mean": float(np.mean(preds)) if preds.size else 0.0,
            "std": float(np.std(preds)) if preds.size else 0.0,
            "min": float(np.min(preds)) if preds.size else 0.0,
            "max": float(np.max(preds)) if preds.size else 0.0,
            "positive_rate": float(np.mean(preds >= self.default_prediction_threshold)) if preds.size else 0.0,
        }
        if self.report_distribution_samples and preds.size:
            summary["sample"] = preds[: self.counterfactual_sample_cap].tolist()
        return summary

    def _resolve_reference_groups(self, groups: Sequence[Any]) -> Dict[str, Any]:
        unique_groups = list(groups)
        if len(unique_groups) < 2:
            raise FairnessEvaluationError(
                "At least two groups are required for disparity analysis.",
                context={"groups": unique_groups},
            )
        if len(unique_groups) == 2:
            ordered = sorted(unique_groups, key=lambda value: str(value))
            return {"unprivileged_group": ordered[0], "privileged_group": ordered[1]}
        ordered = sorted(unique_groups, key=lambda value: str(value))
        return {"unprivileged_group": ordered[0], "privileged_group": ordered[-1]}

    def _causal_graph_info(self) -> Dict[str, Any]:
        if self.causal_model is None:
            return {"status": "unavailable"}
        graph = self.causal_model.graph
        return {
            "nodes": list(graph.nodes()),
            "edges": list(graph.edges()),
            "n_nodes": int(graph.number_of_nodes()),
            "n_edges": int(graph.number_of_edges()),
            "is_dag": bool(nx.is_directed_acyclic_graph(graph)),
            "potential_latent_confounders": list(getattr(graph, "potential_latent_confounders", set())),
        }

    # ------------------------------------------------------------------
    # Memory integration
    # ------------------------------------------------------------------
    def _log_metric(
        self,
        *,
        metric: str,
        value: float,
        threshold: float,
        context: Mapping[str, Any],
        source: str,
        tags: Optional[Iterable[Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not self.enable_memory_logging:
            return
        try:
            self.alignment_memory.log_evaluation(
                metric=metric,
                value=float(value),
                threshold=float(threshold),
                context=dict(normalize_context(context, drop_none=False)),
                source=source,
                tags=list(normalize_tags(tags)),
                metadata=dict(normalize_metadata(metadata)),
            )
        except Exception as exc:
            if self.strict_memory_integration:
                raise wrap_alignment_exception(
                    exc,
                    target_cls=AlignmentMemoryError,
                    message="Failed to log counterfactual audit metric to alignment memory.",
                    context={"metric": metric, "source": source},
                ) from exc
            logger.warning("Counterfactual audit memory logging failed for metric '%s': %s", metric, exc)

    def _record_outcome(
        self,
        *,
        context: Mapping[str, Any],
        outcome: Mapping[str, Any],
        source: str,
        tags: Optional[Iterable[Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not self.enable_memory_logging:
            return
        try:
            self.alignment_memory.record_outcome(
                context=dict(normalize_context(context, drop_none=False)),
                outcome=dict(normalize_metadata(outcome, drop_none=False)),
                source=source,
                tags=list(normalize_tags(tags)),
                metadata=dict(normalize_metadata(metadata)),
            )
        except Exception as exc:
            if self.strict_memory_integration:
                raise wrap_alignment_exception(
                    exc,
                    target_cls=AlignmentMemoryError,
                    message="Failed to record counterfactual audit outcome in alignment memory.",
                    context={"source": source},
                ) from exc
            logger.warning("Counterfactual audit outcome recording failed: %s", exc)


if __name__ == '__main__':
    print("\n=== Running Counterfactual Auditor ===\n")
    printer.status("Init", "Counterfactual Auditor initialized", "success")

    np.random.seed(42)
    n_samples = 1500
    data = pd.DataFrame({
        'age': np.random.normal(35, 10, n_samples).clip(18, 80),
        'gender': np.random.choice([0, 1], n_samples, p=[0.45, 0.55]),
        'education': np.random.choice([0, 1, 2], n_samples, p=[0.2, 0.5, 0.3]),
        'income': np.random.lognormal(3.6, 0.35, n_samples),
    })

    data['loan_approval'] = np.where(
        (data['income'] > np.quantile(data['income'], 0.55)) & (data['education'] > 0),
        1,
        np.where(data['gender'] == 1, 1, 0),
    ).astype(int)

    def ml_predict_func(df: pd.DataFrame) -> np.ndarray:
        logits = (
            0.75 * (df['income'] / float(df['income'].mean()))
            + 0.55 * df['education']
            - 0.20 * (df['age'] / 100.0)
            - 0.25 * df['gender']
        )
        return 1.0 / (1.0 + np.exp(-logits.to_numpy(dtype=float)))

    builder = CausalGraphBuilder()
    builder.required_edges = [('gender', 'loan_approval'), ('education', 'loan_approval'), ('income', 'loan_approval')]
    causal_model = builder.construct_graph(data, sensitive_attrs=['gender'])

    auditor = CounterfactualAuditor(causal_model=causal_model, model_predict_func=ml_predict_func)

    report = auditor.audit(
        data=data,
        sensitive_attrs=['gender'],
        y_true_col='loan_approval',
        metadata={'test_mode': True},
        context={'domain': 'lending', 'task': 'loan_risk_screening'},
    )

    print("\n=== Audit Summary ===")
    print("Audit ID:", report['audit_id'])
    print("Overall Bias:", report['overall_bias'])
    print("Violation Count:", report['violation_count'])
    print("Recommended Action:", report['recommended_action'])
    print("\nIndividual Fairness:")
    print(report['fairness_metrics']['individual_fairness'])
    print("\nGroup Disparity:")
    print(report['fairness_metrics']['group_disparity'])
    print("\nSensitivity Analysis:")
    print(report['sensitivity_analysis'])
    print("\n=== Test ran successfully ===\n")
