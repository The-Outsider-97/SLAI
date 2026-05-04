"""
Formal Bias Detection Framework
Implements intersectional bias analysis and statistical fairness verification from:
- Mitchell et al. (2019) "Model Cards for Model Reporting"
- Barocas & Hardt (2018) "Fairness and Machine Learning"
"""

from __future__ import annotations

import datetime
import math
import random
import numpy as np
import pandas as pd

from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from itertools import combinations
from scipy import stats
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.multitest import multipletests
from statsmodels.multivariate.manova import MANOVA
from ruptures import Binseg

from .utils import *
from .alignment_memory import AlignmentMemory
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Bias Detection")
printer = PrettyPrinter


class BiasDetector:
    """
    Advanced bias detection system implementing:
    - Intersectional fairness analysis
    - Statistical hypothesis testing
    - Longitudinal bias tracking
    - Causal disparity detection

    Supported fairness notions:
    - Demographic Parity (Dwork et al., 2012)
    - Equalized Odds / Equal Opportunity proxy (Hardt et al., 2016)
    - Counterfactual-adjacent disparity telemetry hooks for downstream auditors
    - Sufficiency / Predictive Parity (Barocas et al., 2019)

    Design intent:
    The detector is not a replacement for the fairness evaluator or
    counterfactual auditor. It focuses on robust group-level statistical bias
    assessment, especially for intersectional cohorts, and emits structured
    telemetry suitable for AlignmentMemory and higher-level monitoring.
    """

    HISTORY_COLUMNS = [
        "timestamp",
        "run_id",
        "audit_id",
        "metric",
        "group_id",
        "group_family",
        "group_depth",
        "group_size",
        "eligible_rows",
        "reference_size",
        "metric_value",
        "reference_value",
        "disparity",
        "abs_disparity",
        "ratio_to_reference",
        "ci_lower",
        "ci_upper",
        "p_value",
        "adj_p_value",
        "stat_significance",
        "threshold",
        "violation",
        "group_definition",
        "source",
        "context_hash",
        "tags",
        "metadata",
    ]

    METRIC_ALIASES: Dict[str, str] = {
        "demographic_parity": "demographic_parity",
        "statistical_parity": "demographic_parity",
        "selection_rate_parity": "demographic_parity",
        "equal_opportunity": "equal_opportunity",
        "true_positive_rate_parity": "equal_opportunity",
        "tpr_parity": "equal_opportunity",
        "predictive_parity": "predictive_parity",
        "positive_predictive_value_parity": "predictive_parity",
        "ppv_parity": "predictive_parity",
        "disparate_impact": "disparate_impact",
        "four_fifths_rule": "disparate_impact",
    }

    DEFAULT_THRESHOLDS: Dict[str, float] = {
        "demographic_parity": 0.10,
        "equal_opportunity": 0.10,
        "predictive_parity": 0.10,
        "disparate_impact": 0.80,
    }

    SUMMARY_ALIASES: Dict[str, Tuple[str, ...]] = {
        "demographic_parity": ("demographic_parity", "statistical_parity"),
        "equal_opportunity": ("equal_opportunity",),
        "predictive_parity": ("predictive_parity",),
        "disparate_impact": ("disparate_impact",),
    }

    def __init__(
        self,
        sensitive_attributes: Optional[Sequence[str]] = None,
        alignment_memory: Optional[AlignmentMemory] = None,
        config_section_name: str = "bias_detection",
        config_file_path: Optional[str] = None,
    ):
        self.config = load_global_config()
        self.config_file_path = config_file_path
        self.config_section_name = ensure_non_empty_string(
            config_section_name,
            "config_section_name",
            error_cls=ConfigurationError,
        )
        self.detector_config = self._resolve_config_section(self.config_section_name)
        self._validate_detector_config()

        configured_sensitive = sensitive_attributes or self.detector_config.get("sensitive_attributes_override") or self.config.get("sensitive_attributes")
        self.sensitive_attributes = list(normalize_sensitive_attributes(configured_sensitive, allow_empty=False))
        self.sensitive_attrs = self.sensitive_attributes

        self.metrics = self._resolve_metric_list(self.detector_config.get("metrics"))
        self.alpha = coerce_probability(self.detector_config.get("alpha", 0.05), field_name="alpha")
        self.bootstrap_samples = coerce_int(
            self.detector_config.get("bootstrap_samples", 1000),
            field_name="bootstrap_samples",
            minimum=100,
        )
        self.permutation_samples = coerce_int(
            self.detector_config.get("permutation_samples", 500),
            field_name="permutation_samples",
            minimum=50,
        )
        self.prediction_threshold = coerce_probability(
            self.detector_config.get("prediction_threshold", 0.5),
            field_name="prediction_threshold",
        )
        self.min_group_size = coerce_int(
            self.detector_config.get("min_group_size", 30),
            field_name="min_group_size",
            minimum=1,
        )
        self.min_reference_size = coerce_int(
            self.detector_config.get("min_reference_size", self.min_group_size),
            field_name="min_reference_size",
            minimum=1,
        )
        self.intersectional_depth = coerce_int(
            self.detector_config.get("intersectional_depth", max(1, len(self.sensitive_attributes))),
            field_name="intersectional_depth",
            minimum=1,
        )
        self.max_groups_per_metric = coerce_int(
            self.detector_config.get("max_groups_per_metric", 500),
            field_name="max_groups_per_metric",
            minimum=1,
        )
        self.history_retention_days = coerce_int(
            self.detector_config.get("history_retention_days", 365),
            field_name="history_retention_days",
            minimum=1,
        )
        self.max_history_rows = coerce_int(
            self.detector_config.get("max_history_rows", 100000),
            field_name="max_history_rows",
            minimum=100,
        )
        self.changepoint_penalty = coerce_float(
            self.detector_config.get("changepoint_penalty", 1.0),
            field_name="changepoint_penalty",
            minimum=0.0,
        )
        self.changepoint_min_points = coerce_int(
            self.detector_config.get("changepoint_min_points", 12),
            field_name="changepoint_min_points",
            minimum=3,
        )
        self.seasonality_period = coerce_int(
            self.detector_config.get("seasonality_period", 7),
            field_name="seasonality_period",
            minimum=2,
        )
        self.seasonality_min_points = coerce_int(
            self.detector_config.get("seasonality_min_points", 28),
            field_name="seasonality_min_points",
            minimum=4,
        )
        self.enable_memory_logging = bool(self.detector_config.get("enable_memory_logging", True))
        self.enable_history_retention = bool(self.detector_config.get("enable_history_retention", True))
        self.random_seed = int(self.detector_config.get("random_seed", 42))
        self.metric_thresholds = self._resolve_thresholds(self.detector_config.get("metric_thresholds"))

        self.bias_history = pd.DataFrame(columns=self.HISTORY_COLUMNS)
        self.last_report: Dict[str, Any] = {}
        self.last_run_id: Optional[str] = None
        self._rng = np.random.default_rng(self.random_seed)
        random.seed(self.random_seed)

        self.alignment_memory = alignment_memory or AlignmentMemory()
        self._validate_alignment_memory_contract()

        if config_file_path:
            logger.debug(
                "BiasDetector received config_file_path=%s and retained global config loader handling.",
                config_file_path,
            )

        printer.status(
            "INIT",
            f"Bias Detection initialized | metrics={self.metrics} sensitive_attrs={self.sensitive_attrs}",
            "success",
        )

    # ------------------------------------------------------------------
    # Configuration and initialization
    # ------------------------------------------------------------------
    def _resolve_config_section(self, section_name: str) -> Dict[str, Any]:
        config = get_config_section(section_name)
        if config:
            return dict(config)

        # Backward compatibility with older naming used elsewhere in the stack.
        legacy_name = "bias_detector"
        if section_name != legacy_name:
            legacy = get_config_section(legacy_name)
            if legacy:
                logger.warning(
                    "BiasDetector config section '%s' not found. Falling back to legacy section '%s'.",
                    section_name,
                    legacy_name,
                )
                return dict(legacy)
        return {}

    def _validate_detector_config(self) -> None:
        try:
            ensure_mapping(
                self.detector_config,
                field_name=self.config_section_name,
                allow_empty=False,
                error_cls=ConfigurationError,
            )
            ensure_keys_present = ["metrics", "alpha", "bootstrap_samples", "min_group_size", "intersectional_depth", "metric_thresholds"]
            missing = [key for key in ensure_keys_present if key not in self.detector_config]
            if missing:
                raise ConfigurationError(
                    message=f"BiasDetector configuration is missing required keys: {missing}.",
                    context={"config_section": self.config_section_name, "missing_keys": missing},
                )
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=ConfigurationError,
                message="BiasDetector configuration validation failed.",
                context={
                    "config_section": self.config_section_name,
                    "config_path": self.config.get("__config_path__"),
                },
            ) from exc

    def _validate_alignment_memory_contract(self) -> None:
        required_methods = ["log_evaluation", "record_outcome", "analyze_causes", "get_memory_report"]
        missing = [name for name in required_methods if not hasattr(self.alignment_memory, name)]
        if missing:
            raise ConfigurationError(
                message="AlignmentMemory integration contract is incomplete for BiasDetector.",
                context={"missing_methods": missing, "memory_type": type(self.alignment_memory).__name__},
            )

    def _resolve_metric_list(self, metrics: Any) -> List[str]:
        values = ensure_sequence(metrics, "metrics", allow_empty=False, error_cls=ConfigurationError, allow_strings=False)
        resolved: List[str] = []
        for metric in values:
            canonical = self._canonical_metric_name(metric)
            if canonical not in resolved:
                resolved.append(canonical)
        if not resolved:
            raise ConfigurationError("At least one supported metric must be configured for BiasDetector.")
        return resolved

    def _resolve_thresholds(self, thresholds: Any) -> Dict[str, float]:
        configured = normalize_threshold_mapping(thresholds, minimum=0.0)
        resolved = dict(self.DEFAULT_THRESHOLDS)
        for key, value in configured.items():
            canonical = self._canonical_metric_name(key)
            resolved[canonical] = float(value)
        return resolved

    def _canonical_metric_name(self, metric: Any) -> str:
        normalized = normalize_metric_name(metric, namespace=None)
        if normalized not in self.METRIC_ALIASES:
            raise ConfigurationError(
                message="Unsupported bias metric configured for BiasDetector.",
                context={"metric": normalized, "supported_metrics": sorted(self.METRIC_ALIASES.keys())},
            )
        return self.METRIC_ALIASES[normalized]

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------
    def compute_metrics(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        labels: Optional[np.ndarray] = None,
        *,
        sensitive_attributes: Optional[Sequence[str]] = None,
        context: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        audit_id: Optional[str] = None,
        source: str = "bias_detection",
        tags: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Compute comprehensive bias metrics with:
        - Intersectional group analysis
        - Bootstrap confidence intervals
        - Permutation-based significance estimation
        - Longitudinal telemetry for AlignmentMemory
        """
        try:
            frame, y_pred_raw, y_pred_binary, y_true, resolved_sensitive = self._validate_inputs(
                data=data,
                predictions=predictions,
                labels=labels,
                sensitive_attributes=sensitive_attributes,
            )
            report_context = normalize_context(context, drop_none=True)
            report_metadata = ensure_mapping(metadata, field_name="metadata", allow_empty=True, error_cls=ValidationError)
            normalized_tags = list(normalize_tags(tags))
            source_name = ensure_non_empty_string(source, "source", error_cls=ValidationError)
            run_id = audit_id or self._generate_run_id()
            timestamp = datetime.datetime.utcnow().isoformat() + "Z"
            self.last_run_id = run_id

            metrics_report: Dict[str, Any] = {}
            fairness_metrics_summary: Dict[str, Any] = {}
            violations: List[Dict[str, Any]] = []
            history_rows: List[Dict[str, Any]] = []

            for metric_name in self.metrics:
                metric_frame = self._prepare_metric_frame(
                    metric_name=metric_name,
                    data=frame,
                    y_pred_raw=y_pred_raw,
                    y_pred_binary=y_pred_binary,
                    y_true=y_true,
                )
                metric_report = self._compute_metric_report(
                    metric_name=metric_name,
                    metric_frame=metric_frame,
                    source=source_name,
                    audit_id=run_id,
                    context=report_context,
                    tags=normalized_tags,
                    metadata=report_metadata,
                )
                metrics_report[metric_name] = metric_report
                history_rows.extend(metric_report.get("history_rows", []))

                summary_payload = {
                    key: value
                    for key, value in metric_report.items()
                    if key not in {"groups", "history_rows"}
                }
                for alias in self.SUMMARY_ALIASES.get(metric_name, (metric_name,)):
                    fairness_metrics_summary[alias] = dict(summary_payload)

                if metric_report.get("violation"):
                    violations.append(
                        {
                            "metric": metric_name,
                            "value": metric_report.get("summary_value"),
                            "threshold": metric_report.get("threshold"),
                            "worst_group": metric_report.get("worst_group"),
                        }
                    )

            self._update_history(history_rows)
            if self.enable_history_retention:
                self._enforce_history_retention()

            report = {
                "report_metadata": {
                    "audit_id": run_id,
                    "generated_at": timestamp,
                    "source": source_name,
                    "sensitive_attributes": list(resolved_sensitive),
                    "metrics": list(self.metrics),
                    "prediction_threshold": self.prediction_threshold,
                    "context_hash": stable_context_hash(report_context),
                    "tags": normalized_tags,
                    "metadata": json_safe(report_metadata),
                },
                "dataset_summary": self._dataset_summary(frame, y_pred_raw, y_pred_binary, y_true, resolved_sensitive),
                "metrics": metrics_report,
                "fairness_metrics": fairness_metrics_summary,
                "violations": violations,
                "history_size": int(len(self.bias_history)),
                "event": build_alignment_event(
                    "bias_metrics_computed",
                    severity="medium" if not violations else "high",
                    risk_level="medium" if not violations else "high",
                    source=source_name,
                    tags=normalized_tags,
                    metadata=report_metadata,
                    context=report_context,
                    payload={
                        "audit_id": run_id,
                        "metrics": fairness_metrics_summary,
                        "violation_count": len(violations),
                    },
                ),
            }

            if self.enable_memory_logging:
                self._log_to_alignment_memory(report, report_context, source_name)

            self.last_report = report
            return report
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=BiasDetectionError,
                message="BiasDetector failed to compute bias metrics.",
                context={"source": source, "audit_id": audit_id},
            ) from exc

    def generate_report(self, format: str = "structured") -> Any:
        """Generate a comprehensive detector-level report from accumulated history."""
        try:
            output_format = ensure_non_empty_string(format, "format", error_cls=ValidationError).lower()
            if self.bias_history.empty:
                raise BiasDetectionError(
                    message="No bias history is available. Run compute_metrics() before generating a report.",
                    severity="medium",
                )

            report = {
                "current_state": self._current_state_report(),
                "historical_trends": self._analyze_trends(),
                "statistical_insights": self._compute_aggregate_stats(),
                "history_summary": self._history_summary(),
                "memory_analysis": self._safe_memory_call("get_memory_report", default={}),
                "causal_impact": self._safe_memory_call("analyze_causes", default={}),
                "last_report_metadata": json_safe(self.last_report.get("report_metadata", {})),
            }

            if output_format == "json":
                from .utils.alignment_helpers import stable_json_dumps
                return stable_json_dumps(report)
            if output_format not in {"structured", "dict"}:
                raise ValidationError(
                    message="Unsupported report format requested for BiasDetector.",
                    context={"format": output_format},
                )
            return report
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=BiasDetectionError,
                message="BiasDetector failed to generate a bias report.",
                context={"format": format},
            ) from exc

    # ------------------------------------------------------------------
    # Input preparation and validation
    # ------------------------------------------------------------------
    def _validate_inputs(
        self,
        *,
        data: pd.DataFrame,
        predictions: np.ndarray,
        labels: Optional[np.ndarray],
        sensitive_attributes: Optional[Sequence[str]],
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Optional[np.ndarray], Tuple[str, ...]]:
        if not isinstance(data, pd.DataFrame):
            raise DataValidationError(
                message="'data' must be a pandas DataFrame for BiasDetector.",
                context={"actual_type": type(data).__name__},
            )
        if data.empty:
            raise DataValidationError("'data' must not be empty for BiasDetector.")

        resolved_sensitive = normalize_sensitive_attributes(
            sensitive_attributes or self.sensitive_attrs,
            allow_empty=False,
        )
        validate_sensitive_attributes(data, resolved_sensitive, field_name="data", error_cls=SensitiveAttributeError)

        y_pred_raw = self._coerce_array(predictions, field_name="predictions")
        if len(data) != len(y_pred_raw):
            raise DataValidationError(
                message="Data and predictions length mismatch in BiasDetector.",
                context={"data_rows": len(data), "prediction_rows": len(y_pred_raw)},
            )

        y_true: Optional[np.ndarray] = None
        if labels is not None:
            y_true = self._coerce_array(labels, field_name="labels")
            if len(data) != len(y_true):
                raise DataValidationError(
                    message="Data and labels length mismatch in BiasDetector.",
                    context={"data_rows": len(data), "label_rows": len(y_true)},
                )

        y_pred_binary = self._binarize_predictions(y_pred_raw)
        return data.copy(), y_pred_raw, y_pred_binary, y_true, resolved_sensitive

    def _coerce_array(self, value: Any, *, field_name: str) -> np.ndarray:
        try:
            array = np.asarray(value)
            if array.ndim == 0:
                raise DataValidationError(
                    message=f"'{field_name}' must be array-like, not scalar.",
                    context={"field": field_name, "value": repr(value)},
                )
            return array.reshape(-1)
        except Exception as exc:
            if isinstance(exc, BiasDetectionError):
                raise
            raise DataValidationError(
                message=f"'{field_name}' could not be coerced into a one-dimensional array.",
                context={"field": field_name, "actual_type": type(value).__name__},
                cause=exc,
            ) from exc

    def _binarize_predictions(self, predictions: np.ndarray) -> np.ndarray:
        if predictions.dtype == bool:
            return predictions.astype(int)

        if np.issubdtype(predictions.dtype, np.number):
            unique_values = set(np.unique(predictions[~pd.isna(predictions)]).tolist()) if len(predictions) else set()
            if unique_values and unique_values.issubset({0, 1}):
                return predictions.astype(int)
            return (predictions.astype(float) >= self.prediction_threshold).astype(int)

        try:
            coerced = [1 if str(value).strip().lower() in {"1", "true", "yes", "approved", "positive"} else 0 for value in predictions]
            return np.asarray(coerced, dtype=int)
        except Exception as exc:
            raise DataValidationError(
                message="Predictions could not be binarized for BiasDetector.",
                context={"prediction_dtype": str(predictions.dtype)},
                cause=exc,
            ) from exc

    def _prepare_metric_frame(
        self,
        *,
        metric_name: str,
        data: pd.DataFrame,
        y_pred_raw: np.ndarray,
        y_pred_binary: np.ndarray,
        y_true: Optional[np.ndarray],
    ) -> pd.DataFrame:
        frame = data.copy()
        frame["_prediction_raw_"] = y_pred_raw
        frame["_prediction_binary_"] = y_pred_binary
        if y_true is not None:
            frame["_label_"] = y_true

        if metric_name == "demographic_parity":
            frame["_metric_target_"] = frame["_prediction_binary_"].astype(float)
            frame.attrs["metric_value_name"] = "selection_rate"
            return frame

        if metric_name == "equal_opportunity":
            if y_true is None:
                raise MissingFieldError(
                    message="'labels' are required to compute equal_opportunity in BiasDetector.",
                    context={"metric": metric_name},
                )
            filtered = frame[frame["_label_"] == 1].copy()
            filtered["_metric_target_"] = filtered["_prediction_binary_"].astype(float)
            filtered.attrs["metric_value_name"] = "true_positive_rate"
            return filtered

        if metric_name == "predictive_parity":
            if y_true is None:
                raise MissingFieldError(
                    message="'labels' are required to compute predictive_parity in BiasDetector.",
                    context={"metric": metric_name},
                )
            filtered = frame[frame["_prediction_binary_"] == 1].copy()
            filtered["_metric_target_"] = filtered["_label_"].astype(float)
            filtered.attrs["metric_value_name"] = "positive_predictive_value"
            return filtered

        if metric_name == "disparate_impact":
            frame["_metric_target_"] = frame["_prediction_binary_"].astype(float)
            frame.attrs["metric_value_name"] = "selection_rate"
            return frame

        raise ConfigurationError(
            message="Unsupported bias metric encountered during computation.",
            context={"metric": metric_name},
        )

    # ------------------------------------------------------------------
    # Group construction and metric computation
    # ------------------------------------------------------------------
    def _generate_intersectional_groups(
        self,
        data: pd.DataFrame,
        sensitive_attributes: Sequence[str],
        *,
        intersectional_depth: int,
        min_group_size: int,
    ) -> Dict[str, Dict[str, Any]]:
        groups: Dict[str, Dict[str, Any]] = {}
        max_depth = min(intersectional_depth, len(sensitive_attributes))

        for depth in range(1, max_depth + 1):
            for combo in combinations(sensitive_attributes, depth):
                grouped = data.groupby(list(combo), dropna=False, sort=False)
                family = "|".join(combo)
                for raw_values, group_frame in grouped:
                    values_tuple = raw_values if isinstance(raw_values, tuple) else (raw_values,)
                    if len(group_frame) < min_group_size:
                        continue
                    definition = {attr: value for attr, value in zip(combo, values_tuple)}
                    group_id = "|".join(f"{attr}={value}" for attr, value in definition.items())
                    groups[group_id] = {
                        "group_id": group_id,
                        "group_family": family,
                        "group_depth": depth,
                        "group_definition": definition,
                        "group_size": int(len(group_frame)),
                        "data": group_frame,
                    }

        if len(groups) > self.max_groups_per_metric:
            ranked = sorted(groups.values(), key=lambda item: item["group_size"], reverse=True)
            trimmed = ranked[: self.max_groups_per_metric]
            groups = {item["group_id"]: item for item in trimmed}
            logger.warning(
                "BiasDetector truncated generated groups to max_groups_per_metric=%s.",
                self.max_groups_per_metric,
            )
        return groups

    def _compute_metric_report(
        self,
        *,
        metric_name: str,
        metric_frame: pd.DataFrame,
        source: str,
        audit_id: str,
        context: Mapping[str, Any],
        tags: Sequence[Any],
        metadata: Mapping[str, Any],
    ) -> Dict[str, Any]:
        threshold = float(self.metric_thresholds.get(metric_name, self.DEFAULT_THRESHOLDS[metric_name]))
        metric_value_name = str(metric_frame.attrs.get("metric_value_name", "rate"))

        if metric_frame.empty:
            return {
                "metric": metric_name,
                "metric_value_name": metric_value_name,
                "eligible_rows": 0,
                "group_count": 0,
                "threshold": threshold,
                "summary_value": 0.0 if metric_name != "disparate_impact" else 1.0,
                "reference_value": None,
                "violation": False,
                "status": "insufficient_eligible_rows",
                "worst_group": None,
                "groups": {},
                "history_rows": [],
            }

        groups = self._generate_intersectional_groups(
            metric_frame,
            self.sensitive_attrs,
            intersectional_depth=self.intersectional_depth,
            min_group_size=self.min_group_size,
        )
        if not groups:
            return {
                "metric": metric_name,
                "metric_value_name": metric_value_name,
                "eligible_rows": int(len(metric_frame)),
                "group_count": 0,
                "threshold": threshold,
                "summary_value": 0.0 if metric_name != "disparate_impact" else 1.0,
                "reference_value": float(metric_frame["_metric_target_"].mean()),
                "violation": False,
                "status": "insufficient_group_support",
                "worst_group": None,
                "groups": {},
                "history_rows": [],
            }

        global_reference = float(metric_frame["_metric_target_"].mean())
        group_values: Dict[str, float] = {}
        raw_group_records: Dict[str, Dict[str, Any]] = {}

        for group_id, group_info in groups.items():
            group_frame = group_info["data"]
            group_values[group_id] = float(group_frame["_metric_target_"].mean())

        if metric_name == "disparate_impact":
            reference_for_ratio = max(group_values.values()) if group_values else 0.0
        else:
            reference_for_ratio = global_reference

        for group_id, group_info in groups.items():
            group_frame = group_info["data"]
            group_value = group_values[group_id]
            complement_frame = metric_frame.drop(index=group_frame.index, errors="ignore")
            if len(complement_frame) >= self.min_reference_size:
                comparison_frame = complement_frame
                reference_value = float(comparison_frame["_metric_target_"].mean())
            else:
                comparison_frame = metric_frame
                reference_value = global_reference

            bootstrap = self._bootstrap_mean(
                group_frame["_metric_target_"].to_numpy(dtype=float),
                bootstrap_samples=self.bootstrap_samples,
            )
            p_value = self._permutation_p_value(
                group_frame["_metric_target_"].to_numpy(dtype=float),
                comparison_frame["_metric_target_"].to_numpy(dtype=float),
                n_permutations=self.permutation_samples,
            )

            disparity = group_value - reference_value
            abs_disparity = abs(disparity)
            ratio_to_reference = self._safe_ratio(
                group_value,
                reference_for_ratio if metric_name == "disparate_impact" else reference_value,
            )
            violation = bool(ratio_to_reference < threshold) if metric_name == "disparate_impact" else bool(abs_disparity > threshold)

            raw_group_records[group_id] = {
                "group_id": group_id,
                "group_family": group_info["group_family"],
                "group_depth": group_info["group_depth"],
                "group_definition": json_safe(group_info["group_definition"]),
                "group_size": int(group_info["group_size"]),
                "eligible_rows": int(len(metric_frame)),
                "reference_size": int(len(comparison_frame)),
                "metric_value": float(group_value),
                "reference_value": float(reference_value),
                "disparity": float(disparity),
                "abs_disparity": float(abs_disparity),
                "ratio_to_reference": float(ratio_to_reference) if math.isfinite(ratio_to_reference) else None,
                "ci_lower": float(bootstrap[0]),
                "ci_upper": float(bootstrap[1]),
                "p_value": float(p_value) if p_value is not None else None,
                "threshold": float(threshold),
                "violation": violation,
                "significant": None,
                "adj_p_value": None,
            }

        adjusted_records = self._add_statistical_significance(raw_group_records, alpha=self.alpha)

        history_rows: List[Dict[str, Any]] = []
        groups_output: Dict[str, Dict[str, Any]] = {}
        abs_disparities: List[float] = []
        ratios: List[float] = []

        for group_id, record in adjusted_records.items():
            groups_output[group_id] = {
                "value": record["metric_value"],
                "reference_value": record["reference_value"],
                "disparity": record["disparity"],
                "abs_disparity": record["abs_disparity"],
                "ratio_to_reference": record["ratio_to_reference"],
                "ci_lower": record["ci_lower"],
                "ci_upper": record["ci_upper"],
                "p_value": record["p_value"],
                "adj_p_value": record["adj_p_value"],
                "significant": record["significant"],
                "group_size": record["group_size"],
                "eligible_rows": record["eligible_rows"],
                "reference_size": record["reference_size"],
                "threshold": record["threshold"],
                "violation": record["violation"],
                "group_definition": record["group_definition"],
                "group_family": record["group_family"],
                "group_depth": record["group_depth"],
            }
            abs_disparities.append(float(record["abs_disparity"]))
            if record["ratio_to_reference"] is not None:
                ratios.append(float(record["ratio_to_reference"]))
            history_rows.append(
                {
                    "timestamp": datetime.datetime.utcnow(),
                    "run_id": audit_id,
                    "audit_id": audit_id,
                    "metric": metric_name,
                    "group_id": record["group_id"],
                    "group_family": record["group_family"],
                    "group_depth": int(record["group_depth"]),
                    "group_size": int(record["group_size"]),
                    "eligible_rows": int(record["eligible_rows"]),
                    "reference_size": int(record["reference_size"]),
                    "metric_value": float(record["metric_value"]),
                    "reference_value": float(record["reference_value"]),
                    "disparity": float(record["disparity"]),
                    "abs_disparity": float(record["abs_disparity"]),
                    "ratio_to_reference": record["ratio_to_reference"],
                    "ci_lower": float(record["ci_lower"]),
                    "ci_upper": float(record["ci_upper"]),
                    "p_value": record["p_value"],
                    "adj_p_value": record["adj_p_value"],
                    "stat_significance": record["significant"],
                    "threshold": float(record["threshold"]),
                    "violation": bool(record["violation"]),
                    "group_definition": record["group_definition"],
                    "source": source,
                    "context_hash": stable_context_hash(context),
                    "tags": list(normalize_tags(tags)),
                    "metadata": json_safe(metadata),
                }
            )

        if metric_name == "disparate_impact":
            summary_value = min(ratios) if ratios else 1.0
            summary_violation = bool(summary_value < threshold)
            worst_group = min(groups_output.items(), key=lambda item: item[1]["ratio_to_reference"] if item[1]["ratio_to_reference"] is not None else float("inf"))[0] if groups_output else None
        else:
            observed_values = [record["metric_value"] for record in adjusted_records.values()]
            summary_value = max(observed_values) - min(observed_values) if observed_values else 0.0
            summary_violation = bool(summary_value > threshold)
            worst_group = max(groups_output.items(), key=lambda item: item[1]["abs_disparity"])[0] if groups_output else None

        summary = {
            "metric": metric_name,
            "metric_value_name": metric_value_name,
            "eligible_rows": int(len(metric_frame)),
            "group_count": int(len(groups_output)),
            "threshold": float(threshold),
            "reference_value": float(global_reference),
            "summary_value": float(summary_value),
            "mean_abs_disparity": float(np.mean(abs_disparities)) if abs_disparities else 0.0,
            "max_abs_disparity": float(np.max(abs_disparities)) if abs_disparities else 0.0,
            "min_ratio_to_reference": float(np.min(ratios)) if ratios else None,
            "violation": summary_violation,
            "status": "ok",
            "worst_group": worst_group,
            "groups": groups_output,
            "history_rows": history_rows,
        }
        return summary

    def _bootstrap_mean(self, values: np.ndarray, *, bootstrap_samples: int) -> Tuple[float, float]:
        clean = np.asarray(values, dtype=float)
        clean = clean[~np.isnan(clean)]
        if len(clean) == 0:
            return (0.0, 0.0)
        if len(clean) == 1:
            return (float(clean[0]), float(clean[0]))

        bootstrap_stats = []
        for _ in range(bootstrap_samples):
            sample = self._rng.choice(clean, size=len(clean), replace=True)
            bootstrap_stats.append(float(np.mean(sample)))
        lower = np.percentile(bootstrap_stats, 100 * (self.alpha / 2.0))
        upper = np.percentile(bootstrap_stats, 100 * (1.0 - self.alpha / 2.0))
        return float(lower), float(upper)

    def _permutation_p_value(
        self,
        group_values: np.ndarray,
        reference_values: np.ndarray,
        *,
        n_permutations: int,
    ) -> Optional[float]:
        group_values = np.asarray(group_values, dtype=float)
        reference_values = np.asarray(reference_values, dtype=float)
        group_values = group_values[~np.isnan(group_values)]
        reference_values = reference_values[~np.isnan(reference_values)]

        if len(group_values) < 2 or len(reference_values) < 2:
            return None

        combined = np.concatenate([group_values, reference_values])
        observed = abs(float(np.mean(group_values) - np.mean(reference_values)))
        exceed = 0

        for _ in range(n_permutations):
            permuted = self._rng.permutation(combined)
            group_perm = permuted[: len(group_values)]
            ref_perm = permuted[len(group_values):]
            delta = abs(float(np.mean(group_perm) - np.mean(ref_perm)))
            if delta >= observed:
                exceed += 1

        return float((exceed + 1) / (n_permutations + 1))

    def _add_statistical_significance(self, report: Dict[str, Dict[str, Any]], alpha: float) -> Dict[str, Dict[str, Any]]:
        valid_items = [(group_id, result) for group_id, result in report.items() if result.get("p_value") is not None]
        if not valid_items:
            return report

        p_values = [float(result["p_value"]) for _, result in valid_items]
        reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
        for index, (group_id, result) in enumerate(valid_items):
            result["significant"] = bool(reject[index])
            result["adj_p_value"] = float(pvals_corrected[index])
            report[group_id] = result
        return report

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        if denominator == 0:
            return 1.0 if numerator == 0 else float("inf")
        return float(numerator / denominator)

    # ------------------------------------------------------------------
    # History, memory, and reporting helpers
    # ------------------------------------------------------------------
    def _update_history(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        frame = pd.DataFrame(rows)
        self.bias_history = pd.concat([self.bias_history, frame], ignore_index=True)

    def _enforce_history_retention(self) -> None:
        if self.bias_history.empty:
            return
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=self.history_retention_days)
        self.bias_history["timestamp"] = pd.to_datetime(self.bias_history["timestamp"], errors="coerce")
        self.bias_history = self.bias_history[self.bias_history["timestamp"] >= cutoff]
        if len(self.bias_history) > self.max_history_rows:
            self.bias_history = self.bias_history.sort_values("timestamp").tail(self.max_history_rows).reset_index(drop=True)

    def _log_to_alignment_memory(self, report: Dict[str, Any], context: Mapping[str, Any], source: str) -> None:
        for metric_name, metric_report in report.get("metrics", {}).items():
            if metric_report.get("status") != "ok":
                continue
            summary_value = float(metric_report.get("summary_value", 0.0))
            threshold = float(metric_report.get("threshold", self.metric_thresholds.get(metric_name, 0.0)))
            memory_context = {
                **normalize_context(context, drop_none=True),
                "audit_id": report["report_metadata"]["audit_id"],
                "metric": metric_name,
                "source": source,
                "group_count": metric_report.get("group_count"),
            }
            self._memory_log_evaluation(
                metric=f"bias_{metric_name}",
                value=summary_value,
                threshold=threshold,
                context=memory_context,
                source=source,
                tags=["bias_detection", metric_name],
                audit_id=report["report_metadata"]["audit_id"],
                metadata={"metric_report": json_safe(metric_report)},
            )
            self._memory_record_outcome(
                context=memory_context,
                outcome={
                    "alignment_score": 1.0 - min(summary_value, 1.0) if metric_name != "disparate_impact" else max(min(summary_value, 1.0), 0.0),
                    "bias_rate": min(summary_value, 1.0) if metric_name != "disparate_impact" else max(0.0, 1.0 - min(summary_value, 1.0)),
                    "ethics_violations": 0,
                    "violation": bool(metric_report.get("violation", False)),
                },
                source=source,
                tags=["bias_detection", metric_name],
                metadata={"metric_threshold": threshold},
            )

    def _memory_log_evaluation(self, **kwargs: Any) -> Any:
        try:
            return self.alignment_memory.log_evaluation(**kwargs)
        except TypeError:
            fallback = {key: kwargs[key] for key in ("metric", "value", "threshold", "context") if key in kwargs}
            return self.alignment_memory.log_evaluation(**fallback)

    def _memory_record_outcome(self, **kwargs: Any) -> Any:
        try:
            return self.alignment_memory.record_outcome(**kwargs)
        except TypeError:
            fallback = {key: kwargs[key] for key in ("context", "outcome") if key in kwargs}
            return self.alignment_memory.record_outcome(**fallback)

    def _safe_memory_call(self, method_name: str, *, default: Any) -> Any:
        method = getattr(self.alignment_memory, method_name, None)
        if method is None:
            return default
        try:
            return method()
        except Exception as exc:
            logger.warning("BiasDetector memory integration call '%s' failed: %s", method_name, exc)
            return default

    def _dataset_summary(
        self,
        data: pd.DataFrame,
        y_pred_raw: np.ndarray,
        y_pred_binary: np.ndarray,
        y_true: Optional[np.ndarray],
        sensitive_attributes: Sequence[str],
    ) -> Dict[str, Any]:
        summary = {
            "rows": int(len(data)),
            "columns": list(map(str, data.columns.tolist())),
            "sensitive_attributes": list(sensitive_attributes),
            "prediction_dtype": str(y_pred_raw.dtype),
            "prediction_positive_rate": float(np.mean(y_pred_binary)) if len(y_pred_binary) else 0.0,
            "label_available": y_true is not None,
        }
        if y_true is not None:
            summary["label_positive_rate"] = float(np.mean(y_true.astype(float))) if len(y_true) else 0.0
        return summary

    def _current_state_report(self) -> Dict[str, Any]:
        if self.bias_history.empty:
            return {}

        current = (
            self.bias_history.sort_values("timestamp")
            .groupby(["metric", "group_id"], dropna=False)
            .tail(1)
            .reset_index(drop=True)
        )
        report: Dict[str, Any] = {
            "metrics_summary": {},
            "worst_performers": {},
            "metric_correlations": {},
        }

        for metric in self.metrics:
            metric_data = current[current["metric"] == metric]
            if metric_data.empty:
                continue

            abs_values = metric_data["abs_disparity"].astype(float)
            report["metrics_summary"][metric] = {
                "mean_abs_disparity": float(abs_values.mean()),
                "max_abs_disparity": float(abs_values.max()),
                "min_abs_disparity": float(abs_values.min()),
                "std_dev": float(abs_values.std(ddof=0)) if len(abs_values) > 1 else 0.0,
                "affected_groups": int(metric_data["violation"].astype(bool).sum()),
                "significant_groups": int(metric_data["stat_significance"].fillna(False).astype(bool).sum()),
            }

            worst_idx = abs_values.idxmax()
            worst_group = metric_data.loc[worst_idx]
            report["worst_performers"][metric] = {
                "group": worst_group["group_id"],
                "abs_disparity": float(worst_group["abs_disparity"]),
                "ratio_to_reference": float(worst_group["ratio_to_reference"]) if pd.notna(worst_group["ratio_to_reference"]) else None,
                "significance": bool(worst_group["stat_significance"]) if pd.notna(worst_group["stat_significance"]) else False,
            }

        pivot = current.pivot_table(index="group_id", columns="metric", values="abs_disparity", aggfunc="mean")
        if not pivot.empty and len(pivot.columns) > 1:
            report["metric_correlations"] = pivot.corr(method="spearman").fillna(0.0).to_dict()

        return report

    def _analyze_trends(self, window_size: int = 30) -> Dict[str, Any]:
        if self.bias_history.empty:
            return {}

        trends: Dict[str, Any] = {}
        effective_window = coerce_window_size(window_size, field_name="window_size")
        history = self.bias_history.copy()
        history["timestamp"] = pd.to_datetime(history["timestamp"], errors="coerce")
        history = history.dropna(subset=["timestamp"]) 
        history = history.sort_values("timestamp").set_index("timestamp")

        for metric in self.metrics:
            metric_history = history[history["metric"] == metric]
            if metric_history.empty:
                continue
            series = metric_history["abs_disparity"].resample("D").mean().ffill()
            if len(series) < 2 or series.isna().all() or series.nunique() <= 1:
                continue
            values = series.to_numpy(dtype=float)
            x = np.arange(len(values))
            trend_coeff = float(np.polyfit(x, values, 1)[0]) if len(values) >= 2 else 0.0
            rolling_mean = series.rolling(window=min(effective_window, len(series)), min_periods=1).mean()
            historical_segment = rolling_mean.iloc[:-effective_window] if len(rolling_mean) > effective_window else rolling_mean
            trends[metric] = {
                "trend_direction": "increasing" if trend_coeff > 0 else "decreasing" if trend_coeff < 0 else "stable",
                "trend_magnitude": abs(trend_coeff),
                "changepoints": self._detect_changepoints(series.dropna().to_numpy(dtype=float)),
                "recent_mean": float(rolling_mean.iloc[-min(effective_window, len(rolling_mean)):].mean()),
                "historical_mean": float(historical_segment.mean()) if len(historical_segment) else float(rolling_mean.mean()),
                "seasonality_strength": self._measure_seasonality(series),
            }
        return trends

    def _compute_aggregate_stats(self) -> Dict[str, Any]:
        if self.bias_history.empty:
            return {
                "distribution_analysis": {},
                "hypothesis_testing": {},
                "effect_sizes": {},
            }

        aggregate: Dict[str, Any] = {
            "distribution_analysis": {},
            "hypothesis_testing": {},
            "effect_sizes": {},
        }

        for metric in self.metrics:
            values = self.bias_history[self.bias_history["metric"] == metric]["abs_disparity"].astype(float)
            if values.empty:
                continue
            aggregate["distribution_analysis"][metric] = {
                "skewness": float(values.skew()) if len(values) > 2 else 0.0,
                "kurtosis": float(values.kurtosis()) if len(values) > 3 else 0.0,
                "normality_test": self._shapiro_wilk_test(values),
            }
            aggregate["effect_sizes"][metric] = {
                "cohens_d": self._cohens_d(values),
                "hedges_g": self._hedges_g(values),
                "variance_ratio": float(values.var(ddof=0) / max(self.bias_history["abs_disparity"].astype(float).var(ddof=0), 1e-12)),
            }

        manova_result = self._perform_manova()
        if manova_result:
            aggregate["hypothesis_testing"]["manova"] = manova_result
        return aggregate

    def _history_summary(self) -> Dict[str, Any]:
        if self.bias_history.empty:
            return {"rows": 0, "metrics": [], "groups": 0}
        return {
            "rows": int(len(self.bias_history)),
            "metrics": sorted(self.bias_history["metric"].dropna().unique().tolist()),
            "groups": int(self.bias_history["group_id"].nunique()),
            "latest_timestamp": pd.to_datetime(self.bias_history["timestamp"], errors="coerce").max().isoformat(),
            "violation_rate": float(self.bias_history["violation"].fillna(False).astype(bool).mean()),
        }

    def _detect_changepoints(self, values: np.ndarray) -> List[int]:
        clean = np.asarray(values, dtype=float)
        clean = clean[~np.isnan(clean)]
        if len(clean) < self.changepoint_min_points:
            return []
        variance = float(np.var(clean))
        if variance <= 1e-12:
            return []
        normalized = (clean - np.mean(clean)) / max(np.std(clean), 1e-12)
        try:
            algo = Binseg(model="l2").fit(normalized)
            points = algo.predict(pen=self.changepoint_penalty)
            return [int(point) for point in points[:-1] if 0 < point < len(clean)]
        except Exception as exc:
            logger.warning("BiasDetector changepoint detection failed: %s", exc)
            return []

    def _measure_seasonality(self, series: pd.Series) -> float:
        clean = pd.Series(series).astype(float).dropna()
        if len(clean) < self.seasonality_min_points:
            return 0.0
        try:
            stl = STL(clean, period=self.seasonality_period, robust=True)
            result = stl.fit()
            seasonal_var = float(np.nanvar(result.seasonal))
            total_var = float(np.nanvar(clean))
            return seasonal_var / total_var if total_var > 0 else 0.0
        except Exception as exc:
            logger.warning("BiasDetector seasonality analysis failed: %s", exc)
            return 0.0

    def _shapiro_wilk_test(self, data: pd.Series) -> Dict[str, Any]:
        clean = pd.Series(data).astype(float).dropna()
        if len(clean) < 3 or len(clean) > 5000:
            return {
                "statistic": None,
                "p_value": None,
                "effect_size": None,
                "warning": "Invalid sample size for Shapiro-Wilk (3 <= n <= 5000).",
            }
        stat_value, p_value = stats.shapiro(clean)
        effect_size = float(np.sqrt(max(-2.0 * np.log(max(stat_value, 1e-12)), 0.0)))
        return {
            "statistic": float(stat_value),
            "p_value": float(p_value),
            "effect_size": effect_size,
        }

    def _perform_manova(self) -> Dict[str, Any]:
        if self.bias_history.empty or len(self.metrics) < 2:
            return {}
        try:
            wide = (
                self.bias_history.pivot_table(
                    index=["run_id", "group_family", "group_id"],
                    columns="metric",
                    values="abs_disparity",
                    aggfunc="mean",
                )
                .dropna(axis=0, how="any")
                .reset_index()
            )
            metric_columns = [metric for metric in self.metrics if metric in wide.columns]
            if len(metric_columns) < 2 or wide["group_family"].nunique() < 2:
                return {}
            formula = f"{' + '.join(metric_columns)} ~ group_family"
            result = MANOVA.from_formula(formula, data=wide).mv_test()
            stat = result.results["group_family"]["stat"]
            return {
                "pillai_trace": float(stat.iloc[0, 0]),
                "f_value": float(stat.iloc[0, 4]),
                "p_value": float(stat.iloc[0, 5]),
                "tested_metrics": metric_columns,
            }
        except Exception as exc:
            logger.warning("BiasDetector MANOVA failed: %s", exc)
            return {}

    @staticmethod
    def _cohens_d(values: pd.Series) -> float:
        clean = pd.Series(values).astype(float).dropna()
        if len(clean) < 2:
            return 0.0
        std = float(clean.std(ddof=1))
        if std <= 1e-12:
            return 0.0
        return float(clean.mean() / std)

    def _hedges_g(self, values: pd.Series) -> float:
        clean = pd.Series(values).astype(float).dropna()
        n = len(clean)
        if n < 3:
            return 0.0
        correction = 1.0 - (3.0 / max((4.0 * n) - 9.0, 1.0))
        return float(self._cohens_d(clean) * correction)

    def _generate_run_id(self) -> str:
        return f"bias_audit_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"


if __name__ == "__main__":
    print("\n=== Running Bias Detector ===\n")
    printer.status("TEST", "Bias Detector initialized", "info")

    import random

    rng = np.random.default_rng(42)
    random.seed(42)

    detector = BiasDetector()

    n = 1500
    data = pd.DataFrame(
        {
            "gender": random.choices(["Male", "Female", "Non-binary"], weights=[0.48, 0.48, 0.04], k=n),
            "age_group": random.choices(["18-24", "25-34", "35-44", "45-54", "55-64", "65+"], weights=[0.14, 0.24, 0.20, 0.16, 0.14, 0.12], k=n),
            "race": random.choices(["White", "Black", "Asian", "Hispanic", "Other"], weights=[0.55, 0.14, 0.08, 0.18, 0.05], k=n),
            "education_level": random.choices(["No HS", "HS", "Some College", "Bachelor", "Graduate"], weights=[0.10, 0.25, 0.24, 0.24, 0.17], k=n),
        }
    )

    # Synthetic score generation with structured group disparity to exercise the detector.
    base_scores = rng.normal(loc=0.52, scale=0.12, size=n)
    base_scores += np.where(data["gender"].eq("Female"), -0.06, 0.00)
    base_scores += np.where(data["race"].eq("Black"), -0.08, 0.00)
    base_scores += np.where(data["education_level"].eq("Graduate"), 0.05, 0.00)
    base_scores = np.clip(base_scores, 0.0, 1.0)

    label_probs = np.clip(base_scores + rng.normal(0.0, 0.08, size=n), 0.0, 1.0)
    labels = rng.binomial(1, label_probs, size=n)
    predictions = base_scores

    context = {
        "domain": "synthetic_testing",
        "detector": "BiasDetector",
        "scenario": "production_ready_smoke_test",
    }

    compute_report = detector.compute_metrics(
        data=data,
        predictions=predictions,
        labels=labels,
        context=context,
        metadata={"test_case": "bias_detector_main_block"},
        source="bias_detector_test",
        tags=["test", "bias", "alignment"],
    )
    printer.pretty("compute_metrics", sanitize_for_logging(compute_report, preserve_length=True), "success")

    summary_report = detector.generate_report(format="structured")
    printer.pretty("generate_report", sanitize_for_logging(summary_report, preserve_length=True), "success")

    print("\n=== Test ran successfully ===\n")
