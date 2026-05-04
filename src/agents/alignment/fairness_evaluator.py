"""
Formal Fairness Verification System
Implements:
- Group fairness metrics (Dwork et al., 2012)
- Individual fairness verification (Dwork et al., 2012)
- Disparate impact analysis (Feldman et al., 2015)
- Multi-level statistical testing
"""

from __future__ import annotations

import json
import math
import numpy as np
import pandas as pd

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from scipy import stats
from scipy.stats import linregress
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from .utils import *
from .alignment_memory import AlignmentMemory
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Fairness Evaluator")
printer = PrettyPrinter

class FairnessEvaluator:
    """
    Multi-level fairness assessment system implementing:
    - Group fairness statistical verification
    - Individual fairness consistency checks
    - Disparate impact quantification
    - Longitudinal fairness tracking

    Key Features:
    1. Statistical parity difference with confidence intervals
    2. Equal opportunity / predictive parity disparity estimation
    3. Disparate impact ratio analysis with permutation significance testing
    4. Individual fairness consistency, Lipschitz approximation, and local violation discovery
    5. Shared-memory logging and historical reporting for downstream monitoring
    """

    SUPPORTED_GROUP_METRICS: Tuple[str, ...] = (
        "statistical_parity",
        "equal_opportunity",
        "predictive_parity",
        "disparate_impact",
    )
    SUPPORTED_INDIVIDUAL_METRICS: Tuple[str, ...] = (
        "consistency",
        "consistency_score",
        "lipschitz_constant",
        "fairness_radius",
        "violation_analysis",
        "fairness_violations",
    )
    SUPPORTED_SIMILARITY_METRICS: Tuple[str, ...] = (
        "euclidean",
        "manhattan",
        "cosine",
    )
    HISTORY_COLUMNS: Tuple[str, ...] = (
        "timestamp",
        "evaluation_type",
        "sensitive_attr",
        "metric",
        "metric_name",
        "value",
        "threshold",
        "ci_lower",
        "ci_upper",
        "p_value",
        "adj_p_value",
        "significant",
        "violation_flag",
        "groups",
        "group_sizes",
        "sample_size",
        "source",
        "tags",
        "metadata",
        "fingerprint",
    )

    def __init__(
        self,
        config_section_name: str = "fairness_evaluator",
        config_file_path: Optional[str] = None,
    ):
        self.config = load_global_config()
        self.config_section_name = ensure_non_empty_string(
            config_section_name,
            "config_section_name",
            error_cls=ConfigurationError,
        )
        self.config_file_path = config_file_path

        # Backward-compatible alias support.
        self.fe_config = get_config_section(self.config_section_name)
        if not self.fe_config and self.config_section_name == "fairness_evaluator":
            self.fe_config = get_config_section("fairness") or get_config_section("fairness_evaluator")
        self._validate_config()

        self.sensitive_attributes = tuple(normalize_sensitive_attributes(self.config.get("sensitive_attributes", [])))
        configured_override = self.fe_config.get("sensitive_attributes_override", [])
        self.sensitive_attrs = tuple(
            normalize_sensitive_attributes(configured_override, allow_empty=True)
        ) or self.sensitive_attributes

        self.group_metrics = tuple(
            normalize_metric_name(metric, namespace=None)
            for metric in ensure_sequence(
                self.fe_config["group_metrics"],
                "group_metrics",
                allow_empty=False,
                error_cls=ConfigurationError,
            )
        )
        self.individual_metrics = tuple(
            normalize_metric_name(metric, namespace=None)
            for metric in ensure_sequence(
                self.fe_config["individual_metrics"],
                "individual_metrics",
                allow_empty=False,
                error_cls=ConfigurationError,
            )
        )

        self.alpha = coerce_probability(self.fe_config["alpha"], field_name="alpha")
        self.n_bootstrap = coerce_positive_int(self.fe_config["n_bootstrap"], field_name="n_bootstrap")
        self.n_permutations = coerce_positive_int(self.fe_config["n_permutations"], field_name="n_permutations")
        self.batch_size = coerce_positive_int(self.fe_config["batch_size"], field_name="batch_size")
        self.similarity_metric = ensure_non_empty_string(
            self.fe_config["similarity_metric"],
            "similarity_metric",
            error_cls=ConfigurationError,
        ).strip().lower()
        self.prediction_threshold = coerce_probability(
            self.fe_config["prediction_threshold"],
            field_name="prediction_threshold",
        )
        self.min_group_size = coerce_positive_int(self.fe_config["min_group_size"], field_name="min_group_size")
        self.k_neighbors = coerce_positive_int(self.fe_config["k_neighbors"], field_name="k_neighbors")
        self.max_pair_samples = coerce_positive_int(
            self.fe_config["max_pair_samples"],
            field_name="max_pair_samples",
        )
        self.individual_violation_threshold = coerce_float(
            self.fe_config["individual_violation_threshold"],
            field_name="individual_violation_threshold",
            minimum=0.0,
        )
        self.history_max_rows = coerce_positive_int(
            self.fe_config["history_max_rows"],
            field_name="history_max_rows",
        )
        self.rolling_window = coerce_window_size(
            self.fe_config["rolling_window"],
            field_name="rolling_window",
        )
        self.enable_memory_logging = coerce_bool(self.fe_config["enable_memory_logging"], field_name="enable_memory_logging")
        self.strict_memory_integration = coerce_bool(
            self.fe_config["strict_memory_integration"],
            field_name="strict_memory_integration",
        )
        self.log_distributions = coerce_bool(self.fe_config["log_distributions"], field_name="log_distributions")
        self.distribution_sample_cap = coerce_positive_int(
            self.fe_config["distribution_sample_cap"],
            field_name="distribution_sample_cap",
        )

        self.metric_thresholds = self._load_metric_thresholds(self.fe_config.get("metric_thresholds", {}))
        self.history = pd.DataFrame(columns=self.HISTORY_COLUMNS)
        self.alignment_memory = AlignmentMemory()

        if config_file_path:
            logger.debug(
                "FairnessEvaluator received config_file_path=%s but retained global config loader handling.",
                config_file_path,
            )

        logger.info(
            "FairnessEvaluator initialized | sensitive_attrs=%s group_metrics=%s individual_metrics=%s",
            self.sensitive_attrs,
            self.group_metrics,
            self.individual_metrics,
        )

    # ------------------------------------------------------------------
    # Configuration and validation
    # ------------------------------------------------------------------
    def _validate_config(self) -> None:
        try:
            ensure_mapping(
                self.fe_config,
                self.config_section_name,
                allow_empty=False,
                error_cls=ConfigurationError,
            )
            ensure_keys_present = [
                "group_metrics",
                "individual_metrics",
                "alpha",
                "n_bootstrap",
                "n_permutations",
                "batch_size",
                "similarity_metric",
                "prediction_threshold",
                "min_group_size",
                "k_neighbors",
                "max_pair_samples",
                "individual_violation_threshold",
                "history_max_rows",
                "rolling_window",
                "metric_thresholds",
                "enable_memory_logging",
                "strict_memory_integration",
                "log_distributions",
                "distribution_sample_cap",
            ]
            missing = [key for key in ensure_keys_present if key not in self.fe_config]
            if missing:
                raise ConfigurationError(
                    f"'{self.config_section_name}' is missing required keys: {missing}.",
                    context={"config_section": self.config_section_name, "missing_keys": missing},
                )

            similarity_metric = str(self.fe_config["similarity_metric"]).strip().lower()
            if similarity_metric not in self.SUPPORTED_SIMILARITY_METRICS:
                raise ConfigurationError(
                    f"'similarity_metric' must be one of {self.SUPPORTED_SIMILARITY_METRICS}.",
                    context={"similarity_metric": self.fe_config["similarity_metric"]},
                )

            requested_group_metrics = [normalize_metric_name(metric, namespace=None) for metric in self.fe_config["group_metrics"]]
            unsupported_group = [metric for metric in requested_group_metrics if metric not in self.SUPPORTED_GROUP_METRICS]
            if unsupported_group:
                raise ConfigurationError(
                    "Unsupported group fairness metrics configured.",
                    context={"unsupported_group_metrics": unsupported_group},
                )

            requested_individual = [normalize_metric_name(metric, namespace=None) for metric in self.fe_config["individual_metrics"]]
            unsupported_individual = [metric for metric in requested_individual if metric not in self.SUPPORTED_INDIVIDUAL_METRICS]
            if unsupported_individual:
                raise ConfigurationError(
                    "Unsupported individual fairness metrics configured.",
                    context={"unsupported_individual_metrics": unsupported_individual},
                )

            thresholds = ensure_mapping(
                self.fe_config["metric_thresholds"],
                field_name="metric_thresholds",
                allow_empty=False,
                error_cls=ConfigurationError,
            )
            for metric_name in self.SUPPORTED_GROUP_METRICS:
                if metric_name not in thresholds:
                    raise ConfigurationError(
                        f"Missing threshold for metric '{metric_name}'.",
                        context={"metric_thresholds": list(thresholds.keys())},
                    )
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=ConfigurationError,
                message="FairnessEvaluator configuration validation failed.",
                context={
                    "config_section": self.config_section_name,
                    "config_path": self.config.get("__config_path__"),
                },
            ) from exc

    def _load_metric_thresholds(self, thresholds: Mapping[str, Any]) -> Dict[str, float]:
        normalized: Dict[str, float] = {}
        try:
            threshold_map = ensure_mapping(
                thresholds,
                field_name="metric_thresholds",
                allow_empty=False,
                error_cls=ConfigurationError,
            )
            for key, value in threshold_map.items():
                metric_name = normalize_metric_name(key, namespace=None)
                numeric = coerce_float(value, field_name=f"metric_threshold:{metric_name}", minimum=0.0)
                normalized[metric_name] = numeric
            return normalized
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=ConfigurationError,
                message="Failed to load fairness metric thresholds.",
                context={"threshold_keys": list(thresholds.keys()) if isinstance(thresholds, Mapping) else []},
            ) from exc

    def _validate_group_inputs(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        labels: Optional[np.ndarray],
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        try:
            if not isinstance(data, pd.DataFrame):
                raise DataValidationError(
                    "'data' must be a pandas DataFrame.",
                    context={"actual_type": type(data).__name__},
                )
            ensure_columns_present(data, self.sensitive_attrs, field_name="data", error_cls=SensitiveAttributeError)
            validate_sensitive_attributes(data, self.sensitive_attrs, field_name="data", error_cls=SensitiveAttributeError)

            predictions_array = self._coerce_prediction_array(predictions, len(data), field_name="predictions")
            labels_array = self._coerce_prediction_array(labels, len(data), field_name="labels")

            return data.copy(), predictions_array, labels_array
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=FairnessEvaluationError,
                message="Failed to validate group fairness inputs.",
                context={"n_rows": len(data) if isinstance(data, pd.DataFrame) else None},
            ) from exc

    def _validate_individual_inputs(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        try:
            if not isinstance(data, pd.DataFrame):
                raise DataValidationError(
                    "'data' must be a pandas DataFrame.",
                    context={"actual_type": type(data).__name__},
                )
            if data.empty:
                raise DataValidationError("'data' must contain at least one row for individual fairness evaluation.")
            predictions_array = self._coerce_prediction_array(predictions, len(data), field_name="predictions")
            return data.copy(), predictions_array
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=FairnessEvaluationError,
                message="Failed to validate individual fairness inputs.",
                context={"n_rows": len(data) if isinstance(data, pd.DataFrame) else None},
            ) from exc

    def _coerce_prediction_array(self, values: Any, expected_length: int, *, field_name: str) -> np.ndarray:
        if values is None:
            raise MissingFieldError(f"'{field_name}' must not be None.", context={"field": field_name})
        array = np.asarray(values)
        if array.ndim > 1:
            if 1 in array.shape:
                array = array.reshape(-1)
            else:
                raise DataValidationError(
                    f"'{field_name}' must be one-dimensional.",
                    context={"field": field_name, "shape": array.shape},
                )
        if len(array) != expected_length:
            raise DataValidationError(
                f"'{field_name}' length mismatch: expected {expected_length}, got {len(array)}.",
                context={"field": field_name, "expected_length": expected_length, "actual_length": len(array)},
            )
        if not np.issubdtype(array.dtype, np.number):
            try:
                array = array.astype(float)
            except Exception as exc:
                raise DataValidationError(
                    f"'{field_name}' must be numeric or coercible to numeric.",
                    context={"field": field_name, "dtype": str(array.dtype)},
                    cause=exc,
                ) from exc
        return array.astype(float)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate_group_fairness(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        labels: np.ndarray,
        *,
        source: str = "fairness_evaluator",
        tags: Optional[Sequence[Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive group fairness analysis with:
        - Statistical disparity estimation across configured sensitive attributes
        - Bootstrap confidence intervals
        - Permutation-based p-values
        - Benjamini-Hochberg correction within each sensitive attribute
        - Shared-memory event logging for longitudinal alignment tracking
        """
        try:
            df, prediction_values, label_values = self._validate_group_inputs(data, predictions, labels)
            binary_predictions = self._binarize_predictions(prediction_values)
            df["prediction"] = binary_predictions
            df["prediction_score"] = prediction_values
            df["label"] = label_values.astype(int)

            source_name = ensure_non_empty_string(source, "source", error_cls=DataValidationError)
            normalized_tags = normalize_tags(tags)
            normalized_metadata = normalize_metadata(metadata, drop_none=True)

            report: Dict[str, Any] = {
                "status": "ok",
                "evaluation_type": "group_fairness",
                "evaluated_at": pd.Timestamp.utcnow().isoformat(),
                "sensitive_attributes": list(self.sensitive_attrs),
                "group_metrics": list(self.group_metrics),
                "prediction_threshold": self.prediction_threshold,
                "results": {},
                "summary": {},
            }

            for sensitive_attr in self.sensitive_attrs:
                attribute_results: Dict[str, Dict[str, Any]] = {}
                for metric_name in self.group_metrics:
                    metric_result = self._evaluate_single_group_metric(df, sensitive_attr, metric_name)
                    attribute_results[metric_name] = metric_result
                self._apply_multiple_comparison_correction(attribute_results)
                report["results"][sensitive_attr] = attribute_results
                report["summary"][sensitive_attr] = self._build_attribute_summary(attribute_results)
                self._update_history_from_group_results(
                    sensitive_attr=sensitive_attr,
                    attribute_results=attribute_results,
                    source=source_name,
                    tags=normalized_tags,
                    metadata=normalized_metadata,
                )
                self._log_group_results_to_memory(
                    sensitive_attr=sensitive_attr,
                    attribute_results=attribute_results,
                    source=source_name,
                    tags=normalized_tags,
                    metadata=normalized_metadata,
                    sample_size=len(df),
                )

            report["evaluation_fingerprint"] = stable_record_fingerprint(report, namespace="fairness_group_evaluation")
            return report
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=FairnessEvaluationError,
                message="Group fairness evaluation failed.",
                context={"source": source, "sensitive_attrs": list(self.sensitive_attrs)},
                metadata={"rows": len(data) if isinstance(data, pd.DataFrame) else None},
            ) from exc

    def evaluate_individual_fairness(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        similarity_fn: Optional[Callable[[Any, Any], float]] = None,
        *,
        source: str = "fairness_evaluator",
        tags: Optional[Sequence[Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Individual fairness verification through:
        - k-nearest-neighbour consistency analysis
        - Lipschitz constant approximation on local neighborhoods
        - Violation discovery with example offending pairs
        - Shared-memory logging of aggregate fairness behaviour
        """
        try:
            df, prediction_values = self._validate_individual_inputs(data, predictions)
            encoded = self._prepare_numeric_features(df)
            metric = similarity_fn or self._resolve_similarity_metric()
            nn_info = self._compute_neighbourhood(encoded, metric)

            consistency = self._calculate_consistency(prediction_values, nn_info)
            lipschitz_constant = self._estimate_lipschitz(prediction_values, nn_info)
            fairness_violations = self._identify_violations(prediction_values, nn_info)

            requested = set(self.individual_metrics)
            result: Dict[str, Any] = {
                "status": "ok",
                "evaluation_type": "individual_fairness",
                "evaluated_at": pd.Timestamp.utcnow().isoformat(),
                "sample_size": int(len(df)),
                "feature_dimension": int(encoded.shape[1]),
                "similarity_metric": self.similarity_metric if similarity_fn is None else "callable",
                "k_neighbors": int(nn_info["effective_k"]),
            }
            if requested.intersection({"consistency", "consistency_score"}):
                result["consistency"] = consistency
            if requested.intersection({"lipschitz_constant", "fairness_radius"}):
                result["lipschitz_constant"] = lipschitz_constant
            if requested.intersection({"violation_analysis", "fairness_violations"}):
                result["fairness_violations"] = fairness_violations

            normalized_tags = normalize_tags(tags)
            normalized_metadata = normalize_metadata(metadata, drop_none=True)
            source_name = ensure_non_empty_string(source, "source", error_cls=DataValidationError)
            self._update_history_from_individual_results(
                result=result,
                source=source_name,
                tags=normalized_tags,
                metadata=normalized_metadata,
            )
            self._log_individual_results_to_memory(
                result=result,
                source=source_name,
                tags=normalized_tags,
                metadata=normalized_metadata,
            )
            result["evaluation_fingerprint"] = stable_record_fingerprint(result, namespace="fairness_individual_evaluation")
            return result
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=FairnessEvaluationError,
                message="Individual fairness evaluation failed.",
                context={"source": source},
                metadata={"rows": len(data) if isinstance(data, pd.DataFrame) else None},
            ) from exc

    def generate_report(self, format: str = "structured") -> Dict[str, Any]:
        """Generate a comprehensive fairness report from recorded evaluation history."""
        try:
            requested_format = ensure_non_empty_string(format, "format", error_cls=ValidationError).lower()
            report = {
                "format": requested_format,
                "generated_at": pd.Timestamp.utcnow().isoformat(),
                "current_state": self._current_state(),
                "historical_trends": self._analyze_trends(),
                "statistical_summary": self._statistical_summary(),
                "memory_analysis": self._safe_memory_report(),
            }
            report["report_fingerprint"] = stable_record_fingerprint(report, namespace="fairness_report")
            return report
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=FairnessEvaluationError,
                message="Failed to generate fairness report.",
                context={"format": format},
            ) from exc

    def fairness_records(self) -> pd.DataFrame:
        """Return enriched fairness history records with thresholds and rolling statistics."""
        if self.history.empty:
            return self.history.copy()

        enriched = self.history.copy()
        enriched = enriched.sort_values("timestamp").reset_index(drop=True)
        enriched["timestamp"] = pd.to_datetime(enriched["timestamp"], errors="coerce")
        enriched["threshold"] = enriched["threshold"].astype(float)
        enriched["value"] = enriched["value"].astype(float)
        enriched["violation_flag"] = enriched["violation_flag"].astype(bool)
        enriched["7d_rolling_avg"] = (
            enriched.groupby("metric")["value"].transform(lambda series: series.rolling(self.rolling_window, min_periods=1).mean())
        )
        enriched["drift_score"] = (
            enriched.groupby("metric")["value"].transform(lambda series: series.diff().abs().rolling(self.rolling_window, min_periods=1).mean())
        )
        enriched["day_of_week"] = enriched["timestamp"].dt.dayofweek
        enriched["hour_of_day"] = enriched["timestamp"].dt.hour
        return enriched

    # ------------------------------------------------------------------
    # Group fairness internals
    # ------------------------------------------------------------------
    def _evaluate_single_group_metric(self, df: pd.DataFrame, sensitive_attr: str, metric_name: str) -> Dict[str, Any]:
        try:
            if metric_name == "statistical_parity":
                working_df = df
                group_values, eligible_counts = self._compute_group_values(
                    working_df,
                    sensitive_attr,
                    numerator_fn=lambda group: float(group["prediction"].mean()),
                )
                statistic = self._disparity_from_group_values(group_values)
            elif metric_name == "equal_opportunity":
                working_df = df[df["label"] == 1].copy()
                group_values, eligible_counts = self._compute_group_values(
                    working_df,
                    sensitive_attr,
                    numerator_fn=lambda group: float(group["prediction"].mean()),
                )
                statistic = self._disparity_from_group_values(group_values)
            elif metric_name == "predictive_parity":
                working_df = df[df["prediction"] == 1].copy()
                group_values, eligible_counts = self._compute_group_values(
                    working_df,
                    sensitive_attr,
                    numerator_fn=lambda group: float(group["label"].mean()),
                )
                statistic = self._disparity_from_group_values(group_values)
            elif metric_name == "disparate_impact":
                working_df = df
                group_values, eligible_counts = self._compute_group_values(
                    working_df,
                    sensitive_attr,
                    numerator_fn=lambda group: float(group["prediction"].mean()),
                )
                statistic = self._ratio_from_group_values(group_values)
            else:
                raise FairnessEvaluationError(
                    f"Unsupported fairness metric '{metric_name}'.",
                    context={"metric": metric_name, "attribute": sensitive_attr},
                )

            boot_distribution = self._bootstrap_metric_distribution(working_df, sensitive_attr, metric_name)
            perm_distribution = self._permutation_metric_distribution(working_df, sensitive_attr, metric_name)
            ci_lower, ci_upper = self._confidence_interval(boot_distribution)
            p_value = self._permutation_p_value(metric_name, statistic, perm_distribution)
            threshold = self.metric_thresholds.get(metric_name, 0.0)
            violation_flag = (statistic < threshold) if metric_name == "disparate_impact" else (statistic > threshold)
            best_group, worst_group = self._extreme_groups(group_values, metric_name)

            result = {
                "metric": metric_name,
                "attribute": sensitive_attr,
                "value": float(statistic),
                "threshold": float(threshold),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "p_value": float(p_value),
                "significant": bool(p_value < self.alpha),
                "violation_flag": bool(violation_flag),
                "group_values": {str(key): float(value) for key, value in group_values.items()},
                "group_sizes": {str(key): int(value) for key, value in eligible_counts.items()},
                "sample_size": int(sum(eligible_counts.values())),
                "n_groups": int(len(group_values)),
                "best_group": best_group,
                "worst_group": worst_group,
            }
            if self.log_distributions:
                result["bootstrap_distribution"] = self._sample_distribution(boot_distribution)
                result["permutation_distribution"] = self._sample_distribution(perm_distribution)
            return result
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=FairnessEvaluationError,
                message="Failed to evaluate group fairness metric.",
                context={"metric": metric_name, "attribute": sensitive_attr},
            ) from exc

    def _compute_group_values(
        self,
        df: pd.DataFrame,
        sensitive_attr: str,
        numerator_fn: Callable[[pd.DataFrame], float],
    ) -> Tuple[Dict[str, float], Dict[str, int]]:
        if df.empty:
            raise FairnessEvaluationError(
                "No eligible samples available for fairness metric evaluation.",
                context={"attribute": sensitive_attr},
            )

        group_values: Dict[str, float] = {}
        group_sizes: Dict[str, int] = {}
        for group_name, group_df in df.groupby(sensitive_attr, dropna=False):
            if len(group_df) < self.min_group_size:
                continue
            value = numerator_fn(group_df)
            if pd.isna(value):
                continue
            group_key = str(group_name)
            group_values[group_key] = float(value)
            group_sizes[group_key] = int(len(group_df))

        if len(group_values) < 2:
            raise FairnessEvaluationError(
                "At least two groups meeting the configured minimum group size are required.",
                context={
                    "attribute": sensitive_attr,
                    "min_group_size": self.min_group_size,
                    "available_groups": sorted(str(v) for v in df[sensitive_attr].dropna().unique().tolist()),
                },
            )
        return group_values, group_sizes

    def _metric_statistic(self, df: pd.DataFrame, sensitive_attr: str, metric_name: str) -> float:
        if metric_name == "statistical_parity":
            group_values, _ = self._compute_group_values(df, sensitive_attr, lambda group: float(group["prediction"].mean()))
            return self._disparity_from_group_values(group_values)
        if metric_name == "equal_opportunity":
            filtered = df[df["label"] == 1].copy()
            group_values, _ = self._compute_group_values(filtered, sensitive_attr, lambda group: float(group["prediction"].mean()))
            return self._disparity_from_group_values(group_values)
        if metric_name == "predictive_parity":
            filtered = df[df["prediction"] == 1].copy()
            group_values, _ = self._compute_group_values(filtered, sensitive_attr, lambda group: float(group["label"].mean()))
            return self._disparity_from_group_values(group_values)
        if metric_name == "disparate_impact":
            group_values, _ = self._compute_group_values(df, sensitive_attr, lambda group: float(group["prediction"].mean()))
            return self._ratio_from_group_values(group_values)
        raise FairnessEvaluationError(f"Unsupported fairness metric '{metric_name}'.", context={"metric": metric_name})

    def _bootstrap_metric_distribution(self, df: pd.DataFrame, sensitive_attr: str, metric_name: str) -> np.ndarray:
        values: List[float] = []
        attempts = 0
        max_attempts = max(self.n_bootstrap * 4, self.n_bootstrap)
        while len(values) < self.n_bootstrap and attempts < max_attempts:
            attempts += 1
            sample = df.sample(frac=1.0, replace=True)
            try:
                values.append(float(self._metric_statistic(sample, sensitive_attr, metric_name)))
            except FairnessEvaluationError:
                continue
        if not values:
            raise FairnessEvaluationError(
                "Bootstrap estimation failed because no valid resamples met the fairness requirements.",
                context={"metric": metric_name, "attribute": sensitive_attr},
            )
        return np.asarray(values, dtype=float)

    def _permutation_metric_distribution(self, df: pd.DataFrame, sensitive_attr: str, metric_name: str) -> np.ndarray:
        values: List[float] = []
        base = df.copy()
        attempts = 0
        max_attempts = max(self.n_permutations * 4, self.n_permutations)
        while len(values) < self.n_permutations and attempts < max_attempts:
            attempts += 1
            permuted = base.copy()
            permuted[sensitive_attr] = np.random.permutation(permuted[sensitive_attr].values)
            try:
                values.append(float(self._metric_statistic(permuted, sensitive_attr, metric_name)))
            except FairnessEvaluationError:
                continue
        if not values:
            raise FairnessEvaluationError(
                "Permutation significance estimation failed because no valid permutations met the fairness requirements.",
                context={"metric": metric_name, "attribute": sensitive_attr},
            )
        return np.asarray(values, dtype=float)

    def _confidence_interval(self, values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        alpha_tail = max(0.0, min(1.0, (1.0 - confidence) / 2.0))
        return (
            float(np.quantile(values, alpha_tail)),
            float(np.quantile(values, 1.0 - alpha_tail)),
        )

    def _permutation_p_value(self, metric_name: str, observed: float, null_distribution: np.ndarray) -> float:
        if metric_name == "disparate_impact":
            null_stat = np.abs(1.0 - null_distribution)
            observed_stat = abs(1.0 - observed)
        else:
            null_stat = np.abs(null_distribution)
            observed_stat = abs(observed)
        return float((np.sum(null_stat >= observed_stat) + 1) / (len(null_distribution) + 1))

    @staticmethod
    def _disparity_from_group_values(group_values: Mapping[str, float]) -> float:
        values = np.asarray(list(group_values.values()), dtype=float)
        return float(np.max(values) - np.min(values))

    @staticmethod
    def _ratio_from_group_values(group_values: Mapping[str, float]) -> float:
        values = np.asarray(list(group_values.values()), dtype=float)
        max_value = float(np.max(values))
        min_value = float(np.min(values))
        if math.isclose(max_value, 0.0):
            return 1.0 if math.isclose(min_value, 0.0) else 0.0
        return float(min_value / max_value)

    @staticmethod
    def _extreme_groups(group_values: Mapping[str, float], metric_name: str) -> Tuple[Optional[str], Optional[str]]:
        if not group_values:
            return None, None
        if metric_name == "disparate_impact":
            sorted_groups = sorted(group_values.items(), key=lambda item: item[1])
            worst_group = sorted_groups[0][0]
            best_group = sorted_groups[-1][0]
            return best_group, worst_group
        sorted_groups = sorted(group_values.items(), key=lambda item: item[1])
        worst_group = sorted_groups[0][0]
        best_group = sorted_groups[-1][0]
        return best_group, worst_group

    def _apply_multiple_comparison_correction(self, attribute_results: Dict[str, Dict[str, Any]]) -> None:
        p_items = [(metric_name, result) for metric_name, result in attribute_results.items() if result.get("p_value") is not None]
        if not p_items:
            return

        p_values = np.asarray([float(result["p_value"]) for _, result in p_items], dtype=float)
        order = np.argsort(p_values)
        ranked = p_values[order]
        m = len(ranked)
        adjusted_ranked = np.empty(m, dtype=float)
        prev = 1.0
        for idx in range(m - 1, -1, -1):
            rank = idx + 1
            adjusted_value = min(prev, ranked[idx] * m / rank)
            adjusted_ranked[idx] = adjusted_value
            prev = adjusted_value
        adjusted = np.empty(m, dtype=float)
        adjusted[order] = adjusted_ranked

        for (metric_name, result), adj in zip(p_items, adjusted):
            result["adj_p_value"] = float(min(1.0, max(0.0, adj)))
            result["significant"] = bool(result["adj_p_value"] < self.alpha)

    def _build_attribute_summary(self, attribute_results: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
        violations = [metric for metric, result in attribute_results.items() if result.get("violation_flag")]
        significant = [metric for metric, result in attribute_results.items() if result.get("significant")]
        worst_metric = None
        worst_score = None
        for metric_name, result in attribute_results.items():
            value = float(result.get("value", 0.0))
            comparison_value = abs(1.0 - value) if metric_name == "disparate_impact" else value
            if worst_score is None or comparison_value > worst_score:
                worst_metric = metric_name
                worst_score = comparison_value
        return {
            "metrics_evaluated": int(len(attribute_results)),
            "violating_metrics": violations,
            "significant_metrics": significant,
            "worst_metric": worst_metric,
            "overall_violation": bool(violations),
        }

    # ------------------------------------------------------------------
    # Individual fairness internals
    # ------------------------------------------------------------------
    def _prepare_numeric_features(self, data: pd.DataFrame) -> np.ndarray:
        try:
            encoded = pd.get_dummies(data, drop_first=False)
            if encoded.empty:
                raise DataValidationError("Encoded feature matrix is empty after preprocessing.")
            numeric_matrix = encoded.astype(float).to_numpy()
            if numeric_matrix.shape[1] == 0:
                raise DataValidationError("No numeric feature dimensions available for individual fairness evaluation.")
            return numeric_matrix
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=FairnessEvaluationError,
                message="Failed to prepare numeric features for individual fairness evaluation.",
                context={"columns": list(data.columns)},
            ) from exc

    def _resolve_similarity_metric(self) -> str:
        if self.similarity_metric not in self.SUPPORTED_SIMILARITY_METRICS:
            raise ConfigurationError(
                f"Unsupported similarity metric '{self.similarity_metric}'.",
                context={"similarity_metric": self.similarity_metric},
            )
        return self.similarity_metric

    def _compute_neighbourhood(self, features: np.ndarray, metric: Any) -> Dict[str, Any]:
        n_samples = int(features.shape[0])
        if n_samples < 2:
            raise DataValidationError("At least two samples are required for individual fairness evaluation.")

        effective_k = min(self.k_neighbors, n_samples - 1)
        if effective_k < 1:
            raise DataValidationError("No valid neighbours available for individual fairness evaluation.")

        nn = NearestNeighbors(n_neighbors=effective_k + 1, metric=metric, algorithm="brute")
        nn.fit(features)

        batches_distances: List[np.ndarray] = []
        batches_indices: List[np.ndarray] = []
        for start in range(0, n_samples, self.batch_size):
            stop = min(start + self.batch_size, n_samples)
            distances_batch, indices_batch = nn.kneighbors(features[start:stop])
            batches_distances.append(distances_batch[:, 1:])
            batches_indices.append(indices_batch[:, 1:])

        distances = np.vstack(batches_distances)
        indices = np.vstack(batches_indices)
        return {
            "distances": distances,
            "indices": indices,
            "effective_k": effective_k,
        }

    def _calculate_consistency(self, predictions: np.ndarray, neighbourhood: Dict[str, Any]) -> float:
        neighbor_predictions = predictions[neighbourhood["indices"]]
        local_diffs = np.abs(predictions[:, None] - neighbor_predictions)
        local_consistency = 1.0 - np.mean(local_diffs, axis=1)
        local_consistency = np.clip(local_consistency, 0.0, 1.0)
        return float(np.mean(local_consistency))

    def _estimate_lipschitz(self, predictions: np.ndarray, neighbourhood: Dict[str, Any]) -> float:
        distances = neighbourhood["distances"]
        neighbor_predictions = predictions[neighbourhood["indices"]]
        diffs = np.abs(predictions[:, None] - neighbor_predictions)
        safe_distances = np.maximum(distances, 1e-9)
        lipschitz = diffs / safe_distances
        return float(np.max(lipschitz)) if lipschitz.size else 0.0

    def _identify_violations(self, predictions: np.ndarray, neighbourhood: Dict[str, Any]) -> Dict[str, Any]:
        distances = neighbourhood["distances"]
        indices = neighbourhood["indices"]
        neighbor_predictions = predictions[indices]
        diffs = np.abs(predictions[:, None] - neighbor_predictions)
        safe_distances = np.maximum(distances, 1e-9)
        deviations = diffs / safe_distances

        flattened = deviations.ravel()
        if flattened.size == 0:
            return {
                "max_violation": 0.0,
                "mean_violation": 0.0,
                "violation_rate": 0.0,
                "violation_count": 0,
                "sampled_examples": [],
                "status": "no_neighbourhood_deviation",
            }

        violation_mask = flattened > self.individual_violation_threshold
        violation_count = int(np.sum(violation_mask))
        sampled_examples: List[Dict[str, Any]] = []
        if violation_count > 0:
            pair_examples: List[Tuple[int, int, float, float]] = []
            for row_idx in range(indices.shape[0]):
                for col_idx in range(indices.shape[1]):
                    deviation = float(deviations[row_idx, col_idx])
                    if deviation > self.individual_violation_threshold:
                        pair_examples.append(
                            (
                                int(row_idx),
                                int(indices[row_idx, col_idx]),
                                float(distances[row_idx, col_idx]),
                                deviation,
                            )
                        )
            pair_examples.sort(key=lambda item: item[3], reverse=True)
            for anchor_idx, neighbor_idx, distance, deviation in pair_examples[:10]:
                sampled_examples.append(
                    {
                        "anchor_index": anchor_idx,
                        "neighbor_index": neighbor_idx,
                        "distance": distance,
                        "deviation": deviation,
                    }
                )

        return {
            "max_violation": float(np.max(flattened)),
            "mean_violation": float(np.mean(flattened)),
            "violation_rate": float(np.mean(violation_mask)),
            "violation_count": violation_count,
            "sampled_examples": sampled_examples,
            "status": "ok",
        }

    def _binarize_predictions(self, predictions: np.ndarray) -> np.ndarray:
        if np.array_equal(predictions, predictions.astype(int)) and set(np.unique(predictions)).issubset({0, 1}):
            return predictions.astype(int)
        return (predictions >= self.prediction_threshold).astype(int)

    # ------------------------------------------------------------------
    # History and memory integration
    # ------------------------------------------------------------------
    def _update_history_from_group_results(
        self,
        *,
        sensitive_attr: str,
        attribute_results: Mapping[str, Dict[str, Any]],
        source: str,
        tags: Sequence[Any],
        metadata: Mapping[str, Any],
    ) -> None:
        timestamp = pd.Timestamp.utcnow()
        new_rows: List[Dict[str, Any]] = []
        for metric_name, result in attribute_results.items():
            new_rows.append(
                {
                    "timestamp": timestamp,
                    "evaluation_type": "group",
                    "sensitive_attr": sensitive_attr,
                    "metric": metric_name,
                    "metric_name": f"{sensitive_attr}_{metric_name}",
                    "value": float(result["value"]),
                    "threshold": float(result["threshold"]),
                    "ci_lower": float(result["ci_lower"]),
                    "ci_upper": float(result["ci_upper"]),
                    "p_value": float(result["p_value"]),
                    "adj_p_value": float(result.get("adj_p_value", result["p_value"])),
                    "significant": bool(result["significant"]),
                    "violation_flag": bool(result["violation_flag"]),
                    "groups": json.dumps(result.get("group_values", {}), sort_keys=True),
                    "group_sizes": json.dumps(result.get("group_sizes", {}), sort_keys=True),
                    "sample_size": int(result.get("sample_size", 0)),
                    "source": source,
                    "tags": list(normalize_tags(tags)),
                    "metadata": normalize_metadata(metadata, drop_none=True),
                    "fingerprint": stable_record_fingerprint(
                        {
                            "sensitive_attr": sensitive_attr,
                            "metric": metric_name,
                            "value": result["value"],
                            "sample_size": result.get("sample_size", 0),
                        },
                        namespace="fairness_history_group",
                    ),
                }
            )

        if new_rows:
            self.history = pd.concat([self.history, pd.DataFrame(new_rows)], ignore_index=True)
            self._trim_history()

    def _update_history_from_individual_results(
        self,
        *,
        result: Mapping[str, Any],
        source: str,
        tags: Sequence[Any],
        metadata: Mapping[str, Any],
    ) -> None:
        timestamp = pd.Timestamp.utcnow()
        new_rows: List[Dict[str, Any]] = []

        metric_values = []
        if "consistency" in result:
            metric_values.append(("individual_consistency", float(result["consistency"]), 1.0, False))
        if "lipschitz_constant" in result:
            metric_values.append(("individual_lipschitz_constant", float(result["lipschitz_constant"]), self.individual_violation_threshold, float(result["lipschitz_constant"]) > self.individual_violation_threshold))
        if "fairness_violations" in result:
            violations = result["fairness_violations"]
            metric_values.append(("individual_violation_rate", float(violations["violation_rate"]), 0.0, float(violations["violation_rate"]) > 0.0))
            metric_values.append(("individual_mean_violation", float(violations["mean_violation"]), self.individual_violation_threshold, float(violations["mean_violation"]) > self.individual_violation_threshold))

        for metric_name, value, threshold, violation_flag in metric_values:
            new_rows.append(
                {
                    "timestamp": timestamp,
                    "evaluation_type": "individual",
                    "sensitive_attr": "individual",
                    "metric": metric_name,
                    "metric_name": metric_name,
                    "value": float(value),
                    "threshold": float(threshold),
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "p_value": np.nan,
                    "adj_p_value": np.nan,
                    "significant": False,
                    "violation_flag": bool(violation_flag),
                    "groups": json.dumps({}, sort_keys=True),
                    "group_sizes": json.dumps({}, sort_keys=True),
                    "sample_size": int(result.get("sample_size", 0)),
                    "source": source,
                    "tags": list(normalize_tags(tags)),
                    "metadata": normalize_metadata(metadata, drop_none=True),
                    "fingerprint": stable_record_fingerprint(
                        {"metric": metric_name, "value": value, "sample_size": result.get("sample_size", 0)},
                        namespace="fairness_history_individual",
                    ),
                }
            )

        if new_rows:
            self.history = pd.concat([self.history, pd.DataFrame(new_rows)], ignore_index=True)
            self._trim_history()

    def _trim_history(self) -> None:
        if len(self.history) > self.history_max_rows:
            self.history = self.history.iloc[-self.history_max_rows :].reset_index(drop=True)

    def _log_group_results_to_memory(
        self,
        *,
        sensitive_attr: str,
        attribute_results: Mapping[str, Dict[str, Any]],
        source: str,
        tags: Sequence[Any],
        metadata: Mapping[str, Any],
        sample_size: int,
    ) -> None:
        if not self.enable_memory_logging:
            return

        for metric_name, result in attribute_results.items():
            threshold = float(result["threshold"])
            value = float(result["value"] if metric_name != "disparate_impact" else 1.0 - float(result["value"]))
            context = normalize_context(
                {
                    "module": "fairness_evaluator",
                    "evaluation_type": "group",
                    "sensitive_attr": sensitive_attr,
                    "metric": metric_name,
                    "sample_size": sample_size,
                    "group_sizes": result.get("group_sizes", {}),
                    "group_values": result.get("group_values", {}),
                }
            )
            event = build_alignment_event(
                event_type=f"fairness_{metric_name}",
                source=source,
                tags=tags,
                metadata=merge_mappings(metadata, {"sensitive_attr": sensitive_attr, "metric": metric_name}),
                context=context,
                payload={
                    "value": result["value"],
                    "p_value": result["p_value"],
                    "significant": result["significant"],
                    "violation_flag": result["violation_flag"],
                },
            )
            self._safe_memory_log(
                metric=f"fairness_{metric_name}",
                value=float(value),
                threshold=float(threshold if metric_name != "disparate_impact" else max(0.0, 1.0 - threshold)),
                context=merge_mappings(context, {"event": event}),
                source=source,
                tags=tags,
                metadata=metadata,
            )
            self._safe_memory_outcome(
                context=merge_mappings(context, {"event": event}),
                outcome={
                    "bias_rate": min(1.0, abs(value)),
                    "alignment_score": float(max(0.0, 1.0 - abs(value))),
                    "ethics_violations": 0,
                    "violation": bool(result["violation_flag"]),
                },
                source=source,
                tags=tags,
                metadata=metadata,
            )

    def _log_individual_results_to_memory(
        self,
        *,
        result: Mapping[str, Any],
        source: str,
        tags: Sequence[Any],
        metadata: Mapping[str, Any],
    ) -> None:
        if not self.enable_memory_logging:
            return

        fairness_violations = result.get("fairness_violations", {})
        context = normalize_context(
            {
                "module": "fairness_evaluator",
                "evaluation_type": "individual",
                "sample_size": result.get("sample_size"),
                "feature_dimension": result.get("feature_dimension"),
                "k_neighbors": result.get("k_neighbors"),
            }
        )
        event = build_alignment_event(
            event_type="fairness_individual",
            source=source,
            tags=tags,
            metadata=metadata,
            context=context,
            payload=result,
        )
        self._safe_memory_log(
            metric="fairness_individual_violation_rate",
            value=float(fairness_violations.get("violation_rate", 0.0)),
            threshold=0.0,
            context=merge_mappings(context, {"event": event}),
            source=source,
            tags=tags,
            metadata=metadata,
        )
        self._safe_memory_outcome(
            context=merge_mappings(context, {"event": event}),
            outcome={
                "bias_rate": float(min(1.0, fairness_violations.get("mean_violation", 0.0))),
                "alignment_score": float(result.get("consistency", 0.0)),
                "ethics_violations": 0,
                "violation": bool(fairness_violations.get("violation_count", 0) > 0),
            },
            source=source,
            tags=tags,
            metadata=metadata,
        )

    def _safe_memory_log(
        self,
        *,
        metric: str,
        value: float,
        threshold: float,
        context: Mapping[str, Any],
        source: str,
        tags: Sequence[Any],
        metadata: Mapping[str, Any],
    ) -> None:
        try:
            self.alignment_memory.log_evaluation(
                metric=metric,
                value=value,
                threshold=threshold,
                context=dict(context),
                source=source,
                tags=tags,
                metadata=dict(metadata),
            )
        except Exception as exc:
            wrapped = wrap_alignment_exception(
                exc,
                target_cls=FairnessEvaluationError,
                message="FairnessEvaluator failed to log to AlignmentMemory.",
                context={"metric": metric, "source": source},
            )
            if self.strict_memory_integration:
                raise wrapped
            logger.warning(str(wrapped))

    def _safe_memory_outcome(
        self,
        *,
        context: Mapping[str, Any],
        outcome: Mapping[str, Any],
        source: str,
        tags: Sequence[Any],
        metadata: Mapping[str, Any],
    ) -> None:
        try:
            self.alignment_memory.record_outcome(
                context=dict(context),
                outcome=dict(outcome),
                source=source,
                tags=tags,
                metadata=dict(metadata),
            )
        except Exception as exc:
            wrapped = wrap_alignment_exception(
                exc,
                target_cls=FairnessEvaluationError,
                message="FairnessEvaluator failed to record an outcome in AlignmentMemory.",
                context={"source": source},
            )
            if self.strict_memory_integration:
                raise wrapped
            logger.warning(str(wrapped))

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------
    def _current_state(self) -> Dict[str, Any]:
        if self.history.empty:
            return {"status": "no_data", "message": "No fairness evaluations recorded."}

        latest = self.history.sort_values("timestamp").groupby(["evaluation_type", "metric_name"]).last().reset_index()
        metrics: Dict[str, Any] = {}
        for _, row in latest.iterrows():
            metrics[str(row["metric_name"])] = {
                "evaluation_type": row["evaluation_type"],
                "value": float(row["value"]),
                "threshold": float(row["threshold"]),
                "p_value": None if pd.isna(row["p_value"]) else float(row["p_value"]),
                "adj_p_value": None if pd.isna(row["adj_p_value"]) else float(row["adj_p_value"]),
                "significant": bool(row["significant"]),
                "violation_flag": bool(row["violation_flag"]),
                "last_updated": pd.Timestamp(row["timestamp"]).isoformat(),
            }

        violation_count = int(latest["violation_flag"].astype(bool).sum())
        worst_row = latest.iloc[latest["value"].astype(float).abs().idxmax()]
        return {
            "status": "ok",
            "metrics": metrics,
            "summary": {
                "tracked_metrics": int(len(latest)),
                "violation_count": violation_count,
                "worst_metric": str(worst_row["metric_name"]),
                "latest_timestamp": pd.Timestamp(latest["timestamp"].max()).isoformat(),
            },
        }

    def _analyze_trends(self) -> Dict[str, Any]:
        if self.history.empty:
            return {"status": "no_data", "message": "No historical fairness data available."}

        trends: Dict[str, Any] = {}
        grouped = self.history.sort_values("timestamp").groupby("metric_name")
        for metric_name, metric_data in grouped:
            metric_data = metric_data.dropna(subset=["timestamp", "value"])
            if len(metric_data) < 2:
                continue
            x = metric_data["timestamp"].astype("int64") // 10**9
            y = metric_data["value"].astype(float)
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            recent = metric_data.tail(min(self.rolling_window, len(metric_data)))
            recent_change = float(recent["value"].iloc[-1] - recent["value"].iloc[0]) if len(recent) >= 2 else 0.0
            trends[str(metric_name)] = {
                "trend_slope": float(slope),
                "trend_p_value": float(p_value),
                "r_squared": float(r_value ** 2),
                "recent_change": recent_change,
                "stability": "improving" if slope < 0 else "deteriorating" if slope > 0 else "stable",
            }

        return {
            "status": "ok",
            "temporal_patterns": trends,
            "summary": {
                "metrics_with_trends": int(len(trends)),
                "significant_trends": int(sum(1 for value in trends.values() if value["trend_p_value"] < self.alpha)),
            },
        }

    def _statistical_summary(self) -> Dict[str, Any]:
        if self.history.empty:
            return {"status": "no_data", "message": "No fairness statistics available."}

        summary: Dict[str, Any] = {}
        grouped = self.history.groupby("metric_name")
        for metric_name, metric_data in grouped:
            values = metric_data["value"].astype(float).dropna()
            if values.empty:
                continue
            shapiro_p = None
            if len(values) >= 3:
                sample_for_shapiro = values.iloc[: min(len(values), 5000)]
                shapiro_p = float(stats.shapiro(sample_for_shapiro).pvalue)
            summary[str(metric_name)] = {
                "count": int(values.count()),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
                "median": float(values.median()),
                "kurtosis": float(stats.kurtosis(values)) if len(values) > 3 else 0.0,
                "shapiro_p": shapiro_p,
                "normality": None if shapiro_p is None else bool(shapiro_p > self.alpha),
                "violation_rate": float(metric_data["violation_flag"].astype(bool).mean()),
            }

        correlation_payload: Dict[str, Dict[str, float]] = {}
        pivot_df = self.history.pivot_table(index="timestamp", columns="metric_name", values="value", aggfunc="mean")
        if pivot_df.shape[1] >= 2:
            correlation_df = pivot_df.corr().fillna(0.0)
            correlation_payload = {
                str(column): {str(inner_key): float(inner_value) for inner_key, inner_value in correlation_df[column].to_dict().items()}
                for column in correlation_df.columns
            }

        return {
            "status": "ok",
            "descriptive_statistics": summary,
            "cross_metric_correlation": correlation_payload,
        }

    def _safe_memory_report(self) -> Dict[str, Any]:
        try:
            return self.alignment_memory.get_memory_report()
        except Exception as exc:
            wrapped = wrap_alignment_exception(
                exc,
                target_cls=FairnessEvaluationError,
                message="Failed to obtain AlignmentMemory report for FairnessEvaluator.",
            )
            if self.strict_memory_integration:
                raise wrapped
            logger.warning(str(wrapped))
            return {"status": "unavailable", "message": str(wrapped)}

    def _sample_distribution(self, values: np.ndarray) -> List[float]:
        if len(values) <= self.distribution_sample_cap:
            return [float(value) for value in values.tolist()]
        sampled_idx = np.linspace(0, len(values) - 1, num=self.distribution_sample_cap, dtype=int)
        return [float(values[idx]) for idx in sampled_idx.tolist()]


if __name__ == "__main__":
    print("\n=== Running Fairness Evaluator ===\n")
    printer.status("TEST", "Fairness Evaluator initialized", "info")

    np.random.seed(42)
    n = 3000
    data = pd.DataFrame(
        {
            "feature_1": np.random.normal(0.0, 1.0, n),
            "feature_2": np.random.normal(2.0, 1.5, n),
            "feature_3": np.random.uniform(-1.0, 1.0, n),
            "gender": np.random.choice(["female", "male", "non_binary"], n, p=[0.46, 0.48, 0.06]),
            "age_group": np.random.choice(["18-29", "30-44", "45-59", "60+"], n, p=[0.28, 0.34, 0.24, 0.14]),
            "race": np.random.choice(["white", "black", "asian", "hispanic", "other"], n, p=[0.6, 0.15, 0.1, 0.1, 0.05]),
            "education_level": np.random.choice(["high_school", "bachelor", "master", "phd"], n, p=[0.4, 0.35, 0.15, 0.1]),
        }
    )

    raw_score = (
        0.8 * data["feature_1"]
        - 0.55 * data["feature_2"]
        + 0.25 * data["feature_3"]
        + np.where(data["gender"] == "male", 0.35, 0.0)
        + np.where(data["age_group"] == "60+", -0.20, 0.0)
        + np.where(data["race"] == "black", -0.10, 0.0)
        + np.where(data["education_level"] == "high_school", -0.05, 0.0)
        + np.random.normal(0.0, 0.45, n)
    )
    prediction_scores = 1.0 / (1.0 + np.exp(-raw_score))
    binary_predictions = (prediction_scores >= 0.5).astype(int)

    label_signal = (
        0.7 * data["feature_1"]
        - 0.45 * data["feature_2"]
        + 0.20 * data["feature_3"]
        + np.random.normal(0.0, 0.60, n)
    )
    labels = (label_signal > np.median(label_signal)).astype(int)

    evaluator = FairnessEvaluator()
    printer.pretty("config", {
        "sensitive_attrs": evaluator.sensitive_attrs,
        "group_metrics": evaluator.group_metrics,
        "individual_metrics": evaluator.individual_metrics,
    }, "success")

    group_report = evaluator.evaluate_group_fairness(
        data=data[["gender", "age_group", "race", "education_level"]],
        predictions=binary_predictions,
        labels=labels,
        source="fairness_test",
        tags=["test", "group"],
        metadata={"scenario": "synthetic_group_bias"},
    )
    printer.pretty("group_fairness_summary", group_report["summary"], "success")

    individual_report = evaluator.evaluate_individual_fairness(
        data=data[["feature_1", "feature_2", "feature_3"]],
        predictions=prediction_scores,
        source="fairness_test",
        tags=["test", "individual"],
        metadata={"scenario": "synthetic_individual_bias"},
    )
    printer.pretty("individual_fairness", individual_report, "success")

    report = evaluator.generate_report()
    printer.pretty("report_current_state", report["current_state"], "success")
    printer.pretty("report_historical_trends", report["historical_trends"], "success")
    printer.pretty("report_statistical_summary", report["statistical_summary"], "success")

    records = evaluator.fairness_records()
    printer.pretty(
        "fairness_records_preview",
        records.head(5).to_dict(orient="records") if not records.empty else [],
        "success",
    )

    print("\n=== Test ran successfully ===\n")
