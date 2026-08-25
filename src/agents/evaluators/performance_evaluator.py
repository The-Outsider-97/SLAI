from __future__ import annotations

import json
import math
import importlib
import numpy as np # type: ignore
import yaml # type: ignore

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .utils.config_loader import load_global_config, get_config_section
from .utils.evaluators_calculations import EvaluatorsCalculations
from .utils.evaluation_errors import (ComparisonError, ConfigLoadError, EvaluationError,
                                      MemoryAccessError, ValidationFailureError,
                                      MetricCalculationError, ReportGenerationError,
                                      VisualizationError)
from .modules.report import get_visualizer
from .evaluators_memory import EvaluatorsMemory
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Performance Evaluator")
printer = PrettyPrinter()

MODULE_VERSION = "2.0.0"
_HIGHER_IS_BETTER = frozenset({
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "balanced_accuracy",
    "matthews_corr",
    "composite_score",
})
_LOWER_IS_BETTER = frozenset({"log_loss"})


@dataclass(slots=True)
class BaselineComparison:
    """Structured comparison between current and baseline performance."""

    metric: str
    current: float
    baseline: float
    absolute_difference: float
    relative_change: Optional[float]
    improvement: bool
    direction: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ThresholdAssessment:
    """Decision and score-threshold assessment for a run."""

    decision_threshold: Optional[float]
    composite_threshold: Optional[float]
    composite_below_threshold: Optional[bool]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PerformanceEvaluationResult:
    """Serializable structured performance evaluation artifact."""

    metadata: Dict[str, Any]
    metrics: Dict[str, Any]
    input_summary: Dict[str, Any]
    threshold_assessment: Dict[str, Any]
    baseline_comparison: Dict[str, Dict[str, Any]]
    diagnostics: Dict[str, Any]
    recommendations: List[str]
    memory_entry_id: Optional[str] = None
    report: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": dict(self.metadata),
            "metrics": dict(self.metrics),
            "input_summary": dict(self.input_summary),
            "threshold_assessment": dict(self.threshold_assessment),
            "baseline_comparison": {k: dict(v) for k, v in self.baseline_comparison.items()},
            "diagnostics": dict(self.diagnostics),
            "recommendations": list(self.recommendations),
            "memory_entry_id": self.memory_entry_id,
            "report": self.report,
        }


class PerformanceEvaluator:
    """
    Production-grade evaluator for classification and decision performance.

    Responsibilities
    ----------------
    - Normalize labels, thresholded outputs, and probability predictions
    - Delegate metric calculation to the shared EvaluatorsCalculations service
    - Produce structured diagnostic payloads suitable for storage and reporting
    - Integrate with evaluator memory and the shared reporting visualizer
    - Surface failures through the shared evaluation error model
    """

    def __init__(self) -> None:
        self.config = load_global_config()
        self.config_path = str(self.config.get("__config_path__", "<config>"))
        self.perform_config = get_config_section("performance_evaluator")
        if not isinstance(self.perform_config, Mapping):
            raise ConfigLoadError(
                config_path=self.config_path,
                section="performance_evaluator",
                error_details="Section must be a mapping.",
            )

        self.classes = self.perform_config.get("classes")
        self.average = self.perform_config.get("average", "macro")
        self.enable_composite_score = bool(self.perform_config.get("enable_composite_score", True))
        self.store_results = bool(self.perform_config.get("store_results", True))
        self.threshold = self._coerce_optional_probability(
            self.perform_config.get("threshold", None),
            "performance_evaluator.threshold",
        )
        self.metric_weights = self._normalize_weight_mapping(
            self.perform_config.get("metric_weights", self.perform_config.get("weights", {})),
            "performance_evaluator.metric_weights",
        )
        self.metric_params = self.perform_config.get("metric_params", {})
        self.custom_metrics = self.perform_config.get("custom_metrics", [])

        self.memory = EvaluatorsMemory()
        self.calculations = EvaluatorsCalculations()
        self.history: List[Dict[str, Any]] = []
        self.disabled = False
        self._visualizer = None
        self._validate_config()

        logger.info(
            "Performance Evaluator initialized: average=%s threshold=%s store_results=%s",
            self.average,
            self.threshold,
            self.store_results,
        )

    # ------------------------------------------------------------------
    # Public evaluation API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        outputs: Sequence[Any],
        ground_truths: Sequence[Any],
        report: bool = False,
        *,
        probabilities: Optional[Sequence[Any]] = None,
        baseline_metrics: Optional[Mapping[str, Any]] = None,
        store_result: Optional[bool] = None,
        report_format: str = "markdown",
        include_visualizations: bool = True,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate classifier or decision outputs against reference labels.

        Parameters
        ----------
        outputs:
            Predicted labels, prediction mappings, scalar scores, or class-probability rows.
        ground_truths:
            Ground-truth labels with the same length as outputs.
        report:
            When True, include a rendered report payload in the returned result.
        probabilities:
            Optional explicit probability matrix or binary score vector.
        baseline_metrics:
            Optional mapping used to compare the current run against a baseline.
        store_result:
            Override for memory persistence.
        report_format:
            Report format when ``report=True``. Supported: markdown/json/yaml/dict.
        include_visualizations:
            Include visual assets in report payloads when supported.
        metadata:
            Optional caller-supplied metadata copied into the result.
        """
        if self.disabled:
            raise ValidationFailureError(
                rule_name="performance_evaluator_enabled",
                data=False,
                expected=True,
            )

        normalized_outputs = self._ensure_non_string_sequence(outputs, "outputs")
        normalized_truths = self._ensure_non_string_sequence(ground_truths, "ground_truths")
        if len(normalized_outputs) != len(normalized_truths):
            raise ValidationFailureError(
                rule_name="output_ground_truth_length_match",
                data={"outputs_len": len(normalized_outputs), "ground_truths_len": len(normalized_truths)},
                expected="Equal lengths",
            )
        if not normalized_outputs:
            raise ValidationFailureError(
                rule_name="performance_inputs_non_empty",
                data={"outputs": 0, "ground_truths": 0},
                expected="Non-empty output and ground-truth sequences",
            )

        explicit_probabilities = list(probabilities) if probabilities is not None else None
        persist = self.store_results if store_result is None else bool(store_result)
        caller_metadata = dict(metadata or {})

        try:
            predicted_labels, probability_matrix = self._resolve_prediction_inputs(
                normalized_outputs,
                normalized_truths,
                probabilities=explicit_probabilities,
            )

            metrics = self.calculations.calculate_classification_metrics(
                predicted_labels,
                normalized_truths,
                probabilities=probability_matrix,
            )

            input_summary = self._build_input_summary(
                predicted_labels=predicted_labels,
                ground_truths=normalized_truths,
                probabilities=probability_matrix,
            )
            threshold_assessment = self._build_threshold_assessment(metrics)
            baseline_comparison = self.compare_with_baseline(metrics, baseline_metrics or {})
            diagnostics = self._build_diagnostics(
                predicted_labels=predicted_labels,
                ground_truths=normalized_truths,
                probability_matrix=probability_matrix,
                metrics=metrics,
                threshold_assessment=threshold_assessment,
                baseline_comparison=baseline_comparison,
            )
            recommendations = self._generate_recommendations(metrics, threshold_assessment, diagnostics, baseline_comparison)

            result = PerformanceEvaluationResult(
                metadata={
                    "evaluated_at": _utcnow().isoformat(),
                    "module_version": MODULE_VERSION,
                    "config_path": self.config_path,
                    "average": self.average,
                    "store_result": persist,
                    "enable_composite_score": self.enable_composite_score,
                    "caller_metadata": caller_metadata,
                },
                metrics=metrics,
                input_summary=input_summary,
                threshold_assessment=threshold_assessment.to_dict(),
                baseline_comparison={k: v.to_dict() for k, v in baseline_comparison.items()},
                diagnostics=diagnostics,
                recommendations=recommendations,
            )

            result.memory_entry_id = self._persist_result(result.to_dict(), enabled=persist)
            self._update_visualizer(metrics)
            self.history.append(result.to_dict())

            if report:
                result.report = self.generate_report(
                    result.to_dict(),
                    format=report_format,
                    include_visualizations=include_visualizations,
                )

            return result.to_dict()

        except EvaluationError:
            raise
        except Exception as exc:
            raise MetricCalculationError(
                metric_name="performance_evaluation",
                inputs={
                    "outputs_preview": normalized_outputs[:3],
                    "ground_truths_preview": normalized_truths[:3],
                },
                reason=str(exc),
            ) from exc

    def compare_with_baseline(
        self,
        current_metrics: Mapping[str, Any],
        baseline_metrics: Mapping[str, Any],
    ) -> Dict[str, BaselineComparison]:
        """Compare numeric metrics against a baseline using metric-aware directionality."""
        if not baseline_metrics:
            return {}
        if not isinstance(current_metrics, Mapping):
            raise ComparisonError("current_metrics", "baseline_metrics", "current_metrics must be a mapping")
        if not isinstance(baseline_metrics, Mapping):
            raise ComparisonError("current_metrics", "baseline_metrics", "baseline_metrics must be a mapping")

        comparisons: Dict[str, BaselineComparison] = {}
        for metric, current_value in current_metrics.items():
            baseline_value = baseline_metrics.get(metric)
            if not self._is_numeric(current_value) or not self._is_numeric(baseline_value):
                continue

            current = float(current_value)
            baseline = float(baseline_value)
            absolute_difference = current - baseline
            relative_change = None if baseline == 0 else absolute_difference / abs(baseline)
            direction = self._metric_direction(metric)
            improvement = absolute_difference > 0 if direction == "higher" else absolute_difference < 0

            comparisons[metric] = BaselineComparison(
                metric=metric,
                current=current,
                baseline=baseline,
                absolute_difference=absolute_difference,
                relative_change=relative_change,
                improvement=improvement,
                direction=direction,
            )
        return comparisons

    def generate_report(
        self,
        result: Mapping[str, Any],
        *,
        format: str = "markdown",
        include_visualizations: bool = True,
    ) -> Any:
        """Generate a report from a structured evaluation result."""
        normalized = self._normalize_result_payload(result)
        payload = self._build_report_payload(normalized, include_visualizations=include_visualizations)
        normalized_format = str(format).strip().lower()

        try:
            if normalized_format == "dict":
                return payload
            if normalized_format == "json":
                return json.dumps(payload, indent=2, sort_keys=False, default=str)
            if normalized_format == "yaml":
                return yaml.safe_dump(payload, default_flow_style=False, sort_keys=False)
            if normalized_format == "markdown":
                return self._render_markdown_report(payload)
        except EvaluationError:
            raise
        except Exception as exc:
            raise ReportGenerationError(
                report_type="Performance",
                template="performance_report",
                error_details=str(exc),
            ) from exc

        raise ReportGenerationError(
            report_type="Performance",
            template="performance_report",
            error_details=f"Unsupported report format: {format}",
        )

    def export_report(
        self,
        result: Mapping[str, Any],
        *,
        destination_path: str,
        format: Optional[str] = None,
        include_visualizations: bool = True,
    ) -> str:
        path = destination_path.strip() if isinstance(destination_path, str) else ""
        if not path:
            raise ValidationFailureError(
                rule_name="report_destination_path",
                data=destination_path,
                expected="Non-empty destination path",
            )
        suffix_format = format or path.rsplit(".", 1)[-1].lower() if "." in path else format or "markdown"
        rendered = self.generate_report(result, format=suffix_format, include_visualizations=include_visualizations)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(rendered if isinstance(rendered, str) else json.dumps(rendered, indent=2, default=str))
        return path

    def disable_temporarily(self) -> None:
        """Temporarily disable performance evaluation during degraded mode."""
        self.disabled = True
        logger.warning("Performance Evaluator temporarily disabled.")

    def enable(self) -> None:
        """Re-enable performance evaluation after a temporary disable."""
        self.disabled = False
        logger.info("Performance Evaluator re-enabled.")

    # ------------------------------------------------------------------
    # Configuration and validation helpers
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        valid_averages = ["micro", "macro", "weighted", "samples", None]
        if self.average not in valid_averages:
            raise ConfigLoadError(
                config_path=self.config_path,
                section="performance_evaluator.average",
                error_details=f"Invalid value '{self.average}'. Valid options: {valid_averages}",
            )

        if self.classes is not None and not isinstance(self.classes, list):
            raise ConfigLoadError(
                config_path=self.config_path,
                section="performance_evaluator.classes",
                error_details=f"Expected list or null, got {type(self.classes).__name__}.",
            )

        if self.metric_weights:
            total_weight = sum(self.metric_weights.values())
            if not (0.99 <= total_weight <= 1.01):
                logger.warning("Metric weights sum to %.4f instead of 1.0", total_weight)

        if self.custom_metrics and not isinstance(self.custom_metrics, list):
            raise ConfigLoadError(
                config_path=self.config_path,
                section="performance_evaluator.custom_metrics",
                error_details="custom_metrics must be a list.",
            )

    # ------------------------------------------------------------------
    # Input normalization
    # ------------------------------------------------------------------

    def _resolve_prediction_inputs(
        self,
        outputs: Sequence[Any],
        truths: Sequence[Any],
        *,
        probabilities: Optional[Sequence[Any]],
    ) -> Tuple[List[Any], Optional[List[Any]]]:
        if probabilities is not None:
            probability_matrix = self._coerce_probabilities(probabilities, expected_len=len(truths))
            labels = self._labels_from_probabilities(probability_matrix)
            return labels, probability_matrix

        first = outputs[0]
        if self._looks_like_probability_rows(outputs):
            probability_matrix = self._coerce_probabilities(outputs, expected_len=len(truths))
            labels = self._labels_from_probabilities(probability_matrix)
            return labels, probability_matrix

        if isinstance(first, Mapping):
            labels, probability_matrix = self._extract_predictions_from_mappings(outputs, expected_len=len(truths))
            return labels, probability_matrix

        if self._can_threshold_outputs(outputs, truths):
            threshold = self.threshold if self.threshold is not None else 0.5
            labels = [1 if float(item) >= threshold else 0 for item in outputs]
            return labels, None

        return list(outputs), None

    def _extract_predictions_from_mappings(
        self,
        outputs: Sequence[Mapping[str, Any]],
        *,
        expected_len: int,
    ) -> Tuple[List[Any], Optional[List[Any]]]:
        labels: List[Any] = []
        probability_rows: List[Any] = []
        saw_probabilities = False

        for item in outputs:
            if not isinstance(item, Mapping):
                raise ValidationFailureError(
                    rule_name="prediction_mapping_shape",
                    data=type(item).__name__,
                    expected="Mapping with prediction fields",
                )
            if "probabilities" in item:
                probability_rows.append(item["probabilities"])
                saw_probabilities = True
            elif "scores" in item:
                probability_rows.append(item["scores"])
                saw_probabilities = True

            if "label" in item:
                labels.append(item["label"])
            elif "prediction" in item:
                labels.append(item["prediction"])
            elif "value" in item and not saw_probabilities:
                labels.append(item["value"])
            elif not saw_probabilities:
                raise ValidationFailureError(
                    rule_name="prediction_mapping_fields",
                    data=dict(item),
                    expected="Mapping containing label/prediction/value or probabilities/scores",
                )

        probability_matrix = None
        if saw_probabilities:
            probability_matrix = self._coerce_probabilities(probability_rows, expected_len=expected_len)
            if not labels:
                labels = self._labels_from_probabilities(probability_matrix)

        if len(labels) != expected_len:
            raise ValidationFailureError(
                rule_name="prediction_mapping_length_alignment",
                data=len(labels),
                expected=expected_len,
            )
        return labels, probability_matrix

    def _coerce_probabilities(self, probabilities: Sequence[Any], *, expected_len: int) -> List[Any]:
        if len(probabilities) != expected_len:
            raise ValidationFailureError(
                rule_name="probability_length_alignment",
                data=len(probabilities),
                expected=expected_len,
            )
        normalized: List[Any] = []
        for row in probabilities:
            if isinstance(row, np.ndarray):
                row = row.tolist()
            normalized.append(row)
        return normalized

    def _labels_from_probabilities(self, probabilities: Sequence[Any]) -> List[Any]:
        if not probabilities:
            return []
        first = probabilities[0]
        if self._is_numeric(first):
            threshold = self.threshold if self.threshold is not None else 0.5
            return [1 if float(item) >= threshold else 0 for item in probabilities]

        labels: List[Any] = []
        for row in probabilities:
            if isinstance(row, np.ndarray):
                row = row.tolist()
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) == 0:
                raise ValidationFailureError(
                    rule_name="probability_row_shape",
                    data=row,
                    expected="Non-empty numeric probability row",
                )
            predicted_index = int(np.argmax(np.asarray(row, dtype=float)))
            if isinstance(self.classes, list) and predicted_index < len(self.classes):
                labels.append(self.classes[predicted_index])
            else:
                labels.append(predicted_index)
        return labels

    def _build_input_summary(
        self,
        *,
        predicted_labels: Sequence[Any],
        ground_truths: Sequence[Any],
        probabilities: Optional[Sequence[Any]],
    ) -> Dict[str, Any]:
        truth_distribution = self._distribution(ground_truths)
        prediction_distribution = self._distribution(predicted_labels)
        summary = {
            "sample_count": len(ground_truths),
            "truth_distribution": truth_distribution,
            "prediction_distribution": prediction_distribution,
            "probabilities_provided": probabilities is not None,
            "class_labels": list(self.classes) if isinstance(self.classes, list) else None,
        }
        if probabilities is not None:
            summary["probability_shape"] = self._infer_probability_shape(probabilities)
            summary["average_prediction_confidence"] = self._average_prediction_confidence(probabilities)
        return summary

    def _build_threshold_assessment(self, metrics: Mapping[str, Any]) -> ThresholdAssessment:
        warnings: List[str] = []
        composite_score = metrics.get("composite_score")
        composite_threshold = self.calculations.performance_threshold
        composite_below = None
        if composite_score is not None and composite_threshold is not None:
            composite_below = float(composite_score) < float(composite_threshold)
            if composite_below:
                warnings.append(
                    f"Composite score {float(composite_score):.4f} is below configured threshold {float(composite_threshold):.4f}."
                )
        if metrics.get("log_loss") == float("inf"):
            warnings.append("Log loss could not be computed reliably for the supplied predictions.")
        if metrics.get("roc_auc") == 0.0 and "roc_auc" in metrics:
            warnings.append("ROC AUC is zero or unavailable for the supplied class distribution.")
        return ThresholdAssessment(
            decision_threshold=self.threshold,
            composite_threshold=float(composite_threshold) if composite_threshold is not None else None,
            composite_below_threshold=composite_below,
            warnings=warnings,
        )

    def _build_diagnostics(
        self,
        *,
        predicted_labels: Sequence[Any],
        ground_truths: Sequence[Any],
        probability_matrix: Optional[Sequence[Any]],
        metrics: Mapping[str, Any],
        threshold_assessment: ThresholdAssessment,
        baseline_comparison: Mapping[str, BaselineComparison],
    ) -> Dict[str, Any]:
        confusion = metrics.get("confusion_matrix", [])
        diagnostics: Dict[str, Any] = {
            "confusion_analysis": self._analyze_confusion_matrix(confusion),
            "warnings": list(threshold_assessment.warnings),
            "custom_metrics_present": bool(metrics.get("custom_metrics")),
            "custom_metric_errors": metrics.get("custom_metric_errors", {}),
            "baseline_improvements": {
                key: comparison.improvement for key, comparison in baseline_comparison.items()
            },
            "history_length": len(self.history),
        }
        if probability_matrix is not None:
            diagnostics["average_prediction_confidence"] = self._average_prediction_confidence(probability_matrix)
            diagnostics["average_entropy"] = self._average_entropy(probability_matrix)
        if self._is_binary_label_set(predicted_labels, ground_truths) and confusion:
            diagnostics["binary_failure_modes"] = self._binary_failure_modes(confusion)
        return diagnostics

    def _generate_recommendations(
        self,
        metrics: Mapping[str, Any],
        threshold_assessment: ThresholdAssessment,
        diagnostics: Mapping[str, Any],
        baseline_comparison: Mapping[str, BaselineComparison],
    ) -> List[str]:
        recommendations: List[str] = []

        if threshold_assessment.composite_below_threshold:
            recommendations.append(
                "Review class balance, decision thresholding, and feature quality because the composite score is below the configured threshold."
            )
        if float(metrics.get("recall", 1.0)) < 0.8:
            recommendations.append(
                "Recall is weak. Consider threshold tuning, harder positive examples, or class-rebalancing strategies."
            )
        if float(metrics.get("precision", 1.0)) < 0.8:
            recommendations.append(
                "Precision is weak. Review false-positive drivers, ambiguous classes, and threshold calibration."
            )
        if float(metrics.get("f1", 1.0)) < 0.8:
            recommendations.append(
                "F1 score indicates an imbalanced precision/recall trade-off. Revisit data quality and decision calibration."
            )
        if metrics.get("log_loss") is not None and self._is_numeric(metrics.get("log_loss")) and float(metrics["log_loss"]) > 1.0:
            recommendations.append(
                "Probability calibration appears weak. Consider temperature scaling, isotonic calibration, or improved confidence estimation."
            )
        if diagnostics.get("custom_metric_errors"):
            recommendations.append(
                "One or more configured custom metrics failed to evaluate. Review metric import paths and parameter configuration."
            )
        worsening = [name for name, comparison in baseline_comparison.items() if not comparison.improvement]
        if worsening:
            recommendations.append(
                f"Current run regressed versus baseline for: {', '.join(sorted(worsening))}. Inspect data drift and model-version changes."
            )
        if not recommendations:
            recommendations.append("Performance is stable against configured checks. Continue monitoring drift, calibration, and class balance.")

        deduped: List[str] = []
        seen: set[str] = set()
        for item in recommendations:
            normalized = item.strip()
            if normalized and normalized.casefold() not in seen:
                deduped.append(normalized)
                seen.add(normalized.casefold())
        return deduped

    # ------------------------------------------------------------------
    # Persistence and visualization
    # ------------------------------------------------------------------

    def _persist_result(self, result: Mapping[str, Any], *, enabled: bool) -> Optional[str]:
        if not enabled:
            return None
        try:
            if hasattr(self.memory, "add_evaluation_result"):
                return self.memory.add_evaluation_result(
                    evaluator_name="performance_evaluator",
                    result=result,
                    tags=["performance_eval", "classification"],
                    priority="medium",
                )
            return self.memory.add(
                entry=result,
                tags=["performance_eval", "classification"],
                priority="medium",
                source="performance_evaluator",
                category="evaluation",
            )
        except EvaluationError as exc:
            raise MemoryAccessError("add", "performance_eval_result", str(exc)) from exc
        except Exception as exc:
            raise MemoryAccessError("add", "performance_eval_result", str(exc)) from exc

    def _update_visualizer(self, metrics: Mapping[str, Any]) -> None:
        try:
            visualizer = get_visualizer()
            confusion = np.asarray(metrics.get("confusion_matrix", []), dtype=float)
            total = int(confusion.sum()) if confusion.size else 0
            successes = int(np.trace(confusion)) if confusion.size else 0
            failures = max(total - successes, 0)
            payload = {
                "successes": successes,
                "failures": failures,
                "pass_rate": float(metrics.get("accuracy", 0.0)),
            }
            visualizer.update_metrics(payload)
            self._visualizer = visualizer
        except Exception as exc:
            logger.warning("Visualizer update skipped: %s", exc)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def _normalize_result_payload(self, result: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(result, Mapping):
            raise ValidationFailureError(
                rule_name="performance_result_payload",
                data=type(result).__name__,
                expected="Mapping generated by PerformanceEvaluator.evaluate",
            )
        required = {"metadata", "metrics", "input_summary", "threshold_assessment", "diagnostics", "recommendations"}
        missing = sorted(key for key in required if key not in result)
        if missing:
            raise ValidationFailureError(
                rule_name="performance_result_structure",
                data=list(result.keys()),
                expected=f"Mapping containing keys: {sorted(required)}; missing={missing}",
            )
        return {
            "metadata": dict(result["metadata"]),
            "metrics": dict(result["metrics"]),
            "input_summary": dict(result["input_summary"]),
            "threshold_assessment": dict(result["threshold_assessment"]),
            "baseline_comparison": dict(result.get("baseline_comparison", {})),
            "diagnostics": dict(result["diagnostics"]),
            "recommendations": list(result.get("recommendations", [])),
            "memory_entry_id": result.get("memory_entry_id"),
        }

    def _build_report_payload(self, result: Mapping[str, Any], *, include_visualizations: bool) -> Dict[str, Any]:
        payload = {
            "metadata": dict(result["metadata"]),
            "summary": {
                "accuracy": result["metrics"].get("accuracy"),
                "precision": result["metrics"].get("precision"),
                "recall": result["metrics"].get("recall"),
                "f1": result["metrics"].get("f1"),
                "composite_score": result["metrics"].get("composite_score"),
                "sample_count": result["input_summary"].get("sample_count"),
            },
            "input_summary": dict(result["input_summary"]),
            "metrics": dict(result["metrics"]),
            "threshold_assessment": dict(result["threshold_assessment"]),
            "baseline_comparison": dict(result.get("baseline_comparison", {})),
            "diagnostics": dict(result["diagnostics"]),
            "recommendations": list(result.get("recommendations", [])),
            "memory_entry_id": result.get("memory_entry_id"),
        }
        if include_visualizations:
            payload["visualizations"] = self._generate_visual_assets(result)
        return payload

    def _generate_visual_assets(self, result: Mapping[str, Any]) -> Dict[str, Any]:
        visual_assets: Dict[str, Any] = {}
        try:
            visualizer = self._visualizer or get_visualizer()
            chart = visualizer.render_temporal_chart(getattr(__import__('PyQt5.QtCore', fromlist=['QSize']), 'QSize')(600, 400), 'pass_rate')
            visual_assets["pass_rate_trend"] = {
                "encoding": "base64-png",
                "image": visualizer._chart_to_base64(chart),
            }
        except Exception as exc:
            raise VisualizationError(
                chart_type="pass_rate_trend",
                data=result.get("metrics", {}),
                error_details=str(exc),
            )
        return visual_assets

    def _render_markdown_report(self, payload: Mapping[str, Any]) -> str:
        report: List[str] = []
        report.append("# Performance Evaluation Report\n")
        report.append(f"**Generated**: {payload['metadata'].get('evaluated_at', _utcnow().isoformat())}\n")

        summary = payload["summary"]
        report.append("## Executive Summary\n")
        report.append(f"- **Samples Evaluated**: {summary.get('sample_count', 0)}")
        report.append(f"- **Accuracy**: {self._format_metric(summary.get('accuracy'))}")
        report.append(f"- **Precision**: {self._format_metric(summary.get('precision'))}")
        report.append(f"- **Recall**: {self._format_metric(summary.get('recall'))}")
        report.append(f"- **F1 Score**: {self._format_metric(summary.get('f1'))}")
        if summary.get("composite_score") is not None:
            report.append(f"- **Composite Score**: {self._format_metric(summary.get('composite_score'))}")

        visualizations = payload.get("visualizations", {})
        if visualizations.get("pass_rate_trend"):
            report.append("\n## Performance Trend\n")
            report.append(
                f"![Pass Rate Trend](data:image/png;base64,{visualizations['pass_rate_trend']['image']})"
            )

        report.append("\n## Core Metrics\n")
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "log_loss", "composite_score"]:
            if key in payload["metrics"]:
                report.append(f"- **{key.replace('_', ' ').title()}**: {self._format_metric(payload['metrics'][key])}")

        if payload["baseline_comparison"]:
            report.append("\n## Baseline Comparison\n")
            for metric, comparison in payload["baseline_comparison"].items():
                sign = "improved" if comparison.get("improvement") else "regressed"
                report.append(
                    f"- **{metric}**: current={self._format_metric(comparison.get('current'))}, "
                    f"baseline={self._format_metric(comparison.get('baseline'))}, {sign} by {self._format_metric(comparison.get('absolute_difference'))}"
                )

        threshold = payload["threshold_assessment"]
        if threshold.get("warnings"):
            report.append("\n## Threshold Warnings\n")
            for warning in threshold["warnings"]:
                report.append(f"- {warning}")

        diagnostics = payload["diagnostics"]
        report.append("\n## Diagnostics\n")
        if diagnostics.get("confusion_analysis"):
            for key, value in diagnostics["confusion_analysis"].items():
                report.append(f"- **{key.replace('_', ' ').title()}**: {value}")
        if diagnostics.get("warnings"):
            for warning in diagnostics["warnings"]:
                report.append(f"- Warning: {warning}")

        recommendations = payload.get("recommendations", [])
        if recommendations:
            report.append("\n## Recommendations\n")
            for item in recommendations:
                report.append(f"- {item}")

        report.append(f"\n---\n*Report generated by {self.__class__.__name__}*")
        return "\n".join(report)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _analyze_confusion_matrix(self, confusion: Any) -> Dict[str, Any]:
        array = np.asarray(confusion)
        if array.size == 0:
            return {"available": False}
        if array.ndim != 2:
            return {"available": False, "reason": "non_matrix_confusion"}
        total = int(array.sum())
        successes = int(np.trace(array))
        failures = max(total - successes, 0)
        analysis: Dict[str, Any] = {
            "available": True,
            "total_predictions": total,
            "correct_predictions": successes,
            "incorrect_predictions": failures,
        }
        if array.shape == (2, 2):
            tn, fp, fn, tp = array.ravel()
            analysis.update({
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            })
        return analysis

    def _binary_failure_modes(self, confusion: Any) -> Dict[str, int]:
        array = np.asarray(confusion)
        if array.shape != (2, 2):
            return {}
        tn, fp, fn, tp = array.ravel()
        return {
            "false_positive_count": int(fp),
            "false_negative_count": int(fn),
        }

    def _distribution(self, values: Sequence[Any]) -> Dict[str, int]:
        distribution: Dict[str, int] = {}
        for item in values:
            key = str(item)
            distribution[key] = distribution.get(key, 0) + 1
        return distribution

    def _infer_probability_shape(self, probabilities: Sequence[Any]) -> List[int]:
        first = probabilities[0]
        if self._is_numeric(first):
            return [len(probabilities)]
        if isinstance(first, np.ndarray):
            first = first.tolist()
        if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
            return [len(probabilities), len(first)]
        return [len(probabilities)]

    def _average_prediction_confidence(self, probabilities: Sequence[Any]) -> float:
        if not probabilities:
            return 0.0
        first = probabilities[0]
        if self._is_numeric(first):
            values = [float(item) for item in probabilities]
            confidences = [max(v, 1.0 - v) for v in values]
            return float(np.mean(confidences))
        rows = [np.asarray(row, dtype=float) for row in probabilities]
        maxima = [float(np.max(row)) for row in rows if row.size > 0]
        return float(np.mean(maxima)) if maxima else 0.0

    def _average_entropy(self, probabilities: Sequence[Any]) -> float:
        if not probabilities:
            return 0.0
        first = probabilities[0]
        if self._is_numeric(first):
            entropies = []
            for item in probabilities:
                p = min(max(float(item), 1e-12), 1 - 1e-12)
                entropies.append(-(p * math.log(p) + (1 - p) * math.log(1 - p)))
            return float(np.mean(entropies)) if entropies else 0.0
        entropies = []
        for row in probabilities:
            arr = np.asarray(row, dtype=float)
            if arr.size == 0:
                continue
            arr = np.clip(arr, 1e-12, None)
            total = arr.sum()
            if total <= 0:
                continue
            arr = arr / total
            entropies.append(float(-(arr * np.log(arr)).sum()))
        return float(np.mean(entropies)) if entropies else 0.0

    def _metric_direction(self, metric: str) -> str:
        name = str(metric).strip().lower()
        if name in _LOWER_IS_BETTER:
            return "lower"
        return "higher"

    def _is_binary_label_set(self, predicted_labels: Sequence[Any], ground_truths: Sequence[Any]) -> bool:
        values = set(predicted_labels) | set(ground_truths)
        return values.issubset({0, 1, False, True, "0", "1"}) and len(values) <= 2

    def _can_threshold_outputs(self, outputs: Sequence[Any], truths: Sequence[Any]) -> bool:
        if self.threshold is None:
            return False
        if not all(self._is_numeric(item) for item in outputs):
            return False
        values = set(truths)
        return values.issubset({0, 1, False, True, "0", "1"}) or (isinstance(self.classes, list) and len(self.classes) == 2)

    def _looks_like_probability_rows(self, outputs: Sequence[Any]) -> bool:
        first = outputs[0]
        if isinstance(first, np.ndarray):
            return first.ndim >= 1 and first.size > 1
        return isinstance(first, Sequence) and not isinstance(first, (str, bytes)) and len(first) > 1 and all(self._is_numeric(v) for v in first)

    def _ensure_non_string_sequence(self, value: Sequence[Any], field_name: str) -> List[Any]:
        if value is None or isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ValidationFailureError(
                rule_name=field_name,
                data=type(value).__name__,
                expected="Sequence",
            )
        return list(value)

    def _normalize_weight_mapping(self, value: Any, field_name: str) -> Dict[str, float]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ConfigLoadError(
                config_path=self.config_path,
                section=field_name,
                error_details=f"Expected mapping, got {type(value).__name__}.",
            )
        normalized: Dict[str, float] = {}
        for key, raw in value.items():
            if not self._is_numeric(raw):
                raise ConfigLoadError(
                    config_path=self.config_path,
                    section=field_name,
                    error_details=f"Metric weight '{key}' must be numeric.",
                )
            normalized[str(key)] = float(raw)
        return normalized

    def _coerce_optional_probability(self, value: Any, field_name: str) -> Optional[float]:
        if value is None:
            return None
        if not self._is_numeric(value):
            raise ConfigLoadError(
                config_path=self.config_path,
                section=field_name,
                error_details="Value must be numeric or null.",
            )
        numeric = float(value)
        if not 0.0 <= numeric <= 1.0:
            raise ConfigLoadError(
                config_path=self.config_path,
                section=field_name,
                error_details="Probability values must be in [0, 1].",
            )
        return numeric

    @staticmethod
    def _is_numeric(value: Any) -> bool:
        return isinstance(value, (int, float, np.floating, np.integer)) and not isinstance(value, bool)

    @staticmethod
    def _format_metric(value: Any) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, (int, float, np.floating, np.integer)):
            return f"{float(value):.4f}"
        return str(value)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Performance Evaluator ===\n")
    import sys
    qt_widgets = importlib.import_module("PyQt5.QtWidgets")
    app = qt_widgets.QApplication(sys.argv)

    performance = PerformanceEvaluator()
    logger.info(f"{performance}")
    print(f"\n* * * * * Phase 2 * * * * *\n")
    outputs = [1, 0, 1, 1]
    ground_truths = [1, 0, 0, 1]
    report = True

    results = performance.evaluate(outputs, ground_truths, report)
    printer.pretty("FINAL", results, "success" if results else "error")
    if 'report' in results:
        print(results['report'])
    print(f"\n* * * * * Phase 3 * * * * *\n")

    print("\n=== Successfully Ran Performance Evaluator ===\n")
