from __future__ import annotations

import json
import math
import sys
import importlib
import numpy as np

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from scipy.stats import f as f_dist
from scipy.stats import norm, shapiro, t as student_t
from sklearn.metrics import (confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)

from .evaluation_errors import (ConfigLoadError, MetricCalculationError, OperationalError,
                                ThresholdViolationError, ValidationFailureError)
from .config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Evaluators Calculations")
printer = PrettyPrinter


class EvaluatorsCalculations:
    """
    Centralized calculation service for evaluation, efficiency, safety, and
    statistical analytics.

    The class keeps backward-compatible helper names where practical while
    providing production-grade validation, configuration normalization, and
    structured calculation flows.
    """

    DEBT_WEIGHTS: Dict[str, float] = {
        "nested_control": 0.3,
        "nested_loop": 0.3,
        "duplicate_code": 0.4,
        "violation_of_law_of_demeter": 0.2,
        "security_risk": 0.5,
    }

    def __init__(self) -> None:
        self.config = load_global_config()
        self.config_path = str(self.config.get("__config_path__", "<config>"))

        self.avg_token_baseline = self._require_positive_number(
            self.config.get("avg_token_baseline", 50),
            "avg_token_baseline",
        )

        self.base_config = get_config_section("baselines")
        self.time_per_output = self._require_positive_number(
            self.base_config.get("time_per_output", 0.5),
            "baselines.time_per_output",
        )
        self.time_baseline = self._require_non_negative_number(
            self.base_config.get("time_baseline", 0.0),
            "baselines.time_baseline",
        )
        self.memory_baseline = self._require_positive_number(
            self.base_config.get("memory_baseline", 1024),
            "baselines.memory_baseline",
        )
        self.flops = self._require_positive_number(
            self.base_config.get("flops", 1_000_000),
            "baselines.flops",
        )
        self.zero_division = self.base_config.get("zero_division", "warn")

        self.perform_config = get_config_section("performance_evaluator")
        self.average = str(self.perform_config.get("average", "macro"))
        self.classes = self.perform_config.get("classes")
        self.performance_metric_weights = self._normalize_weight_mapping(
            self.perform_config.get("metric_weights", self.perform_config.get("weights", {})),
            "performance_evaluator.metric_weights",
        )
        self.metric_params = self.perform_config.get("metric_params", {})
        self.enable_composite_score = bool(self.perform_config.get("enable_composite_score", True))
        self.performance_threshold = self.perform_config.get("threshold")
        self.custom_metrics_config = self.perform_config.get("custom_metrics", [])

        self.efficiency_config = get_config_section("efficiency_evaluator")
        self.efficiency_weights = self._normalize_weight_mapping(
            self.efficiency_config.get("efficiency_weights", {}),
            "efficiency_evaluator.efficiency_weights",
        )
        self.linguistic_weights = self._normalize_weight_mapping(
            self.efficiency_config.get("linguistic_weights", {}),
            "efficiency_evaluator.linguistic_weights",
        )
        self.current_flops = self._require_positive_number(
            self.efficiency_config.get("current_flops", self.flops),
            "efficiency_evaluator.current_flops",
        )

        self.rue_config = get_config_section("resource_utilization_evaluator")
        self.thresholds = self._normalize_threshold_mapping(
            self.rue_config.get("thresholds", {}),
            "resource_utilization_evaluator.thresholds",
        )
        self.resource_weights = self._normalize_weight_mapping(
            self.rue_config.get("weights", {}),
            "resource_utilization_evaluator.weights",
        )

        self.statistic_config = get_config_section("statistical_evaluator")
        self.significance_threshold = self._coerce_probability(
            self.statistic_config.get("alpha", 0.05),
            "statistical_evaluator.alpha",
            inclusive_zero=False,
            inclusive_one=False,
        )
        self.min_sample_size = self._require_positive_integer(
            self.statistic_config.get("min_sample_size", 10),
            "statistical_evaluator.min_sample_size",
        )
        self.confidence_level = self._coerce_probability(
            self.statistic_config.get("confidence_level", 0.95),
            "statistical_evaluator.confidence_level",
            inclusive_zero=False,
            inclusive_one=False,
        )

        self.eval_config = get_config_section("safety_evaluator")
        self.risk_categories = self._normalize_string_list(
            self.eval_config.get(
                "risk_categories",
                [
                    "collision",
                    "pinch_point",
                    "crush_hazard",
                    "electrical",
                    "environmental",
                    "control_failure",
                ],
            ),
            "safety_evaluator.risk_categories",
        )

        self.constraints: Dict[str, Any] = {
            "safety_tolerance": 0.0,
            "ethical_patterns": [],
        }
        self.violation_history: List[Dict[str, float]] = []
        self._lagrangian_multipliers: Dict[str, float] = {"safety": 0.1, "ethics": 0.2}
        self.hazard_data: List[Dict[str, float]] = []

        self.tokenizer: Any = None
        self.resource: Any = None
        self.nlp_engine: Any = None

        logger.info("Evaluators Calculations successfully initialized")

    # ------------------------------------------------------------------
    # Public integration helpers
    # ------------------------------------------------------------------

    def attach_dependencies(
        self,
        tokenizer: Any = None,
        resource: Any = None,
        nlp_engine: Any = None,
    ) -> None:
        """Attach optional runtime dependencies used by advanced calculations."""
        self.tokenizer = tokenizer if tokenizer is not None else self.tokenizer
        self.resource = resource if resource is not None else self.resource
        self.nlp_engine = nlp_engine if nlp_engine is not None else self.nlp_engine

    def set_current_flops(self, current_flops: float) -> float:
        self.current_flops = self._require_positive_number(current_flops, "current_flops")
        return self.current_flops

    def set_constraints(self, constraints: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(constraints, Mapping):
            raise ValidationFailureError("constraints_mapping", type(constraints).__name__, "mapping")

        updated = dict(self.constraints)
        if "safety_tolerance" in constraints:
            updated["safety_tolerance"] = self._coerce_probability(
                constraints["safety_tolerance"],
                "constraints.safety_tolerance",
                inclusive_zero=True,
                inclusive_one=True,
            )
        if "ethical_patterns" in constraints:
            updated["ethical_patterns"] = self._normalize_string_list(
                constraints["ethical_patterns"],
                "constraints.ethical_patterns",
            )

        self.constraints = updated
        return dict(self.constraints)

    def add_hazard_records(self, hazard_records: Sequence[Mapping[str, Any]]) -> int:
        if not isinstance(hazard_records, Sequence):
            raise ValidationFailureError("hazard_records", type(hazard_records).__name__, "sequence of mappings")

        appended = 0
        for record in hazard_records:
            if not isinstance(record, Mapping):
                raise ValidationFailureError("hazard_record_item", type(record).__name__, "mapping")
            normalized = {}
            for category in self.risk_categories:
                if category in record:
                    normalized[category] = self._require_non_negative_number(record[category], f"hazard[{category}]")
            if normalized:
                self.hazard_data.append(normalized)
                appended += 1
        return appended

    # ------------------------------------------------------------------
    # Efficiency and linguistic analysis
    # ------------------------------------------------------------------

    def calculate_efficiency_metrics(
        self,
        outputs: Sequence[Any],
        current_flops: Optional[float] = None,
    ) -> Dict[str, Any]:
        validated_outputs = self._ensure_non_string_sequence(outputs, "outputs")
        if current_flops is not None:
            self.set_current_flops(current_flops)

        temporal = self._calculate_temporal(validated_outputs)
        spatial = self._calculate_spatial(validated_outputs)
        computational = self._calculate_computational()
        token_efficiency = self._calculate_token_efficiency(validated_outputs)
        linguistic_complexity = self._calculate_linguistic_complexity(validated_outputs)
        linguistic_score = self._calculate_linguistic_score(linguistic_complexity)

        efficiency_metrics = {
            "temporal": temporal,
            "spatial": spatial,
            "computational": computational,
            "token_efficiency": token_efficiency,
            "linguistic_score": linguistic_score,
        }

        return {
            "metrics": efficiency_metrics,
            "linguistic_complexity": linguistic_complexity,
            "composite_score": self._calculate_efficiency_composite_score(efficiency_metrics),
        }

    def _calculate_linguistic_complexity(self, outputs: Sequence[Any]) -> Dict[str, float]:
        """Analyze text complexity using an attached NLP engine when available."""
        complexity = {
            "avg_sentence_length": 0.0,
            "pos_diversity": 0.0,
            "dependency_complexity": 0.0,
            "entity_density": 0.0,
        }

        texts = [text for text in (self._extract_text_from_output(item) for item in outputs) if text]
        if not texts:
            return complexity

        if self.nlp_engine is None:
            total_tokens = sum(max(1, len(text.split())) for text in texts)
            total_sentences = sum(max(1, self._estimate_sentence_count(text)) for text in texts)
            complexity["avg_sentence_length"] = total_tokens / max(total_sentences, 1)
            return complexity

        total_sentences = 0
        total_tokens = 0
        total_dependencies = 0
        total_entities = 0
        pos_counts: Dict[str, int] = defaultdict(int)

        for text in texts:
            try:
                tokens = self.nlp_engine.process_text(text)
                token_list = list(tokens) if not isinstance(tokens, list) else tokens
                sentence_count = max(1, self._estimate_sentence_count(text))
                dependencies = self._safe_nlp_call("apply_dependency_rules", token_list, default=[])
                entities = self._safe_nlp_call("resolve_coreferences", [token_list], default=[])

                total_sentences += sentence_count
                total_tokens += len(token_list)
                total_dependencies += len(dependencies)
                total_entities += len(entities)

                for token in token_list:
                    pos_tag = getattr(token, "pos", getattr(token, "pos_", None))
                    if pos_tag:
                        pos_counts[str(pos_tag)] += 1
            except Exception as exc:
                logger.warning("Complexity analysis failed for output segment: %s", exc)

        if total_sentences > 0:
            complexity["avg_sentence_length"] = total_tokens / total_sentences
        if total_tokens > 0:
            complexity["pos_diversity"] = len(pos_counts) / total_tokens if pos_counts else 0.0
            complexity["dependency_complexity"] = total_dependencies / total_tokens
            complexity["entity_density"] = total_entities / total_tokens

        return complexity

    def _calculate_token_efficiency(self, outputs: Sequence[Any]) -> float:
        """Token efficiency based on tokenizer counts or conservative text fallback."""
        if not outputs:
            raise MetricCalculationError("token_efficiency", outputs, "outputs must not be empty")

        total_tokens = 0
        for output in outputs:
            token_count = self._estimate_token_count(output)
            total_tokens += max(token_count, 0)

        avg_tokens = total_tokens / len(outputs)
        return self.avg_token_baseline / (avg_tokens + sys.float_info.epsilon)

    def _calculate_temporal(self, outputs: Sequence[Any]) -> float:
        """Response latency efficiency using generation_time metadata where present."""
        total_time = 0.0
        valid_outputs = 0

        for output in outputs:
            if isinstance(output, Mapping) and "generation_time" in output:
                total_time += self._require_non_negative_number(
                    output["generation_time"],
                    "output.generation_time",
                )
                valid_outputs += 1

        if valid_outputs == 0:
            logger.warning("Temporal efficiency skipped: no generation_time metadata found")
            return 0.0

        baseline = self.time_baseline if self.time_baseline > 0 else self.time_per_output * valid_outputs
        return baseline / (total_time + sys.float_info.epsilon)

    def _calculate_spatial(self, outputs: Sequence[Any]) -> float:
        """Memory footprint efficiency using serialized payload sizes or explicit metadata."""
        total_size = 0

        for output in outputs:
            if isinstance(output, Mapping) and "serialized_size" in output:
                total_size += int(self._require_non_negative_number(output["serialized_size"], "output.serialized_size"))
                continue

            try:
                if isinstance(output, bytes):
                    serialized = output
                elif isinstance(output, str):
                    serialized = output.encode("utf-8")
                else:
                    serialized = json.dumps(output, sort_keys=True, default=str).encode("utf-8")
                total_size += len(serialized)
            except (TypeError, ValueError, OverflowError):
                total_size += sys.getsizeof(output)

        return self.memory_baseline / (total_size + sys.float_info.epsilon)

    def _calculate_computational(self) -> float:
        """Compute FLOP efficiency relative to the configured baseline."""
        baseline = self.flops
        current = self.current_flops or baseline
        return baseline / (current + sys.float_info.epsilon)

    def _calculate_efficiency_composite_score(self, results: Mapping[str, float]) -> float:
        if not self.efficiency_weights:
            return 0.0

        score = 0.0
        total_weight = 0.0
        for metric, weight in self.efficiency_weights.items():
            if metric in results:
                score += weight * float(results[metric])
                total_weight += weight
        return score / total_weight if total_weight > 0 else 0.0

    def _calculate_composite_score(self, results: Mapping[str, Any]) -> float:
        """Backward-compatible composite score router.

        Routes to the most appropriate configured weighting scheme depending on
        the metrics available in the supplied result mapping.
        """
        if not isinstance(results, Mapping):
            raise ValidationFailureError("composite_score.results", type(results).__name__, "mapping")

        if any(key in results for key in ("temporal", "spatial", "computational", "token_efficiency")):
            numeric_results = {key: float(value) for key, value in results.items() if isinstance(value, (int, float, np.floating))}
            return self._calculate_efficiency_composite_score(numeric_results)

        if any(key in results for key in ("accuracy", "precision", "recall", "f1")):
            return self._calculate_performance_composite_score(results)

        if all(key in results for key in ("accuracy", "safety_score", "resource_usage")):
            weights = self.performance_metric_weights
            total = (
                weights.get("accuracy", 0.6) * float(results["accuracy"]) +
                weights.get("safety", 0.3) * float(results["safety_score"]) +
                weights.get("efficiency", 0.1) * (1.0 - float(results["resource_usage"]))
            )
            return float(total)

        return 0.0

    def _calculate_linguistic_score(self, complexity: Mapping[str, float]) -> float:
        if not self.linguistic_weights:
            return 0.0

        normalized = {
            "syntactic_complexity": self._bounded_inverse_scale(complexity.get("avg_sentence_length", 0.0), reference=20.0),
            "semantic_density": self._bounded_ratio(complexity.get("entity_density", 0.0), reference=0.2),
            "structural_variety": self._bounded_ratio(complexity.get("pos_diversity", 0.0), reference=0.5),
        }
        total = 0.0
        total_weight = 0.0
        for metric, weight in self.linguistic_weights.items():
            if metric in normalized:
                total += normalized[metric] * weight
                total_weight += weight
        return total / total_weight if total_weight > 0 else 0.0

    # ------------------------------------------------------------------
    # Performance and classification metrics
    # ------------------------------------------------------------------

    def calculate_classification_metrics(
        self,
        outputs: Sequence[Any],
        truths: Sequence[Any],
        probabilities: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        labels = self._coerce_prediction_labels(outputs)
        truths_array = self._coerce_truth_labels(truths)

        if labels.size != truths_array.size:
            raise ValidationFailureError(
                "prediction_length_alignment",
                int(labels.size),
                int(truths_array.size),
            )
        if truths_array.size == 0:
            raise MetricCalculationError("classification_metrics", truths, "truth labels must not be empty")

        metrics = {
            "accuracy": self._calculate_accuracy(labels, truths_array),
            "precision": self._calculate_precision(labels, truths_array),
            "recall": self._calculate_recall(labels, truths_array),
            "f1": self._calculate_f1(labels, truths_array),
            "confusion_matrix": self._calculate_confusion_matrix(labels, truths_array),
        }

        probability_matrix = self._resolve_probability_inputs(outputs, truths_array, probabilities)
        if probability_matrix is not None:
            metrics["roc_auc"] = self._calculate_roc_auc(probability_matrix, truths_array)
            metrics["log_loss"] = self._calculate_log_loss(probability_matrix, truths_array)

        custom_metrics, custom_metric_errors = self._calculate_custom_metrics(
            truths_array,
            labels,
            probability_matrix,
        )
        if custom_metrics:
            metrics["custom_metrics"] = custom_metrics
        if custom_metric_errors:
            metrics["custom_metric_errors"] = custom_metric_errors

        if self.enable_composite_score:
            metrics["composite_score"] = self._calculate_performance_composite_score(metrics)
            if self.performance_threshold is not None and metrics["composite_score"] < float(self.performance_threshold):
                logger.warning(
                    "Performance composite score %.4f is below configured threshold %.4f",
                    metrics["composite_score"],
                    float(self.performance_threshold),
                )

        return metrics

    def _calculate_accuracy(self, outputs: np.ndarray, truths: np.ndarray) -> float:
        correct = np.sum(outputs == truths)
        return float(correct / truths.size)

    def _calculate_precision(self, outputs: np.ndarray, truths: np.ndarray) -> float:
        return float(
            precision_score(
                truths,
                outputs,
                average=self.average,
                zero_division=self.zero_division,
            )
        )

    def _calculate_recall(self, outputs: np.ndarray, truths: np.ndarray) -> float:
        return float(
            recall_score(
                truths,
                outputs,
                average=self.average,
                zero_division=self.zero_division,
            )
        )

    def _calculate_f1(self, outputs: np.ndarray, truths: np.ndarray) -> float:
        return float(
            f1_score(
                truths,
                outputs,
                average=self.average,
                zero_division=self.zero_division,
            )
        )

    def _calculate_confusion_matrix(self, outputs: np.ndarray, truths: np.ndarray) -> List[List[int]]:
        labels = self.classes if self.classes is not None else None
        return confusion_matrix(truths, outputs, labels=labels).tolist()

    def _calculate_roc_auc(self, outputs: np.ndarray, truths: np.ndarray) -> float:
        try:
            unique_classes = np.unique(truths)
            roc_params = dict(self.metric_params.get("roc_auc", {}))
            if unique_classes.size > 2:
                multi_class = roc_params.get("multi_class", "ovo")
                return float(roc_auc_score(truths, outputs, multi_class=multi_class))
            if outputs.ndim == 2:
                positive_scores = outputs[:, 1] if outputs.shape[1] > 1 else outputs[:, 0]
                return float(roc_auc_score(truths, positive_scores))
            return float(roc_auc_score(truths, outputs))
        except ValueError as exc:
            logger.warning("ROC AUC calculation failed: %s", exc)
            return 0.0

    def _calculate_log_loss(self, outputs: np.ndarray, truths: np.ndarray) -> float:
        try:
            return float(log_loss(truths, outputs))
        except (ValueError, TypeError) as exc:
            logger.warning("Log loss calculation failed: %s", exc)
            return float("inf")

    def _calculate_adaptive_weights(self, ground_truths: Sequence[int]) -> np.ndarray:
        truth_array = np.asarray(ground_truths, dtype=int)
        if truth_array.size == 0:
            raise MetricCalculationError("adaptive_class_weights", ground_truths, "ground_truths must not be empty")
        class_counts = np.bincount(truth_array)
        non_zero = np.where(class_counts > 0, class_counts, 1)
        weights = 1.0 / non_zero
        return weights / weights.sum()

    def _calculate_performance_composite_score(self, results: Mapping[str, Any]) -> float:
        if not self.performance_metric_weights:
            return 0.0

        total = 0.0
        total_weight = 0.0
        for metric, weight in self.performance_metric_weights.items():
            value = results.get(metric)
            if isinstance(value, (int, float, np.floating)):
                total += float(value) * weight
                total_weight += weight
        return total / total_weight if total_weight > 0 else 0.0

    # ------------------------------------------------------------------
    # Resource utilization calculations
    # ------------------------------------------------------------------

    def calculate_resource_efficiency(self, metrics: Mapping[str, Any]) -> Dict[str, Any]:
        validated_metrics = self._validate_resource_metrics(metrics)
        scores = self._calculate_scores(validated_metrics)
        weighted_score = self._calculate_resource_weighted_score(scores)

        threshold_violations = {
            name: validated_metrics[name]
            for name, threshold in self.thresholds.items()
            if validated_metrics[name] > threshold
        }
        violation_events = []
        for metric_name, observed_value in threshold_violations.items():
            violation = ThresholdViolationError(metric_name, float(observed_value), float(self.thresholds[metric_name]))
            violation_events.append(violation.to_audit_dict())

        return {
            "raw_metrics": validated_metrics,
            "scores": scores,
            "weighted_score": weighted_score,
            "threshold_violations": threshold_violations,
            "violation_events": violation_events,
        }

    def _calculate_scores(self, metrics: Mapping[str, float]) -> Dict[str, float]:
        """Calculate normalized resource efficiency scores."""
        return {
            metric: max(0.0, 1.0 - (float(metrics[metric]) / float(self.thresholds[metric])))
            for metric in self.thresholds
        }

    def _calculate_resource_weighted_score(self, scores: Mapping[str, float]) -> float:
        if not self.resource_weights:
            return float(np.mean(list(scores.values()))) if scores else 0.0

        total = 0.0
        total_weight = 0.0
        for metric, weight in self.resource_weights.items():
            if metric in scores:
                total += scores[metric] * weight
                total_weight += weight
        return total / total_weight if total_weight > 0 else 0.0

    # ------------------------------------------------------------------
    # Statistical calculations
    # ------------------------------------------------------------------

    def calculate_statistical_analysis(
        self,
        datasets: Mapping[str, Sequence[float]],
        confidence_level: Optional[float] = None,
    ) -> Dict[str, Any]:
        normalized = self._normalize_datasets(datasets)
        confidence = (
            self._coerce_probability(confidence_level, "confidence_level", inclusive_zero=False, inclusive_one=False)
            if confidence_level is not None
            else self.confidence_level
        )

        descriptive = self._calculate_descriptive_stats(normalized)
        if not descriptive:
            raise MetricCalculationError(
                "descriptive_statistics",
                list(normalized.keys()),
                f"all datasets are smaller than min_sample_size={self.min_sample_size}",
            )

        normality = self._calculate_normality_tests(normalized)
        confidence_intervals = {
            name: self._calculate_confidence_interval(values, confidence)
            for name, values in normalized.items()
            if len(values) >= 2
        }

        return {
            "descriptive_stats": descriptive,
            "normality": normality,
            "pairwise_tests": self._calculate_pairwise_tests(normalized, confidence),
            "effect_sizes": self._calculate_all_effect_sizes(normalized),
            "power_analysis": self._calculate_power_analysis(normalized),
            "confidence_intervals": confidence_intervals,
            "alpha": self.significance_threshold,
            "confidence_level": confidence,
        }

    def _calculate_descriptive_stats(self, datasets: Mapping[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        for name, data in datasets.items():
            if len(data) < self.min_sample_size:
                logger.warning("Insufficient sample size for %s: %d", name, len(data))
                continue

            stats[name] = {
                "mean": float(np.mean(data)),
                "median": float(np.median(data)),
                "std_dev": float(np.std(data, ddof=1)) if len(data) > 1 else 0.0,
                "variance": float(np.var(data, ddof=1)) if len(data) > 1 else 0.0,
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "sample_size": int(len(data)),
            }
        return stats

    def _calculate_normality_tests(self, datasets: Mapping[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
        normality_results: Dict[str, Dict[str, Any]] = {}
        for name, data in datasets.items():
            if len(data) < 3:
                normality_results[name] = {
                    "test": "shapiro",
                    "p_value": None,
                    "is_normal": None,
                    "status": "insufficient_samples",
                }
                continue
            if len(data) > 5000:
                data = data[:5000]

            statistic, p_value = shapiro(data)
            normality_results[name] = {
                "test": "shapiro",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "is_normal": bool(p_value >= self.significance_threshold),
                "status": "ok",
            }
        return normality_results

    def _calculate_all_effect_sizes(self, datasets: Mapping[str, np.ndarray]) -> Dict[str, float]:
        effects: Dict[str, float] = {}
        names = list(datasets.keys())
        for index, first_name in enumerate(names):
            for second_name in names[index + 1 :]:
                key = f"{first_name}_vs_{second_name}"
                effects[key] = self.effect_size(datasets[first_name], datasets[second_name])
        return effects

    def _calculate_power_analysis(self, datasets: Mapping[str, np.ndarray]) -> Dict[str, float]:
        power: Dict[str, float] = {}
        zero_baseline = np.array([0.0, 0.0], dtype=float)
        for name, data in datasets.items():
            if len(data) < 2:
                power[name] = 0.0
                continue
            effect = self.effect_size(data, zero_baseline)
            power[name] = self._calculate_power(len(data), effect)
        return power

    def _calculate_power(self, n: int, effect_size: float) -> float:
        alpha = self.significance_threshold
        z_alpha = norm.ppf(1 - alpha / 2)
        z_power = effect_size * math.sqrt(max(n, 1)) - z_alpha
        return float(norm.cdf(z_power))

    def effect_size(self, sample_a: Sequence[float], sample_b: Sequence[float]) -> float:
        a = self._coerce_numeric_array(sample_a, "sample_a")
        b = self._coerce_numeric_array(sample_b, "sample_b")
        if len(a) < 2 or len(b) < 2:
            return 0.0

        mean_a = float(np.mean(a))
        mean_b = float(np.mean(b))
        var_a = float(np.var(a, ddof=1))
        var_b = float(np.var(b, ddof=1))
        pooled_std = math.sqrt((var_a + var_b) / 2)
        return 0.0 if pooled_std == 0 else (mean_a - mean_b) / pooled_std

    def _calculate_confidence_interval(self, data: Sequence[float], confidence_level: float) -> Dict[str, float]:
        sample = self._coerce_numeric_array(data, "confidence_interval_data")
        if len(sample) < 2:
            raise MetricCalculationError("confidence_interval", list(sample), "at least two observations are required")

        mean = float(np.mean(sample))
        std = float(np.std(sample, ddof=1))
        standard_error = std / math.sqrt(len(sample))
        alpha = 1 - confidence_level
        critical = float(student_t.ppf(1 - alpha / 2, df=len(sample) - 1))
        margin = critical * standard_error

        return {
            "mean": mean,
            "lower": mean - margin,
            "upper": mean + margin,
            "margin": margin,
        }

    def _calculate_pairwise_tests(
        self,
        datasets: Mapping[str, np.ndarray],
        confidence_level: float,
    ) -> Dict[str, Dict[str, Any]]:
        comparisons: Dict[str, Dict[str, Any]] = {}
        names = list(datasets.keys())
        for index, first_name in enumerate(names):
            for second_name in names[index + 1 :]:
                sample_a = datasets[first_name]
                sample_b = datasets[second_name]
                comparison_key = f"{first_name}_vs_{second_name}"
                comparisons[comparison_key] = {
                    "t_test": self._welch_t_test(sample_a, sample_b),
                    "variance_test": self._variance_ratio_test(sample_a, sample_b, confidence_level),
                    "effect_size": self.effect_size(sample_a, sample_b),
                }
        return comparisons

    def _welch_t_test(self, sample_a: np.ndarray, sample_b: np.ndarray) -> Dict[str, Any]:
        if len(sample_a) < 2 or len(sample_b) < 2:
            return {
                "statistic": None,
                "p_value": None,
                "degrees_of_freedom": None,
                "significant": None,
                "status": "insufficient_samples",
            }

        mean_a = float(np.mean(sample_a))
        mean_b = float(np.mean(sample_b))
        var_a = float(np.var(sample_a, ddof=1))
        var_b = float(np.var(sample_b, ddof=1))
        n_a = len(sample_a)
        n_b = len(sample_b)

        denominator = math.sqrt((var_a / n_a) + (var_b / n_b))
        if denominator == 0:
            return {
                "statistic": 0.0,
                "p_value": 1.0,
                "degrees_of_freedom": None,
                "significant": False,
                "status": "zero_variance",
            }

        statistic = (mean_a - mean_b) / denominator
        numerator = ((var_a / n_a) + (var_b / n_b)) ** 2
        denominator_df = 0.0
        if n_a > 1:
            denominator_df += ((var_a / n_a) ** 2) / (n_a - 1)
        if n_b > 1:
            denominator_df += ((var_b / n_b) ** 2) / (n_b - 1)
        degrees_of_freedom = numerator / denominator_df if denominator_df > 0 else None
        p_value = (
            float(2 * student_t.sf(abs(statistic), df=degrees_of_freedom))
            if degrees_of_freedom is not None
            else 1.0
        )

        return {
            "statistic": float(statistic),
            "p_value": p_value,
            "degrees_of_freedom": float(degrees_of_freedom) if degrees_of_freedom is not None else None,
            "significant": bool(p_value < self.significance_threshold),
            "status": "ok",
        }

    def _variance_ratio_test(
        self,
        sample_a: np.ndarray,
        sample_b: np.ndarray,
        confidence_level: float,
    ) -> Dict[str, Any]:
        if len(sample_a) < 2 or len(sample_b) < 2:
            return {
                "f_statistic": None,
                "p_value": None,
                "variance_equal": None,
                "status": "insufficient_samples",
            }

        var_a = float(np.var(sample_a, ddof=1))
        var_b = float(np.var(sample_b, ddof=1))
        if var_a == 0 and var_b == 0:
            return {
                "f_statistic": 1.0,
                "p_value": 1.0,
                "variance_equal": True,
                "status": "zero_variance",
            }
        if var_a == 0 or var_b == 0:
            return {
                "f_statistic": float("inf"),
                "p_value": 0.0,
                "variance_equal": False,
                "status": "degenerate_variance",
            }

        f_statistic = var_a / var_b
        df_a = len(sample_a) - 1
        df_b = len(sample_b) - 1
        lower_tail = f_dist.cdf(f_statistic, df_a, df_b)
        p_value = float(2 * min(lower_tail, 1 - lower_tail))

        alpha = 1 - confidence_level
        variance_equal = p_value >= alpha
        return {
            "f_statistic": float(f_statistic),
            "p_value": p_value,
            "variance_equal": bool(variance_equal),
            "status": "ok",
        }

    # ------------------------------------------------------------------
    # Safety and risk calculations
    # ------------------------------------------------------------------

    def _calculate_risk_distribution(self) -> Dict[str, float]:
        distribution = {category: 0.0 for category in self.risk_categories}
        total_weight = 0.0

        for hazard in self.hazard_data:
            for category in self.risk_categories:
                if category in hazard:
                    value = self._require_non_negative_number(hazard[category], f"hazard[{category}]")
                    distribution[category] += value
                    total_weight += value

        if total_weight > 0:
            for category in distribution:
                distribution[category] /= total_weight

        return distribution

    def calculate_diagnostic_coverage(self) -> Dict[str, Any]:
        if not self.hazard_data:
            logger.warning("No hazard data available for diagnostic coverage calculation")
            return {"coverage": 0.0, "details": {}, "status": "No data"}

        coverage_by_category = {
            category: {"detected": 0, "total": 0}
            for category in self.risk_categories
        }

        for hazard in self.hazard_data:
            for category in self.risk_categories:
                if category in hazard:
                    coverage_by_category[category]["total"] += 1
                    if float(hazard[category]) >= 1.0:
                        coverage_by_category[category]["detected"] += 1

        results = {}
        total_detected = 0
        total_possible = 0
        for category, stats in coverage_by_category.items():
            total = stats["total"]
            detected = stats["detected"]
            coverage = detected / total if total > 0 else 0.0
            results[category] = round(coverage, 3)
            total_detected += detected
            total_possible += total

        overall_coverage = total_detected / total_possible if total_possible > 0 else 0.0
        return {
            "coverage": round(overall_coverage, 3),
            "details": results,
            "status": "OK" if overall_coverage >= 0.85 else "Insufficient",
            "risk_distribution": self._calculate_risk_distribution(),
        }

    # ------------------------------------------------------------------
    # Technical debt and remediation prioritization
    # ------------------------------------------------------------------

    def calculate_debt(self, issues: Sequence[Mapping[str, Any]]) -> float:
        validated_issues = self._validate_issue_list(issues)
        return float(
            sum(
                issue["severity"] * self.DEBT_WEIGHTS.get(issue["type"], 0.1)
                for issue in validated_issues
            )
        )

    def prioritize_remediation(self, issues: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        validated_issues = self._validate_issue_list(issues)
        ranked = sorted(
            validated_issues,
            key=lambda issue: (
                issue["severity"] * self.DEBT_WEIGHTS.get(issue["type"], 0.1)
                / max(1.0, float(issue.get("estimated_fix_time", 1.0)))
            ),
            reverse=True,
        )
        return [dict(issue) for issue in ranked]

    # ------------------------------------------------------------------
    # Reward and constraint optimization
    # ------------------------------------------------------------------

    def calculate_reward(self, state: Mapping[str, Any], action: Mapping[str, Any], outcome: Mapping[str, Any]) -> float:
        if not isinstance(action, Mapping) or not isinstance(outcome, Mapping):
            raise ValidationFailureError("reward_inputs", "non-mapping", "mapping action and outcome")

        base_reward = float(outcome.get("performance", 0.0))
        safety_penalty = self._calculate_safety_violation(outcome)
        ethics_penalty = self._calculate_ethical_violation(action)

        constrained_reward = (
            base_reward
            - self._lagrangian_multipliers["safety"] * safety_penalty
            - self._lagrangian_multipliers["ethics"] * ethics_penalty
        )

        self.violation_history.append(
            {
                "safety_penalty": float(safety_penalty),
                "ethics_penalty": float(ethics_penalty),
            }
        )
        self._update_multipliers(safety_penalty, ethics_penalty)
        return float(constrained_reward)

    def _calculate_safety_violation(self, outcome: Mapping[str, Any]) -> float:
        hazard_prob = self._coerce_probability(
            outcome.get("hazard_prob", 0.0),
            "outcome.hazard_prob",
            inclusive_zero=True,
            inclusive_one=True,
            error_mode="runtime",
        )
        return max(0.0, hazard_prob - float(self.constraints.get("safety_tolerance", 0.0)))

    def _calculate_ethical_violation(self, action: Mapping[str, Any]) -> float:
        decision_path = action.get("decision_path", [])
        if isinstance(decision_path, str):
            path_tokens = [decision_path]
        elif isinstance(decision_path, Sequence):
            path_tokens = [str(item) for item in decision_path]
        else:
            raise ValidationFailureError("action.decision_path", type(decision_path).__name__, "string or sequence")

        patterns = self.constraints.get("ethical_patterns", [])
        return float(sum(1 for pattern in patterns if pattern in path_tokens))

    def _update_multipliers(self, safety_viol: float, ethics_viol: float) -> None:
        self._lagrangian_multipliers["safety"] *= 1 + max(0.0, float(safety_viol))
        self._lagrangian_multipliers["ethics"] *= 1 + max(0.0, float(ethics_viol))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_probability_inputs(
        self,
        outputs: Sequence[Any],
        truths: np.ndarray,
        probabilities: Optional[Sequence[Any]],
    ) -> Optional[np.ndarray]:
        raw = probabilities if probabilities is not None else outputs
        try:
            matrix = np.asarray(raw, dtype=float)
        except (TypeError, ValueError):
            return None

        if matrix.ndim == 1:
            if np.unique(truths).size > 2:
                return None
            return np.column_stack([1.0 - matrix, matrix])

        if matrix.ndim == 2:
            return matrix

        return None

    def _calculate_custom_metrics(
        self,
        truths: np.ndarray,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray],
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        metric_results: Dict[str, float] = {}
        errors: List[Dict[str, Any]] = []

        for metric_spec in self.custom_metrics_config:
            if not isinstance(metric_spec, Mapping):
                continue
            metric_name = str(metric_spec.get("name", "custom_metric"))
            function_path = metric_spec.get("function")
            if not function_path:
                continue

            try:
                metric_callable = self._resolve_callable(function_path)
                if probabilities is not None and metric_name in {"roc_auc", "log_loss"}:
                    value = metric_callable(truths, probabilities)
                else:
                    value = metric_callable(truths, labels)
                metric_results[metric_name] = float(value)
            except Exception as exc:
                error = MetricCalculationError(metric_name, {"truths": truths.tolist(), "labels": labels.tolist()}, str(exc))
                logger.warning("Custom metric '%s' failed: %s", metric_name, exc)
                errors.append(error.to_audit_dict())

        return metric_results, errors

    def _resolve_callable(self, function_path: str):
        if not isinstance(function_path, str) or "." not in function_path:
            raise OperationalError("Invalid callable path for custom metric", {"function_path": function_path})
        module_name, attribute_name = function_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, attribute_name)

    def _validate_resource_metrics(self, metrics: Mapping[str, Any]) -> Dict[str, float]:
        if not isinstance(metrics, Mapping):
            raise ValidationFailureError("resource_metrics", type(metrics).__name__, "mapping")

        validated: Dict[str, float] = {}
        for metric_name in self.thresholds:
            if metric_name not in metrics:
                raise ValidationFailureError(f"resource_metrics.{metric_name}", "missing", "present")
            validated[metric_name] = self._require_non_negative_number(metrics[metric_name], f"resource_metrics.{metric_name}")
        return validated

    def _normalize_datasets(self, datasets: Mapping[str, Sequence[float]]) -> Dict[str, np.ndarray]:
        if not isinstance(datasets, Mapping) or not datasets:
            raise ValidationFailureError("datasets", type(datasets).__name__, "non-empty mapping")
        normalized: Dict[str, np.ndarray] = {}
        for name, values in datasets.items():
            normalized[str(name)] = self._coerce_numeric_array(values, f"datasets.{name}")
        return normalized

    def _validate_issue_list(self, issues: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(issues, Sequence):
            raise ValidationFailureError("issues", type(issues).__name__, "sequence of mappings")

        validated: List[Dict[str, Any]] = []
        for issue in issues:
            if not isinstance(issue, Mapping):
                raise ValidationFailureError("issue_item", type(issue).__name__, "mapping")
            if "severity" not in issue or "type" not in issue:
                raise ValidationFailureError("issue_required_fields", list(issue.keys()), "severity and type")
            validated.append(
                {
                    **dict(issue),
                    "severity": self._require_non_negative_number(issue["severity"], "issue.severity"),
                    "type": str(issue["type"]).strip(),
                }
            )
        return validated

    def _coerce_prediction_labels(self, outputs: Sequence[Any]) -> np.ndarray:
        sequence = self._ensure_non_string_sequence(outputs, "outputs")
        if not sequence:
            raise MetricCalculationError("prediction_labels", outputs, "outputs must not be empty")

        first = sequence[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            matrix = np.asarray(sequence)
            if matrix.ndim == 1:
                return matrix.astype(int)
            return np.argmax(matrix, axis=1).astype(int)
        return np.asarray(sequence)

    def _coerce_truth_labels(self, truths: Sequence[Any]) -> np.ndarray:
        sequence = self._ensure_non_string_sequence(truths, "truths")
        return np.asarray(sequence)

    def _coerce_numeric_array(self, values: Sequence[float], field_name: str) -> np.ndarray:
        sequence = self._ensure_non_string_sequence(values, field_name)
        if len(sequence) == 0:   # changed from 'if not sequence'
            raise MetricCalculationError(field_name, values, "sequence must not be empty")
        try:
            array = np.asarray(sequence, dtype=float)
        except (TypeError, ValueError) as exc:
            raise MetricCalculationError(field_name, values, f"numeric conversion failed: {exc}") from exc
        if np.isnan(array).any() or np.isinf(array).any():
            raise ValidationFailureError(field_name, "NaN/Inf detected", "finite numeric values")
        return array

    def _ensure_non_string_sequence(self, value: Any, field_name: str) -> Sequence[Any]:
        if isinstance(value, (str, bytes)):
            raise ValidationFailureError(field_name, type(value).__name__, "sequence (non-string)")
        # Allow numpy arrays as sequences
        if not isinstance(value, (Sequence, np.ndarray)):
            raise ValidationFailureError(field_name, type(value).__name__, "sequence")
        return value

    def _extract_text_from_output(self, output: Any) -> str:
        if isinstance(output, str):
            return output.strip()
        if isinstance(output, Mapping):
            for key in ("text", "content", "output", "response"):
                value = output.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return str(output).strip()

    def _estimate_token_count(self, output: Any) -> int:
        text = self._extract_text_from_output(output)
        if not text:
            return 0

        tokenizer = self.tokenizer
        if tokenizer is not None:
            try:
                if hasattr(tokenizer, "encode"):
                    encoded = tokenizer.encode(text)
                    return len(encoded)
                if hasattr(tokenizer, "tokenize"):
                    tokens = tokenizer.tokenize(text)
                    return len(tokens)
            except Exception as exc:
                logger.warning("Tokenizer-based token count failed; falling back to whitespace split: %s", exc)

        return len(text.split())

    def _estimate_sentence_count(self, text: str) -> int:
        sentence_markers = sum(text.count(marker) for marker in (".", "!", "?"))
        return max(1, sentence_markers)

    def _safe_nlp_call(self, method_name: str, *args, default: Optional[Any] = None) -> Any:
        method = getattr(self.nlp_engine, method_name, None)
        if method is None:
            return default if default is not None else []
        try:
            return method(*args)
        except Exception as exc:
            logger.warning("NLP engine method '%s' failed: %s", method_name, exc)
            return default if default is not None else []

    def _normalize_weight_mapping(self, weights: Any, field_name: str) -> Dict[str, float]:
        if weights in (None, {}):
            return {}
        if not isinstance(weights, Mapping):
            raise ConfigLoadError(self.config_path, field_name, "weights must be a mapping")
        normalized = {}
        for key, value in weights.items():
            normalized[str(key)] = self._require_non_negative_number(value, f"{field_name}.{key}")
        return normalized

    def _normalize_threshold_mapping(self, thresholds: Any, field_name: str) -> Dict[str, float]:
        if not isinstance(thresholds, Mapping) or not thresholds:
            raise ConfigLoadError(self.config_path, field_name, "thresholds must be a non-empty mapping")
        normalized = {}
        for key, value in thresholds.items():
            normalized[str(key)] = self._require_positive_number(value, f"{field_name}.{key}")
        return normalized

    def _normalize_string_list(self, values: Any, field_name: str) -> List[str]:
        if values is None:
            return []
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ConfigLoadError(self.config_path, field_name, "value must be a sequence of strings")
        normalized = []
        for item in values:
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized

    def _require_positive_integer(self, value: Any, field_name: str) -> int:
        if not isinstance(value, int) or value <= 0:
            raise ConfigLoadError(self.config_path, field_name, "value must be a positive integer")
        return value

    def _require_positive_number(self, value: Any, field_name: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigLoadError(self.config_path, field_name, f"value must be numeric: {exc}") from exc
        if number <= 0:
            raise ConfigLoadError(self.config_path, field_name, "value must be greater than zero")
        return number

    @staticmethod
    def _require_non_negative_number(value: Any, field_name: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise MetricCalculationError(field_name, value, f"value must be numeric: {exc}") from exc
        if number < 0:
            raise ValidationFailureError(field_name, number, "non-negative number")
        return number

    def _coerce_probability(
        self,
        value: Any,
        field_name: str,
        inclusive_zero: bool = True,
        inclusive_one: bool = True,
        error_mode: str = "config",
    ) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            if error_mode == "config":
                raise ConfigLoadError(self.config_path, field_name, f"value must be numeric: {exc}") from exc
            raise ValidationFailureError(field_name, value, "numeric probability") from exc

        lower_ok = number >= 0.0 if inclusive_zero else number > 0.0
        upper_ok = number <= 1.0 if inclusive_one else number < 1.0
        if not (lower_ok and upper_ok):
            expectation = "0..1 inclusive"
            if not inclusive_zero and not inclusive_one:
                expectation = "0..1 exclusive"
            elif not inclusive_zero:
                expectation = ">0 and <=1"
            elif not inclusive_one:
                expectation = ">=0 and <1"
            if error_mode == "config":
                raise ConfigLoadError(self.config_path, field_name, f"value must be in {expectation}")
            raise ValidationFailureError(field_name, number, expectation)
        return number

    @staticmethod
    def _bounded_ratio(value: float, reference: float) -> float:
        if reference <= 0:
            return 0.0
        return float(max(0.0, min(value / reference, 1.0)))

    @staticmethod
    def _bounded_inverse_scale(value: float, reference: float) -> float:
        if reference <= 0:
            return 0.0
        return float(max(0.0, min(reference / max(value, sys.float_info.epsilon), 1.0)))


if __name__ == "__main__":
    print("\n=== Running Evaluators Calculations ===\n")
    calculations = EvaluatorsCalculations()

    example_outputs = [
        {"generation_time": 0.6, "serialized_size": 800, "text": "System response nominal."},
        {"generation_time": 0.5, "serialized_size": 900, "text": "Fallback path validated."},
    ]
    example_truths = [1, 0]
    example_predictions = [1, 0]
    example_probabilities = [[0.1, 0.9], [0.8, 0.2]]

    printer.pretty("Efficiency:", calculations.calculate_efficiency_metrics(example_outputs), "success")
    printer.pretty(
        "Classification:",
        calculations.calculate_classification_metrics(example_predictions, example_truths, example_probabilities),
        "success",
    )
    printer.pretty(
        "Statistics:",
        calculations.calculate_statistical_analysis(
            {"baseline": [0.71, 0.72, 0.75, 0.77, 0.76, 0.74, 0.73, 0.72, 0.75, 0.78],
             "candidate": [0.81, 0.82, 0.80, 0.83, 0.84, 0.81, 0.82, 0.85, 0.84, 0.83]}
        ),
        "success",
    )
    print("\n=== Successfully Ran Evaluators Calculations ===")
