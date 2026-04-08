
from __future__ import annotations

import json
import os
import sys
import time
import hashlib
import numpy as np
import torch
import yaml

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from PyQt5.QtCore import QSize, QBuffer

if sys.platform != "win32":
    import resource
else:
    resource = None

from .utils.config_loader import load_global_config, get_config_section
from .utils.evaluators_calculations import EvaluatorsCalculations
from .utils.evaluation_errors import (ComparisonError, ConfigLoadError, EvaluationError,
                                      MemoryAccessError, MetricCalculationError, OperationalError,
                                      ReportGenerationError, TemplateError, ValidationFailureError,
                                      VisualizationError)
from .modules.report import get_visualizer
from .evaluators_memory import EvaluatorsMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Efficiency Evaluator")
printer = PrettyPrinter


@dataclass(slots=True)
class EfficiencyHistoryRecord:
    """Compact historical record retained for trend analysis and visualization."""

    evaluated_at: str
    score: float
    temporal: float
    spatial: float
    computational: float
    token_efficiency: float
    runtime_seconds: float
    memory_delta_mb: float
    output_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BaselineComparison:
    """Comparison between current metrics and configured baselines."""

    name: str
    observed: float
    baseline: float
    ratio: float
    status: str
    direction: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EfficiencyEvaluationResult:
    """Structured output produced by the efficiency evaluator."""

    metadata: Dict[str, Any]
    metrics: Dict[str, float]
    linguistic_complexity: Dict[str, float]
    resource_observation: Dict[str, float]
    baseline_comparison: Dict[str, Dict[str, Any]]
    diagnostics: Dict[str, Any]
    recommendations: List[str]
    memory_entry_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": dict(self.metadata),
            "metrics": dict(self.metrics),
            "linguistic_complexity": dict(self.linguistic_complexity),
            "resource_observation": dict(self.resource_observation),
            "baseline_comparison": {
                key: dict(value) for key, value in self.baseline_comparison.items()
            },
            "diagnostics": dict(self.diagnostics),
            "recommendations": list(self.recommendations),
            "memory_entry_id": self.memory_entry_id,
        }


class EfficiencyEvaluator:
    """
    Production-grade efficiency evaluator for evaluator outputs and generated artifacts.

    Responsibilities
    ----------------
    - Calculate temporal, spatial, computational, token, and linguistic efficiency
    - Compare observed measurements against configured baselines
    - Produce structured reports suitable for dashboards, markdown, JSON, or YAML
    - Integrate with evaluator memory and the shared reporting/visualization pipeline
    - Surface failures through the shared evaluation error model

    Notes
    -----
    The evaluator intentionally does not auto-import local NLP or text-generation
    components at runtime. Instead, those dependencies can be attached explicitly
    through :meth:`attach_dependencies`, which keeps the module deterministic and
    avoids brittle runtime import behavior.
    """

    MODULE_VERSION = "2.0.0"

    def __init__(
        self,
        *,
        tokenizer: Any = None,
        nlp_engine: Any = None,
        text_encoder: Any = None,
        text_decoder: Any = None,
    ) -> None:
        self.config = load_global_config()
        self.config_path = str(self.config.get("__config_path__", "<config>"))

        self.section = get_config_section("efficiency_evaluator")
        if not isinstance(self.section, Mapping):
            raise ConfigLoadError(
                config_path=self.config_path,
                section="efficiency_evaluator",
                error_details="Section must be a mapping.",
            )

        self.store_results = bool(self.config.get("store_results", True))
        self.enable_historical = bool(self.config.get("enable_historical", True))

        self.energy_model_path = self._resolve_optional_path(self.section.get("energy_model"))
        self.report_template_path = self._resolve_optional_path(self.section.get("report_template"))
        self.recommendation_template_path = self._resolve_optional_path(
            self.section.get("recommendation_template_path")
        )

        self.complexity_metrics = bool(self.section.get("complexity_metrics", True))
        self.current_flops = self._require_positive_number(
            self.section.get("current_flops", 1_000_000),
            "efficiency_evaluator.current_flops",
        )

        self.efficiency_weights = self._normalize_weight_mapping(
            self.section.get("efficiency_weights", {}),
            "efficiency_evaluator.efficiency_weights",
        )
        self.linguistic_weights = self._normalize_weight_mapping(
            self.section.get("linguistic_weights", {}),
            "efficiency_evaluator.linguistic_weights",
        )
        self.baselines = self._normalize_baselines(
            self.section.get("baselines", {}),
            "efficiency_evaluator.baselines",
        )
        self.operational_modes = self.section.get("operational_modes", {})

        self.memory = EvaluatorsMemory()
        self.calculations = EvaluatorsCalculations()

        self.tokenizer = None
        self.nlp_engine = None
        self.text_encoder = None
        self.text_decoder = None
        self.attach_dependencies(
            tokenizer=tokenizer,
            nlp_engine=nlp_engine,
            text_encoder=text_encoder,
            text_decoder=text_decoder,
        )

        self.history: List[EfficiencyHistoryRecord] = []

        logger.info(
            "Efficiency Evaluator initialized: store_results=%s, complexity_metrics=%s, current_flops=%s",
            self.store_results,
            self.complexity_metrics,
            self.current_flops,
        )

    # ------------------------------------------------------------------
    # Dependency integration
    # ------------------------------------------------------------------
    def attach_dependencies(
        self,
        *,
        tokenizer: Any = None,
        nlp_engine: Any = None,
        text_encoder: Any = None,
        text_decoder: Any = None,
    ) -> None:
        """
        Attach optional runtime dependencies.

        Supported dependencies
        ----------------------
        tokenizer:
            Tokenizer-like object used for token counting or optional summarization.
        nlp_engine:
            NLP engine providing process_text/apply_dependency_rules/resolve_coreferences.
        text_encoder, text_decoder:
            Optional text-generation components used to synthesize an executive
            summary when available.
        """
        if tokenizer is not None:
            self.tokenizer = tokenizer
        if nlp_engine is not None:
            self.nlp_engine = nlp_engine
        if text_encoder is not None:
            self.text_encoder = text_encoder
        if text_decoder is not None:
            self.text_decoder = text_decoder

        self.calculations.attach_dependencies(
            tokenizer=self.tokenizer,
            nlp_engine=self.nlp_engine,
        )

    def set_current_flops(self, current_flops: float) -> float:
        """Update the evaluator's active FLOP measurement."""
        self.current_flops = self._require_positive_number(
            current_flops,
            "current_flops",
        )
        self.calculations.set_current_flops(self.current_flops)
        return self.current_flops

    # ------------------------------------------------------------------
    # Evaluation lifecycle
    # ------------------------------------------------------------------
    def evaluate(
        self,
        outputs: Sequence[Any],
        ground_truths: Optional[Sequence[Any]] = None,
        *,
        current_flops: Optional[float] = None,
        execution_metadata: Optional[Mapping[str, Any]] = None,
        store_result: Optional[bool] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a structured efficiency evaluation over model or system outputs.

        Parameters
        ----------
        outputs:
            Sequence of outputs. Entries may be strings, bytes, mappings with
            generation metadata, or other serializable structures.
        ground_truths:
            Optional sequence of reference outputs. These are not required for
            efficiency scoring, but are used for diagnostic comparisons.
        current_flops:
            Optional override for the active FLOP count used by computational
            efficiency.
        execution_metadata:
            Optional metadata that will be included in the result payload.
        store_result:
            Override for whether the result should be persisted to evaluator memory.
        mode:
            Optional operational mode key used to override weightings from config.

        Returns
        -------
        dict
            A serializable result payload suitable for reporting or persistence.
        """
        normalized_outputs = self._ensure_non_string_sequence(outputs, "outputs")
        normalized_truths = (
            self._ensure_non_string_sequence(ground_truths, "ground_truths")
            if ground_truths is not None
            else None
        )
        execution_metadata = dict(execution_metadata or {})
        persist = self.store_results if store_result is None else bool(store_result)

        if current_flops is not None:
            self.set_current_flops(current_flops)
        else:
            self.calculations.set_current_flops(self.current_flops)

        start_wall = time.perf_counter()
        start_mem_mb = self._get_process_memory_mb()

        try:
            efficiency_result = self.calculations.calculate_efficiency_metrics(
                normalized_outputs,
                current_flops=self.current_flops,
            )
            base_metrics = dict(efficiency_result.get("metrics", {}))
            linguistic_complexity = dict(efficiency_result.get("linguistic_complexity", {}))
            if not self.complexity_metrics:
                linguistic_complexity = {
                    "avg_sentence_length": 0.0,
                    "pos_diversity": 0.0,
                    "dependency_complexity": 0.0,
                    "entity_density": 0.0,
                }
                base_metrics["linguistic_score"] = 0.0

            runtime_seconds = time.perf_counter() - start_wall
            end_mem_mb = self._get_process_memory_mb()
            memory_delta_mb = max(0.0, end_mem_mb - start_mem_mb)

            token_stats = self._calculate_token_statistics(normalized_outputs)
            energy_estimate = self._estimate_energy_usage(
                outputs=normalized_outputs,
                runtime_seconds=runtime_seconds,
            )

            metrics: Dict[str, float] = {
                "temporal": float(base_metrics.get("temporal", 0.0)),
                "spatial": float(base_metrics.get("spatial", 0.0)),
                "computational": float(base_metrics.get("computational", 0.0)),
                "token_efficiency": float(base_metrics.get("token_efficiency", 0.0)),
                "linguistic_score": float(base_metrics.get("linguistic_score", 0.0)),
                "syntactic_complexity": float(linguistic_complexity.get("dependency_complexity", 0.0)),
                "semantic_density": float(linguistic_complexity.get("entity_density", 0.0)),
                "structural_variety": float(linguistic_complexity.get("pos_diversity", 0.0)),
                "avg_sentence_length": float(linguistic_complexity.get("avg_sentence_length", 0.0)),
                "estimated_energy_joules": float(energy_estimate),
                "memory_usage_mb": float(memory_delta_mb),
                "execution_time": float(runtime_seconds),
                "output_count": float(len(normalized_outputs)),
                "avg_output_tokens": float(token_stats["avg_tokens"]),
                "avg_output_chars": float(token_stats["avg_chars"]),
                "text_output_fraction": float(token_stats["text_fraction"]),
            }

            if normalized_truths is not None:
                metrics.update(self._calculate_reference_diagnostics(normalized_outputs, normalized_truths))

            composite_score = self._calculate_composite_score(metrics, mode=mode)
            metrics["score"] = composite_score

            resource_observation = {
                "process_memory_start_mb": start_mem_mb,
                "process_memory_end_mb": end_mem_mb,
                "memory_delta_mb": memory_delta_mb,
                "runtime_seconds": runtime_seconds,
                "estimated_energy_joules": energy_estimate,
                "current_flops": float(self.current_flops),
            }

            comparisons = self._compare_against_baselines(metrics)
            diagnostics = self._build_diagnostics(
                metrics=metrics,
                comparisons=comparisons,
                outputs=normalized_outputs,
                ground_truths=normalized_truths,
                mode=mode,
            )
            recommendations = self._generate_recommendations(metrics, diagnostics, comparisons)

            result = EfficiencyEvaluationResult(
                metadata={
                    "evaluated_at": _utcnow().isoformat(),
                    "module_version": self.MODULE_VERSION,
                    "config_path": self.config_path,
                    "mode": mode or "default",
                    "store_result": persist,
                    "complexity_metrics_enabled": self.complexity_metrics,
                    "execution_metadata": execution_metadata,
                },
                metrics=metrics,
                linguistic_complexity=linguistic_complexity,
                resource_observation=resource_observation,
                baseline_comparison={key: value.to_dict() for key, value in comparisons.items()},
                diagnostics=diagnostics,
                recommendations=recommendations,
            )

            result.memory_entry_id = self._persist_evaluation_result(result.to_dict(), enabled=persist)
            self._update_visualizer(result.to_dict())
            self._append_history(result.to_dict())
            return result.to_dict()

        except EvaluationError:
            raise
        except Exception as exc:
            raise MetricCalculationError(
                metric_name="efficiency_evaluation",
                inputs={"outputs": normalized_outputs[:3], "ground_truths": (normalized_truths or [])[:3]},
                reason=str(exc),
            ) from exc

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def generate_report(
        self,
        result: Mapping[str, Any],
        *,
        format: str = "markdown",
        include_visualizations: bool = True,
        include_recommendations: bool = True,
    ) -> Any:
        """
        Generate a report from a prior efficiency evaluation result.

        Supported formats: ``markdown``, ``json``, ``yaml``, ``dict``.
        """
        normalized = self._normalize_result_payload(result)
        report_payload = self._build_report_payload(
            normalized,
            include_visualizations=include_visualizations,
            include_recommendations=include_recommendations,
        )

        normalized_format = str(format).strip().lower()
        try:
            if normalized_format == "dict":
                return report_payload
            if normalized_format == "json":
                return json.dumps(report_payload, indent=2, sort_keys=False, default=str)
            if normalized_format == "yaml":
                return yaml.safe_dump(report_payload, default_flow_style=False, sort_keys=False)
            if normalized_format == "markdown":
                return self._render_markdown_report(report_payload)
        except EvaluationError:
            raise
        except Exception as exc:
            raise ReportGenerationError(
                report_type="Efficiency",
                template=str(self.report_template_path or "efficiency_report"),
                error_details=str(exc),
            ) from exc

        raise ReportGenerationError(
            report_type="Efficiency",
            template=str(self.report_template_path or "efficiency_report"),
            error_details=f"Unsupported report format: {format}",
        )

    def export_report(
        self,
        result: Mapping[str, Any],
        *,
        destination_path: str,
        format: Optional[str] = None,
    ) -> str:
        """Generate and persist a report to disk."""
        path = Path(destination_path)
        output_format = format or path.suffix.lstrip(".") or "markdown"
        rendered = self.generate_report(result, format=output_format)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(rendered if isinstance(rendered, str) else json.dumps(rendered, indent=2, default=str))
        return str(path)

    # ------------------------------------------------------------------
    # Diagnostics and recommendations
    # ------------------------------------------------------------------
    def _build_diagnostics(
        self,
        *,
        metrics: Mapping[str, float],
        comparisons: Mapping[str, BaselineComparison],
        outputs: Sequence[Any],
        ground_truths: Optional[Sequence[Any]],
        mode: Optional[str],
    ) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {
            "mode": mode or "default",
            "output_types": self._summarize_output_types(outputs),
            "baseline_status": {
                key: comparison.status for key, comparison in comparisons.items()
            },
            "warnings": [],
        }

        if metrics.get("temporal", 0.0) == 0.0:
            diagnostics["warnings"].append(
                "Temporal efficiency could not be derived from generation_time metadata."
            )
        if metrics.get("token_efficiency", 0.0) < 0.5:
            diagnostics["warnings"].append(
                "Outputs are substantially larger than the configured token baseline."
            )
        if metrics.get("computational", 0.0) < 1.0:
            diagnostics["warnings"].append(
                "Observed FLOP cost exceeds the configured computational baseline."
            )
        if metrics.get("memory_usage_mb", 0.0) > float(self.baselines.get("memory_threshold", float("inf"))):
            diagnostics["warnings"].append(
                "Observed memory growth exceeds the configured memory threshold."
            )

        diagnostics["report_template_available"] = bool(self.report_template_path and self.report_template_path.exists())
        diagnostics["recommendation_template_available"] = bool(
            self.recommendation_template_path and self.recommendation_template_path.exists()
        )
        diagnostics["energy_model_configured"] = bool(self.energy_model_path)
        diagnostics["uses_nlp_engine"] = self.nlp_engine is not None
        diagnostics["uses_tokenizer"] = self.tokenizer is not None
        diagnostics["uses_text_generation"] = (
            self.tokenizer is not None and self.text_encoder is not None and self.text_decoder is not None
        )

        if ground_truths is not None:
            diagnostics["has_reference_outputs"] = True
            diagnostics["reference_count"] = len(ground_truths)
        else:
            diagnostics["has_reference_outputs"] = False

        if self.enable_historical and self.history:
            scores = [record.score for record in self.history]
            diagnostics["history"] = {
                "evaluations": len(self.history),
                "best_score": max(scores),
                "average_score": float(np.mean(scores)),
                "latest_score": scores[-1],
            }

        return diagnostics

    def _generate_recommendations(
        self,
        metrics: Mapping[str, float],
        diagnostics: Mapping[str, Any],
        comparisons: Mapping[str, BaselineComparison],
    ) -> List[str]:
        recommendations: List[str] = []
        template_recommendations = self._evaluate_recommendation_template(metrics)
        recommendations.extend(template_recommendations)

        if metrics.get("temporal", 0.0) < 1.0:
            recommendations.append(
                "Reduce latency by batching requests more carefully, pruning expensive post-processing, or precomputing recurrent features."
            )
        if metrics.get("spatial", 0.0) < 1.0:
            recommendations.append(
                "Improve spatial efficiency by reducing payload verbosity, compressing structured outputs, or trimming unnecessary metadata."
            )
        if metrics.get("computational", 0.0) < 1.0:
            recommendations.append(
                "Observed FLOP demand exceeds the configured baseline. Review model size, decoding strategy, and expensive auxiliary analysis stages."
            )
        if metrics.get("token_efficiency", 0.0) < 0.75:
            recommendations.append(
                "Token efficiency is low relative to baseline. Consider shorter response policies, more targeted prompts, or stricter decoding controls."
            )
        if metrics.get("linguistic_score", 0.0) < 0.3 and self.complexity_metrics:
            recommendations.append(
                "Linguistic efficiency is weak. Review verbosity, entity density, and syntactic complexity relative to the task requirements."
            )
        if not diagnostics.get("uses_tokenizer"):
            recommendations.append(
                "Attach a tokenizer for more accurate token-efficiency estimates instead of relying on fallback token counts."
            )
        if self.complexity_metrics and not diagnostics.get("uses_nlp_engine"):
            recommendations.append(
                "Attach the NLP engine to enable richer linguistic diagnostics and dependency-level complexity analysis."
            )

        deduplicated: List[str] = []
        seen: set[str] = set()
        for item in recommendations:
            text = str(item).strip()
            if text and text.casefold() not in seen:
                deduplicated.append(text)
                seen.add(text.casefold())
        return deduplicated

    def _evaluate_recommendation_template(self, metrics: Mapping[str, float]) -> List[str]:
        if not self.recommendation_template_path:
            return []

        template = self._load_json_template(
            self.recommendation_template_path,
            template_name="recommendation_template",
        )
        recommendations: List[str] = []
        for section in template.get("recommendation_sections", []):
            if not isinstance(section, Mapping):
                continue
            title = str(section.get("section_title", "")).strip()
            section_lines: List[str] = []
            for candidate in section.get("recommendations", []):
                if not isinstance(candidate, Mapping):
                    continue
                condition = candidate.get("condition", {})
                if self._condition_matches(metrics, condition):
                    message = str(candidate.get("message", "")).strip()
                    if message:
                        section_lines.append(f"- {message}")
            if section_lines and title:
                recommendations.append(f"{title}: " + " ".join(item[2:] if item.startswith("- ") else item for item in section_lines))
            elif section_lines:
                recommendations.extend(item[2:] if item.startswith("- ") else item for item in section_lines)
        return recommendations

    def _condition_matches(self, metrics: Mapping[str, float], condition: Mapping[str, Any]) -> bool:
        if not isinstance(condition, Mapping):
            return False
        metric_key = str(condition.get("metric", "")).strip()
        operator = str(condition.get("operator", "")).strip()
        threshold = condition.get("threshold")

        if not metric_key or metric_key not in metrics:
            return False
        try:
            threshold_value = float(threshold)
            current_value = float(metrics[metric_key])
        except (TypeError, ValueError):
            return False

        if operator == "<":
            return current_value < threshold_value
        if operator == "<=":
            return current_value <= threshold_value
        if operator == ">":
            return current_value > threshold_value
        if operator == ">=":
            return current_value >= threshold_value
        if operator == "==":
            return current_value == threshold_value
        return False

    # ------------------------------------------------------------------
    # Comparisons and metrics
    # ------------------------------------------------------------------
    def _calculate_composite_score(self, metrics: Mapping[str, float], *, mode: Optional[str]) -> float:
        weights = dict(self.efficiency_weights)
        if mode:
            mode_config = self.operational_modes.get(mode, {})
            if isinstance(mode_config, Mapping):
                weights_override = mode_config.get("weights", {})
                if isinstance(weights_override, Mapping) and weights_override:
                    weights = {
                        str(key): float(value)
                        for key, value in weights_override.items()
                        if isinstance(value, (int, float))
                    }

        selected = {key: float(metrics[key]) for key in weights if key in metrics}
        if not selected:
            return 0.0

        total_weight = sum(float(weights[key]) for key in selected)
        if total_weight <= 0:
            raise MetricCalculationError(
                metric_name="efficiency_composite_score",
                inputs=weights,
                reason="configured efficiency weights must sum to a positive value",
            )

        score = sum(float(weights[key]) * float(selected[key]) for key in selected) / total_weight
        return float(score)

    def _compare_against_baselines(self, metrics: Mapping[str, float]) -> Dict[str, BaselineComparison]:
        comparisons: Dict[str, BaselineComparison] = {}

        baseline_pairs = {
            "flops": ("current_flops", float(self.current_flops), float(self.baselines.get("flops", self.current_flops)), "lower_is_better"),
            "memory_usage_mb": ("memory_usage_mb", float(metrics.get("memory_usage_mb", 0.0)), float(self.baselines.get("memory_threshold", 1.0)), "lower_is_better"),
            "estimated_energy_joules": ("estimated_energy_joules", float(metrics.get("estimated_energy_joules", 0.0)), float(self.baselines.get("energy_usage", 1.0)), "lower_is_better"),
            "token_efficiency": ("token_efficiency", float(metrics.get("token_efficiency", 0.0)), float(self.config.get("avg_token_baseline", 1.0)), "higher_is_better"),
        }

        for key, (name, observed, baseline, direction) in baseline_pairs.items():
            if baseline <= 0:
                raise ComparisonError(
                    baseline=name,
                    current=str(observed),
                    error_details="Baseline must be greater than zero for comparison.",
                )

            ratio = observed / baseline
            if direction == "lower_is_better":
                status = "meets" if ratio <= 1.0 else "exceeds"
            else:
                status = "meets" if ratio >= 1.0 else "below"
            comparisons[key] = BaselineComparison(
                name=name,
                observed=float(observed),
                baseline=float(baseline),
                ratio=float(ratio),
                status=status,
                direction=direction,
            )

        return comparisons

    def _calculate_reference_diagnostics(
        self,
        outputs: Sequence[Any],
        ground_truths: Sequence[Any],
    ) -> Dict[str, float]:
        if len(ground_truths) == 0:
            return {}
        output_token_total = sum(self._estimate_token_count(item) for item in outputs)
        truth_token_total = sum(self._estimate_token_count(item) for item in ground_truths)
        output_chars = sum(len(self._extract_text(item)) for item in outputs)
        truth_chars = sum(len(self._extract_text(item)) for item in ground_truths)

        token_ratio = output_token_total / truth_token_total if truth_token_total > 0 else 0.0
        char_ratio = output_chars / truth_chars if truth_chars > 0 else 0.0
        alignment = 1.0 / (1.0 + abs(1.0 - token_ratio)) if token_ratio > 0 else 0.0

        return {
            "reference_token_ratio": float(token_ratio),
            "reference_char_ratio": float(char_ratio),
            "reference_alignment": float(alignment),
        }

    def _calculate_token_statistics(self, outputs: Sequence[Any]) -> Dict[str, float]:
        token_counts = [self._estimate_token_count(item) for item in outputs]
        char_counts = [len(self._extract_text(item)) for item in outputs]
        text_like = [1 for item in outputs if bool(self._extract_text(item))]
        return {
            "avg_tokens": float(np.mean(token_counts)) if token_counts else 0.0,
            "avg_chars": float(np.mean(char_counts)) if char_counts else 0.0,
            "text_fraction": float(sum(text_like) / len(outputs)) if outputs else 0.0,
        }

    def _estimate_energy_usage(self, outputs: Sequence[Any], runtime_seconds: float) -> float:
        total_energy = 0.0
        observed = False
        baseline_power = float(self.baselines.get("energy_usage", 0.0))

        for output in outputs:
            if isinstance(output, Mapping):
                if "energy_joules" in output:
                    total_energy += self._coerce_non_negative_number(output["energy_joules"], "output.energy_joules")
                    observed = True
                elif "energy_usage_joules" in output:
                    total_energy += self._coerce_non_negative_number(output["energy_usage_joules"], "output.energy_usage_joules")
                    observed = True
                elif "power_watts" in output and "generation_time" in output:
                    watts = self._coerce_non_negative_number(output["power_watts"], "output.power_watts")
                    duration = self._coerce_non_negative_number(output["generation_time"], "output.generation_time")
                    total_energy += watts * duration
                    observed = True
                elif "energy_wh" in output:
                    total_energy += self._coerce_non_negative_number(output["energy_wh"], "output.energy_wh") * 3600.0
                    observed = True

        if observed:
            return float(total_energy)
        if baseline_power > 0 and runtime_seconds > 0:
            return float(baseline_power * runtime_seconds)
        return 0.0

    def _estimate_token_count(self, output: Any) -> int:
        text = self._extract_text(output)
        if not text:
            return 0
        if self.tokenizer is not None:
            try:
                if hasattr(self.tokenizer, "tokenize"):
                    return len(self.tokenizer.tokenize(text))
                if callable(self.tokenizer):
                    encoded = self.tokenizer(text)
                    if isinstance(encoded, Mapping) and "input_ids" in encoded:
                        input_ids = encoded["input_ids"]
                        if hasattr(input_ids, "numel"):
                            return int(input_ids.numel())
                        return len(input_ids)
            except Exception as exc:
                logger.warning("Tokenizer-based token estimate failed; falling back to whitespace split: %s", exc)
        return max(1, len(text.split()))

    # ------------------------------------------------------------------
    # Persistence and visualization integration
    # ------------------------------------------------------------------
    def _persist_evaluation_result(self, result: Mapping[str, Any], *, enabled: bool) -> Optional[str]:
        if not enabled:
            return None

        priority = "medium"
        score = float(result.get("metrics", {}).get("score", 0.0))
        if score < 0.5:
            priority = "high"
        elif score >= 1.0:
            priority = "low"

        try:
            if hasattr(self.memory, "add_evaluation_result"):
                return self.memory.add_evaluation_result(
                    "efficiency_evaluator",
                    dict(result),
                    tags=["efficiency", "evaluation"],
                    priority=priority,
                    metadata={"module_version": self.MODULE_VERSION},
                )
            return self.memory.add(
                entry=dict(result),
                tags=["efficiency", "evaluation"],
                priority=priority,
                source="efficiency_evaluator",
                category="evaluation",
                metadata={"module_version": self.MODULE_VERSION},
            )
        except EvaluationError:
            raise
        except Exception as exc:
            raise MemoryAccessError(
                operation="add",
                key="efficiency_evaluation_result",
                error_details=str(exc),
            ) from exc

    def _update_visualizer(self, result: Mapping[str, Any]) -> None:
        try:
            visualizer = get_visualizer()
            metrics = result.get("metrics", {})
            history_scores = [record.score for record in self.history] + [float(metrics.get("score", 0.0))]
            visualizer.update_metrics(
                {
                    "successes": 1 if float(metrics.get("score", 0.0)) >= 1.0 else 0,
                    "failures": 1 if float(metrics.get("score", 0.0)) < 1.0 else 0,
                    "operational_time": float(result.get("resource_observation", {}).get("runtime_seconds", 0.0)),
                    "reward": float(metrics.get("score", 0.0)),
                    "pass_rate": float(np.mean(history_scores)) if history_scores else float(metrics.get("score", 0.0)),
                }
            )
        except Exception as exc:
            logger.warning("Visualizer integration failed: %s", exc)

    def _append_history(self, result: Mapping[str, Any]) -> None:
        if not self.enable_historical:
            return
        record = EfficiencyHistoryRecord(
            evaluated_at=str(result.get("metadata", {}).get("evaluated_at", _utcnow().isoformat())),
            score=float(result.get("metrics", {}).get("score", 0.0)),
            temporal=float(result.get("metrics", {}).get("temporal", 0.0)),
            spatial=float(result.get("metrics", {}).get("spatial", 0.0)),
            computational=float(result.get("metrics", {}).get("computational", 0.0)),
            token_efficiency=float(result.get("metrics", {}).get("token_efficiency", 0.0)),
            runtime_seconds=float(result.get("resource_observation", {}).get("runtime_seconds", 0.0)),
            memory_delta_mb=float(result.get("resource_observation", {}).get("memory_delta_mb", 0.0)),
            output_count=int(result.get("metrics", {}).get("output_count", 0.0)),
        )
        self.history.append(record)

    # ------------------------------------------------------------------
    # Report construction helpers
    # ------------------------------------------------------------------
    def _build_report_payload(
        self,
        result: Mapping[str, Any],
        *,
        include_visualizations: bool,
        include_recommendations: bool,
    ) -> Dict[str, Any]:
        metrics = dict(result["metrics"])
        comparisons = dict(result["baseline_comparison"])
        diagnostics = dict(result["diagnostics"])
        recommendations = list(result.get("recommendations", [])) if include_recommendations else []

        payload: Dict[str, Any] = {
            "metadata": {
                "generated_at": _utcnow().isoformat(),
                "report_type": "Efficiency Evaluation",
                "module_version": self.MODULE_VERSION,
                "source_evaluated_at": result["metadata"].get("evaluated_at"),
            },
            "summary": {
                "score": metrics.get("score", 0.0),
                "overall_status": self._classify_score(metrics.get("score", 0.0)),
                "key_strengths": self._identify_strengths(metrics),
                "key_concerns": diagnostics.get("warnings", []),
            },
            "metrics": metrics,
            "linguistic_complexity": dict(result.get("linguistic_complexity", {})),
            "resource_observation": dict(result.get("resource_observation", {})),
            "baseline_comparison": comparisons,
            "diagnostics": diagnostics,
            "recommendations": recommendations,
            "executive_summary": self._generate_executive_summary(metrics, diagnostics, recommendations),
            "visualizations": {},
        }

        if include_visualizations:
            payload["visualizations"] = self._generate_visualizations()
        return payload

    def _render_markdown_report(self, report: Mapping[str, Any]) -> str:
        metrics = report["metrics"]
        comparisons = report["baseline_comparison"]
        lines = [
            "# Efficiency Evaluation Report",
            "",
            f"**Generated**: {report['metadata']['generated_at']}",
            "",
            "## Executive Summary",
            report["executive_summary"],
            "",
            "## Score Summary",
            f"- **Overall Score**: {metrics.get('score', 0.0):.4f}",
            f"- **Status**: {report['summary']['overall_status']}",
            f"- **Strengths**: {', '.join(report['summary']['key_strengths']) if report['summary']['key_strengths'] else 'None identified'}",
            f"- **Concerns**: {', '.join(report['summary']['key_concerns']) if report['summary']['key_concerns'] else 'None'}",
            "",
            "## Core Metrics",
        ]

        metric_order = [
            "temporal",
            "spatial",
            "computational",
            "token_efficiency",
            "linguistic_score",
            "estimated_energy_joules",
            "memory_usage_mb",
            "execution_time",
            "avg_output_tokens",
            "avg_output_chars",
            "reference_alignment",
            "score",
        ]
        for key in metric_order:
            if key in metrics:
                lines.append(f"- **{key.replace('_', ' ').title()}**: {metrics[key]:.4f}")

        lines.extend(["", "## Baseline Comparison"])
        for name, comparison in comparisons.items():
            lines.append(
                f"- **{name.replace('_', ' ').title()}**: observed={comparison['observed']:.4f}, "
                f"baseline={comparison['baseline']:.4f}, ratio={comparison['ratio']:.4f}, status={comparison['status']}"
            )

        lines.extend(["", "## Diagnostics"])
        for warning in report["diagnostics"].get("warnings", []):
            lines.append(f"- {warning}")
        if not report["diagnostics"].get("warnings"):
            lines.append("- No critical diagnostic warnings.")

        if report.get("recommendations"):
            lines.extend(["", "## Recommendations"])
            for item in report["recommendations"]:
                lines.append(f"- {item}")

        if report.get("visualizations"):
            lines.extend(["", "## Visualizations"])
            for key, asset in report["visualizations"].items():
                if isinstance(asset, Mapping) and asset.get("image"):
                    lines.append(f"### {key.replace('_', ' ').title()}")
                    lines.append(f"![{key}](data:image/png;base64,{asset['image']})")
                    lines.append("")

        lines.extend(["", "---", f"*Report generated by {self.__class__.__name__}*"])
        return "\n".join(lines)

    def _generate_visualizations(self) -> Dict[str, Any]:
        assets: Dict[str, Any] = {}
        if not self.enable_historical or not self.history:
            return assets

        try:
            visualizer = get_visualizer()
        except Exception as exc:
            raise VisualizationError(
                chart_type="visualizer",
                data={"history_length": len(self.history)},
                error_details=str(exc),
            ) from exc

        try:
            scores = [record.score for record in self.history]
            runtimes = [record.runtime_seconds for record in self.history]

            if scores:
                score_chart = visualizer.render_temporal_chart(
                    QSize(700, 400),
                    "pass_rate",
                    data=scores,
                )
                assets["score_trend"] = {
                    "chart_type": "pass_rate",
                    "encoding": "base64",
                    "image": visualizer._chart_to_base64(score_chart),
                }

            if runtimes:
                runtime_chart = visualizer.render_temporal_chart(
                    QSize(700, 400),
                    "operational_times",
                    data=runtimes,
                )
                assets["runtime_trend"] = {
                    "chart_type": "operational_times",
                    "encoding": "base64",
                    "image": visualizer._chart_to_base64(runtime_chart),
                }
            return assets
        except EvaluationError:
            raise
        except Exception as exc:
            raise VisualizationError(
                chart_type="efficiency_trends",
                data={"history_length": len(self.history)},
                error_details=str(exc),
            ) from exc

    def _generate_executive_summary(
        self,
        metrics: Mapping[str, float],
        diagnostics: Mapping[str, Any],
        recommendations: Sequence[str],
    ) -> str:
        prompt = (
            "Summarize the efficiency evaluation with emphasis on overall score, latency, "
            "memory growth, computational demand, and the main recommended improvement areas."
        )
        generated = self._generate_model_assisted_summary(prompt, metrics, recommendations)
        if generated:
            return generated

        status = self._classify_score(metrics.get("score", 0.0))
        parts = [
            f"The evaluated outputs are in a **{status.lower()}** efficiency state with an overall score of {metrics.get('score', 0.0):.3f}.",
            f"Temporal efficiency is {metrics.get('temporal', 0.0):.3f}, spatial efficiency is {metrics.get('spatial', 0.0):.3f}, and computational efficiency is {metrics.get('computational', 0.0):.3f}.",
        ]
        if diagnostics.get("warnings"):
            parts.append(
                "The main concerns are: " + "; ".join(str(item) for item in diagnostics["warnings"][:3]) + "."
            )
        elif recommendations:
            parts.append(
                "The evaluation did not raise critical warnings, but optimization opportunities remain around "
                + "; ".join(str(item) for item in recommendations[:2]).rstrip(".")
                + "."
            )
        else:
            parts.append("The measured outputs remain within the expected operating envelope and do not require immediate intervention.")
        return " ".join(parts)

    def _generate_model_assisted_summary(
        self,
        prompt: str,
        metrics: Mapping[str, float],
        recommendations: Sequence[str],
    ) -> Optional[str]:
        if not (self.tokenizer is not None and self.text_encoder is not None and self.text_decoder is not None):
            return None

        try:
            summary_prompt = (
                f"{prompt}\n"
                f"Score: {metrics.get('score', 0.0):.3f}\n"
                f"Temporal: {metrics.get('temporal', 0.0):.3f}\n"
                f"Spatial: {metrics.get('spatial', 0.0):.3f}\n"
                f"Computational: {metrics.get('computational', 0.0):.3f}\n"
                f"Token efficiency: {metrics.get('token_efficiency', 0.0):.3f}\n"
                f"Recommendations: {'; '.join(recommendations[:3]) if recommendations else 'None'}"
            )

            encoded = self.tokenizer(summary_prompt)
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")

            if hasattr(input_ids, "dim") and input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if attention_mask is not None and hasattr(attention_mask, "dim") and attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)

            device = next(self.text_encoder.parameters()).device if hasattr(self.text_encoder, "parameters") else torch.device("cpu")
            if hasattr(input_ids, "to"):
                input_ids = input_ids.to(device)
            if attention_mask is not None and hasattr(attention_mask, "to"):
                attention_mask = attention_mask.to(device)

            encoded_prompt = self.text_encoder(input_ids, attention_mask=attention_mask)
            generated_ids = self.text_decoder.inference(
                memory=encoded_prompt,
                strategy="sampling",
            )
            if hasattr(generated_ids, "dim") and generated_ids.dim() > 1:
                generated_ids = generated_ids[0]
            if hasattr(self.tokenizer, "decode"):
                return str(self.tokenizer.decode(generated_ids, skip_special_tokens=True)).strip() or None
            return None
        except Exception as exc:
            logger.warning("Model-assisted summary generation failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Template and serialization helpers
    # ------------------------------------------------------------------
    def _load_json_template(self, path: Path, *, template_name: str) -> Dict[str, Any]:
        if not path.exists():
            raise TemplateError(
                template_path=str(path),
                error_details=f"{template_name} file not found",
            )
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise TemplateError(
                template_path=str(path),
                error_details=f"{template_name} is not valid JSON: {exc}",
            ) from exc
        except OSError as exc:
            raise TemplateError(
                template_path=str(path),
                error_details=f"{template_name} could not be read: {exc}",
            ) from exc

        if not isinstance(payload, Mapping):
            raise TemplateError(
                template_path=str(path),
                error_details=f"{template_name} must deserialize to a JSON object",
            )
        return dict(payload)

    def _normalize_result_payload(self, result: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(result, Mapping):
            raise ValidationFailureError(
                rule_name="efficiency_result_mapping",
                data=type(result).__name__,
                expected="mapping",
            )

        required_top_level = {
            "metadata",
            "metrics",
            "linguistic_complexity",
            "resource_observation",
            "baseline_comparison",
            "diagnostics",
            "recommendations",
        }
        missing = sorted(key for key in required_top_level if key not in result)
        if missing:
            raise ValidationFailureError(
                rule_name="efficiency_result_shape",
                data=list(result.keys()),
                expected=f"missing required keys: {', '.join(missing)}",
            )
        return dict(result)

    # ------------------------------------------------------------------
    # Low-level normalization helpers
    # ------------------------------------------------------------------
    def _ensure_non_string_sequence(self, value: Optional[Sequence[Any]], field_name: str) -> List[Any]:
        if value is None:
            if field_name == "ground_truths":
                return []
            raise ValidationFailureError(field_name, value, "non-empty sequence")
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise ValidationFailureError(field_name, type(value).__name__, "sequence")
        result = list(value)
        if field_name == "outputs" and not result:
            raise ValidationFailureError(field_name, result, "non-empty sequence")
        return result

    def _resolve_optional_path(self, value: Any) -> Optional[Path]:
        if value in (None, ""):
            return None
        text = str(value).strip()
        if not text:
            return None
        candidate = Path(text)
        if candidate.is_absolute():
            return candidate
        # Compute project root once (cached)
        if not hasattr(self, '_project_root'):
            self._project_root = Path(__file__).parent.parent.parent.parent
        return self._project_root / candidate

    def _normalize_baselines(self, baselines: Any, field_name: str) -> Dict[str, float]:
        if not isinstance(baselines, Mapping):
            raise ConfigLoadError(self.config_path, field_name, "baselines must be a mapping")
        normalized: Dict[str, float] = {}
        for key, value in baselines.items():
            if isinstance(value, (list, dict, tuple, set)):
                continue
            try:
                normalized[str(key)] = float(value)
            except (TypeError, ValueError) as exc:
                raise ConfigLoadError(self.config_path, f"{field_name}.{key}", f"baseline must be numeric: {exc}") from exc
        return normalized

    def _normalize_weight_mapping(self, weights: Any, field_name: str) -> Dict[str, float]:
        if weights in (None, {}):
            return {}
        if not isinstance(weights, Mapping):
            raise ConfigLoadError(self.config_path, field_name, "weights must be a mapping")
        normalized: Dict[str, float] = {}
        for key, value in weights.items():
            normalized[str(key)] = self._require_non_negative_number(value, f"{field_name}.{key}")
        return normalized

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

    def _coerce_non_negative_number(self, value: Any, field_name: str) -> float:
        return self._require_non_negative_number(value, field_name)

    def _extract_text(self, output: Any) -> str:
        if output is None:
            return ""
        if isinstance(output, str):
            return output
        if isinstance(output, bytes):
            return output.decode("utf-8", errors="replace")
        if isinstance(output, Mapping):
            for key in ("text", "output", "value", "content", "response"):
                if key in output:
                    return self._extract_text(output[key])
            return json.dumps(output, default=str, sort_keys=True)
        return str(output)

    def _summarize_output_types(self, outputs: Sequence[Any]) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for item in outputs:
            key = type(item).__name__
            summary[key] = summary.get(key, 0) + 1
        return summary

    def _identify_strengths(self, metrics: Mapping[str, float]) -> List[str]:
        strengths: List[str] = []
        if metrics.get("temporal", 0.0) >= 1.0:
            strengths.append("temporal efficiency")
        if metrics.get("spatial", 0.0) >= 1.0:
            strengths.append("spatial efficiency")
        if metrics.get("computational", 0.0) >= 1.0:
            strengths.append("computational efficiency")
        if metrics.get("token_efficiency", 0.0) >= 1.0:
            strengths.append("token efficiency")
        if metrics.get("linguistic_score", 0.0) >= 0.5:
            strengths.append("linguistic compactness")
        return strengths

    def _classify_score(self, score: Any) -> str:
        try:
            value = float(score)
        except (TypeError, ValueError):
            return "UNKNOWN"
        if value >= 1.0:
            return "STRONG"
        if value >= 0.8:
            return "ACCEPTABLE"
        if value >= 0.5:
            return "DEGRADED"
        return "CRITICAL"

    def _get_process_memory_mb(self) -> float:
        if resource is None:
            return 0.0
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                return float(usage) / (1024.0 * 1024.0)
            return float(usage) / 1024.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Administrative helpers
    # ------------------------------------------------------------------
    def get_history(self) -> List[Dict[str, Any]]:
        """Return retained historical evaluation summaries."""
        return [record.to_dict() for record in self.history]

    def reset(self) -> None:
        """Reset retained historical evaluation state."""
        self.history.clear()

    def disable_temporarily(self) -> None:
        """Compatibility helper for degraded-mode suspension."""
        self.reset()
        logger.warning("Efficiency Evaluator temporarily disabled.")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Efficiency Evaluator ===\n")
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    eval = EfficiencyEvaluator()

    logger.info(f"{eval}")
    print(f"\n* * * * * Phase 2 * * * * *\n")
    outputs = ["The quick brown fox jumps over the lazy dog.", 
               "Sample output for efficiency analysis."]
    ground_truths = ["Expected reference text for comparison."]

    results = eval.evaluate(outputs, ground_truths)
    printer.pretty(f"Evaluation results:", results, "success" if results else "error")
    print(f"\n* * * * * Phase 3 * * * * *\n")


    report = eval.generate_report(results)
    printer.pretty(f"Evaluation report:", report, "success" if report else "error")
    print("\n=== Successfully Ran Efficiency Evaluator ===\n")
