"""
    Implements non-dominated sorting from:
    Deb et al. (2002) "A Fast Elitist Multiobjective Genetic Algorith
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import yaml

from .utils.config_loader import load_global_config, get_config_section
from .utils.evaluators_calculations import EvaluatorsCalculations
from .utils.evaluation_errors import (ComparisonError, StatisticalAnalysisError, EvaluationError,
                                      MemoryAccessError, ValidationFailureError,
                                      MetricCalculationError, ReportGenerationError)
from .modules.report import get_visualizer
from .evaluators_memory import EvaluatorsMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Statistical Evaluator")
printer = PrettyPrinter

MODULE_VERSION = "2.0.0"


@dataclass(slots=True)
class StatisticalEvaluationResult:
    metadata: Dict[str, Any]
    analysis: Dict[str, Any]
    pairwise_summary: Dict[str, Any]
    advanced_analysis: Dict[str, Any]
    pareto_frontiers: List[List[Dict[str, Any]]]
    diagnostics: Dict[str, Any]
    recommendations: List[str]
    memory_entry_id: Optional[str] = None
    report: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": dict(self.metadata),
            "analysis": dict(self.analysis),
            "pairwise_summary": dict(self.pairwise_summary),
            "advanced_analysis": dict(self.advanced_analysis),
            "pareto_frontiers": list(self.pareto_frontiers),
            "diagnostics": dict(self.diagnostics),
            "recommendations": list(self.recommendations),
            "memory_entry_id": self.memory_entry_id,
            "report": self.report,
        }


class StatisticalEvaluator:
    """
    Production-grade statistical evaluator.

    Responsibilities
    ----------------
    - Validate numeric datasets and run centralized statistical analysis
    - Expose confidence intervals, effect sizes, pairwise tests, and power analysis
    - Provide multi-objective frontier utilities for comparison studies
    - Persist results and integrate summary metrics with the shared visualizer
    """

    def __init__(self) -> None:
        self.config = load_global_config()
        self.config_path = str(self.config.get("__config_path__", "<config>"))
        self.section = get_config_section("statistical_evaluator")
        self.significance_threshold = self._coerce_probability(
            self.section.get("alpha", 0.05),
            "statistical_evaluator.alpha",
            include_zero=False,
            include_one=False,
        )
        self.confidence_level = self._coerce_probability(
            self.section.get("confidence_level", 0.95),
            "statistical_evaluator.confidence_level",
            include_zero=False,
            include_one=False,
        )
        self.min_sample_size = self._require_positive_int(
            self.section.get("min_sample_size", 10),
            "statistical_evaluator.min_sample_size",
        )

        self.memory = EvaluatorsMemory()
        self.calculations = EvaluatorsCalculations()
        self.objectives: List[str] = []
        self.weights: Dict[str, float] = {}
        self.history: List[Dict[str, Any]] = []
        self.store_results = bool(self.config.get("store_results", True))
        self.disabled = False

        logger.info("Statistical Evaluator initialized: alpha=%s confidence=%s", self.significance_threshold, self.confidence_level)

    # ------------------------------------------------------------------
    # Public evaluation API
    # ------------------------------------------------------------------
    def evaluate(
        self,
        datasets: Mapping[str, Sequence[float]],
        report: bool = False,
        *,
        report_format: str = "markdown",
        store_result: Optional[bool] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.disabled:
            raise ValidationFailureError("statistical_evaluator_enabled", False, True)
        normalized_datasets = self._validate_datasets(datasets)

        persist = self.store_results if store_result is None else bool(store_result)
        analysis = self.calculations.calculate_statistical_analysis(
            normalized_datasets,
            confidence_level=self.confidence_level,
        )
        pairwise_summary = self._build_pairwise_summary(analysis)
        advanced_analysis = {
            "demsar_protocol": self._build_demsar_protocol(normalized_datasets),
            "dataset_cardinality": {name: len(values) for name, values in normalized_datasets.items()},
        }
        diagnostics = self._build_diagnostics(normalized_datasets, analysis)
        recommendations = self._generate_recommendations(analysis, diagnostics)

        result = StatisticalEvaluationResult(
            metadata={
                "evaluated_at": _utcnow().isoformat(),
                "module_version": MODULE_VERSION,
                "config_path": self.config_path,
                "alpha": self.significance_threshold,
                "confidence_level": self.confidence_level,
                "store_result": persist,
                "caller_metadata": dict(metadata or {}),
            },
            analysis=analysis,
            pairwise_summary=pairwise_summary,
            advanced_analysis=advanced_analysis,
            pareto_frontiers=[],
            diagnostics=diagnostics,
            recommendations=recommendations,
        )

        self._record_history(result)
        self._update_visualizer(result)
        if persist:
            result.memory_entry_id = self._store_result(result)

        payload = result.to_dict()
        if report:
            payload["report"] = self.generate_report(payload, format=report_format)
        return payload

    def compute_confidence_interval(self, samples: Sequence[float], confidence: Optional[float] = None) -> Dict[str, float]:
        level = self.confidence_level if confidence is None else self._coerce_probability(
            confidence,
            "confidence",
            include_zero=False,
            include_one=False,
        )
        return self.calculations._calculate_confidence_interval(samples, level)

    def t_test(self, sample_a: Sequence[float], sample_b: Sequence[float]) -> Dict[str, Any]:
        return self.calculations._welch_t_test(
            np.asarray(sample_a, dtype=float),
            np.asarray(sample_b, dtype=float),
        )

    def anova(self, groups: Sequence[Sequence[float]]) -> Dict[str, Any]:
        if len(groups) < 2:
            raise StatisticalAnalysisError("anova", groups, "At least two groups are required.")
        flattened = {f"group_{index+1}": list(group) for index, group in enumerate(groups)}
        analysis = self.calculations.calculate_statistical_analysis(flattened, confidence_level=self.confidence_level)
        if "pairwise_tests" not in analysis:
            raise StatisticalAnalysisError("anova", groups, "Pairwise statistical analysis could not be completed.")
        names = list(flattened.keys())
        grand_values = [value for group in groups for value in group]
        if not grand_values:
            raise StatisticalAnalysisError("anova", groups, "No observations available for ANOVA.")
        k = len(groups)
        n_total = sum(len(group) for group in groups)
        grand_mean = float(np.mean(grand_values))
        ss_between = sum(len(group) * (float(np.mean(group)) - grand_mean) ** 2 for group in groups)
        ss_within = sum(sum((value - float(np.mean(group))) ** 2 for value in group) for group in groups)
        df_between = k - 1
        df_within = n_total - k
        ms_between = ss_between / df_between if df_between > 0 else 0.0
        ms_within = ss_within / df_within if df_within > 0 else 1e-12
        f_stat = ms_between / ms_within if ms_within > 0 else 0.0
        # simple tail approximation is intentionally not reimplemented here; rely on calculations for pairwise tests
        return {
            "f_stat": f_stat,
            "degrees_of_freedom": {"between": df_between, "within": df_within},
            "group_names": names,
            "status": "ok",
        }

    def pareto_frontier(
        self,
        solutions: Sequence[Mapping[str, Any]],
        *,
        objectives: Optional[Sequence[str]] = None,
        weights: Optional[Mapping[str, float]] = None,
    ) -> List[List[Dict[str, Any]]]:
        if not isinstance(solutions, Sequence) or isinstance(solutions, (str, bytes)):
            raise ValidationFailureError("solutions", type(solutions).__name__, "sequence of mappings")

        objectives_list = [str(obj) for obj in (objectives or self.objectives) if str(obj).strip()]
        if not objectives_list:
            raise ValidationFailureError("objectives", objectives_list, "non-empty objective list")
        weight_map = {obj: float(weights.get(obj, self.weights.get(obj, 1.0))) if weights else float(self.weights.get(obj, 1.0)) for obj in objectives_list}

        remaining = [dict(solution) for solution in solutions]
        frontiers: List[List[Dict[str, Any]]] = []
        while remaining:
            current_front: List[Dict[str, Any]] = []
            dominated: List[Dict[str, Any]] = []
            for candidate in remaining:
                if not any(self._dominates(other, candidate, objectives_list, weight_map) for other in remaining if other is not candidate):
                    current_front.append(candidate)
                else:
                    dominated.append(candidate)
            frontiers.append(current_front)
            remaining = dominated
        return frontiers

    def statistical_analysis(self, baseline: Sequence[float], treatment: Sequence[float]) -> Dict[str, Any]:
        baseline_values = np.asarray(baseline, dtype=float)
        treatment_values = np.asarray(treatment, dtype=float)
        if baseline_values.size != treatment_values.size or baseline_values.size == 0:
            raise ValidationFailureError(
                "demsar_protocol_inputs",
                {"baseline": baseline_values.size, "treatment": treatment_values.size},
                "Equal non-empty paired samples",
            )
        diffs = baseline_values - treatment_values
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1)) if diffs.size > 1 else 0.0
        z_scores = [float(diff / math.sqrt(max(len(diffs), 1))) for diff in diffs]
        return {
            "mean_diff": mean_diff,
            "effect_size": self._hedges_g(baseline_values, treatment_values),
            "significance": any(abs(z) > 2.58 for z in z_scores),
            "paired_test": self.t_test(baseline_values, treatment_values),
            "std_diff": std_diff,
        }

    def _hedges_g(self, a: Sequence[float], b: Sequence[float]) -> float:
        sample_a = np.asarray(a, dtype=float)
        sample_b = np.asarray(b, dtype=float)
        if len(sample_a) < 2 or len(sample_b) < 2:
            return 0.0
        return float(self.calculations.effect_size(sample_a, sample_b))

    def generate_report(self, results: Mapping[str, Any], *, format: str = "markdown") -> Any:
        if not isinstance(results, Mapping):
            raise ReportGenerationError("Statistical Analysis", "statistical_report", "Results must be a mapping.")
        normalized = str(format).strip().lower()
        if normalized == "dict":
            return dict(results)
        if normalized == "json":
            return json.dumps(dict(results), indent=2, sort_keys=False, default=str)
        if normalized == "yaml":
            return yaml.safe_dump(dict(results), default_flow_style=False, sort_keys=False)
        if normalized != "markdown":
            raise ReportGenerationError("Statistical Analysis", "statistical_report", f"Unsupported report format: {format}")

        try:
            analysis = results["analysis"]
            report: List[str] = [
                "# Statistical Analysis Report",
                f"**Generated**: {results['metadata']['evaluated_at']}",
                "",
                "## Descriptive Statistics",
            ]
            for name, stats in analysis.get("descriptive_stats", {}).items():
                report.append(
                    f"- **{name}**: mean={stats['mean']:.3f}, std={stats['std_dev']:.3f}, n={stats['sample_size']}, min={stats['min']:.3f}, max={stats['max']:.3f}"
                )

            report.extend(["", "## Pairwise Tests"])
            for name, comparison in analysis.get("pairwise_tests", {}).items():
                t_test = comparison.get("t_test", {})
                report.append(
                    f"- **{name}**: p={t_test.get('p_value')} | significant={t_test.get('significant')} | effect={comparison.get('effect_size', 0):.3f}"
                )

            report.extend(["", "## Confidence Intervals"])
            for name, interval in analysis.get("confidence_intervals", {}).items():
                report.append(f"- **{name}**: [{interval['lower']:.3f}, {interval['upper']:.3f}] (mean={interval['mean']:.3f})")

            if results.get("recommendations"):
                report.extend(["", "## Recommendations"])
                for recommendation in results["recommendations"]:
                    report.append(f"- {recommendation}")

            report.extend(["", f"*Significance threshold: α={results['metadata']['alpha']}*", f"*Report generated by {self.__class__.__name__}*"])
            return "\n".join(report)
        except Exception as exc:
            raise ReportGenerationError("Statistical Analysis", "statistical_report", str(exc)) from exc

    def disable_temporarily(self) -> None:
        self.disabled = True
        logger.warning("Statistical Evaluator temporarily disabled.")

    def enable(self) -> None:
        self.disabled = False
        logger.info("Statistical Evaluator re-enabled.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_datasets(self, datasets: Mapping[str, Sequence[float]]) -> Dict[str, List[float]]:
        if not isinstance(datasets, Mapping) or not datasets:
            raise ValidationFailureError("dataset_validation", datasets, "Non-empty mapping of datasets")
        normalized: Dict[str, List[float]] = {}
        for name, values in datasets.items():
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise ValidationFailureError("dataset_values", type(values).__name__, "sequence of numeric values")
            coerced = []
            for value in values:
                try:
                    coerced.append(float(value))
                except (TypeError, ValueError) as exc:
                    raise ValidationFailureError(f"dataset.{name}", value, "numeric value") from exc
            if len(coerced) < self.min_sample_size:
                raise ValidationFailureError("sample_size_validation", {"dataset": name, "size": len(coerced)}, f"Minimum sample size {self.min_sample_size}")
            normalized[str(name)] = coerced
        return normalized

    def _build_pairwise_summary(self, analysis: Mapping[str, Any]) -> Dict[str, Any]:
        pairwise = analysis.get("pairwise_tests", {})
        significant = [name for name, value in pairwise.items() if value.get("t_test", {}).get("significant")]
        strongest = None
        strongest_value = -1.0
        for name, value in analysis.get("effect_sizes", {}).items():
            magnitude = abs(float(value))
            if magnitude > strongest_value:
                strongest_value = magnitude
                strongest = {"comparison": name, "effect_size": float(value)}
        return {
            "significant_comparisons": significant,
            "significant_count": len(significant),
            "strongest_effect": strongest,
        }

    def _build_demsar_protocol(self, datasets: Mapping[str, Sequence[float]]) -> Dict[str, Any]:
        names = list(datasets.keys())
        if len(names) != 2:
            return {"status": "skipped", "reason": "Requires exactly two paired datasets."}
        return self.statistical_analysis(datasets[names[0]], datasets[names[1]])

    def _build_diagnostics(self, datasets: Mapping[str, Sequence[float]], analysis: Mapping[str, Any]) -> Dict[str, Any]:
        normality = analysis.get("normality", {})
        non_normal = [name for name, result in normality.items() if result.get("is_normal") is False]
        low_power = [name for name, power in analysis.get("power_analysis", {}).items() if float(power) < 0.8]
        return {
            "dataset_count": len(datasets),
            "non_normal_datasets": non_normal,
            "low_power_datasets": low_power,
            "significant_pairwise_tests": self._build_pairwise_summary(analysis).get("significant_count", 0),
        }

    def _generate_recommendations(self, analysis: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> List[str]:
        recommendations: List[str] = []
        if diagnostics.get("non_normal_datasets"):
            recommendations.append("Consider non-parametric alternatives for datasets that fail normality assumptions.")
        if diagnostics.get("low_power_datasets"):
            recommendations.append("Increase sample size for low-power datasets before drawing strong conclusions.")
        if not recommendations:
            recommendations.append("Statistical assumptions and power look acceptable for the evaluated datasets.")
        if self._build_pairwise_summary(analysis).get("significant_count", 0) == 0:
            recommendations.append("No statistically significant pairwise differences were detected at the configured alpha level.")
        return recommendations

    def _record_history(self, result: StatisticalEvaluationResult) -> None:
        summary = {
            "evaluated_at": result.metadata["evaluated_at"],
            "dataset_count": result.diagnostics["dataset_count"],
            "significant_pairwise_tests": result.diagnostics["significant_pairwise_tests"],
        }
        self.history.append(summary)
        if len(self.history) > 100:
            self.history = self.history[-100:]

    def _update_visualizer(self, result: StatisticalEvaluationResult) -> None:
        try:
            visualizer = get_visualizer()
            visualizer.update_metrics({
                "reward": max(0.0, min(1.0, 1.0 - (result.diagnostics["significant_pairwise_tests"] * 0.05))),
                "pass_rate": max(0.0, min(1.0, len(result.analysis.get("descriptive_stats", {})) / max(result.diagnostics["dataset_count"], 1))),
                "operational_time": float(result.diagnostics["dataset_count"]),
            })
        except Exception as exc:
            logger.warning("Visualizer update skipped for statistical evaluator: %s", exc)

    def _store_result(self, result: StatisticalEvaluationResult) -> str:
        try:
            return self.memory.add(
                entry=result.to_dict(),
                tags=["statistical_analysis", "statistics"],
                priority="medium",
                category="statistics",
                source=self.__class__.__name__,
            )
        except Exception as exc:
            raise MemoryAccessError("add", "statistical_results", str(exc)) from exc

    @staticmethod
    def _require_positive_int(value: Any, field_name: str) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise ValidationFailureError(field_name, value, "positive integer") from exc
        if number <= 0:
            raise ValidationFailureError(field_name, value, "positive integer")
        return number

    @staticmethod
    def _coerce_probability(value: Any, field_name: str, *, include_zero: bool, include_one: bool) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValidationFailureError(field_name, value, "probability") from exc
        if number < 0 or number > 1 or (not include_zero and number == 0) or (not include_one and number == 1):
            raise ValidationFailureError(field_name, value, "probability within configured bounds")
        return number

    def _dominates(
        self,
        a: Mapping[str, Any],
        b: Mapping[str, Any],
        objectives: Sequence[str],
        weights: Mapping[str, float],
    ) -> bool:
        better = False
        for objective in objectives:
            if objective not in a or objective not in b:
                continue
            a_val = float(a[objective]) * float(weights.get(objective, 1.0))
            b_val = float(b[objective]) * float(weights.get(objective, 1.0))
            if a_val < b_val:
                return False
            if a_val > b_val:
                better = True
        return better



def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Statistical Evaluator ===\n")
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    try:
        statistics = StatisticalEvaluator()
        datasets = {
            "Control":      [12.3, 14.2, 15.1, 13.8, 16.0, 14.4, 15.2, 13.5, 14.9, 15.3],
            "Treatment_A":  [18.4, 19.2, 17.8, 20.1, 19.5, 18.6, 19.7, 18.2, 19.1, 20.0],
            "Treatment_B":  [15.6, 16.9, 14.2, 17.3, 16.1, 15.8, 17.0, 15.3, 16.2, 16.5]
        }
        results = statistics.evaluate(datasets, report=True)
        logger.info(f"{statistics}")
        if 'report' in results:
            print(results['report'])
        elif 'error' in results:
            printer.pretty("Evaluation failed", results, "error")
            
    except Exception as e:
        printer.pretty("Fatal error during evaluation", str(e), "error")
    

    print("\n=== Statistical Evaluation Complete ===\n")
