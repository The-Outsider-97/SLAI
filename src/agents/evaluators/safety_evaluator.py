from __future__ import annotations

import json
import matplotlib.pyplot as plt
import numpy as np
import yaml

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .utils.config_loader import load_global_config, get_config_section
from .utils.evaluators_calculations import EvaluatorsCalculations
from .utils.evaluation_errors import (ComparisonError, EvaluationError, ValidationFailureError,
                                      MemoryAccessError, ReportGenerationError, VisualizationError)
from .modules.report import get_visualizer
from .evaluators_memory import EvaluatorsMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Safety Evaluator")
printer = PrettyPrinter

MODULE_VERSION = "2.0.0"


@dataclass(slots=True)
class SafetyIncidentMetrics:
    incident_id: str
    risk_level: float
    hazard_detection_time: float
    emergency_stop_time: float
    safety_margin: float
    collision_avoidance: bool
    standards_compliance: float
    risk_assessment: Dict[str, float] = field(default_factory=dict)
    threshold_violations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SafetyEvaluationResult:
    metadata: Dict[str, Any]
    aggregates: Dict[str, Any]
    incidents: List[Dict[str, Any]]
    threshold_assessment: Dict[str, Any]
    risk_distribution: Dict[str, float]
    diagnostics: Dict[str, Any]
    recommendations: List[str]
    memory_entry_id: Optional[str] = None
    report: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": dict(self.metadata),
            "aggregates": dict(self.aggregates),
            "incidents": list(self.incidents),
            "threshold_assessment": dict(self.threshold_assessment),
            "risk_distribution": dict(self.risk_distribution),
            "diagnostics": dict(self.diagnostics),
            "recommendations": list(self.recommendations),
            "memory_entry_id": self.memory_entry_id,
            "report": self.report,
        }


class SafetyEvaluator:
    """
    Production-grade evaluator for safety-critical robotics and automation scenarios.

    Responsibilities
    ----------------
    - Evaluate safety incidents against configured thresholds and standards
    - Build aggregate operational safety metrics and threshold assessments
    - Summarize risk-category distribution and compliance trends
    - Integrate with evaluator memory, shared calculations, and reporting
    """

    def __init__(self) -> None:
        self.config = load_global_config()
        self.config_path = str(self.config.get("__config_path__", "<config>"))
        self.section = get_config_section("safety_evaluator")

        self.thresholds = self._load_thresholds(self.section.get("thresholds", {}))
        self.safety_standards = self._normalize_string_list(
            self.section.get("safety_standards", ["ISO 13849", "IEC 61508", "ANSI/RIA R15.06"]),
            "safety_evaluator.safety_standards",
        )
        self.metric_weights = self._normalize_weight_mapping(self.section.get("weights", {}))
        self.risk_categories = self._normalize_string_list(
            self.section.get(
                "risk_categories",
                ["collision", "pinch_point", "crush_hazard", "electrical", "environmental", "control_failure"],
            ),
            "safety_evaluator.risk_categories",
        )
        self.store_results = bool(self.config.get("store_results", True))

        self.memory = EvaluatorsMemory()
        self.calculations = EvaluatorsCalculations()
        self.safety_incidents: List[SafetyIncidentMetrics] = []
        self.raw_incidents: List[Dict[str, Any]] = []
        self.compliance_history: List[float] = []
        self.disabled = False

        logger.info("Safety Evaluator initialized with standards: %s", self.safety_standards)

    # ------------------------------------------------------------------
    # Public evaluation API
    # ------------------------------------------------------------------

    def evaluate_incident(self, incident: Mapping[str, Any]) -> SafetyIncidentMetrics:
        normalized = self._normalize_incident(incident)
        risk_level = float(np.mean([normalized["risk_assessment"].get(cat, 0.0) for cat in self.risk_categories]))
        standards_compliance = float(np.mean([1.0 if normalized["standards"].get(std, False) else 0.0 for std in self.safety_standards]))

        threshold_violations = self._detect_incident_violations(
            risk_level=risk_level,
            hazard_detection_time=normalized["hazard_detection_time"],
            emergency_stop_time=normalized["emergency_stop_time"],
            safety_margin=normalized["safety_margin"],
            standards_compliance=standards_compliance,
        )
        recommendations = self._generate_incident_recommendations(threshold_violations)

        metrics = SafetyIncidentMetrics(
            incident_id=normalized["incident_id"],
            risk_level=risk_level,
            hazard_detection_time=normalized["hazard_detection_time"],
            emergency_stop_time=normalized["emergency_stop_time"],
            safety_margin=normalized["safety_margin"],
            collision_avoidance=normalized["collision_avoided"],
            standards_compliance=standards_compliance,
            risk_assessment=normalized["risk_assessment"],
            threshold_violations=threshold_violations,
            recommendations=recommendations,
        )

        self.safety_incidents.append(metrics)
        self.raw_incidents.append(dict(normalized))
        self.compliance_history.append(standards_compliance)
        if normalized.get("hazard_details"):
            self.calculations.add_hazard_records([normalized["hazard_details"]])
        return metrics

    def evaluate_operation(
        self,
        incidents: Sequence[Mapping[str, Any]],
        *,
        report: bool = False,
        report_format: str = "markdown",
        baseline_metrics: Optional[Mapping[str, Any]] = None,
        store_result: Optional[bool] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.disabled:
            raise ValidationFailureError("safety_evaluator_enabled", False, True)
        if not isinstance(incidents, Sequence) or isinstance(incidents, (str, bytes)):
            raise ValidationFailureError("safety_incidents", type(incidents).__name__, "sequence of mappings")

        persist = self.store_results if store_result is None else bool(store_result)
        incident_results: List[SafetyIncidentMetrics] = []
        anomalies: List[Dict[str, Any]] = []
        for incident in incidents:
            try:
                incident_results.append(self.evaluate_incident(incident))
            except EvaluationError:
                raise
            except Exception as exc:
                anomalies.append({"incident": str(incident)[:300], "error": str(exc)})
                logger.error("Incident evaluation failed: %s", exc)

        aggregates = self._build_aggregates(incident_results)
        threshold_assessment = self._build_threshold_assessment(aggregates)
        risk_distribution = self.calculations._calculate_risk_distribution() if self.calculations.hazard_data else {}
        baseline_comparison = self.compare_with_baseline(aggregates, baseline_metrics or {})
        diagnostics = self._build_diagnostics(aggregates, threshold_assessment, baseline_comparison, anomalies)
        recommendations = self._generate_operation_recommendations(aggregates, threshold_assessment)

        result = SafetyEvaluationResult(
            metadata={
                "evaluated_at": _utcnow().isoformat(),
                "module_version": MODULE_VERSION,
                "config_path": self.config_path,
                "incident_count_requested": len(incidents),
                "incident_count_evaluated": len(incident_results),
                "store_result": persist,
                "caller_metadata": dict(metadata or {}),
            },
            aggregates=aggregates,
            incidents=[incident.to_dict() for incident in incident_results],
            threshold_assessment=threshold_assessment,
            risk_distribution=risk_distribution,
            diagnostics=diagnostics,
            recommendations=recommendations,
        )

        self._update_visualizer(result)
        if persist:
            result.memory_entry_id = self._store_result(result)

        payload = result.to_dict()
        if report:
            payload["report"] = self.generate_report(payload, format=report_format)
        return payload

    def compare_with_baseline(
        self,
        current_metrics: Mapping[str, Any],
        baseline_metrics: Mapping[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        if not baseline_metrics:
            return {}
        comparison: Dict[str, Dict[str, Any]] = {}
        relevant = ["avg_risk_level", "avg_detection_time", "avg_stop_time", "avg_safety_margin", "compliance_rate", "composite_score"]
        for metric in relevant:
            if metric not in current_metrics or metric not in baseline_metrics:
                continue
            current_value = float(current_metrics[metric])
            baseline_value = float(baseline_metrics[metric])
            lower_is_better = metric in {"avg_risk_level", "avg_detection_time", "avg_stop_time"}
            diff = current_value - baseline_value
            comparison[metric] = {
                "current": current_value,
                "baseline": baseline_value,
                "absolute_difference": diff,
                "relative_change": (diff / baseline_value) if baseline_value not in (0, 0.0) else None,
                "improvement": current_value < baseline_value if lower_is_better else current_value > baseline_value,
                "direction": "lower_is_better" if lower_is_better else "higher_is_better",
            }
        return comparison

    def generate_report(self, results: Mapping[str, Any], *, format: str = "markdown") -> Any:
        if not isinstance(results, Mapping):
            raise ReportGenerationError("Safety Evaluation", "safety_report", "Results must be a mapping.")

        normalized = str(format).strip().lower()
        if normalized == "dict":
            return dict(results)
        if normalized == "json":
            return json.dumps(dict(results), indent=2, sort_keys=False, default=str)
        if normalized == "yaml":
            return yaml.safe_dump(dict(results), default_flow_style=False, sort_keys=False)
        if normalized != "markdown":
            raise ReportGenerationError("Safety Evaluation", "safety_report", f"Unsupported report format: {format}")

        try:
            aggregates = results["aggregates"]
            report: List[str] = [
                "# Safety Evaluation Report",
                f"**Generated**: {results['metadata']['evaluated_at']}",
                "",
                "## Executive Summary",
                f"- **Incidents Evaluated**: {aggregates['total_incidents']}",
                f"- **Critical Incidents**: {aggregates['critical_incidents']}",
                f"- **Collision Avoidance Rate**: {aggregates['collision_avoidance_rate']:.2%}",
                f"- **Average Risk Level**: {aggregates['avg_risk_level']:.3f}",
                f"- **Compliance Rate**: {aggregates['compliance_rate']:.2%}",
                f"- **Composite Safety Score**: {aggregates['composite_score']:.3f}/1.0",
                "",
                "## Aggregate Metrics",
                f"- **Average Hazard Detection Time**: {aggregates['avg_detection_time']:.3f}s",
                f"- **Average Emergency Stop Time**: {aggregates['avg_stop_time']:.3f}s",
                f"- **Average Safety Margin**: {aggregates['avg_safety_margin']:.3f}",
                f"- **Standards Compliance Mean**: {aggregates['compliance_rate']:.3f}",
            ]

            if results.get("risk_distribution"):
                report.extend(["", "## Risk Distribution"])
                for category, value in results["risk_distribution"].items():
                    report.append(f"- **{category.replace('_', ' ').title()}**: {value:.2%}")

            threshold_assessment = results.get("threshold_assessment", {})
            if threshold_assessment.get("violations"):
                report.extend(["", "## Threshold Violations"])
                for violation in threshold_assessment["violations"]:
                    report.append(f"- {violation}")

            if results.get("recommendations"):
                report.extend(["", "## Recommendations"])
                for recommendation in results["recommendations"]:
                    report.append(f"- {recommendation}")

            report.extend(["", f"*Report generated by {self.__class__.__name__}*"])
            return "\n".join(report)
        except Exception as exc:
            raise ReportGenerationError(
                report_type="Safety Evaluation",
                template="safety_report_template",
                error_details=str(exc),
            ) from exc

    def disable_temporarily(self) -> None:
        self.disabled = True
        logger.warning("SafetyEvaluator temporarily disabled.")

    def enable(self) -> None:
        self.disabled = False
        logger.info("SafetyEvaluator re-enabled.")

    # ------------------------------------------------------------------
    # Internal evaluation helpers
    # ------------------------------------------------------------------

    def _normalize_incident(self, incident: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(incident, Mapping):
            raise ValidationFailureError("incident_mapping", type(incident).__name__, "mapping")

        required = ["risk_assessment", "hazard_detection_time", "emergency_stop_time", "safety_margin", "collision_avoided"]
        missing = [key for key in required if key not in incident]
        if missing:
            raise ValidationFailureError("incident_structure", dict(incident), f"Missing keys: {', '.join(missing)}")

        risk_assessment = incident.get("risk_assessment", {})
        if not isinstance(risk_assessment, Mapping):
            raise ValidationFailureError("risk_assessment", type(risk_assessment).__name__, "mapping")

        normalized_risk = {category: self._coerce_probability(risk_assessment.get(category, 0.0), f"risk_assessment.{category}") for category in self.risk_categories}
        standards = {standard: bool(incident.get(standard, False)) for standard in self.safety_standards}
        hazard_details = incident.get("hazard_details", {})
        normalized_hazard_details = {}
        if isinstance(hazard_details, Mapping):
            for key, value in hazard_details.items():
                if key in self.risk_categories:
                    normalized_hazard_details[str(key)] = self._coerce_probability(value, f"hazard_details.{key}")

        return {
            "incident_id": str(incident.get("id", f"INC-{len(self.raw_incidents)+1:05d}")),
            "risk_assessment": normalized_risk,
            "hazard_detection_time": self._require_non_negative_number(incident["hazard_detection_time"], "hazard_detection_time"),
            "emergency_stop_time": self._require_non_negative_number(incident["emergency_stop_time"], "emergency_stop_time"),
            "safety_margin": self._require_non_negative_number(incident["safety_margin"], "safety_margin"),
            "collision_avoided": bool(incident["collision_avoided"]),
            "standards": standards,
            "hazard_details": normalized_hazard_details,
        }

    def _build_aggregates(self, incidents: Sequence[SafetyIncidentMetrics]) -> Dict[str, Any]:
        count = len(incidents)
        if count == 0:
            return {
                "total_incidents": 0,
                "critical_incidents": 0,
                "collisions_prevented": 0,
                "collision_avoidance_rate": 0.0,
                "avg_risk_level": 0.0,
                "avg_detection_time": 0.0,
                "avg_stop_time": 0.0,
                "avg_safety_margin": 0.0,
                "compliance_rate": 0.0,
                "detection_score": 0.0,
                "response_score": 0.0,
                "margin_score": 0.0,
                "composite_score": 0.0,
            }

        avg_risk = float(np.mean([incident.risk_level for incident in incidents]))
        avg_detection = float(np.mean([incident.hazard_detection_time for incident in incidents]))
        avg_stop = float(np.mean([incident.emergency_stop_time for incident in incidents]))
        avg_margin = float(np.mean([incident.safety_margin for incident in incidents]))
        compliance_rate = float(np.mean([incident.standards_compliance for incident in incidents]))
        collisions_prevented = int(sum(1 for incident in incidents if incident.collision_avoidance))
        critical_incidents = int(sum(1 for incident in incidents if incident.risk_level > self.thresholds["max_risk_level"]))

        detection_score = max(0.0, 1.0 - (avg_detection / self.thresholds["max_hazard_detection_time"]))
        response_score = max(0.0, 1.0 - (avg_stop / self.thresholds["max_emergency_stop_time"]))
        margin_score = min(1.0, avg_margin / self.thresholds["min_safety_margin"]) if self.thresholds["min_safety_margin"] > 0 else 0.0
        composite = (
            self.metric_weights["risk_level"] * (1.0 - avg_risk) +
            self.metric_weights["hazard_detection"] * detection_score +
            self.metric_weights["emergency_response"] * response_score +
            self.metric_weights["standards_compliance"] * compliance_rate
        )

        return {
            "total_incidents": count,
            "critical_incidents": critical_incidents,
            "collisions_prevented": collisions_prevented,
            "collision_avoidance_rate": collisions_prevented / count,
            "avg_risk_level": avg_risk,
            "avg_detection_time": avg_detection,
            "avg_stop_time": avg_stop,
            "avg_safety_margin": avg_margin,
            "compliance_rate": compliance_rate,
            "detection_score": detection_score,
            "response_score": response_score,
            "margin_score": margin_score,
            "composite_score": composite,
        }

    def _build_threshold_assessment(self, aggregates: Mapping[str, Any]) -> Dict[str, Any]:
        violations: List[str] = []
        if aggregates["avg_risk_level"] > self.thresholds["max_risk_level"]:
            violations.append("Average risk level exceeds configured maximum.")
        if aggregates["avg_detection_time"] > self.thresholds["max_hazard_detection_time"]:
            violations.append("Average hazard detection time exceeds configured maximum.")
        if aggregates["avg_stop_time"] > self.thresholds["max_emergency_stop_time"]:
            violations.append("Average emergency stop time exceeds configured maximum.")
        if aggregates["avg_safety_margin"] < self.thresholds["min_safety_margin"]:
            violations.append("Average safety margin is below the configured minimum.")
        if aggregates["compliance_rate"] < self.thresholds["min_standards_compliance"]:
            violations.append("Standards compliance rate is below the configured minimum.")
        return {
            "violations": violations,
            "passed": not violations,
        }

    def _build_diagnostics(
        self,
        aggregates: Mapping[str, Any],
        threshold_assessment: Mapping[str, Any],
        baseline_comparison: Mapping[str, Any],
        anomalies: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "anomaly_count": len(anomalies),
            "critical_incident_ratio": (aggregates["critical_incidents"] / aggregates["total_incidents"]) if aggregates["total_incidents"] else 0.0,
            "threshold_violation_count": len(threshold_assessment.get("violations", [])),
            "baseline_metrics_compared": sorted(baseline_comparison.keys()),
            "compliance_history_length": len(self.compliance_history),
        }

    def _detect_incident_violations(
        self,
        *,
        risk_level: float,
        hazard_detection_time: float,
        emergency_stop_time: float,
        safety_margin: float,
        standards_compliance: float,
    ) -> List[str]:
        violations: List[str] = []
        if risk_level > self.thresholds["max_risk_level"]:
            violations.append("risk_level")
        if hazard_detection_time > self.thresholds["max_hazard_detection_time"]:
            violations.append("hazard_detection_time")
        if emergency_stop_time > self.thresholds["max_emergency_stop_time"]:
            violations.append("emergency_stop_time")
        if safety_margin < self.thresholds["min_safety_margin"]:
            violations.append("safety_margin")
        if standards_compliance < self.thresholds["min_standards_compliance"]:
            violations.append("standards_compliance")
        return violations

    def _generate_incident_recommendations(self, violations: Sequence[str]) -> List[str]:
        recommendations: List[str] = []
        if "risk_level" in violations:
            recommendations.append("Reduce risk exposure through additional controls or safer motion envelopes.")
        if "hazard_detection_time" in violations:
            recommendations.append("Improve hazard-detection latency through faster sensing or more frequent evaluation cycles.")
        if "emergency_stop_time" in violations:
            recommendations.append("Optimize emergency-stop actuation and confirm redundancy in safety shutdown paths.")
        if "safety_margin" in violations:
            recommendations.append("Increase operational safety margins or slow the operating envelope under uncertainty.")
        if "standards_compliance" in violations:
            recommendations.append("Address gaps in applicable safety-standard coverage and verification evidence.")
        return recommendations

    def _generate_operation_recommendations(
        self,
        aggregates: Mapping[str, Any],
        threshold_assessment: Mapping[str, Any],
    ) -> List[str]:
        recommendations: List[str] = []
        if not threshold_assessment.get("violations"):
            recommendations.append("Operational safety indicators are within configured bounds; continue periodic review.")
        else:
            recommendations.extend(threshold_assessment["violations"])

        if aggregates["critical_incidents"] > 0:
            recommendations.append("Prioritize mitigation of high-risk incident categories and increase scenario-based testing in those modes.")
        if aggregates["collision_avoidance_rate"] < 1.0:
            recommendations.append("Investigate missed avoidance events and add guard policies or fallback behaviors.")
        return recommendations

    def _store_result(self, result: SafetyEvaluationResult) -> str:
        try:
            priority = "high" if result.threshold_assessment.get("violations") else "medium"
            return self.memory.add(
                entry=result.to_dict(),
                tags=["safety_evaluation", "robotics", "automation"],
                priority=priority,
                category="safety",
                source=self.__class__.__name__,
            )
        except Exception as exc:
            raise MemoryAccessError("add", "safety_evaluation_result", str(exc)) from exc

    def _update_visualizer(self, result: SafetyEvaluationResult) -> None:
        try:
            visualizer = get_visualizer()
            visualizer.update_metrics({
                "risk": result.aggregates["avg_risk_level"],
                "reward": result.aggregates["composite_score"],
                "pass_rate": max(0.0, min(1.0, result.aggregates["compliance_rate"])),
                "operational_time": result.aggregates["avg_detection_time"],
            })
        except Exception as exc:
            logger.warning("Visualizer update skipped for safety evaluator: %s", exc)

    def _load_thresholds(self, value: Any) -> Dict[str, float]:
        required = {
            "max_risk_level": 0.3,
            "max_hazard_detection_time": 0.5,
            "max_emergency_stop_time": 0.2,
            "min_safety_margin": 0.5,
            "min_standards_compliance": 0.95,
        }
        if not isinstance(value, Mapping):
            value = required
        thresholds = {}
        for key, default in required.items():
            thresholds[key] = self._require_non_negative_number(value.get(key, default), key)
        return thresholds

    def _normalize_weight_mapping(self, value: Any) -> Dict[str, float]:
        defaults = {
            "risk_level": 0.3,
            "hazard_detection": 0.25,
            "emergency_response": 0.25,
            "standards_compliance": 0.2,
        }
        if not isinstance(value, Mapping):
            return defaults
        normalized = {key: self._require_non_negative_number(value.get(key, defaults[key]), key) for key in defaults}
        total = sum(normalized.values())
        return {key: val / total for key, val in normalized.items()} if total > 0 else defaults

    @staticmethod
    def _normalize_string_list(value: Any, field_name: str) -> List[str]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValidationFailureError(field_name, type(value).__name__, "sequence of strings")
        normalized: List[str] = []
        seen: set[str] = set()
        for item in value:
            text = str(item).strip()
            if not text:
                continue
            key = text.casefold()
            if key not in seen:
                normalized.append(text)
                seen.add(key)
        if not normalized:
            raise ValidationFailureError(field_name, value, "non-empty sequence of strings")
        return normalized

    @staticmethod
    def _require_non_negative_number(value: Any, field_name: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValidationFailureError(field_name, value, "non-negative number") from exc
        if number < 0:
            raise ValidationFailureError(field_name, number, "non-negative number")
        return number

    @staticmethod
    def _coerce_probability(value: Any, field_name: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValidationFailureError(field_name, value, "probability in [0, 1]") from exc
        if number < 0 or number > 1:
            raise ValidationFailureError(field_name, number, "probability in [0, 1]")
        return number



def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Safety Evaluator ===\n")
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # Sample safety incidents
    incidents = [
        {
            'risk_assessment': {'collision': 0.4, 'pinch_point': 0.2},
            'hazard_detection_time': 0.3,
            'emergency_stop_time': 0.15,
            'safety_margin': 0.8,
            'collision_avoided': True,
            'ISO 13849': True,
            'IEC 61508': True,
            'hazard_details': {'collision': 0.6, 'pinch_point': 0.4}
        },
        {
            'risk_assessment': {'crush_hazard': 0.6, 'control_failure': 0.3},
            'hazard_detection_time': 0.7,
            'emergency_stop_time': 0.25,
            'safety_margin': 0.3,
            'collision_avoided': False,
            'ISO 13849': False,
            'IEC 61508': True,
            'hazard_details': {'crush_hazard': 0.8, 'control_failure': 0.2}
        }
    ]

    evaluator = SafetyEvaluator()
    results = evaluator.evaluate_operation(incidents)

    printer.pretty("Results:", results, "success" if results else "error")
    print(f"\nReport:\n{evaluator.generate_report(results)}")

    print("\n=== Safety Evaluation Complete ===\n")
