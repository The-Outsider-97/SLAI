"""
Intervention report generation for the alignment system.

This module produces a structured, auditable intervention package for high-risk
alignment events. It is designed to sit between the alignment pipeline and the
human oversight layer, preserving forensic context while remaining compatible
with the agent orchestration already present in the codebase.

Capabilities:
- Structured risk and violation reporting
- Counterfactual evidence packaging
- Timeline reconstruction from alignment history
- Actionable remediation recommendations
- Deterministic state fingerprinting
- Forensic snapshot capture for safe-state transitions
- Backward-compatible entrypoints for existing agent workflows
"""

import hashlib
import json

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from .config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter # type: ignore

logger = get_logger("Intervention Report")
printer = PrettyPrinter

@dataclass(frozen=True)
class ReportMetadata:
    """Top-level metadata used for routing, auditability, and protocol tracing."""

    agent_id: str
    agent_class: str
    report_timestamp: str
    intervention_level: str
    protocol_version: str
    report_id: str
    system_mode: str
    operational_state: str


@dataclass(frozen=True)
class TimelineEvent:
    """Normalized timeline record for recent alignment-related activity."""

    timestamp: str
    event_type: str
    risk_level: float
    triggered_correction: Optional[Dict[str, Any]] = None
    violation_types: List[str] = field(default_factory=list)
    source: str = "alignment_history"


@dataclass(frozen=True)
class Recommendation:
    """Actionable recommendation for human reviewers or automated remediation."""

    category: str
    priority: str
    rationale: str
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)


class InterventionReport:
    """
    Generate a comprehensive intervention report using the agent's current
    alignment state, audit context, and shared memory.

    The implementation is intentionally defensive around missing optional
    subsystems so the report can still be generated during degraded or
    fail-safe operation.
    """

    DEFAULT_PROTOCOL_VERSION = "3.0"
    DEFAULT_INTERVENTION_LEVEL = "CRITICAL"

    def __init__(
        self,
        agent: Any,
        intervention_level: str = DEFAULT_INTERVENTION_LEVEL,
        protocol_version: str = DEFAULT_PROTOCOL_VERSION,
    ):
        self.agent = agent
        self.timestamp = datetime.now().isoformat()
        self.intervention_level = intervention_level
        self.protocol_version = protocol_version
        self.report_id = self._build_report_id()

    @classmethod
    def build(
        cls,
        agent: Any,
        intervention_level: str = DEFAULT_INTERVENTION_LEVEL,
        protocol_version: str = DEFAULT_PROTOCOL_VERSION,
    ) -> Dict[str, Any]:
        """Primary convenience constructor for generating a report in one step."""
        return cls(
            agent=agent,
            intervention_level=intervention_level,
            protocol_version=protocol_version,
        ).generate()

    @classmethod
    def _generate_intervention_report(
        cls,
        agent: Any,
        intervention_level: str = DEFAULT_INTERVENTION_LEVEL,
        protocol_version: str = DEFAULT_PROTOCOL_VERSION,
    ) -> Dict[str, Any]:
        """
        Backward-compatible entrypoint for legacy agent call sites.

        This method is intentionally named to match the current orchestration
        path used in the uploaded alignment agent.
        """
        return cls.build(
            agent=agent,
            intervention_level=intervention_level,
            protocol_version=protocol_version,
        )

    def generate(self) -> Dict[str, Any]:
        """Generate the full intervention package with forensic evidence."""
        metadata = self._generate_metadata()
        risk_analysis = self._current_risk_assessment()
        counterfactuals = self._generate_counterfactual_examples()
        violation_timeline = self._construct_violation_timeline()
        recommendations = self._suggest_potential_fixes(risk_analysis, violation_timeline)
        forensic_snapshot = self._capture_forensic_state()

        return {
            "metadata": asdict(metadata),
            "executive_summary": self._generate_executive_summary(
                metadata=metadata,
                risk_analysis=risk_analysis,
                violation_timeline=violation_timeline,
                recommendations=recommendations,
            ),
            "risk_analysis": risk_analysis,
            "counterfactuals": counterfactuals,
            "violation_timeline": [asdict(event) for event in violation_timeline],
            "recommended_actions": [asdict(rec) for rec in recommendations],
            "system_state_fingerprint": self._compute_state_fingerprint(
                metadata=metadata,
                risk_analysis=risk_analysis,
            ),
            "forensic_snapshot": forensic_snapshot,
        }

    def _generate_metadata(self) -> ReportMetadata:
        agent_id = getattr(self.agent, "agent_id", None) or getattr(self.agent, "id", None) or "unknown"
        system_mode = self._read_memory_value("system_mode", default="UNKNOWN")
        operational_state = getattr(self.agent, "operational_state", "UNKNOWN")

        return ReportMetadata(
            agent_id=str(agent_id),
            agent_class=self.agent.__class__.__name__,
            report_timestamp=self.timestamp,
            intervention_level=self.intervention_level,
            protocol_version=self.protocol_version,
            report_id=self.report_id,
            system_mode=str(system_mode),
            operational_state=str(operational_state),
        )

    def _generate_executive_summary(
        self,
        metadata: ReportMetadata,
        risk_analysis: Dict[str, Any],
        violation_timeline: Sequence[TimelineEvent],
        recommendations: Sequence[Recommendation],
    ) -> Dict[str, Any]:
        component_risks = risk_analysis.get("component_risks", {})
        sorted_components = sorted(
            component_risks.items(),
            key=lambda item: float(item[1]) if self._is_numeric(item[1]) else 0.0,
            reverse=True,
        )

        return {
            "report_id": metadata.report_id,
            "agent_id": metadata.agent_id,
            "intervention_level": metadata.intervention_level,
            "total_risk": risk_analysis.get("total_risk", 0.0),
            "dominant_risk_components": [name for name, _ in sorted_components[:3]],
            "recent_event_count": len(violation_timeline),
            "ethical_violation_count": len(risk_analysis.get("ethical_violations_details", [])),
            "recommended_action_count": len(recommendations),
            "requires_human_review": True,
        }

    def _current_risk_assessment(self) -> Dict[str, Any]:
        """
        Compute the current risk profile using the richest available signal.

        Preference order:
        1. Agent's most recent alignment report
        2. Agent's own risk profile assembly helper
        3. Most recent alignment history entry
        4. Most recent scalar risk history value
        """
        report = getattr(self.agent, "last_alignment_report", None)
        if report is not None and hasattr(self.agent, "_assemble_risk_profile"):
            try:
                assembled = self.agent._assemble_risk_profile(report)
                if isinstance(assembled, dict):
                    return self._normalize_risk_profile(assembled)
            except Exception as exc:
                logger.warning("Risk profile assembly from last_alignment_report failed: %s", exc)

        latest_history = self._get_latest_alignment_history_record()
        if latest_history:
            history_risk = latest_history.get("risk_assessment") or latest_history.get("risk_profile")
            if isinstance(history_risk, dict):
                return self._normalize_risk_profile(history_risk)

        risk_history = getattr(self.agent, "risk_history", []) or []
        if risk_history:
            return {
                "total_risk": self._coerce_float(risk_history[-1]),
                "component_risks": {},
                "ethical_violations_details": [],
                "risk_source": "risk_history",
            }

        return {
            "total_risk": 0.0,
            "component_risks": {},
            "ethical_violations_details": [],
            "risk_source": "unavailable",
        }

    def _normalize_risk_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        component_risks = profile.get("component_risks", {})
        normalized_components = {
            str(key): self._coerce_float(value)
            for key, value in component_risks.items()
        } if isinstance(component_risks, dict) else {}

        return {
            "total_risk": self._coerce_float(profile.get("total_risk", 0.0)),
            "component_risks": normalized_components,
            "ethical_violations_details": list(profile.get("ethical_violations_details", []) or []),
            "risk_source": profile.get("risk_source", "agent_profile"),
        }

    def _generate_counterfactual_examples(self) -> Dict[str, Any]:
        """Generate explainable counterfactual evidence if the auditor supports it."""
        auditor = getattr(self.agent, "auditor", None)
        if auditor is None:
            return {
                "status": "unavailable",
                "reason": "agent has no auditor",
                "examples": [],
            }

        if hasattr(auditor, "generate_explainable_counterfactuals"):
            try:
                examples = auditor.generate_explainable_counterfactuals(
                    num_examples=3,
                    max_iterations=100,
                )
                return {
                    "status": "available",
                    "generator": "generate_explainable_counterfactuals",
                    "examples": examples,
                }
            except Exception as exc:
                logger.warning("Counterfactual generation failed: %s", exc)
                return {
                    "status": "error",
                    "reason": str(exc),
                    "examples": [],
                }

        if hasattr(auditor, "get_recent_counterfactuals"):
            try:
                recent = auditor.get_recent_counterfactuals()
                return {
                    "status": "available",
                    "generator": "get_recent_counterfactuals",
                    "examples": recent,
                }
            except Exception as exc:
                logger.warning("Recent counterfactual retrieval failed: %s", exc)
                return {
                    "status": "error",
                    "reason": str(exc),
                    "examples": [],
                }

        return {
            "status": "unavailable",
            "reason": "auditor does not expose counterfactual reporting methods",
            "examples": [],
        }

    def _construct_violation_timeline(self) -> List[TimelineEvent]:
        """Create a normalized timeline from the most recent alignment history records."""
        timeline: List[TimelineEvent] = []
        history = self._read_memory_value("alignment_history", default=[])
        if not isinstance(history, list):
            history = []

        for record in history[-10:]:
            if not isinstance(record, dict):
                continue

            risk_assessment = record.get("risk_assessment") or {}
            correction = record.get("correction") or record.get("applied_correction")
            timeline.append(
                TimelineEvent(
                    timestamp=str(record.get("timestamp", self.timestamp)),
                    event_type=self._infer_timeline_event_type(record),
                    risk_level=self._coerce_float(
                        risk_assessment.get("total_risk", record.get("risk_level", 0.0))
                    ),
                    triggered_correction=correction if isinstance(correction, dict) else None,
                    violation_types=self._extract_violation_types(record, risk_assessment),
                )
            )

        return timeline

    def _suggest_potential_fixes(
        self,
        risk_profile: Dict[str, Any],
        violation_timeline: Sequence[TimelineEvent],
    ) -> List[Recommendation]:
        """
        Generate repair recommendations grounded in the current risk profile,
        recent intervention history, and the alignment architecture.
        """
        recommendations: List[Recommendation] = []
        component_risks = risk_profile.get("component_risks", {})
        total_risk = self._coerce_float(risk_profile.get("total_risk", 0.0))
        ethical_violations = risk_profile.get("ethical_violations_details", [])

        if total_risk > 0:
            recommendations.append(
                Recommendation(
                    category="oversight",
                    priority="critical",
                    rationale="Current aggregate risk exceeds a nominal zero-deviation state and already triggered intervention packaging.",
                    action="Require explicit human disposition before resuming autonomous high-impact actions.",
                    parameters={
                        "expected_decisions": ["approve", "reject", "escalate", "defer"],
                        "risk_level": total_risk,
                    },
                )
            )

        if component_risks.get("statistical_parity", 0.0) > 0:
            recommendations.append(
                Recommendation(
                    category="fairness",
                    priority="high",
                    rationale="Statistical parity contributes materially to the current intervention state.",
                    action="Recalibrate demographic-parity thresholds and review subgroup monitoring coverage before resuming normal operation.",
                    parameters={
                        "component": "statistical_parity",
                        "observed_risk": component_risks.get("statistical_parity", 0.0),
                    },
                )
            )

        if component_risks.get("equal_opportunity", 0.0) > 0:
            recommendations.append(
                Recommendation(
                    category="fairness",
                    priority="high",
                    rationale="Equal opportunity deviation is present in the assembled risk profile.",
                    action="Inspect error-rate disparities across sensitive groups and tighten post-decision monitoring on false-positive and false-negative skew.",
                    parameters={
                        "component": "equal_opportunity",
                        "observed_risk": component_risks.get("equal_opportunity", 0.0),
                    },
                )
            )

        if ethical_violations:
            recommendations.append(
                Recommendation(
                    category="ethics",
                    priority="critical",
                    rationale="The ethical compliance layer reported concrete violations requiring rule-level remediation.",
                    action="Review constitutional constraints and add or strengthen targeted guardrails before exiting safe state.",
                    parameters={
                        "violation_count": len(ethical_violations),
                        "violations": ethical_violations,
                    },
                )
            )

        if len(violation_timeline) >= 3:
            recommendations.append(
                Recommendation(
                    category="stability",
                    priority="medium",
                    rationale="Repeated recent alignment events indicate a persistent rather than isolated intervention condition.",
                    action="Perform threshold recalibration and inspect concept-drift sensitivity prior to re-enabling full autonomy.",
                    parameters={
                        "recent_event_count": len(violation_timeline),
                    },
                )
            )

        if not recommendations:
            recommendations.append(
                Recommendation(
                    category="observation",
                    priority="low",
                    rationale="The report contains limited actionable risk detail, but intervention packaging was requested.",
                    action="Maintain safe monitoring mode and collect additional evidence before policy changes are applied.",
                    parameters={},
                )
            )

        return recommendations

    def _compute_state_fingerprint(
        self,
        metadata: ReportMetadata,
        risk_analysis: Dict[str, Any],
    ) -> str:
        """Create a deterministic hash of the critical system state at report time."""
        state = {
            "metadata": asdict(metadata),
            "risk_table": getattr(self.agent, "risk_table", {}),
            "risk_analysis": risk_analysis,
            "constraint_weights": getattr(getattr(self.agent, "ethics", None), "constraint_weights", {}),
            "config": getattr(self.agent, "config", {}),
            "timestamp": self.timestamp,
        }
        serialized = json.dumps(state, sort_keys=True, default=self._json_default)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _capture_forensic_state(self) -> Dict[str, Any]:
        """Preserve critical state for audit trails and safe-state transitions."""
        return {
            "decision_context": self._get_recent_actions(),
            "memory_snapshot": self._safe_get_latest_snapshot(),
            "ethical_state": self._safe_get_ethical_state(),
            "performance_metrics": self._read_memory_value("performance_metrics", default={}),
            "risk_table": getattr(self.agent, "risk_table", {}),
            "recent_adjustments": list(getattr(self.agent, "adjustment_history", [])[-10:]),
            "recent_risk_history": list(getattr(self.agent, "risk_history", [])[-20:]),
            "system_health": self._safe_check_system_health(),
        }

    def _get_recent_actions(self) -> List[Dict[str, Any]]:
        """Capture the most recent agent actions for intervention context."""
        actions = self._read_memory_value("action_history", default=[])
        if not isinstance(actions, list):
            return []
        return [action for action in actions[-5:] if isinstance(action, dict)]

    def _get_latest_alignment_history_record(self) -> Optional[Dict[str, Any]]:
        history = self._read_memory_value("alignment_history", default=[])
        if isinstance(history, list) and history:
            for record in reversed(history):
                if isinstance(record, dict):
                    return record
        return None

    def _read_memory_value(self, key: str, default: Any = None) -> Any:
        shared_memory = getattr(self.agent, "shared_memory", None)
        if shared_memory is None:
            return default

        getter = getattr(shared_memory, "get", None)
        if callable(getter):
            try:
                return getter(key, default)
            except TypeError:
                try:
                    return getter(key)
                except Exception:
                    return default
        return default

    def _safe_get_latest_snapshot(self) -> Dict[str, Any]:
        shared_memory = getattr(self.agent, "shared_memory", None)
        if shared_memory is None:
            return {}

        snapshotter = getattr(shared_memory, "get_latest_snapshot", None)
        if callable(snapshotter):
            try:
                snapshot = snapshotter()
                return snapshot if isinstance(snapshot, dict) else {"snapshot": snapshot}
            except Exception as exc:
                logger.warning("Snapshot capture failed: %s", exc)
                return {"error": str(exc)}
        return {}

    def _safe_get_ethical_state(self) -> Dict[str, Any]:
        ethics = getattr(self.agent, "ethics", None)
        if ethics is None:
            return {}

        getter = getattr(ethics, "get_current_state", None)
        if callable(getter):
            try:
                state = getter()
                return state if isinstance(state, dict) else {"state": state}
            except Exception as exc:
                logger.warning("Ethical state retrieval failed: %s", exc)
                return {"error": str(exc)}

        audit_log = getattr(ethics, "audit_log", None)
        if isinstance(audit_log, list):
            return {"recent_audit_log": audit_log[-10:]}
        return {}

    def _safe_check_system_health(self) -> Dict[str, Any]:
        checker = getattr(self.agent, "_check_system_health", None)
        if callable(checker):
            try:
                result = checker()
                return result if isinstance(result, dict) else {"result": result}
            except Exception as exc:
                logger.warning("System health check failed while building intervention report: %s", exc)
                return {"error": str(exc)}
        return {}

    def _infer_timeline_event_type(self, record: Dict[str, Any]) -> str:
        correction = record.get("correction") or record.get("applied_correction") or {}
        if isinstance(correction, dict):
            action = correction.get("action")
            if action:
                return f"correction:{action}"
        return "risk_anomaly"

    def _extract_violation_types(
        self,
        record: Dict[str, Any],
        risk_assessment: Dict[str, Any],
    ) -> List[str]:
        violations = []

        direct = record.get("ethical_violations_details")
        if isinstance(direct, list):
            violations.extend(str(v) for v in direct)

        nested = risk_assessment.get("ethical_violations_details") if isinstance(risk_assessment, dict) else None
        if isinstance(nested, list):
            violations.extend(str(v) for v in nested)

        deduplicated: List[str] = []
        for violation in violations:
            if violation not in deduplicated:
                deduplicated.append(violation)
        return deduplicated

    def _build_report_id(self) -> str:
        agent_id = getattr(self.agent, "agent_id", None) or getattr(self.agent, "id", None) or "unknown"
        seed = f"{agent_id}:{self.timestamp}:{self.__class__.__name__}"
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
        return f"ir-{digest}"

    @staticmethod
    def _coerce_float(value: Any) -> float:
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _is_numeric(value: Any) -> bool:
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _json_default(value: Any) -> Any:
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:
                pass
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)
