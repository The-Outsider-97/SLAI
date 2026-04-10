"""
- Auto-generated incident briefs.
- Suspected root-cause ranking.
- Suggested remediation runbooks.

This module is the incident-synthesis layer of the Observability subsystem.
It converts raw operational evidence into operator-facing incident intelligence:
ranked hypotheses, concise briefs, escalation level, and remediation guidance.

Design goals:
- Preserve a clear separation between signal ingestion and incident synthesis.
- Turn heterogeneous evidence into normalized, explainable incident signals.
- Keep classification, root-cause ranking, and runbook selection deterministic
  enough for automation, while still surfacing confidence and evidence.
- Integrate with waterfall analysis, capacity/performance outputs, and
  observability memory without hard-coding the rest of the subsystem.
- Use the shared observability error model for ambiguity, clustering,
  runbook lookup, and ranking failures rather than silently degrading.
"""

from __future__ import annotations

import hashlib
import json
import time

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from threading import RLock
from typing import Any, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .utils import (load_global_config, get_config_section,
                                  # waterfall analysis
                                  WaterfallAnalyzer, summarize_waterfall,
                                  # observability error
                                  ObservabilityErrorType, ObservabilitySeverity,
                                  ObservabilityError, normalize_observability_exception)
from .observability_memory import ObservabilityMemory
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Observability Intelligence")
printer = PrettyPrinter


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class IncidentSignal:
    signal_type: str
    source: str
    level: str
    title: str
    description: str
    score: float
    timestamp_ms: float
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RootCauseHypothesis:
    cause_id: str
    label: str
    confidence: float
    score: float
    rationale: str
    evidence: List[str] = field(default_factory=list)
    related_agents: List[str] = field(default_factory=list)
    related_spans: List[str] = field(default_factory=list)
    related_signals: List[str] = field(default_factory=list)
    remediation_hint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunbookRecommendation:
    playbook_id: str
    title: str
    priority: int
    confidence: float
    rationale: str
    actions: List[str] = field(default_factory=list)
    automation_safe: bool = False
    estimated_impact: Optional[str] = None
    historical_success_rate: Optional[float] = None
    historical_attempt_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IncidentBrief:
    incident_id: str
    incident_level: str
    status: str
    summary: str
    customer_impact: str
    trace_id: Optional[str]
    error_signature: str
    started_at_ms: float
    generated_at_ms: float
    primary_symptoms: List[str] = field(default_factory=list)
    top_root_causes: List[Dict[str, Any]] = field(default_factory=list)
    recommended_runbooks: List[Dict[str, Any]] = field(default_factory=list)
    evidence_snapshot: Dict[str, Any] = field(default_factory=dict)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    similar_incidents: List[Dict[str, Any]] = field(default_factory=list)
    suppress_duplicate_alert: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IncidentAssessment:
    incident_id: str
    incident_level: str
    status: str
    score: float
    trace_id: Optional[str]
    error_signature: str
    signals: List[Dict[str, Any]]
    root_causes: List[Dict[str, Any]]
    runbooks: List[Dict[str, Any]]
    brief: Dict[str, Any]
    waterfall_summary: Dict[str, Any] = field(default_factory=dict)
    capacity_summary: Dict[str, Any] = field(default_factory=dict)
    performance_summary: Dict[str, Any] = field(default_factory=dict)
    alert_fatigue: Dict[str, Any] = field(default_factory=dict)
    similar_incidents: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
_LEVEL_ORDER = {"info": 10, "warning": 20, "critical": 30}


def _now_ms() -> float:
    return time.time() * 1000.0


def _coerce_mapping(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"expected mapping, received {type(value).__name__}")
    return {str(key): payload for key, payload in value.items()}

def _iter_record_mappings(
    value: Any,
    *,
    key_field: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if value is None:
        return []

    if isinstance(value, Mapping):
        records: List[Dict[str, Any]] = []

        # Dict-of-records: {"subject_a": {...}, "subject_b": {...}}
        if all(isinstance(item, Mapping) for item in value.values()):
            for key, item in value.items():
                record = dict(item)
                if key_field and key_field not in record:
                    record[key_field] = str(key)
                records.append(record)
            return records

        # Single record: {"level": "warning", ...}
        return [dict(value)]

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        records = []
        for item in value:
            if isinstance(item, Mapping):
                records.append(dict(item))
        return records

    return []

def _coerce_sequence(value: Optional[Iterable[Any]]) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    return list(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _truncate_text(value: str, limit: int = 280) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    output: List[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _level_rank(level: str) -> int:
    return _LEVEL_ORDER.get(str(level or "info").lower(), 10)


def _normalize_level(level: Any) -> str:
    text = str(level or "info").strip().lower()
    if text in {"critical", "crit", "sev1", "p1", "fatal"}:
        return "critical"
    if text in {"warning", "warn", "high", "medium", "sev2", "p2"}:
        return "warning"
    return "info"


# ---------------------------------------------------------------------------
# Main intelligence implementation
# ---------------------------------------------------------------------------
class ObservabilityIntelligence:
    def __init__(self) -> None:
        self.config = load_global_config()
        self.intel_config = get_config_section("observability_intelligence")
        self._lock = RLock()

        self.enabled = bool(self.intel_config.get("enabled", True))
        self.enable_memory_integration = bool(self.intel_config.get("enable_memory_integration", False))
        self.similar_incident_limit = int(self.intel_config.get("similar_incident_limit", 5))
        self.root_cause_top_k = int(self.intel_config.get("root_cause_top_k", 5))
        self.runbook_top_k = int(self.intel_config.get("runbook_top_k", 4))
        self.timeline_event_limit = int(self.intel_config.get("timeline_event_limit", 100))
        self.incident_history_limit = int(self.intel_config.get("incident_history_limit", 500))
        self.alert_fatigue_window_seconds = float(self.intel_config.get("alert_fatigue_window_seconds", 1800.0))
        self.alert_fatigue_repeat_threshold = int(self.intel_config.get("alert_fatigue_repeat_threshold", 4))

        self.latency_warning_ms = float(self.intel_config.get("latency_warning_ms", 1000.0))
        self.latency_critical_ms = float(self.intel_config.get("latency_critical_ms", 2500.0))
        self.critical_path_ratio_warning = float(self.intel_config.get("critical_path_ratio_warning", 0.60))
        self.critical_path_ratio_critical = float(self.intel_config.get("critical_path_ratio_critical", 0.85))
        self.anomaly_warning_count = int(self.intel_config.get("anomaly_warning_count", 1))
        self.anomaly_critical_count = int(self.intel_config.get("anomaly_critical_count", 3))
        self.retry_chain_warning_count = int(self.intel_config.get("retry_chain_warning_count", 1))
        self.retry_chain_critical_count = int(self.intel_config.get("retry_chain_critical_count", 3))
        self.bottleneck_warning_count = int(self.intel_config.get("bottleneck_warning_count", 1))
        self.bottleneck_critical_count = int(self.intel_config.get("bottleneck_critical_count", 3))

        score_thresholds = _coerce_mapping(self.intel_config.get("incident_score_thresholds", {}))
        self.warning_score_threshold = float(score_thresholds.get("warning", 4.0))
        self.critical_score_threshold = float(score_thresholds.get("critical", 9.0))

        self.signal_weights = {
            key: float(value)
            for key, value in _coerce_mapping(
                self.intel_config.get(
                    "signal_weights",
                    {"info": 1.0, "warning": 3.0, "critical": 6.0},
                )
            ).items()
        }

        self.root_cause_weights = {
            key: float(value)
            for key, value in _coerce_mapping(self.intel_config.get("root_cause_weights", {})).items()
        }
        self.runbook_catalog = _coerce_mapping(self.intel_config.get("runbook_catalog", {}))
        self.default_runbooks = _coerce_sequence(self.intel_config.get("default_runbooks", []))
        self.impact_templates = _coerce_mapping(self.intel_config.get("impact_templates", {}))

        self.memory: Optional[ObservabilityMemory] = None
        if self.enable_memory_integration:
            try:
                self.memory = ObservabilityMemory()
            except Exception as exc:
                logger.warning("ObservabilityMemory initialization failed; proceeding without memory integration: %s", exc)
                self.memory = None

        self.waterfall_analyzer = WaterfallAnalyzer()
        self._recent_incidents: Deque[Tuple[float, str]] = deque(maxlen=self.incident_history_limit)
        self._recent_signatures: Counter[str] = Counter()

    # ------------------------------------------------------------------
    # Public orchestration APIs
    # ------------------------------------------------------------------
    def synthesize_incident(
        self,
        *,
        incident_id: str,
        spans: Optional[Sequence[Mapping[str, Any]]] = None,
        waterfall_summary: Optional[Mapping[str, Any]] = None,
        performance_report: Optional[Mapping[str, Any]] = None,
        capacity_report: Optional[Mapping[str, Any]] = None,
        alerts: Optional[Sequence[Mapping[str, Any]]] = None,
        error_events: Optional[Sequence[Mapping[str, Any]]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        incident_started_at_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        try:
            self._ensure_enabled(operation="synthesize_incident")
            incident_id = self._require_non_empty_str(
                incident_id,
                field_name="incident_id",
                operation="synthesize_incident",
            )

            metadata_dict = _coerce_mapping(metadata)
            performance_view = _coerce_mapping(performance_report)
            capacity_view = _coerce_mapping(capacity_report)
            waterfall_view = self._derive_waterfall_summary(
                spans=spans,
                waterfall_summary=waterfall_summary,
                performance_report=performance_view,
            )

            alert_records = self._normalize_event_records(alerts or [], source="alerts")
            error_records = self._normalize_event_records(error_events or [], source="errors")
            signals = self._collect_signals(
                incident_id=incident_id,
                waterfall_summary=waterfall_view,
                performance_report=performance_view,
                capacity_report=capacity_view,
                alert_records=alert_records,
                error_records=error_records,
            )

            if not signals and not waterfall_view and not performance_view and not capacity_view and not alert_records and not error_records:
                raise ObservabilityError(
                    message=f"Incident '{incident_id}' has insufficient evidence for classification",
                    error_type=ObservabilityErrorType.INCIDENT_CLASSIFICATION_AMBIGUOUS,
                    severity=ObservabilitySeverity.MEDIUM,
                    retryable=True,
                    context={"incident_id": incident_id},
                    remediation="Provide spans, performance findings, capacity findings, or alert evidence before incident synthesis.",
                )

            error_signature = self._build_error_signature(
                incident_id=incident_id,
                signals=signals,
                waterfall_summary=waterfall_view,
                capacity_report=capacity_view,
                performance_report=performance_view,
            )
            similar_incidents = self.cluster_related_incidents(error_signature=error_signature)
            alert_fatigue = self._assess_alert_fatigue(error_signature, signals, alert_records)
            incident_level, incident_status, incident_score = self._classify_incident_level(signals, alert_fatigue)

            root_causes = self.rank_root_causes(
                incident_id=incident_id,
                signals=signals,
                waterfall_summary=waterfall_view,
                capacity_report=capacity_view,
                performance_report=performance_view,
                similar_incidents=similar_incidents,
            )
            runbooks = self.suggest_runbooks(
                incident_id=incident_id,
                incident_level=incident_level,
                root_causes=root_causes,
                signals=signals,
            )
            timeline = self._build_incident_timeline(alert_records, error_records)
            brief = self.generate_incident_brief(
                incident_id=incident_id,
                incident_level=incident_level,
                incident_status=incident_status,
                signals=signals,
                root_causes=root_causes,
                runbooks=runbooks,
                timeline=timeline,
                error_signature=error_signature,
                trace_id=self._extract_trace_id(waterfall_view, performance_view, metadata_dict),
                incident_started_at_ms=float(incident_started_at_ms or self._infer_incident_start(timeline) or _now_ms()),
                similar_incidents=similar_incidents,
                alert_fatigue=alert_fatigue,
                waterfall_summary=waterfall_view,
                capacity_report=capacity_view,
                performance_report=performance_view,
            )

            self._record_recent_signature(error_signature)
            self._persist_incident_context(
                incident_id=incident_id,
                error_signature=error_signature,
                incident_level=incident_level,
                signals=signals,
                root_causes=root_causes,
                runbooks=runbooks,
                timeline=timeline,
                metadata=metadata_dict,
            )

            assessment = IncidentAssessment(
                incident_id=incident_id,
                incident_level=incident_level,
                status=incident_status,
                score=round(incident_score, 6),
                trace_id=brief.trace_id,
                error_signature=error_signature,
                signals=[signal.to_dict() for signal in signals],
                root_causes=[candidate.to_dict() for candidate in root_causes],
                runbooks=[runbook.to_dict() for runbook in runbooks],
                brief=brief.to_dict(),
                waterfall_summary=waterfall_view,
                capacity_summary=capacity_view,
                performance_summary=performance_view,
                alert_fatigue=alert_fatigue,
                similar_incidents=similar_incidents,
            )
            return assessment.to_dict()
        except Exception as exc:
            raise normalize_observability_exception(
                exc,
                stage="incident.classify",
                context={"incident_id": incident_id if "incident_id" in locals() else None},
            ) from exc

    def cluster_related_incidents(self, *, error_signature: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        try:
            normalized_signature = self._require_non_empty_str(
                error_signature,
                field_name="error_signature",
                operation="cluster_related_incidents",
            )
            max_results = int(limit if limit is not None else self.similar_incident_limit)
            if self.memory is None or not hasattr(self.memory, "incident_similarities"):
                return []
            result = self.memory.incident_similarities(normalized_signature, limit=max_results)
            return [dict(item) for item in _coerce_sequence(result)]
        except Exception as exc:
            raise normalize_observability_exception(
                exc,
                stage="incident.cluster",
                context={"error_signature": error_signature},
            ) from exc

    def rank_root_causes(
        self,
        *,
        incident_id: str,
        signals: Sequence[IncidentSignal],
        waterfall_summary: Optional[Mapping[str, Any]] = None,
        capacity_report: Optional[Mapping[str, Any]] = None,
        performance_report: Optional[Mapping[str, Any]] = None,
        similar_incidents: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> List[RootCauseHypothesis]:
        try:
            if not signals and not waterfall_summary and not capacity_report and not performance_report:
                raise ObservabilityError(
                    message=f"No evidence available to rank root causes for '{incident_id}'",
                    error_type=ObservabilityErrorType.RCA_GENERATION_FAILED,
                    severity=ObservabilitySeverity.HIGH,
                    retryable=True,
                    context={"incident_id": incident_id},
                    remediation="Provide incident signals or supporting reports before ranking root causes.",
                )

            candidate_state: MutableMapping[str, Dict[str, Any]] = defaultdict(
                lambda: {
                    "label": "",
                    "score": 0.0,
                    "evidence": [],
                    "related_agents": [],
                    "related_spans": [],
                    "related_signals": [],
                    "remediation_hint": None,
                }
            )

            def add_candidate(
                cause_id: str,
                *,
                label: str,
                score: float,
                evidence: Optional[str] = None,
                agent_names: Optional[Sequence[str]] = None,
                span_ids: Optional[Sequence[str]] = None,
                signal_type: Optional[str] = None,
                remediation_hint: Optional[str] = None,
            ) -> None:
                state = candidate_state[cause_id]
                state["label"] = state["label"] or label
                state["score"] += float(score)
                if evidence:
                    state["evidence"].append(str(evidence))
                if agent_names:
                    state["related_agents"].extend(str(agent) for agent in agent_names if agent)
                if span_ids:
                    state["related_spans"].extend(str(span) for span in span_ids if span)
                if signal_type:
                    state["related_signals"].append(str(signal_type))
                if remediation_hint and not state["remediation_hint"]:
                    state["remediation_hint"] = remediation_hint

            for signal in signals:
                level_weight = self._signal_weight(signal.level)
                signal_type = signal.signal_type
                context = signal.context
                evidence = signal.description

                if signal_type in {"resource_pressure", "resource_saturation", "cpu_pressure", "memory_pressure", "gpu_pressure"}:
                    add_candidate(
                        "resource_saturation",
                        label="Resource saturation",
                        score=1.4 * level_weight,
                        evidence=evidence,
                        agent_names=[context.get("agent_name")] if context.get("agent_name") else None,
                        signal_type=signal_type,
                        remediation_hint="Relieve host pressure, rebalance work, or scale the saturated resource.",
                    )
                if signal_type in {"queue_backlog", "queue_growth", "queue_drain_failure"}:
                    add_candidate(
                        "queue_saturation",
                        label="Queue saturation or consumer imbalance",
                        score=1.3 * level_weight,
                        evidence=evidence,
                        signal_type=signal_type,
                        remediation_hint="Drain backlog, inspect worker health, and rebalance consumers.",
                    )
                if signal_type in {"latency_regression", "critical_path_latency", "bottleneck_hotspot"}:
                    agent_name = context.get("agent_name")
                    dominant_span = context.get("span_id")
                    add_candidate(
                        "agent_latency_bottleneck" if not agent_name else f"agent_latency_bottleneck:{agent_name}",
                        label=(
                            "Agent latency bottleneck"
                            if not agent_name
                            else f"Agent latency bottleneck: {agent_name}"
                        ),
                        score=1.25 * level_weight,
                        evidence=evidence,
                        agent_names=[agent_name] if agent_name else None,
                        span_ids=[dominant_span] if dominant_span else None,
                        signal_type=signal_type,
                        remediation_hint="Inspect the critical path and reduce the dominant agent's exclusive latency.",
                    )
                if signal_type in {"retry_storm", "retry_waterfall"}:
                    add_candidate(
                        "retry_amplification",
                        label="Retry amplification loop",
                        score=1.2 * level_weight,
                        evidence=evidence,
                        signal_type=signal_type,
                        remediation_hint="Cap retries, add jitter/backoff, and remove recursive retry cascades.",
                    )
                if signal_type in {"timeout_error", "error_burst", "multi_agent_failure"}:
                    add_candidate(
                        "downstream_dependency_failure",
                        label="Downstream dependency failure or timeout",
                        score=1.35 * level_weight,
                        evidence=evidence,
                        signal_type=signal_type,
                        remediation_hint="Inspect external dependencies, timeout budgets, and fallback behavior.",
                    )
                if signal_type in {"slo_breach", "throughput_regression"}:
                    add_candidate(
                        "service_degradation",
                        label="User-facing service degradation",
                        score=1.1 * level_weight,
                        evidence=evidence,
                        signal_type=signal_type,
                        remediation_hint="Apply degraded mode, reduce load, and restore contractual performance.",
                    )

            waterfall_view = _coerce_mapping(waterfall_summary)
            top_bottlenecks = _coerce_sequence(waterfall_view.get("bottleneck_spans"))
            anomalies = _coerce_sequence(waterfall_view.get("anomalies"))
            critical_path = _coerce_sequence(waterfall_view.get("critical_path_span_ids"))
            per_agent_duration = _coerce_mapping(waterfall_view.get("per_agent_duration_ms"))

            if top_bottlenecks:
                first = _coerce_mapping(top_bottlenecks[0])
                agent_name = first.get("agent_name")
                span_id = first.get("span_id")
                add_candidate(
                    "agent_latency_bottleneck" if not agent_name else f"agent_latency_bottleneck:{agent_name}",
                    label="Critical path bottleneck" if not agent_name else f"Critical path bottleneck: {agent_name}",
                    score=2.0,
                    evidence=f"Top waterfall bottleneck is span '{span_id}' owned by '{agent_name}'.",
                    agent_names=[agent_name] if agent_name else None,
                    span_ids=[span_id] if span_id else None,
                    signal_type="bottleneck_hotspot",
                    remediation_hint="Inspect the dominant bottleneck span and reduce synchronous waiting on the critical path.",
                )

            if anomalies:
                timeout_like = [item for item in anomalies if str(_coerce_mapping(item).get("type", "")).lower().startswith("timeout")]
                error_like = [item for item in anomalies if _coerce_mapping(item).get("type") == "error_status"]
                if timeout_like or error_like:
                    add_candidate(
                        "downstream_dependency_failure",
                        label="Downstream dependency failure or timeout",
                        score=1.5 + (0.25 * len(timeout_like)) + (0.15 * len(error_like)),
                        evidence=f"Waterfall anomalies include {len(timeout_like)} timeout-like and {len(error_like)} error-like spans.",
                        agent_names=[_coerce_mapping(item).get("agent_name") for item in anomalies],
                        span_ids=[_coerce_mapping(item).get("span_id") for item in anomalies],
                        signal_type="timeout_error",
                        remediation_hint="Inspect dependency timeouts and error bursts along the trace path.",
                    )

            if len(set(str(agent) for agent in per_agent_duration if agent)) >= 3 and len(anomalies) >= 2:
                add_candidate(
                    "cross_agent_cascade",
                    label="Cross-agent cascading degradation",
                    score=1.9,
                    evidence="Multiple agents appear in the trace breakdown together with repeated anomalies.",
                    agent_names=list(per_agent_duration.keys()),
                    span_ids=critical_path,
                    signal_type="multi_agent_failure",
                    remediation_hint="Stabilize the highest-traffic edge first and isolate the dependency fan-out causing cross-agent spread.",
                )

            for incident in _coerce_sequence(similar_incidents):
                incident_view = _coerce_mapping(incident)
                for cause_name in _coerce_sequence(incident_view.get("root_cause_candidates")):
                    normalized_cause = str(cause_name).strip().lower().replace(" ", "_")
                    add_candidate(
                        normalized_cause,
                        label=str(cause_name),
                        score=0.8,
                        evidence=f"Similar incident history references root cause '{cause_name}'.",
                        signal_type="historical_similarity",
                    )

            if not candidate_state:
                add_candidate(
                    "insufficient_discriminators",
                    label="Insufficient discriminating evidence",
                    score=1.0,
                    evidence="The available evidence does not isolate a single dominant failure mode.",
                    remediation_hint="Collect more trace, queue, and dependency evidence before automating remediation.",
                )

            ranked = sorted(
                candidate_state.items(),
                key=lambda item: (item[1]["score"] + self.root_cause_weights.get(item[0], 0.0)),
                reverse=True,
            )
            max_score = max((state["score"] + self.root_cause_weights.get(cause_id, 0.0) for cause_id, state in ranked), default=1.0)

            output: List[RootCauseHypothesis] = []
            for cause_id, state in ranked[: self.root_cause_top_k]:
                adjusted_score = state["score"] + self.root_cause_weights.get(cause_id, 0.0)
                confidence = min(1.0, adjusted_score / max_score) if max_score > 0 else 0.0
                rationale = self._build_root_cause_rationale(
                    label=state["label"],
                    evidence=state["evidence"],
                    related_signals=state["related_signals"],
                )
                output.append(
                    RootCauseHypothesis(
                        cause_id=cause_id,
                        label=state["label"] or cause_id.replace("_", " ").title(),
                        confidence=round(confidence, 6),
                        score=round(adjusted_score, 6),
                        rationale=rationale,
                        evidence=_dedupe_preserve_order(state["evidence"])[:6],
                        related_agents=_dedupe_preserve_order(state["related_agents"]),
                        related_spans=_dedupe_preserve_order(state["related_spans"]),
                        related_signals=_dedupe_preserve_order(state["related_signals"]),
                        remediation_hint=state["remediation_hint"],
                    )
                )
            return output
        except Exception as exc:
            raise normalize_observability_exception(
                exc,
                stage="incident.rca",
                context={"incident_id": incident_id},
            ) from exc

    def suggest_runbooks(
        self,
        *,
        incident_id: str,
        incident_level: str,
        root_causes: Sequence[RootCauseHypothesis],
        signals: Sequence[IncidentSignal],
    ) -> List[RunbookRecommendation]:
        try:
            catalog = self.runbook_catalog
            candidate_runbooks: Dict[str, RunbookRecommendation] = {}
            signal_types = {signal.signal_type for signal in signals}

            def merge_runbook(
                playbook_id: str,
                *,
                title: str,
                confidence: float,
                priority: int,
                rationale: str,
                actions: Optional[Sequence[str]] = None,
                automation_safe: bool = False,
                estimated_impact: Optional[str] = None,
            ) -> None:
                existing = candidate_runbooks.get(playbook_id)
                if existing is None:
                    candidate_runbooks[playbook_id] = RunbookRecommendation(
                        playbook_id=playbook_id,
                        title=title,
                        priority=priority,
                        confidence=round(confidence, 6),
                        rationale=rationale,
                        actions=[str(item) for item in (actions or []) if str(item).strip()],
                        automation_safe=bool(automation_safe),
                        estimated_impact=estimated_impact,
                    )
                    return

                existing.priority = min(existing.priority, priority)
                existing.confidence = max(existing.confidence, round(confidence, 6))
                if rationale and rationale not in existing.rationale:
                    existing.rationale = f"{existing.rationale} {rationale}".strip()
                existing.actions = _dedupe_preserve_order(existing.actions + [str(item) for item in (actions or [])])
                existing.automation_safe = existing.automation_safe or bool(automation_safe)
                if existing.estimated_impact is None and estimated_impact:
                    existing.estimated_impact = estimated_impact

            for index, cause in enumerate(root_causes, start=1):
                raw_entries = _coerce_sequence(catalog.get(cause.cause_id))
                fallback_entries = _coerce_sequence(catalog.get(cause.cause_id.split(":", 1)[0]))
                runbook_entries = raw_entries or fallback_entries
                if not runbook_entries and index == 1 and self.default_runbooks:
                    runbook_entries = self.default_runbooks

                for entry in runbook_entries:
                    entry_view = _coerce_mapping(entry)
                    playbook_id = str(entry_view.get("playbook_id") or "").strip()
                    if not playbook_id:
                        raise ObservabilityError(
                            message=f"Runbook entry for cause '{cause.cause_id}' is missing 'playbook_id'",
                            error_type=ObservabilityErrorType.RUNBOOK_LOOKUP_FAILED,
                            severity=ObservabilitySeverity.MEDIUM,
                            retryable=True,
                            context={"incident_id": incident_id, "cause_id": cause.cause_id, "entry": entry_view},
                            remediation="Add a valid playbook_id to the runbook catalog configuration.",
                        )

                    historical_success_rate, attempt_count = self._lookup_runbook_history(playbook_id)
                    historical_bonus = 0.0
                    if historical_success_rate is not None:
                        historical_bonus = max(-0.15, min(0.15, (historical_success_rate - 0.5) * 0.5))

                    merge_runbook(
                        playbook_id,
                        title=str(entry_view.get("title") or playbook_id),
                        confidence=min(1.0, cause.confidence + historical_bonus),
                        priority=_safe_int(entry_view.get("priority"), default=index),
                        rationale=(
                            f"Selected for root cause '{cause.label}'"
                            + (f" with historical success rate {historical_success_rate:.2f}" if historical_success_rate is not None else "")
                        ),
                        actions=_coerce_sequence(entry_view.get("actions")),
                        automation_safe=bool(entry_view.get("automation_safe", False)),
                        estimated_impact=entry_view.get("estimated_impact"),
                    )
                    candidate_runbooks[playbook_id].historical_success_rate = historical_success_rate
                    candidate_runbooks[playbook_id].historical_attempt_count = attempt_count

            if not candidate_runbooks and incident_level in {"warning", "critical"}:
                merge_runbook(
                    "fallback-observability-triage",
                    title="Fallback observability triage",
                    confidence=0.45,
                    priority=99,
                    rationale="No catalog runbook matched the current incident evidence; use the generic triage workflow.",
                    actions=[
                        "Inspect the critical path and dominant bottlenecks.",
                        "Compare queue backlog and resource pressure against recent baseline.",
                        "Apply the platform degraded-mode policy if user-facing latency is at risk.",
                    ],
                    automation_safe=False,
                    estimated_impact="Provides safe manual stabilization while a specific runbook is identified.",
                )

            prioritized = sorted(
                candidate_runbooks.values(),
                key=lambda item: (item.priority, -item.confidence, item.title),
            )[: self.runbook_top_k]

            if not prioritized and signal_types:
                raise ObservabilityError(
                    message=f"No remediation runbooks could be resolved for incident '{incident_id}'",
                    error_type=ObservabilityErrorType.RUNBOOK_LOOKUP_FAILED,
                    severity=ObservabilitySeverity.MEDIUM,
                    retryable=True,
                    context={"incident_id": incident_id, "signal_types": sorted(signal_types)},
                    remediation="Populate the runbook catalog for the observed signal families or add default runbooks.",
                )

            return prioritized
        except Exception as exc:
            raise normalize_observability_exception(
                exc,
                stage="incident.runbook",
                context={"incident_id": incident_id},
            ) from exc

    def generate_incident_brief(
        self,
        *,
        incident_id: str,
        incident_level: str,
        incident_status: str,
        signals: Sequence[IncidentSignal],
        root_causes: Sequence[RootCauseHypothesis],
        runbooks: Sequence[RunbookRecommendation],
        timeline: Sequence[Mapping[str, Any]],
        error_signature: str,
        trace_id: Optional[str],
        incident_started_at_ms: float,
        similar_incidents: Sequence[Mapping[str, Any]],
        alert_fatigue: Mapping[str, Any],
        waterfall_summary: Optional[Mapping[str, Any]] = None,
        capacity_report: Optional[Mapping[str, Any]] = None,
        performance_report: Optional[Mapping[str, Any]] = None,
    ) -> IncidentBrief:
        try:
            sorted_signals = sorted(signals, key=lambda signal: (-_level_rank(signal.level), -signal.score, signal.title))
            primary_symptoms = [signal.title for signal in sorted_signals[: int(self.intel_config.get("incident_brief_max_symptoms", 6))]]
            top_root_causes = [candidate.to_dict() for candidate in root_causes[: self.root_cause_top_k]]
            recommended_runbooks = [runbook.to_dict() for runbook in runbooks[: self.runbook_top_k]]
            evidence_snapshot = self._build_evidence_snapshot(
                signals=signals,
                waterfall_summary=waterfall_summary,
                capacity_report=capacity_report,
                performance_report=performance_report,
            )
            customer_impact = self._derive_customer_impact(incident_level, signals, waterfall_summary, capacity_report)
            summary = self._build_incident_summary(
                incident_id=incident_id,
                incident_level=incident_level,
                signals=sorted_signals,
                root_causes=root_causes,
            )

            return IncidentBrief(
                incident_id=incident_id,
                incident_level=incident_level,
                status=incident_status,
                summary=summary,
                customer_impact=customer_impact,
                trace_id=trace_id,
                error_signature=error_signature,
                started_at_ms=float(incident_started_at_ms),
                generated_at_ms=_now_ms(),
                primary_symptoms=primary_symptoms,
                top_root_causes=top_root_causes,
                recommended_runbooks=recommended_runbooks,
                evidence_snapshot=evidence_snapshot,
                timeline=[dict(item) for item in timeline[: self.timeline_event_limit]],
                similar_incidents=[dict(item) for item in similar_incidents[: self.similar_incident_limit]],
                suppress_duplicate_alert=bool(_coerce_mapping(alert_fatigue).get("suppress_duplicate_alert", False)),
            )
        except Exception as exc:
            raise normalize_observability_exception(
                exc,
                stage="incident.brief",
                context={"incident_id": incident_id},
            ) from exc

    # ------------------------------------------------------------------
    # Evidence collection and normalization
    # ------------------------------------------------------------------
    def _collect_signals(
        self,
        *,
        incident_id: str,
        waterfall_summary: Mapping[str, Any],
        performance_report: Mapping[str, Any],
        capacity_report: Mapping[str, Any],
        alert_records: Sequence[Mapping[str, Any]],
        error_records: Sequence[Mapping[str, Any]],
    ) -> List[IncidentSignal]:
        signals: List[IncidentSignal] = []
        signals.extend(self._signals_from_waterfall(waterfall_summary))
        signals.extend(self._signals_from_performance(performance_report))
        signals.extend(self._signals_from_capacity(capacity_report))
        signals.extend(self._signals_from_event_records(alert_records, default_source="alerts"))
        signals.extend(self._signals_from_event_records(error_records, default_source="errors"))

        deduped: Dict[Tuple[str, str, str, str], IncidentSignal] = {}
        for signal in signals:
            dedupe_key = (signal.signal_type, signal.source, signal.level, signal.title)
            current = deduped.get(dedupe_key)
            if current is None or signal.score > current.score:
                deduped[dedupe_key] = signal

        output = sorted(
            deduped.values(),
            key=lambda signal: (-_level_rank(signal.level), -signal.score, signal.title),
        )
        logger.info("Synthesized %s incident signals for '%s'.", len(output), incident_id)
        return output

    def _derive_waterfall_summary(
        self,
        *,
        spans: Optional[Sequence[Mapping[str, Any]]],
        waterfall_summary: Optional[Mapping[str, Any]],
        performance_report: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if spans:
            report = self.waterfall_analyzer.analyze(spans)
            return summarize_waterfall(report)

        provided_summary = _coerce_mapping(waterfall_summary)
        if provided_summary:
            return provided_summary

        for key in ("waterfall_summary", "trace_summary", "trace_report"):
            nested = _coerce_mapping(performance_report.get(key))
            if nested:
                return nested
        return {}

    def _signals_from_waterfall(self, summary: Mapping[str, Any]) -> List[IncidentSignal]:
        if not summary:
            return []

        signals: List[IncidentSignal] = []
        trace_id = summary.get("trace_id")
        total_duration_ms = _safe_float(summary.get("total_duration_ms"))
        critical_path_ms = _safe_float(summary.get("critical_path_ms"))
        anomaly_count = _safe_int(summary.get("anomaly_count", len(_coerce_sequence(summary.get("anomalies")))))
        retry_chain_count = _safe_int(summary.get("retry_chain_count", len(_coerce_sequence(summary.get("retry_chains")))))
        bottleneck_count = _safe_int(summary.get("bottleneck_count", len(_coerce_sequence(summary.get("bottleneck_spans")))))
        anomalies = [_coerce_mapping(item) for item in _coerce_sequence(summary.get("anomalies"))]
        bottlenecks = [_coerce_mapping(item) for item in _coerce_sequence(summary.get("bottleneck_spans"))]
        critical_path_ratio = (critical_path_ms / total_duration_ms) if total_duration_ms > 0 else 0.0

        if critical_path_ms >= self.latency_warning_ms:
            level = "critical" if critical_path_ms >= self.latency_critical_ms else "warning"
            signals.append(
                IncidentSignal(
                    signal_type="critical_path_latency",
                    source="waterfall",
                    level=level,
                    title="Critical path latency is elevated",
                    description=(
                        f"Critical path is {critical_path_ms:.2f} ms over total trace duration {total_duration_ms:.2f} ms "
                        f"(ratio={critical_path_ratio:.2%})."
                    ),
                    score=critical_path_ratio * self._signal_weight(level),
                    timestamp_ms=_now_ms(),
                    context={"trace_id": trace_id, "critical_path_ms": critical_path_ms, "critical_path_ratio": critical_path_ratio},
                )
            )

        if critical_path_ratio >= self.critical_path_ratio_warning:
            level = "critical" if critical_path_ratio >= self.critical_path_ratio_critical else "warning"
            signals.append(
                IncidentSignal(
                    signal_type="critical_path_dominance",
                    source="waterfall",
                    level=level,
                    title="Trace latency is dominated by the critical path",
                    description=f"The critical path accounts for {critical_path_ratio:.2%} of total trace duration.",
                    score=critical_path_ratio * self._signal_weight(level),
                    timestamp_ms=_now_ms(),
                    context={"trace_id": trace_id, "critical_path_ratio": critical_path_ratio},
                )
            )

        if anomaly_count >= self.anomaly_warning_count:
            anomaly_types = Counter(str(item.get("type", "unknown")) for item in anomalies)
            level = "critical" if anomaly_count >= self.anomaly_critical_count else "warning"
            signals.append(
                IncidentSignal(
                    signal_type="timeout_error" if any(name.startswith("timeout") or name == "error_status" for name in anomaly_types) else "trace_anomaly",
                    source="waterfall",
                    level=level,
                    title="Waterfall anomalies detected",
                    description=f"Observed {anomaly_count} waterfall anomalies: {dict(anomaly_types)}.",
                    score=anomaly_count * self._signal_weight(level),
                    timestamp_ms=_now_ms(),
                    context={"trace_id": trace_id, "anomaly_types": dict(anomaly_types)},
                )
            )

        if retry_chain_count >= self.retry_chain_warning_count:
            level = "critical" if retry_chain_count >= self.retry_chain_critical_count else "warning"
            signals.append(
                IncidentSignal(
                    signal_type="retry_storm",
                    source="waterfall",
                    level=level,
                    title="Retry waterfall detected",
                    description=f"Observed {retry_chain_count} retry chains in the trace waterfall.",
                    score=retry_chain_count * self._signal_weight(level),
                    timestamp_ms=_now_ms(),
                    context={"trace_id": trace_id, "retry_chain_count": retry_chain_count},
                )
            )

        if bottleneck_count >= self.bottleneck_warning_count and bottlenecks:
            top = bottlenecks[0]
            level = "critical" if bottleneck_count >= self.bottleneck_critical_count else "warning"
            signals.append(
                IncidentSignal(
                    signal_type="bottleneck_hotspot",
                    source="waterfall",
                    level=level,
                    title="Trace bottleneck hotspot identified",
                    description=(
                        f"Top bottleneck span '{top.get('span_id')}' owned by '{top.get('agent_name')}' consumed "
                        f"{_safe_float(top.get('duration_ms')):.2f} ms."
                    ),
                    score=max(1.0, _safe_float(top.get("ratio"), 0.0) * self._signal_weight(level)),
                    timestamp_ms=_now_ms(),
                    context={
                        "trace_id": trace_id,
                        "span_id": top.get("span_id"),
                        "agent_name": top.get("agent_name"),
                        "duration_ms": top.get("duration_ms"),
                    },
                )
            )

        per_agent_duration = _coerce_mapping(summary.get("per_agent_duration_ms"))
        if per_agent_duration:
            dominant_agent, dominant_value = max(per_agent_duration.items(), key=lambda item: _safe_float(item[1]))
            if total_duration_ms > 0 and (_safe_float(dominant_value) / total_duration_ms) >= 0.40:
                signals.append(
                    IncidentSignal(
                        signal_type="agent_hotspot",
                        source="waterfall",
                        level="warning",
                        title=f"Agent '{dominant_agent}' dominates trace time",
                        description=(
                            f"Agent '{dominant_agent}' accounts for {_safe_float(dominant_value):.2f} ms of the "
                            f"{total_duration_ms:.2f} ms trace."
                        ),
                        score=(_safe_float(dominant_value) / total_duration_ms) * self._signal_weight("warning"),
                        timestamp_ms=_now_ms(),
                        context={"trace_id": trace_id, "agent_name": dominant_agent, "duration_ms": dominant_value},
                    )
                )
        return signals

    def _signals_from_performance(self, report: Mapping[str, Any]) -> List[IncidentSignal]:
        if not report:
            return []
    
        signals: List[IncidentSignal] = []
    
        for entry in _iter_record_mappings(report.get("latency_regressions"), key_field="subject"):
            level = _normalize_level(entry.get("level") or entry.get("severity") or "warning")
            subject = entry.get("subject") or entry.get("agent_name") or "unknown"
            current_value = _safe_float(entry.get("recent_value") or entry.get("current_ms") or entry.get("value"))
            baseline_value = _safe_float(entry.get("baseline_value") or entry.get("baseline_ms"))
            signals.append(
                IncidentSignal(
                    signal_type="latency_regression",
                    source="performance",
                    level=level,
                    title=f"Latency regression for '{subject}'",
                    description=(
                        f"Recent latency is {current_value:.2f} ms versus baseline "
                        f"{baseline_value:.2f} ms for subject '{subject}'."
                    ),
                    score=max(1.0, _safe_float(entry.get("delta_ratio"), 0.0) * self._signal_weight(level)),
                    timestamp_ms=_safe_float(entry.get("timestamp_ms"), _now_ms()),
                    context=entry,
                )
            )
    
        for entry in _iter_record_mappings(report.get("throughput_regressions"), key_field="subject"):
            level = _normalize_level(entry.get("level") or entry.get("severity") or "warning")
            subject = entry.get("subject") or entry.get("service") or "unknown"
            recent_value = _safe_float(entry.get("recent_value") or entry.get("current_rps") or entry.get("value"))
            baseline_value = _safe_float(entry.get("baseline_value") or entry.get("baseline_rps"))
            signals.append(
                IncidentSignal(
                    signal_type="throughput_regression",
                    source="performance",
                    level=level,
                    title=f"Throughput regression for '{subject}'",
                    description=(
                        f"Recent throughput is {recent_value:.2f} versus baseline "
                        f"{baseline_value:.2f} for '{subject}'."
                    ),
                    score=max(1.0, (1.0 - _safe_float(entry.get("delta_ratio"), 1.0)) * self._signal_weight(level)),
                    timestamp_ms=_safe_float(entry.get("timestamp_ms"), _now_ms()),
                    context=entry,
                )
            )
    
        slo_view = report.get("slo_evaluation") or report.get("slo_status")
        if isinstance(slo_view, Mapping):
            slo_view = dict(slo_view)
        else:
            slo_view = {}
    
        if slo_view:
            status = str(slo_view.get("status", "")).lower()
            if status in {"breach", "violated", "critical", "failed"} or bool(slo_view.get("breached")):
                level = "critical"
                signals.append(
                    IncidentSignal(
                        signal_type="slo_breach",
                        source="performance",
                        level=level,
                        title="Latency SLO breach detected",
                        description=(
                            f"Observed {_safe_float(slo_view.get('observed') or slo_view.get('observed_ms')):.2f} "
                            f"against target {_safe_float(slo_view.get('target') or slo_view.get('target_ms')):.2f} "
                            f"for SLO '{slo_view.get('slo_name', 'default')}'."
                        ),
                        score=2.0 * self._signal_weight(level),
                        timestamp_ms=_now_ms(),
                        context=slo_view,
                    )
                )
    
        return signals

    def _signals_from_capacity(self, report: Mapping[str, Any]) -> List[IncidentSignal]:
        if not report:
            return []
    
        signals: List[IncidentSignal] = []
    
        for entry in _iter_record_mappings(report.get("alerts")):
            level = _normalize_level(entry.get("level") or entry.get("severity") or "warning")
            signal_type = str(entry.get("signal") or entry.get("signal_type") or "capacity_alert")
            title = str(entry.get("title") or entry.get("message") or signal_type.replace("_", " ").title())
            description = str(entry.get("message") or title)
            signals.append(
                IncidentSignal(
                    signal_type=signal_type,
                    source="capacity",
                    level=level,
                    title=title,
                    description=description,
                    score=max(1.0, self._signal_weight(level)),
                    timestamp_ms=_safe_float(entry.get("timestamp_ms"), _now_ms()),
                    context=entry,
                )
            )
    
        queue_items = (
            report.get("queue_findings")
            or report.get("queues")
            or report.get("queue_analysis")
            or report.get("queue_reports")
        )
        for queue_view in _iter_record_mappings(queue_items, key_field="queue_name"):
            queue_name = queue_view.get("queue_name") or queue_view.get("subject") or "queue"
            backlog_depth = _safe_float(queue_view.get("depth") or queue_view.get("backlog_depth"))
            growth_rate = _safe_float(queue_view.get("growth_rate_per_min") or queue_view.get("slope_per_min"))
            if backlog_depth > 0 or growth_rate > 0:
                level = _normalize_level(queue_view.get("level") or queue_view.get("severity") or "warning")
                signal_type = "queue_growth" if growth_rate > 0 else "queue_backlog"
                signals.append(
                    IncidentSignal(
                        signal_type=signal_type,
                        source="capacity",
                        level=level,
                        title=f"Queue pressure on '{queue_name}'",
                        description=(
                            f"Queue '{queue_name}' has depth {backlog_depth:.2f} and "
                            f"growth rate {growth_rate:.2f}/min."
                        ),
                        score=max(1.0, self._signal_weight(level)),
                        timestamp_ms=_safe_float(queue_view.get("timestamp_ms"), _now_ms()),
                        context=queue_view,
                    )
                )
    
        resource_items = (
            report.get("resource_findings")
            or report.get("resources")
            or report.get("resource_analysis")
            or report.get("resource_reports")
        )
        for resource_view in _iter_record_mappings(resource_items, key_field="resource_name"):
            resource_name = resource_view.get("resource_name") or resource_view.get("host") or resource_view.get("resource_type") or "resource"
            utilization_pct = _safe_float(resource_view.get("utilization_pct") or resource_view.get("observed") or resource_view.get("value"))
            if utilization_pct <= 0:
                continue
            level = _normalize_level(resource_view.get("level") or resource_view.get("severity") or "warning")
            resource_type = str(resource_view.get("resource_type") or "resource")
            signals.append(
                IncidentSignal(
                    signal_type=f"{resource_type}_pressure" if resource_type in {"cpu", "memory", "gpu"} else "resource_pressure",
                    source="capacity",
                    level=level,
                    title=f"{resource_type.upper()} pressure on '{resource_name}'",
                    description=f"Resource '{resource_name}' is at {utilization_pct:.2f}% utilization.",
                    score=max(1.0, (utilization_pct / 100.0) * self._signal_weight(level)),
                    timestamp_ms=_safe_float(resource_view.get("timestamp_ms"), _now_ms()),
                    context=resource_view,
                )
            )
    
        return signals

    def _signals_from_event_records(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        default_source: str,
    ) -> List[IncidentSignal]:
        signals: List[IncidentSignal] = []
        for record in records:
            entry = _coerce_mapping(record)
            level = _normalize_level(entry.get("level") or entry.get("severity") or "warning")
            signal_type = str(entry.get("signal_type") or entry.get("event_type") or entry.get("type") or "event")
            title = str(entry.get("title") or entry.get("message") or signal_type.replace("_", " ").title())
            description = str(entry.get("message") or entry.get("description") or title)
            signals.append(
                IncidentSignal(
                    signal_type=signal_type,
                    source=str(entry.get("source") or default_source),
                    level=level,
                    title=_truncate_text(title, 120),
                    description=_truncate_text(description, 320),
                    score=max(1.0, self._signal_weight(level)),
                    timestamp_ms=_safe_float(entry.get("timestamp_ms"), _now_ms()),
                    context=entry,
                )
            )
        return signals

    def _normalize_event_records(self, records: Sequence[Mapping[str, Any]], *, source: str) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for item in records:
            entry = _coerce_mapping(item)
            normalized.append(
                {
                    "source": source,
                    "event_type": entry.get("event_type") or entry.get("type") or source,
                    "message": entry.get("message") or entry.get("description") or entry.get("title") or source,
                    "level": _normalize_level(entry.get("level") or entry.get("severity") or "warning"),
                    "timestamp_ms": _safe_float(entry.get("timestamp_ms"), _now_ms()),
                    **entry,
                }
            )
        normalized.sort(key=lambda item: _safe_float(item.get("timestamp_ms"), 0.0))
        return normalized

    # ------------------------------------------------------------------
    # Classification and presentation
    # ------------------------------------------------------------------
    def _classify_incident_level(
        self,
        signals: Sequence[IncidentSignal],
        alert_fatigue: Mapping[str, Any],
    ) -> Tuple[str, str, float]:
        if not signals:
            return "info", "monitoring", 0.0

        level_counter = Counter(signal.level for signal in signals)
        score = sum(signal.score for signal in signals)
        critical_count = level_counter.get("critical", 0)
        warning_count = level_counter.get("warning", 0)
        repeated_bonus = 1.0 if bool(alert_fatigue.get("historically_repeated")) else 0.0
        total_score = score + repeated_bonus

        if critical_count > 0 or total_score >= self.critical_score_threshold:
            return "critical", "open", total_score
        if warning_count > 0 or total_score >= self.warning_score_threshold:
            return "warning", "open", total_score
        return "info", "monitoring", total_score

    def _build_incident_summary(
        self,
        *,
        incident_id: str,
        incident_level: str,
        signals: Sequence[IncidentSignal],
        root_causes: Sequence[RootCauseHypothesis],
    ) -> str:
        top_signal = signals[0].title if signals else "limited evidence"
        top_root_cause = root_causes[0].label if root_causes else "undetermined"
        return (
            f"Incident {incident_id} is classified as {incident_level}. "
            f"Primary symptom: {top_signal}. Most likely cause: {top_root_cause}."
        )

    def _derive_customer_impact(self, incident_level: str,
        signals: Sequence[IncidentSignal],
        waterfall_summary: Optional[Mapping[str, Any]],
        capacity_report: Optional[Mapping[str, Any]],
    ) -> str:
        impact_template = self.impact_templates.get(incident_level)
        if impact_template:
            return str(impact_template)

        signal_types = {signal.signal_type for signal in signals}
        if incident_level == "critical":
            if {"queue_growth", "resource_pressure", "slo_breach"} & signal_types:
                return "User-facing degradation is likely active or imminent; latency, backlog, or saturation signals exceed acceptable operational bounds."
            return "User-facing degradation is likely active and requires immediate mitigation."
        if incident_level == "warning":
            return "User-facing impact is possible if current regressions continue; close monitoring and remediation are warranted."
        return "Current evidence suggests limited or localized user impact, but continued observation is recommended."

    def _build_evidence_snapshot(self, *,
        signals: Sequence[IncidentSignal],
        waterfall_summary: Optional[Mapping[str, Any]],
        capacity_report: Optional[Mapping[str, Any]],
        performance_report: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        signal_counts = Counter(signal.signal_type for signal in signals)
        snapshot = {
            "signal_counts": dict(signal_counts),
            "top_signals": [signal.to_dict() for signal in list(signals)[: int(self.intel_config.get("incident_brief_max_evidence_items", 12))]],
        }
        if waterfall_summary:
            snapshot["waterfall"] = {
                "total_duration_ms": _safe_float(waterfall_summary.get("total_duration_ms")),
                "critical_path_ms": _safe_float(waterfall_summary.get("critical_path_ms")),
                "bottleneck_count": _safe_int(waterfall_summary.get("bottleneck_count", len(_coerce_sequence(waterfall_summary.get("bottleneck_spans"))))),
                "anomaly_count": _safe_int(waterfall_summary.get("anomaly_count", len(_coerce_sequence(waterfall_summary.get("anomalies"))))),
                "retry_chain_count": _safe_int(waterfall_summary.get("retry_chain_count", len(_coerce_sequence(waterfall_summary.get("retry_chains"))))),
            }
        if capacity_report:
            capacity_records = _coerce_mapping(capacity_report)
            snapshot["capacity"] = {
                "alert_count": len(_iter_record_mappings(capacity_records.get("alerts"))),
                "queue_count": len(
                    _iter_record_mappings(
                        capacity_records.get("queue_findings")
                        or capacity_records.get("queues")
                        or capacity_records.get("queue_reports")
                    )
                ),
                "resource_count": len(
                    _iter_record_mappings(
                        capacity_records.get("resource_findings")
                        or capacity_records.get("resources")
                        or capacity_records.get("resource_reports")
                    )
                ),
            }
        
        if performance_report:
            performance_records = _coerce_mapping(performance_report)
            snapshot["performance"] = {
                "latency_regression_count": len(_iter_record_mappings(performance_records.get("latency_regressions"), key_field="subject")),
                "throughput_regression_count": len(_iter_record_mappings(performance_records.get("throughput_regressions"), key_field="subject")),
            }
        return snapshot

    def _build_incident_timeline(
        self,
        alert_records: Sequence[Mapping[str, Any]],
        error_records: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        timeline: List[Dict[str, Any]] = []
        for record in list(alert_records) + list(error_records):
            entry = _coerce_mapping(record)
            timeline.append(
                {
                    "timestamp_ms": _safe_float(entry.get("timestamp_ms"), _now_ms()),
                    "source": entry.get("source"),
                    "level": _normalize_level(entry.get("level") or entry.get("severity") or "info"),
                    "event_type": entry.get("event_type") or entry.get("type"),
                    "message": entry.get("message") or entry.get("description") or entry.get("title"),
                }
            )
        timeline.sort(key=lambda item: _safe_float(item.get("timestamp_ms"), 0.0))
        return timeline[: self.timeline_event_limit]

    def _assess_alert_fatigue(
        self,
        error_signature: str,
        signals: Sequence[IncidentSignal],
        alert_records: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        recent_count = self._recent_signatures.get(error_signature, 0)
        repeated_alert_types = Counter(str(record.get("event_type") or record.get("type") or "alert") for record in alert_records)
        repeated_signal_types = Counter(signal.signal_type for signal in signals)
        suppress_duplicate_alert = recent_count >= self.alert_fatigue_repeat_threshold and len(repeated_signal_types) <= 3
        return {
            "error_signature": error_signature,
            "recent_repeat_count": recent_count,
            "repeated_alert_types": dict(repeated_alert_types),
            "repeated_signal_types": dict(repeated_signal_types),
            "historically_repeated": recent_count > 0,
            "suppress_duplicate_alert": suppress_duplicate_alert,
        }

    # ------------------------------------------------------------------
    # Persistence and historical context
    # ------------------------------------------------------------------
    def _persist_incident_context(self, *, incident_id: str, error_signature: str, incident_level: str,
        signals: Sequence[IncidentSignal],
        root_causes: Sequence[RootCauseHypothesis],
        runbooks: Sequence[RunbookRecommendation],
        timeline: Sequence[Mapping[str, Any]],
        metadata: Mapping[str, Any]) -> None:
        if self.memory is None:
            return

        try:
            related_agents: List[str] = []
            for cause in root_causes:
                related_agents.extend(cause.related_agents)

            if hasattr(self.memory, "record_incident_fingerprint"):
                self.memory.record_incident_fingerprint(
                    incident_id=incident_id,
                    error_signature=error_signature,
                    severity=incident_level,
                    related_agents=_dedupe_preserve_order(related_agents),
                    root_cause_candidates=[cause.label for cause in root_causes[: self.root_cause_top_k]],
                    recommended_actions=[runbook.playbook_id for runbook in runbooks[: self.runbook_top_k]],
                    metadata={
                        "signal_types": [signal.signal_type for signal in signals],
                        **metadata,
                    },
                )

            if hasattr(self.memory, "append_timeline_event"):
                self.memory.append_timeline_event(
                    incident_id,
                    event_type="incident_synthesized",
                    message=f"Incident synthesized with level '{incident_level}'.",
                    severity=incident_level,
                    payload={
                        "root_causes": [cause.label for cause in root_causes[: self.root_cause_top_k]],
                        "runbooks": [runbook.playbook_id for runbook in runbooks[: self.runbook_top_k]],
                        "timeline_event_count": len(timeline),
                    },
                )
        except Exception as exc:
            logger.warning("Failed to persist intelligence context for incident '%s': %s", incident_id, exc)

    def _lookup_runbook_history(self, playbook_id: str) -> Tuple[Optional[float], int]:
        if self.memory is None or not hasattr(self.memory, "runbook_outcome"):
            return None, 0
        try:
            outcome = _coerce_mapping(self.memory.runbook_outcome(playbook_id))
            if not outcome:
                return None, 0
            return outcome.get("success_rate"), _safe_int(outcome.get("attempt_count"))
        except Exception as exc:
            logger.warning("Runbook history lookup failed for '%s': %s", playbook_id, exc)
            return None, 0

    def _record_recent_signature(self, error_signature: str) -> None:
        cutoff_ms = _now_ms() - (self.alert_fatigue_window_seconds * 1000.0)
        with self._lock:
            while self._recent_incidents and self._recent_incidents[0][0] < cutoff_ms:
                _, expired_signature = self._recent_incidents.popleft()
                if self._recent_signatures.get(expired_signature, 0) > 1:
                    self._recent_signatures[expired_signature] -= 1
                else:
                    self._recent_signatures.pop(expired_signature, None)

            self._recent_incidents.append((_now_ms(), error_signature))
            self._recent_signatures[error_signature] += 1

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _ensure_enabled(self, *, operation: str) -> None:
        if self.enabled:
            return
        raise ObservabilityError(
            message=f"Observability intelligence is disabled; cannot execute '{operation}'",
            error_type=ObservabilityErrorType.INCIDENT_CLASSIFICATION_AMBIGUOUS,
            severity=ObservabilitySeverity.MEDIUM,
            retryable=False,
            context={"operation": operation},
            remediation="Enable observability_intelligence in the configuration before invoking incident synthesis.",
        )

    def _require_non_empty_str(self, value: Any, *, field_name: str, operation: str) -> str:
        text = str(value).strip() if value is not None else ""
        if text:
            return text
        raise ObservabilityError(
            message=f"{operation} requires a non-empty '{field_name}'",
            error_type=ObservabilityErrorType.INCIDENT_CLASSIFICATION_AMBIGUOUS,
            severity=ObservabilitySeverity.MEDIUM,
            retryable=False,
            context={"field_name": field_name, "operation": operation},
            remediation=f"Provide a valid {field_name} before calling {operation}.",
        )

    def _extract_trace_id(self,
        waterfall_summary: Mapping[str, Any],
        performance_report: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> Optional[str]:
        for source in (waterfall_summary, performance_report, metadata):
            if not source:
                continue
            for key in ("trace_id", "observability.trace_id"):
                value = source.get(key)
                if value:
                    return str(value)
        return None

    def _infer_incident_start(self, timeline: Sequence[Mapping[str, Any]]) -> Optional[float]:
        if not timeline:
            return None
        return min(_safe_float(item.get("timestamp_ms"), _now_ms()) for item in timeline)

    def _build_error_signature(self, *, incident_id: str,
        signals: Sequence[IncidentSignal],
        waterfall_summary: Mapping[str, Any],
        capacity_report: Mapping[str, Any],
        performance_report: Mapping[str, Any],
    ) -> str:
        payload = {
            "incident_id": incident_id,
            "signal_types": [signal.signal_type for signal in signals[:12]],
            "signal_levels": [signal.level for signal in signals[:12]],
            "critical_path_ms": round(_safe_float(waterfall_summary.get("critical_path_ms")), 3),
            "anomaly_count": _safe_int(waterfall_summary.get("anomaly_count", len(_coerce_sequence(waterfall_summary.get("anomalies"))))),
            "bottleneck_count": _safe_int(waterfall_summary.get("bottleneck_count", len(_coerce_sequence(waterfall_summary.get("bottleneck_spans"))))),
            "capacity_alert_count": len(_coerce_sequence(capacity_report.get("alerts"))),
            "latency_regression_count": len(_coerce_sequence(performance_report.get("latency_regressions"))),
            "throughput_regression_count": len(_coerce_sequence(performance_report.get("throughput_regressions"))),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return f"incsig:{digest}"

    def _signal_weight(self, level: str) -> float:
        return float(self.signal_weights.get(level, 1.0))

    def _build_root_cause_rationale(
        self,
        *,
        label: str,
        evidence: Sequence[str],
        related_signals: Sequence[str],
    ) -> str:
        evidence_fragment = evidence[0] if evidence else "available evidence"
        signal_fragment = ", ".join(_dedupe_preserve_order(related_signals)[:3]) if related_signals else "mixed signals"
        return f"{label} is favored because {evidence_fragment} and the strongest supporting signals are {signal_fragment}."


# ---------------------------------------------------------------------------
# Standalone script test block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Running Observability Intelligence  ===\n")
    printer.status("TEST", "Observability Intelligence  initialized", "info")

    intelligence = ObservabilityIntelligence()

    sample_spans = [
        {
            "span_id": "root-1",
            "agent_name": "planning_agent",
            "start_ms": 0.0,
            "end_ms": 1250.0,
            "status": "ok",
        },
        {
            "span_id": "retrieval-1",
            "agent_name": "retrieval_agent",
            "start_ms": 50.0,
            "end_ms": 900.0,
            "status": "retry",
            "parent_span_id": "root-1",
        },
        {
            "span_id": "tool-1",
            "agent_name": "tool_agent",
            "start_ms": 920.0,
            "end_ms": 4100.0,
            "status": "timeout",
            "parent_span_id": "root-1",
        },
    ]

    sample_capacity_report = {
        "alerts": [
            {
                "level": "critical",
                "signal": "queue_growth",
                "message": "scheduler queue backlog is growing faster than consumers can drain it",
                "queue_name": "agent_scheduler",
            },
            {
                "level": "warning",
                "signal": "resource_pressure",
                "message": "GPU utilization exceeded the warning threshold",
                "resource_type": "gpu",
                "resource_name": "gpu-node-a",
                "utilization_pct": 94.0,
            },
        ],
        "queues": [
            {
                "queue_name": "agent_scheduler",
                "backlog_depth": 180,
                "growth_rate_per_min": 32.0,
                "level": "critical",
            }
        ],
        "resources": [
            {
                "resource_name": "gpu-node-a",
                "resource_type": "gpu",
                "utilization_pct": 94.0,
                "level": "warning",
            }
        ],
    }

    sample_performance_report = {
        "latency_regressions": [
            {
                "subject": "tool_agent",
                "recent_value": 4100.0,
                "baseline_value": 1400.0,
                "delta_ratio": 2.928571,
                "level": "critical",
            }
        ],
        "throughput_regressions": [
            {
                "service": "slai_runtime",
                "recent_value": 7.5,
                "baseline_value": 18.0,
                "delta_ratio": 0.416667,
                "level": "warning",
            }
        ],
        "slo_status": {
            "status": "breach",
            "slo_name": "response_p95",
            "observed": 4100.0,
            "target": 1500.0,
        },
    }

    alerts = [
        {
            "event_type": "latency_page",
            "message": "p95 latency exceeded the incident paging threshold",
            "level": "critical",
            "timestamp_ms": _now_ms() - 20_000.0,
        },
        {
            "event_type": "queue_alert",
            "message": "queue growth persisted for five consecutive windows",
            "level": "warning",
            "timestamp_ms": _now_ms() - 10_000.0,
        },
    ]

    error_events = [
        {
            "event_type": "dependency_timeout",
            "message": "tool_agent timed out while waiting on the external inference backend",
            "level": "critical",
            "timestamp_ms": _now_ms() - 15_000.0,
        }
    ]

    assessment = intelligence.synthesize_incident(
        incident_id="incident-obs-001",
        spans=sample_spans,
        capacity_report=sample_capacity_report,
        performance_report=sample_performance_report,
        alerts=alerts,
        error_events=error_events,
        metadata={"service": "slai_runtime"},
    )

    brief = assessment["brief"]
    print("Incident level:", assessment["incident_level"])
    print("Incident status:", assessment["status"])
    print("Error signature:", assessment["error_signature"])
    print("Summary:", brief["summary"])
    print("Top root cause:", brief["top_root_causes"][0]["label"] if brief["top_root_causes"] else "n/a")
    print("Top runbook:", brief["recommended_runbooks"][0]["playbook_id"] if brief["recommended_runbooks"] else "n/a")
    print("Signal count:", len(assessment["signals"]))
    print("Timeline events:", len(brief["timeline"]))

    print("\n=== Test ran successfully ===\n")
