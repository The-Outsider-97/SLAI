"""Privacy-specific lineage and dissemination analysis for SLAI.

``PrivacyFlowAnalyzer`` reasons over privacy lineage evidence that already exists
in the subsystem.  It does not route messages, validate network security,
perform consent checks, or own lineage persistence.  Its responsibility is to
interpret an ordered set of privacy lineage events and identify cumulative
privacy risks such as destination expansion, purpose drift, repeated
cross-context propagation, processing after verified deletion, and suspicious
context cycles.

The analyzer is intentionally data-source agnostic.  The caller (normally the
Privacy Agent) supplies lineage events from ``PrivacyMemory`` or an audit bundle.
That avoids importing ``PrivacyMemory`` here and keeps dependency direction
strictly ``modules -> utils``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.privacy_error import *
from ..utils.privacy_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Privacy Flow Analyzer")
printer = PrettyPrinter()

MODULE_VERSION = "2.3.0"
ASSESSMENT_SCHEMA_VERSION = "privacy.flow_analysis.v1"


@dataclass(frozen=True, slots=True)
class PrivacyFlowEdge:
    """Normalized context transition extracted from a privacy lineage event."""

    source_context: str
    destination_context: str
    purpose: Optional[str]
    operation: Optional[str]
    request_id: Optional[str]
    record_id: Optional[str]
    subject_id: Optional[str]
    timestamp: Optional[float]
    event_ref: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PrivacyFlowFinding:
    """Audit-safe flow finding."""

    code: str
    severity: str
    message: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["evidence"] = sanitize_privacy_mapping(data.get("evidence"), enabled=True)
        return data


@dataclass(frozen=True, slots=True)
class PrivacyFlowAssessment:
    """Structured privacy dissemination assessment."""

    request_id: Optional[str]
    record_id: Optional[str]
    subject_id: Optional[str]
    decision: str
    status: str
    stage: str
    event_count: int
    edge_count: int
    cross_context_transition_count: int
    unique_context_count: int
    unique_destination_count: int
    max_context_fanout: int
    unique_purpose_count: int
    cycle_detected: bool
    unauthorized_destinations: Tuple[str, ...]
    high_risk_destinations: Tuple[str, ...]
    post_deletion_processing_count: int
    incomplete_event_count: int
    flow_risk_score: float
    findings: Tuple[PrivacyFlowFinding, ...]
    policy_fingerprint: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["findings"] = [item.to_dict() for item in self.findings]
        return data


class PrivacyFlowAnalyzer:
    """Analyze cumulative privacy movement without duplicating consent/routing logic."""

    CONFIG_SECTION = "privacy_flow_analyzer"

    def __init__(self, *, config: Optional[Mapping[str, Any]] = None) -> None:
        root_config = dict(config) if config is not None else load_global_config()
        self.config = get_config_section(self.CONFIG_SECTION, config=root_config)
        if not self.config:
            raise PrivacyConfigurationError(
                section=self.CONFIG_SECTION,
                details=(
                    "Missing privacy_flow_analyzer configuration. Add the documented "
                    "section to src/agents/privacy/configs/privacy_config.yaml."
                ),
            )

        self.enabled = self._require_bool("enabled")
        self.strict_mode = self._require_bool("strict_mode")
        self.default_stage = self._require_text("default_decision_stage")
        self.block_on_post_deletion_processing = self._require_bool(
            "block_on_post_deletion_processing"
        )
        self.block_on_unauthorized_destination = self._require_bool(
            "block_on_unauthorized_destination"
        )
        self.escalate_on_unauthorized_destination = self._require_bool(
            "escalate_on_unauthorized_destination"
        )
        self.escalate_on_purpose_drift = self._require_bool("escalate_on_purpose_drift")
        self.escalate_on_context_cycle = self._require_bool("escalate_on_context_cycle")
        self.max_events = require_integer(
            self.config.get("max_events"), "privacy_flow_analyzer.max_events", minimum=1
        )
        self.max_findings = require_integer(
            self.config.get("max_findings"), "privacy_flow_analyzer.max_findings", minimum=1
        )
        self.max_authorized_contexts = require_integer(
            self.config.get("max_authorized_contexts"),
            "privacy_flow_analyzer.max_authorized_contexts",
            minimum=1,
        )

        thresholds = self._require_mapping("thresholds")
        self.max_unique_destinations = require_integer(
            thresholds.get("max_unique_destinations"),
            "privacy_flow_analyzer.thresholds.max_unique_destinations",
            minimum=1,
        )
        self.max_cross_context_transitions = require_integer(
            thresholds.get("max_cross_context_transitions"),
            "privacy_flow_analyzer.thresholds.max_cross_context_transitions",
            minimum=1,
        )
        self.max_unique_purposes = require_integer(
            thresholds.get("max_unique_purposes"),
            "privacy_flow_analyzer.thresholds.max_unique_purposes",
            minimum=1,
        )
        self.max_context_fanout_threshold = require_integer(
            thresholds.get("max_context_fanout"),
            "privacy_flow_analyzer.thresholds.max_context_fanout",
            minimum=1,
        )
        self.modify_score_threshold = require_probability(
            thresholds.get("modify_score"),
            "privacy_flow_analyzer.thresholds.modify_score",
        )
        self.escalate_score_threshold = require_probability(
            thresholds.get("escalate_score"),
            "privacy_flow_analyzer.thresholds.escalate_score",
        )
        if self.modify_score_threshold > self.escalate_score_threshold:
            raise PrivacyConfigurationError(
                section=self.CONFIG_SECTION,
                details="thresholds.modify_score must be <= thresholds.escalate_score.",
            )

        weights = self._require_mapping("risk_weights")
        self.risk_weights = {
            "destination_expansion": require_probability(
                weights.get("destination_expansion"),
                "privacy_flow_analyzer.risk_weights.destination_expansion",
            ),
            "cross_context": require_probability(
                weights.get("cross_context"),
                "privacy_flow_analyzer.risk_weights.cross_context",
            ),
            "purpose_drift": require_probability(
                weights.get("purpose_drift"),
                "privacy_flow_analyzer.risk_weights.purpose_drift",
            ),
            "unauthorized_destination": require_probability(
                weights.get("unauthorized_destination"),
                "privacy_flow_analyzer.risk_weights.unauthorized_destination",
            ),
            "post_deletion_processing": require_probability(
                weights.get("post_deletion_processing"),
                "privacy_flow_analyzer.risk_weights.post_deletion_processing",
            ),
            "cycle": require_probability(
                weights.get("cycle"),
                "privacy_flow_analyzer.risk_weights.cycle",
            ),
        }
        self._validate_weights(self.risk_weights)

        self.high_risk_contexts = set(
            normalize_string_sequence(
                self.config.get("high_risk_contexts"),
                field_name="privacy_flow_analyzer.high_risk_contexts",
                casefold=True,
            )
        )
        self.ignored_contexts = set(
            normalize_string_sequence(
                self.config.get("ignored_contexts"),
                field_name="privacy_flow_analyzer.ignored_contexts",
                casefold=True,
            )
        )
        self.allowed_cycle_contexts = set(
            normalize_string_sequence(
                self.config.get("allowed_cycle_contexts"),
                field_name="privacy_flow_analyzer.allowed_cycle_contexts",
                casefold=True,
            )
        )

        self.policy_fingerprint = stable_privacy_fingerprint(
            {"section": self.CONFIG_SECTION, "config": self.config}, length=24
        )
        logger.info(
            "PrivacyFlowAnalyzer initialized (schema=%s, policy=%s)",
            ASSESSMENT_SCHEMA_VERSION,
            self.policy_fingerprint,
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def _require_bool(self, key: str) -> bool:
        if key not in self.config or not isinstance(self.config[key], bool):
            raise PrivacyConfigurationError(
                section=self.CONFIG_SECTION,
                details=f"'{key}' must be explicitly configured as a boolean.",
            )
        return bool(self.config[key])

    def _require_text(self, key: str) -> str:
        value = str(self.config.get(key) or "").strip()
        if not value:
            raise PrivacyConfigurationError(
                section=self.CONFIG_SECTION,
                details=f"'{key}' must be a non-empty string.",
            )
        return value

    def _require_mapping(self, key: str) -> Dict[str, Any]:
        value = self.config.get(key)
        if not isinstance(value, Mapping):
            raise PrivacyConfigurationError(
                section=self.CONFIG_SECTION,
                details=f"'{key}' must be a mapping.",
            )
        return dict(value)

    @staticmethod
    def _validate_weights(weights: Mapping[str, float]) -> None:
        total = sum(float(v) for v in weights.values())
        if abs(total - 1.0) > 1e-9:
            raise PrivacyConfigurationError(
                section=PrivacyFlowAnalyzer.CONFIG_SECTION,
                details=f"risk_weights must sum to exactly 1.0; received {total:.12f}.",
            )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_context_name(value: Any) -> Optional[str]:
        text = str(value or "").strip()
        return text or None

    @staticmethod
    def _mapping(value: Any, field_name: str) -> Dict[str, Any]:
        if value is None:
            return {}
        if hasattr(value, "to_dict") and callable(value.to_dict):
            value = value.to_dict()
        if not isinstance(value, Mapping):
            raise TypeError(f"'{field_name}' must be a mapping or expose to_dict().")
        return normalize_mapping(value, field_name=field_name, deep_copy=True)

    @staticmethod
    def _event_payload(event: Mapping[str, Any]) -> Mapping[str, Any]:
        payload = event.get("payload")
        return payload if isinstance(payload, Mapping) else event

    def _edge_from_event(self, event: Mapping[str, Any], index: int) -> Tuple[Optional[PrivacyFlowEdge], bool]:
        payload = self._event_payload(event)
        source = self._normalize_context_name(payload.get("source_context") or event.get("source_context"))
        destination = self._normalize_context_name(
            payload.get("destination_context") or event.get("destination_context")
        )
        purpose = self._normalize_context_name(payload.get("purpose") or event.get("purpose"))
        operation = self._normalize_context_name(payload.get("operation") or event.get("operation"))
        request_id = self._normalize_context_name(event.get("request_id") or payload.get("request_id"))
        record_id = self._normalize_context_name(event.get("record_id") or payload.get("record_id"))
        subject_id = self._normalize_context_name(event.get("subject_id") or payload.get("subject_id"))

        timestamp = None
        raw_timestamp = event.get("timestamp", payload.get("timestamp"))
        if raw_timestamp is not None:
            try:
                timestamp = normalize_optional_timestamp(raw_timestamp, "lineage_event.timestamp")
            except ValueError:
                timestamp = None

        event_ref = str(
            event.get("event_id")
            or event.get("event_ref")
            or stable_privacy_fingerprint(
                {
                    "index": index,
                    "source": source,
                    "destination": destination,
                    "purpose": purpose,
                    "operation": operation,
                    "request_id": request_id,
                    "record_id": record_id,
                    "timestamp": timestamp,
                },
                length=20,
            )
        )

        incomplete = not source or not destination
        if incomplete:
            return None, True

        assert source is not None
        assert destination is not None
        if source.casefold() in self.ignored_contexts or destination.casefold() in self.ignored_contexts:
            return None, False

        return (
            PrivacyFlowEdge(
                source_context=source,
                destination_context=destination,
                purpose=purpose,
                operation=operation,
                request_id=request_id,
                record_id=record_id,
                subject_id=subject_id,
                timestamp=timestamp,
                event_ref=event_ref,
            ),
            False,
        )

    @staticmethod
    def _dedupe_edges(edges: Sequence[PrivacyFlowEdge]) -> List[PrivacyFlowEdge]:
        seen: Set[str] = set()
        result: List[PrivacyFlowEdge] = []
        for edge in edges:
            key = edge.event_ref or stable_privacy_fingerprint(edge.to_dict(), length=20)
            if key in seen:
                continue
            seen.add(key)
            result.append(edge)
        result.sort(key=lambda e: (e.timestamp is None, e.timestamp or 0.0, e.event_ref))
        return result

    @staticmethod
    def _has_cycle(adjacency: Mapping[str, Set[str]], allowed_cycle_contexts: Set[str]) -> bool:
        visited: Set[str] = set()
        active: Set[str] = set()

        def visit(node: str) -> bool:
            if node in active:
                return node.casefold() not in allowed_cycle_contexts
            if node in visited:
                return False
            visited.add(node)
            active.add(node)
            for neighbor in adjacency.get(node, set()):
                if neighbor == node:
                    continue
                if visit(neighbor):
                    return True
            active.remove(node)
            return False

        return any(visit(node) for node in adjacency if node not in visited)

    @staticmethod
    def _deletion_completed_at(retention: Mapping[str, Any]) -> Optional[float]:
        candidates: List[Any] = []
        deletion = retention.get("deletion")
        if isinstance(deletion, Mapping):
            candidates.extend(
                [
                    deletion.get("completed_at"),
                    deletion.get("deleted_at"),
                ]
            )
        candidates.extend([retention.get("completed_at"), retention.get("deleted_at")])
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                return normalize_optional_timestamp(candidate, "retention.deletion_completed_at")
            except ValueError:
                continue
        return None

    @staticmethod
    def _bounded_ratio(value: int, threshold: int) -> float:
        if value <= 0:
            return 0.0
        return min(1.0, value / float(max(threshold, 1)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(
        self,
        *,
        lineage_events: Sequence[Mapping[str, Any]],
        request_id: Optional[str] = None,
        record_id: Optional[str] = None,
        subject_id: Optional[str] = None,
        authorized_contexts: Optional[Sequence[str]] = None,
        retention: Optional[Mapping[str, Any]] = None,
        stage: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Analyze privacy lineage events for cumulative dissemination risk."""

        if not self.enabled:
            result = PrivacyFlowAssessment(
                request_id=request_id,
                record_id=record_id,
                subject_id=subject_id,
                decision=PrivacyDecision.ALLOW.value,
                status="disabled",
                stage=stage or self.default_stage,
                event_count=0,
                edge_count=0,
                cross_context_transition_count=0,
                unique_context_count=0,
                unique_destination_count=0,
                max_context_fanout=0,
                unique_purpose_count=0,
                cycle_detected=False,
                unauthorized_destinations=(),
                high_risk_destinations=(),
                post_deletion_processing_count=0,
                incomplete_event_count=0,
                flow_risk_score=0.0,
                findings=(),
                policy_fingerprint=self.policy_fingerprint,
                created_at=utc_iso(),
            )
            return result.to_dict()

        try:
            if isinstance(lineage_events, (str, bytes, bytearray)) or not isinstance(lineage_events, Sequence):
                raise TypeError("'lineage_events' must be a sequence of mappings.")
            if len(lineage_events) > self.max_events:
                raise ValueError(
                    f"lineage_events exceeds configured maximum of {self.max_events}."
                )

            normalized_authorized = {
                item.casefold()
                for item in normalize_string_sequence(
                    authorized_contexts,
                    field_name="authorized_contexts",
                    max_items=self.max_authorized_contexts,
                    overflow="raise",
                )
            }
            retention_map = self._mapping(retention, "retention") if retention is not None else {}
            safe_context = sanitize_privacy_mapping(context, enabled=True)

            raw_edges: List[PrivacyFlowEdge] = []
            incomplete_event_count = 0
            for index, event in enumerate(lineage_events):
                if not isinstance(event, Mapping):
                    incomplete_event_count += 1
                    continue
                edge, incomplete = self._edge_from_event(event, index)
                if incomplete:
                    incomplete_event_count += 1
                if edge is not None:
                    raw_edges.append(edge)
            edges = self._dedupe_edges(raw_edges)

            contexts: Set[str] = set()
            destinations: Set[str] = set()
            purposes: Set[str] = set()
            adjacency: Dict[str, Set[str]] = defaultdict(set)
            fanout: Dict[str, Set[str]] = defaultdict(set)
            cross_context_count = 0

            for edge in edges:
                contexts.add(edge.source_context)
                contexts.add(edge.destination_context)
                destinations.add(edge.destination_context)
                if edge.purpose:
                    purposes.add(edge.purpose)
                adjacency[edge.source_context].add(edge.destination_context)
                fanout[edge.source_context].add(edge.destination_context)
                if edge.source_context != edge.destination_context:
                    cross_context_count += 1

            max_context_fanout = max((len(values) for values in fanout.values()), default=0)
            cycle_detected = self._has_cycle(adjacency, self.allowed_cycle_contexts)

            unauthorized_destinations: Set[str] = set()
            if normalized_authorized:
                for destination in destinations:
                    if destination.casefold() not in normalized_authorized:
                        unauthorized_destinations.add(destination)

            high_risk_destinations = {
                destination
                for destination in destinations
                if destination.casefold() in self.high_risk_contexts
            }

            deletion_completed_at = self._deletion_completed_at(retention_map)
            post_deletion_processing: List[PrivacyFlowEdge] = []
            if deletion_completed_at is not None:
                post_deletion_processing = [
                    edge
                    for edge in edges
                    if edge.timestamp is not None and edge.timestamp > deletion_completed_at
                ]

            findings: List[PrivacyFlowFinding] = []

            def add_finding(
                code: str,
                severity: str,
                message: str,
                evidence: Optional[Mapping[str, Any]] = None,
            ) -> None:
                if len(findings) >= self.max_findings:
                    return
                findings.append(
                    PrivacyFlowFinding(
                        code=code,
                        severity=severity,
                        message=message,
                        evidence=sanitize_privacy_mapping(evidence, enabled=True),
                    )
                )

            if len(destinations) > self.max_unique_destinations:
                add_finding(
                    "destination_expansion",
                    "medium",
                    "Sensitive lineage has expanded to more destinations than the configured threshold.",
                    {
                        "destination_count": len(destinations),
                        "threshold": self.max_unique_destinations,
                    },
                )
            if cross_context_count > self.max_cross_context_transitions:
                add_finding(
                    "repeated_cross_context_propagation",
                    "medium",
                    "Cross-context propagation exceeds the configured transition threshold.",
                    {
                        "transition_count": cross_context_count,
                        "threshold": self.max_cross_context_transitions,
                    },
                )
            if len(purposes) > self.max_unique_purposes:
                add_finding(
                    "purpose_drift",
                    "high",
                    "Lineage evidence contains more distinct purposes than the configured privacy bound.",
                    {
                        "purpose_count": len(purposes),
                        "threshold": self.max_unique_purposes,
                        "purposes": sorted(purposes),
                    },
                )
            if max_context_fanout > self.max_context_fanout_threshold:
                add_finding(
                    "context_fanout",
                    "medium",
                    "A single processing context has propagated data to too many distinct destinations.",
                    {
                        "max_context_fanout": max_context_fanout,
                        "threshold": self.max_context_fanout_threshold,
                    },
                )
            if unauthorized_destinations:
                add_finding(
                    "unauthorized_destination_observed",
                    "critical" if self.block_on_unauthorized_destination else "high",
                    "Lineage contains destinations outside the supplied authorization context set.",
                    {"unauthorized_destinations": sorted(unauthorized_destinations)},
                )
            if high_risk_destinations:
                add_finding(
                    "high_risk_destination_observed",
                    "high",
                    "Lineage reached one or more contexts configured as high-risk for privacy propagation.",
                    {"high_risk_destinations": sorted(high_risk_destinations)},
                )
            if cycle_detected:
                add_finding(
                    "context_cycle_detected",
                    "medium",
                    "Privacy lineage contains a non-whitelisted context cycle.",
                )
            if post_deletion_processing:
                add_finding(
                    "processing_after_verified_deletion",
                    "critical",
                    "Lineage indicates processing after a verified deletion completion timestamp.",
                    {
                        "post_deletion_event_count": len(post_deletion_processing),
                        "deletion_completed_at": deletion_completed_at,
                    },
                )
            if incomplete_event_count:
                add_finding(
                    "incomplete_lineage_events",
                    "low",
                    "Some lineage records lacked enough source/destination evidence for flow analysis.",
                    {"incomplete_event_count": incomplete_event_count},
                )

            destination_expansion_score = self._bounded_ratio(
                len(destinations), self.max_unique_destinations
            )
            cross_context_score = self._bounded_ratio(
                cross_context_count, self.max_cross_context_transitions
            )
            purpose_drift_score = self._bounded_ratio(len(purposes), self.max_unique_purposes)
            unauthorized_score = 1.0 if unauthorized_destinations else 0.0
            post_deletion_score = 1.0 if post_deletion_processing else 0.0
            cycle_score = 1.0 if cycle_detected else 0.0

            flow_risk_score = (
                self.risk_weights["destination_expansion"] * destination_expansion_score
                + self.risk_weights["cross_context"] * cross_context_score
                + self.risk_weights["purpose_drift"] * purpose_drift_score
                + self.risk_weights["unauthorized_destination"] * unauthorized_score
                + self.risk_weights["post_deletion_processing"] * post_deletion_score
                + self.risk_weights["cycle"] * cycle_score
            )
            flow_risk_score = round(max(0.0, min(1.0, flow_risk_score)), 6)

            score_decision = PrivacyDecision.ALLOW.value
            if flow_risk_score >= self.escalate_score_threshold:
                score_decision = PrivacyDecision.ESCALATE.value
            elif flow_risk_score >= self.modify_score_threshold:
                score_decision = PrivacyDecision.MODIFY.value

            explicit_decisions: List[str] = [score_decision]
            if post_deletion_processing:
                explicit_decisions.append(
                    PrivacyDecision.BLOCK.value
                    if self.block_on_post_deletion_processing
                    else PrivacyDecision.ESCALATE.value
                )
            if unauthorized_destinations:
                if self.block_on_unauthorized_destination:
                    explicit_decisions.append(PrivacyDecision.BLOCK.value)
                elif self.escalate_on_unauthorized_destination:
                    explicit_decisions.append(PrivacyDecision.ESCALATE.value)
            if len(purposes) > self.max_unique_purposes and self.escalate_on_purpose_drift:
                explicit_decisions.append(PrivacyDecision.ESCALATE.value)
            if cycle_detected and self.escalate_on_context_cycle:
                explicit_decisions.append(PrivacyDecision.ESCALATE.value)
            if high_risk_destinations:
                explicit_decisions.append(PrivacyDecision.ESCALATE.value)

            decision = combine_privacy_decisions(*explicit_decisions)
            status = "pass" if decision == PrivacyDecision.ALLOW.value else "review"
            if decision == PrivacyDecision.BLOCK.value:
                status = "fail"

            result = PrivacyFlowAssessment(
                request_id=request_id,
                record_id=record_id,
                subject_id=subject_id,
                decision=decision_value(decision),
                status=status,
                stage=stage or self.default_stage,
                event_count=len(lineage_events),
                edge_count=len(edges),
                cross_context_transition_count=cross_context_count,
                unique_context_count=len(contexts),
                unique_destination_count=len(destinations),
                max_context_fanout=max_context_fanout,
                unique_purpose_count=len(purposes),
                cycle_detected=cycle_detected,
                unauthorized_destinations=tuple(sorted(unauthorized_destinations)),
                high_risk_destinations=tuple(sorted(high_risk_destinations)),
                post_deletion_processing_count=len(post_deletion_processing),
                incomplete_event_count=incomplete_event_count,
                flow_risk_score=flow_risk_score,
                findings=tuple(findings),
                policy_fingerprint=self.policy_fingerprint,
                created_at=utc_iso(),
            )
            output = result.to_dict()
            output["schema_version"] = ASSESSMENT_SCHEMA_VERSION
            output["edges"] = [edge.to_dict() for edge in edges]
            if safe_context:
                output["context"] = safe_context
            return output
        except Exception as exc:
            if isinstance(exc, PrivacyError):
                raise
            normalized = normalize_privacy_exception(
                exc,
                stage="privacy_flow_analyzer.analyze",
                context={
                    "request_id": request_id,
                    "record_id": record_id,
                    "subject_id": subject_id,
                },
            )
            if self.strict_mode:
                raise normalized from exc
            return {
                "schema_version": ASSESSMENT_SCHEMA_VERSION,
                "request_id": request_id,
                "record_id": record_id,
                "subject_id": subject_id,
                "decision": PrivacyDecision.ESCALATE.value,
                "status": "error",
                "stage": stage or self.default_stage,
                "policy_fingerprint": self.policy_fingerprint,
                "created_at": utc_iso(),
                "error": {
                    "error_type": str(getattr(getattr(normalized, "error_type", None), "value", "policy_evaluation_failed")),
                    "error_code": str(getattr(normalized, "error_code", "PRV-3025")),
                    "severity": str(getattr(getattr(normalized, "severity", None), "value", "high")),
                    "retryable": bool(getattr(normalized, "retryable", True)),
                    "message": str(getattr(normalized, "message", "Privacy flow analysis failed.")),
                },
            }


__all__ = [
    "MODULE_VERSION",
    "ASSESSMENT_SCHEMA_VERSION",
    "PrivacyFlowEdge",
    "PrivacyFlowFinding",
    "PrivacyFlowAssessment",
    "PrivacyFlowAnalyzer",
]
