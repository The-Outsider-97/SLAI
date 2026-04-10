"""
- Stores trace/span archives and correlated event timelines.
- Maintains historical incident fingerprints for fast regression matching.
- Persists SLO/SLA history and prior alert suppressions for context-aware alerting.
- Records remediation actions and resulting recovery metrics for playbook learning.
- Exposes retrieval APIs for:
    - incident_similarities(error_signature)
    - latency_trend(agent_name, percentile, window)
    - runbook_outcome(playbook_id)
"""

from __future__ import annotations

import json
import math
import time

from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from threading import RLock
from typing import Any, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .utils.config_loader import load_global_config, get_config_section
from .utils.observability_error import (ObservabilityError, ObservabilityMemoryError,
                                        normalize_observability_exception)
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Observability Memory")
printer = PrettyPrinter


@dataclass
class TimelineEvent:
    trace_id: str
    event_type: str
    timestamp_ms: float
    agent_name: Optional[str] = None
    severity: str = "info"
    message: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_keys: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentLatencySample:
    agent_name: str
    timestamp_ms: float
    duration_ms: float
    trace_id: str
    span_id: str
    status: str = "ok"
    operation_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TraceArchive:
    trace_id: str
    stored_at_ms: float
    incident_level: str
    agent_spans: List[Dict[str, Any]]
    timeline: List[Dict[str, Any]]
    summary: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_signature: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IncidentFingerprint:
    incident_id: str
    error_signature: str
    signature_hash: str
    severity: str
    first_seen_ms: float
    last_seen_ms: float
    occurrence_count: int = 1
    related_trace_ids: List[str] = field(default_factory=list)
    related_agents: List[str] = field(default_factory=list)
    root_cause_candidates: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ObjectiveRecord:
    objective_type: str
    service: str
    objective_name: str
    observed: float
    target: float
    comparator: str
    status: str
    window_seconds: Optional[float]
    timestamp_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AlertSuppressionRecord:
    alert_key: str
    reason: str
    source: str
    created_at_ms: float
    expires_at_ms: float
    incident_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.expires_at_ms > time.time() * 1000.0

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["is_active"] = self.is_active
        return payload


@dataclass
class RemediationRecord:
    playbook_id: str
    incident_id: Optional[str]
    action_name: str
    status: str
    started_at_ms: float
    completed_at_ms: Optional[float] = None
    recovery_metrics: Dict[str, float] = field(default_factory=dict)
    notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def recovery_time_ms(self) -> Optional[float]:
        if self.completed_at_ms is None:
            return None
        return max(0.0, self.completed_at_ms - self.started_at_ms)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["recovery_time_ms"] = self.recovery_time_ms
        return payload


class ObservabilityMemory:
    def __init__(self) -> None:
        self.config = load_global_config()
        self.memory_config = get_config_section("observability_memory")
        self._lock = RLock()

        self.enabled = bool(self.memory_config.get("enabled", True))
        self.archive_raw_spans = bool(self.memory_config.get("archive_raw_spans", True))
        self.persist_timelines = bool(self.memory_config.get("persist_timelines", True))
        self.persist_incidents = bool(self.memory_config.get("persist_incidents", True))
        self.persist_slo_history = bool(self.memory_config.get("persist_slo_history", True))
        self.persist_suppressions = bool(self.memory_config.get("persist_suppressions", True))
        self.persist_remediations = bool(self.memory_config.get("persist_remediations", True))
        self.enable_snapshot_persistence = bool(self.memory_config.get("enable_snapshot_persistence", False))
        self.flush_snapshot_on_write = bool(self.memory_config.get("flush_snapshot_on_write", False))
        self.write_pretty_json = bool(self.memory_config.get("write_pretty_json", True))

        self.max_trace_archives = int(self.memory_config.get("max_trace_archives", 2000))
        self.max_raw_spans_per_trace = int(self.memory_config.get("max_raw_spans_per_trace", 5000))
        self.max_timeline_events_per_trace = int(self.memory_config.get("max_timeline_events_per_trace", 1000))
        self.max_incident_fingerprints = int(self.memory_config.get("max_incident_fingerprints", 2000))
        self.max_trace_ids_per_incident = int(self.memory_config.get("max_trace_ids_per_incident", 50))
        self.max_related_agents_per_incident = int(self.memory_config.get("max_related_agents_per_incident", 25))
        self.max_root_cause_candidates = int(self.memory_config.get("max_root_cause_candidates", 10))
        self.max_recommended_actions = int(self.memory_config.get("max_recommended_actions", 10))
        self.max_objective_history_per_key = int(self.memory_config.get("max_objective_history_per_key", 365))
        self.max_alert_suppressions = int(self.memory_config.get("max_alert_suppressions", 1000))
        self.max_runbook_records_per_playbook = int(self.memory_config.get("max_runbook_records_per_playbook", 1000))
        self.max_agent_latency_samples_per_agent = int(self.memory_config.get("max_agent_latency_samples_per_agent", 20000))
        self.max_shared_memory_actions = int(self.memory_config.get("max_shared_memory_actions", 5))

        self.similarity_top_k_default = int(self.memory_config.get("similarity_top_k_default", 5))
        self.similarity_min_score = float(self.memory_config.get("similarity_min_score", 0.2))
        self.default_latency_window_seconds = float(self.memory_config.get("default_latency_window_seconds", 86400))
        self.default_latency_bucket_seconds = int(self.memory_config.get("default_latency_bucket_seconds", 3600))
        self.default_suppression_ttl_seconds = float(self.memory_config.get("default_suppression_ttl_seconds", 1800))
        self.supported_percentiles = tuple(
            int(value) for value in self.memory_config.get("supported_percentiles", [50, 75, 90, 95, 99])
        )
        self.snapshot_path = self._resolve_storage_path(self.memory_config.get("snapshot_path", "observability/data/observability_memory_snapshot.json"))

        self._trace_archives: "OrderedDict[str, TraceArchive]" = OrderedDict()
        self._trace_timelines: Dict[str, Deque[TimelineEvent]] = {}
        self._incident_fingerprints: "OrderedDict[str, IncidentFingerprint]" = OrderedDict()
        self._signature_index: Dict[str, str] = {}
        self._objective_history: Dict[Tuple[str, str, str], Deque[ObjectiveRecord]] = defaultdict(
            lambda: deque(maxlen=self.max_objective_history_per_key)
        )
        self._alert_suppressions: "OrderedDict[str, AlertSuppressionRecord]" = OrderedDict()
        self._remediation_history: Dict[str, Deque[RemediationRecord]] = defaultdict(
            lambda: deque(maxlen=self.max_runbook_records_per_playbook)
        )
        self._agent_latency_samples: Dict[str, Deque[AgentLatencySample]] = defaultdict(
            lambda: deque(maxlen=self.max_agent_latency_samples_per_agent)
        )

        if self.enable_snapshot_persistence:
            self._load_snapshot()

    # ---------------------------------------------------------------------
    # Public write APIs
    # ---------------------------------------------------------------------
    def archive_trace(self, trace_id: str, agent_spans: Sequence[Mapping[str, Any]], *,
        timeline: Optional[Sequence[Mapping[str, Any]]] = None,
        summary: Optional[Mapping[str, Any]] = None,
        incident_level: str = "info",
        error_signature: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            self._ensure_enabled(operation="archive_trace")
            trace_id = self._require_non_empty_str(trace_id, field_name="trace_id", operation="archive_trace")
            normalized_spans, span_warnings = self._normalize_spans(trace_id, agent_spans)
            normalized_timeline = self._normalize_timeline(trace_id, timeline or [])
            archive_summary = self._build_trace_summary(
                trace_id=trace_id,
                normalized_spans=normalized_spans,
                normalized_timeline=normalized_timeline,
                provided_summary=summary,
                incident_level=incident_level,
                error_signature=error_signature,
            )
            if span_warnings:
                archive_summary.setdefault("parse_warnings", []).extend(span_warnings)

            stored_spans = normalized_spans if self.archive_raw_spans else []
            archive = TraceArchive(
                trace_id=trace_id,
                stored_at_ms=self._now_ms(),
                incident_level=str(incident_level or "info").lower(),
                agent_spans=stored_spans,
                timeline=normalized_timeline,
                summary=archive_summary,
                metadata=self._safe_mapping(metadata),
                error_signature=error_signature or archive_summary.get("error_signature"),
            )

            with self._lock:
                self._trace_archives[trace_id] = archive
                self._trace_archives.move_to_end(trace_id)
                self._trace_timelines[trace_id] = deque(
                    (TimelineEvent(**event) for event in normalized_timeline),
                    maxlen=self.max_timeline_events_per_trace,
                )
                self._evict_ordered_dict(self._trace_archives, self.max_trace_archives)
                self._evict_trace_timeline_orphans()
                self._index_trace_latency_samples(trace_id, normalized_spans)

            self._flush_snapshot_if_configured()
            logger.info("Archived observability trace '%s' with %s spans.", trace_id, len(normalized_spans))
            return archive.to_dict()
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="memory.write",
                operation="archive_trace",
                context={"trace_id": trace_id},
            ) from exc

    def append_timeline_event(self, trace_id: str, *, event_type: str,
        message: Optional[str] = None,
        agent_name: Optional[str] = None,
        severity: str = "info",
        timestamp_ms: Optional[float] = None,
        payload: Optional[Mapping[str, Any]] = None,
        correlation_keys: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, Any]:
        try:
            self._ensure_enabled(operation="append_timeline_event")
            trace_id = self._require_non_empty_str(trace_id, field_name="trace_id", operation="append_timeline_event")
            event_type = self._require_non_empty_str(event_type, field_name="event_type", operation="append_timeline_event")
            event = TimelineEvent(
                trace_id=trace_id,
                event_type=event_type,
                timestamp_ms=float(timestamp_ms if timestamp_ms is not None else self._now_ms()),
                agent_name=self._optional_str(agent_name),
                severity=str(severity or "info").lower(),
                message=self._optional_str(message),
                payload=self._safe_mapping(payload),
                correlation_keys={str(k): str(v) for k, v in self._safe_mapping(correlation_keys).items()},
            )

            with self._lock:
                timeline = self._trace_timelines.setdefault(
                    trace_id,
                    deque(maxlen=self.max_timeline_events_per_trace),
                )
                timeline.append(event)
                archive = self._trace_archives.get(trace_id)
                if archive is not None and self.persist_timelines:
                    archive.timeline = [entry.to_dict() for entry in timeline]
                    archive.summary["event_count"] = len(archive.timeline)
                    archive.summary["latest_event_type"] = event.event_type

            self._flush_snapshot_if_configured()
            return event.to_dict()
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="memory.write",
                operation="append_timeline_event",
                context={"trace_id": trace_id},
            ) from exc

    def record_incident_fingerprint(self, *, incident_id: str, error_signature: str,
        severity: str = "warning",
        trace_id: Optional[str] = None,
        related_agents: Optional[Sequence[str]] = None,
        root_cause_candidates: Optional[Sequence[str]] = None,
        recommended_actions: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            self._ensure_enabled(operation="record_incident_fingerprint")
            if not self.persist_incidents:
                raise ObservabilityMemoryError(
                    operation="record_incident_fingerprint",
                    details="incident persistence is disabled by configuration",
                    context={"incident_id": incident_id},
                )

            incident_id = self._require_non_empty_str(
                incident_id,
                field_name="incident_id",
                operation="record_incident_fingerprint",
            )
            normalized_signature = self._normalize_signature(error_signature)
            if not normalized_signature:
                raise ObservabilityMemoryError(
                    operation="record_incident_fingerprint",
                    details="error_signature cannot be empty",
                    context={"incident_id": incident_id},
                )

            signature_hash = self._signature_hash(normalized_signature)
            related_trace_ids = [trace_id] if trace_id else []
            incident = self._incident_fingerprints.get(incident_id)
            if incident is None:
                incident = IncidentFingerprint(
                    incident_id=incident_id,
                    error_signature=normalized_signature,
                    signature_hash=signature_hash,
                    severity=str(severity or "warning").lower(),
                    first_seen_ms=self._now_ms(),
                    last_seen_ms=self._now_ms(),
                    related_trace_ids=related_trace_ids,
                    related_agents=self._dedupe_strings(related_agents, self.max_related_agents_per_incident),
                    root_cause_candidates=self._dedupe_strings(root_cause_candidates, self.max_root_cause_candidates),
                    recommended_actions=self._dedupe_strings(recommended_actions, self.max_recommended_actions),
                    metadata=self._safe_mapping(metadata),
                )
            else:
                incident.last_seen_ms = self._now_ms()
                incident.occurrence_count += 1
                incident.severity = str(severity or incident.severity).lower()
                incident.related_trace_ids = self._merge_limited_strings(
                    incident.related_trace_ids,
                    related_trace_ids,
                    limit=self.max_trace_ids_per_incident,
                )
                incident.related_agents = self._merge_limited_strings(
                    incident.related_agents,
                    related_agents,
                    limit=self.max_related_agents_per_incident,
                )
                incident.root_cause_candidates = self._merge_limited_strings(
                    incident.root_cause_candidates,
                    root_cause_candidates,
                    limit=self.max_root_cause_candidates,
                )
                incident.recommended_actions = self._merge_limited_strings(
                    incident.recommended_actions,
                    recommended_actions,
                    limit=self.max_recommended_actions,
                )
                incident.metadata.update(self._safe_mapping(metadata))

            with self._lock:
                self._incident_fingerprints[incident_id] = incident
                self._incident_fingerprints.move_to_end(incident_id)
                self._signature_index[signature_hash] = incident_id
                self._evict_ordered_dict(self._incident_fingerprints, self.max_incident_fingerprints)
                self._rebuild_signature_index_if_needed()

            self._flush_snapshot_if_configured()
            return incident.to_dict()
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="memory.write",
                operation="record_incident_fingerprint",
                context={"incident_id": incident_id},
            ) from exc

    def record_slo_history(self, *, service: str, slo_name: str,
                           observed: float, target: float, comparator: str = "lte",
        timestamp_ms: Optional[float] = None,
        window_seconds: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._record_objective_history(
            objective_type="slo",
            service=service,
            objective_name=slo_name,
            observed=observed,
            target=target,
            comparator=comparator,
            timestamp_ms=timestamp_ms,
            window_seconds=window_seconds,
            metadata=metadata,
        )

    def record_sla_history(
        self,
        *,
        service: str,
        sla_name: str,
        observed: float,
        target: float,
        comparator: str = "lte",
        timestamp_ms: Optional[float] = None,
        window_seconds: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._record_objective_history(
            objective_type="sla",
            service=service,
            objective_name=sla_name,
            observed=observed,
            target=target,
            comparator=comparator,
            timestamp_ms=timestamp_ms,
            window_seconds=window_seconds,
            metadata=metadata,
        )

    def record_alert_suppression(
        self,
        *,
        alert_key: str,
        reason: str,
        source: str,
        ttl_seconds: Optional[float] = None,
        incident_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            self._ensure_enabled(operation="record_alert_suppression")
            if not self.persist_suppressions:
                raise ObservabilityMemoryError(
                    operation="record_alert_suppression",
                    details="alert suppression persistence is disabled by configuration",
                    context={"alert_key": alert_key},
                )

            alert_key = self._require_non_empty_str(alert_key, field_name="alert_key", operation="record_alert_suppression")
            reason = self._require_non_empty_str(reason, field_name="reason", operation="record_alert_suppression")
            source = self._require_non_empty_str(source, field_name="source", operation="record_alert_suppression")
            created_at_ms = self._now_ms()
            ttl = float(ttl_seconds if ttl_seconds is not None else self.default_suppression_ttl_seconds)
            expires_at_ms = created_at_ms + max(0.0, ttl) * 1000.0

            suppression = AlertSuppressionRecord(
                alert_key=alert_key,
                reason=reason,
                source=source,
                created_at_ms=created_at_ms,
                expires_at_ms=expires_at_ms,
                incident_id=self._optional_str(incident_id),
                metadata=self._safe_mapping(metadata),
            )

            with self._lock:
                self._alert_suppressions[alert_key] = suppression
                self._alert_suppressions.move_to_end(alert_key)
                self._evict_ordered_dict(self._alert_suppressions, self.max_alert_suppressions)
                self._purge_expired_suppressions_locked()

            self._flush_snapshot_if_configured()
            return suppression.to_dict()
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="memory.write",
                operation="record_alert_suppression",
                context={"alert_key": alert_key},
            ) from exc

    def record_remediation_action(
        self,
        *,
        playbook_id: str,
        action_name: str,
        status: str,
        started_at_ms: Optional[float] = None,
        completed_at_ms: Optional[float] = None,
        incident_id: Optional[str] = None,
        recovery_metrics: Optional[Mapping[str, float]] = None,
        notes: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            self._ensure_enabled(operation="record_remediation_action")
            if not self.persist_remediations:
                raise ObservabilityMemoryError(
                    operation="record_remediation_action",
                    details="remediation persistence is disabled by configuration",
                    context={"playbook_id": playbook_id},
                )

            playbook_id = self._require_non_empty_str(playbook_id, field_name="playbook_id", operation="record_remediation_action")
            action_name = self._require_non_empty_str(action_name, field_name="action_name", operation="record_remediation_action")
            started = float(started_at_ms if started_at_ms is not None else self._now_ms())
            completed = float(completed_at_ms) if completed_at_ms is not None else None
            if completed is not None and completed < started:
                raise ObservabilityMemoryError(
                    operation="record_remediation_action",
                    details="completed_at_ms cannot be earlier than started_at_ms",
                    context={"playbook_id": playbook_id, "started_at_ms": started, "completed_at_ms": completed},
                )

            record = RemediationRecord(
                playbook_id=playbook_id,
                incident_id=self._optional_str(incident_id),
                action_name=action_name,
                status=str(status or "unknown").lower(),
                started_at_ms=started,
                completed_at_ms=completed,
                recovery_metrics=self._safe_numeric_mapping(recovery_metrics),
                notes=self._optional_str(notes),
                metadata=self._safe_mapping(metadata),
            )

            with self._lock:
                self._remediation_history[playbook_id].append(record)

            self._flush_snapshot_if_configured()
            return record.to_dict()
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="memory.write",
                operation="record_remediation_action",
                context={"playbook_id": playbook_id},
            ) from exc

    # ---------------------------------------------------------------------
    # Public read APIs
    # ---------------------------------------------------------------------
    def get_trace_archive(self, trace_id: str) -> Optional[Dict[str, Any]]:
        try:
            trace_id = self._require_non_empty_str(trace_id, field_name="trace_id", operation="get_trace_archive")
            with self._lock:
                archive = self._trace_archives.get(trace_id)
                return archive.to_dict() if archive is not None else None
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="memory.read",
                operation="get_trace_archive",
                context={"trace_id": trace_id},
            ) from exc

    def get_event_timeline(self, trace_id: str) -> List[Dict[str, Any]]:
        try:
            trace_id = self._require_non_empty_str(trace_id, field_name="trace_id", operation="get_event_timeline")
            with self._lock:
                timeline = self._trace_timelines.get(trace_id, deque())
                return [event.to_dict() for event in timeline]
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="memory.read",
                operation="get_event_timeline",
                context={"trace_id": trace_id},
            ) from exc

    def incident_similarities(self, error_signature: str, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        try:
            normalized_signature = self._normalize_signature(error_signature)
            if not normalized_signature:
                raise ObservabilityMemoryError(
                    operation="incident_similarities",
                    details="error_signature cannot be empty",
                )

            top_k = int(limit if limit is not None else self.similarity_top_k_default)
            query_hash = self._signature_hash(normalized_signature)
            query_tokens = self._signature_tokens(normalized_signature)

            matches: List[Dict[str, Any]] = []
            with self._lock:
                for incident in self._incident_fingerprints.values():
                    exact_hash = 1.0 if incident.signature_hash == query_hash else 0.0
                    token_score = self._jaccard_similarity(query_tokens, self._signature_tokens(incident.error_signature))
                    sequence_score = SequenceMatcher(None, normalized_signature, incident.error_signature).ratio()
                    combined_score = max(exact_hash, (0.25 * exact_hash) + (0.35 * token_score) + (0.40 * sequence_score))
                    if combined_score < self.similarity_min_score:
                        continue
                    matches.append(
                        {
                            "incident_id": incident.incident_id,
                            "error_signature": incident.error_signature,
                            "severity": incident.severity,
                            "occurrence_count": incident.occurrence_count,
                            "similarity_score": round(combined_score, 6),
                            "related_trace_ids": incident.related_trace_ids,
                            "related_agents": incident.related_agents,
                            "recommended_actions": incident.recommended_actions,
                            "root_cause_candidates": incident.root_cause_candidates,
                            "last_seen_ms": incident.last_seen_ms,
                        }
                    )

            matches.sort(key=lambda item: (item["similarity_score"], item["occurrence_count"], item["last_seen_ms"]), reverse=True)
            return matches[: max(0, top_k)]
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="memory.index",
                operation="incident_similarities",
                context={"limit": limit},
            ) from exc

    def latency_trend(self, agent_name: str, percentile: int, window: Any) -> Dict[str, Any]:
        try:
            agent_name = self._require_non_empty_str(agent_name, field_name="agent_name", operation="latency_trend")
            percentile_value = int(percentile)
            if percentile_value not in self.supported_percentiles:
                raise ObservabilityMemoryError(
                    operation="latency_trend",
                    details=f"unsupported percentile '{percentile_value}'",
                    context={"supported_percentiles": list(self.supported_percentiles)},
                )

            window_seconds = self._parse_window_seconds(window)
            bucket_seconds = min(self.default_latency_bucket_seconds, max(60, int(window_seconds // 24) or self.default_latency_bucket_seconds))
            cutoff_ms = self._now_ms() - (window_seconds * 1000.0)

            with self._lock:
                samples = [sample for sample in self._agent_latency_samples.get(agent_name, deque()) if sample.timestamp_ms >= cutoff_ms]

            if not samples:
                return {
                    "agent_name": agent_name,
                    "percentile": percentile_value,
                    "window_seconds": window_seconds,
                    "bucket_seconds": bucket_seconds,
                    "sample_count": 0,
                    "overall_percentile_ms": 0.0,
                    "trend_points": [],
                }

            buckets: Dict[int, List[float]] = defaultdict(list)
            for sample in samples:
                bucket_id = int(sample.timestamp_ms // (bucket_seconds * 1000.0))
                buckets[bucket_id].append(sample.duration_ms)

            trend_points: List[Dict[str, Any]] = []
            for bucket_id in sorted(buckets):
                values = buckets[bucket_id]
                bucket_start_ms = bucket_id * bucket_seconds * 1000.0
                trend_points.append(
                    {
                        "bucket_start_ms": float(bucket_start_ms),
                        "bucket_end_ms": float(bucket_start_ms + bucket_seconds * 1000.0),
                        "sample_count": len(values),
                        "percentile_ms": round(self._percentile(values, percentile_value), 6),
                        "average_ms": round(sum(values) / len(values), 6),
                        "max_ms": round(max(values), 6),
                    }
                )

            overall = self._percentile([sample.duration_ms for sample in samples], percentile_value)
            return {
                "agent_name": agent_name,
                "percentile": percentile_value,
                "window_seconds": window_seconds,
                "bucket_seconds": bucket_seconds,
                "sample_count": len(samples),
                "overall_percentile_ms": round(overall, 6),
                "trend_points": trend_points,
            }
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="memory.read",
                operation="latency_trend",
                context={"agent_name": agent_name, "percentile": percentile, "window": window},
            ) from exc

    def runbook_outcome(self, playbook_id: str) -> Dict[str, Any]:
        try:
            playbook_id = self._require_non_empty_str(playbook_id, field_name="playbook_id", operation="runbook_outcome")
            with self._lock:
                records = list(self._remediation_history.get(playbook_id, deque()))

            if not records:
                return {
                    "playbook_id": playbook_id,
                    "attempt_count": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "success_rate": 0.0,
                    "recovery_time_ms": {
                        "average": 0.0,
                        "median": 0.0,
                        "p95": 0.0,
                    },
                    "last_outcome": None,
                    "status_distribution": {},
                }

            success_count = sum(1 for record in records if record.status in {"success", "completed", "resolved"})
            failure_count = sum(1 for record in records if record.status in {"failed", "error", "timeout"})
            durations = [record.recovery_time_ms for record in records if record.recovery_time_ms is not None]
            status_distribution = Counter(record.status for record in records)
            last_record = max(records, key=lambda record: record.started_at_ms)

            return {
                "playbook_id": playbook_id,
                "attempt_count": len(records),
                "success_count": success_count,
                "failure_count": failure_count,
                "success_rate": round(success_count / len(records), 6),
                "recovery_time_ms": {
                    "average": round(sum(durations) / len(durations), 6) if durations else 0.0,
                    "median": round(self._percentile(durations, 50), 6) if durations else 0.0,
                    "p95": round(self._percentile(durations, 95), 6) if durations else 0.0,
                },
                "last_outcome": last_record.to_dict(),
                "status_distribution": dict(status_distribution),
            }
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="memory.read",
                operation="runbook_outcome",
                context={"playbook_id": playbook_id},
            ) from exc

    def get_objective_history(
        self,
        *,
        objective_type: str,
        service: str,
        objective_name: str,
    ) -> List[Dict[str, Any]]:
        try:
            objective_type = self._require_non_empty_str(objective_type, field_name="objective_type", operation="get_objective_history")
            service = self._require_non_empty_str(service, field_name="service", operation="get_objective_history")
            objective_name = self._require_non_empty_str(objective_name, field_name="objective_name", operation="get_objective_history")
            key = (objective_type.lower(), service, objective_name)
            with self._lock:
                return [record.to_dict() for record in self._objective_history.get(key, deque())]
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="memory.read",
                operation="get_objective_history",
                context={"objective_type": objective_type, "service": service, "objective_name": objective_name},
            ) from exc

    def is_alert_suppressed(self, alert_key: str, *, as_of_ms: Optional[float] = None) -> Dict[str, Any]:
        try:
            alert_key = self._require_non_empty_str(alert_key, field_name="alert_key", operation="is_alert_suppressed")
            ts_ms = float(as_of_ms if as_of_ms is not None else self._now_ms())
            with self._lock:
                self._purge_expired_suppressions_locked(now_ms=ts_ms)
                record = self._alert_suppressions.get(alert_key)

            return {
                "alert_key": alert_key,
                "is_suppressed": bool(record and record.expires_at_ms > ts_ms),
                "suppression": record.to_dict() if record is not None else None,
            }
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="memory.read",
                operation="is_alert_suppressed",
                context={"alert_key": alert_key},
            ) from exc

    def build_shared_memory_context(
        self,
        *,
        trace_id: Optional[str] = None,
        incident_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            context: Dict[str, Any] = {
                "observability.trace_id": None,
                "observability.agent_spans": [],
                "observability.error_clusters": [],
                "observability.latency_p95": {},
                "observability.incident_level": "info",
                "observability.recommended_actions": [],
            }

            if trace_id:
                archive = self.get_trace_archive(trace_id)
                if archive is not None:
                    context["observability.trace_id"] = trace_id
                    context["observability.agent_spans"] = archive.get("agent_spans", [])
                    context["observability.incident_level"] = archive.get("incident_level", "info")
                    agent_names = archive.get("summary", {}).get("agent_names", [])
                    context["observability.latency_p95"] = {
                        agent: self.latency_trend(agent, 95, self.default_latency_window_seconds).get("overall_percentile_ms", 0.0)
                        for agent in agent_names
                    }
                    signature = archive.get("error_signature") or archive.get("summary", {}).get("error_signature")
                    if signature:
                        context["observability.error_clusters"] = self.incident_similarities(signature)

            if incident_id:
                with self._lock:
                    incident = self._incident_fingerprints.get(incident_id)
                if incident is not None:
                    context["observability.incident_level"] = incident.severity
                    context["observability.recommended_actions"] = incident.recommended_actions[: self.max_shared_memory_actions]
                    if not context["observability.error_clusters"]:
                        context["observability.error_clusters"] = [
                            {
                                "incident_id": incident.incident_id,
                                "error_signature": incident.error_signature,
                                "severity": incident.severity,
                                "occurrence_count": incident.occurrence_count,
                            }
                        ]

            return context
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="memory.read",
                operation="build_shared_memory_context",
                context={"trace_id": trace_id, "incident_id": incident_id},
            ) from exc

    def snapshot(self) -> Dict[str, Any]:
        try:
            with self._lock:
                return {
                    "trace_archives": {trace_id: archive.to_dict() for trace_id, archive in self._trace_archives.items()},
                    "incident_fingerprints": {
                        incident_id: incident.to_dict()
                        for incident_id, incident in self._incident_fingerprints.items()
                    },
                    "objective_history": {
                        f"{objective_type}:{service}:{objective_name}": [record.to_dict() for record in records]
                        for (objective_type, service, objective_name), records in self._objective_history.items()
                    },
                    "alert_suppressions": {
                        alert_key: record.to_dict() for alert_key, record in self._alert_suppressions.items()
                    },
                    "remediation_history": {
                        playbook_id: [record.to_dict() for record in records]
                        for playbook_id, records in self._remediation_history.items()
                    },
                    "agent_latency_samples": {
                        agent_name: [sample.to_dict() for sample in samples]
                        for agent_name, samples in self._agent_latency_samples.items()
                    },
                }
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="memory.read",
                operation="snapshot",
            ) from exc

    def persist_snapshot(self) -> Dict[str, Any]:
        try:
            payload = self.snapshot()
            self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            with self.snapshot_path.open("w", encoding="utf-8") as handle:
                if self.write_pretty_json:
                    json.dump(payload, handle, indent=2, sort_keys=True)
                else:
                    json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
            return {"snapshot_path": str(self.snapshot_path), "status": "written"}
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="memory.write",
                operation="persist_snapshot",
                context={"snapshot_path": str(self.snapshot_path)},
            ) from exc

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _record_objective_history(
        self,
        *,
        objective_type: str,
        service: str,
        objective_name: str,
        observed: float,
        target: float,
        comparator: str,
        timestamp_ms: Optional[float],
        window_seconds: Optional[float],
        metadata: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        try:
            self._ensure_enabled(operation="record_objective_history")
            if not self.persist_slo_history:
                raise ObservabilityMemoryError(
                    operation="record_objective_history",
                    details="objective history persistence is disabled by configuration",
                    context={"objective_type": objective_type},
                )

            objective_type = self._require_non_empty_str(objective_type, field_name="objective_type", operation="record_objective_history")
            service = self._require_non_empty_str(service, field_name="service", operation="record_objective_history")
            objective_name = self._require_non_empty_str(objective_name, field_name="objective_name", operation="record_objective_history")
            comparator = str(comparator or "lte").lower()
            if comparator not in {"lte", "gte"}:
                raise ObservabilityMemoryError(
                    operation="record_objective_history",
                    details=f"unsupported comparator '{comparator}'",
                    context={"supported": ["lte", "gte"]},
                )

            observed_value = float(observed)
            target_value = float(target)
            status = self._evaluate_objective_status(observed_value, target_value, comparator)
            record = ObjectiveRecord(
                objective_type=objective_type.lower(),
                service=service,
                objective_name=objective_name,
                observed=observed_value,
                target=target_value,
                comparator=comparator,
                status=status,
                window_seconds=float(window_seconds) if window_seconds is not None else None,
                timestamp_ms=float(timestamp_ms if timestamp_ms is not None else self._now_ms()),
                metadata=self._safe_mapping(metadata),
            )

            with self._lock:
                self._objective_history[(record.objective_type, record.service, record.objective_name)].append(record)

            self._flush_snapshot_if_configured()
            return record.to_dict()
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="memory.write",
                operation="record_objective_history",
                context={"objective_type": objective_type, "service": service, "objective_name": objective_name},
            ) from exc

    def _load_snapshot(self) -> None:
        if not self.snapshot_path.exists():
            return

        try:
            with self.snapshot_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

            with self._lock:
                self._trace_archives.clear()
                self._trace_timelines.clear()
                self._incident_fingerprints.clear()
                self._signature_index.clear()
                self._objective_history.clear()
                self._alert_suppressions.clear()
                self._remediation_history.clear()
                self._agent_latency_samples.clear()

                for trace_id, raw_archive in payload.get("trace_archives", {}).items():
                    archive = TraceArchive(**raw_archive)
                    self._trace_archives[trace_id] = archive
                    self._trace_timelines[trace_id] = deque(
                        (TimelineEvent(**event) for event in archive.timeline),
                        maxlen=self.max_timeline_events_per_trace,
                    )
                self._evict_ordered_dict(self._trace_archives, self.max_trace_archives)

                for incident_id, raw_incident in payload.get("incident_fingerprints", {}).items():
                    incident = IncidentFingerprint(**raw_incident)
                    self._incident_fingerprints[incident_id] = incident
                    self._signature_index[incident.signature_hash] = incident_id
                self._evict_ordered_dict(self._incident_fingerprints, self.max_incident_fingerprints)

                for composite_key, records in payload.get("objective_history", {}).items():
                    objective_type, service, objective_name = composite_key.split(":", 2)
                    queue: Deque[ObjectiveRecord] = deque(maxlen=self.max_objective_history_per_key)
                    for record in records:
                        queue.append(ObjectiveRecord(**record))
                    self._objective_history[(objective_type, service, objective_name)] = queue

                for alert_key, raw_record in payload.get("alert_suppressions", {}).items():
                    raw_copy = dict(raw_record)
                    raw_copy.pop("is_active", None)
                    self._alert_suppressions[alert_key] = AlertSuppressionRecord(**raw_copy)
                self._purge_expired_suppressions_locked()
                self._evict_ordered_dict(self._alert_suppressions, self.max_alert_suppressions)

                for playbook_id, records in payload.get("remediation_history", {}).items():
                    queue = deque(maxlen=self.max_runbook_records_per_playbook)
                    for record in records:
                        raw_copy = dict(record)
                        raw_copy.pop("recovery_time_ms", None)
                        queue.append(RemediationRecord(**raw_copy))
                    self._remediation_history[playbook_id] = queue

                for agent_name, records in payload.get("agent_latency_samples", {}).items():
                    queue = deque(maxlen=self.max_agent_latency_samples_per_agent)
                    for record in records:
                        queue.append(AgentLatencySample(**record))
                    self._agent_latency_samples[agent_name] = queue

            logger.info("Loaded observability memory snapshot from '%s'.", self.snapshot_path)
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="memory.read",
                operation="load_snapshot",
                context={"snapshot_path": str(self.snapshot_path)},
            ) from exc

    def _normalize_spans(
        self,
        trace_id: str,
        spans: Sequence[Mapping[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if spans is None:
            raise ObservabilityMemoryError(
                operation="archive_trace",
                details="agent_spans cannot be None",
                context={"trace_id": trace_id},
            )

        normalized: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        for index, raw_span in enumerate(list(spans)[: self.max_raw_spans_per_trace]):
            span = dict(raw_span)
            agent_name = str(
                span.get("agent_name")
                or span.get("agent")
                or span.get("service")
                or "unknown_agent"
            )
            span_id = str(span.get("span_id") or span.get("id") or f"{trace_id}-span-{index}")
            start_ms = self._coerce_float(span.get("start_ms", span.get("start_time_ms", 0.0)), 0.0)
            end_ms = self._coerce_float(span.get("end_ms", span.get("end_time_ms", start_ms)), start_ms)
            if end_ms < start_ms:
                warnings.append(
                    {
                        "type": "negative_duration",
                        "span_id": span_id,
                        "agent_name": agent_name,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                    }
                )
                end_ms = start_ms

            status = str(span.get("status", span.get("result", "ok")) or "ok").lower()
            duration_ms = max(0.0, end_ms - start_ms)
            normalized.append(
                {
                    "trace_id": str(span.get("trace_id") or trace_id),
                    "span_id": span_id,
                    "agent_name": agent_name,
                    "operation_name": self._optional_str(span.get("operation_name") or span.get("operation") or span.get("name")),
                    "parent_span_id": self._optional_str(span.get("parent_span_id") or span.get("parent_id")),
                    "attempt": int(self._coerce_float(span.get("attempt", 1), 1.0)),
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "duration_ms": duration_ms,
                    "status": status,
                    "metadata": self._safe_mapping(span.get("metadata") or {}),
                }
            )

        if len(spans) > self.max_raw_spans_per_trace:
            warnings.append(
                {
                    "type": "trace_span_truncated",
                    "trace_id": trace_id,
                    "stored_spans": len(normalized),
                    "discarded_spans": len(spans) - len(normalized),
                }
            )

        normalized.sort(key=lambda item: (item["start_ms"], item["end_ms"], item["span_id"]))
        return normalized, warnings

    def _normalize_timeline(
        self,
        trace_id: str,
        timeline: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for raw_event in list(timeline)[: self.max_timeline_events_per_trace]:
            event = TimelineEvent(
                trace_id=trace_id,
                event_type=str(raw_event.get("event_type") or raw_event.get("type") or "event"),
                timestamp_ms=float(raw_event.get("timestamp_ms") or raw_event.get("ts_ms") or self._now_ms()),
                agent_name=self._optional_str(raw_event.get("agent_name") or raw_event.get("agent")),
                severity=str(raw_event.get("severity") or "info").lower(),
                message=self._optional_str(raw_event.get("message")),
                payload=self._safe_mapping(raw_event.get("payload")),
                correlation_keys={
                    str(key): str(value)
                    for key, value in self._safe_mapping(raw_event.get("correlation_keys")).items()
                },
            )
            normalized.append(event.to_dict())
        normalized.sort(key=lambda item: item["timestamp_ms"])
        return normalized

    def _build_trace_summary(
        self,
        *,
        trace_id: str,
        normalized_spans: Sequence[Mapping[str, Any]],
        normalized_timeline: Sequence[Mapping[str, Any]],
        provided_summary: Optional[Mapping[str, Any]],
        incident_level: str,
        error_signature: Optional[str],
    ) -> Dict[str, Any]:
        start_ms = min((span["start_ms"] for span in normalized_spans), default=0.0)
        end_ms = max((span["end_ms"] for span in normalized_spans), default=start_ms)
        per_agent_duration_ms: Dict[str, float] = defaultdict(float)
        per_agent_span_count: Dict[str, int] = defaultdict(int)
        status_counts: Dict[str, int] = defaultdict(int)
        related_agents: List[str] = []
        error_spans: List[str] = []

        for span in normalized_spans:
            agent_name = str(span["agent_name"])
            per_agent_duration_ms[agent_name] += float(span["duration_ms"])
            per_agent_span_count[agent_name] += 1
            status_counts[str(span["status"])] += 1
            related_agents.append(agent_name)
            if str(span["status"]) in {"error", "timeout", "failed"}:
                error_spans.append(f"{agent_name}:{span.get('operation_name') or span['span_id']}:{span['status']}")

        provided = dict(provided_summary or {})
        computed_error_signature = error_signature or (" | ".join(error_spans) if error_spans else None)
        return {
            "trace_id": trace_id,
            "span_count": len(normalized_spans),
            "event_count": len(normalized_timeline),
            "wall_clock_start_ms": start_ms,
            "wall_clock_end_ms": end_ms,
            "total_duration_ms": max(0.0, end_ms - start_ms),
            "incident_level": str(incident_level or "info").lower(),
            "agent_names": sorted(set(related_agents)),
            "per_agent_duration_ms": {agent: round(duration, 6) for agent, duration in per_agent_duration_ms.items()},
            "per_agent_span_count": dict(per_agent_span_count),
            "status_counts": dict(status_counts),
            "error_signature": computed_error_signature,
            **provided,
        }

    def _index_trace_latency_samples(self, trace_id: str, normalized_spans: Sequence[Mapping[str, Any]]) -> None:
        for span in normalized_spans:
            agent_name = str(span["agent_name"])
            self._agent_latency_samples[agent_name].append(
                AgentLatencySample(
                    agent_name=agent_name,
                    timestamp_ms=float(span["end_ms"]),
                    duration_ms=float(span["duration_ms"]),
                    trace_id=trace_id,
                    span_id=str(span["span_id"]),
                    status=str(span["status"]),
                    operation_name=self._optional_str(span.get("operation_name")),
                )
            )

    def _resolve_storage_path(self, raw_path: str) -> Path:
        candidate = Path(str(raw_path))
        if candidate.is_absolute():
            return candidate

        config_path = Path(str(self.config.get("__config_path__", ""))).resolve() if self.config.get("__config_path__") else None
        if config_path is not None and str(config_path):
            return (config_path.parent / candidate).resolve()
        return candidate.resolve()

    def _ensure_enabled(self, *, operation: str) -> None:
        if not self.enabled:
            raise ObservabilityMemoryError(
                operation=operation,
                details="observability memory is disabled by configuration",
            )

    def _handle_exception(
        self,
        exc: Exception,
        *,
        stage: str,
        operation: str,
        context: Optional[Mapping[str, Any]] = None,
    ) -> ObservabilityError:
        error = normalize_observability_exception(exc, stage=stage, context={"operation": operation, **self._safe_mapping(context)})
        error.report()
        logger.error("Observability memory operation '%s' failed: %s | context=%s", operation, error, error.to_dict())
        return error

    def _flush_snapshot_if_configured(self) -> None:
        if self.enable_snapshot_persistence and self.flush_snapshot_on_write:
            self.persist_snapshot()

    def _evict_ordered_dict(self, dictionary: MutableMapping[str, Any], limit: int) -> None:
        while len(dictionary) > limit:
            oldest_key = next(iter(dictionary))
            dictionary.pop(oldest_key, None)

    def _evict_trace_timeline_orphans(self) -> None:
        valid_trace_ids = set(self._trace_archives.keys())
        for trace_id in list(self._trace_timelines.keys()):
            if trace_id not in valid_trace_ids:
                self._trace_timelines.pop(trace_id, None)

    def _rebuild_signature_index_if_needed(self) -> None:
        valid_incident_ids = set(self._incident_fingerprints.keys())
        for signature_hash, incident_id in list(self._signature_index.items()):
            if incident_id not in valid_incident_ids:
                self._signature_index.pop(signature_hash, None)

    def _purge_expired_suppressions_locked(self, *, now_ms: Optional[float] = None) -> None:
        ts_ms = float(now_ms if now_ms is not None else self._now_ms())
        for alert_key, record in list(self._alert_suppressions.items()):
            if record.expires_at_ms <= ts_ms:
                self._alert_suppressions.pop(alert_key, None)

    def _evaluate_objective_status(self, observed: float, target: float, comparator: str) -> str:
        if comparator == "lte":
            return "met" if observed <= target else "breached"
        return "met" if observed >= target else "breached"

    def _parse_window_seconds(self, window: Any) -> float:
        if window is None:
            return self.default_latency_window_seconds
        if isinstance(window, (int, float)):
            if float(window) <= 0:
                raise ObservabilityMemoryError(
                    operation="latency_trend",
                    details="window must be positive",
                    context={"window": window},
                )
            return float(window)
        if not isinstance(window, str):
            raise ObservabilityMemoryError(
                operation="latency_trend",
                details="window must be a number of seconds or a duration string",
                context={"window_type": type(window).__name__},
            )

        raw = window.strip().lower()
        if raw.isdigit():
            return float(raw)
        unit_map = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
        unit = raw[-1]
        magnitude = raw[:-1]
        if unit not in unit_map or not magnitude:
            raise ObservabilityMemoryError(
                operation="latency_trend",
                details=f"unsupported window format '{window}'",
                context={"expected_examples": ["3600", "1h", "24h", "7d"]},
            )
        seconds = float(magnitude) * unit_map[unit]
        if seconds <= 0:
            raise ObservabilityMemoryError(
                operation="latency_trend",
                details="window must be positive",
                context={"window": window},
            )
        return seconds

    def _percentile(self, values: Sequence[float], percentile: int) -> float:
        if not values:
            return 0.0
        ordered = sorted(float(value) for value in values)
        if len(ordered) == 1:
            return ordered[0]
        rank = (len(ordered) - 1) * (percentile / 100.0)
        lower = math.floor(rank)
        upper = math.ceil(rank)
        if lower == upper:
            return ordered[lower]
        weight = rank - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * weight

    def _normalize_signature(self, value: Optional[str]) -> str:
        raw = str(value or "").strip().lower()
        return " ".join(raw.split())

    def _signature_hash(self, signature: str) -> str:
        import hashlib

        return hashlib.sha256(signature.encode("utf-8")).hexdigest()

    def _signature_tokens(self, signature: str) -> set[str]:
        return {token for token in self._normalize_signature(signature).replace("|", " ").replace(":", " ").split(" ") if token}

    def _jaccard_similarity(self, left: set[str], right: set[str]) -> float:
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0
        intersection = len(left.intersection(right))
        union = len(left.union(right))
        return intersection / union if union else 0.0

    def _safe_mapping(self, value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if value is None:
            return {}
        return {str(key): self._safe_scalar(item) for key, item in dict(value).items()}

    def _safe_numeric_mapping(self, value: Optional[Mapping[str, Any]]) -> Dict[str, float]:
        if value is None:
            return {}
        return {str(key): float(item) for key, item in dict(value).items()}

    def _safe_scalar(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, Mapping):
            return {str(key): self._safe_scalar(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._safe_scalar(item) for item in value]
        return str(value)

    def _dedupe_strings(self, values: Optional[Iterable[Any]], limit: int) -> List[str]:
        ordered: List[str] = []
        if values is None:
            return ordered
        for value in values:
            string_value = str(value).strip()
            if string_value and string_value not in ordered:
                ordered.append(string_value)
            if len(ordered) >= limit:
                break
        return ordered

    def _merge_limited_strings(
        self,
        existing: Sequence[str],
        incoming: Optional[Iterable[Any]],
        *,
        limit: int,
    ) -> List[str]:
        merged = list(existing)
        for value in incoming or []:
            string_value = str(value).strip()
            if not string_value or string_value in merged:
                continue
            merged.append(string_value)
            if len(merged) >= limit:
                break
        return merged

    def _require_non_empty_str(self, value: Any, *, field_name: str, operation: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ObservabilityMemoryError(
                operation=operation,
                details=f"{field_name} cannot be empty",
                context={"field_name": field_name},
            )
        return text

    def _optional_str(self, value: Any) -> Optional[str]:
        text = str(value).strip() if value is not None else ""
        return text or None

    def _coerce_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _now_ms(self) -> float:
        return time.time() * 1000.0


if __name__ == "__main__":
    print("\n=== Running Observability Memory ===\n")
    printer.status("TEST", "Observability Memory initialized", "info")

    memory = ObservabilityMemory()
    now_ms = time.time() * 1000.0
    trace_id = "trace-demo-001"

    spans = [
        {
            "trace_id": trace_id,
            "span_id": "span-1",
            "agent_name": "planner_agent",
            "operation_name": "plan_request",
            "start_ms": now_ms - 4200,
            "end_ms": now_ms - 3600,
            "status": "ok",
            "metadata": {"queue": "planner"},
        },
        {
            "trace_id": trace_id,
            "span_id": "span-2",
            "agent_name": "observability_agent",
            "operation_name": "analyze_waterfall",
            "start_ms": now_ms - 3500,
            "end_ms": now_ms - 1800,
            "status": "retry",
            "attempt": 2,
            "metadata": {"retry": True, "reason": "slow_downstream_dependency"},
        },
        {
            "trace_id": trace_id,
            "span_id": "span-3",
            "agent_name": "handler_agent",
            "operation_name": "fallback_route",
            "start_ms": now_ms - 1700,
            "end_ms": now_ms - 900,
            "status": "timeout",
            "metadata": {"policy": "degraded_mode"},
        },
    ]

    timeline = [
        {
            "event_type": "task_received",
            "timestamp_ms": now_ms - 4300,
            "agent_name": "planner_agent",
            "severity": "info",
            "message": "Incoming request accepted for orchestration.",
        },
        {
            "event_type": "retry_detected",
            "timestamp_ms": now_ms - 2100,
            "agent_name": "observability_agent",
            "severity": "warning",
            "message": "Retry waterfall detected during latency attribution.",
            "payload": {"attempt": 2},
        },
    ]

    archive = memory.archive_trace(
        trace_id,
        spans,
        timeline=timeline,
        summary={
            "critical_path_ms": 2500.0,
            "bottleneck_count": 1,
            "anomaly_count": 1,
            "per_agent_duration_ms": {
                "planner_agent": 600.0,
                "observability_agent": 1700.0,
                "handler_agent": 800.0,
            },
        },
        incident_level="warning",
        error_signature="handler_agent:fallback_route:timeout | observability_agent:analyze_waterfall:retry",
        metadata={"task_id": "task-42", "tenant": "internal"},
    )
    printer.status("ARCHIVE", f"Archived trace {trace_id}", "success")
    printer.pretty("TRACE", archive, "info")

    event = memory.append_timeline_event(
        trace_id,
        event_type="incident_promoted",
        severity="warning",
        agent_name="observability_agent",
        message="Incident level promoted after timeout confirmation.",
        payload={"incident_level": "warning"},
        correlation_keys={"incident_id": "incident-001"},
    )
    printer.pretty("TIMELINE", event, "info")

    incident = memory.record_incident_fingerprint(
        incident_id="incident-001",
        error_signature="handler_agent:fallback_route:timeout | observability_agent:analyze_waterfall:retry",
        severity="warning",
        trace_id=trace_id,
        related_agents=["handler_agent", "observability_agent"],
        root_cause_candidates=["downstream_timeout", "retry_amplification"],
        recommended_actions=["enable degraded mode", "inspect downstream queue backlog"],
        metadata={"cluster": "timeout.retry"},
    )
    printer.pretty("INCIDENT", incident, "info")

    slo_record = memory.record_slo_history(
        service="slai",
        slo_name="response_latency_p95_ms",
        observed=2450.0,
        target=2000.0,
        comparator="lte",
        window_seconds=300,
        metadata={"trace_id": trace_id},
    )
    printer.pretty("SLO", slo_record, "info")

    suppression = memory.record_alert_suppression(
        alert_key="slai.latency.p95.warning",
        reason="Known incident already routed to handler policy",
        source="observability_agent",
        ttl_seconds=1800,
        incident_id="incident-001",
        metadata={"policy": "contextual_suppression"},
    )
    printer.pretty("SUPPRESS", suppression, "info")

    remediation = memory.record_remediation_action(
        playbook_id="pb-degraded-mode",
        incident_id="incident-001",
        action_name="enable_degraded_mode",
        status="success",
        started_at_ms=now_ms - 800,
        completed_at_ms=now_ms - 200,
        recovery_metrics={"latency_reduction_ms": 900.0, "queue_delta": -14.0},
        notes="Fallback path reduced end-to-end latency.",
        metadata={"executor": "handler_agent"},
    )
    printer.pretty("RUNBOOK", remediation, "info")

    similarities = memory.incident_similarities(
        "handler_agent:fallback_route:timeout | observability_agent:analyze_waterfall:retry"
    )
    printer.pretty("SIMILAR", similarities, "info")

    latency = memory.latency_trend("observability_agent", 95, "24h")
    printer.pretty("LATENCY", latency, "info")

    runbook = memory.runbook_outcome("pb-degraded-mode")
    printer.pretty("OUTCOME", runbook, "info")

    shared_context = memory.build_shared_memory_context(trace_id=trace_id, incident_id="incident-001")
    printer.pretty("SHARED", shared_context, "info")

    suppression_status = memory.is_alert_suppressed("slai.latency.p95.warning")
    printer.pretty("SUPPRESSION_STATUS", suppression_status, "info")

    snapshot_info = memory.persist_snapshot() if memory.enable_snapshot_persistence else {"status": "skipped"}
    printer.pretty("SNAPSHOT", snapshot_info, "info")

    print("\n=== Test ran successfully ===\n")
