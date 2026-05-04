"""
- End-to-end task trace IDs.
- Per-agent span timing.
- Critical path reconstruction.
"""

from __future__ import annotations

import time
import uuid

from collections import OrderedDict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from .utils import (load_global_config, get_config_section,
                    WaterfallAnalyzer,
                    # Error handling
                    ObservabilityError, ObservabilityErrorType, TraceContextMissingError,
                    ObservabilitySeverity, TelemetryContractError, TraceCollectionError,
                    normalize_observability_exception)
from .observability_memory import ObservabilityMemory
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Observability Tracing")
printer = PrettyPrinter

_STATUS_ALIASES: Dict[str, str] = {
    "ok": "ok",
    "success": "ok",
    "succeeded": "ok",
    "completed": "ok",
    "complete": "ok",
    "done": "ok",
    "running": "running",
    "in_progress": "running",
    "queued": "queued",
    "pending": "queued",
    "retry": "retry",
    "retried": "retry",
    "retrying": "retry",
    "error": "error",
    "failed": "error",
    "failure": "error",
    "exception": "error",
    "timeout": "timeout",
    "timed_out": "timeout",
    "deadline_exceeded": "timeout",
    "cancelled": "cancelled",
    "canceled": "cancelled",
    "skipped": "skipped",
}

_TERMINAL_STATUSES = {"ok", "error", "timeout", "cancelled", "skipped"}
_DEFAULT_REQUIRED_CONTEXT_FIELDS = ("trace_id", "agent_name", "operation_name")


@dataclass
class TraceEvent:
    trace_id: str
    event_type: str
    timestamp_ms: float
    message: Optional[str] = None
    severity: str = "info"
    agent_name: Optional[str] = None
    span_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_keys: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "event_type": self.event_type,
            "timestamp_ms": self.timestamp_ms,
            "message": self.message,
            "severity": self.severity,
            "agent_name": self.agent_name,
            "span_id": self.span_id,
            "payload": dict(self.payload),
            "correlation_keys": dict(self.correlation_keys),
        }


@dataclass
class TraceSpanRecord:
    trace_id: str
    span_id: str
    agent_name: str
    operation_name: str
    start_ms: float
    end_ms: Optional[float] = None
    status: str = "running"
    parent_span_id: Optional[str] = None
    attempt: int = 1
    service: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return self.end_ms is None

    @property
    def duration_ms(self) -> float:
        end_ms = self.end_ms if self.end_ms is not None else self.start_ms
        return max(0.0, float(end_ms) - float(self.start_ms))

    def to_waterfall_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "agent_name": self.agent_name,
            "operation_name": self.operation_name,
            "start_ms": float(self.start_ms),
            "end_ms": float(self.end_ms if self.end_ms is not None else self.start_ms),
            "status": self.status,
            "parent_span_id": self.parent_span_id,
            "attempt": int(self.attempt),
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> Dict[str, Any]:
        payload = self.to_waterfall_dict()
        payload["is_open"] = self.is_open
        payload["duration_ms"] = self.duration_ms
        payload["service"] = self.service
        return payload


@dataclass
class TraceSession:
    trace_id: str
    task_name: str
    root_agent_name: str
    root_operation_name: str
    created_at_ms: float
    updated_at_ms: float
    service: str
    incident_level: str = "info"
    status: str = "running"
    metadata: Dict[str, Any] = field(default_factory=dict)
    root_span_id: Optional[str] = None
    completed_at_ms: Optional[float] = None
    archived: bool = False
    analysis_summary: Optional[Dict[str, Any]] = None
    shared_memory_snapshot: Dict[str, Any] = field(default_factory=dict)
    spans: "OrderedDict[str, TraceSpanRecord]" = field(default_factory=OrderedDict)
    events: List[TraceEvent] = field(default_factory=list)

    @property
    def span_count(self) -> int:
        return len(self.spans)

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def wall_clock_end_ms(self) -> float:
        if self.completed_at_ms is not None:
            return float(self.completed_at_ms)
        return max((span.end_ms if span.end_ms is not None else span.start_ms) for span in self.spans.values()) if self.spans else self.updated_at_ms

    @property
    def total_duration_ms(self) -> float:
        return max(0.0, self.wall_clock_end_ms - self.created_at_ms)

    def to_dict(
        self,
        *,
        include_spans: bool = True,
        include_events: bool = True,
    ) -> Dict[str, Any]:
        payload = {
            "trace_id": self.trace_id,
            "task_name": self.task_name,
            "root_agent_name": self.root_agent_name,
            "root_operation_name": self.root_operation_name,
            "created_at_ms": self.created_at_ms,
            "updated_at_ms": self.updated_at_ms,
            "completed_at_ms": self.completed_at_ms,
            "service": self.service,
            "incident_level": self.incident_level,
            "status": self.status,
            "metadata": dict(self.metadata),
            "root_span_id": self.root_span_id,
            "archived": self.archived,
            "span_count": self.span_count,
            "event_count": self.event_count,
            "total_duration_ms": self.total_duration_ms,
            "analysis_summary": dict(self.analysis_summary) if isinstance(self.analysis_summary, dict) else self.analysis_summary,
            "shared_memory_snapshot": dict(self.shared_memory_snapshot),
        }
        if include_spans:
            payload["spans"] = [span.to_dict() for span in self.spans.values()]
        if include_events:
            payload["events"] = [event.to_dict() for event in self.events]
        return payload


class ObservabilityTracing:
    def __init__(self, *,
        memory: Optional[ObservabilityMemory] = None,
        analyzer: Optional[WaterfallAnalyzer] = None,
    ) -> None:
        self.config = load_global_config()
        self.tracing_config = get_config_section("observability_tracing")
        self._lock = RLock()

        self.enabled = bool(self.tracing_config.get("enabled", True))
        self.enable_memory_integration = bool(self.tracing_config.get("enable_memory_integration", False))
        self.auto_archive_on_finalize = bool(self.tracing_config.get("auto_archive_on_finalize", True))
        self.auto_analyze_on_finalize = bool(self.tracing_config.get("auto_analyze_on_finalize", True))
        self.preserve_completed_traces = bool(self.tracing_config.get("preserve_completed_traces", True))
        self.finalize_open_spans_on_close = bool(self.tracing_config.get("finalize_open_spans_on_close", True))
        self.append_timeline_events = bool(self.tracing_config.get("append_timeline_events", True))
        self.record_lifecycle_events = bool(self.tracing_config.get("record_lifecycle_events", True))
        self.record_contract_failures_as_events = bool(
            self.tracing_config.get("record_contract_failures_as_events", True)
        )
        self.enforce_contract_validation = bool(self.tracing_config.get("enforce_contract_validation", True))
        self.allow_orphan_spans = bool(self.tracing_config.get("allow_orphan_spans", False))
        self.allow_external_trace_creation = bool(self.tracing_config.get("allow_external_trace_creation", False))
        self.archive_trace_reports = bool(self.tracing_config.get("archive_trace_reports", True))
        self.include_shared_memory_snapshot = bool(self.tracing_config.get("include_shared_memory_snapshot", True))
        self.analyze_on_reconstruct = bool(self.tracing_config.get("analyze_on_reconstruct", True))

        self.max_active_traces = int(self.tracing_config.get("max_active_traces", 1000))
        self.max_completed_traces = int(self.tracing_config.get("max_completed_traces", 2000))
        self.max_spans_per_trace = int(self.tracing_config.get("max_spans_per_trace", 5000))
        self.max_events_per_trace = int(self.tracing_config.get("max_events_per_trace", 1000))
        self.default_span_timeout_ms = float(self.tracing_config.get("default_span_timeout_ms", 30000.0))
        self.max_metadata_keys = int(self.tracing_config.get("max_metadata_keys", 64))
        self.max_metadata_value_length = int(self.tracing_config.get("max_metadata_value_length", 1024))
        self.max_payload_keys = int(self.tracing_config.get("max_payload_keys", 64))
        self.max_payload_value_length = int(self.tracing_config.get("max_payload_value_length", 2048))

        self.trace_id_prefix = str(self.tracing_config.get("trace_id_prefix", "trace"))
        self.span_id_prefix = str(self.tracing_config.get("span_id_prefix", "span"))
        self.default_service = str(self.tracing_config.get("default_service", "slai"))
        self.default_incident_level = str(self.tracing_config.get("default_incident_level", "info")).lower()
        self.default_open_span_close_status = self._normalize_status(
            self.tracing_config.get("default_open_span_close_status", "cancelled")
        )

        configured_required_fields = self.tracing_config.get(
            "required_trace_context_fields",
            list(_DEFAULT_REQUIRED_CONTEXT_FIELDS),
        )
        self.required_trace_context_fields = tuple(str(field) for field in configured_required_fields)

        self.memory = memory if memory is not None else (
            ObservabilityMemory() if self.enable_memory_integration else None
        )
        self.analyzer = analyzer or WaterfallAnalyzer()

        self._active_traces: "OrderedDict[str, TraceSession]" = OrderedDict()
        self._completed_traces: "OrderedDict[str, TraceSession]" = OrderedDict()

    # ------------------------------------------------------------------
    # Public trace lifecycle
    # ------------------------------------------------------------------
    def start_trace(self, *, task_name: str, agent_name: str, operation_name: str,
        service: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        trace_id: Optional[str] = None,
        incident_level: Optional[str] = None,
        start_root_span: bool = True,
        root_span_metadata: Optional[Mapping[str, Any]] = None,
        root_span_status: str = "running") -> Dict[str, Any]:
        try:
            self._ensure_enabled(operation="start_trace")
            task_name = self._require_non_empty_str(task_name, field_name="task_name", operation="start_trace")
            agent_name = self._require_non_empty_str(agent_name, field_name="agent_name", operation="start_trace")
            operation_name = self._require_non_empty_str(
                operation_name,
                field_name="operation_name",
                operation="start_trace",
            )

            with self._lock:
                self._enforce_trace_capacity_locked()
                resolved_trace_id = self._ensure_unique_trace_id_locked(trace_id)
                now_ms = self._now_ms()
                session = TraceSession(
                    trace_id=resolved_trace_id,
                    task_name=task_name,
                    root_agent_name=agent_name,
                    root_operation_name=operation_name,
                    created_at_ms=now_ms,
                    updated_at_ms=now_ms,
                    service=self._optional_str(service) or self.default_service,
                    incident_level=str(incident_level or self.default_incident_level).lower(),
                    metadata=self._sanitize_mapping(metadata, max_items=self.max_metadata_keys),
                )
                self._active_traces[resolved_trace_id] = session
                self._active_traces.move_to_end(resolved_trace_id)

                if self.record_lifecycle_events:
                    self._append_event_locked(
                        session,
                        event_type="trace_started",
                        message=f"Trace '{resolved_trace_id}' started for task '{task_name}'.",
                        severity="info",
                        agent_name=agent_name,
                        payload={
                            "task_name": task_name,
                            "operation_name": operation_name,
                            "service": session.service,
                        },
                    )

                if start_root_span:
                    root_span = self._create_span_locked(
                        session,
                        agent_name=agent_name,
                        operation_name=operation_name,
                        parent_span_id=None,
                        metadata=root_span_metadata,
                        status=root_span_status,
                        service=session.service,
                    )
                    session.root_span_id = root_span.span_id
                    if self.record_lifecycle_events:
                        self._append_event_locked(
                            session,
                            event_type="root_span_started",
                            message=f"Root span '{root_span.span_id}' opened for '{agent_name}'.",
                            severity="info",
                            agent_name=agent_name,
                            span_id=root_span.span_id,
                            payload={"operation_name": operation_name},
                        )

                logger.info("Started trace '%s' for task '%s'.", resolved_trace_id, task_name)
                return session.to_dict()
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="tracing.collect",
                operation="start_trace",
                context={"task_name": task_name, "agent_name": agent_name},
            ) from exc

    def start_span(self, *, trace_id: str, agent_name: str, operation_name: str,
        parent_span_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        status: str = "running",
        attempt: int = 1,
        service: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            self._ensure_enabled(operation="start_span")
            trace_id = self._require_non_empty_str(trace_id, field_name="trace_id", operation="start_span")
            agent_name = self._require_non_empty_str(agent_name, field_name="agent_name", operation="start_span")
            operation_name = self._require_non_empty_str(
                operation_name,
                field_name="operation_name",
                operation="start_span",
            )

            with self._lock:
                session = self._require_trace_locked(trace_id, operation="start_span")
                span = self._create_span_locked(
                    session,
                    agent_name=agent_name,
                    operation_name=operation_name,
                    parent_span_id=parent_span_id,
                    metadata=metadata,
                    status=status,
                    attempt=attempt,
                    service=service,
                )
                if self.record_lifecycle_events:
                    self._append_event_locked(
                        session,
                        event_type="span_started",
                        message=f"Span '{span.span_id}' started for agent '{agent_name}'.",
                        severity="info",
                        agent_name=agent_name,
                        span_id=span.span_id,
                        payload={
                            "parent_span_id": parent_span_id,
                            "operation_name": operation_name,
                            "attempt": span.attempt,
                        },
                    )
                return span.to_dict()
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="tracing.collect",
                operation="start_span",
                context={"trace_id": trace_id, "agent_name": agent_name},
            ) from exc

    def ingest_span(self, span_payload: Mapping[str, Any], *,
                    allow_create_trace: Optional[bool] = None) -> Dict[str, Any]:
        try:
            self._ensure_enabled(operation="ingest_span")
            payload = dict(span_payload or {})
            self.validate_trace_context(payload, agent_name=str(payload.get("agent_name") or "unknown_agent"))

            trace_id = self._require_non_empty_str(payload.get("trace_id"), field_name="trace_id", operation="ingest_span")
            agent_name = self._require_non_empty_str(payload.get("agent_name"), field_name="agent_name", operation="ingest_span")
            operation_name = self._require_non_empty_str(
                payload.get("operation_name"),
                field_name="operation_name",
                operation="ingest_span",
            )
            span_id = self._optional_str(payload.get("span_id"))
            parent_span_id = self._optional_str(payload.get("parent_span_id"))
            start_ms = self._coerce_float(payload.get("start_ms"), default=self._now_ms())
            end_ms = payload.get("end_ms")
            end_ms_value = self._coerce_float(end_ms, default=start_ms) if end_ms is not None else None
            status = self._normalize_status(payload.get("status", "running"))
            attempt = max(1, int(self._coerce_float(payload.get("attempt"), default=1.0)))
            metadata = self._sanitize_mapping(payload.get("metadata"), max_items=self.max_metadata_keys)
            service = self._optional_str(payload.get("service"))

            create_trace = self.allow_external_trace_creation if allow_create_trace is None else bool(allow_create_trace)

            with self._lock:
                session = self._active_traces.get(trace_id)
                if session is None:
                    if not create_trace:
                        raise TraceCollectionError(
                            source="ingest_span",
                            details=f"trace '{trace_id}' is unknown and external creation is disabled",
                            context={"trace_id": trace_id, "agent_name": agent_name},
                        )
                    session = TraceSession(
                        trace_id=trace_id,
                        task_name=str(payload.get("task_name") or operation_name),
                        root_agent_name=agent_name,
                        root_operation_name=operation_name,
                        created_at_ms=start_ms,
                        updated_at_ms=start_ms,
                        service=service or self.default_service,
                        incident_level=self.default_incident_level,
                        metadata=self._sanitize_mapping(payload.get("trace_metadata"), max_items=self.max_metadata_keys),
                    )
                    self._enforce_trace_capacity_locked()
                    self._active_traces[trace_id] = session

                if span_id and span_id in session.spans:
                    span = session.spans[span_id]
                    span.agent_name = agent_name
                    span.operation_name = operation_name
                    span.parent_span_id = parent_span_id
                    span.start_ms = min(span.start_ms, start_ms)
                    span.end_ms = end_ms_value if end_ms_value is not None else span.end_ms
                    span.status = status
                    span.attempt = attempt
                    span.service = service or span.service
                    span.metadata.update(metadata)
                else:
                    if len(session.spans) >= self.max_spans_per_trace:
                        raise TraceCollectionError(
                            source="ingest_span",
                            details=f"trace '{trace_id}' exceeded max_spans_per_trace={self.max_spans_per_trace}",
                            context={"trace_id": trace_id, "span_count": len(session.spans)},
                        )
                    resolved_span_id = span_id or self._new_span_id()
                    if parent_span_id and parent_span_id not in session.spans and not self.allow_orphan_spans:
                        raise TraceCollectionError(
                            source="ingest_span",
                            details=f"parent span '{parent_span_id}' is not registered for trace '{trace_id}'",
                            context={"trace_id": trace_id, "parent_span_id": parent_span_id},
                        )
                    span = TraceSpanRecord(
                        trace_id=trace_id,
                        span_id=resolved_span_id,
                        agent_name=agent_name,
                        operation_name=operation_name,
                        start_ms=start_ms,
                        end_ms=end_ms_value,
                        status=status,
                        parent_span_id=parent_span_id,
                        attempt=attempt,
                        service=service or session.service,
                        metadata=metadata,
                    )
                    session.spans[span.span_id] = span
                    if session.root_span_id is None:
                        session.root_span_id = span.span_id

                session.updated_at_ms = max(session.updated_at_ms, end_ms_value or start_ms)
                if self.append_timeline_events:
                    self._append_event_locked(
                        session,
                        event_type="span_ingested",
                        message=f"Span '{span.span_id}' ingested from external telemetry.",
                        severity="info",
                        agent_name=agent_name,
                        span_id=span.span_id,
                        payload={"status": status, "attempt": attempt},
                    )
                return span.to_dict()
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="tracing.collect",
                operation="ingest_span",
                context={"span_payload": dict(span_payload or {})},
            ) from exc

    def end_span(self, trace_id: str, span_id: str, *, status: str = "ok",
        metadata: Optional[Mapping[str, Any]] = None,
        end_ms: Optional[float] = None) -> Dict[str, Any]:
        try:
            self._ensure_enabled(operation="end_span")
            trace_id = self._require_non_empty_str(trace_id, field_name="trace_id", operation="end_span")
            span_id = self._require_non_empty_str(span_id, field_name="span_id", operation="end_span")
            resolved_status = self._normalize_status(status)
            resolved_end_ms = self._coerce_float(end_ms, default=self._now_ms())

            with self._lock:
                session = self._require_trace_locked(trace_id, operation="end_span")
                span = self._require_span_locked(session, span_id, operation="end_span")
                span.status = resolved_status
                span.end_ms = max(float(span.start_ms), resolved_end_ms)
                if metadata:
                    span.metadata.update(self._sanitize_mapping(metadata, max_items=self.max_metadata_keys))
                session.updated_at_ms = max(session.updated_at_ms, span.end_ms)
                if self.record_lifecycle_events:
                    self._append_event_locked(
                        session,
                        event_type="span_completed",
                        message=f"Span '{span_id}' completed with status '{resolved_status}'.",
                        severity=self._severity_for_status(resolved_status),
                        agent_name=span.agent_name,
                        span_id=span_id,
                        payload={"duration_ms": span.duration_ms, "operation_name": span.operation_name},
                    )
                return span.to_dict()
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="tracing.collect",
                operation="end_span",
                context={"trace_id": trace_id, "span_id": span_id},
            ) from exc

    def fail_span(self, trace_id: str, span_id: str, *, exc: Exception, status: str = "error",
                  metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        try:
            error = normalize_observability_exception(
                exc,
                stage="tracing.collect",
                context={"trace_id": trace_id, "span_id": span_id},
            )
            error.report()
            merged_metadata = dict(metadata or {})
            merged_metadata["error"] = error.to_dict()
            return self.end_span(trace_id, span_id, status=status, metadata=merged_metadata)
        except Exception as nested_exc:
            raise self._handle_exception(
                nested_exc,
                stage="tracing.collect",
                operation="fail_span",
                context={"trace_id": trace_id, "span_id": span_id},
            ) from nested_exc

    def append_event(self, trace_id: str, *, event_type: str,
        message: Optional[str] = None,
        severity: str = "info",
        agent_name: Optional[str] = None,
        span_id: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
        correlation_keys: Optional[Mapping[str, Any]] = None,
        timestamp_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        try:
            self._ensure_enabled(operation="append_event")
            trace_id = self._require_non_empty_str(trace_id, field_name="trace_id", operation="append_event")
            event_type = self._require_non_empty_str(event_type, field_name="event_type", operation="append_event")
            with self._lock:
                session = self._require_trace_locked(trace_id, operation="append_event")
                event = self._append_event_locked(
                    session,
                    event_type=event_type,
                    message=message,
                    severity=severity,
                    agent_name=agent_name,
                    span_id=span_id,
                    payload=payload,
                    correlation_keys=correlation_keys,
                    timestamp_ms=timestamp_ms,
                )
                return event.to_dict()
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="tracing.collect",
                operation="append_event",
                context={"trace_id": trace_id, "event_type": event_type},
            ) from exc

    def build_trace_context(self, *, trace_id: str, agent_name: str, operation_name: str,
                            span_id: Optional[str] = None, parent_span_id: Optional[str] = None,
                            service: Optional[str] = None, metadata: Optional[Mapping[str, Any]] = None,
                            ) -> Dict[str, Any]:
        context = {
            "trace_id": self._require_non_empty_str(trace_id, field_name="trace_id", operation="build_trace_context"),
            "agent_name": self._require_non_empty_str(agent_name, field_name="agent_name", operation="build_trace_context"),
            "operation_name": self._require_non_empty_str(
                operation_name,
                field_name="operation_name",
                operation="build_trace_context",
            ),
            "span_id": self._optional_str(span_id),
            "parent_span_id": self._optional_str(parent_span_id),
            "service": self._optional_str(service) or self.default_service,
            "metadata": self._sanitize_mapping(metadata, max_items=self.max_metadata_keys),
        }
        self.validate_trace_context(context, agent_name=agent_name)
        return context

    def derive_child_context(self, parent_context: Mapping[str, Any], *, agent_name: str,
                             operation_name: str, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        base = dict(parent_context or {})
        self.validate_trace_context(base, agent_name=str(base.get("agent_name") or "unknown_agent"))
        return self.build_trace_context(
            trace_id=str(base["trace_id"]),
            agent_name=agent_name,
            operation_name=operation_name,
            parent_span_id=self._optional_str(base.get("span_id")) or self._optional_str(base.get("parent_span_id")),
            service=self._optional_str(base.get("service")),
            metadata={**self._coerce_mapping(base.get("metadata")), **self._coerce_mapping(metadata)},
        )

    def validate_trace_context(
        self,
        context: Mapping[str, Any],
        *,
        agent_name: str,
        required_fields: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        try:
            required = tuple(str(field) for field in (required_fields or self.required_trace_context_fields))
            missing_fields = [field for field in required if not str((context or {}).get(field, "")).strip()]
            if missing_fields:
                exc = TraceContextMissingError(
                    agent_name=agent_name,
                    missing_fields=missing_fields,
                    context={"required_fields": list(required), "provided_keys": sorted((context or {}).keys())},
                )
                if self.record_contract_failures_as_events:
                    trace_id = self._optional_str((context or {}).get("trace_id"))
                    if trace_id:
                        try:
                            self.append_event(
                                trace_id,
                                event_type="trace_context_missing",
                                message=str(exc),
                                severity="warning",
                                agent_name=agent_name,
                                payload={"missing_fields": missing_fields},
                            )
                        except Exception:
                            pass
                raise exc

            if self.enforce_contract_validation:
                normalized_context = {str(k): v for k, v in dict(context or {}).items()}
                if "metadata" in normalized_context and not isinstance(normalized_context["metadata"], Mapping):
                    raise TelemetryContractError(
                        agent_name=agent_name,
                        missing_fields=[],
                        context={
                            "contract_violation": "metadata must be a mapping",
                            "offending_type": type(normalized_context["metadata"]).__name__,
                        },
                    )
                return normalized_context
            return dict(context or {})
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="tracing.context",
                operation="validate_trace_context",
                context={"agent_name": agent_name},
            ) from exc

    def reconstruct_critical_path(self, trace_id: str) -> Dict[str, Any]:
        try:
            session = self._get_trace_session(trace_id)
            if session is None:
                raise TraceCollectionError(
                    source="reconstruct_critical_path",
                    details=f"trace '{trace_id}' was not found",
                    context={"trace_id": trace_id},
                )
            if not session.spans:
                raise TraceCollectionError(
                    source="reconstruct_critical_path",
                    details=f"trace '{trace_id}' has no spans",
                    context={"trace_id": trace_id},
                )

            report = self.analyzer.analyze([span.to_waterfall_dict() for span in session.spans.values()])
            session.analysis_summary = report.to_dict()
            session.updated_at_ms = max(session.updated_at_ms, self._now_ms())
            return {
                "trace_id": trace_id,
                "critical_path_ms": report.critical_path_ms,
                "critical_path_span_ids": report.critical_path_span_ids,
                "critical_path_agent_names": report.critical_path_agent_names,
                "critical_path_exclusive_ms": report.critical_path_exclusive_ms,
                "total_duration_ms": report.total_duration_ms,
                "span_count": report.span_count,
                "anomaly_count": len(report.anomalies),
                "bottleneck_count": len(report.bottleneck_spans),
            }
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="tracing.integrity",
                operation="reconstruct_critical_path",
                context={"trace_id": trace_id},
            ) from exc

    def finalize_trace(
        self,
        trace_id: str,
        *,
        incident_level: Optional[str] = None,
        status: str = "completed",
        metadata: Optional[Mapping[str, Any]] = None,
        archive: Optional[bool] = None,
        analyze: Optional[bool] = None,
    ) -> Dict[str, Any]:
        try:
            self._ensure_enabled(operation="finalize_trace")
            trace_id = self._require_non_empty_str(trace_id, field_name="trace_id", operation="finalize_trace")
            do_archive = self.auto_archive_on_finalize if archive is None else bool(archive)
            do_analyze = self.auto_analyze_on_finalize if analyze is None else bool(analyze)

            with self._lock:
                session = self._require_trace_locked(trace_id, operation="finalize_trace")
                if metadata:
                    session.metadata.update(self._sanitize_mapping(metadata, max_items=self.max_metadata_keys))
                session.incident_level = str(incident_level or session.incident_level or self.default_incident_level).lower()
                open_spans = [span for span in session.spans.values() if span.is_open]
                if open_spans:
                    if not self.finalize_open_spans_on_close:
                        raise ObservabilityError(
                            message=f"Trace '{trace_id}' cannot be finalized while {len(open_spans)} spans are still open.",
                            error_type=ObservabilityErrorType.TRACE_CORRUPTION_DETECTED,
                            severity=ObservabilitySeverity.HIGH,
                            retryable=False,
                            component="observability_tracing",
                            operation="finalize_trace",
                            trace_id=trace_id,
                            context={
                                "trace_id": trace_id,
                                "open_span_ids": [span.span_id for span in open_spans],
                            },
                            remediation="Close or cancel all open spans before finalizing the trace.",
                        )
                    now_ms = self._now_ms()
                    for span in open_spans:
                        span.end_ms = max(span.start_ms, now_ms)
                        span.status = self.default_open_span_close_status
                    if self.record_lifecycle_events:
                        self._append_event_locked(
                            session,
                            event_type="open_spans_closed",
                            message=f"{len(open_spans)} open span(s) were force-closed during finalization.",
                            severity="warning",
                            payload={"open_span_ids": [span.span_id for span in open_spans]},
                        )

                session.status = str(status or "completed").lower()
                session.completed_at_ms = self._now_ms()
                session.updated_at_ms = session.completed_at_ms

                if do_analyze and session.spans:
                    report = self.analyzer.analyze([span.to_waterfall_dict() for span in session.spans.values()])
                    session.analysis_summary = report.to_dict()
                    session.shared_memory_snapshot = self._build_shared_memory_snapshot(session, report.to_dict())
                elif self.include_shared_memory_snapshot:
                    session.shared_memory_snapshot = self._build_shared_memory_snapshot(session, None)

                if self.record_lifecycle_events:
                    self._append_event_locked(
                        session,
                        event_type="trace_finalized",
                        message=f"Trace '{trace_id}' finalized with status '{session.status}'.",
                        severity="info",
                        payload={"incident_level": session.incident_level, "status": session.status},
                    )

                if self.preserve_completed_traces:
                    self._completed_traces[trace_id] = session
                    self._completed_traces.move_to_end(trace_id)
                    self._evict_ordered_dict(self._completed_traces, self.max_completed_traces)

                del self._active_traces[trace_id]

            if do_archive:
                self._archive_trace(session)
            logger.info("Finalized trace '%s' with %s spans.", trace_id, session.span_count)
            return session.to_dict()
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="tracing.integrity",
                operation="finalize_trace",
                context={"trace_id": trace_id},
            ) from exc

    # ------------------------------------------------------------------
    # Public read / export APIs
    # ------------------------------------------------------------------
    def get_trace(self, trace_id: str, *, include_analysis: bool = True) -> Optional[Dict[str, Any]]:
        session = self._get_trace_session(trace_id)
        if session is None:
            return None
        if include_analysis and self.analyze_on_reconstruct and session.analysis_summary is None and session.spans:
            try:
                self.reconstruct_critical_path(trace_id)
            except Exception:
                pass
        return session.to_dict()

    def list_active_traces(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [session.to_dict(include_spans=False, include_events=False) for session in self._active_traces.values()]

    def list_completed_traces(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [session.to_dict(include_spans=False, include_events=False) for session in self._completed_traces.values()]

    def export_trace_spans(self, trace_id: str) -> List[Dict[str, Any]]:
        session = self._get_trace_session(trace_id)
        if session is None:
            return []
        return [span.to_waterfall_dict() for span in session.spans.values()]

    def build_shared_memory_context(self, trace_id: str) -> Dict[str, Any]:
        session = self._get_trace_session(trace_id)
        if session is None:
            raise TraceCollectionError(
                source="build_shared_memory_context",
                details=f"trace '{trace_id}' was not found",
                context={"trace_id": trace_id},
            )
        if session.analysis_summary is None and session.spans:
            try:
                self.reconstruct_critical_path(trace_id)
            except Exception:
                pass
        return self._build_shared_memory_snapshot(session, session.analysis_summary)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _archive_trace(self, session: TraceSession) -> None:
        if self.memory is None:
            return
        if not hasattr(self.memory, "archive_trace"):
            return
        try:
            timeline = [event.to_dict() for event in session.events] if self.append_timeline_events else []
            summary = dict(session.analysis_summary or {}) if self.archive_trace_reports else {}
            summary.setdefault("trace_id", session.trace_id)
            summary.setdefault("task_name", session.task_name)
            summary.setdefault("status", session.status)
            summary.setdefault("service", session.service)
            summary.setdefault("span_count", session.span_count)
            summary.setdefault("event_count", session.event_count)
            summary.setdefault("total_duration_ms", session.total_duration_ms)
            if self.include_shared_memory_snapshot:
                summary.setdefault("shared_memory_snapshot", dict(session.shared_memory_snapshot))

            archive_payload = self.memory.archive_trace(
                session.trace_id,
                [span.to_waterfall_dict() for span in session.spans.values()],
                timeline=timeline,
                summary=summary,
                incident_level=session.incident_level,
                metadata={
                    "task_name": session.task_name,
                    "service": session.service,
                    "status": session.status,
                },
            )
            session.archived = bool(archive_payload)
        except Exception as exc:
            normalized = normalize_observability_exception(
                exc,
                stage="memory.write",
                context={"trace_id": session.trace_id, "operation": "archive_trace"},
            )
            normalized.report()

    def _build_shared_memory_snapshot(
        self,
        session: TraceSession,
        report: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {
            "observability.trace_id": session.trace_id,
            "observability.agent_spans": [span.to_waterfall_dict() for span in session.spans.values()],
            "observability.incident_level": session.incident_level,
            "observability.recommended_actions": [],
        }
        if not report:
            return snapshot

        bottlenecks = list(report.get("bottleneck_spans", []))
        anomalies = list(report.get("anomalies", []))
        p95_ms = 0.0
        durations = [float(span.duration_ms) for span in session.spans.values()]
        if durations:
            sorted_values = sorted(durations)
            index = max(0, min(len(sorted_values) - 1, int(round((len(sorted_values) - 1) * 0.95))))
            p95_ms = float(sorted_values[index])

        snapshot["observability.latency_p95"] = p95_ms
        snapshot["observability.error_clusters"] = self._cluster_error_families(session.spans.values())
        if bottlenecks:
            top = bottlenecks[0]
            snapshot["observability.recommended_actions"] = [
                f"Inspect bottleneck span '{top.get('span_id')}' on agent '{top.get('agent_name')}'.",
                "Review retry amplification and dependency waits on the critical path.",
                "Check whether degraded mode or throttling is needed if latency remains elevated.",
            ]
        if anomalies and not snapshot["observability.recommended_actions"]:
            snapshot["observability.recommended_actions"] = [
                "Inspect anomalous spans and validate telemetry timing integrity.",
                "Confirm trace propagation across upstream and downstream agents.",
            ]
        return snapshot

    def _cluster_error_families(self, spans: Iterable[TraceSpanRecord]) -> List[Dict[str, Any]]:
        clusters: Dict[str, Dict[str, Any]] = {}
        for span in spans:
            if span.status not in {"error", "timeout", "retry"}:
                continue
            key = f"{span.agent_name}:{span.operation_name}:{span.status}"
            cluster = clusters.setdefault(
                key,
                {
                    "signature": key,
                    "count": 0,
                    "trace_id": span.trace_id,
                    "agent_name": span.agent_name,
                    "operation_name": span.operation_name,
                    "status": span.status,
                    "span_ids": [],
                },
            )
            cluster["count"] += 1
            cluster["span_ids"].append(span.span_id)
        return sorted(clusters.values(), key=lambda item: (item["count"], item["signature"]), reverse=True)

    def _create_span_locked(
        self,
        session: TraceSession,
        *,
        agent_name: str,
        operation_name: str,
        parent_span_id: Optional[str],
        metadata: Optional[Mapping[str, Any]],
        status: str,
        attempt: int = 1,
        service: Optional[str] = None,
    ) -> TraceSpanRecord:
        if len(session.spans) >= self.max_spans_per_trace:
            raise TraceCollectionError(
                source="start_span",
                details=f"trace '{session.trace_id}' exceeded max_spans_per_trace={self.max_spans_per_trace}",
                context={"trace_id": session.trace_id, "span_count": len(session.spans)},
            )
        if parent_span_id and parent_span_id not in session.spans and not self.allow_orphan_spans:
            raise TraceCollectionError(
                source="start_span",
                details=f"parent span '{parent_span_id}' is not registered for trace '{session.trace_id}'",
                context={"trace_id": session.trace_id, "parent_span_id": parent_span_id},
            )

        span = TraceSpanRecord(
            trace_id=session.trace_id,
            span_id=self._new_span_id(),
            agent_name=agent_name,
            operation_name=operation_name,
            start_ms=self._now_ms(),
            end_ms=None,
            status=self._normalize_status(status),
            parent_span_id=self._optional_str(parent_span_id),
            attempt=max(1, int(attempt)),
            service=self._optional_str(service) or session.service,
            metadata=self._sanitize_mapping(metadata, max_items=self.max_metadata_keys),
        )
        session.spans[span.span_id] = span
        session.updated_at_ms = max(session.updated_at_ms, span.start_ms)
        return span

    def _append_event_locked(
        self,
        session: TraceSession,
        *,
        event_type: str,
        message: Optional[str] = None,
        severity: str = "info",
        agent_name: Optional[str] = None,
        span_id: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
        correlation_keys: Optional[Mapping[str, Any]] = None,
        timestamp_ms: Optional[float] = None,
    ) -> TraceEvent:
        event = TraceEvent(
            trace_id=session.trace_id,
            event_type=self._require_non_empty_str(event_type, field_name="event_type", operation="append_event"),
            timestamp_ms=self._coerce_float(timestamp_ms, default=self._now_ms()),
            message=self._optional_str(message),
            severity=self._normalize_severity(severity),
            agent_name=self._optional_str(agent_name),
            span_id=self._optional_str(span_id),
            payload=self._sanitize_mapping(payload, max_items=self.max_payload_keys),
            correlation_keys={
                str(key): self._truncate_text(str(value), self.max_payload_value_length)
                for key, value in self._coerce_mapping(correlation_keys).items()
            },
        )
        session.events.append(event)
        if len(session.events) > self.max_events_per_trace:
            del session.events[: len(session.events) - self.max_events_per_trace]
        session.updated_at_ms = max(session.updated_at_ms, event.timestamp_ms)

        if self.memory is not None and self.append_timeline_events and hasattr(self.memory, "append_timeline_event"):
            try:
                self.memory.append_timeline_event(
                    session.trace_id,
                    event_type=event.event_type,
                    message=event.message,
                    agent_name=event.agent_name,
                    severity=event.severity,
                    timestamp_ms=event.timestamp_ms,
                    payload=event.payload,
                    correlation_keys=event.correlation_keys,
                )
            except Exception as exc:
                normalized = normalize_observability_exception(
                    exc,
                    stage="memory.write",
                    context={"trace_id": session.trace_id, "operation": "append_timeline_event"},
                )
                normalized.report()

        return event

    def _ensure_enabled(self, *, operation: str) -> None:
        if not self.enabled:
            raise TraceCollectionError(
                source=operation,
                details="observability tracing is disabled by configuration",
                context={"operation": operation},
            )

    def _require_trace_locked(self, trace_id: str, *, operation: str) -> TraceSession:
        session = self._active_traces.get(trace_id)
        if session is not None:
            return session
        completed = self._completed_traces.get(trace_id)
        if completed is not None and operation in {"get_trace", "build_shared_memory_context"}:
            return completed
        raise TraceCollectionError(
            source=operation,
            details=f"trace '{trace_id}' was not found",
            context={"trace_id": trace_id},
        )

    def _require_span_locked(self, session: TraceSession, span_id: str, *, operation: str) -> TraceSpanRecord:
        span = session.spans.get(span_id)
        if span is None:
            raise TraceCollectionError(
                source=operation,
                details=f"span '{span_id}' was not found in trace '{session.trace_id}'",
                context={"trace_id": session.trace_id, "span_id": span_id},
            )
        return span

    def _get_trace_session(self, trace_id: str) -> Optional[TraceSession]:
        trace_id = self._optional_str(trace_id)
        if not trace_id:
            return None
        with self._lock:
            return self._active_traces.get(trace_id) or self._completed_traces.get(trace_id)

    def _ensure_unique_trace_id_locked(self, trace_id: Optional[str]) -> str:
        resolved = self._optional_str(trace_id) or self._new_trace_id()
        if resolved in self._active_traces or resolved in self._completed_traces:
            raise TraceCollectionError(
                source="start_trace",
                details=f"trace_id '{resolved}' already exists",
                context={"trace_id": resolved},
            )
        return resolved

    def _enforce_trace_capacity_locked(self) -> None:
        if len(self._active_traces) >= self.max_active_traces:
            raise TraceCollectionError(
                source="start_trace",
                details=f"active trace capacity exceeded: max_active_traces={self.max_active_traces}",
                context={"active_trace_count": len(self._active_traces)},
            )

    def _evict_ordered_dict(self, ordered: MutableMapping[str, Any], limit: int) -> None:
        while len(ordered) > max(0, limit):
            ordered.pop(next(iter(ordered)))

    def _normalize_status(self, value: Any) -> str:
        text = str(value or "running").strip().lower()
        return _STATUS_ALIASES.get(text, text or "running")

    def _normalize_severity(self, value: Any) -> str:
        text = str(value or "info").strip().lower()
        if text not in {"info", "warning", "error", "critical"}:
            return "info"
        return text

    def _severity_for_status(self, status: str) -> str:
        normalized = self._normalize_status(status)
        if normalized in {"error", "timeout"}:
            return "error"
        if normalized == "retry":
            return "warning"
        return "info"

    def _now_ms(self) -> float:
        return time.time() * 1000.0

    def _new_trace_id(self) -> str:
        return f"{self.trace_id_prefix}-{uuid.uuid4().hex}"

    def _new_span_id(self) -> str:
        return f"{self.span_id_prefix}-{uuid.uuid4().hex}"

    def _coerce_mapping(self, value: Any) -> Dict[str, Any]:
        return dict(value) if isinstance(value, Mapping) else {}

    def _sanitize_mapping(self, value: Any, *, max_items: int) -> Dict[str, Any]:
        mapping = self._coerce_mapping(value)
        sanitized: Dict[str, Any] = {}
        for index, (key, item) in enumerate(mapping.items()):
            if index >= max_items:
                sanitized["__truncated__"] = True
                break
            sanitized[str(key)] = self._sanitize_value(item)
        return sanitized

    def _sanitize_value(self, value: Any, *, depth: int = 0) -> Any:
        if depth >= 3:
            return "<max_depth_exceeded>"
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return self._truncate_text(value, self.max_metadata_value_length)
        if isinstance(value, Mapping):
            return {
                str(key): self._sanitize_value(item, depth=depth + 1)
                for key, item in list(value.items())[: self.max_payload_keys]
            }
        if isinstance(value, (list, tuple, set)):
            items = list(value)
            sanitized_items = [self._sanitize_value(item, depth=depth + 1) for item in items[: self.max_payload_keys]]
            if len(items) > self.max_payload_keys:
                sanitized_items.append("<truncated>")
            return sanitized_items
        if hasattr(value, "to_dict") and callable(value.to_dict):
            try:
                return self._sanitize_value(value.to_dict(), depth=depth + 1)
            except Exception:
                return self._truncate_text(repr(value), self.max_metadata_value_length)
        return self._truncate_text(repr(value), self.max_metadata_value_length)

    def _truncate_text(self, value: str, limit: int) -> str:
        if len(value) <= limit:
            return value
        return f"{value[: max(0, limit - 3)]}..."

    def _require_non_empty_str(self, value: Any, *, field_name: str, operation: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise TraceContextMissingError(
                agent_name="observability_tracing",
                missing_fields=[field_name],
                context={"operation": operation},
            )
        return text

    def _optional_str(self, value: Any) -> Optional[str]:
        text = str(value).strip() if value is not None else ""
        return text or None

    def _coerce_float(self, value: Any, *, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _handle_exception(
        self,
        exc: Exception,
        *,
        stage: str,
        operation: str,
        context: Optional[Mapping[str, Any]] = None,
    ) -> ObservabilityError:
        normalized = normalize_observability_exception(
            exc,
            stage=stage,
            context={"component": "observability_tracing", "operation": operation, **dict(context or {})},
        )
        if "trace_id" in (context or {}) and normalized.trace_id is None:
            normalized.trace_id = str((context or {})["trace_id"])
            normalized.tags = normalized._build_tags(normalized.tags)
        normalized.report()
        logger.error("Tracing operation '%s' failed: %s", operation, normalized)
        return normalized


if __name__ == "__main__":
    print("\n=== Running Observability Tracing  ===\n")
    printer.status("TEST", "Observability Tracing  initialized", "info")

    tracing = ObservabilityTracing()

    trace = tracing.start_trace(
        task_name="slai_observability_demo",
        agent_name="planner_agent",
        operation_name="plan_request",
        metadata={"request_id": "req-001", "tenant": "demo"},
        start_root_span=True,
    )
    printer.pretty("TRACE_STARTED", trace, "info")

    trace_id = trace["trace_id"]
    root_span_id = trace.get("root_span_id")

    retrieval_span = tracing.start_span(
        trace_id=trace_id,
        agent_name="retrieval_agent",
        operation_name="retrieve_context",
        parent_span_id=root_span_id,
        metadata={"corpus": "observability_docs"},
    )
    printer.pretty("SPAN_RETRIEVAL", retrieval_span, "info")

    tracing.append_event(
        trace_id,
        event_type="queue_wait",
        message="Retrieval agent waiting on queue dispatch.",
        severity="warning",
        agent_name="retrieval_agent",
        span_id=retrieval_span["span_id"],
        payload={"queue_depth": 17, "scheduler": "default"},
    )

    time.sleep(0.002)
    tracing.end_span(
        trace_id,
        retrieval_span["span_id"],
        status="retry",
        metadata={"retry": True, "attempt": 2, "reason": "slow_dependency"},
    )

    synthesis_span = tracing.start_span(
        trace_id=trace_id,
        agent_name="handler_agent",
        operation_name="fallback_route",
        parent_span_id=root_span_id,
        metadata={"policy": "degraded_mode"},
    )
    time.sleep(0.002)
    tracing.end_span(
        trace_id,
        synthesis_span["span_id"],
        status="timeout",
        metadata={"timeout_ms": 1500},
    )

    if root_span_id:
        tracing.end_span(
            trace_id,
            root_span_id,
            status="ok",
            metadata={"workflow_state": "completed_with_degradation"},
        )

    critical_path = tracing.reconstruct_critical_path(trace_id)
    printer.pretty("CRITICAL_PATH", critical_path, "info")

    shared_memory_context = tracing.build_shared_memory_context(trace_id)
    printer.pretty("SHARED_MEMORY", shared_memory_context, "info")

    finalized = tracing.finalize_trace(
        trace_id,
        incident_level="warning",
        metadata={"finalized_by": "__main__", "demo": True},
    )
    printer.pretty("TRACE_FINALIZED", finalized, "info")

    active = tracing.list_active_traces()
    completed = tracing.list_completed_traces()
    printer.pretty("ACTIVE_TRACES", active, "info")
    printer.pretty("COMPLETED_TRACES", completed, "info")

    print("\n=== Test ran successfully ===\n")
