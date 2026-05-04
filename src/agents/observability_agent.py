"""
The Observability Agent converts multi-agent runtime behavior into actionable operational intelligence:
traces, bottlenecks, error families, saturation warnings, and incident summaries.

Interfaces and dependencies
Inputs:
- Logs/events from all agents
- Execution state transitions
- Queue and scheduler metrics

Outputs:
- Alert severity and incident status
- Root-cause hypotheses
- Remediation recommendations

KPIs
- Mean time to detect (MTTD)
- Mean time to resolve (MTTR)
- Alert precision (signal/noise ratio)
- Recurring incident rate
- User-facing degraded response rate

Failure modes & mitigations
- Alert fatigue: dedupe, rate-limit, and contextual suppression.
- Blind spots: enforce telemetry contract in BaseAgent hooks.
- High cardinality metrics: bounded label strategy and rollups.

This agent is the orchestration layer for the Observability subsystem. It is
responsible for normalizing heterogeneous operational inputs, delegating signal
production to tracing/capacity/performance submodules, synthesizing incidents
through the intelligence layer, and publishing operator-facing state into
shared memory for the wider multi-agent runtime.
"""

from __future__ import annotations

__version__ = "2.1.0"

import inspect
import time

from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from .base_agent import BaseAgent
from .base.utils.main_config_loader import load_global_config, get_config_section
from .observability import ObservabilityCapacity, ObservabilityIntelligence, ObservabilityTracing, ObservabilityPerformance
from .observability.utils.observability_error import ObservabilityError, ObservabilityErrorType, ObservabilitySeverity, normalize_observability_exception
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Observability Agent")
printer = PrettyPrinter

_LEVEL_RANK = {"info": 10, "warning": 20, "critical": 30}


class ObservabilityAgent(BaseAgent):
    """Production-grade orchestrator for the Observability subsystem."""

    def __init__(self, shared_memory, agent_factory, config=None, **kwargs):
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory, config=config)

        self.shared_memory = shared_memory or self.shared_memory
        self.agent_factory = agent_factory or self.agent_factory

        self.config = load_global_config()
        self.observability_config = get_config_section("observability_agent") or {}
        if isinstance(config, Mapping):
            self.observability_config.update(dict(config))
        if isinstance(kwargs, Mapping):
            self.observability_config.update(dict(kwargs))

        self.enabled = bool(self.observability_config.get("enabled", True))
        self.default_service = str(self.observability_config.get("default_service", "slai"))
        self.default_task_name = str(self.observability_config.get("default_task_name", "agent_workflow"))
        self.default_trace_operation = str(self.observability_config.get("default_trace_operation", "observe"))
        self.default_incident_status = str(self.observability_config.get("default_incident_status", "open"))

        self.auto_start_trace = bool(self.observability_config.get("auto_start_trace", True))
        self.auto_finalize_trace = bool(self.observability_config.get("auto_finalize_trace", True))
        self.enable_trace_span_ingestion = bool(self.observability_config.get("enable_trace_span_ingestion", True))
        self.enable_trace_event_ingestion = bool(self.observability_config.get("enable_trace_event_ingestion", True))
        self.enable_state_transition_ingestion = bool(
            self.observability_config.get("enable_state_transition_ingestion", True)
        )
        self.enable_log_ingestion = bool(self.observability_config.get("enable_log_ingestion", True))
        self.enable_performance_trace_analysis = bool(
            self.observability_config.get("enable_performance_trace_analysis", True)
        )
        self.enable_health_snapshots = bool(self.observability_config.get("enable_health_snapshots", True))
        self.enable_kpi_tracking = bool(self.observability_config.get("enable_kpi_tracking", True))
        self.enable_shared_context_export = bool(
            self.observability_config.get("enable_shared_context_export", True)
        )
        self.allow_degraded_reports = bool(self.observability_config.get("allow_degraded_reports", True))

        self.max_recent_reports = int(self.observability_config.get("max_recent_reports", 100))
        self.max_recent_errors = int(self.observability_config.get("max_recent_errors", 200))
        self.max_signature_history = int(self.observability_config.get("max_signature_history", 512))
        self.max_event_records_per_run = int(self.observability_config.get("max_event_records_per_run", 200))
        self.max_log_records_per_run = int(self.observability_config.get("max_log_records_per_run", 200))
        self.max_alert_records_per_run = int(self.observability_config.get("max_alert_records_per_run", 100))
        self.max_error_records_per_run = int(self.observability_config.get("max_error_records_per_run", 100))
        self.max_state_transition_records_per_run = int(
            self.observability_config.get("max_state_transition_records_per_run", 200)
        )
        self.max_objective_records_per_run = int(self.observability_config.get("max_objective_records_per_run", 50))
        self.max_related_agents = int(self.observability_config.get("max_related_agents", 16))

        self.alert_dedupe_window_seconds = float(
            self.observability_config.get("alert_dedupe_window_seconds", 1800.0)
        )
        self.alert_dedupe_repeat_threshold = int(
            self.observability_config.get("alert_dedupe_repeat_threshold", 3)
        )
        self.recurring_incident_threshold = int(
            self.observability_config.get("recurring_incident_threshold", 2)
        )
        self.alert_precision_decay = float(self.observability_config.get("alert_precision_decay", 1.0))

        self.degraded_status_levels = {
            str(level).lower() for level in self.observability_config.get("degraded_status_levels", ["warning", "critical"])
        }
        self.trace_context_required_fields = tuple(
            str(field) for field in self.observability_config.get(
                "required_trace_context_fields",
                ["task_name", "agent_name", "operation_name"],
            )
        )

        routing_config = self._mapping(self.observability_config.get("routing"))
        self.routing_config = routing_config
        self.auto_route_handler_on_warning = bool(routing_config.get("handler_on_warning", True))
        self.auto_route_handler_on_critical = bool(routing_config.get("handler_on_critical", True))
        self.auto_route_planning_on_capacity = bool(routing_config.get("planning_on_capacity", True))
        self.auto_route_safety_on_critical = bool(routing_config.get("safety_on_critical", True))
        self.auto_route_evaluation_on_degradation = bool(routing_config.get("evaluation_on_degradation", True))

        self.handler_agent_names = self._coerce_name_list(
            routing_config.get("handler_agent_names", ["handler", "handler_agent", "HandlerAgent"])
        )
        self.planning_agent_names = self._coerce_name_list(
            routing_config.get("planning_agent_names", ["planning", "planning_agent", "PlanningAgent"])
        )
        self.safety_agent_names = self._coerce_name_list(
            routing_config.get("safety_agent_names", ["safety", "safety_agent", "SafetyAgent"])
        )
        self.evaluation_agent_names = self._coerce_name_list(
            routing_config.get("evaluation_agent_names", ["evaluation", "evaluation_agent", "EvaluationAgent"])
        )

        self._init_shared_memory_keys()

        self.observability_capacity = self._build_subsystem(ObservabilityCapacity)
        self.observability_performance = self._build_subsystem(ObservabilityPerformance)
        self.observability_intel = self._build_subsystem(ObservabilityIntelligence)
        self.observability_tracing = self._build_subsystem(ObservabilityTracing)

    # ------------------------------------------------------------------
    # Core task entrypoint
    # ------------------------------------------------------------------
    def perform_task(self, task_data: Any) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "status": "disabled",
                "agent": self.name,
                "timestamp_ms": self._now_ms(),
                "reason": "observability_agent disabled by config",
            }

        payload = self._normalize_task_payload(task_data)
        self._validate_trace_seed(payload)

        run_state: Dict[str, Any] = {
            "started_at_ms": self._now_ms(),
            "pipeline_errors": [],
            "trace_analysis": None,
            "objective_results": [],
            "ingestion_stats": {
                "queue_samples": 0,
                "resource_samples": 0,
                "latency_samples": 0,
                "throughput_samples": 0,
                "span_records": 0,
                "event_records": 0,
                "state_transition_records": 0,
                "log_records": 0,
                "alert_records": 0,
                "error_records": 0,
            },
        }

        trace_context: Optional[Dict[str, Any]] = None
        report: Optional[Dict[str, Any]] = None
        fatal_error: Optional[ObservabilityError] = None

        try:
            trace_context = self._start_trace(payload, run_state)
            self._ingest_capacity_signals(payload, trace_context, run_state)
            self._ingest_performance_signals(payload, trace_context, run_state)
            self._ingest_trace_signals(payload, trace_context, run_state)
            self._ingest_runtime_events(payload, trace_context, run_state)
            self._evaluate_objectives(payload, trace_context, run_state)
            report = self._build_observability_report(payload, trace_context, run_state)
            self._persist_shared_memory_context(report)
            self._publish_shared_memory_contract(report)
            if self.enable_kpi_tracking:
                report["kpis"] = self._update_kpis(report, payload, trace_context)
            return report
        except Exception as exc:
            fatal_error = self._normalize_agent_exception(
                exc,
                stage="incident.brief",
                operation="perform_task",
                trace_context=trace_context,
            )
            self._record_error(fatal_error, trace_context=trace_context)
            raise fatal_error
        finally:
            self._finalize_trace(trace_context, report=report, fatal_error=fatal_error)

    # ------------------------------------------------------------------
    # BaseAgent compatibility hooks
    # ------------------------------------------------------------------
    def extract_performance_metrics(self, result: Any) -> dict:
        if not isinstance(result, Mapping):
            return {}

        incident = self._mapping(result.get("incident"))
        suppression = self._mapping(result.get("suppression"))
        metrics = {
            "report_status": str(result.get("status", "unknown")),
            "incident_level": str(result.get("incident_level") or incident.get("incident_level") or "info"),
            "pipeline_error_count": len(self._sequence(result.get("pipeline_errors"))),
            "suppressed_duplicate_alert": bool(suppression.get("suppress_duplicate_alert", False)),
            "recurring_incident": bool(suppression.get("recurring_incident", False)),
        }
        trace_metrics = self._mapping(self._mapping(result.get("trace")).get("analysis"))
        if trace_metrics:
            metrics["trace_total_duration_ms"] = float(trace_metrics.get("total_duration_ms", 0.0) or 0.0)
            metrics["trace_critical_path_ms"] = float(trace_metrics.get("critical_path_ms", 0.0) or 0.0)
            metrics["trace_anomaly_count"] = int(trace_metrics.get("anomaly_count", 0) or 0)
        return metrics

    def evaluate_performance(self, metrics: dict):
        if not metrics:
            return
        agent_metrics = self.performance_metrics["observability_agent"]
        agent_metrics.append(
            {
                "timestamp_ms": self._now_ms(),
                "incident_level": metrics.get("incident_level"),
                "pipeline_error_count": metrics.get("pipeline_error_count", 0),
                "trace_total_duration_ms": metrics.get("trace_total_duration_ms", 0.0),
                "trace_critical_path_ms": metrics.get("trace_critical_path_ms", 0.0),
                "trace_anomaly_count": metrics.get("trace_anomaly_count", 0),
                "suppressed_duplicate_alert": metrics.get("suppressed_duplicate_alert", False),
                "recurring_incident": metrics.get("recurring_incident", False),
            }
        )

    def alternative_execute(self, task_data, original_error=None):
        """Produce a degraded but operator-usable fallback report."""
        latest_health = self._sm_get(self.sm_keys["health"], default={}) or {}
        latest_report = self._sm_get(self.sm_keys["latest_report"], default={}) or {}

        fallback = {
            "status": "degraded",
            "agent": self.name,
            "timestamp_ms": self._now_ms(),
            "incident_level": "warning",
            "reason": "observability_agent fallback path engaged",
            "error": self._serialize_exception(original_error),
            "latest_health": latest_health,
            "latest_report_summary": {
                "trace_id": latest_report.get("trace_id"),
                "incident_id": latest_report.get("incident_id"),
                "incident_level": latest_report.get("incident_level"),
            },
            "recommended_actions": [
                "Inspect latest observability health snapshot.",
                "Review recent pipeline errors for recurring patterns.",
                "Re-run observability analysis with reduced payload if cardinality is high.",
            ],
        }
        self.shared_memory.set(self.sm_keys["latest_report"], fallback)
        return fallback

    # ------------------------------------------------------------------
    # Shared-memory wiring
    # ------------------------------------------------------------------
    def _init_shared_memory_keys(self) -> None:
        prefix = f"observability:{self.name}"
        self.sm_keys = {
            "latest_report": f"{prefix}:latest_report",
            "latest_incident": f"{prefix}:latest_incident",
            "recent_reports": f"{prefix}:recent_reports",
            "health": f"{prefix}:health",
            "errors": f"{prefix}:errors",
            "kpis": f"{prefix}:kpis",
            "signature_history": f"{prefix}:signature_history",
            "routing": f"{prefix}:routing",
        }

    def _persist_shared_memory_context(self, report: Mapping[str, Any]) -> None:
        self.shared_memory.set(self.sm_keys["latest_report"], dict(report))
        self.shared_memory.set(self.sm_keys["latest_incident"], dict(self._mapping(report.get("incident"))))

        recent_reports = self._sm_get(self.sm_keys["recent_reports"], default=[]) or []
        recent_reports = [dict(item) for item in recent_reports if isinstance(item, Mapping)]
        recent_reports.append(dict(report))
        if len(recent_reports) > self.max_recent_reports:
            recent_reports = recent_reports[-self.max_recent_reports :]
        self.shared_memory.set(self.sm_keys["recent_reports"], recent_reports)

        if self.enable_health_snapshots:
            self.shared_memory.set(self.sm_keys["health"], dict(report.get("health") or {}))
        self.shared_memory.set(self.sm_keys["routing"], list(report.get("routing") or []))

    def _publish_shared_memory_contract(self, report: Mapping[str, Any]) -> None:
        if not self.enable_shared_context_export:
            return

        trace = self._mapping(report.get("trace"))
        incident = self._mapping(report.get("incident"))
        performance = self._mapping(report.get("performance"))
        recommended_actions = self._collect_recommended_actions(report)

        contract_entries = {
            "observability.trace_id": report.get("trace_id"),
            "observability.agent_spans": trace.get("spans", []),
            "observability.error_clusters": self._mapping(trace.get("shared_context")).get("observability.error_clusters", []),
            "observability.latency_p95": self._extract_latency_p95(performance),
            "observability.incident_level": report.get("incident_level") or incident.get("incident_level"),
            "observability.recommended_actions": recommended_actions,
        }

        for key, value in contract_entries.items():
            if value is not None:
                self.shared_memory.set(key, value)

    # ------------------------------------------------------------------
    # Payload normalization and validation
    # ------------------------------------------------------------------
    def _normalize_task_payload(self, task_data: Any) -> Dict[str, Any]:
        if task_data is None:
            return {}
        if isinstance(task_data, Mapping):
            return dict(task_data)
        if isinstance(task_data, Sequence) and not isinstance(task_data, (str, bytes, bytearray)):
            return {"events": [item for item in task_data if isinstance(item, Mapping)]}
        return {"raw_input": task_data}

    def _validate_trace_seed(self, payload: Mapping[str, Any]) -> None:
        trace_info = self._mapping(payload.get("trace"))
        missing: List[str] = []
    
        for field_name in self.trace_context_required_fields:
            if field_name == "task_name":
                value = trace_info.get("task_name") or payload.get("task_name") or self.default_task_name
            elif field_name == "agent_name":
                value = trace_info.get("agent_name") or payload.get("agent_name") or self.name
            elif field_name == "operation_name":
                value = trace_info.get("operation_name") or payload.get("operation_name") or self.default_trace_operation
            else:
                value = trace_info.get(field_name) or payload.get(field_name)
    
            is_missing = (
                value is None
                or value == ""
                or (isinstance(value, (list, tuple, set, dict)) and len(value) == 0)
            )
    
            if is_missing:
                missing.append(field_name)
    
        if missing:
            raise ObservabilityError(
                message=f"Observability task payload is missing required trace seed fields: {missing}",
                error_type=ObservabilityErrorType.TELEMETRY_CONTRACT_VIOLATION,
                severity=ObservabilitySeverity.HIGH,
                retryable=False,
                context={"missing_fields": missing, "operation": "perform_task", "agent_name": self.name},
                remediation="Provide the required trace/task seed fields before invoking ObservabilityAgent.",
            )

    # ------------------------------------------------------------------
    # Subsystem orchestration
    # ------------------------------------------------------------------
    def _start_trace(self, payload: Mapping[str, Any], run_state: MutableMapping[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.auto_start_trace:
            return self._build_synthetic_trace_context(payload)

        trace_info = self._mapping(payload.get("trace"))
        task_name = str(trace_info.get("task_name") or payload.get("task_name") or self.default_task_name)
        operation_name = str(
            trace_info.get("operation_name") or payload.get("operation_name") or self.default_trace_operation
        )
        source_agent = str(trace_info.get("agent_name") or payload.get("agent_name") or self.name)

        if not hasattr(self.observability_tracing, "start_trace"):
            return self._build_synthetic_trace_context(payload, agent_name=source_agent, operation_name=operation_name)

        try:
            trace_response = self._invoke_supported(
                self.observability_tracing,
                "start_trace",
                task_name=task_name,
                agent_name=source_agent,
                operation_name=operation_name,
                service=str(trace_info.get("service") or payload.get("service") or self.default_service),
                metadata={
                    "source": payload.get("source", "runtime"),
                    "pipeline": payload.get("pipeline", "default"),
                    "observer": self.name,
                    **self._mapping(trace_info.get("metadata")),
                },
                trace_id=trace_info.get("trace_id"),
                incident_level=trace_info.get("incident_level"),
                start_root_span=True,
                root_span_metadata=self._mapping(trace_info.get("root_span_metadata") or trace_info.get("metadata")),
            ) or {}
            return {
                "trace_id": trace_response.get("trace_id") or trace_info.get("trace_id") or self._new_trace_id(),
                "root_span_id": trace_response.get("root_span_id"),
                "agent_name": source_agent,
                "operation_name": operation_name,
                "service": str(trace_info.get("service") or payload.get("service") or self.default_service),
                "managed_by_tracer": True,
            }
        except Exception as exc:
            normalized = self._capture_stage_error(
                run_state,
                exc,
                stage="tracing.collect",
                operation="start_trace",
                trace_context={"agent_name": source_agent, "operation_name": operation_name},
                escalate=not self.allow_degraded_reports,
            )
            if not self.allow_degraded_reports:
                raise normalized
            return self._build_synthetic_trace_context(payload, agent_name=source_agent, operation_name=operation_name)

    def _ingest_capacity_signals(
        self,
        payload: Mapping[str, Any],
        trace_context: Optional[Mapping[str, Any]],
        run_state: MutableMapping[str, Any],
    ) -> None:
        for queue in self._bounded_records(payload.get("queues"), self.max_event_records_per_run):
            try:
                self._invoke_supported(
                    self.observability_capacity,
                    "record_queue_sample",
                    queue_name=str(queue.get("queue_name") or queue.get("name") or "default"),
                    depth=queue.get("depth", queue.get("queue_depth", 0.0)),
                    inflow_per_sec=queue.get("inflow_per_sec"),
                    outflow_per_sec=queue.get("outflow_per_sec"),
                    timestamp_ms=queue.get("timestamp_ms"),
                    metadata=self._mapping(queue.get("metadata")),
                )
                run_state["ingestion_stats"]["queue_samples"] += 1
            except Exception as exc:
                self._capture_stage_error(
                    run_state,
                    exc,
                    stage="capacity.queue",
                    operation="record_queue_sample",
                    trace_context=trace_context,
                )

        for resource in self._bounded_records(payload.get("resources"), self.max_event_records_per_run):
            try:
                self._invoke_supported(
                    self.observability_capacity,
                    "record_resource_sample",
                    resource_name=str(resource.get("resource_name") or resource.get("name") or "resource"),
                    resource_type=str(resource.get("resource_type") or resource.get("type") or "unknown"),
                    utilization_pct=resource.get("utilization_pct", resource.get("utilization", 0.0)),
                    used=resource.get("used"),
                    capacity=resource.get("capacity"),
                    timestamp_ms=resource.get("timestamp_ms"),
                    metadata=self._mapping(resource.get("metadata")),
                )
                run_state["ingestion_stats"]["resource_samples"] += 1
            except Exception as exc:
                self._capture_stage_error(
                    run_state,
                    exc,
                    stage="capacity.resource",
                    operation="record_resource_sample",
                    trace_context=trace_context,
                )

    def _ingest_performance_signals(
        self,
        payload: Mapping[str, Any],
        trace_context: Optional[Mapping[str, Any]],
        run_state: MutableMapping[str, Any],
    ) -> None:
        trace_id = self._trace_id(trace_context)
        default_subject = str(payload.get("service") or self.default_service)

        for latency in self._bounded_records(payload.get("latencies"), self.max_event_records_per_run):
            try:
                self._invoke_supported(
                    self.observability_performance,
                    "record_latency_sample",
                    subject=str(latency.get("subject") or default_subject),
                    duration_ms=latency.get("duration_ms", latency.get("latency_ms", 0.0)),
                    trace_id=trace_id,
                    timestamp_ms=latency.get("timestamp_ms"),
                    status=str(latency.get("status") or "ok"),
                    metadata=self._mapping(latency.get("metadata")),
                )
                run_state["ingestion_stats"]["latency_samples"] += 1
            except Exception as exc:
                self._capture_stage_error(
                    run_state,
                    exc,
                    stage="performance.latency",
                    operation="record_latency_sample",
                    trace_context=trace_context,
                )

        for throughput in self._bounded_records(payload.get("throughput"), self.max_event_records_per_run):
            try:
                self._invoke_supported(
                    self.observability_performance,
                    "record_throughput_sample",
                    subject=str(throughput.get("subject") or default_subject),
                    count=int(throughput.get("count", 1) or 1),
                    success_count=int(throughput.get("success_count", 0) or 0),
                    failure_count=int(throughput.get("failure_count", 0) or 0),
                    timestamp_ms=throughput.get("timestamp_ms"),
                    metadata=self._mapping(throughput.get("metadata")),
                )
                run_state["ingestion_stats"]["throughput_samples"] += 1
            except Exception as exc:
                self._capture_stage_error(
                    run_state,
                    exc,
                    stage="performance.throughput",
                    operation="record_throughput_sample",
                    trace_context=trace_context,
                )

    def _ingest_trace_signals(
        self,
        payload: Mapping[str, Any],
        trace_context: Optional[Mapping[str, Any]],
        run_state: MutableMapping[str, Any],
    ) -> None:
        if not self.enable_trace_span_ingestion:
            return

        trace_id = self._trace_id(trace_context)
        spans = self._bounded_records(payload.get("spans"), self.max_event_records_per_run)
        for span in spans:
            normalized_span = dict(span)
            if trace_id and not normalized_span.get("trace_id"):
                normalized_span["trace_id"] = trace_id
            if trace_context and not normalized_span.get("agent_name"):
                normalized_span["agent_name"] = trace_context.get("agent_name")
            if trace_context and not normalized_span.get("operation_name"):
                normalized_span["operation_name"] = trace_context.get("operation_name")
            try:
                if hasattr(self.observability_tracing, "ingest_span"):
                    self._invoke_supported(
                        self.observability_tracing,
                        "ingest_span",
                        span_payload=normalized_span,
                        allow_create_trace=True,
                    )
                run_state["ingestion_stats"]["span_records"] += 1
            except Exception as exc:
                self._capture_stage_error(
                    run_state,
                    exc,
                    stage="tracing.collect",
                    operation="ingest_span",
                    trace_context=trace_context,
                )

        resolved_trace_id = self._trace_id(trace_context)
        exported_spans = []
        if resolved_trace_id and hasattr(self.observability_tracing, "export_trace_spans"):
            try:
                exported_spans = list(
                    self._invoke_supported(self.observability_tracing, "export_trace_spans", trace_id=resolved_trace_id) or []
                )
            except Exception as exc:
                self._capture_stage_error(
                    run_state,
                    exc,
                    stage="tracing.collect",
                    operation="export_trace_spans",
                    trace_context=trace_context,
                )
        if not exported_spans:
            exported_spans = spans
        run_state["trace_spans"] = [dict(item) for item in exported_spans if isinstance(item, Mapping)]

        if self.enable_performance_trace_analysis and resolved_trace_id and run_state["trace_spans"]:
            try:
                trace_analysis = self._invoke_supported(
                    self.observability_performance,
                    "analyze_trace",
                    trace_id=resolved_trace_id,
                    spans=run_state["trace_spans"],
                    metadata={
                        "source_agent": self._trace_agent_name(trace_context),
                        "service": self._trace_service(trace_context, payload),
                        "observer": self.name,
                    },
                )
                if isinstance(trace_analysis, Mapping):
                    run_state["trace_analysis"] = dict(trace_analysis)
            except Exception as exc:
                self._capture_stage_error(
                    run_state,
                    exc,
                    stage="performance.latency",
                    operation="analyze_trace",
                    trace_context=trace_context,
                )

    def _ingest_runtime_events(
        self,
        payload: Mapping[str, Any],
        trace_context: Optional[Mapping[str, Any]],
        run_state: MutableMapping[str, Any],
    ) -> None:
        if not self.enable_trace_event_ingestion:
            return

        trace_id = self._trace_id(trace_context)
        if not trace_id:
            return

        for event in self._bounded_records(payload.get("events"), self.max_event_records_per_run):
            self._append_trace_event(
                trace_context,
                run_state,
                event_type=str(event.get("event_type") or event.get("type") or "event"),
                message=event.get("message"),
                severity=str(event.get("severity") or "info"),
                agent_name=event.get("agent_name"),
                span_id=event.get("span_id"),
                payload=self._mapping(event.get("payload")),
                correlation_keys=self._mapping(event.get("correlation_keys")),
                timestamp_ms=event.get("timestamp_ms"),
                counter_key="event_records",
            )

        if self.enable_state_transition_ingestion:
            for transition in self._bounded_records(
                payload.get("state_transitions"),
                self.max_state_transition_records_per_run,
            ):
                transition_payload = {
                    "from_state": transition.get("from_state") or transition.get("from"),
                    "to_state": transition.get("to_state") or transition.get("to"),
                    "reason": transition.get("reason"),
                    **self._mapping(transition.get("metadata")),
                }
                self._append_trace_event(
                    trace_context,
                    run_state,
                    event_type="state_transition",
                    message=transition.get("message") or f"State transitioned to '{transition_payload.get('to_state')}'.",
                    severity=str(transition.get("severity") or "info"),
                    agent_name=transition.get("agent_name") or self._trace_agent_name(trace_context),
                    span_id=transition.get("span_id"),
                    payload=transition_payload,
                    correlation_keys=self._mapping(transition.get("correlation_keys")),
                    timestamp_ms=transition.get("timestamp_ms"),
                    counter_key="state_transition_records",
                )

        if self.enable_log_ingestion:
            for log_record in self._bounded_records(payload.get("logs"), self.max_log_records_per_run):
                self._append_trace_event(
                    trace_context,
                    run_state,
                    event_type=f"log.{str(log_record.get('level') or 'info').lower()}",
                    message=log_record.get("message") or log_record.get("text") or "log event",
                    severity=str(log_record.get("severity") or log_record.get("level") or "info"),
                    agent_name=log_record.get("agent_name") or self._trace_agent_name(trace_context),
                    span_id=log_record.get("span_id"),
                    payload={
                        "logger": log_record.get("logger"),
                        "code": log_record.get("code"),
                        **self._mapping(log_record.get("metadata")),
                    },
                    correlation_keys=self._mapping(log_record.get("correlation_keys")),
                    timestamp_ms=log_record.get("timestamp_ms"),
                    counter_key="log_records",
                )

        run_state["ingestion_stats"]["alert_records"] = len(
            self._bounded_records(payload.get("alerts"), self.max_alert_records_per_run)
        )
        run_state["ingestion_stats"]["error_records"] = len(
            self._bounded_records(payload.get("error_events"), self.max_error_records_per_run)
        )

    def _evaluate_objectives(
        self,
        payload: Mapping[str, Any],
        trace_context: Optional[Mapping[str, Any]],
        run_state: MutableMapping[str, Any],
    ) -> None:
        objectives = self._bounded_records(payload.get("latency_objectives"), self.max_objective_records_per_run)
        if not objectives:
            return

        for objective in objectives:
            try:
                result = self._invoke_supported(
                    self.observability_performance,
                    "evaluate_latency_slo",
                    service=str(objective.get("service") or payload.get("service") or self.default_service),
                    slo_name=str(objective.get("slo_name") or "p95_latency"),
                    target_ms=float(objective.get("target_ms", objective.get("target", 0.0)) or 0.0),
                    percentile=int(objective.get("percentile", 95) or 95),
                    raise_on_breach=bool(objective.get("raise_on_breach", False)),
                    subject=str(objective.get("subject") or payload.get("service") or self.default_service),
                    metadata={
                        "trace_id": self._trace_id(trace_context),
                        **self._mapping(objective.get("metadata")),
                    },
                )
                if isinstance(result, Mapping):
                    run_state["objective_results"].append(dict(result))
            except Exception as exc:
                self._capture_stage_error(
                    run_state,
                    exc,
                    stage="slo.evaluate",
                    operation="evaluate_latency_slo",
                    trace_context=trace_context,
                )

    def _build_observability_report(
        self,
        payload: Mapping[str, Any],
        trace_context: Optional[Mapping[str, Any]],
        run_state: MutableMapping[str, Any],
    ) -> Dict[str, Any]:
        trace_id = self._trace_id(trace_context)
        trace_spans = [dict(span) for span in self._sequence(run_state.get("trace_spans")) if isinstance(span, Mapping)]
        trace_analysis = self._mapping(run_state.get("trace_analysis"))

        capacity_summary = self._safe_subsystem_report(
            target=self.observability_capacity,
            method_name="summarize_capacity",
            stage="capacity.queue",
            operation="summarize_capacity",
            trace_context=trace_context,
            run_state=run_state,
            default={"status": "unavailable", "scope": "capacity", "alerts": []},
        )
        performance_summary = self._safe_subsystem_report(
            target=self.observability_performance,
            method_name="summarize_performance",
            stage="performance.latency",
            operation="summarize_performance",
            trace_context=trace_context,
            run_state=run_state,
            default={"status": "unavailable", "scope": "performance"},
        )
        if trace_analysis:
            performance_summary.setdefault("current_trace", trace_analysis)

        incident_id = str(payload.get("incident_id") or f"obs-{int(self._now_ms())}")
        alerts = self._bounded_records(payload.get("alerts"), self.max_alert_records_per_run)
        error_events = self._bounded_records(payload.get("error_events"), self.max_error_records_per_run)
        error_events.extend(self._pipeline_errors_as_events(run_state.get("pipeline_errors")))

        incident_assessment = self._safe_subsystem_report(
            target=self.observability_intel,
            method_name="synthesize_incident",
            stage="incident.brief",
            operation="synthesize_incident",
            trace_context=trace_context,
            run_state=run_state,
            default=self._fallback_incident_assessment(
                incident_id=incident_id,
                trace_id=trace_id,
                trace_analysis=trace_analysis,
                capacity_summary=capacity_summary,
                performance_summary=performance_summary,
                error_events=error_events,
            ),
            incident_id=incident_id,
            spans=trace_spans,
            performance_report=performance_summary,
            capacity_report=capacity_summary,
            alerts=alerts,
            error_events=error_events,
            metadata={
                "source_agent": self._trace_agent_name(trace_context),
                "service": self._trace_service(trace_context, payload),
                "trace_id": trace_id,
                "observer": self.name,
                "pipeline_errors": len(self._sequence(run_state.get("pipeline_errors"))),
            },
        )

        incident_level = str(
            self._mapping(incident_assessment).get("incident_level")
            or self._mapping(self._mapping(incident_assessment).get("brief")).get("incident_level")
            or self._derive_incident_level_from_errors(run_state.get("pipeline_errors"))
            or "info"
        ).lower()
        suppression = self._apply_alert_policy(incident_assessment)
        routing = self._build_routing_recommendations(
            incident_assessment=incident_assessment,
            capacity_summary=capacity_summary,
            performance_summary=performance_summary,
            trace_analysis=trace_analysis,
            suppression=suppression,
        )
        trace_summary = self._build_trace_summary(trace_context, trace_spans, trace_analysis)
        health = self._build_health_snapshot(
            trace_id=trace_id,
            incident_id=incident_id,
            incident_level=incident_level,
            suppression=suppression,
            run_state=run_state,
            routing=routing,
        )

        status = "degraded" if incident_level in self.degraded_status_levels or run_state["pipeline_errors"] else "ok"
        return {
            "status": status,
            "agent": self.name,
            "timestamp_ms": self._now_ms(),
            "started_at_ms": run_state.get("started_at_ms"),
            "trace_id": trace_id,
            "incident_id": incident_id,
            "incident_level": incident_level,
            "service": self._trace_service(trace_context, payload),
            "trace": trace_summary,
            "capacity": capacity_summary,
            "performance": performance_summary,
            "incident": incident_assessment,
            "suppression": suppression,
            "routing": routing,
            "health": health,
            "ingestion": dict(run_state.get("ingestion_stats") or {}),
            "objective_results": list(run_state.get("objective_results") or []),
            "pipeline_errors": list(run_state.get("pipeline_errors") or []),
        }

    def _finalize_trace(
        self,
        trace_context: Optional[Mapping[str, Any]],
        *,
        report: Optional[Mapping[str, Any]],
        fatal_error: Optional[ObservabilityError],
    ) -> None:
        if not self.auto_finalize_trace or not trace_context or not trace_context.get("managed_by_tracer"):
            return
        trace_id = self._trace_id(trace_context)
        root_span_id = trace_context.get("root_span_id")
        if not trace_id or not hasattr(self.observability_tracing, "finalize_trace"):
            return

        final_status = self._resolve_trace_final_status(report=report, fatal_error=fatal_error)
        try:
            if root_span_id and hasattr(self.observability_tracing, "end_span"):
                self._invoke_supported(
                    self.observability_tracing,
                    "end_span",
                    trace_id=trace_id,
                    span_id=root_span_id,
                    status=final_status,
                    metadata={"observer": self.name},
                )
            self._invoke_supported(
                self.observability_tracing,
                "finalize_trace",
                trace_id=trace_id,
                status=final_status,
                incident_level=self._resolve_final_incident_level(report=report, fatal_error=fatal_error),
                metadata={"observer": self.name},
            )
        except Exception as exc:
            normalized = self._normalize_agent_exception(
                exc,
                stage="tracing.collect",
                operation="finalize_trace",
                trace_context=trace_context,
            )
            self._record_error(normalized, trace_context=trace_context)

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------
    def _build_trace_summary(
        self,
        trace_context: Optional[Mapping[str, Any]],
        trace_spans: Sequence[Mapping[str, Any]],
        trace_analysis: Mapping[str, Any],
    ) -> Dict[str, Any]:
        shared_context: Dict[str, Any] = {}
        trace_id = self._trace_id(trace_context)
        if trace_id and hasattr(self.observability_tracing, "build_shared_memory_context"):
            try:
                shared_context = dict(
                    self._invoke_supported(self.observability_tracing, "build_shared_memory_context", trace_id=trace_id)
                    or {}
                )
            except Exception:
                shared_context = {}

        return {
            "trace_id": trace_id,
            "root_span_id": trace_context.get("root_span_id") if trace_context else None,
            "agent_name": self._trace_agent_name(trace_context),
            "operation_name": self._trace_operation_name(trace_context),
            "service": self._trace_service(trace_context, {}),
            "span_count": len(trace_spans),
            "spans": [dict(span) for span in trace_spans if isinstance(span, Mapping)],
            "analysis": dict(trace_analysis),
            "shared_context": shared_context,
        }

    def _build_health_snapshot(
        self,
        *,
        trace_id: Optional[str],
        incident_id: str,
        incident_level: str,
        suppression: Mapping[str, Any],
        run_state: Mapping[str, Any],
        routing: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        pipeline_errors = self._sequence(run_state.get("pipeline_errors"))
        return {
            "status": "degraded" if incident_level in self.degraded_status_levels or pipeline_errors else "ok",
            "trace_id": trace_id,
            "incident_id": incident_id,
            "incident_level": incident_level,
            "updated_at_ms": self._now_ms(),
            "pipeline_error_count": len(pipeline_errors),
            "suppressed_duplicate_alert": bool(self._mapping(suppression).get("suppress_duplicate_alert", False)),
            "recurring_incident": bool(self._mapping(suppression).get("recurring_incident", False)),
            "routing_targets": [item.get("target_agent") for item in routing if isinstance(item, Mapping)],
        }

    def _fallback_incident_assessment(
        self,
        *,
        incident_id: str,
        trace_id: Optional[str],
        trace_analysis: Mapping[str, Any],
        capacity_summary: Mapping[str, Any],
        performance_summary: Mapping[str, Any],
        error_events: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        incident_level = self._derive_incident_level_from_trace(trace_analysis)
        if incident_level == "info":
            incident_level = self._derive_incident_level_from_errors(error_events)

        error_signature = self._build_error_signature(trace_analysis, error_events)
        summary = "Fallback incident synthesis used because the intelligence module was unavailable."
        brief = {
            "incident_id": incident_id,
            "incident_level": incident_level,
            "status": self.default_incident_status,
            "summary": summary,
            "customer_impact": "Potential runtime degradation requires review." if incident_level != "info" else "Limited impact currently inferred.",
            "trace_id": trace_id,
            "error_signature": error_signature,
            "started_at_ms": self._now_ms(),
            "generated_at_ms": self._now_ms(),
            "primary_symptoms": self._fallback_symptoms(trace_analysis, capacity_summary, performance_summary),
            "top_root_causes": [],
            "recommended_runbooks": [],
            "evidence_snapshot": {
                "trace_analysis": dict(trace_analysis),
                "capacity_summary": dict(capacity_summary),
                "performance_summary": dict(performance_summary),
            },
            "timeline": [],
            "similar_incidents": [],
            "suppress_duplicate_alert": False,
        }
        return {
            "incident_id": incident_id,
            "incident_level": incident_level,
            "status": self.default_incident_status,
            "score": float(_LEVEL_RANK.get(incident_level, 10)),
            "trace_id": trace_id,
            "error_signature": error_signature,
            "signals": [],
            "root_causes": [],
            "runbooks": [],
            "brief": brief,
            "waterfall_summary": dict(trace_analysis),
            "capacity_summary": dict(capacity_summary),
            "performance_summary": dict(performance_summary),
            "alert_fatigue": {"suppress_duplicate_alert": False},
            "similar_incidents": [],
        }

    def _apply_alert_policy(self, incident_assessment: Mapping[str, Any]) -> Dict[str, Any]:
        incident = self._mapping(incident_assessment)
        brief = self._mapping(incident.get("brief"))
        error_signature = str(brief.get("error_signature") or incident.get("error_signature") or "").strip()
        if not error_signature:
            return {
                "error_signature": "",
                "repeat_count": 0,
                "recurring_incident": False,
                "suppress_duplicate_alert": False,
                "window_seconds": self.alert_dedupe_window_seconds,
            }

        now_ms = self._now_ms()
        window_start_ms = now_ms - (self.alert_dedupe_window_seconds * 1000.0)
        history = self._sm_get(self.sm_keys["signature_history"], default=[]) or []
        normalized_history: List[Dict[str, Any]] = []
        for record in history:
            if not isinstance(record, Mapping):
                continue
            timestamp_ms = float(record.get("timestamp_ms", 0.0) or 0.0)
            if timestamp_ms >= window_start_ms:
                normalized_history.append(
                    {
                        "error_signature": str(record.get("error_signature") or ""),
                        "timestamp_ms": timestamp_ms,
                        "incident_level": str(record.get("incident_level") or "info"),
                        "incident_id": record.get("incident_id"),
                    }
                )

        repeat_count = sum(1 for entry in normalized_history if entry["error_signature"] == error_signature) + 1
        recurring_incident = repeat_count >= self.recurring_incident_threshold
        suppress_duplicate_alert = repeat_count >= self.alert_dedupe_repeat_threshold and str(
            brief.get("incident_level") or incident.get("incident_level") or "info"
        ).lower() != "critical"

        normalized_history.append(
            {
                "error_signature": error_signature,
                "timestamp_ms": now_ms,
                "incident_level": str(brief.get("incident_level") or incident.get("incident_level") or "info"),
                "incident_id": incident.get("incident_id") or brief.get("incident_id"),
            }
        )
        if len(normalized_history) > self.max_signature_history:
            normalized_history = normalized_history[-self.max_signature_history :]
        self.shared_memory.set(self.sm_keys["signature_history"], normalized_history)

        return {
            "error_signature": error_signature,
            "repeat_count": repeat_count,
            "recurring_incident": recurring_incident,
            "suppress_duplicate_alert": suppress_duplicate_alert,
            "window_seconds": self.alert_dedupe_window_seconds,
        }

    def _build_routing_recommendations(
        self,
        *,
        incident_assessment: Mapping[str, Any],
        capacity_summary: Mapping[str, Any],
        performance_summary: Mapping[str, Any],
        trace_analysis: Mapping[str, Any],
        suppression: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        incident = self._mapping(incident_assessment)
        brief = self._mapping(incident.get("brief"))
        incident_level = str(brief.get("incident_level") or incident.get("incident_level") or "info").lower()
        suppression_state = self._mapping(suppression)

        recommendations: List[Dict[str, Any]] = []
        if self.auto_route_handler_on_warning and incident_level in {"warning", "critical"}:
            recommendations.append(
                self._make_routing_action(
                    target_role="handler",
                    target_names=self.handler_agent_names,
                    reason="Use fallback/degraded execution when runtime instability is detected.",
                    priority=10 if incident_level == "critical" else 20,
                    context={"incident_level": incident_level, "suppress_duplicate_alert": suppression_state.get("suppress_duplicate_alert")},
                )
            )

        if self.auto_route_planning_on_capacity and self._capacity_indicates_replanning(capacity_summary):
            recommendations.append(
                self._make_routing_action(
                    target_role="planning",
                    target_names=self.planning_agent_names,
                    reason="Runtime capacity pressure suggests replanning or load-shedding is needed.",
                    priority=30,
                    context={"capacity_alerts": self._sequence(self._mapping(capacity_summary).get("alerts"))},
                )
            )

        if self.auto_route_safety_on_critical and incident_level == "critical":
            recommendations.append(
                self._make_routing_action(
                    target_role="safety",
                    target_names=self.safety_agent_names,
                    reason="Critical user-facing degradation should be escalated through the safety/risk path.",
                    priority=5,
                    context={"incident_id": incident.get("incident_id") or brief.get("incident_id")},
                )
            )

        if self.auto_route_evaluation_on_degradation and self._requires_quality_reassessment(
            performance_summary,
            trace_analysis,
            incident_level,
        ):
            recommendations.append(
                self._make_routing_action(
                    target_role="evaluation",
                    target_names=self.evaluation_agent_names,
                    reason="Degradation and retries suggest downstream quality should be re-evaluated.",
                    priority=40,
                    context={"incident_level": incident_level},
                )
            )

        recommendations.sort(key=lambda item: int(item.get("priority", 999)))
        return recommendations[: self.max_related_agents]

    # ------------------------------------------------------------------
    # KPI and error handling
    # ------------------------------------------------------------------
    def _update_kpis(
        self,
        report: Mapping[str, Any],
        payload: Mapping[str, Any],
        trace_context: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        stored = self._mapping(self._sm_get(self.sm_keys["kpis"], default={}) or {})
        total_reports = int(stored.get("total_reports", 0) or 0) + 1
        total_incidents = int(stored.get("total_incidents", 0) or 0) + 1
        critical_incidents = int(stored.get("critical_incidents", 0) or 0)
        if str(report.get("incident_level") or "info").lower() == "critical":
            critical_incidents += 1

        recurring_incidents = int(stored.get("recurring_incidents", 0) or 0)
        if bool(self._mapping(report.get("suppression")).get("recurring_incident", False)):
            recurring_incidents += 1

        degraded_reports = int(stored.get("degraded_reports", 0) or 0)
        if str(report.get("status") or "ok").lower() == "degraded":
            degraded_reports += 1

        suppressed_alerts = int(stored.get("suppressed_alerts", 0) or 0)
        if bool(self._mapping(report.get("suppression")).get("suppress_duplicate_alert", False)):
            suppressed_alerts += 1

        actionable_alerts = int(stored.get("actionable_alerts", 0) or 0)
        if not bool(self._mapping(report.get("suppression")).get("suppress_duplicate_alert", False)):
            actionable_alerts += 1

        detection_started_ms = self._detect_observation_start(payload, trace_context, report)
        detection_latency_ms = max(0.0, float(report.get("timestamp_ms", self._now_ms())) - detection_started_ms)
        mttd_ms = self._rolling_average(stored.get("mttd_ms"), detection_latency_ms, total_reports)

        resolution_latency_ms = None
        resolution = self._mapping(payload.get("resolution"))
        resolved_at_ms = resolution.get("resolved_at_ms")
        if resolved_at_ms is not None:
            resolution_latency_ms = max(0.0, float(resolved_at_ms) - detection_started_ms)
        resolved_incidents = int(stored.get("resolved_incidents", 0) or 0)
        if resolution_latency_ms is not None:
            resolved_incidents += 1
        mttr_ms = stored.get("mttr_ms")
        if resolution_latency_ms is not None:
            mttr_ms = self._rolling_average(mttr_ms, resolution_latency_ms, resolved_incidents)

        alert_precision = 0.0 if total_incidents <= 0 else actionable_alerts / float(total_incidents)
        recurring_incident_rate = 0.0 if total_incidents <= 0 else recurring_incidents / float(total_incidents)
        degraded_response_rate = 0.0 if total_reports <= 0 else degraded_reports / float(total_reports)

        snapshot = {
            "updated_at_ms": self._now_ms(),
            "total_reports": total_reports,
            "total_incidents": total_incidents,
            "critical_incidents": critical_incidents,
            "recurring_incidents": recurring_incidents,
            "resolved_incidents": resolved_incidents,
            "degraded_reports": degraded_reports,
            "suppressed_alerts": suppressed_alerts,
            "actionable_alerts": actionable_alerts,
            "mttd_ms": mttd_ms,
            "mttr_ms": mttr_ms,
            "alert_precision": alert_precision,
            "recurring_incident_rate": recurring_incident_rate,
            "user_facing_degraded_response_rate": degraded_response_rate,
            "latest_incident_level": report.get("incident_level"),
            "latest_trace_id": report.get("trace_id"),
            "latest_incident_id": report.get("incident_id"),
        }
        self.shared_memory.set(self.sm_keys["kpis"], snapshot)
        return snapshot

    def _record_error(self, error: ObservabilityError, *, trace_context: Optional[Mapping[str, Any]]) -> None:
        payload = self._error_payload(error)
        logger.error("Observability pipeline failure: %s", payload.get("message"))

        recent_errors = self._sm_get(self.sm_keys["errors"], default=[]) or []
        recent_errors = [dict(item) for item in recent_errors if isinstance(item, Mapping)]
        recent_errors.append(payload)
        if len(recent_errors) > self.max_recent_errors:
            recent_errors = recent_errors[-self.max_recent_errors :]
        self.shared_memory.set(self.sm_keys["errors"], recent_errors)

        base_error_info = {
            "timestamp": time.time(),
            "error_type": payload.get("error_type", type(error).__name__),
            "error_message": payload.get("message", str(error)),
            "traceback": payload.get("context", {}).get("traceback") if isinstance(payload.get("context"), Mapping) else None,
        }
        try:
            self._log_error_to_shared_memory(base_error_info)
        except Exception:
            pass

        trace_id = self._trace_id(trace_context)
        if trace_id and hasattr(self.observability_tracing, "append_event"):
            try:
                self._invoke_supported(
                    self.observability_tracing,
                    "append_event",
                    trace_id=trace_id,
                    event_type="observability_agent.error",
                    message=payload.get("message"),
                    severity=payload.get("severity", "warning"),
                    agent_name=self.name,
                    payload={
                        "error_type": payload.get("error_type"),
                        "error_code": payload.get("error_code"),
                    },
                )
            except Exception:
                logger.debug("Failed to append tracing error event for trace '%s'.", trace_id)

    def _capture_stage_error(
        self,
        run_state: MutableMapping[str, Any],
        exc: Exception,
        *,
        stage: str,
        operation: str,
        trace_context: Optional[Mapping[str, Any]],
        escalate: bool = False,
    ) -> ObservabilityError:
        normalized = self._normalize_agent_exception(
            exc,
            stage=stage,
            operation=operation,
            trace_context=trace_context,
        )
        run_state.setdefault("pipeline_errors", []).append(self._error_payload(normalized))
        self._record_error(normalized, trace_context=trace_context)
        if escalate:
            raise normalized
        return normalized

    def _normalize_agent_exception(
        self,
        exc: Exception,
        *,
        stage: str,
        operation: str,
        trace_context: Optional[Mapping[str, Any]],
    ) -> ObservabilityError:
        return normalize_observability_exception(
            exc,
            stage=stage,
            context={
                "agent_name": self.name,
                "trace_id": self._trace_id(trace_context),
                "operation": operation,
                "service": self._trace_service(trace_context, {}),
            },
            default_error_type=(
                ObservabilityErrorType.INCIDENT_BRIEF_GENERATION_FAILED
                if stage.startswith("incident")
                else None
            ),
            default_severity=(
                ObservabilitySeverity.HIGH if stage.startswith("incident") else None
            ),
        )

    # ------------------------------------------------------------------
    # Introspection and compatibility helpers
    # ------------------------------------------------------------------
    def _build_subsystem(self, subsystem_cls):
        try:
            signature = inspect.signature(subsystem_cls.__init__)
        except (TypeError, ValueError):
            return subsystem_cls()

        kwargs: Dict[str, Any] = {}
        if "memory" in signature.parameters:
            kwargs["memory"] = None
        if "analyzer" in signature.parameters:
            kwargs["analyzer"] = None
        return subsystem_cls(**kwargs)

    def _invoke_supported(self, target: Any, method_name: str, /, **candidate_kwargs: Any) -> Any:
        method = getattr(target, method_name, None)
        if not callable(method):
            return None

        signature = inspect.signature(method)
        accepts_varkw = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        filtered_kwargs: Dict[str, Any] = {}
        for key, value in candidate_kwargs.items():
            if accepts_varkw or key in signature.parameters:
                filtered_kwargs[key] = value
        return method(**filtered_kwargs)

    def _safe_subsystem_report(
        self,
        *,
        target: Any,
        method_name: str,
        stage: str,
        operation: str,
        trace_context: Optional[Mapping[str, Any]],
        run_state: MutableMapping[str, Any],
        default: Mapping[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if not hasattr(target, method_name):
            return dict(default)
        try:
            response = self._invoke_supported(target, method_name, **kwargs)
            return dict(response) if isinstance(response, Mapping) else dict(default)
        except Exception as exc:
            self._capture_stage_error(
                run_state,
                exc,
                stage=stage,
                operation=operation,
                trace_context=trace_context,
            )
            return dict(default)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _append_trace_event(
        self,
        trace_context: Optional[Mapping[str, Any]],
        run_state: MutableMapping[str, Any],
        *,
        event_type: str,
        message: Optional[str],
        severity: str,
        agent_name: Optional[str],
        span_id: Optional[str],
        payload: Optional[Mapping[str, Any]],
        correlation_keys: Optional[Mapping[str, Any]],
        timestamp_ms: Optional[float],
        counter_key: str,
    ) -> None:
        trace_id = self._trace_id(trace_context)
        if not trace_id or not hasattr(self.observability_tracing, "append_event"):
            return
        try:
            self._invoke_supported(
                self.observability_tracing,
                "append_event",
                trace_id=trace_id,
                event_type=event_type,
                message=message,
                severity=severity,
                agent_name=agent_name,
                span_id=span_id,
                payload=self._mapping(payload),
                correlation_keys={str(key): str(value) for key, value in self._mapping(correlation_keys).items()},
                timestamp_ms=timestamp_ms,
            )
            run_state["ingestion_stats"][counter_key] += 1
        except Exception as exc:
            self._capture_stage_error(
                run_state,
                exc,
                stage="tracing.collect",
                operation="append_event",
                trace_context=trace_context,
            )

    def _bounded_records(self, value: Any, limit: int) -> List[Dict[str, Any]]:
        records = self._record_list(value)
        if len(records) > limit:
            records = records[:limit]
        return records

    def _record_list(self, value: Any) -> List[Dict[str, Any]]:
        if value is None:
            return []
        if isinstance(value, Mapping):
            return [dict(value)]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [dict(item) for item in value if isinstance(item, Mapping)]
        return []

    def _pipeline_errors_as_events(self, errors: Optional[Iterable[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        for error in errors or []:
            if not isinstance(error, Mapping):
                continue
            output.append(
                {
                    "event_type": "observability.pipeline_error",
                    "severity": error.get("severity", "warning"),
                    "message": error.get("message"),
                    "payload": {
                        "error_type": error.get("error_type"),
                        "error_code": error.get("error_code"),
                    },
                }
            )
        return output

    def _resolve_trace_final_status(
        self,
        *,
        report: Optional[Mapping[str, Any]],
        fatal_error: Optional[ObservabilityError],
    ) -> str:
        if fatal_error is not None:
            return "error"
        if not report:
            return "ok"
        if self._mapping(report.get("suppression")).get("suppress_duplicate_alert", False):
            return "ok"
        if str(report.get("incident_level") or "info").lower() == "critical":
            return "error"
        return "ok"

    def _resolve_final_incident_level(
        self,
        *,
        report: Optional[Mapping[str, Any]],
        fatal_error: Optional[ObservabilityError],
    ) -> str:
        if fatal_error is not None:
            return getattr(fatal_error, "severity", ObservabilitySeverity.HIGH).to_incident_level() if hasattr(getattr(fatal_error, "severity", None), "to_incident_level") else "critical"
        if not report:
            return "info"
        return str(report.get("incident_level") or "info").lower()

    def _derive_incident_level_from_trace(self, trace_analysis: Mapping[str, Any]) -> str:
        anomaly_count = int(trace_analysis.get("anomaly_count", 0) or 0)
        bottleneck_count = int(trace_analysis.get("bottleneck_count", 0) or 0)
        retry_chain_count = int(trace_analysis.get("retry_chain_count", 0) or 0)
        critical_path_ms = float(trace_analysis.get("critical_path_ms", 0.0) or 0.0)
        total_duration_ms = float(trace_analysis.get("total_duration_ms", 0.0) or 0.0)
        if anomaly_count >= 3 or retry_chain_count >= 2:
            return "critical"
        if anomaly_count >= 1 or bottleneck_count >= 1:
            return "warning"
        if total_duration_ms > 0 and critical_path_ms / max(total_duration_ms, 1.0) > 0.85:
            return "warning"
        return "info"

    def _derive_incident_level_from_errors(self, error_events: Optional[Iterable[Mapping[str, Any]]]) -> str:
        highest = "info"
        for record in error_events or []:
            if not isinstance(record, Mapping):
                continue
            level = str(record.get("severity") or record.get("level") or "warning").lower()
            if _LEVEL_RANK.get(level, 10) > _LEVEL_RANK.get(highest, 10):
                highest = "critical" if level in {"critical", "error"} else "warning"
        return highest

    def _build_error_signature(
        self,
        trace_analysis: Mapping[str, Any],
        error_events: Optional[Iterable[Mapping[str, Any]]],
    ) -> str:
        parts: List[str] = []
        for bottleneck in self._sequence(trace_analysis.get("bottleneck_spans"))[:3]:
            if isinstance(bottleneck, Mapping):
                parts.append(
                    f"{bottleneck.get('agent_name', 'unknown')}:{bottleneck.get('span_id', 'span')}:{bottleneck.get('status', 'ok')}"
                )
        for anomaly in self._sequence(trace_analysis.get("anomalies"))[:3]:
            if isinstance(anomaly, Mapping):
                parts.append(
                    f"{anomaly.get('agent_name', 'unknown')}:{anomaly.get('type', 'anomaly')}:{anomaly.get('severity', 'warning')}"
                )
        for event in (error_events or []):
            if isinstance(event, Mapping):
                parts.append(
                    f"{event.get('agent_name', 'unknown')}:{event.get('event_type', 'event')}:{event.get('severity', 'warning')}"
                )
            if len(parts) >= 6:
                break
        if not parts:
            parts.append("observability_agent:generic:info")
        return " | ".join(parts)

    def _fallback_symptoms(
        self,
        trace_analysis: Mapping[str, Any],
        capacity_summary: Mapping[str, Any],
        performance_summary: Mapping[str, Any],
    ) -> List[str]:
        symptoms: List[str] = []
        if trace_analysis.get("anomaly_count"):
            symptoms.append(f"Trace anomalies detected: {trace_analysis.get('anomaly_count')}")
        if trace_analysis.get("bottleneck_count"):
            symptoms.append(f"Trace bottlenecks detected: {trace_analysis.get('bottleneck_count')}")
        alerts = self._sequence(capacity_summary.get("alerts"))
        if alerts:
            symptoms.append(f"Capacity alerts active: {len(alerts)}")
        latency_subjects = self._sequence(performance_summary.get("latency_subjects"))
        if latency_subjects:
            symptoms.append(f"Latency subjects monitored: {len(latency_subjects)}")
        return symptoms or ["Operational evidence available but incident synthesis ran in fallback mode."]

    def _make_routing_action(
        self,
        *,
        target_role: str,
        target_names: Sequence[str],
        reason: str,
        priority: int,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        selected_name = self._resolve_agent_name(target_names)
        return {
            "target_role": target_role,
            "target_agent": selected_name,
            "available": bool(selected_name),
            "priority": int(priority),
            "reason": reason,
            "context": dict(context or {}),
        }

    def _resolve_agent_name(self, candidates: Sequence[str]) -> Optional[str]:
        if self.agent_factory is None:
            return candidates[0] if candidates else None
        active_agents = getattr(self.agent_factory, "active_agents", {})
        for candidate in candidates:
            if candidate in active_agents:
                return candidate
        if hasattr(self.agent_factory, "registry"):
            registry = getattr(self.agent_factory, "registry")
            for candidate in candidates:
                try:
                    if hasattr(registry, "get") and registry.get(candidate) is not None:
                        return candidate
                except Exception:
                    continue
        return candidates[0] if candidates else None

    def _capacity_indicates_replanning(self, capacity_summary: Mapping[str, Any]) -> bool:
        for alert in self._sequence(capacity_summary.get("alerts")):
            if not isinstance(alert, Mapping):
                continue
            signal = str(alert.get("signal") or "").lower()
            if signal in {"queue_backlog_growth", "drain_ratio_low", "resource_saturation", "resource_stale"}:
                return True
        return False

    def _requires_quality_reassessment(
        self,
        performance_summary: Mapping[str, Any],
        trace_analysis: Mapping[str, Any],
        incident_level: str,
    ) -> bool:
        if incident_level == "critical":
            return True
        if int(trace_analysis.get("retry_chain_count", 0) or 0) > 0:
            return True
        regressions = self._sequence(performance_summary.get("regressions"))
        return any(isinstance(item, Mapping) and str(item.get("level") or "").lower() == "critical" for item in regressions)

    def _detect_observation_start(
        self,
        payload: Mapping[str, Any],
        trace_context: Optional[Mapping[str, Any]],
        report: Mapping[str, Any],
    ) -> float:
        candidates = [
            payload.get("started_at_ms"),
            self._mapping(payload.get("trace")).get("started_at_ms"),
            self._mapping(self._mapping(report.get("trace")).get("analysis")).get("wall_clock_start_ms"),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                return float(candidate)
            except (TypeError, ValueError):
                continue
        return float(report.get("started_at_ms", self._now_ms()))

    def _collect_recommended_actions(self, report: Mapping[str, Any]) -> List[str]:
        actions: List[str] = []
        incident = self._mapping(report.get("incident"))
        for runbook in self._sequence(incident.get("runbooks")):
            if isinstance(runbook, Mapping):
                for action in self._sequence(runbook.get("actions")):
                    text = str(action).strip()
                    if text and text not in actions:
                        actions.append(text)
        brief = self._mapping(incident.get("brief"))
        for runbook in self._sequence(brief.get("recommended_runbooks")):
            if isinstance(runbook, Mapping):
                for action in self._sequence(runbook.get("actions")):
                    text = str(action).strip()
                    if text and text not in actions:
                        actions.append(text)
        for route in self._sequence(report.get("routing")):
            if isinstance(route, Mapping):
                reason = str(route.get("reason") or "").strip()
                if reason and reason not in actions:
                    actions.append(reason)
        return actions[: self.max_related_agents * 2]

    def _extract_latency_p95(self, performance_summary: Mapping[str, Any]) -> Optional[float]:
        current_trace = self._mapping(performance_summary.get("current_trace"))
        if current_trace:
            per_agent = self._mapping(current_trace.get("per_agent_stats"))
            values: List[float] = []
            for stats in per_agent.values():
                if isinstance(stats, Mapping):
                    try:
                        values.append(float(stats.get("p95_duration_ms", 0.0) or 0.0))
                    except (TypeError, ValueError):
                        continue
            if values:
                return max(values)
        subjects = self._mapping(performance_summary.get("latency_subjects"))
        values = []
        for subject_stats in subjects.values():
            if isinstance(subject_stats, Mapping):
                try:
                    values.append(float(subject_stats.get("p95_duration_ms", 0.0) or 0.0))
                except (TypeError, ValueError):
                    continue
        return max(values) if values else None

    def _error_payload(self, error: ObservabilityError) -> Dict[str, Any]:
        payload = error.to_dict() if hasattr(error, "to_dict") else {"message": str(error)}
        try:
            reported = error.report() if hasattr(error, "report") else None
            if isinstance(reported, Mapping):
                payload = dict(reported)
        except Exception:
            pass
        return dict(payload)

    def _serialize_exception(self, exc: Any) -> Dict[str, Any]:
        if exc is None:
            return {}
        if isinstance(exc, ObservabilityError):
            return self._error_payload(exc)
        return {"type": type(exc).__name__, "message": str(exc)}

    def _build_synthetic_trace_context(
        self,
        payload: Mapping[str, Any],
        *,
        agent_name: Optional[str] = None,
        operation_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        trace_info = self._mapping(payload.get("trace"))
        return {
            "trace_id": str(trace_info.get("trace_id") or self._new_trace_id()),
            "root_span_id": None,
            "agent_name": str(agent_name or trace_info.get("agent_name") or payload.get("agent_name") or self.name),
            "operation_name": str(operation_name or trace_info.get("operation_name") or payload.get("operation_name") or self.default_trace_operation),
            "service": str(trace_info.get("service") or payload.get("service") or self.default_service),
            "managed_by_tracer": False,
        }

    def _trace_id(self, trace_context: Optional[Mapping[str, Any]]) -> Optional[str]:
        if not trace_context:
            return None
        value = trace_context.get("trace_id")
        return str(value) if value else None

    def _trace_agent_name(self, trace_context: Optional[Mapping[str, Any]]) -> str:
        if trace_context and trace_context.get("agent_name"):
            return str(trace_context["agent_name"])
        return self.name

    def _trace_operation_name(self, trace_context: Optional[Mapping[str, Any]]) -> Optional[str]:
        if trace_context and trace_context.get("operation_name"):
            return str(trace_context["operation_name"])
        return None

    def _trace_service(self, trace_context: Optional[Mapping[str, Any]], payload: Mapping[str, Any]) -> str:
        if trace_context and trace_context.get("service"):
            return str(trace_context["service"])
        return str(payload.get("service") or self.default_service)

    def _new_trace_id(self) -> str:
        return f"obs-trace-{int(self._now_ms())}"

    def _rolling_average(self, current_average: Any, new_value: float, count: int) -> float:
        try:
            current = float(current_average)
        except (TypeError, ValueError):
            current = 0.0
        if count <= 1:
            return float(new_value)
        return ((current * (count - 1)) + float(new_value)) / float(count)

    def _sm_get(self, key: str, *, default: Any = None) -> Any:
        return self.shared_memory.get(key, default=default)

    @staticmethod
    def _mapping(value: Any) -> Dict[str, Any]:
        return dict(value) if isinstance(value, Mapping) else {}

    @staticmethod
    def _sequence(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return list(value)
        return []

    @staticmethod
    def _coerce_name_list(value: Any) -> List[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            return [str(item) for item in value if str(item).strip()]
        return []

    @staticmethod
    def _now_ms() -> float:
        return time.time() * 1000.0


if __name__ == "__main__":
    print("\n=== Running Observability Agent  ===\n")
    printer.status("TEST", "Observability Agent  initialized", "info")
    from .collaborative.shared_memory import SharedMemory
    from .agent_factory import AgentFactory

    shared_memory = SharedMemory()
    factory = AgentFactory()
    agent = ObservabilityAgent(shared_memory=shared_memory, agent_factory=factory)

    payload = {
        "task_name": "multi_agent_runtime_observation",
        "service": "slai",
        "source": "runtime_test",
        "trace": {
            "task_name": "multi_agent_runtime_observation",
            "agent_name": "planning_agent",
            "operation_name": "coordinate_request",
            "metadata": {"request_id": "req-obs-001", "tenant": "internal"},
        },
        "queues": [
            {"queue_name": "planner_queue", "depth": 24, "inflow_per_sec": 8.0, "outflow_per_sec": 6.5},
            {"queue_name": "handler_queue", "depth": 11, "inflow_per_sec": 3.0, "outflow_per_sec": 3.2},
        ],
        "resources": [
            {"resource_name": "node-a-cpu", "resource_type": "cpu", "utilization_pct": 82.0, "used": 82, "capacity": 100},
            {"resource_name": "node-a-memory", "resource_type": "memory", "utilization_pct": 76.5, "used": 61.2, "capacity": 80.0},
        ],
        "latencies": [
            {"subject": "slai", "duration_ms": 1420.0, "status": "ok"},
            {"subject": "planner_agent", "duration_ms": 680.0, "status": "ok"},
            {"subject": "handler_agent", "duration_ms": 930.0, "status": "retry"},
        ],
        "throughput": [
            {"subject": "slai", "count": 18, "success_count": 16, "failure_count": 2},
        ],
        "spans": [
            {
                "span_id": "span-root",
                "agent_name": "planning_agent",
                "operation_name": "coordinate_request",
                "start_ms": 0.0,
                "end_ms": 1600.0,
                "status": "ok",
            },
            {
                "span_id": "span-retrieval",
                "parent_span_id": "span-root",
                "agent_name": "knowledge_agent",
                "operation_name": "retrieve_context",
                "start_ms": 120.0,
                "end_ms": 740.0,
                "status": "retry",
                "attempt": 2,
                "metadata": {"retry": True},
            },
            {
                "span_id": "span-handler",
                "parent_span_id": "span-root",
                "agent_name": "handler_agent",
                "operation_name": "fallback_route",
                "start_ms": 760.0,
                "end_ms": 1510.0,
                "status": "timeout",
            },
        ],
        "events": [
            {
                "event_type": "queue_wait",
                "severity": "warning",
                "agent_name": "knowledge_agent",
                "message": "Knowledge agent waited for retrieval dispatch.",
                "payload": {"queue_depth": 24},
            }
        ],
        "state_transitions": [
            {"from_state": "planned", "to_state": "executing", "agent_name": "planning_agent", "severity": "info"},
            {"from_state": "executing", "to_state": "degraded", "agent_name": "handler_agent", "severity": "warning", "reason": "fallback path engaged"},
        ],
        "logs": [
            {"level": "warning", "agent_name": "handler_agent", "message": "Timeout pressure detected on fallback route."},
        ],
        "alerts": [
            {"event_type": "latency_breach", "severity": "warning", "agent_name": "observability_agent", "message": "p95 latency warning exceeded."},
        ],
        "error_events": [
            {"event_type": "dependency_timeout", "severity": "critical", "agent_name": "handler_agent", "message": "Downstream timeout observed."},
        ],
        "latency_objectives": [
            {"service": "slai", "subject": "slai", "slo_name": "response_latency_p95_ms", "target_ms": 1500.0, "percentile": 95},
        ],
    }

    result = agent.perform_task(payload)
    printer.pretty("OBSERVABILITY_REPORT", result, "info")
    printer.pretty("HEALTH", shared_memory.get(agent.sm_keys["health"], default={}), "info")
    printer.pretty("KPIS", shared_memory.get(agent.sm_keys["kpis"], default={}), "info")

    print("\n=== Test ran successfully ===\n")
