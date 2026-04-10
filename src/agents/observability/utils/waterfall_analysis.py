"""
It gives your Observability subsystem a dedicated place to transform raw spans into operational signals
(critical path, bottlenecks, anomalies, retry patterns),
which is exactly what you need for incident triage and performance debugging.

This script should own these responsibilities:
- Span normalization from mixed event payloads into one canonical structure.
- Critical-path identification to show where end-to-end latency is actually spent.
- Per-agent runtime accounting to reveal heavy/slow subsystems.
- Retry waterfall detection to expose silent cost/latency loops.
- Bottleneck ranking (threshold-based + later percentile-based).
- Anomaly extraction (timeouts, invalid timings, repeated error spans).
- Summary serialization for dashboards, alert routing, and postmortems.
- All of these are now represented in the module design and methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .config_loader import get_config_section, load_global_config
from .observability_error import normalize_observability_exception


_STATUS_ALIASES = {
    "ok": "ok",
    "success": "ok",
    "succeeded": "ok",
    "completed": "ok",
    "complete": "ok",
    "done": "ok",
    "error": "error",
    "failed": "error",
    "failure": "error",
    "exception": "error",
    "timeout": "timeout",
    "timed_out": "timeout",
    "deadline_exceeded": "timeout",
    "retry": "retry",
    "retried": "retry",
    "retrying": "retry",
    "queued": "queued",
    "queue": "queued",
    "running": "running",
    "in_progress": "running",
    "cancelled": "cancelled",
    "canceled": "cancelled",
    "skipped": "skipped",
}

_ERROR_STATUSES = {"error", "timeout"}
_TIMEOUT_STATUSES = {"timeout"}
_NON_TERMINAL_STATUSES = {"running", "queued"}


@dataclass
class WaterfallSpan:
    """Canonical observability span used by the waterfall analyzer."""

    span_id: str
    agent_name: str
    start_ms: float
    end_ms: float
    status: str = "ok"
    parent_span_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    operation_name: Optional[str] = None
    attempt: int = 1
    exclusive_duration_ms: float = 0.0

    @property
    def duration_ms(self) -> float:
        return max(0.0, self.end_ms - self.start_ms)

    @property
    def timing_valid(self) -> bool:
        return self.end_ms >= self.start_ms

    @property
    def is_error(self) -> bool:
        return self.status in _ERROR_STATUSES

    @property
    def is_timeout(self) -> bool:
        return self.status in _TIMEOUT_STATUSES

    @property
    def is_retry(self) -> bool:
        retry_flag = self.metadata.get("retry", False)
        return bool(retry_flag) or self.attempt > 1 or self.status == "retry"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "agent_name": self.agent_name,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "exclusive_duration_ms": self.exclusive_duration_ms,
            "status": self.status,
            "parent_span_id": self.parent_span_id,
            "trace_id": self.trace_id,
            "operation_name": self.operation_name,
            "attempt": self.attempt,
            "metadata": self.metadata,
        }


@dataclass
class WaterfallRetryChain:
    agent_name: str
    span_ids: List[str]
    attempts: List[int]
    statuses: List[str]
    parent_span_id: Optional[str]
    operation_name: Optional[str]
    trace_id: Optional[str]
    start_ms: float
    end_ms: float
    total_duration_ms: float
    retry_overhead_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "span_ids": self.span_ids,
            "attempts": self.attempts,
            "attempt_count": len(self.span_ids),
            "statuses": self.statuses,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "trace_id": self.trace_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "total_duration_ms": self.total_duration_ms,
            "retry_overhead_ms": self.retry_overhead_ms,
        }


@dataclass
class WaterfallAnomaly:
    anomaly_type: str
    severity: str
    description: str
    span_id: Optional[str] = None
    agent_name: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "type": self.anomaly_type,
            "severity": self.severity,
            "description": self.description,
            "details": self.details,
        }
        if self.span_id is not None:
            payload["span_id"] = self.span_id
        if self.agent_name is not None:
            payload["agent_name"] = self.agent_name
        return payload


@dataclass
class WaterfallAgentStats:
    agent_name: str
    span_count: int
    total_duration_ms: float
    exclusive_duration_ms: float
    average_duration_ms: float
    max_duration_ms: float
    p95_duration_ms: float
    share_of_total_duration: float
    error_count: int
    timeout_count: int
    retry_count: int
    root_span_count: int
    leaf_span_count: int
    critical_path_duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "span_count": self.span_count,
            "total_duration_ms": self.total_duration_ms,
            "exclusive_duration_ms": self.exclusive_duration_ms,
            "average_duration_ms": self.average_duration_ms,
            "max_duration_ms": self.max_duration_ms,
            "p95_duration_ms": self.p95_duration_ms,
            "share_of_total_duration": self.share_of_total_duration,
            "error_count": self.error_count,
            "timeout_count": self.timeout_count,
            "retry_count": self.retry_count,
            "root_span_count": self.root_span_count,
            "leaf_span_count": self.leaf_span_count,
            "critical_path_duration_ms": self.critical_path_duration_ms,
        }


@dataclass
class WaterfallBottleneck:
    rank: int
    span_id: str
    agent_name: str
    duration_ms: float
    exclusive_duration_ms: float
    ratio_of_total: float
    exclusive_ratio_of_total: float
    percentile_rank: float
    on_critical_path: bool
    status: str
    reason: List[str]
    operation_name: Optional[str] = None
    parent_span_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "span_id": self.span_id,
            "agent_name": self.agent_name,
            "duration_ms": self.duration_ms,
            "exclusive_duration_ms": self.exclusive_duration_ms,
            "ratio": self.ratio_of_total,
            "exclusive_ratio": self.exclusive_ratio_of_total,
            "percentile_rank": self.percentile_rank,
            "on_critical_path": self.on_critical_path,
            "status": self.status,
            "reason": self.reason,
            "operation_name": self.operation_name,
            "parent_span_id": self.parent_span_id,
        }


@dataclass
class WaterfallReport:
    total_duration_ms: float
    critical_path_ms: float
    critical_path_span_ids: List[str]
    per_agent_duration_ms: Dict[str, float]
    retry_chains: List[List[str]]
    bottleneck_spans: List[Dict[str, Any]]
    anomalies: List[Dict[str, Any]]
    trace_id: Optional[str] = None
    span_count: int = 0
    wall_clock_start_ms: float = 0.0
    wall_clock_end_ms: float = 0.0
    critical_path_agent_names: List[str] = field(default_factory=list)
    critical_path_exclusive_ms: Dict[str, float] = field(default_factory=dict)
    per_agent_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    retry_chain_details: List[Dict[str, Any]] = field(default_factory=list)
    status_counts: Dict[str, int] = field(default_factory=dict)
    unique_agents: List[str] = field(default_factory=list)
    root_span_ids: List[str] = field(default_factory=list)
    orphan_span_ids: List[str] = field(default_factory=list)
    duplicate_span_ids: List[str] = field(default_factory=list)
    parse_warnings: List[Dict[str, Any]] = field(default_factory=list)
    max_concurrency: int = 0
    avg_concurrency: float = 0.0
    total_exclusive_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_count": self.span_count,
            "wall_clock_start_ms": self.wall_clock_start_ms,
            "wall_clock_end_ms": self.wall_clock_end_ms,
            "total_duration_ms": self.total_duration_ms,
            "total_exclusive_duration_ms": self.total_exclusive_duration_ms,
            "critical_path_ms": self.critical_path_ms,
            "critical_path_span_ids": self.critical_path_span_ids,
            "critical_path_agent_names": self.critical_path_agent_names,
            "critical_path_exclusive_ms": self.critical_path_exclusive_ms,
            "per_agent_duration_ms": self.per_agent_duration_ms,
            "per_agent_stats": self.per_agent_stats,
            "status_counts": self.status_counts,
            "retry_chains": self.retry_chains,
            "retry_chain_details": self.retry_chain_details,
            "bottleneck_spans": self.bottleneck_spans,
            "anomalies": self.anomalies,
            "unique_agents": self.unique_agents,
            "root_span_ids": self.root_span_ids,
            "orphan_span_ids": self.orphan_span_ids,
            "duplicate_span_ids": self.duplicate_span_ids,
            "parse_warnings": self.parse_warnings,
            "max_concurrency": self.max_concurrency,
            "avg_concurrency": self.avg_concurrency,
        }


class WaterfallAnalyzer:
    """
    Transform raw execution spans into operationally useful diagnostics.

    Expected minimum information per span:
      - span_id
      - agent_name
      - start_ms
      - end_ms

    Supported optional fields:
      - status
      - parent_span_id
      - metadata
      - trace_id
      - operation_name
      - attempt

    The analyzer is intentionally optimized for post-execution observability,
    incident triage, regression analysis, and dashboard summarization rather
    than for continuously mutating realtime graphs.
    """

    def __init__(
        self,
        bottleneck_threshold_ratio: Optional[float] = None,
        timeout_ms: Optional[float] = None,
    ) -> None:
        self.config = load_global_config()
        self.analysis_config = get_config_section("waterfall_analysis")

        self.bottleneck_threshold_ratio = float(
            bottleneck_threshold_ratio
            if bottleneck_threshold_ratio is not None
            else self.analysis_config.get("bottleneck_threshold_ratio", 0.25)
        )
        self.timeout_ms = float(
            timeout_ms if timeout_ms is not None else self.analysis_config.get("timeout_ms", 30_000.0)
        )
        self.percentile_threshold = float(self.analysis_config.get("percentile_threshold", 0.95))
        self.max_ranked_bottlenecks = int(self.analysis_config.get("max_ranked_bottlenecks", 10))
        self.min_retry_chain_length = int(self.analysis_config.get("min_retry_chain_length", 2))
        self.zero_duration_epsilon_ms = float(self.analysis_config.get("zero_duration_epsilon_ms", 0.001))
        self.repeated_error_threshold = int(self.analysis_config.get("repeated_error_threshold", 3))
        self.treat_orphans_as_roots = bool(self.analysis_config.get("treat_orphans_as_roots", True))
        self.require_parent_child_time_alignment = bool(
            self.analysis_config.get("require_parent_child_time_alignment", True)
        )

    def parse_spans(self, spans: Iterable[Dict[str, Any]]) -> List[WaterfallSpan]:
        parsed, _ = self._parse_spans_with_warnings(spans)
        return parsed

    def analyze(self, spans: Iterable[Dict[str, Any]]) -> WaterfallReport:
        try:
            parsed_spans, parse_warnings = self._parse_spans_with_warnings(spans)
            if not parsed_spans:
                return WaterfallReport(
                    total_duration_ms=0.0,
                    critical_path_ms=0.0,
                    critical_path_span_ids=[],
                    per_agent_duration_ms={},
                    retry_chains=[],
                    bottleneck_spans=[],
                    anomalies=[],
                    parse_warnings=parse_warnings,
                )

            unique_spans, duplicate_span_ids = self._deduplicate_spans(parsed_spans)
            by_id = {span.span_id: span for span in unique_spans}
            children_map, root_span_ids, orphan_span_ids = self._build_topology(unique_spans, by_id)
            self._compute_exclusive_durations(unique_spans, children_map, by_id)

            sorted_unique_spans = sorted(unique_spans, key=self._span_sort_key)
            wall_clock_start_ms = min(span.start_ms for span in sorted_unique_spans)
            wall_clock_end_ms = max(span.end_ms for span in sorted_unique_spans)
            total_duration_ms = max(0.0, wall_clock_end_ms - wall_clock_start_ms)
            total_exclusive_duration_ms = sum(span.exclusive_duration_ms for span in sorted_unique_spans)

            analysis_roots = list(root_span_ids)
            if self.treat_orphans_as_roots:
                analysis_roots.extend([span_id for span_id in orphan_span_ids if span_id not in analysis_roots])
            if not analysis_roots:
                analysis_roots = [span.span_id for span in sorted_unique_spans]

            critical_path_ms, critical_path_span_ids = self._critical_path(analysis_roots, children_map, by_id)
            critical_path_agent_names, critical_path_exclusive_ms = self._critical_path_breakdown(
                critical_path_span_ids,
                by_id,
            )

            per_agent_duration_ms = self._per_agent_duration(sorted_unique_spans)
            status_counts = self._status_counts(sorted_unique_spans)
            retry_chain_objects = self._retry_chains(sorted_unique_spans)
            retry_chains = [chain.span_ids for chain in retry_chain_objects]
            per_agent_stats = self._per_agent_stats(
                sorted_unique_spans,
                children_map,
                root_span_ids,
                critical_path_span_ids,
                total_duration_ms,
            )
            bottleneck_spans = self._detect_bottlenecks(
                sorted_unique_spans,
                total_duration_ms,
                critical_path_span_ids,
            )
            anomalies = self._detect_anomalies(
                sorted_unique_spans,
                children_map,
                by_id,
                duplicate_span_ids,
                orphan_span_ids,
                parse_warnings,
            )
            max_concurrency, avg_concurrency = self._concurrency_profile(sorted_unique_spans, total_duration_ms)
            trace_id = self._select_trace_id(sorted_unique_spans)

            return WaterfallReport(
                trace_id=trace_id,
                span_count=len(sorted_unique_spans),
                wall_clock_start_ms=wall_clock_start_ms,
                wall_clock_end_ms=wall_clock_end_ms,
                total_duration_ms=total_duration_ms,
                total_exclusive_duration_ms=total_exclusive_duration_ms,
                critical_path_ms=critical_path_ms,
                critical_path_span_ids=critical_path_span_ids,
                critical_path_agent_names=critical_path_agent_names,
                critical_path_exclusive_ms=critical_path_exclusive_ms,
                per_agent_duration_ms=per_agent_duration_ms,
                per_agent_stats=per_agent_stats,
                status_counts=status_counts,
                retry_chains=retry_chains,
                retry_chain_details=[chain.to_dict() for chain in retry_chain_objects],
                bottleneck_spans=[bottleneck.to_dict() for bottleneck in bottleneck_spans],
                anomalies=[anomaly.to_dict() for anomaly in anomalies],
                unique_agents=sorted({span.agent_name for span in sorted_unique_spans}),
                root_span_ids=root_span_ids,
                orphan_span_ids=orphan_span_ids,
                duplicate_span_ids=duplicate_span_ids,
                parse_warnings=parse_warnings,
                max_concurrency=max_concurrency,
                avg_concurrency=avg_concurrency,
            )
        except Exception as exc:
            raise normalize_observability_exception(
                exc,
                stage="waterfall_analysis",
                context={"component": "waterfall_analysis"},
            ) from exc

    def _parse_spans_with_warnings(
        self,
        spans: Iterable[Dict[str, Any]],
    ) -> Tuple[List[WaterfallSpan], List[Dict[str, Any]]]:
        parsed: List[WaterfallSpan] = []
        warnings: List[Dict[str, Any]] = []

        for index, raw in enumerate(spans):
            if isinstance(raw, WaterfallSpan):
                raw.status = self._normalize_status(raw.status)
                raw.metadata = self._coerce_mapping(raw.metadata)
                raw.attempt = max(1, self._safe_int(raw.attempt, 1))
                parsed.append(raw)
                continue

            if not isinstance(raw, Mapping):
                warnings.append(
                    {
                        "type": "invalid_span_payload",
                        "index": index,
                        "reason": "span payload is not a mapping",
                    }
                )
                continue

            span_id = self._extract_string(raw, "span_id", "id", "spanId")
            agent_name = self._extract_string(raw, "agent_name", "agent", "component", "service")
            start_ms = self._extract_float(raw, "start_ms", "start", "start_time_ms", "started_ms")
            end_ms = self._extract_float(raw, "end_ms", "end", "end_time_ms", "finished_ms")

            missing = []
            if span_id is None:
                missing.append("span_id")
            if agent_name is None:
                missing.append("agent_name")
            if start_ms is None:
                missing.append("start_ms")
            if end_ms is None:
                missing.append("end_ms")

            if missing:
                warnings.append(
                    {
                        "type": "invalid_span_payload",
                        "index": index,
                        "reason": "missing required fields",
                        "missing_fields": missing,
                    }
                )
                continue

            metadata = self._coerce_mapping(raw.get("metadata", raw.get("tags", raw.get("context", {}))))
            operation_name = self._extract_string(raw, "operation_name", "operation", "name", "step")
            status = self._normalize_status(
                self._extract_string(raw, "status", "state", "outcome") or metadata.get("status", "ok")
            )
            attempt = max(
                1,
                self._safe_int(
                    raw.get("attempt", metadata.get("attempt", metadata.get("retry_attempt", 1))),
                    1,
                ),
            )
            trace_id = self._extract_string(raw, "trace_id", "traceId") or self._safe_str(metadata.get("trace_id"))
            parent_span_id = self._extract_string(raw, "parent_span_id", "parent_id", "parentSpanId")

            parsed.append(
                WaterfallSpan(
                    span_id=span_id,
                    agent_name=agent_name,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    status=status,
                    parent_span_id=parent_span_id,
                    metadata=metadata,
                    trace_id=trace_id,
                    operation_name=operation_name,
                    attempt=attempt,
                )
            )

        parsed.sort(key=self._span_sort_key)
        return parsed, warnings

    def _deduplicate_spans(
        self,
        spans: Sequence[WaterfallSpan],
    ) -> Tuple[List[WaterfallSpan], List[str]]:
        seen: Dict[str, WaterfallSpan] = {}
        duplicates: List[str] = []

        for span in spans:
            if span.span_id in seen:
                duplicates.append(span.span_id)
                continue
            seen[span.span_id] = span

        return list(seen.values()), duplicates

    def _build_topology(
        self,
        spans: Sequence[WaterfallSpan],
        by_id: Mapping[str, WaterfallSpan],
    ) -> Tuple[Dict[str, List[str]], List[str], List[str]]:
        children_map: Dict[str, List[str]] = {span.span_id: [] for span in spans}
        root_span_ids: List[str] = []
        orphan_span_ids: List[str] = []

        for span in spans:
            if not span.parent_span_id:
                root_span_ids.append(span.span_id)
                continue

            if span.parent_span_id not in by_id:
                orphan_span_ids.append(span.span_id)
                continue

            children_map.setdefault(span.parent_span_id, []).append(span.span_id)

        for child_ids in children_map.values():
            child_ids.sort(key=lambda span_id: self._span_sort_key(by_id[span_id]))

        return children_map, root_span_ids, orphan_span_ids

    def _compute_exclusive_durations(
        self,
        spans: Sequence[WaterfallSpan],
        children_map: Mapping[str, List[str]],
        by_id: Mapping[str, WaterfallSpan],
    ) -> None:
        for span in spans:
            child_intervals: List[Tuple[float, float]] = []
            for child_id in children_map.get(span.span_id, []):
                child = by_id[child_id]
                overlap_start = max(span.start_ms, child.start_ms)
                overlap_end = min(span.end_ms, child.end_ms)
                if overlap_end > overlap_start:
                    child_intervals.append((overlap_start, overlap_end))
            covered_ms = self._merge_intervals_total(child_intervals)
            span.exclusive_duration_ms = max(0.0, span.duration_ms - covered_ms)

    def _critical_path(
        self,
        root_span_ids: Sequence[str],
        children_map: Mapping[str, List[str]],
        by_id: Mapping[str, WaterfallSpan],
    ) -> Tuple[float, List[str]]:
        memo: Dict[str, Tuple[float, List[str]]] = {}

        def visit(span_id: str) -> Tuple[float, List[str]]:
            if span_id in memo:
                return memo[span_id]

            span = by_id[span_id]
            best_child_ms = 0.0
            best_child_path: List[str] = []

            for child_id in children_map.get(span_id, []):
                child_ms, child_path = visit(child_id)
                if child_ms > best_child_ms:
                    best_child_ms = child_ms
                    best_child_path = child_path

            result = (span.exclusive_duration_ms + best_child_ms, [span_id, *best_child_path])
            memo[span_id] = result
            return result

        best_ms = 0.0
        best_path: List[str] = []
        for root_id in root_span_ids:
            if root_id not in by_id:
                continue
            path_ms, path_ids = visit(root_id)
            if path_ms > best_ms:
                best_ms = path_ms
                best_path = path_ids

        return best_ms, best_path

    def _critical_path_breakdown(
        self,
        critical_path_span_ids: Sequence[str],
        by_id: Mapping[str, WaterfallSpan],
    ) -> Tuple[List[str], Dict[str, float]]:
        agent_names: List[str] = []
        exclusive_by_agent: Dict[str, float] = {}

        for span_id in critical_path_span_ids:
            span = by_id.get(span_id)
            if span is None:
                continue
            agent_names.append(span.agent_name)
            exclusive_by_agent[span.agent_name] = exclusive_by_agent.get(span.agent_name, 0.0) + span.exclusive_duration_ms

        return agent_names, exclusive_by_agent

    def _per_agent_duration(self, spans: Sequence[WaterfallSpan]) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        for span in spans:
            totals[span.agent_name] = totals.get(span.agent_name, 0.0) + span.duration_ms
        return totals

    def _status_counts(self, spans: Sequence[WaterfallSpan]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for span in spans:
            counts[span.status] = counts.get(span.status, 0) + 1
        return counts

    def _retry_chains(self, spans: Sequence[WaterfallSpan]) -> List[WaterfallRetryChain]:
        grouped: Dict[Tuple[str, Optional[str], Optional[str], Optional[str]], List[WaterfallSpan]] = {}

        for span in spans:
            key = (
                span.agent_name,
                span.parent_span_id,
                span.operation_name,
                span.trace_id,
            )
            grouped.setdefault(key, []).append(span)

        chains: List[WaterfallRetryChain] = []
        for (agent_name, parent_span_id, operation_name, trace_id), group in grouped.items():
            group = sorted(group, key=self._span_sort_key)
            has_retry_signal = any(span.is_retry for span in group)
            if len(group) < self.min_retry_chain_length and not has_retry_signal:
                continue

            attempts = [span.attempt for span in group]
            statuses = [span.status for span in group]
            total_duration_ms = sum(span.duration_ms for span in group)
            retry_overhead_ms = total_duration_ms - group[-1].duration_ms if len(group) > 1 else 0.0

            chains.append(
                WaterfallRetryChain(
                    agent_name=agent_name,
                    span_ids=[span.span_id for span in group],
                    attempts=attempts,
                    statuses=statuses,
                    parent_span_id=parent_span_id,
                    operation_name=operation_name,
                    trace_id=trace_id,
                    start_ms=min(span.start_ms for span in group),
                    end_ms=max(span.end_ms for span in group),
                    total_duration_ms=total_duration_ms,
                    retry_overhead_ms=max(0.0, retry_overhead_ms),
                )
            )

        chains.sort(key=lambda chain: (-chain.retry_overhead_ms, -chain.total_duration_ms, chain.agent_name))
        return chains

    def _per_agent_stats(
        self,
        spans: Sequence[WaterfallSpan],
        children_map: Mapping[str, List[str]],
        root_span_ids: Sequence[str],
        critical_path_span_ids: Sequence[str],
        total_duration_ms: float,
    ) -> Dict[str, Dict[str, Any]]:
        spans_by_agent: Dict[str, List[WaterfallSpan]] = {}
        root_span_set = set(root_span_ids)
        critical_path_set = set(critical_path_span_ids)

        for span in spans:
            spans_by_agent.setdefault(span.agent_name, []).append(span)

        results: Dict[str, Dict[str, Any]] = {}
        for agent_name, agent_spans in spans_by_agent.items():
            durations = [span.duration_ms for span in agent_spans]
            total_duration = sum(durations)
            exclusive_duration = sum(span.exclusive_duration_ms for span in agent_spans)
            critical_path_duration = sum(
                span.exclusive_duration_ms for span in agent_spans if span.span_id in critical_path_set
            )
            leaf_count = sum(1 for span in agent_spans if not children_map.get(span.span_id))
            stats = WaterfallAgentStats(
                agent_name=agent_name,
                span_count=len(agent_spans),
                total_duration_ms=total_duration,
                exclusive_duration_ms=exclusive_duration,
                average_duration_ms=(total_duration / len(agent_spans)) if agent_spans else 0.0,
                max_duration_ms=max(durations) if durations else 0.0,
                p95_duration_ms=self._percentile(durations, 0.95),
                share_of_total_duration=(total_duration / total_duration_ms) if total_duration_ms else 0.0,
                error_count=sum(1 for span in agent_spans if span.is_error),
                timeout_count=sum(1 for span in agent_spans if span.is_timeout),
                retry_count=sum(1 for span in agent_spans if span.is_retry),
                root_span_count=sum(1 for span in agent_spans if span.span_id in root_span_set),
                leaf_span_count=leaf_count,
                critical_path_duration_ms=critical_path_duration,
            )
            results[agent_name] = stats.to_dict()

        return dict(sorted(results.items(), key=lambda item: (-item[1]["exclusive_duration_ms"], item[0])))

    def _detect_bottlenecks(
        self,
        spans: Sequence[WaterfallSpan],
        total_duration_ms: float,
        critical_path_span_ids: Sequence[str],
    ) -> List[WaterfallBottleneck]:
        if not spans:
            return []

        critical_path_set = set(critical_path_span_ids)
        duration_values = sorted(span.duration_ms for span in spans)
        threshold_duration = total_duration_ms * self.bottleneck_threshold_ratio
        percentile_cutoff = self._percentile(duration_values, self.percentile_threshold)

        candidates: List[Tuple[float, WaterfallSpan, List[str], float]] = []
        for span in spans:
            reasons: List[str] = []
            if total_duration_ms and span.duration_ms >= threshold_duration:
                reasons.append("duration_ratio_threshold")
            if span.duration_ms >= percentile_cutoff:
                reasons.append("percentile_threshold")
            if span.span_id in critical_path_set and span.exclusive_duration_ms > 0:
                reasons.append("critical_path_exclusive_contribution")
            if span.is_timeout:
                reasons.append("timeout_span")
            elif span.is_error:
                reasons.append("error_path_span")

            if not reasons:
                continue

            percentile_rank = self._percentile_rank(duration_values, span.duration_ms)
            score = (
                span.exclusive_duration_ms,
                1.0 if span.span_id in critical_path_set else 0.0,
                span.duration_ms,
                percentile_rank,
            )
            candidates.append((score, span, reasons, percentile_rank))

        candidates.sort(key=lambda item: item[0], reverse=True)

        bottlenecks: List[WaterfallBottleneck] = []
        for rank, (_, span, reasons, percentile_rank) in enumerate(
            candidates[: self.max_ranked_bottlenecks],
            start=1,
        ):
            bottlenecks.append(
                WaterfallBottleneck(
                    rank=rank,
                    span_id=span.span_id,
                    agent_name=span.agent_name,
                    duration_ms=span.duration_ms,
                    exclusive_duration_ms=span.exclusive_duration_ms,
                    ratio_of_total=(span.duration_ms / total_duration_ms) if total_duration_ms else 0.0,
                    exclusive_ratio_of_total=(span.exclusive_duration_ms / total_duration_ms) if total_duration_ms else 0.0,
                    percentile_rank=percentile_rank,
                    on_critical_path=span.span_id in critical_path_set,
                    status=span.status,
                    reason=reasons,
                    operation_name=span.operation_name,
                    parent_span_id=span.parent_span_id,
                )
            )

        return bottlenecks

    def _detect_anomalies(
        self,
        spans: Sequence[WaterfallSpan],
        children_map: Mapping[str, List[str]],
        by_id: Mapping[str, WaterfallSpan],
        duplicate_span_ids: Sequence[str],
        orphan_span_ids: Sequence[str],
        parse_warnings: Sequence[Dict[str, Any]],
    ) -> List[WaterfallAnomaly]:
        anomalies: List[WaterfallAnomaly] = []

        for warning in parse_warnings:
            anomalies.append(
                WaterfallAnomaly(
                    anomaly_type="invalid_span_payload",
                    severity="medium",
                    description="A span payload could not be normalized into the canonical waterfall model.",
                    details=dict(warning),
                )
            )

        for duplicate_span_id in duplicate_span_ids:
            anomalies.append(
                WaterfallAnomaly(
                    anomaly_type="duplicate_span_id",
                    severity="high",
                    description="Duplicate span IDs were detected and later duplicates were excluded from analysis.",
                    span_id=duplicate_span_id,
                    details={"span_id": duplicate_span_id},
                )
            )

        for orphan_span_id in orphan_span_ids:
            span = by_id.get(orphan_span_id)
            anomalies.append(
                WaterfallAnomaly(
                    anomaly_type="orphan_span",
                    severity="medium",
                    description="Span references a parent span that is missing from the analyzed waterfall.",
                    span_id=orphan_span_id,
                    agent_name=span.agent_name if span else None,
                    details={
                        "parent_span_id": span.parent_span_id if span else None,
                    },
                )
            )

        error_fingerprint_counts: Dict[Tuple[str, Optional[str], str], int] = {}
        for span in spans:
            if not span.timing_valid:
                anomalies.append(
                    WaterfallAnomaly(
                        anomaly_type="negative_duration",
                        severity="critical",
                        description="Span end timestamp is earlier than its start timestamp.",
                        span_id=span.span_id,
                        agent_name=span.agent_name,
                        details={"start_ms": span.start_ms, "end_ms": span.end_ms},
                    )
                )

            if span.duration_ms <= self.zero_duration_epsilon_ms:
                anomalies.append(
                    WaterfallAnomaly(
                        anomaly_type="zero_duration",
                        severity="low",
                        description="Span duration is zero or close to zero, which can indicate incomplete timing instrumentation.",
                        span_id=span.span_id,
                        agent_name=span.agent_name,
                        details={"duration_ms": span.duration_ms},
                    )
                )

            if span.duration_ms > self.timeout_ms:
                anomalies.append(
                    WaterfallAnomaly(
                        anomaly_type="timeout_threshold_exceeded",
                        severity="high",
                        description="Span exceeded the configured timeout threshold.",
                        span_id=span.span_id,
                        agent_name=span.agent_name,
                        details={"duration_ms": span.duration_ms, "timeout_ms": self.timeout_ms},
                    )
                )

            if span.status not in _STATUS_ALIASES.values():
                anomalies.append(
                    WaterfallAnomaly(
                        anomaly_type="unknown_status",
                        severity="medium",
                        description="Span status is outside the normalized observability status vocabulary.",
                        span_id=span.span_id,
                        agent_name=span.agent_name,
                        details={"status": span.status},
                    )
                )

            if span.is_error:
                error_fingerprint = (span.agent_name, span.operation_name, span.status)
                error_fingerprint_counts[error_fingerprint] = error_fingerprint_counts.get(error_fingerprint, 0) + 1
                anomalies.append(
                    WaterfallAnomaly(
                        anomaly_type="error_status",
                        severity="high" if span.is_timeout else "medium",
                        description="Span completed with an error-class status.",
                        span_id=span.span_id,
                        agent_name=span.agent_name,
                        details={"status": span.status, "operation_name": span.operation_name},
                    )
                )

            if span.status in _NON_TERMINAL_STATUSES:
                anomalies.append(
                    WaterfallAnomaly(
                        anomaly_type="non_terminal_span_status",
                        severity="low",
                        description="Span remained in a non-terminal state at analysis time.",
                        span_id=span.span_id,
                        agent_name=span.agent_name,
                        details={"status": span.status},
                    )
                )

            if self.require_parent_child_time_alignment:
                for child_id in children_map.get(span.span_id, []):
                    child = by_id[child_id]
                    if child.start_ms < span.start_ms or child.end_ms > span.end_ms:
                        anomalies.append(
                            WaterfallAnomaly(
                                anomaly_type="parent_child_time_violation",
                                severity="high",
                                description="Child span timing falls outside of its parent span boundary.",
                                span_id=child.span_id,
                                agent_name=child.agent_name,
                                details={
                                    "parent_span_id": span.span_id,
                                    "parent_start_ms": span.start_ms,
                                    "parent_end_ms": span.end_ms,
                                    "child_start_ms": child.start_ms,
                                    "child_end_ms": child.end_ms,
                                },
                            )
                        )

        for (agent_name, operation_name, status), count in error_fingerprint_counts.items():
            if count < self.repeated_error_threshold:
                continue
            anomalies.append(
                WaterfallAnomaly(
                    anomaly_type="repeated_error_pattern",
                    severity="high",
                    description="Repeated error-class spans suggest a clustered or regressing failure mode.",
                    agent_name=agent_name,
                    details={
                        "operation_name": operation_name,
                        "status": status,
                        "count": count,
                        "threshold": self.repeated_error_threshold,
                    },
                )
            )

        anomalies.sort(
            key=lambda anomaly: (
                {"critical": 3, "high": 2, "medium": 1, "low": 0}.get(anomaly.severity, 0),
                anomaly.agent_name or "",
                anomaly.span_id or "",
            ),
            reverse=True,
        )
        return anomalies

    def _concurrency_profile(
        self,
        spans: Sequence[WaterfallSpan],
        total_duration_ms: float,
    ) -> Tuple[int, float]:
        if not spans or total_duration_ms <= 0.0:
            return 0, 0.0

        events: List[Tuple[float, int]] = []
        for span in spans:
            events.append((span.start_ms, 1))
            events.append((span.end_ms, -1))

        events.sort(key=lambda event: (event[0], event[1]))

        current = 0
        max_concurrency = 0
        weighted_sum = 0.0
        last_time = events[0][0]

        for timestamp, delta in events:
            elapsed = max(0.0, timestamp - last_time)
            weighted_sum += current * elapsed
            current += delta
            max_concurrency = max(max_concurrency, current)
            last_time = timestamp

        avg_concurrency = weighted_sum / total_duration_ms if total_duration_ms else 0.0
        return max_concurrency, avg_concurrency

    def _select_trace_id(self, spans: Sequence[WaterfallSpan]) -> Optional[str]:
        counts: Dict[str, int] = {}
        for span in spans:
            if span.trace_id:
                counts[span.trace_id] = counts.get(span.trace_id, 0) + 1
        if not counts:
            return None
        return max(counts.items(), key=lambda item: (item[1], item[0]))[0]

    def _normalize_status(self, status: Any) -> str:
        value = self._safe_str(status, default="ok").strip().lower()
        return _STATUS_ALIASES.get(value, value)

    def _extract_string(self, raw: Mapping[str, Any], *keys: str) -> Optional[str]:
        for key in keys:
            if key in raw and raw[key] is not None:
                value = self._safe_str(raw[key])
                if value:
                    return value
        return None

    def _extract_float(self, raw: Mapping[str, Any], *keys: str) -> Optional[float]:
        for key in keys:
            if key in raw and raw[key] is not None:
                try:
                    return float(raw[key])
                except (TypeError, ValueError):
                    return None
        return None

    def _coerce_mapping(self, value: Any) -> Dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        if value is None:
            return {}
        return {"raw_metadata": value}

    def _safe_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _safe_str(self, value: Any, default: str = "") -> str:
        if value is None:
            return default
        return str(value)

    def _span_sort_key(self, span: WaterfallSpan) -> Tuple[float, float, str]:
        return (span.start_ms, span.end_ms, span.span_id)

    def _merge_intervals_total(self, intervals: Sequence[Tuple[float, float]]) -> float:
        if not intervals:
            return 0.0

        merged_total = 0.0
        sorted_intervals = sorted(intervals)
        current_start, current_end = sorted_intervals[0]

        for start, end in sorted_intervals[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
                continue
            merged_total += max(0.0, current_end - current_start)
            current_start, current_end = start, end

        merged_total += max(0.0, current_end - current_start)
        return merged_total

    def _percentile(self, values: Sequence[float], percentile: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return float(values[0])

        ordered = sorted(float(value) for value in values)
        clamped = min(max(percentile, 0.0), 1.0)
        position = clamped * (len(ordered) - 1)
        lower_index = int(position)
        upper_index = min(lower_index + 1, len(ordered) - 1)
        lower_weight = upper_index - position
        upper_weight = position - lower_index
        return ordered[lower_index] * lower_weight + ordered[upper_index] * upper_weight

    def _percentile_rank(self, ordered_values: Sequence[float], value: float) -> float:
        if not ordered_values:
            return 0.0
        less_or_equal = sum(1 for candidate in ordered_values if candidate <= value)
        return less_or_equal / len(ordered_values)


def summarize_waterfall(report: WaterfallReport) -> Dict[str, Any]:
    """Convert report dataclass into a dashboard and incident-routing friendly dictionary."""
    return {
        "trace_id": report.trace_id,
        "span_count": report.span_count,
        "total_duration_ms": report.total_duration_ms,
        "critical_path_ms": report.critical_path_ms,
        "critical_path_span_ids": report.critical_path_span_ids,
        "critical_path_agent_names": report.critical_path_agent_names,
        "total_exclusive_duration_ms": report.total_exclusive_duration_ms,
        "per_agent_duration_ms": report.per_agent_duration_ms,
        "status_counts": report.status_counts,
        "retry_chain_count": len(report.retry_chains),
        "bottleneck_count": len(report.bottleneck_spans),
        "anomaly_count": len(report.anomalies),
        "max_concurrency": report.max_concurrency,
        "avg_concurrency": report.avg_concurrency,
        "root_span_ids": report.root_span_ids,
        "orphan_span_ids": report.orphan_span_ids,
        "duplicate_span_ids": report.duplicate_span_ids,
        "parse_warning_count": len(report.parse_warnings),
        "bottleneck_spans": report.bottleneck_spans,
        "anomalies": report.anomalies,
    }