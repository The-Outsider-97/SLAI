"""
- Quarantine suspect records.
- Route severe findings to Handler/Safety.
- Emit remediation suggestions (clean, impute, drop, re-fetch).
"""

from __future__ import annotations

import time
import uuid

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .utils.config_loader import load_global_config, get_config_section
from .utils.quality_error import (DataQualityError, QualityErrorType, QualitySeverity,
                                  QualityMemoryError, normalize_quality_exception)
from .quality_memory import QualityMemory
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Workflow Control")
printer = PrettyPrinter

ERR_CONFIG_INVALID = getattr(QualityErrorType, "CONFIGURATION_INVALID", QualityErrorType.POLICY_THRESHOLD_INVALID)
ERR_ROUTING_FAILED = getattr(QualityErrorType, "ROUTING_FAILED", QualityErrorType.REMEDIATION_FAILED)
ERR_PRIVACY_CONFLICT = getattr(QualityErrorType, "PRIVACY_POLICY_CONFLICT", QualityErrorType.PROVENANCE_UNTRUSTED)
ERR_PROVENANCE_MISSING = getattr(QualityErrorType, "PROVENANCE_MISSING", QualityErrorType.PROVENANCE_UNTRUSTED)


@dataclass(slots=True)
class QuarantineEntry:
    entry_id: str
    source_id: str
    batch_id: str
    record_id: Optional[str]
    severity: str
    verdict: str
    reason: str
    flags: List[str]
    created_at: float
    ttl_seconds: Optional[int]
    context: Dict[str, Any] = field(default_factory=dict)
    payload: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["created_at_iso"] = datetime.fromtimestamp(self.created_at, tz=timezone.utc).isoformat()
        return data


@dataclass(slots=True)
class RouteRecord:
    route_id: str
    target: str
    status: str
    reason: str
    payload: Dict[str, Any]
    created_at: float
    response: Optional[Dict[str, Any]] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["created_at_iso"] = datetime.fromtimestamp(self.created_at, tz=timezone.utc).isoformat()
        return data


@dataclass(slots=True)
class RemediationPlan:
    actions: List[str]
    rationale: List[str]
    priority: str
    created_at: float
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["created_at_iso"] = datetime.fromtimestamp(self.created_at, tz=timezone.utc).isoformat()
        return data


@dataclass(slots=True)
class WorkflowDecision:
    decision_id: str
    source_id: str
    batch_id: str
    verdict: str
    batch_score: float
    downstream_action: str
    flags: List[str]
    quarantine_entries: List[Dict[str, Any]]
    route_records: List[Dict[str, Any]]
    remediation_plan: Dict[str, Any]
    checker_findings: List[Dict[str, Any]]
    conflict_resolution: Optional[Dict[str, Any]]
    created_at: float
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["created_at_iso"] = datetime.fromtimestamp(self.created_at, tz=timezone.utc).isoformat()
        return data


class WorkflowControl:
    """Operational coordinator for downstream quality-control actions.

    Workflow Control is the execution layer that turns quality findings into
    enforceable operational outcomes:
    - quarantine entries for suspect records or batches,
    - structured handoff payloads for Handler and Safety,
    - remediation suggestions for cleanup or re-fetch workflows,
    - final downstream gating decisions.

    Architectural recommendation:
    keep cross-agent communication as an *injected boundary*. ``WorkflowControl``
    can write handoff intents to shared memory and can optionally invoke injected
    bridges, but it should not import or instantiate HandlerAgent / SafetyAgent
    directly. The parent ``QualityAgent`` already owns the shared-memory handle
    and is the best place to wire concrete agent instances, which avoids circular
    imports and repeated initialization.
    """

    def __init__(self,
        shared_memory: Any = None,
        handler_bridge: Any = None,
        safety_bridge: Any = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.config = load_global_config()
        self.quality_config = get_config_section("workflow_control")
        if config:
            self.quality_config.update(dict(config))

        self.memory = QualityMemory()
        self.shared_memory = shared_memory
        self.handler_bridge = handler_bridge
        self.safety_bridge = safety_bridge

        self._lock = RLock()
        self._quarantine_by_batch: Dict[str, List[Dict[str, Any]]] = {}
        self._route_history: List[Dict[str, Any]] = []
        self._decisions_by_batch: Dict[str, Dict[str, Any]] = {}

        self.enabled = bool(self.quality_config.get("enabled", True))
        self.auto_record_to_memory = bool(self.quality_config.get("auto_record_to_memory", True))
        self.default_window = str(self.quality_config.get("default_window", "latest")).strip() or "latest"

        self.policy_config = dict(self.quality_config.get("policy", {}))
        self.scoring_config = dict(self.quality_config.get("scoring", {}))
        self.quarantine_config = dict(self.quality_config.get("quarantine", {}))
        self.routing_config = dict(self.quality_config.get("routing", {}))
        self.remediation_config = dict(self.quality_config.get("remediation", {}))
        self.memory_integration_config = dict(self.quality_config.get("memory", {}))
        self.shared_memory_config = dict(self.quality_config.get("shared_memory", {}))

        self.pass_threshold = self._bounded_score(
            self.policy_config.get("pass_threshold", 0.90),
            field_name="workflow_control.policy.pass_threshold",
        )
        self.warn_threshold = self._bounded_score(
            self.policy_config.get("warn_threshold", 0.75),
            field_name="workflow_control.policy.warn_threshold",
        )
        self.reconcile_conflicts = bool(self.policy_config.get("reconcile_conflicts", True))
        self.block_on_critical = bool(self.policy_config.get("block_on_critical", True))
        self.block_on_privacy_or_safety_conflict = bool(
            self.policy_config.get("block_on_privacy_or_safety_conflict", True)
        )
        self.quarantine_on_warn = bool(self.policy_config.get("quarantine_on_warn", True))
        self.quarantine_on_block = bool(self.policy_config.get("quarantine_on_block", True))
        self.max_quarantine_entries_per_batch = self._positive_int(
            self.policy_config.get("max_quarantine_entries_per_batch", 5000),
            "workflow_control.policy.max_quarantine_entries_per_batch",
        )

        severity_penalty_cfg = dict(self.scoring_config.get("severity_penalty", {}))
        self.severity_penalty = {
            "low": float(severity_penalty_cfg.get("low", 0.08)),
            "medium": float(severity_penalty_cfg.get("medium", 0.22)),
            "high": float(severity_penalty_cfg.get("high", 0.50)),
            "critical": float(severity_penalty_cfg.get("critical", 0.85)),
        }
        verdict_penalty_cfg = dict(self.scoring_config.get("verdict_penalty", {}))
        self.verdict_penalty = {
            "pass": float(verdict_penalty_cfg.get("pass", 0.0)),
            "warn": float(verdict_penalty_cfg.get("warn", 0.18)),
            "quarantine": float(verdict_penalty_cfg.get("quarantine", 0.32)),
            "block": float(verdict_penalty_cfg.get("block", 0.70)),
            "escalate": float(verdict_penalty_cfg.get("escalate", 0.40)),
            "retry": float(verdict_penalty_cfg.get("retry", 0.20)),
            "fallback": float(verdict_penalty_cfg.get("fallback", 0.15)),
        }

        self.quarantine_enabled = bool(self.quarantine_config.get("enabled", True))
        self.include_record_payload = bool(self.quarantine_config.get("include_record_payload", False))
        self.quarantine_ttl_seconds = self._optional_nonnegative_int(
            self.quarantine_config.get("ttl_seconds", 604800),
            "workflow_control.quarantine.ttl_seconds",
        )
        self.record_queue_key = str(
            self.quarantine_config.get("record_queue_key", "quality:quarantine:records")
        ).strip() or "quality:quarantine:records"
        self.batch_queue_key = str(
            self.quarantine_config.get("batch_queue_key", "quality:quarantine:batches")
        ).strip() or "quality:quarantine:batches"

        self.publish_events_to_shared_memory = bool(
            self.routing_config.get("publish_events_to_shared_memory", True)
        )
        self.event_channel = str(self.routing_config.get("event_channel", "quality_events")).strip() or "quality_events"
        self.handler_verdicts = {
            self._normalize_verdict(item)
            for item in self.routing_config.get("handler_on_verdicts", ["block"])
        }
        self.handler_severities = {
            self._normalize_severity(item)
            for item in self.routing_config.get("handler_on_severities", ["high", "critical"])
        }
        self.safety_severities = {
            self._normalize_severity(item)
            for item in self.routing_config.get("safety_on_severities", ["critical"])
        }
        self.safety_error_types = {
            str(item).strip()
            for item in self.routing_config.get(
                "safety_on_error_types",
                [
                    ERR_PRIVACY_CONFLICT.value,
                    QualityErrorType.LEAKAGE_DETECTED.value,
                    QualityErrorType.PROVENANCE_UNTRUSTED.value,
                    ERR_PROVENANCE_MISSING.value,
                ],
            )
            if str(item).strip()
        }
        self.fail_closed_on_handler_error = bool(
            self.routing_config.get("fail_closed_on_handler_error", False)
        )
        self.fail_closed_on_safety_error = bool(
            self.routing_config.get("fail_closed_on_safety_error", False)
        )

        self.max_remediation_actions = self._positive_int(
            self.remediation_config.get("max_actions", 10),
            "workflow_control.remediation.max_actions",
        )
        self.default_remediation_actions = [
            str(item)
            for item in self.remediation_config.get(
                "default_actions",
                ["clean_records", "impute_missing_values", "drop_invalid_records", "re_fetch_source"],
            )
        ]
        self.action_catalog = {
            str(k): [str(item) for item in v]
            for k, v in dict(self.remediation_config.get("action_catalog", {})).items()
        }

        self.store_decision_state = bool(self.shared_memory_config.get("store_decision_state", True))
        self.publish_notifications = bool(self.shared_memory_config.get("publish_notifications", True))
        self.shared_decision_key_prefix = str(
            self.shared_memory_config.get("decision_state_key_prefix", "quality:workflow")
        ).strip() or "quality:workflow"
        self.shared_ttl_seconds = self._optional_nonnegative_int(
            self.shared_memory_config.get("ttl_seconds", 604800),
            "workflow_control.shared_memory.ttl_seconds",
        )

        self._validate_runtime_configuration()
        logger.info(
            "Workflow Control initialized | shared_memory=%s | handler_bridge=%s | safety_bridge=%s",
            self.shared_memory is not None,
            self.handler_bridge is not None,
            self.safety_bridge is not None,
        )

    # ------------------------------------------------------------------
    # Runtime attachment
    # ------------------------------------------------------------------
    def attach_runtime(
        self,
        *,
        shared_memory: Any = None,
        handler_bridge: Any = None,
        safety_bridge: Any = None,
    ) -> None:
        if shared_memory is not None:
            self.shared_memory = shared_memory
        if handler_bridge is not None:
            self.handler_bridge = handler_bridge
        if safety_bridge is not None:
            self.safety_bridge = safety_bridge

    # ------------------------------------------------------------------
    # Public orchestration APIs
    # ------------------------------------------------------------------
    def coordinate_batch(
        self,
        *,
        source_id: str,
        batch_id: str,
        findings: Optional[Sequence[Mapping[str, Any]]] = None,
        batch_score: Optional[float] = None,
        records: Optional[Sequence[Mapping[str, Any]]] = None,
        context: Optional[Mapping[str, Any]] = None,
        shared_memory: Any = None,
        handler_bridge: Any = None,
        safety_bridge: Any = None,
    ) -> Dict[str, Any]:
        with self._boundary("coordinate_batch", "routing", {"source_id": source_id, "batch_id": batch_id}):
            source_key = self._nonempty(source_id, "source_id")
            batch_key = self._nonempty(batch_id, "batch_id")
            normalized_context = self._normalized_mapping(context)
            active_shared_memory = shared_memory if shared_memory is not None else self.shared_memory
            active_handler = handler_bridge if handler_bridge is not None else self.handler_bridge
            active_safety = safety_bridge if safety_bridge is not None else self.safety_bridge

            normalized_findings = self._normalize_findings(findings)
            conflict_resolution = self._reconcile_findings(
                findings=normalized_findings,
                source_id=source_key,
                batch_id=batch_key,
                context=normalized_context,
            )
            resolved_score = self._derive_batch_score(normalized_findings, explicit_score=batch_score)
            final_verdict = self._resolve_final_verdict(
                findings=normalized_findings,
                batch_score=resolved_score,
                conflict_resolution=conflict_resolution,
            )
            downstream_action = self._downstream_action(final_verdict)
            flags = self._collect_flags(normalized_findings, final_verdict)
            remediation_plan = self.emit_remediation_suggestions(
                findings=normalized_findings,
                verdict=final_verdict,
                context=normalized_context,
            )
            quarantine_entries = self.quarantine_records(
                source_id=source_key,
                batch_id=batch_key,
                findings=normalized_findings,
                verdict=final_verdict,
                records=records,
                context=normalized_context,
                shared_memory=active_shared_memory,
            )
            route_records = self.route_severe_findings(
                source_id=source_key,
                batch_id=batch_key,
                verdict=final_verdict,
                findings=normalized_findings,
                remediation_plan=remediation_plan,
                context=normalized_context,
                handler_bridge=active_handler,
                safety_bridge=active_safety,
                shared_memory=active_shared_memory,
            )

            decision = WorkflowDecision(
                decision_id=self._new_id("workflow_decision"),
                source_id=source_key,
                batch_id=batch_key,
                verdict=final_verdict,
                batch_score=resolved_score,
                downstream_action=downstream_action,
                flags=flags,
                quarantine_entries=deepcopy(quarantine_entries),
                route_records=deepcopy(route_records),
                remediation_plan=deepcopy(remediation_plan),
                checker_findings=deepcopy(normalized_findings),
                conflict_resolution=deepcopy(conflict_resolution),
                created_at=time.time(),
                context=normalized_context,
            )

            decision_dict = decision.to_dict()
            with self._lock:
                self._quarantine_by_batch[batch_key] = deepcopy(quarantine_entries)
                self._route_history.extend(deepcopy(route_records))
                self._decisions_by_batch[batch_key] = deepcopy(decision_dict)

            self._publish_decision_state(decision_dict, shared_memory=active_shared_memory)
            self._record_decision_to_memory(decision_dict)
            return decision_dict

    def process_quality_result(self, **kwargs: Any) -> Dict[str, Any]:
        return self.coordinate_batch(**kwargs)

    def emit_remediation_suggestions(
        self,
        *,
        findings: Sequence[Mapping[str, Any]],
        verdict: str,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._boundary("emit_remediation_suggestions", "remediation"):
            normalized_findings = self._normalize_findings(findings)
            actions: List[str] = []
            rationale: List[str] = []

            for finding in normalized_findings:
                error_type = str(finding.get("error_type") or "").strip()
                checker = str(finding.get("checker") or finding.get("domain") or "unknown")
                finding_actions = [str(item) for item in finding.get("remediation_actions", [])]
                for action in finding_actions:
                    if action not in actions:
                        actions.append(action)
                        rationale.append(f"{checker} suggested '{action}'.")

                for catalog_key in (error_type, checker, verdict):
                    for action in self.action_catalog.get(catalog_key, []):
                        if action not in actions:
                            actions.append(action)
                            rationale.append(f"Policy catalog matched '{catalog_key}' -> '{action}'.")

            for action in self.default_remediation_actions:
                if len(actions) >= self.max_remediation_actions:
                    break
                if action not in actions:
                    actions.append(action)
                    rationale.append("Fallback workflow policy supplied a default remediation action.")

            priority = "high" if verdict == "block" else "medium" if verdict == "warn" else "low"
            plan = RemediationPlan(
                actions=actions[: self.max_remediation_actions],
                rationale=rationale[: self.max_remediation_actions],
                priority=priority,
                created_at=time.time(),
                context=self._normalized_mapping(context),
            )
            return plan.to_dict()

    def quarantine_records(
        self,
        *,
        source_id: str,
        batch_id: str,
        findings: Sequence[Mapping[str, Any]],
        verdict: str,
        records: Optional[Sequence[Mapping[str, Any]]] = None,
        context: Optional[Mapping[str, Any]] = None,
        shared_memory: Any = None,
    ) -> List[Dict[str, Any]]:
        with self._boundary("quarantine_records", "quarantine", {"source_id": source_id, "batch_id": batch_id}):
            if not self.quarantine_enabled:
                return []
            if verdict == "pass":
                return []
            if verdict == "warn" and not self.quarantine_on_warn:
                return []
            if verdict == "block" and not self.quarantine_on_block:
                return []

            normalized_findings = self._normalize_findings(findings)
            record_lookup = self._record_lookup(records)
            entries: List[Dict[str, Any]] = []
            seen_record_ids: set[str] = set()

            for finding in normalized_findings:
                candidate_ids = self._extract_record_ids(finding)
                candidate_flags = self._string_list(finding.get("flags"))
                severity = self._normalize_severity(finding.get("severity", "medium"))
                reason = str(finding.get("message") or finding.get("error_type") or "quality finding")

                if not candidate_ids:
                    continue

                for record_id in candidate_ids:
                    if len(entries) >= self.max_quarantine_entries_per_batch:
                        break
                    record_key = str(record_id)
                    if record_key in seen_record_ids:
                        continue
                    seen_record_ids.add(record_key)
                    payload = deepcopy(record_lookup.get(record_key)) if self.include_record_payload else None
                    entry = QuarantineEntry(
                        entry_id=self._new_id("quarantine"),
                        source_id=source_id,
                        batch_id=batch_id,
                        record_id=record_key,
                        severity=severity,
                        verdict=verdict,
                        reason=reason,
                        flags=candidate_flags,
                        created_at=time.time(),
                        ttl_seconds=self.quarantine_ttl_seconds,
                        context=self._normalized_mapping(context),
                        payload=payload,
                    )
                    entries.append(entry.to_dict())

            if not entries and verdict == "block":
                batch_entry = QuarantineEntry(
                    entry_id=self._new_id("quarantine_batch"),
                    source_id=source_id,
                    batch_id=batch_id,
                    record_id=None,
                    severity="critical",
                    verdict=verdict,
                    reason="No record-level targets were supplied; quarantining the full batch.",
                    flags=["batch_quarantine"],
                    created_at=time.time(),
                    ttl_seconds=self.quarantine_ttl_seconds,
                    context=self._normalized_mapping(context),
                    payload=None,
                )
                entries.append(batch_entry.to_dict())

            if shared_memory is not None and entries:
                try:
                    self._shared_append(shared_memory, self.record_queue_key, entries, ttl=self.shared_ttl_seconds)
                    self._shared_append(
                        shared_memory,
                        self.batch_queue_key,
                        [{"source_id": source_id, "batch_id": batch_id, "count": len(entries), "verdict": verdict}],
                        ttl=self.shared_ttl_seconds,
                    )
                except Exception as exc:
                    raise DataQualityError(
                        message=f"Failed to quarantine batch '{batch_id}': {exc}",
                        error_type=QualityErrorType.QUARANTINE_OPERATION_FAILED,
                        severity=QualitySeverity.HIGH,
                        retryable=True,
                        context={"source_id": source_id, "batch_id": batch_id, "entry_count": len(entries)},
                        remediation="Persist the quarantine failure locally and retry with an idempotent handoff.",
                    ) from exc

            return entries

    def route_severe_findings(
        self,
        *,
        source_id: str,
        batch_id: str,
        verdict: str,
        findings: Sequence[Mapping[str, Any]],
        remediation_plan: Mapping[str, Any],
        context: Optional[Mapping[str, Any]] = None,
        handler_bridge: Any = None,
        safety_bridge: Any = None,
        shared_memory: Any = None,
    ) -> List[Dict[str, Any]]:
        with self._boundary("route_severe_findings", "routing", {"source_id": source_id, "batch_id": batch_id}):
            normalized_findings = self._normalize_findings(findings)
            route_records: List[Dict[str, Any]] = []

            if self._should_route_to_handler(verdict=verdict, findings=normalized_findings):
                payload = self._build_handler_payload(
                    source_id=source_id,
                    batch_id=batch_id,
                    verdict=verdict,
                    findings=normalized_findings,
                    remediation_plan=remediation_plan,
                    context=context,
                )
                route_records.append(
                    self._invoke_route(
                        target="handler",
                        bridge=handler_bridge,
                        payload=payload,
                        fail_closed=self.fail_closed_on_handler_error,
                        shared_memory=shared_memory,
                    )
                )

            if self._should_route_to_safety(verdict=verdict, findings=normalized_findings):
                payload = self._build_safety_payload(
                    source_id=source_id,
                    batch_id=batch_id,
                    verdict=verdict,
                    findings=normalized_findings,
                    remediation_plan=remediation_plan,
                    context=context,
                )
                route_records.append(
                    self._invoke_route(
                        target="safety",
                        bridge=safety_bridge,
                        payload=payload,
                        fail_closed=self.fail_closed_on_safety_error,
                        shared_memory=shared_memory,
                    )
                )

            return route_records

    def latest_decision(self, batch_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            decision = self._decisions_by_batch.get(str(batch_id))
            return None if decision is None else deepcopy(decision)

    def quarantine_entries(self, batch_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return deepcopy(self._quarantine_by_batch.get(str(batch_id), []))

    def route_history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return deepcopy(self._route_history)

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "decisions": len(self._decisions_by_batch),
                "batches_with_quarantine": len(self._quarantine_by_batch),
                "route_events": len(self._route_history),
                "has_shared_memory": self.shared_memory is not None,
                "has_handler_bridge": self.handler_bridge is not None,
                "has_safety_bridge": self.safety_bridge is not None,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @contextmanager
    def _boundary(self, operation: str, stage: str, context: Optional[Mapping[str, Any]] = None):
        try:
            yield
        except Exception as exc:
            raise self._normalize_exception(
                exc,
                stage=stage,
                context={"operation": operation, **dict(context or {})},
            ) from exc

    def _normalize_exception(
        self,
        exc: Exception,
        *,
        stage: str,
        context: Optional[Mapping[str, Any]] = None,
        error_type: Optional[QualityErrorType] = None,
        severity: QualitySeverity = QualitySeverity.HIGH,
        retryable: bool = True,
        remediation: Optional[str] = None,
    ) -> DataQualityError:
        if isinstance(exc, DataQualityError):
            return exc
        try:
            normalized = normalize_quality_exception(exc, stage=stage, context=dict(context or {}))
            if isinstance(normalized, DataQualityError) and (error_type is None or normalized.error_type != QualityErrorType.REMEDIATION_FAILED):
                return normalized
        except Exception:
            pass
        return DataQualityError(
            message=f"Unhandled exception during workflow stage '{stage}': {exc}",
            error_type=error_type or ERR_ROUTING_FAILED,
            severity=severity,
            retryable=retryable,
            context=dict(context or {}),
            remediation=remediation or "Retry the workflow step with preserved payload and escalate through Handler if the failure persists.",
        )

    def _validate_runtime_configuration(self) -> None:
        if self.warn_threshold > self.pass_threshold:
            raise DataQualityError(
                message="workflow_control.policy.warn_threshold must be <= pass_threshold",
                error_type=ERR_CONFIG_INVALID,
                severity=QualitySeverity.HIGH,
                retryable=False,
                context={"warn_threshold": self.warn_threshold, "pass_threshold": self.pass_threshold},
                remediation="Correct the workflow thresholds so warn_threshold <= pass_threshold.",
            )
        for key, value in {**self.severity_penalty, **self.verdict_penalty}.items():
            if float(value) < 0:
                raise DataQualityError(
                    message=f"workflow_control scoring penalty for '{key}' must be non-negative",
                    error_type=ERR_CONFIG_INVALID,
                    severity=QualitySeverity.HIGH,
                    retryable=False,
                    context={"penalty_key": key, "value": value},
                    remediation="Set workflow scoring penalties to non-negative values.",
                )

    def _normalize_findings(self, findings: Optional[Sequence[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for item in findings or []:
            finding = self._normalized_mapping(item)
            verdict = self._normalize_verdict(finding.get("verdict", "warn"))
            severity = self._normalize_severity(finding.get("severity", "medium"))
            confidence = self._clamp(float(finding.get("confidence", 1.0)), 0.0, 1.0)
            checker = str(finding.get("checker") or finding.get("domain") or "unknown")
            error_type = finding.get("error_type")
            if isinstance(error_type, QualityErrorType):
                error_type_value = error_type.value
            else:
                error_type_value = str(error_type).strip() if error_type is not None else ""
            disposition = str(finding.get("disposition") or verdict).strip().lower()
            finding.update(
                {
                    "checker": checker,
                    "verdict": verdict,
                    "severity": severity,
                    "confidence": confidence,
                    "error_type": error_type_value,
                    "disposition": disposition,
                    "flags": self._string_list(finding.get("flags")),
                    "remediation_actions": self._string_list(finding.get("remediation_actions")),
                    "message": str(finding.get("message") or ""),
                }
            )
            normalized.append(finding)
        return normalized

    def _reconcile_findings(
        self,
        *,
        findings: Sequence[Mapping[str, Any]],
        source_id: str,
        batch_id: str,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.reconcile_conflicts or len(findings) <= 1:
            return None
        if not hasattr(self.memory, "reconcile_conflicts"):
            return {
                "conflict_id": self._new_id("conflict_local"),
                "scope_key": f"workflow:{source_id}:{batch_id}",
                "source_id": source_id,
                "batch_id": batch_id,
                "reconciled_verdict": self._fallback_conflict_verdict(findings),
                "weighted_score": None,
                "rationale": "quality_memory does not expose reconcile_conflicts; local fallback reconciliation was used.",
                "findings": deepcopy(list(findings)),
                "recorded_at": time.time(),
                "context": self._normalized_mapping(context),
            }
        try:
            result = self.memory.reconcile_conflicts(
                findings=findings,
                source_id=source_id,
                batch_id=batch_id,
                scope_key=f"workflow:{source_id}:{batch_id}",
                context=context,
            )
            return self._normalized_mapping(result)
        except Exception as exc:
            normalized = self._normalize_exception(
                exc,
                stage="routing",
                context={"source_id": source_id, "batch_id": batch_id, "operation": "reconcile_conflicts"},
                error_type=QualityErrorType.QUALITY_MEMORY_UNAVAILABLE,
                severity=QualitySeverity.HIGH,
                retryable=False,
                remediation="Review conflicting findings and re-run the workflow reconciliation stage.",
            )
            normalized.report()
            return {
                "conflict_id": self._new_id("conflict_fallback"),
                "scope_key": f"workflow:{source_id}:{batch_id}",
                "source_id": source_id,
                "batch_id": batch_id,
                "reconciled_verdict": self._fallback_conflict_verdict(findings),
                "weighted_score": None,
                "rationale": f"Conflict reconciliation fell back locally after memory failure: {normalized.message}",
                "findings": deepcopy(list(findings)),
                "recorded_at": time.time(),
                "context": {"fallback": True},
            }

    def _fallback_conflict_verdict(self, findings: Sequence[Mapping[str, Any]]) -> str:
        verdicts = {str(item.get("verdict", "warn")) for item in findings}
        if "block" in verdicts:
            return "block"
        if "warn" in verdicts:
            return "warn"
        return "pass"

    def _derive_batch_score(
        self,
        findings: Sequence[Mapping[str, Any]],
        *,
        explicit_score: Optional[float] = None,
    ) -> float:
        if explicit_score is not None:
            return self._bounded_score(explicit_score, field_name="batch_score")
        if not findings:
            return 1.0

        penalties: List[float] = []
        for finding in findings:
            severity_penalty = self.severity_penalty.get(str(finding.get("severity", "medium")), 0.22)
            disposition = str(finding.get("disposition", finding.get("verdict", "warn"))).strip().lower()
            verdict_penalty = self.verdict_penalty.get(disposition, self.verdict_penalty.get(str(finding.get("verdict", "warn")), 0.18))
            confidence = self._clamp(float(finding.get("confidence", 1.0)), 0.0, 1.0)
            penalties.append(max(severity_penalty, verdict_penalty) * confidence)

        mean_penalty = sum(penalties) / len(penalties)
        return self._clamp(1.0 - mean_penalty, 0.0, 1.0)

    def _resolve_final_verdict(
        self,
        *,
        findings: Sequence[Mapping[str, Any]],
        batch_score: float,
        conflict_resolution: Optional[Mapping[str, Any]],
    ) -> str:
        if self.block_on_critical and any(str(item.get("severity")) == "critical" for item in findings):
            return "block"

        if self.block_on_privacy_or_safety_conflict and any(
            str(item.get("error_type")) in {
                ERR_PRIVACY_CONFLICT.value,
                QualityErrorType.LEAKAGE_DETECTED.value,
                QualityErrorType.PROVENANCE_UNTRUSTED.value,
                ERR_PROVENANCE_MISSING.value,
            }
            for item in findings
        ):
            return "block"

        if conflict_resolution is not None:
            reconciled = self._normalize_verdict(conflict_resolution.get("reconciled_verdict", "warn"))
            if reconciled == "block":
                return "block"
            if reconciled == "warn" and batch_score < self.pass_threshold:
                return "warn"

        if any(str(item.get("verdict")) == "block" for item in findings):
            return "block"
        if batch_score < self.warn_threshold:
            return "block"
        if any(str(item.get("verdict")) == "warn" for item in findings):
            return "warn"
        if batch_score < self.pass_threshold:
            return "warn"
        return "pass"

    def _downstream_action(self, verdict: str) -> str:
        resolved = self._normalize_verdict(verdict)
        if resolved == "block":
            return "hold_and_escalate"
        if resolved == "warn":
            return "allow_with_review"
        return "allow"

    def _collect_flags(self, findings: Sequence[Mapping[str, Any]], verdict: str) -> List[str]:
        flags: List[str] = [f"workflow_verdict:{verdict}"]
        for finding in findings:
            for flag in self._string_list(finding.get("flags")):
                if flag not in flags:
                    flags.append(flag)
            error_type = str(finding.get("error_type") or "").strip()
            if error_type:
                marker = f"error:{error_type}"
                if marker not in flags:
                    flags.append(marker)
        return flags

    def _should_route_to_handler(self, *, verdict: str, findings: Sequence[Mapping[str, Any]]) -> bool:
        if self._normalize_verdict(verdict) in self.handler_verdicts:
            return True
        return any(str(item.get("severity")) in self.handler_severities for item in findings)

    def _should_route_to_safety(self, *, verdict: str, findings: Sequence[Mapping[str, Any]]) -> bool:
        if self._normalize_verdict(verdict) == "block" and any(
            str(item.get("error_type")) in self.safety_error_types for item in findings
        ):
            return True
        return any(
            str(item.get("severity")) in self.safety_severities
            or str(item.get("error_type")) in self.safety_error_types
            for item in findings
        )

    def _build_handler_payload(
        self,
        *,
        source_id: str,
        batch_id: str,
        verdict: str,
        findings: Sequence[Mapping[str, Any]],
        remediation_plan: Mapping[str, Any],
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        critical_findings = [item for item in findings if item.get("severity") in {"high", "critical"}]
        headline = critical_findings[0] if critical_findings else (findings[0] if findings else {})
        error_type = str(headline.get("error_type") or ERR_ROUTING_FAILED.value)
        error_message = str(headline.get("message") or f"Quality workflow requires handler attention for batch '{batch_id}'")
        return {
            "error_info": {
                "error_type": error_type,
                "error_message": error_message,
            },
            "task_data": {
                "source_id": source_id,
                "batch_id": batch_id,
                "verdict": verdict,
                "findings": deepcopy(list(findings)),
                "remediation_plan": deepcopy(dict(remediation_plan)),
                "route": "quality_workflow",
            },
            "context": {
                "agent": "QualityAgent",
                "route": "quality_workflow_to_handler",
                "task_id": f"quality-handler-{batch_id}",
                **dict(context or {}),
            },
        }

    def _build_safety_payload(
        self,
        *,
        source_id: str,
        batch_id: str,
        verdict: str,
        findings: Sequence[Mapping[str, Any]],
        remediation_plan: Mapping[str, Any],
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "source_id": source_id,
            "batch_id": batch_id,
            "verdict": verdict,
            "findings": deepcopy(list(findings)),
            "remediation_plan": deepcopy(dict(remediation_plan)),
            "context": {
                "source": "quality_workflow",
                "route": "quality_workflow_to_safety",
                **dict(context or {}),
            },
        }

    def _invoke_route(
        self,
        *,
        target: str,
        bridge: Any,
        payload: Mapping[str, Any],
        fail_closed: bool,
        shared_memory: Any,
    ) -> Dict[str, Any]:
        record = RouteRecord(
            route_id=self._new_id(f"route_{target}"),
            target=target,
            status="pending",
            reason="Prepared route payload.",
            payload=deepcopy(dict(payload)),
            created_at=time.time(),
            response=None,
            context={"fail_closed": fail_closed},
        )

        if bridge is None:
            record.status = "deferred"
            record.reason = f"No {target} bridge attached; route payload retained for parent-agent delivery."
            if shared_memory is not None and self.publish_events_to_shared_memory:
                self._publish_route_intent(record.to_dict(), shared_memory=shared_memory)
            return record.to_dict()

        try:
            if callable(bridge):
                response = bridge(deepcopy(dict(payload)))
            elif hasattr(bridge, "perform_task"):
                response = bridge.perform_task(deepcopy(dict(payload)))
            elif target == "safety" and hasattr(bridge, "predict"):
                response = bridge.predict(
                    deepcopy(dict(payload)),
                    context=deepcopy(dict(payload.get("context") or {})),
                )
            else:
                raise TypeError(f"Unsupported {target} bridge interface")

            record.status = "delivered"
            record.reason = f"Route delivered to {target}."
            record.response = self._normalized_mapping(response if isinstance(response, Mapping) else {"response": response})
        except Exception as exc:
            normalized = self._normalize_exception(
                exc,
                stage="routing",
                context={"target": target, "payload": dict(payload)},
                error_type=ERR_ROUTING_FAILED,
                severity=QualitySeverity.HIGH,
                retryable=True,
                remediation="Persist the route payload, retry through WorkflowControl or QualityAgent, and escalate if repeated.",
            )
            normalized.report()
            record.status = "failed"
            record.reason = normalized.message
            record.response = normalized.to_dict()
            if fail_closed:
                raise normalized from exc
            if shared_memory is not None and self.publish_events_to_shared_memory:
                self._publish_route_intent(record.to_dict(), shared_memory=shared_memory)
        else:
            if shared_memory is not None and self.publish_events_to_shared_memory:
                self._publish_route_intent(record.to_dict(), shared_memory=shared_memory)

        return record.to_dict()

    def _publish_decision_state(self, decision: Mapping[str, Any], *, shared_memory: Any) -> None:
        if shared_memory is None or not self.store_decision_state:
            return
        key = f"{self.shared_decision_key_prefix}:{decision['batch_id']}"
        self._shared_set(shared_memory, key, deepcopy(dict(decision)), ttl=self.shared_ttl_seconds)
        if self.publish_notifications:
            self._shared_publish(
                shared_memory,
                self.event_channel,
                {"event_type": "quality_workflow_decision", "decision": deepcopy(dict(decision))},
            )

    def _publish_route_intent(self, route_record: Mapping[str, Any], *, shared_memory: Any) -> None:
        if not self.publish_notifications:
            return
        self._shared_publish(
            shared_memory,
            self.event_channel,
            {"event_type": "quality_workflow_route", "route": deepcopy(dict(route_record))},
        )

    def _record_decision_to_memory(self, decision: Mapping[str, Any]) -> None:
        if not self.auto_record_to_memory:
            return
        if not bool(self.memory_integration_config.get("record_quality_snapshot", True)):
            return

        if not hasattr(self.memory, "record_quality_snapshot"):
            return

        try:
            source_reliability = 0.5
            if hasattr(self.memory, "latest_source_reliability"):
                latest_reliability = self.memory.latest_source_reliability(decision["source_id"])
                if isinstance(latest_reliability, Mapping) and latest_reliability.get("reliability") is not None:
                    source_reliability = float(latest_reliability.get("reliability"))
                elif hasattr(self.memory, "default_source_reliability"):
                    source_reliability = float(getattr(self.memory, "default_source_reliability"))
            elif hasattr(self.memory, "default_source_reliability"):
                source_reliability = float(getattr(self.memory, "default_source_reliability"))

            self.memory.record_quality_snapshot(
                source_id=decision["source_id"],
                batch_id=decision["batch_id"],
                batch_score=float(decision["batch_score"]),
                verdict=str(decision["verdict"]),
                flags=self._string_list(decision.get("flags")),
                quarantine_count=len(decision.get("quarantine_entries", [])),
                shift_metrics={},
                remediation_actions=self._string_list(
                    decision.get("remediation_plan", {}).get("actions", [])
                ),
                source_reliability=source_reliability,
                schema_version=None,
                window=self.default_window,
                checker_findings=self._mapping_list(decision.get("checker_findings", [])),
                context={
                    **self._normalized_mapping(decision.get("context")),
                    "workflow_decision_id": decision.get("decision_id"),
                    "route_count": len(decision.get("route_records", [])),
                },
            )
        except Exception as exc:
            normalized = self._normalize_exception(
                exc,
                stage="persistence",
                context={"operation": "record_workflow_snapshot", "batch_id": decision.get("batch_id")},
                error_type=QualityErrorType.QUALITY_MEMORY_UNAVAILABLE,
                severity=QualitySeverity.CRITICAL,
                retryable=True,
                remediation="Preserve the workflow decision locally and retry persistence into quality_memory.",
            )
            normalized.report()
            logger.error("Workflow Control memory persistence failed: %s", normalized.to_dict())

    def _record_lookup(self, records: Optional[Sequence[Mapping[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        lookup: Dict[str, Dict[str, Any]] = {}
        for record in records or []:
            record_id = self._infer_record_id(record)
            if record_id is not None:
                lookup[record_id] = self._normalized_mapping(record)
        return lookup

    def _infer_record_id(self, record: Mapping[str, Any]) -> Optional[str]:
        for candidate in ("record_id", "id", "row_id", "sample_id", "uuid"):
            value = record.get(candidate)
            if value is not None and str(value).strip():
                return str(value)
        return None

    def _extract_record_ids(self, finding: Mapping[str, Any]) -> List[str]:
        record_ids: List[str] = []
        for key in ("quarantine_record_ids", "affected_record_ids", "record_ids"):
            for item in self._string_list(finding.get(key)):
                if item not in record_ids:
                    record_ids.append(item)
        scalar_record_id = finding.get("record_id")
        if scalar_record_id is not None and str(scalar_record_id).strip():
            record_key = str(scalar_record_id)
            if record_key not in record_ids:
                record_ids.append(record_key)
        return record_ids

    def _shared_set(self, shared_memory: Any, key: str, value: Any, *, ttl: Optional[int]) -> None:
        if hasattr(shared_memory, "set"):
            try:
                shared_memory.set(key, value, ttl=ttl)
            except TypeError:
                shared_memory.set(key, value)
            return
        if hasattr(shared_memory, "put"):
            try:
                shared_memory.put(key, value, ttl=ttl)
            except TypeError:
                shared_memory.put(key, value)
            return
        raise TypeError("shared_memory object does not expose set(...) or put(...)")

    def _shared_append(self, shared_memory: Any, key: str, values: Sequence[Any], *, ttl: Optional[int]) -> None:
        existing = []
        if hasattr(shared_memory, "get"):
            try:
                existing = list(shared_memory.get(key, default=[])) or []
            except TypeError:
                current = shared_memory.get(key)
                existing = list(current) if current else []
        existing.extend(deepcopy(list(values)))
        self._shared_set(shared_memory, key, existing, ttl=ttl)

    def _shared_publish(self, shared_memory: Any, channel: str, message: Mapping[str, Any]) -> None:
        if hasattr(shared_memory, "publish"):
            shared_memory.publish(channel=channel, message=deepcopy(dict(message)))
            return
        event_key = f"{channel}:events"
        self._shared_append(shared_memory, event_key, [deepcopy(dict(message))], ttl=self.shared_ttl_seconds)

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

    def _positive_int(self, value: Any, field_name: str) -> int:
        try:
            resolved = int(value)
        except Exception as exc:
            raise DataQualityError(
                message=f"{field_name} must be a positive integer",
                error_type=ERR_CONFIG_INVALID,
                severity=QualitySeverity.HIGH,
                retryable=False,
                context={"field_name": field_name, "value": value},
                remediation="Use a positive integer in workflow_control configuration.",
            ) from exc
        if resolved <= 0:
            raise DataQualityError(
                message=f"{field_name} must be a positive integer",
                error_type=ERR_CONFIG_INVALID,
                severity=QualitySeverity.HIGH,
                retryable=False,
                context={"field_name": field_name, "value": value},
                remediation="Use a positive integer in workflow_control configuration.",
            )
        return resolved

    def _optional_nonnegative_int(self, value: Any, field_name: str) -> Optional[int]:
        if value is None:
            return None
        try:
            resolved = int(value)
        except Exception as exc:
            raise DataQualityError(
                message=f"{field_name} must be a non-negative integer or null",
                error_type=ERR_CONFIG_INVALID,
                severity=QualitySeverity.HIGH,
                retryable=False,
                context={"field_name": field_name, "value": value},
                remediation="Use a non-negative integer or null in workflow_control configuration.",
            ) from exc
        if resolved < 0:
            raise DataQualityError(
                message=f"{field_name} must be a non-negative integer or null",
                error_type=ERR_CONFIG_INVALID,
                severity=QualitySeverity.HIGH,
                retryable=False,
                context={"field_name": field_name, "value": value},
                remediation="Use a non-negative integer or null in workflow_control configuration.",
            )
        return resolved

    def _bounded_score(self, value: Any, *, field_name: str) -> float:
        try:
            score = float(value)
        except Exception as exc:
            raise DataQualityError(
                message=f"{field_name} must be numeric within [0.0, 1.0]",
                error_type=QualityErrorType.POLICY_THRESHOLD_INVALID,
                severity=QualitySeverity.HIGH,
                retryable=False,
                context={"field_name": field_name, "value": value},
                remediation="Use a numeric workflow threshold between 0.0 and 1.0.",
            ) from exc
        if score < 0.0 or score > 1.0:
            raise DataQualityError(
                message=f"{field_name} must be within [0.0, 1.0]",
                error_type=QualityErrorType.POLICY_THRESHOLD_INVALID,
                severity=QualitySeverity.HIGH,
                retryable=False,
                context={"field_name": field_name, "value": score},
                remediation="Use a numeric workflow threshold between 0.0 and 1.0.",
            )
        return score

    def _nonempty(self, value: Any, field_name: str) -> str:
        text = str(value).strip() if value is not None else ""
        if not text:
            raise DataQualityError(
                message=f"{field_name} must not be empty",
                error_type=ERR_CONFIG_INVALID,
                severity=QualitySeverity.MEDIUM,
                retryable=False,
                context={"field_name": field_name, "value": value},
                remediation="Provide a non-empty identifier.",
            )
        return text

    def _normalize_verdict(self, verdict: Any) -> str:
        value = str(verdict).strip().lower()
        if value not in {"pass", "warn", "block"}:
            raise DataQualityError(
                message=f"Unsupported workflow verdict '{verdict}'",
                error_type=ERR_CONFIG_INVALID,
                severity=QualitySeverity.MEDIUM,
                retryable=False,
                context={"verdict": verdict},
                remediation="Use one of: pass, warn, block.",
            )
        return value

    def _normalize_severity(self, severity: Any) -> str:
        value = str(severity).strip().lower()
        if value not in {"low", "medium", "high", "critical"}:
            return "medium"
        return value

    def _clamp(self, value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    def _string_list(self, values: Any) -> List[str]:
        if values is None:
            return []
        if isinstance(values, str):
            return [values]
        if isinstance(values, Mapping):
            return [str(k) for k in values.keys()]
        if isinstance(values, Iterable):
            return [str(item) for item in values]
        return [str(values)]

    def _mapping_list(self, values: Optional[Sequence[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
        if values is None:
            return []
        return [self._normalized_mapping(item) for item in values]

    def _normalized_mapping(self, value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if value is None:
            return {}
        return {str(k): self._safe_value(v) for k, v in dict(value).items()}

    def _safe_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Mapping):
            return {str(k): self._safe_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._safe_value(item) for item in value]
        return str(value)


if __name__ == "__main__":
    print("\n=== Running Workflow Control ===\n")
    printer.status("TEST", "Workflow Control initialized", "info")
    from ..collaborative.shared_memory import SharedMemory

    class DemoHandlerBridge:
        def perform_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "status": "ok",
                "route": "handler",
                "received_error_type": task_data.get("error_info", {}).get("error_type"),
            }

    class DemoSafetyBridge:
        def perform_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "status": "reviewed",
                "route": "safety",
                "finding_count": len(task_data.get("findings", [])),
            }

    shared = SharedMemory()
    workflow = WorkflowControl(
        shared_memory=shared,
        handler_bridge=DemoHandlerBridge(),
        safety_bridge=DemoSafetyBridge(),
    )

    findings = [
        {
            "checker": "structural",
            "verdict": "warn",
            "severity": "high",
            "confidence": 0.88,
            "error_type": QualityErrorType.REQUIRED_FIELD_MISSING.value,
            "message": "Required field 'country' missing for several records.",
            "flags": ["missing_required_fields"],
            "remediation_actions": ["impute_required_fields", "quarantine_incomplete_records"],
            "affected_record_ids": ["rec-002", "rec-005"],
        },
        {
            "checker": "statistical",
            "verdict": "warn",
            "severity": "medium",
            "confidence": 0.81,
            "error_type": QualityErrorType.DISTRIBUTION_DRIFT_DETECTED.value,
            "message": "Recent batch drifted against the trusted baseline.",
            "flags": ["distribution_shift"],
            "remediation_actions": ["compare_to_recent_baseline", "monitor_next_batches"],
            "affected_record_ids": ["rec-005", "rec-009"],
        },
        {
            "checker": "semantic",
            "verdict": "block",
            "severity": "critical",
            "confidence": 0.97,
            "error_type": ERR_PRIVACY_CONFLICT.value,
            "message": "PII-bearing record violates privacy policy and must not reach downstream systems.",
            "flags": ["privacy_conflict", "manual_review_required"],
            "remediation_actions": ["quarantine_records", "manual_attestation", "verify_lineage"],
            "affected_record_ids": ["rec-009"],
        },
    ]

    records = [
        {"record_id": "rec-001", "text": "clean", "country": "NL"},
        {"record_id": "rec-002", "text": "missing country"},
        {"record_id": "rec-005", "text": "drift candidate"},
        {"record_id": "rec-009", "text": "contains sensitive personal information"},
    ]

    print("\n* * * * * Phase 1: remediation plan generation * * * * *\n")
    remediation = workflow.emit_remediation_suggestions(
        findings=findings,
        verdict="block",
        context={"source_id": "source_alpha", "batch_id": "batch_demo_001"},
    )
    printer.pretty("REMEDIATION", remediation, "success")

    print("\n* * * * * Phase 2: quarantine preparation * * * * *\n")
    quarantine = workflow.quarantine_records(
        source_id="source_alpha",
        batch_id="batch_demo_001",
        findings=findings,
        verdict="block",
        records=records,
        context={"origin": "unit_test"},
        shared_memory=shared,
    )
    printer.pretty("QUARANTINE", quarantine, "success")

    print("\n* * * * * Phase 3: full workflow coordination * * * * *\n")
    decision = workflow.coordinate_batch(
        source_id="source_alpha",
        batch_id="batch_demo_001",
        findings=findings,
        records=records,
        context={"origin": "unit_test", "ingestion_path": "reader->quality_agent"},
    )
    printer.pretty("DECISION", decision, "success")

    stored_decision = workflow.latest_decision("batch_demo_001")
    printer.pretty("LATEST DECISION", stored_decision, "success")

    memory_key = f"{workflow.shared_decision_key_prefix}:batch_demo_001"
    shared_state = shared.get(memory_key)
    printer.pretty("SHARED MEMORY STATE", shared_state, "success")

    summary = workflow.summary()
    printer.pretty("SUMMARY", summary, "success")

    assert decision["verdict"] == "block"
    assert len(decision["quarantine_entries"]) >= 1
    assert len(decision["route_records"]) == 2
    assert shared_state is not None
    assert summary["route_events"] >= 2

    print("\n=== Test ran successfully ===\n")
