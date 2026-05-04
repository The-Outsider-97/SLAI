"""
- Stores canonical quality snapshots per batch/source/window.
- Persists drift baselines, schema versions, and historical threshold decisions.
- Tracks remediation outcomes (what fix was applied, and whether it worked).
- Supports conflict reconciliation when multiple quality checks disagree.
- Exposes retrieval APIs for:
    - latest_quality_state(source_id)
    - historical_drift(source_id, metric, window)
    - remediation_effectiveness(rule_id)
"""

from __future__ import annotations

import json
import os
import tempfile
import time

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .utils.config_loader import load_global_config, get_config_section
from .utils.quality_error import ( ThresholdConfigurationError, QualityMemoryError,
                                  DataQualityError, QualityDisposition, QualityDomain,
                                  QualityErrorType, QualitySeverity, QualityStage,
                                  normalize_quality_exception, quality_error_boundary)
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Quality Memory")
printer = PrettyPrinter


@dataclass(slots=True)
class QualitySnapshot:
    snapshot_id: str
    source_id: str
    batch_id: str
    verdict: str
    batch_score: float
    flags: List[str]
    quarantine_count: int
    shift_metrics: Dict[str, float]
    remediation_actions: List[str]
    source_reliability: float
    schema_version: Optional[str]
    window: Optional[str]
    created_at: float
    shared_memory: Dict[str, Any]
    checker_findings: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["created_at_iso"] = datetime.fromtimestamp(self.created_at, tz=timezone.utc).isoformat()
        return payload


@dataclass(slots=True)
class DriftBaseline:
    baseline_id: str
    source_id: str
    metric: str
    baseline: Dict[str, Any]
    window: Optional[str]
    schema_version: Optional[str]
    recorded_at: float
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["recorded_at_iso"] = datetime.fromtimestamp(self.recorded_at, tz=timezone.utc).isoformat()
        return payload


@dataclass(slots=True)
class DriftObservation:
    observation_id: str
    source_id: str
    metric: str
    observed: float
    baseline_value: Optional[float]
    drift_score: float
    threshold: Optional[float]
    window: Optional[str]
    batch_id: Optional[str]
    is_alert: bool
    recorded_at: float
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["recorded_at_iso"] = datetime.fromtimestamp(self.recorded_at, tz=timezone.utc).isoformat()
        return payload


@dataclass(slots=True)
class SchemaVersionRecord:
    version_id: str
    source_id: str
    schema_version: str
    schema_hash: str
    required_fields: List[str]
    field_types: Dict[str, str]
    is_active: bool
    recorded_at: float
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["recorded_at_iso"] = datetime.fromtimestamp(self.recorded_at, tz=timezone.utc).isoformat()
        return payload


@dataclass(slots=True)
class ThresholdDecision:
    decision_id: str
    rule_id: str
    metric: str
    threshold: float
    rationale: str
    approved_by: Optional[str]
    source_id: Optional[str]
    recorded_at: float
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["recorded_at_iso"] = datetime.fromtimestamp(self.recorded_at, tz=timezone.utc).isoformat()
        return payload


@dataclass(slots=True)
class RemediationOutcome:
    outcome_id: str
    rule_id: str
    action: str
    success: bool
    source_id: Optional[str]
    batch_id: Optional[str]
    before_score: Optional[float]
    after_score: Optional[float]
    affected_records: int
    duration_ms: Optional[float]
    notes: Optional[str]
    recorded_at: float
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def uplift(self) -> Optional[float]:
        if self.before_score is None or self.after_score is None:
            return None
        return self.after_score - self.before_score

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["uplift"] = self.uplift
        payload["recorded_at_iso"] = datetime.fromtimestamp(self.recorded_at, tz=timezone.utc).isoformat()
        return payload


@dataclass(slots=True)
class ConflictResolutionRecord:
    conflict_id: str
    scope_key: str
    source_id: Optional[str]
    batch_id: Optional[str]
    reconciled_verdict: str
    weighted_score: float
    rationale: str
    findings: List[Dict[str, Any]]
    recorded_at: float
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["recorded_at_iso"] = datetime.fromtimestamp(self.recorded_at, tz=timezone.utc).isoformat()
        return payload


@dataclass(slots=True)
class SourceReliabilityRecord:
    event_id: str
    source_id: str
    reliability: float
    reason: Optional[str]
    recorded_at: float
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["recorded_at_iso"] = datetime.fromtimestamp(self.recorded_at, tz=timezone.utc).isoformat()
        return payload


class QualityMemory:
    """Durable, source-aware memory subsystem for the Data Quality Agent.

    This module is intentionally stateful. It tracks current and historical quality
    decisions while remaining JSON-serializable so that the same state can be used
    for persistence, audit export, and shared-memory handoff to sibling agents.
    """

    def __init__(self) -> None:
        self.config = load_global_config()
        self.memory_config = get_config_section("quality_memory")
        self._lock = RLock()
        self._state = self._empty_state()
        self._write_counter = 0

        self.enabled = bool(self.memory_config.get("enabled", True))
        self.backend = str(self.memory_config.get("backend", "hybrid")).strip().lower() or "hybrid"
        self.auto_persist = bool(self.memory_config.get("auto_persist", True))
        self.bootstrap_on_start = bool(self.memory_config.get("bootstrap_on_start", True))
        self.persist_every_n_writes = int(self.memory_config.get("persist_every_n_writes", 1))
        self.journaling_enabled = bool(self.memory_config.get("journaling", {}).get("enabled", True))
        self.fsync_on_write = bool(self.memory_config.get("persistence", {}).get("fsync_on_write", False))

        self.snapshot_limits = {
            "snapshots_per_source": self._positive_int(
                self.memory_config.get("retention", {}).get("max_snapshots_per_source", 2000),
                "retention.max_snapshots_per_source",
            ),
            "drift_baselines_per_metric": self._positive_int(
                self.memory_config.get("retention", {}).get("max_drift_baselines_per_metric", 250),
                "retention.max_drift_baselines_per_metric",
            ),
            "drift_observations_per_metric": self._positive_int(
                self.memory_config.get("retention", {}).get("max_drift_observations_per_metric", 2000),
                "retention.max_drift_observations_per_metric",
            ),
            "schema_versions_per_source": self._positive_int(
                self.memory_config.get("retention", {}).get("max_schema_versions_per_source", 200),
                "retention.max_schema_versions_per_source",
            ),
            "threshold_decisions_per_rule": self._positive_int(
                self.memory_config.get("retention", {}).get("max_threshold_decisions_per_rule", 250),
                "retention.max_threshold_decisions_per_rule",
            ),
            "remediation_outcomes_per_rule": self._positive_int(
                self.memory_config.get("retention", {}).get("max_remediation_outcomes_per_rule", 1000),
                "retention.max_remediation_outcomes_per_rule",
            ),
            "conflicts_per_scope": self._positive_int(
                self.memory_config.get("retention", {}).get("max_conflicts_per_scope", 500),
                "retention.max_conflicts_per_scope",
            ),
            "source_reliability_events": self._positive_int(
                self.memory_config.get("retention", {}).get("max_source_reliability_events", 1000),
                "retention.max_source_reliability_events",
            ),
        }

        self.score_bounds = self.memory_config.get("validation", {}).get("score_bounds", {"min": 0.0, "max": 1.0})
        self.default_window = self.memory_config.get("defaults", {}).get("default_window", "latest")
        self.default_verdict = self.memory_config.get("defaults", {}).get("default_verdict", "warn")
        self.default_source_reliability = float(
            self.memory_config.get("defaults", {}).get("default_source_reliability", 0.5)
        )
        self.default_conflict_strategy = str(
            self.memory_config.get("conflict_resolution", {}).get("default_strategy", "weighted_consensus")
        ).strip().lower() or "weighted_consensus"

        self._project_root = self._resolve_project_root()
        self._storage_dir = self._resolve_storage_path(
            self.memory_config.get("persistence", {}).get("storage_dir", "quality/storage/quality_memory")
        )
        self._state_file = self._storage_dir / self.memory_config.get("persistence", {}).get("state_filename", "quality_memory_state.json")
        self._journal_file = self._storage_dir / self.memory_config.get("persistence", {}).get("journal_filename", "quality_memory_journal.jsonl")

        self._validate_runtime_configuration()
        self._ensure_storage_paths()

        if self.enabled and self.bootstrap_on_start:
            self._bootstrap_from_disk()

        logger.info(
            "Quality Memory initialized | backend=%s | storage=%s | auto_persist=%s",
            self.backend,
            self._storage_dir,
            self.auto_persist,
        )

    # ---------------------------------------------------------------------
    # Public write APIs
    # ---------------------------------------------------------------------
    def record_quality_snapshot(
        self,
        *,
        source_id: str,
        batch_id: str,
        batch_score: float,
        verdict: Optional[str] = None,
        flags: Optional[Sequence[str]] = None,
        quarantine_count: int = 0,
        shift_metrics: Optional[Mapping[str, Any]] = None,
        remediation_actions: Optional[Sequence[str]] = None,
        source_reliability: Optional[float] = None,
        schema_version: Optional[str] = None,
        window: Optional[str] = None,
        checker_findings: Optional[Sequence[Mapping[str, Any]]] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._boundary(
            operation="record_quality_snapshot",
            stage=QualityStage.PERSISTENCE,
            context={"source_id": source_id, "batch_id": batch_id},
        ):
            normalized_score = self._bounded_score(batch_score, field_name="batch_score")
            normalized_reliability = self._bounded_score(
                self.default_source_reliability if source_reliability is None else source_reliability,
                field_name="source_reliability",
            )
            snapshot = QualitySnapshot(
                snapshot_id=self._new_id("snapshot"),
                source_id=self._nonempty(source_id, "source_id"),
                batch_id=self._nonempty(batch_id, "batch_id"),
                verdict=self._normalize_verdict(verdict or self.default_verdict),
                batch_score=normalized_score,
                flags=self._string_list(flags),
                quarantine_count=self._nonnegative_int(quarantine_count, "quarantine_count"),
                shift_metrics=self._float_mapping(shift_metrics),
                remediation_actions=self._string_list(remediation_actions),
                source_reliability=normalized_reliability,
                schema_version=schema_version,
                window=window or self.default_window,
                created_at=time.time(),
                shared_memory={
                    "data_quality.batch_score": normalized_score,
                    "data_quality.flags": self._string_list(flags),
                    "data_quality.quarantine_count": self._nonnegative_int(quarantine_count, "quarantine_count"),
                    "data_quality.shift_metrics": self._float_mapping(shift_metrics),
                    "data_quality.remediation_actions": self._string_list(remediation_actions),
                    "data_quality.source_reliability": normalized_reliability,
                },
                checker_findings=self._mapping_list(checker_findings),
                context=self._normalized_mapping(context),
            )

            with self._lock:
                source_history = self._state["snapshots_by_source"].setdefault(snapshot.source_id, [])
                source_history.append(snapshot.to_dict())
                self._trim_history(source_history, self.snapshot_limits["snapshots_per_source"])
                self._state["snapshots_by_batch"][snapshot.batch_id] = snapshot.to_dict()
                self._touch_state()
                self._record_mutation(
                    event_type="quality_snapshot_recorded",
                    payload=snapshot.to_dict(),
                )
                return deepcopy(snapshot.to_dict())

    def register_schema_version(
        self,
        *,
        source_id: str,
        schema_version: str,
        schema_hash: str,
        required_fields: Optional[Sequence[str]] = None,
        field_types: Optional[Mapping[str, Any]] = None,
        is_active: bool = True,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._boundary(
            operation="register_schema_version",
            stage=QualityStage.PERSISTENCE,
            context={"source_id": source_id, "schema_version": schema_version},
        ):
            record = SchemaVersionRecord(
                version_id=self._new_id("schema"),
                source_id=self._nonempty(source_id, "source_id"),
                schema_version=self._nonempty(schema_version, "schema_version"),
                schema_hash=self._nonempty(schema_hash, "schema_hash"),
                required_fields=self._string_list(required_fields),
                field_types={str(k): str(v) for k, v in dict(field_types or {}).items()},
                is_active=bool(is_active),
                recorded_at=time.time(),
                context=self._normalized_mapping(context),
            )

            with self._lock:
                entries = self._state["schema_versions"].setdefault(record.source_id, [])
                if record.is_active:
                    for entry in entries:
                        entry["is_active"] = False
                entries.append(record.to_dict())
                self._trim_history(entries, self.snapshot_limits["schema_versions_per_source"])
                self._touch_state()
                self._record_mutation("schema_version_registered", record.to_dict())
                return deepcopy(record.to_dict())

    def record_drift_baseline(
        self,
        *,
        source_id: str,
        metric: str,
        baseline: Mapping[str, Any],
        window: Optional[str] = None,
        schema_version: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._boundary(
            operation="record_drift_baseline",
            stage=QualityStage.BASELINE,
            context={"source_id": source_id, "metric": metric},
        ):
            record = DriftBaseline(
                baseline_id=self._new_id("baseline"),
                source_id=self._nonempty(source_id, "source_id"),
                metric=self._nonempty(metric, "metric"),
                baseline=self._normalized_mapping(baseline),
                window=window or self.default_window,
                schema_version=schema_version,
                recorded_at=time.time(),
                context=self._normalized_mapping(context),
            )

            with self._lock:
                metrics = self._state["drift_baselines"].setdefault(record.source_id, {})
                history = metrics.setdefault(record.metric, [])
                history.append(record.to_dict())
                self._trim_history(history, self.snapshot_limits["drift_baselines_per_metric"])
                self._touch_state()
                self._record_mutation("drift_baseline_recorded", record.to_dict())
                return deepcopy(record.to_dict())

    def record_drift_observation(
        self,
        *,
        source_id: str,
        metric: str,
        observed: float,
        drift_score: float,
        threshold: Optional[float] = None,
        baseline_value: Optional[float] = None,
        window: Optional[str] = None,
        batch_id: Optional[str] = None,
        is_alert: Optional[bool] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._boundary(
            operation="record_drift_observation",
            stage=QualityStage.PROFILING,
            context={"source_id": source_id, "metric": metric, "batch_id": batch_id},
        ):
            observed_value = float(observed)
            drift_value = float(drift_score)
            threshold_value = None if threshold is None else float(threshold)
            alert = bool(is_alert) if is_alert is not None else (
                threshold_value is not None and drift_value >= threshold_value
            )

            record = DriftObservation(
                observation_id=self._new_id("drift"),
                source_id=self._nonempty(source_id, "source_id"),
                metric=self._nonempty(metric, "metric"),
                observed=observed_value,
                baseline_value=None if baseline_value is None else float(baseline_value),
                drift_score=drift_value,
                threshold=threshold_value,
                window=window or self.default_window,
                batch_id=batch_id,
                is_alert=alert,
                recorded_at=time.time(),
                context=self._normalized_mapping(context),
            )

            with self._lock:
                metrics = self._state["drift_observations"].setdefault(record.source_id, {})
                history = metrics.setdefault(record.metric, [])
                history.append(record.to_dict())
                self._trim_history(history, self.snapshot_limits["drift_observations_per_metric"])
                self._touch_state()
                self._record_mutation("drift_observation_recorded", record.to_dict())
                return deepcopy(record.to_dict())

    def record_threshold_decision(
        self,
        *,
        rule_id: str,
        metric: str,
        threshold: float,
        rationale: str,
        approved_by: Optional[str] = None,
        source_id: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._boundary(
            operation="record_threshold_decision",
            stage=QualityStage.SCORING,
            context={"rule_id": rule_id, "metric": metric},
        ):
            threshold_value = float(threshold)
            if threshold_value < 0:
                raise ThresholdConfigurationError(
                    rule_id=rule_id,
                    details="threshold must be non-negative",
                    context={"threshold": threshold_value, "metric": metric},
                )

            record = ThresholdDecision(
                decision_id=self._new_id("threshold"),
                rule_id=self._nonempty(rule_id, "rule_id"),
                metric=self._nonempty(metric, "metric"),
                threshold=threshold_value,
                rationale=self._nonempty(rationale, "rationale"),
                approved_by=approved_by,
                source_id=source_id,
                recorded_at=time.time(),
                context=self._normalized_mapping(context),
            )

            with self._lock:
                history = self._state["threshold_decisions"].setdefault(record.rule_id, [])
                history.append(record.to_dict())
                self._trim_history(history, self.snapshot_limits["threshold_decisions_per_rule"])
                self._touch_state()
                self._record_mutation("threshold_decision_recorded", record.to_dict())
                return deepcopy(record.to_dict())

    def record_remediation_outcome(
        self,
        *,
        rule_id: str,
        action: str,
        success: bool,
        source_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        before_score: Optional[float] = None,
        after_score: Optional[float] = None,
        affected_records: int = 0,
        duration_ms: Optional[float] = None,
        notes: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._boundary(
            operation="record_remediation_outcome",
            stage=QualityStage.REMEDIATION,
            context={"rule_id": rule_id, "action": action, "batch_id": batch_id},
        ):
            record = RemediationOutcome(
                outcome_id=self._new_id("remediation"),
                rule_id=self._nonempty(rule_id, "rule_id"),
                action=self._nonempty(action, "action"),
                success=bool(success),
                source_id=source_id,
                batch_id=batch_id,
                before_score=None if before_score is None else self._bounded_score(before_score, field_name="before_score"),
                after_score=None if after_score is None else self._bounded_score(after_score, field_name="after_score"),
                affected_records=self._nonnegative_int(affected_records, "affected_records"),
                duration_ms=None if duration_ms is None else float(duration_ms),
                notes=None if notes is None else str(notes),
                recorded_at=time.time(),
                context=self._normalized_mapping(context),
            )

            with self._lock:
                history = self._state["remediation_history"].setdefault(record.rule_id, [])
                history.append(record.to_dict())
                self._trim_history(history, self.snapshot_limits["remediation_outcomes_per_rule"])
                self._touch_state()
                self._record_mutation("remediation_outcome_recorded", record.to_dict())
                return deepcopy(record.to_dict())

    def record_source_reliability(
        self,
        *,
        source_id: str,
        reliability: float,
        reason: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._boundary(
            operation="record_source_reliability",
            stage=QualityStage.VALIDATION,
            context={"source_id": source_id},
        ):
            record = SourceReliabilityRecord(
                event_id=self._new_id("source_reliability"),
                source_id=self._nonempty(source_id, "source_id"),
                reliability=self._bounded_score(reliability, field_name="reliability"),
                reason=None if reason is None else str(reason),
                recorded_at=time.time(),
                context=self._normalized_mapping(context),
            )

            with self._lock:
                history = self._state["source_reliability_history"].setdefault(record.source_id, [])
                history.append(record.to_dict())
                self._trim_history(history, self.snapshot_limits["source_reliability_events"])
                self._touch_state()
                self._record_mutation("source_reliability_recorded", record.to_dict())
                return deepcopy(record.to_dict())

    def reconcile_conflicts(
        self,
        *,
        findings: Sequence[Mapping[str, Any]],
        source_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        scope_key: Optional[str] = None,
        strategy: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._boundary(
            operation="reconcile_conflicts",
            stage=QualityStage.ROUTING,
            context={"source_id": source_id, "batch_id": batch_id, "scope_key": scope_key},
            error_type=QualityErrorType.QUALITY_MEMORY_CONFLICT,
            severity=QualitySeverity.HIGH,
            retryable=False,
            remediation="Review the conflicting findings, align policies, and re-run the reconciliation workflow.",
            disposition=QualityDisposition.ESCALATE,
        ):
            normalized_findings = self._mapping_list(findings)
            if not normalized_findings:
                raise DataQualityError(
                    message="Conflict reconciliation requires at least one finding",
                    error_type=QualityErrorType.QUALITY_MEMORY_CONFLICT,
                    severity=QualitySeverity.HIGH,
                    retryable=False,
                    stage=QualityStage.ROUTING,
                    domain=QualityDomain.MEMORY,
                    disposition=QualityDisposition.ESCALATE,
                    source_id=source_id,
                    batch_id=batch_id,
                    context={"scope_key": scope_key, **self._normalized_mapping(context)},
                    remediation="Supply at least one normalized finding before attempting reconciliation.",
                )

            strategy_name = (strategy or self.default_conflict_strategy).strip().lower()
            if strategy_name != "weighted_consensus":
                raise DataQualityError(
                    message=f"Unsupported conflict resolution strategy '{strategy_name}'",
                    error_type=QualityErrorType.CONFIGURATION_INVALID,
                    severity=QualitySeverity.HIGH,
                    retryable=False,
                    stage=QualityStage.ROUTING,
                    domain=QualityDomain.SYSTEM,
                    disposition=QualityDisposition.ESCALATE,
                    source_id=source_id,
                    batch_id=batch_id,
                    context={"strategy": strategy_name},
                    remediation="Use a supported strategy or extend the conflict resolver implementation.",
                )

            resolved_scope_key = scope_key or self._build_scope_key(source_id=source_id, batch_id=batch_id)
            verdict_rank = self.memory_config.get("conflict_resolution", {}).get(
                "verdict_rank", {"pass": 0, "warn": 1, "block": 2}
            )
            severity_rank = self.memory_config.get("conflict_resolution", {}).get(
                "severity_rank", {"low": 1, "medium": 2, "high": 3, "critical": 4}
            )
            checker_weight = self.memory_config.get("conflict_resolution", {}).get("checker_weight", {})
            block_on_critical = bool(self.memory_config.get("conflict_resolution", {}).get("block_on_critical", True))
            quarantine_on_high_disagreement = bool(
                self.memory_config.get("conflict_resolution", {}).get("quarantine_on_high_disagreement", True)
            )

            weighted_sum = 0.0
            total_weight = 0.0
            saw_critical_blocker = False
            saw_block = False
            rationale_parts: List[str] = []

            for finding in normalized_findings:
                checker = str(finding.get("checker") or finding.get("domain") or "unknown")
                verdict = self._normalize_verdict(str(finding.get("verdict", "warn")))
                severity = str(finding.get("severity", "medium")).strip().lower()
                confidence = float(finding.get("confidence", 1.0))
                if confidence < 0:
                    confidence = 0.0
                weight = float(checker_weight.get(checker, checker_weight.get("default", 1.0))) * max(confidence, 0.0)
                weighted_sum += float(verdict_rank.get(verdict, 1)) * weight
                total_weight += weight
                rationale_parts.append(f"{checker}:{verdict}@{confidence:.2f}")
                if verdict == "block":
                    saw_block = True
                    if severity_rank.get(severity, 2) >= severity_rank.get("critical", 4):
                        saw_critical_blocker = True

            average_rank = weighted_sum / total_weight if total_weight > 0 else float(verdict_rank.get("warn", 1))

            if block_on_critical and saw_critical_blocker:
                reconciled_verdict = "block"
                rationale = "Critical blocker present; escalation policy forced final block verdict."
            elif saw_block and quarantine_on_high_disagreement and len({str(item.get('verdict', 'warn')) for item in normalized_findings}) > 1:
                reconciled_verdict = "warn"
                rationale = "Block-level disagreement detected across checks; returning warn so workflow control can quarantine or review."
            elif average_rank >= 1.5:
                reconciled_verdict = "block"
                rationale = "Weighted consensus exceeded the block threshold."
            elif average_rank >= 0.5:
                reconciled_verdict = "warn"
                rationale = "Weighted consensus indicates elevated risk but not a hard block."
            else:
                reconciled_verdict = "pass"
                rationale = "Weighted consensus indicates the findings are acceptable for downstream use."

            record = ConflictResolutionRecord(
                conflict_id=self._new_id("conflict"),
                scope_key=resolved_scope_key,
                source_id=source_id,
                batch_id=batch_id,
                reconciled_verdict=reconciled_verdict,
                weighted_score=average_rank,
                rationale=f"{rationale} Inputs={'; '.join(rationale_parts)}",
                findings=normalized_findings,
                recorded_at=time.time(),
                context=self._normalized_mapping(context),
            )

            with self._lock:
                history = self._state["conflict_history"].setdefault(record.scope_key, [])
                history.append(record.to_dict())
                self._trim_history(history, self.snapshot_limits["conflicts_per_scope"])
                self._touch_state()
                self._record_mutation("conflict_reconciled", record.to_dict())
                return deepcopy(record.to_dict())

    # ---------------------------------------------------------------------
    # Public read APIs
    # ---------------------------------------------------------------------
    def latest_quality_state(self, source_id: str) -> Optional[Dict[str, Any]]:
        with self._boundary(
            operation="latest_quality_state",
            stage=QualityStage.RETRIEVAL,
            context={"source_id": source_id},
        ):
            source_key = self._nonempty(source_id, "source_id")
            with self._lock:
                history = self._state["snapshots_by_source"].get(source_key, [])
                if not history:
                    return None
                return deepcopy(history[-1])

    def get_batch_snapshot(self, batch_id: str) -> Optional[Dict[str, Any]]:
        with self._boundary(
            operation="get_batch_snapshot",
            stage=QualityStage.RETRIEVAL,
            context={"batch_id": batch_id},
        ):
            batch_key = self._nonempty(batch_id, "batch_id")
            with self._lock:
                snapshot = self._state["snapshots_by_batch"].get(batch_key)
                return None if snapshot is None else deepcopy(snapshot)

    def latest_schema_version(self, source_id: str) -> Optional[Dict[str, Any]]:
        with self._boundary(
            operation="latest_schema_version",
            stage=QualityStage.RETRIEVAL,
            context={"source_id": source_id},
        ):
            source_key = self._nonempty(source_id, "source_id")
            with self._lock:
                history = self._state["schema_versions"].get(source_key, [])
                if not history:
                    return None
                active = [item for item in history if item.get("is_active")]
                selected = active[-1] if active else history[-1]
                return deepcopy(selected)

    def latest_threshold_decision(self, rule_id: str) -> Optional[Dict[str, Any]]:
        with self._boundary(
            operation="latest_threshold_decision",
            stage=QualityStage.RETRIEVAL,
            context={"rule_id": rule_id},
        ):
            rule_key = self._nonempty(rule_id, "rule_id")
            with self._lock:
                history = self._state["threshold_decisions"].get(rule_key, [])
                if not history:
                    return None
                return deepcopy(history[-1])

    def latest_source_reliability(self, source_id: str) -> Optional[Dict[str, Any]]:
        with self._boundary(
            operation="latest_source_reliability",
            stage=QualityStage.RETRIEVAL,
            context={"source_id": source_id},
        ):
            source_key = self._nonempty(source_id, "source_id")
            with self._lock:
                history = self._state["source_reliability_history"].get(source_key, [])
                if not history:
                    return None
                return deepcopy(history[-1])

    def historical_drift(self, source_id: str, metric: str, window: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._boundary(
            operation="historical_drift",
            stage=QualityStage.RETRIEVAL,
            context={"source_id": source_id, "metric": metric, "window": window},
        ):
            source_key = self._nonempty(source_id, "source_id")
            metric_key = self._nonempty(metric, "metric")
            cutoff = self._window_cutoff(window)
            with self._lock:
                history = deepcopy(self._state["drift_observations"].get(source_key, {}).get(metric_key, []))
            if cutoff is None:
                return history
            return [item for item in history if float(item.get("recorded_at", 0.0)) >= cutoff]

    def remediation_effectiveness(self, rule_id: str) -> Dict[str, Any]:
        with self._boundary(
            operation="remediation_effectiveness",
            stage=QualityStage.RETRIEVAL,
            context={"rule_id": rule_id},
        ):
            rule_key = self._nonempty(rule_id, "rule_id")
            with self._lock:
                history = deepcopy(self._state["remediation_history"].get(rule_key, []))

            total = len(history)
            if total == 0:
                return {
                    "rule_id": rule_key,
                    "total_runs": 0,
                    "success_count": 0,
                    "success_rate": 0.0,
                    "average_uplift": None,
                    "average_duration_ms": None,
                    "actions": {},
                    "last_outcome": None,
                }

            success_count = sum(1 for item in history if bool(item.get("success")))
            action_breakdown: Dict[str, Dict[str, Any]] = {}
            uplift_values: List[float] = []
            durations: List[float] = []

            for item in history:
                action = str(item.get("action", "unknown"))
                bucket = action_breakdown.setdefault(
                    action,
                    {"count": 0, "success_count": 0, "success_rate": 0.0},
                )
                bucket["count"] += 1
                bucket["success_count"] += 1 if bool(item.get("success")) else 0
                uplift = item.get("uplift")
                if uplift is not None:
                    uplift_values.append(float(uplift))
                duration = item.get("duration_ms")
                if duration is not None:
                    durations.append(float(duration))

            for bucket in action_breakdown.values():
                bucket["success_rate"] = bucket["success_count"] / bucket["count"] if bucket["count"] else 0.0

            return {
                "rule_id": rule_key,
                "total_runs": total,
                "success_count": success_count,
                "success_rate": success_count / total,
                "average_uplift": sum(uplift_values) / len(uplift_values) if uplift_values else None,
                "average_duration_ms": sum(durations) / len(durations) if durations else None,
                "actions": action_breakdown,
                "last_outcome": history[-1],
            }

    def recent_conflicts(self, scope_key: str) -> List[Dict[str, Any]]:
        with self._boundary(
            operation="recent_conflicts",
            stage=QualityStage.RETRIEVAL,
            context={"scope_key": scope_key},
        ):
            scope = self._nonempty(scope_key, "scope_key")
            with self._lock:
                return deepcopy(self._state["conflict_history"].get(scope, []))

    def export_state(self) -> Dict[str, Any]:
        with self._lock:
            return deepcopy(self._state)

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "backend": self.backend,
                "storage_dir": str(self._storage_dir),
                "state_file": str(self._state_file),
                "journal_file": str(self._journal_file),
                "counts": {
                    "sources_with_snapshots": len(self._state["snapshots_by_source"]),
                    "batch_snapshots": len(self._state["snapshots_by_batch"]),
                    "schema_sources": len(self._state["schema_versions"]),
                    "baseline_sources": len(self._state["drift_baselines"]),
                    "drift_sources": len(self._state["drift_observations"]),
                    "threshold_rules": len(self._state["threshold_decisions"]),
                    "remediation_rules": len(self._state["remediation_history"]),
                    "conflict_scopes": len(self._state["conflict_history"]),
                    "reliability_sources": len(self._state["source_reliability_history"]),
                },
                "last_updated_at": self._state["metadata"].get("updated_at"),
                "last_updated_at_iso": self._iso_timestamp(self._state["metadata"].get("updated_at")),
            }

    def flush(self) -> None:
        with self._boundary(operation="flush", stage=QualityStage.PERSISTENCE):
            with self._lock:
                self._persist_state()

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _boundary(
        self,
        *,
        operation: str,
        stage: QualityStage,
        context: Optional[Mapping[str, Any]] = None,
        error_type: Optional[QualityErrorType] = None,
        severity: Optional[QualitySeverity] = None,
        retryable: Optional[bool] = None,
        remediation: Optional[str] = None,
        disposition: Optional[QualityDisposition] = None,
    ):
        return quality_error_boundary(
            stage=stage,
            context={"operation": operation, **dict(context or {})},
            error_type=error_type,
            severity=severity,
            retryable=retryable,
            remediation=remediation,
            disposition=disposition,
        )

    def _validate_runtime_configuration(self) -> None:
        if self.backend not in {"memory", "disk", "hybrid"}:
            raise DataQualityError(
                message=f"Unsupported quality_memory backend '{self.backend}'",
                error_type=QualityErrorType.CONFIGURATION_INVALID,
                severity=QualitySeverity.HIGH,
                retryable=False,
                stage=QualityStage.PERSISTENCE,
                domain=QualityDomain.SYSTEM,
                disposition=QualityDisposition.ESCALATE,
                context={"backend": self.backend},
                remediation="Use one of the supported backends: memory, disk, or hybrid.",
            )

        if self.persist_every_n_writes <= 0:
            raise DataQualityError(
                message="persist_every_n_writes must be greater than zero",
                error_type=QualityErrorType.CONFIGURATION_INVALID,
                severity=QualitySeverity.HIGH,
                retryable=False,
                stage=QualityStage.PERSISTENCE,
                domain=QualityDomain.SYSTEM,
                disposition=QualityDisposition.ESCALATE,
                context={"persist_every_n_writes": self.persist_every_n_writes},
                remediation="Set persistence.persist_every_n_writes to a positive integer.",
            )

        score_min = float(self.score_bounds.get("min", 0.0))
        score_max = float(self.score_bounds.get("max", 1.0))
        if score_min >= score_max:
            raise DataQualityError(
                message="quality_memory.validation.score_bounds.min must be less than max",
                error_type=QualityErrorType.CONFIGURATION_INVALID,
                severity=QualitySeverity.HIGH,
                retryable=False,
                stage=QualityStage.SCORING,
                domain=QualityDomain.SYSTEM,
                disposition=QualityDisposition.ESCALATE,
                context={"score_bounds": self.score_bounds},
                remediation="Correct the configured score bounds so min < max.",
            )

        self._normalize_verdict(self.default_verdict)
        self._bounded_score(self.default_source_reliability, field_name="defaults.default_source_reliability")

    def _empty_state(self) -> Dict[str, Any]:
        now = time.time()
        return {
            "metadata": {
                "memory_version": "1.0.0",
                "created_at": now,
                "updated_at": now,
            },
            "snapshots_by_source": {},
            "snapshots_by_batch": {},
            "schema_versions": {},
            "drift_baselines": {},
            "drift_observations": {},
            "threshold_decisions": {},
            "remediation_history": {},
            "conflict_history": {},
            "source_reliability_history": {},
        }

    def _touch_state(self) -> None:
        self._state["metadata"]["updated_at"] = time.time()

    def _resolve_project_root(self) -> Path:
        config_path = Path(str(self.config.get("__config_path__", "quality/configs/quality_config.yaml"))).resolve()
        parents = list(config_path.parents)
        if len(parents) >= 3:
            return parents[2]
        return config_path.parent

    def _resolve_storage_path(self, raw_path: str) -> Path:
        path = Path(str(raw_path))
        if path.is_absolute():
            return path
        return (self._project_root / path).resolve()

    def _ensure_storage_paths(self) -> None:
        if self.backend == "memory":
            return
        try:
            self._storage_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise QualityMemoryError(
                "initialize_storage",
                f"unable to create storage directory '{self._storage_dir}': {exc}",
                context={"storage_dir": str(self._storage_dir)},
            ) from exc

    def _bootstrap_from_disk(self) -> None:
        if self.backend == "memory":
            return

        with self._lock:
            if self._state_file.exists():
                try:
                    with self._state_file.open("r", encoding="utf-8") as handle:
                        loaded = json.load(handle)
                    if not isinstance(loaded, dict):
                        raise ValueError("state file must contain a JSON object")
                    self._state = loaded
                    logger.info("Quality Memory state restored from %s", self._state_file)
                    return
                except Exception as exc:
                    normalized = normalize_quality_exception(
                        exc,
                        stage=QualityStage.PERSISTENCE,
                        context={"operation": "bootstrap_state", "state_file": str(self._state_file)},
                        error_type=QualityErrorType.QUALITY_MEMORY_UNAVAILABLE,
                        severity=QualitySeverity.CRITICAL,
                        retryable=True,
                        remediation="Repair or replace the persisted state file, then replay the journal if available.",
                        disposition=QualityDisposition.FALLBACK,
                    )
                    normalized.report()
                    logger.error("Quality Memory bootstrap failed: %s", normalized.to_json())

            if self.journaling_enabled and self._journal_file.exists():
                try:
                    self._rebuild_state_from_journal()
                    logger.info("Quality Memory state rebuilt from journal %s", self._journal_file)
                except Exception as exc:
                    raise QualityMemoryError(
                        "bootstrap_from_journal",
                        str(exc),
                        context={"journal_file": str(self._journal_file)},
                    ) from exc

    def _rebuild_state_from_journal(self) -> None:
        self._state = self._empty_state()
        with self._journal_file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise QualityMemoryError(
                        "rebuild_from_journal",
                        f"invalid journal JSON on line {line_number}: {exc}",
                    ) from exc
                self._apply_event(event)

    def _apply_event(self, event: Mapping[str, Any]) -> None:
        event_type = str(event.get("event_type", "")).strip()
        payload = deepcopy(dict(event.get("payload") or {}))
        if not event_type:
            raise QualityMemoryError("apply_event", "journal event missing event_type")

        if event_type == "quality_snapshot_recorded":
            source_history = self._state["snapshots_by_source"].setdefault(str(payload["source_id"]), [])
            source_history.append(payload)
            self._trim_history(source_history, self.snapshot_limits["snapshots_per_source"])
            self._state["snapshots_by_batch"][str(payload["batch_id"])] = payload
        elif event_type == "schema_version_registered":
            entries = self._state["schema_versions"].setdefault(str(payload["source_id"]), [])
            if payload.get("is_active"):
                for entry in entries:
                    entry["is_active"] = False
            entries.append(payload)
            self._trim_history(entries, self.snapshot_limits["schema_versions_per_source"])
        elif event_type == "drift_baseline_recorded":
            metrics = self._state["drift_baselines"].setdefault(str(payload["source_id"]), {})
            history = metrics.setdefault(str(payload["metric"]), [])
            history.append(payload)
            self._trim_history(history, self.snapshot_limits["drift_baselines_per_metric"])
        elif event_type == "drift_observation_recorded":
            metrics = self._state["drift_observations"].setdefault(str(payload["source_id"]), {})
            history = metrics.setdefault(str(payload["metric"]), [])
            history.append(payload)
            self._trim_history(history, self.snapshot_limits["drift_observations_per_metric"])
        elif event_type == "threshold_decision_recorded":
            history = self._state["threshold_decisions"].setdefault(str(payload["rule_id"]), [])
            history.append(payload)
            self._trim_history(history, self.snapshot_limits["threshold_decisions_per_rule"])
        elif event_type == "remediation_outcome_recorded":
            history = self._state["remediation_history"].setdefault(str(payload["rule_id"]), [])
            history.append(payload)
            self._trim_history(history, self.snapshot_limits["remediation_outcomes_per_rule"])
        elif event_type == "conflict_reconciled":
            history = self._state["conflict_history"].setdefault(str(payload["scope_key"]), [])
            history.append(payload)
            self._trim_history(history, self.snapshot_limits["conflicts_per_scope"])
        elif event_type == "source_reliability_recorded":
            history = self._state["source_reliability_history"].setdefault(str(payload["source_id"]), [])
            history.append(payload)
            self._trim_history(history, self.snapshot_limits["source_reliability_events"])
        else:
            raise QualityMemoryError("apply_event", f"unsupported event_type '{event_type}'")

        self._touch_state()

    def _record_mutation(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if not self.enabled:
            return

        if self.backend in {"disk", "hybrid"} and self.journaling_enabled:
            self._append_journal_event(event_type=event_type, payload=payload)

        self._write_counter += 1
        if self.auto_persist and self.backend in {"disk", "hybrid"} and self._write_counter >= self.persist_every_n_writes:
            self._persist_state()
            self._write_counter = 0

    def _append_journal_event(self, *, event_type: str, payload: Mapping[str, Any]) -> None:
        event = {
            "event_id": self._new_id("event"),
            "event_type": str(event_type),
            "created_at": time.time(),
            "payload": deepcopy(dict(payload)),
        }
        try:
            with self._journal_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
                if self.fsync_on_write:
                    handle.flush()
                    os.fsync(handle.fileno())
        except Exception as exc:
            raise QualityMemoryError(
                "append_journal_event",
                str(exc),
                context={"journal_file": str(self._journal_file), "event_type": event_type},
            ) from exc

    def _persist_state(self) -> None:
        if self.backend == "memory":
            return
        try:
            serialized = json.dumps(self._state, indent=2, sort_keys=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(self._storage_dir),
                delete=False,
                prefix="quality_memory_",
                suffix=".tmp",
            ) as temp_handle:
                temp_handle.write(serialized)
                temp_handle.flush()
                if self.fsync_on_write:
                    os.fsync(temp_handle.fileno())
                temp_path = Path(temp_handle.name)
            os.replace(temp_path, self._state_file)
        except Exception as exc:
            raise QualityMemoryError(
                "persist_state",
                str(exc),
                context={"state_file": str(self._state_file)},
            ) from exc

    def _window_cutoff(self, window: Optional[str]) -> Optional[float]:
        if window is None:
            return None
        value = str(window).strip().lower()
        if not value or value == "all":
            return None
        if value == "latest":
            return time.time() - 1.0
        if value.endswith("s"):
            return time.time() - float(value[:-1])
        if value.endswith("m"):
            return time.time() - float(value[:-1]) * 60.0
        if value.endswith("h"):
            return time.time() - float(value[:-1]) * 3600.0
        if value.endswith("d"):
            return time.time() - float(value[:-1]) * 86400.0
        if value.endswith("w"):
            return time.time() - float(value[:-1]) * 604800.0
        raise DataQualityError(
            message=f"Unsupported window expression '{window}'",
            error_type=QualityErrorType.CONFIGURATION_INVALID,
            severity=QualitySeverity.MEDIUM,
            retryable=False,
            stage=QualityStage.RETRIEVAL,
            domain=QualityDomain.SYSTEM,
            disposition=QualityDisposition.WARN,
            context={"window": window},
            remediation="Use one of the supported window formats: latest, all, <n>s, <n>m, <n>h, <n>d, or <n>w.",
        )

    def _build_scope_key(self, *, source_id: Optional[str], batch_id: Optional[str]) -> str:
        if source_id and batch_id:
            return f"{source_id}:{batch_id}"
        if batch_id:
            return f"batch:{batch_id}"
        if source_id:
            return f"source:{source_id}"
        return "global"

    def _trim_history(self, history: List[Dict[str, Any]], max_size: int) -> None:
        if max_size <= 0:
            history.clear()
            return
        overflow = len(history) - max_size
        if overflow > 0:
            del history[:overflow]

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}_{int(time.time() * 1000)}_{os.urandom(4).hex()}"

    def _positive_int(self, value: Any, field_name: str) -> int:
        try:
            resolved = int(value)
        except Exception as exc:
            raise ThresholdConfigurationError(
                rule_id=field_name,
                details=f"expected positive integer, received {value!r}",
            ) from exc
        if resolved <= 0:
            raise ThresholdConfigurationError(
                rule_id=field_name,
                details=f"expected positive integer, received {resolved}",
            )
        return resolved

    def _nonnegative_int(self, value: Any, field_name: str) -> int:
        try:
            resolved = int(value)
        except Exception as exc:
            raise DataQualityError(
                message=f"{field_name} must be a non-negative integer",
                error_type=QualityErrorType.CONFIGURATION_INVALID,
                severity=QualitySeverity.MEDIUM,
                retryable=False,
                stage=QualityStage.VALIDATION,
                domain=QualityDomain.SYSTEM,
                disposition=QualityDisposition.WARN,
                context={"field_name": field_name, "value": value},
                remediation="Provide a non-negative integer value.",
                cause=exc,
            ) from exc
        if resolved < 0:
            raise DataQualityError(
                message=f"{field_name} must be a non-negative integer",
                error_type=QualityErrorType.CONFIGURATION_INVALID,
                severity=QualitySeverity.MEDIUM,
                retryable=False,
                stage=QualityStage.VALIDATION,
                domain=QualityDomain.SYSTEM,
                disposition=QualityDisposition.WARN,
                context={"field_name": field_name, "value": value},
                remediation="Provide a non-negative integer value.",
            )
        return resolved

    def _bounded_score(self, value: Any, *, field_name: str) -> float:
        try:
            score = float(value)
        except Exception as exc:
            raise DataQualityError(
                message=f"{field_name} must be numeric",
                error_type=QualityErrorType.CONFIGURATION_INVALID,
                severity=QualitySeverity.MEDIUM,
                retryable=False,
                stage=QualityStage.SCORING,
                domain=QualityDomain.SYSTEM,
                disposition=QualityDisposition.WARN,
                context={"field_name": field_name, "value": value},
                remediation="Provide a numeric value within the configured score bounds.",
                cause=exc,
            ) from exc

        minimum = float(self.score_bounds.get("min", 0.0))
        maximum = float(self.score_bounds.get("max", 1.0))
        if score < minimum or score > maximum:
            raise DataQualityError(
                message=f"{field_name} must be within [{minimum}, {maximum}]",
                error_type=QualityErrorType.POLICY_THRESHOLD_INVALID,
                severity=QualitySeverity.MEDIUM,
                retryable=False,
                stage=QualityStage.SCORING,
                domain=QualityDomain.SYSTEM,
                disposition=QualityDisposition.WARN,
                context={"field_name": field_name, "value": score, "min": minimum, "max": maximum},
                remediation="Adjust the score or widen the configured score bounds if the policy is too strict.",
            )
        return score

    def _nonempty(self, value: Any, field_name: str) -> str:
        text = str(value).strip() if value is not None else ""
        if not text:
            raise DataQualityError(
                message=f"{field_name} must not be empty",
                error_type=QualityErrorType.CONFIGURATION_INVALID,
                severity=QualitySeverity.MEDIUM,
                retryable=False,
                stage=QualityStage.VALIDATION,
                domain=QualityDomain.SYSTEM,
                disposition=QualityDisposition.WARN,
                context={"field_name": field_name, "value": value},
                remediation="Provide a non-empty identifier or field value.",
            )
        return text

    def _string_list(self, values: Optional[Iterable[Any]]) -> List[str]:
        if values is None:
            return []
        return [str(item) for item in values]

    def _float_mapping(self, values: Optional[Mapping[str, Any]]) -> Dict[str, float]:
        if values is None:
            return {}
        normalized: Dict[str, float] = {}
        for key, value in dict(values).items():
            normalized[str(key)] = float(value)
        return normalized

    def _mapping_list(self, values: Optional[Sequence[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
        if values is None:
            return []
        normalized: List[Dict[str, Any]] = []
        for item in values:
            normalized.append(self._normalized_mapping(item))
        return normalized

    def _normalized_mapping(self, value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if value is None:
            return {}
        return json.loads(json.dumps(dict(value), default=str))

    def _normalize_verdict(self, verdict: str) -> str:
        value = str(verdict).strip().lower()
        if value not in {"pass", "warn", "block"}:
            raise DataQualityError(
                message=f"Unsupported quality verdict '{verdict}'",
                error_type=QualityErrorType.CONFIGURATION_INVALID,
                severity=QualitySeverity.MEDIUM,
                retryable=False,
                stage=QualityStage.ROUTING,
                domain=QualityDomain.SYSTEM,
                disposition=QualityDisposition.WARN,
                context={"verdict": verdict},
                remediation="Use one of the supported verdict values: pass, warn, or block.",
            )
        return value

    def _iso_timestamp(self, timestamp_value: Optional[float]) -> Optional[str]:
        if timestamp_value is None:
            return None
        return datetime.fromtimestamp(float(timestamp_value), tz=timezone.utc).isoformat()


if __name__ == "__main__":
    print("\n=== Running Quality Memory ===\n")
    printer.status("TEST", "Quality Memory initialized", "info")

    memory = QualityMemory()
    printer.status("CONFIG", f"Loaded quality_memory config from {memory.config.get('__config_path__', 'unknown')}", "success")

    schema_record = memory.register_schema_version(
        source_id="source_alpha",
        schema_version="v1.2.0",
        schema_hash="schema_hash_9f42",
        required_fields=["id", "text", "label", "timestamp"],
        field_types={"id": "str", "text": "str", "label": "str", "timestamp": "datetime"},
        context={"owner": "Data Quality Agent"},
    )
    printer.pretty("SCHEMA", schema_record, "success")

    baseline_record = memory.record_drift_baseline(
        source_id="source_alpha",
        metric="missing_rate",
        baseline={"mean": 0.02, "std": 0.01, "p95": 0.04},
        window="30d",
        schema_version="v1.2.0",
    )
    printer.pretty("BASELINE", baseline_record, "success")

    threshold_record = memory.record_threshold_decision(
        rule_id="missing_rate_rule",
        metric="missing_rate",
        threshold=0.08,
        rationale="Historical baseline and incident postmortems show 8% as the safe intervention threshold.",
        approved_by="quality_ops",
        source_id="source_alpha",
    )
    printer.pretty("THRESHOLD", threshold_record, "success")

    reliability_record = memory.record_source_reliability(
        source_id="source_alpha",
        reliability=0.94,
        reason="Trusted upstream with stable schema and recent successful remediations.",
    )
    printer.pretty("RELIABILITY", reliability_record, "success")

    snapshot_record = memory.record_quality_snapshot(
        source_id="source_alpha",
        batch_id="batch_2026_04_08_001",
        batch_score=0.91,
        verdict="pass",
        flags=["low_missingness", "trusted_source"],
        quarantine_count=2,
        shift_metrics={"missing_rate": 0.03, "duplicate_rate": 0.01},
        remediation_actions=["drop_invalid_rows", "normalize_timestamp"],
        source_reliability=0.94,
        schema_version="v1.2.0",
        window="24h",
        checker_findings=[
            {"checker": "structural", "verdict": "pass", "severity": "low", "confidence": 0.98},
            {"checker": "statistical", "verdict": "warn", "severity": "medium", "confidence": 0.85},
            {"checker": "semantic", "verdict": "pass", "severity": "low", "confidence": 0.92},
        ],
        context={"ingestion_path": "reader->knowledge_ingestion"},
    )
    printer.pretty("SNAPSHOT", snapshot_record, "success")

    drift_record = memory.record_drift_observation(
        source_id="source_alpha",
        metric="missing_rate",
        observed=0.031,
        baseline_value=0.02,
        drift_score=0.011,
        threshold=0.08,
        window="24h",
        batch_id="batch_2026_04_08_001",
    )
    printer.pretty("DRIFT", drift_record, "success")

    remediation_record = memory.record_remediation_outcome(
        rule_id="missing_rate_rule",
        action="drop_invalid_rows",
        success=True,
        source_id="source_alpha",
        batch_id="batch_2026_04_08_001",
        before_score=0.81,
        after_score=0.91,
        affected_records=14,
        duration_ms=126.4,
        notes="Recovered batch quality without introducing label inconsistency.",
    )
    printer.pretty("REMEDIATION", remediation_record, "success")

    conflict_record = memory.reconcile_conflicts(
        source_id="source_alpha",
        batch_id="batch_2026_04_08_001",
        findings=[
            {"checker": "structural", "verdict": "pass", "severity": "low", "confidence": 0.95},
            {"checker": "statistical", "verdict": "warn", "severity": "high", "confidence": 0.90},
            {"checker": "semantic", "verdict": "pass", "severity": "medium", "confidence": 0.80},
        ],
        context={"reason": "Cross-checking final batch release decision"},
    )
    printer.pretty("CONFLICT", conflict_record, "success")

    latest_state = memory.latest_quality_state("source_alpha")
    drift_history = memory.historical_drift("source_alpha", "missing_rate", "7d")
    remediation_summary = memory.remediation_effectiveness("missing_rate_rule")
    summary = memory.summary()

    assert latest_state is not None, "latest_quality_state should return a snapshot"
    assert latest_state["batch_id"] == "batch_2026_04_08_001"
    assert len(drift_history) >= 1, "historical_drift should include the recorded observation"
    assert remediation_summary["success_rate"] == 1.0, "remediation_effectiveness should compute success rate"
    assert summary["counts"]["sources_with_snapshots"] >= 1, "summary should reflect persisted state"

    printer.pretty("LATEST_STATE", latest_state, "success")
    printer.pretty("DRIFT_HISTORY", drift_history, "success")
    printer.pretty("REMEDIATION_EFFECTIVENESS", remediation_summary, "success")
    printer.pretty("SUMMARY", summary, "success")

    memory.flush()
    printer.status("TEST", "Quality Memory flush completed", "success")
    print("\n=== Test ran successfully ===\n")
