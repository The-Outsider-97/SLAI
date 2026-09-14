"""Trusted baseline lifecycle governance for the SLAI Quality Agent.

Baseline governance answers one narrow question: is a reference profile
trustworthy enough to be used as a baseline? StatisticalQuality remains the
owner of drift measurement and scoring.

Configuration source:
    src/agents/quality/configs/quality_config.yaml -> baseline_governance

QualityMemory may be supplied by dependency injection; this module never imports
or constructs it.
"""
from __future__ import annotations

import time

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.quality_error import *
from ..utils.quality_helpers import *
from logs.logger import get_logger, PrettyPrinter, get_log_queue # pyright: ignore[reportMissingImports]


logger = get_logger("Quality Baseline Governance")
printer = PrettyPrinter()


class BaselineStatus(str, Enum):
    UNESTABLISHED = "unestablished"
    CANDIDATE = "candidate"
    TRUSTED = "trusted"
    ACTIVE = "active"
    STALE = "stale"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


@dataclass(frozen=True, slots=True)
class BaselineDecision:
    source_id: str
    metric: str
    status: BaselineStatus
    usable: bool
    profile: Optional[Dict[str, Any]]
    profile_hash: Optional[str]
    schema_version: Optional[str]
    observed_stable_batches: int
    required_stable_batches: int
    confidence: float
    reason_codes: Tuple[str, ...] = ()
    rationale: Tuple[str, ...] = ()
    persisted_baseline_id: Optional[str] = None
    recorded_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["reason_codes"] = list(self.reason_codes)
        payload["rationale"] = list(self.rationale)
        return payload


class BaselineGovernor:
    """Govern candidate/trusted/active baselines using explicit evidence."""

    def __init__(self, *, memory: Any = None, config: Optional[Mapping[str, Any]] = None) -> None:
        global_config = load_global_config()
        section = get_config_section("baseline_governance", config=global_config)
        if config:
            section.update(dict(config))
        self.config = section
        self.memory = memory

        self.enabled = bool(section.get("enabled", True))
        self.metric_name = nonempty_text(
            section.get("metric_name", "dataset_profile"),
            "baseline_governance.metric_name",
        )
        self.min_stable_batches = self._positive_int(
            section.get("min_stable_batches", 3),
            "baseline_governance.min_stable_batches",
        )
        self.min_source_reliability = bounded_score(
            section.get("min_source_reliability", 0.70),
            field_name="baseline_governance.min_source_reliability",
        )
        self.allowed_structural_verdicts = {
            normalize_verdict(item)
            for item in section.get("allowed_structural_verdicts", ["pass"])
        }
        self.allowed_semantic_verdicts = {
            normalize_verdict(item)
            for item in section.get("allowed_semantic_verdicts", ["pass", "warn"])
        }
        self.max_quality_score_delta = bounded_score(
            section.get("max_quality_score_delta", 0.08),
            field_name="baseline_governance.max_quality_score_delta",
        )
        self.max_baseline_age_seconds = self._positive_int(
            section.get("max_baseline_age_seconds", 2_592_000),
            "baseline_governance.max_baseline_age_seconds",
        )
        self.trust_explicit_baseline = bool(section.get("trust_explicit_baseline", False))
        self.require_schema_match = bool(section.get("require_schema_match", True))
        self.persist_trusted_baselines = bool(section.get("persist_trusted_baselines", True))
        self.allow_export_state_fallback = bool(section.get("allow_export_state_fallback", True))
        self.accept_legacy_baselines = bool(section.get("accept_legacy_baselines", False))
        self.use_memory_snapshot_history = bool(section.get("use_memory_snapshot_history", False))
        self.history_limit = self._positive_int(
            section.get("history_limit", 12),
            "baseline_governance.history_limit",
        )
        self._runtime_active: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def resolve(
        self,
        *,
        source_id: str,
        current_profile: Mapping[str, Any],
        metric: Optional[str] = None,
        explicit_baseline: Optional[Mapping[str, Any]] = None,
        structural_verdict: str = "pass",
        semantic_verdict: str = "pass",
        source_reliability: Optional[float] = None,
        current_quality_score: Optional[float] = None,
        schema_version: Optional[str] = None,
        window: Optional[str] = None,
        history: Optional[Sequence[Mapping[str, Any]]] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> BaselineDecision:
        source_key = nonempty_text(source_id, "source_id")
        metric_key = nonempty_text(metric or self.metric_name, "metric")
        profile = normalized_mapping(current_profile, field_name="current_profile", allow_none=False)
        runtime_context = normalized_mapping(context)

        if not self.enabled:
            return BaselineDecision(
                source_id=source_key,
                metric=metric_key,
                status=BaselineStatus.UNESTABLISHED,
                usable=False,
                profile=None,
                profile_hash=None,
                schema_version=schema_version,
                observed_stable_batches=0,
                required_stable_batches=self.min_stable_batches,
                confidence=0.0,
                reason_codes=("baseline_governance_disabled",),
                rationale=("Baseline governance is disabled by configuration.",),
            )

        # Explicit baselines are not silently trusted unless the caller marks them
        # trusted or configuration opts into that behavior.
        if explicit_baseline is not None:
            explicit = normalized_mapping(explicit_baseline, field_name="explicit_baseline", allow_none=False)
            trusted_marker = bool(runtime_context.get("baseline_trusted", False) or self.trust_explicit_baseline)
            if not self._schema_compatible(
                baseline_schema=self._extract_schema_version(explicit),
                current_schema=schema_version,
            ):
                return self._rejected(
                    source_key,
                    metric_key,
                    explicit,
                    schema_version,
                    "explicit_baseline_schema_mismatch",
                    "Explicit baseline schema is incompatible with the current schema.",
                )
            status = BaselineStatus.ACTIVE if trusted_marker else BaselineStatus.CANDIDATE
            return BaselineDecision(
                source_id=source_key,
                metric=metric_key,
                status=status,
                usable=trusted_marker,
                profile=explicit,
                profile_hash=stable_hash(explicit),
                schema_version=schema_version,
                observed_stable_batches=self.min_stable_batches if trusted_marker else 1,
                required_stable_batches=self.min_stable_batches,
                confidence=1.0 if trusted_marker else 0.55,
                reason_codes=("explicit_baseline_trusted" if trusted_marker else "explicit_baseline_unverified",),
                rationale=(
                    "Caller supplied an explicitly trusted baseline."
                    if trusted_marker
                    else "Caller supplied a baseline, but no explicit trust evidence was provided."
                ,),
            )

        # Prefer an already trusted runtime/persisted baseline.
        existing = self._get_existing_baseline(source_key, metric_key)
        if existing is not None:
            existing_profile = normalized_mapping(existing.get("baseline", existing))
            baseline_schema = existing.get("schema_version")
            recorded_at = float(existing.get("recorded_at", 0.0) or 0.0)
            age = max(0.0, time.time() - recorded_at) if recorded_at else None
            if not self._schema_compatible(baseline_schema=baseline_schema, current_schema=schema_version):
                return BaselineDecision(
                    source_id=source_key,
                    metric=metric_key,
                    status=BaselineStatus.SUPERSEDED,
                    usable=False,
                    profile=existing_profile,
                    profile_hash=stable_hash(existing_profile),
                    schema_version=schema_version,
                    observed_stable_batches=0,
                    required_stable_batches=self.min_stable_batches,
                    confidence=0.25,
                    reason_codes=("baseline_schema_superseded",),
                    rationale=("Stored baseline belongs to an incompatible schema version.",),
                )
            if age is not None and age > self.max_baseline_age_seconds:
                return BaselineDecision(
                    source_id=source_key,
                    metric=metric_key,
                    status=BaselineStatus.STALE,
                    usable=False,
                    profile=existing_profile,
                    profile_hash=stable_hash(existing_profile),
                    schema_version=schema_version,
                    observed_stable_batches=0,
                    required_stable_batches=self.min_stable_batches,
                    confidence=0.35,
                    reason_codes=("baseline_stale",),
                    rationale=(
                        f"Stored baseline age ({age:.0f}s) exceeds the configured maximum "
                        f"({self.max_baseline_age_seconds}s).",
                    ),
                )
            return BaselineDecision(
                source_id=source_key,
                metric=metric_key,
                status=BaselineStatus.ACTIVE,
                usable=True,
                profile=existing_profile,
                profile_hash=stable_hash(existing_profile),
                schema_version=schema_version,
                observed_stable_batches=self.min_stable_batches,
                required_stable_batches=self.min_stable_batches,
                confidence=0.95,
                reason_codes=("trusted_baseline_available",),
                rationale=("A non-stale, schema-compatible trusted baseline is available.",),
                persisted_baseline_id=existing.get("baseline_id"),
            )

        structural = normalize_verdict(structural_verdict)
        semantic = normalize_verdict(semantic_verdict)
        reliability = (
            None
            if source_reliability is None
            else bounded_score(source_reliability, field_name="source_reliability")
        )
        reasons: List[str] = []
        eligible = True
        if structural not in self.allowed_structural_verdicts:
            eligible = False
            reasons.append("structural_verdict_not_eligible")
        if semantic not in self.allowed_semantic_verdicts:
            eligible = False
            reasons.append("semantic_verdict_not_eligible")
        if reliability is None:
            eligible = False
            reasons.append("source_reliability_unavailable")
        elif reliability < self.min_source_reliability:
            eligible = False
            reasons.append("source_reliability_below_floor")

        if not eligible:
            return self._rejected(
                source_key,
                metric_key,
                profile,
                schema_version,
                ",".join(reasons),
                "Current profile is not eligible to seed a trusted baseline.",
            )

        snapshots = self._deduplicated_history(
            history if history is not None else self._get_history(source_key)
        )
        current_score = (
            None
            if current_quality_score is None
            else bounded_score(current_quality_score, field_name="current_quality_score")
        )
        stable_count = self._stable_batch_count(snapshots=snapshots, current_score=current_score)

        if stable_count < self.min_stable_batches:
            return BaselineDecision(
                source_id=source_key,
                metric=metric_key,
                status=BaselineStatus.CANDIDATE,
                usable=False,
                profile=profile,
                profile_hash=stable_hash(profile),
                schema_version=schema_version,
                observed_stable_batches=stable_count,
                required_stable_batches=self.min_stable_batches,
                confidence=min(0.85, 0.40 + 0.45 * (stable_count / self.min_stable_batches)),
                reason_codes=("insufficient_stable_history",),
                rationale=(
                    f"Candidate has {stable_count} stable qualifying batch(es); "
                    f"{self.min_stable_batches} are required.",
                ),
            )

        persisted_id = None
        if self.persist_trusted_baselines:
            persisted = self._persist_baseline(
                source_id=source_key,
                metric=metric_key,
                profile=profile,
                window=window,
                schema_version=schema_version,
                context={
                    **runtime_context,
                    "baseline_governance_status": BaselineStatus.TRUSTED.value,
                    "stable_batch_count": stable_count,
                    "profile_hash": stable_hash(profile),
                },
            )
            if isinstance(persisted, Mapping):
                persisted_id = persisted.get("baseline_id")

        record = {
            "baseline": profile,
            "schema_version": schema_version,
            "recorded_at": time.time(),
            "baseline_id": persisted_id,
        }
        self._runtime_active[(source_key, metric_key)] = record

        return BaselineDecision(
            source_id=source_key,
            metric=metric_key,
            status=BaselineStatus.TRUSTED,
            usable=True,
            profile=profile,
            profile_hash=stable_hash(profile),
            schema_version=schema_version,
            observed_stable_batches=stable_count,
            required_stable_batches=self.min_stable_batches,
            confidence=0.95,
            reason_codes=("candidate_promoted_to_trusted",),
            rationale=(
                "Candidate satisfied structural, semantic, source-reliability, schema, and stability requirements.",
            ),
            persisted_baseline_id=persisted_id,
        )

    def _get_existing_baseline(self, source_id: str, metric: str) -> Optional[Dict[str, Any]]:
        cached = self._runtime_active.get((source_id, metric))
        if cached:
            return dict(cached)
        if self.memory is None:
            return None

        direct = getattr(self.memory, "latest_drift_baseline", None)
        if callable(direct):
            value = direct(source_id, metric)
            if isinstance(value, Mapping) and self._is_governed_baseline(value):
                return dict(value)

        if not self.allow_export_state_fallback:
            return None
        exporter = getattr(self.memory, "export_state", None)
        if not callable(exporter):
            return None
        state = exporter()
        if not isinstance(state, Mapping):
            return None
        baselines = state.get("drift_baselines", {})
        if not isinstance(baselines, Mapping):
            return None
        source_map = baselines.get(source_id, {})
        if not isinstance(source_map, Mapping):
            return None
        history = source_map.get(metric, [])
        if isinstance(history, Sequence):
            for item in reversed(history):
                if isinstance(item, Mapping) and self._is_governed_baseline(item):
                    return dict(item)
        return None

    def _is_governed_baseline(self, record: Mapping[str, Any]) -> bool:
        context = record.get("context", {})
        status = ""
        if isinstance(context, Mapping):
            status = str(context.get("baseline_governance_status", "")).strip().lower()
        if status in {BaselineStatus.TRUSTED.value, BaselineStatus.ACTIVE.value}:
            return True
        return self.accept_legacy_baselines

    def _get_history(self, source_id: str) -> Sequence[Mapping[str, Any]]:
        if self.memory is None or not self.use_memory_snapshot_history:
            return ()
        direct = getattr(self.memory, "recent_quality_states", None)
        if callable(direct):
            value = direct(source_id, limit=self.history_limit)
            if isinstance(value, Sequence):
                return [item for item in value if isinstance(item, Mapping)]

        if self.allow_export_state_fallback:
            exporter = getattr(self.memory, "export_state", None)
            if callable(exporter):
                state = exporter()
                if isinstance(state, Mapping):
                    source_map = state.get("snapshots_by_source", {})
                    if isinstance(source_map, Mapping):
                        history = source_map.get(source_id, [])
                        if isinstance(history, Sequence):
                            return [
                                item for item in history[-self.history_limit :]
                                if isinstance(item, Mapping)
                            ]

        latest = getattr(self.memory, "latest_quality_state", None)
        if callable(latest):
            value = latest(source_id)
            if isinstance(value, Mapping):
                return [value]
        return ()

    def _deduplicated_history(self, history: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        by_batch: Dict[str, Dict[str, Any]] = {}
        no_batch: List[Dict[str, Any]] = []
        for item in history:
            payload = dict(item)
            batch_id = payload.get("batch_id")
            if batch_id is None:
                no_batch.append(payload)
                continue
            key = str(batch_id)
            previous = by_batch.get(key)
            if previous is None or float(payload.get("created_at", 0.0) or 0.0) >= float(
                previous.get("created_at", 0.0) or 0.0
            ):
                by_batch[key] = payload
        combined = list(by_batch.values()) + no_batch
        combined.sort(key=lambda item: float(item.get("created_at", 0.0) or 0.0))
        return combined[-self.history_limit :]

    def _stable_batch_count(
        self,
        *,
        snapshots: Sequence[Mapping[str, Any]],
        current_score: Optional[float],
    ) -> int:
        scores: List[float] = []
        for snapshot in snapshots:
            raw_verdict = snapshot.get("verdict")
            if raw_verdict is not None:
                try:
                    if normalize_verdict(raw_verdict) == "block":
                        continue
                except DataQualityError:
                    continue
            raw_score = snapshot.get("batch_score")
            if raw_score is None:
                continue
            try:
                scores.append(bounded_score(raw_score, field_name="history.batch_score"))
            except DataQualityError:
                continue
        if current_score is not None:
            scores.append(current_score)
        if not scores:
            return 1

        stable_suffix = 1
        for left, right in zip(reversed(scores[:-1]), reversed(scores[1:])):
            if abs(right - left) <= self.max_quality_score_delta:
                stable_suffix += 1
            else:
                break
        return stable_suffix

    def _persist_baseline(
        self,
        *,
        source_id: str,
        metric: str,
        profile: Mapping[str, Any],
        window: Optional[str],
        schema_version: Optional[str],
        context: Mapping[str, Any],
    ) -> Optional[Mapping[str, Any]]:
        if self.memory is None:
            return None
        method = getattr(self.memory, "record_drift_baseline", None)
        if not callable(method):
            return None
        result = method(
            source_id=source_id,
            metric=metric,
            baseline=dict(profile),
            window=window,
            schema_version=schema_version,
            context=dict(context),
        )
        return result if isinstance(result, Mapping) else None

    def _schema_compatible(self, *, baseline_schema: Optional[Any], current_schema: Optional[Any]) -> bool:
        if not self.require_schema_match:
            return True
        if baseline_schema in (None, "") or current_schema in (None, ""):
            return True
        return str(baseline_schema) == str(current_schema)

    @staticmethod
    def _extract_schema_version(profile: Mapping[str, Any]) -> Optional[str]:
        for key in ("schema_version", "version"):
            value = profile.get(key)
            if value not in (None, ""):
                return str(value)
        metadata = profile.get("metadata")
        if isinstance(metadata, Mapping):
            value = metadata.get("schema_version")
            if value not in (None, ""):
                return str(value)
        return None

    def _rejected(
        self,
        source_id: str,
        metric: str,
        profile: Mapping[str, Any],
        schema_version: Optional[str],
        code: str,
        rationale: str,
    ) -> BaselineDecision:
        return BaselineDecision(
            source_id=source_id,
            metric=metric,
            status=BaselineStatus.REJECTED,
            usable=False,
            profile=dict(profile),
            profile_hash=stable_hash(profile),
            schema_version=schema_version,
            observed_stable_batches=0,
            required_stable_batches=self.min_stable_batches,
            confidence=0.20,
            reason_codes=tuple(part.strip() for part in code.split(",") if part.strip()),
            rationale=(rationale,),
        )

    @staticmethod
    def _positive_int(value: Any, field_name: str) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise BaselineGovernor._config_error(
                f"{field_name} must be a positive integer",
                context={"value": value},
                cause=exc,
            ) from exc
        if parsed <= 0:
            raise BaselineGovernor._config_error(
                f"{field_name} must be > 0",
                context={"value": value},
            )
        return parsed

    @staticmethod
    def _config_error(
        message: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> DataQualityError:
        return DataQualityError(
            message=message,
            error_type=QualityErrorType.CONFIGURATION_INVALID,
            severity=QualitySeverity.HIGH,
            retryable=False,
            stage=QualityStage.VALIDATION,
            domain=QualityDomain.SYSTEM,
            disposition=QualityDisposition.ESCALATE,
            context=dict(context or {}),
            remediation="Correct baseline_governance in quality_config.yaml.",
            cause=cause,
        )


__all__ = ["BaselineStatus", "BaselineDecision", "BaselineGovernor"]