"""Relationship and joint-structure quality analysis for SLAI Quality.

The module detects relationship drift that marginal per-field statistics can
miss. Initial scope is intentionally bounded to pairwise numeric correlation and
missingness-correlation drift with support-aware Fisher r-to-z evidence.

It does not classify relationships as label leakage; SemanticQuality owns that
responsibility.

Configuration source:
    src/agents/quality/configs/quality_config.yaml -> relationship_quality

QualityMemory may be supplied by dependency injection; this module never imports
or constructs it.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.quality_error import *
from ..utils.quality_helpers import *
from logs.logger import get_logger, PrettyPrinter, get_log_queue # pyright: ignore[reportMissingImports]


logger = get_logger("Relationship Quality")
printer = PrettyPrinter()


@dataclass(frozen=True, slots=True)
class RelationshipFinding:
    pair: Tuple[str, str]
    relationship_type: str
    verdict: str
    severity: str
    score: float
    confidence: float
    current_value: float
    baseline_value: float
    absolute_delta: float
    current_support: int
    baseline_support: int
    fisher_z: Optional[float]
    p_value: Optional[float]
    statistically_supported: bool
    flags: Tuple[str, ...] = ()
    remediation_actions: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["pair"] = list(self.pair)
        payload["flags"] = list(self.flags)
        payload["remediation_actions"] = list(self.remediation_actions)
        return payload


@dataclass(frozen=True, slots=True)
class RelationshipAssessment:
    source_id: str
    batch_id: str
    verdict: str
    batch_score: float
    confidence: float
    profile: Dict[str, Any]
    baseline_available: bool
    evaluated_pairs: int
    skipped_pairs: int
    findings: Tuple[RelationshipFinding, ...]
    flags: Tuple[str, ...]
    remediation_actions: Tuple[str, ...]
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "batch_id": self.batch_id,
            "verdict": self.verdict,
            "batch_score": self.batch_score,
            "confidence": self.confidence,
            "profile": self.profile,
            "baseline_available": self.baseline_available,
            "evaluated_pairs": self.evaluated_pairs,
            "skipped_pairs": self.skipped_pairs,
            "findings": [item.to_dict() for item in self.findings],
            "flags": list(self.flags),
            "remediation_actions": list(self.remediation_actions),
            "generated_at": self.generated_at,
        }


class RelationshipQuality:
    """Support-aware relationship drift detector with bounded pair growth."""

    def __init__(self, *, memory: Any = None, config: Optional[Mapping[str, Any]] = None) -> None:
        global_config = load_global_config()
        section = get_config_section("relationship_quality", config=global_config)
        if config:
            section.update(dict(config))
        self.config = section
        self.memory = memory

        self.enabled = bool(section.get("enabled", True))
        self.max_numeric_fields = self._positive_int(
            section.get("max_numeric_fields", 30),
            "relationship_quality.max_numeric_fields",
        )
        self.min_pair_observations = self._positive_int(
            section.get("min_pair_observations", 20),
            "relationship_quality.min_pair_observations",
        )
        self.min_absolute_baseline_correlation = self._bounded_nonnegative(
            section.get("min_absolute_baseline_correlation", 0.20),
            maximum=1.0,
            field_name="relationship_quality.min_absolute_baseline_correlation",
        )
        self.warn_correlation_delta = self._bounded_nonnegative(
            section.get("warn_correlation_delta", 0.20),
            maximum=2.0,
            field_name="relationship_quality.warn_correlation_delta",
        )
        self.block_correlation_delta = self._bounded_nonnegative(
            section.get("block_correlation_delta", 0.40),
            maximum=2.0,
            field_name="relationship_quality.block_correlation_delta",
        )
        self.warn_missingness_delta = self._bounded_nonnegative(
            section.get("warn_missingness_delta", 0.20),
            maximum=2.0,
            field_name="relationship_quality.warn_missingness_delta",
        )
        self.block_missingness_delta = self._bounded_nonnegative(
            section.get("block_missingness_delta", 0.40),
            maximum=2.0,
            field_name="relationship_quality.block_missingness_delta",
        )
        if self.warn_correlation_delta > self.block_correlation_delta:
            raise self._config_error("warn_correlation_delta must be <= block_correlation_delta")
        if self.warn_missingness_delta > self.block_missingness_delta:
            raise self._config_error("warn_missingness_delta must be <= block_missingness_delta")
        self.significance_alpha = self._bounded_nonnegative(
            section.get("significance_alpha", 0.05),
            maximum=1.0,
            field_name="relationship_quality.significance_alpha",
        )
        if self.significance_alpha <= 0.0:
            raise self._config_error("significance_alpha must be > 0")
        self.require_statistical_support_for_block = bool(
            section.get("require_statistical_support_for_block", True)
        )
        self.exclude_fields = {
            str(item)
            for item in section.get(
                "exclude_fields",
                ["record_id", "id", "uuid", "created_at", "updated_at", "timestamp"],
            )
        }
        self.pass_threshold = bounded_score(
            section.get("pass_threshold", 0.90),
            field_name="relationship_quality.pass_threshold",
        )
        self.warn_threshold = bounded_score(
            section.get("warn_threshold", 0.75),
            field_name="relationship_quality.warn_threshold",
        )
        if self.warn_threshold > self.pass_threshold:
            raise self._config_error("relationship_quality.warn_threshold must be <= pass_threshold")
        self.profile_metric_name = nonempty_text(
            section.get("profile_metric_name", "relationship_profile"),
            "relationship_quality.profile_metric_name",
        )
        remediation = section.get("remediation_actions", {})
        if not isinstance(remediation, Mapping):
            raise self._config_error("relationship_quality.remediation_actions must be a mapping")
        self.warn_actions = tuple(
            str(item)
            for item in remediation.get(
                "warn",
                ["review_feature_relationships", "compare_recent_batches"],
            )
        )
        self.block_actions = tuple(
            str(item)
            for item in remediation.get(
                "block",
                [
                    "quarantine_relationship_shifted_batch",
                    "revalidate_upstream_transformations",
                    "rebuild_trusted_baseline",
                ],
            )
        )

    def build_profile(self, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        prepared = normalize_records(records, require_nonempty=True)
        fields = self._select_numeric_fields(prepared)
        numeric_pairs: Dict[str, Dict[str, Any]] = {}
        missingness_pairs: Dict[str, Dict[str, Any]] = {}

        for left, right in combinations(fields, 2):
            values = [
                (float(record[left]), float(record[right]))
                for record in prepared
                if self._is_numeric(record.get(left)) and self._is_numeric(record.get(right))
            ]
            if len(values) >= self.min_pair_observations:
                x = [item[0] for item in values]
                y = [item[1] for item in values]
                correlation = self._pearson(x, y)
                if correlation is not None:
                    numeric_pairs[self._pair_key(left, right)] = {
                        "fields": [left, right],
                        "correlation": correlation,
                        "support": len(values),
                    }

            missing_left = [1.0 if self._is_missing(record.get(left)) else 0.0 for record in prepared]
            missing_right = [1.0 if self._is_missing(record.get(right)) else 0.0 for record in prepared]
            missing_correlation = self._pearson(missing_left, missing_right)
            if missing_correlation is not None:
                missingness_pairs[self._pair_key(left, right)] = {
                    "fields": [left, right],
                    "correlation": missing_correlation,
                    "support": len(prepared),
                }

        return {
            "record_count": len(prepared),
            "numeric_fields": fields,
            "numeric_correlations": numeric_pairs,
            "missingness_correlations": missingness_pairs,
            "generated_at": time.time(),
        }

    def assess(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        source_id: str,
        batch_id: str,
        baseline_profile: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        source_key = nonempty_text(source_id, "source_id")
        batch_key = nonempty_text(batch_id, "batch_id")
        profile = self.build_profile(records)

        if not self.enabled:
            return RelationshipAssessment(
                source_id=source_key,
                batch_id=batch_key,
                verdict="pass",
                batch_score=1.0,
                confidence=1.0,
                profile=profile,
                baseline_available=False,
                evaluated_pairs=0,
                skipped_pairs=0,
                findings=(),
                flags=("relationship_quality_disabled",),
                remediation_actions=(),
            ).to_dict()

        baseline = (
            normalized_mapping(baseline_profile)
            if baseline_profile is not None
            else self._resolve_memory_baseline(source_key)
        )
        if not baseline:
            return RelationshipAssessment(
                source_id=source_key,
                batch_id=batch_key,
                verdict="warn",
                batch_score=self.warn_threshold,
                confidence=0.35,
                profile=profile,
                baseline_available=False,
                evaluated_pairs=0,
                skipped_pairs=0,
                findings=(),
                flags=("relationship_baseline_unavailable",),
                remediation_actions=("establish_trusted_relationship_baseline",),
            ).to_dict()

        findings: List[RelationshipFinding] = []
        skipped = 0
        specifications = (
            (
                "numeric_correlation",
                "numeric_correlations",
                self.warn_correlation_delta,
                self.block_correlation_delta,
            ),
            (
                "missingness_correlation",
                "missingness_correlations",
                self.warn_missingness_delta,
                self.block_missingness_delta,
            ),
        )

        for relationship_type, key, warn_delta, block_delta in specifications:
            current_pairs = profile.get(key, {})
            baseline_pairs = baseline.get(key, {})
            if not isinstance(current_pairs, Mapping) or not isinstance(baseline_pairs, Mapping):
                continue
            for pair_key in sorted(set(current_pairs) & set(baseline_pairs)):
                current = current_pairs[pair_key]
                reference = baseline_pairs[pair_key]
                if not isinstance(current, Mapping) or not isinstance(reference, Mapping):
                    skipped += 1
                    continue
                pair = tuple(str(item) for item in current.get("fields", []))
                if len(pair) != 2:
                    skipped += 1
                    continue
                current_value = float(current.get("correlation", 0.0))
                baseline_value = float(reference.get("correlation", 0.0))
                current_n = int(current.get("support", 0))
                baseline_n = int(reference.get("support", 0))
                if min(current_n, baseline_n) < self.min_pair_observations:
                    skipped += 1
                    continue
                if (
                    relationship_type == "numeric_correlation"
                    and abs(baseline_value) < self.min_absolute_baseline_correlation
                    and abs(current_value) < self.min_absolute_baseline_correlation
                ):
                    skipped += 1
                    continue

                delta = abs(current_value - baseline_value)
                z_value, p_value = self._fisher_difference(
                    current_value,
                    baseline_value,
                    current_n,
                    baseline_n,
                )
                statistically_supported = p_value is not None and p_value <= self.significance_alpha

                verdict = "pass"
                severity = "low"
                flags: List[str] = []
                if delta >= block_delta:
                    if self.require_statistical_support_for_block and not statistically_supported:
                        verdict = "warn"
                        severity = "medium"
                        flags.append("large_relationship_delta_low_support")
                    else:
                        verdict = "block"
                        severity = "high"
                        flags.append("relationship_shift_block")
                elif delta >= warn_delta:
                    verdict = "warn"
                    severity = "medium"
                    flags.append("relationship_shift_warn")

                score = self._score_from_delta(delta, warn_delta, block_delta)
                confidence = self._support_confidence(
                    current_n,
                    baseline_n,
                    statistically_supported=statistically_supported,
                )
                actions = self.block_actions if verdict == "block" else self.warn_actions if verdict == "warn" else ()
                findings.append(
                    RelationshipFinding(
                        pair=(pair[0], pair[1]),
                        relationship_type=relationship_type,
                        verdict=verdict,
                        severity=severity,
                        score=score,
                        confidence=confidence,
                        current_value=current_value,
                        baseline_value=baseline_value,
                        absolute_delta=delta,
                        current_support=current_n,
                        baseline_support=baseline_n,
                        fisher_z=z_value,
                        p_value=p_value,
                        statistically_supported=statistically_supported,
                        flags=tuple(flags),
                        remediation_actions=tuple(actions),
                    )
                )

        if not findings:
            batch_score = 1.0
            verdict = "pass"
            confidence = 0.55
        else:
            batch_score = min(item.score for item in findings)
            if any(item.verdict == "block" for item in findings):
                verdict = "block"
            elif any(item.verdict == "warn" for item in findings):
                verdict = "warn"
            else:
                verdict = verdict_from_score(
                    batch_score,
                    pass_threshold=self.pass_threshold,
                    warn_threshold=self.warn_threshold,
                )
            confidence = sum(item.confidence for item in findings) / len(findings)

        flags = merge_unique_strings(*(item.flags for item in findings))
        actions = merge_unique_strings(*(item.remediation_actions for item in findings))
        return RelationshipAssessment(
            source_id=source_key,
            batch_id=batch_key,
            verdict=verdict,
            batch_score=bounded_score(batch_score, field_name="relationship.batch_score"),
            confidence=bounded_score(confidence, field_name="relationship.confidence"),
            profile=profile,
            baseline_available=True,
            evaluated_pairs=len(findings),
            skipped_pairs=skipped,
            findings=tuple(findings),
            flags=tuple(flags),
            remediation_actions=tuple(actions),
        ).to_dict()

    def persist_profile_baseline(
        self,
        *,
        source_id: str,
        profile: Mapping[str, Any],
        governance_status: str,
        window: Optional[str] = None,
        schema_version: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Persist only a baseline already promoted by BaselineGovernor."""
        status = str(governance_status).strip().lower()
        if status not in {"trusted", "active"}:
            raise self._config_error(
                "Relationship baseline persistence requires governance_status 'trusted' or 'active'",
                context={"governance_status": governance_status},
            )
        if self.memory is None:
            return None
        method = getattr(self.memory, "record_drift_baseline", None)
        if not callable(method):
            return None
        value = method(
            source_id=nonempty_text(source_id, "source_id"),
            metric=self.profile_metric_name,
            baseline=normalized_mapping(profile, allow_none=False),
            window=window,
            schema_version=schema_version,
            context={
                **normalized_mapping(context),
                "baseline_kind": "relationship_profile",
                "baseline_governance_status": status,
            },
        )
        return dict(value) if isinstance(value, Mapping) else None

    def _resolve_memory_baseline(self, source_id: str) -> Dict[str, Any]:
        if self.memory is None:
            return {}
        direct = getattr(self.memory, "latest_drift_baseline", None)
        if callable(direct):
            value = direct(source_id, self.profile_metric_name)
            if isinstance(value, Mapping) and self._trusted_baseline_record(value):
                baseline = value.get("baseline", value)
                if isinstance(baseline, Mapping):
                    return dict(baseline)

        exporter = getattr(self.memory, "export_state", None)
        if callable(exporter):
            state = exporter()
            if isinstance(state, Mapping):
                baselines = state.get("drift_baselines", {})
                if isinstance(baselines, Mapping):
                    source_map = baselines.get(source_id, {})
                    if isinstance(source_map, Mapping):
                        history = source_map.get(self.profile_metric_name, [])
                        if isinstance(history, Sequence):
                            for latest in reversed(history):
                                if isinstance(latest, Mapping) and self._trusted_baseline_record(latest):
                                    baseline = latest.get("baseline", {})
                                    if isinstance(baseline, Mapping):
                                        return dict(baseline)
        return {}

    @staticmethod
    def _trusted_baseline_record(record: Mapping[str, Any]) -> bool:
        context = record.get("context", {})
        if not isinstance(context, Mapping):
            return False
        return str(context.get("baseline_governance_status", "")).strip().lower() in {"trusted", "active"}

    def _select_numeric_fields(self, records: Sequence[Mapping[str, Any]]) -> List[str]:
        candidates: List[Tuple[str, int, float]] = []
        field_names = sorted(
            {
                str(key)
                for record in records
                for key in record
                if str(key) not in self.exclude_fields
            }
        )
        for field in field_names:
            values = [float(record[field]) for record in records if self._is_numeric(record.get(field))]
            if len(values) < self.min_pair_observations:
                continue
            variance = self._variance(values)
            if variance <= 0.0:
                continue
            candidates.append((field, len(values), variance))
        candidates.sort(key=lambda item: (item[1], item[2], item[0]), reverse=True)
        return [item[0] for item in candidates[: self.max_numeric_fields]]

    @staticmethod
    def _pair_key(left: str, right: str) -> str:
        a, b = sorted((str(left), str(right)))
        return f"{a}||{b}"

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, (list, tuple, set, dict)):
            return len(value) == 0
        return False

    @staticmethod
    def _is_numeric(value: Any) -> bool:
        if value is None or isinstance(value, bool):
            return False
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return False
        return math.isfinite(numeric)

    @staticmethod
    def _variance(values: Sequence[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean_value = sum(values) / len(values)
        return sum((item - mean_value) ** 2 for item in values) / len(values)

    @classmethod
    def _pearson(cls, x: Sequence[float], y: Sequence[float]) -> Optional[float]:
        if len(x) != len(y) or len(x) < 2:
            return None
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)
        dx = [item - mean_x for item in x]
        dy = [item - mean_y for item in y]
        denominator = math.sqrt(sum(item * item for item in dx)) * math.sqrt(sum(item * item for item in dy))
        if denominator <= 0.0:
            return None
        value = sum(a * b for a, b in zip(dx, dy)) / denominator
        return max(-1.0, min(1.0, value))

    @staticmethod
    def _fisher_difference(
        current_r: float,
        baseline_r: float,
        current_n: int,
        baseline_n: int,
    ) -> Tuple[Optional[float], Optional[float]]:
        if current_n <= 3 or baseline_n <= 3:
            return None, None
        eps = 1e-12
        current = max(-1.0 + eps, min(1.0 - eps, current_r))
        baseline = max(-1.0 + eps, min(1.0 - eps, baseline_r))
        z_current = 0.5 * math.log((1.0 + current) / (1.0 - current))
        z_baseline = 0.5 * math.log((1.0 + baseline) / (1.0 - baseline))
        standard_error = math.sqrt((1.0 / (current_n - 3)) + (1.0 / (baseline_n - 3)))
        if standard_error <= 0.0:
            return None, None
        z_value = (z_current - z_baseline) / standard_error
        p_value = math.erfc(abs(z_value) / math.sqrt(2.0))
        return z_value, max(0.0, min(1.0, p_value))

    @staticmethod
    def _score_from_delta(delta: float, warn_threshold: float, block_threshold: float) -> float:
        if delta <= warn_threshold:
            if warn_threshold <= 0.0:
                return 1.0
            return max(0.90, 1.0 - 0.10 * (delta / warn_threshold))
        if block_threshold <= warn_threshold or delta >= block_threshold:
            return 0.0
        fraction = (delta - warn_threshold) / (block_threshold - warn_threshold)
        return max(0.0, 0.90 - 0.90 * fraction)

    def _support_confidence(
        self,
        current_n: int,
        baseline_n: int,
        *,
        statistically_supported: bool,
    ) -> float:
        minimum = min(current_n, baseline_n)
        sample_component = 1.0 - math.exp(-minimum / max(float(self.min_pair_observations), 1.0))
        support_bonus = 0.10 if statistically_supported else 0.0
        return max(0.0, min(1.0, 0.45 + 0.45 * sample_component + support_bonus))

    @staticmethod
    def _positive_int(value: Any, field_name: str) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise RelationshipQuality._config_error(
                f"{field_name} must be a positive integer",
                context={"value": value},
                cause=exc,
            ) from exc
        if parsed <= 0:
            raise RelationshipQuality._config_error(
                f"{field_name} must be > 0",
                context={"value": value},
            )
        return parsed

    @staticmethod
    def _bounded_nonnegative(value: Any, *, maximum: float, field_name: str) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise RelationshipQuality._config_error(
                f"{field_name} must be numeric",
                context={"value": value},
                cause=exc,
            ) from exc
        if not math.isfinite(parsed) or parsed < 0.0 or parsed > maximum:
            raise RelationshipQuality._config_error(
                f"{field_name} must be within [0, {maximum}]",
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
            domain=QualityDomain.STATISTICAL,
            disposition=QualityDisposition.ESCALATE,
            context=dict(context or {}),
            remediation="Correct relationship_quality in quality_config.yaml.",
            cause=cause,
        )


__all__ = ["RelationshipFinding", "RelationshipAssessment", "RelationshipQuality"]