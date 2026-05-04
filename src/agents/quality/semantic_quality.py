"""
- Label leakage checks.
- Cross-field consistency rules.
- Source trust weighting and provenance confidence.
"""

from __future__ import annotations

import json
import math
import os
import re
import time

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .utils.config_loader import load_global_config, get_config_section
from .utils.quality_error import ( ProvenanceTrustError, DataQualityError, QualityDomain,
                                  QualityDisposition, QualityErrorType, QualitySeverity,
                                  QualityStage, quality_error_boundary)
from .quality_memory import QualityMemory
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Semantic Quality")
printer = PrettyPrinter


@dataclass(slots=True)
class SemanticFinding:
    finding_id: str
    check_name: str
    verdict: str
    severity: str
    confidence: float
    score: float
    message: str
    flags: List[str] = field(default_factory=list)
    affected_records: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    remediation_actions: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_conflict_input(self) -> Dict[str, Any]:
        return {
            "checker": self.check_name,
            "domain": "semantic",
            "verdict": self.verdict,
            "severity": self.severity,
            "confidence": self.confidence,
            "flags": list(self.flags),
        }


@dataclass(slots=True)
class SemanticAssessment:
    assessment_id: str
    source_id: str
    batch_id: str
    label_field: str
    verdict: str
    batch_score: float
    record_count: int
    reviewed_record_count: int
    quarantine_count: int
    flags: List[str]
    remediation_actions: List[str]
    shift_metrics: Dict[str, float]
    source_reliability: float
    provenance_confidence: float
    conflict_resolution: Dict[str, Any]
    findings: List[Dict[str, Any]]
    context: Dict[str, Any]
    created_at: float
    schema_version: Optional[str] = None
    window: Optional[str] = None
    memory_snapshot: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["created_at_iso"] = datetime.fromtimestamp(self.created_at, tz=timezone.utc).isoformat()
        return payload


class SemanticQuality:
    """Production-grade semantic gate for label integrity, field coherence, and provenance trust.

    The semantic layer sits after raw structural validity but before downstream usage. Its job is
    to answer a different question than schema validation: even if the record *fits* the schema,
    is it semantically safe and trustworthy enough to influence ingestion, learning, inference,
    or memory updates?
    """

    def __init__(self) -> None:
        self.config = load_global_config()
        self.quality_config = get_config_section("semantic_quality")
        self.memory = QualityMemory()

        self.enabled = bool(self.quality_config.get("enabled", True))
        self.auto_record_to_memory = bool(self.quality_config.get("auto_record_to_memory", True))
        self.default_label_field = str(self.quality_config.get("default_label_field", "label")).strip() or "label"
        self.default_window = str(self.quality_config.get("default_window", "latest")).strip() or "latest"
        self.record_id_candidates = self._string_list(
            self.quality_config.get("ids", {}).get(
                "record_id_candidates",
                ["record_id", "id", "row_id", "sample_id", "uuid"],
            )
        )

        scoring = self.quality_config.get("scoring", {})
        self.pass_threshold = float(scoring.get("pass_threshold", 0.90))
        self.warn_threshold = float(scoring.get("warn_threshold", 0.75))
        self.component_weights = {
            "leakage": float(scoring.get("weights", {}).get("leakage", 0.45)),
            "consistency": float(scoring.get("weights", {}).get("consistency", 0.30)),
            "provenance": float(scoring.get("weights", {}).get("provenance", 0.25)),
        }

        self.leakage_config = self.quality_config.get("leakage_detection", {})
        self.consistency_config = self.quality_config.get("cross_field_consistency", {})
        self.provenance_config = self.quality_config.get("provenance", {})
        self.remediation_config = self.quality_config.get("remediation", {})
        self.memory_options = self.quality_config.get("memory", {})

        self._validate_runtime_configuration()

        logger.info(
            "Semantic Quality initialized | enabled=%s | auto_record_to_memory=%s | default_label_field=%s",
            self.enabled,
            self.auto_record_to_memory,
            self.default_label_field,
        )

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------
    def evaluate_batch(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        source_id: str,
        batch_id: str,
        label_field: Optional[str] = None,
        feature_fields: Optional[Sequence[str]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
        source_metadata: Optional[Mapping[str, Any]] = None,
        schema_version: Optional[str] = None,
        window: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._boundary(
            operation="evaluate_batch",
            stage=QualityStage.VALIDATION,
            context={"source_id": source_id, "batch_id": batch_id},
        ):
            source_key = self._nonempty(source_id, "source_id")
            batch_key = self._nonempty(batch_id, "batch_id")
            label_key = self._nonempty(label_field or self.default_label_field, "label_field")
            normalized_records = self._normalize_records(records)

            if not normalized_records:
                raise DataQualityError(
                    message="Semantic quality evaluation requires at least one record",
                    error_type=QualityErrorType.SCHEMA_VALIDATION_FAILED,
                    severity=QualitySeverity.HIGH,
                    retryable=False,
                    stage=QualityStage.VALIDATION,
                    domain=QualityDomain.SEMANTIC,
                    disposition=QualityDisposition.BLOCK,
                    source_id=source_key,
                    batch_id=batch_key,
                    context={"label_field": label_key},
                    remediation="Provide at least one normalized record before running semantic validation.",
                )

            runtime_context = self._normalized_mapping(context)
            provenance_payload = self._merge_provenance_inputs(
                source_id=source_key,
                provenance=provenance,
                source_metadata=source_metadata,
                schema_version=schema_version,
                batch_id=batch_key,
            )

            leakage = self._detect_label_leakage(
                normalized_records,
                label_field=label_key,
                feature_fields=feature_fields,
                source_id=source_key,
                batch_id=batch_key,
            )
            consistency = self._validate_cross_field_consistency(
                normalized_records,
                label_field=label_key,
                source_id=source_key,
                batch_id=batch_key,
            )
            provenance_finding = self._assess_provenance(
                normalized_records,
                source_id=source_key,
                batch_id=batch_key,
                provenance=provenance_payload,
            )

            findings = [leakage, consistency, provenance_finding]
            conflict_resolution = self.memory.reconcile_conflicts(
                findings=[finding.to_conflict_input() for finding in findings],
                source_id=source_key,
                batch_id=batch_key,
                scope_key=f"semantic:{source_key}:{batch_key}",
                context={"label_field": label_key},
            )

            batch_score = self._aggregate_batch_score(findings)
            final_verdict = self._resolve_final_verdict(batch_score, findings, conflict_resolution)
            flags = self._merge_string_sets(*(finding.flags for finding in findings), [f"semantic_verdict:{final_verdict}"])
            remediation_actions = self._merge_string_sets(*(finding.remediation_actions for finding in findings))
            quarantine_record_ids = self._merge_string_sets(*(finding.affected_records for finding in findings if finding.verdict != "pass"))
            shift_metrics = {
                "semantic.leakage_rate": float(leakage.metrics.get("contaminated_record_rate", 0.0)),
                "semantic.max_field_leakage_rate": float(leakage.metrics.get("max_field_leakage_rate", 0.0)),
                "semantic.inconsistency_rate": float(consistency.metrics.get("record_inconsistency_rate", 0.0)),
                "semantic.entity_label_conflict_rate": float(consistency.metrics.get("entity_label_conflict_rate", 0.0)),
                "semantic.source_trust_score": float(provenance_finding.metrics.get("source_trust_score", 0.0)),
                "semantic.provenance_confidence": float(provenance_finding.metrics.get("provenance_confidence", 0.0)),
            }
            source_reliability = float(provenance_finding.metrics.get("source_trust_score", 0.0))

            if self.memory_options.get("record_source_reliability", True):
                self.memory.record_source_reliability(
                    source_id=source_key,
                    reliability=source_reliability,
                    reason=(
                        f"semantic evaluation for batch '{batch_key}' produced verdict '{final_verdict}' "
                        f"with provenance confidence {provenance_finding.metrics.get('provenance_confidence', 0.0):.3f}"
                    ),
                    context={
                        "batch_id": batch_key,
                        "component": "semantic_quality",
                        "label_field": label_key,
                    },
                )

            memory_snapshot: Optional[Dict[str, Any]] = None
            if self.auto_record_to_memory and self.memory_options.get("record_quality_snapshot", True):
                memory_snapshot = self.memory.record_quality_snapshot(
                    source_id=source_key,
                    batch_id=batch_key,
                    batch_score=batch_score,
                    verdict=final_verdict,
                    flags=flags,
                    quarantine_count=len(quarantine_record_ids),
                    shift_metrics=shift_metrics,
                    remediation_actions=remediation_actions,
                    source_reliability=source_reliability,
                    schema_version=schema_version or provenance_payload.get("schema_version"),
                    window=window or self.default_window,
                    checker_findings=[finding.to_conflict_input() for finding in findings],
                    context={
                        **runtime_context,
                        "component": "semantic_quality",
                        "label_field": label_key,
                        "record_count": len(normalized_records),
                    },
                )

            assessment = SemanticAssessment(
                assessment_id=self._new_id("semantic_assessment"),
                source_id=source_key,
                batch_id=batch_key,
                label_field=label_key,
                verdict=final_verdict,
                batch_score=batch_score,
                record_count=len(normalized_records),
                reviewed_record_count=len(normalized_records),
                quarantine_count=len(quarantine_record_ids),
                flags=flags,
                remediation_actions=remediation_actions,
                shift_metrics=shift_metrics,
                source_reliability=source_reliability,
                provenance_confidence=float(provenance_finding.metrics.get("provenance_confidence", 0.0)),
                conflict_resolution=conflict_resolution,
                findings=[finding.to_dict() for finding in findings],
                context=runtime_context,
                created_at=time.time(),
                schema_version=schema_version or provenance_payload.get("schema_version"),
                window=window or self.default_window,
                memory_snapshot=memory_snapshot,
            )
            return assessment.to_dict()

    def assess_batch(self, records: Sequence[Mapping[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        return self.evaluate_batch(records, **kwargs)

    def detect_label_leakage(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        label_field: Optional[str] = None,
        feature_fields: Optional[Sequence[str]] = None,
        source_id: str = "unknown_source",
        batch_id: str = "unknown_batch",
    ) -> Dict[str, Any]:
        with self._boundary(
            operation="detect_label_leakage",
            stage=QualityStage.VALIDATION,
            context={"source_id": source_id, "batch_id": batch_id},
            error_type=QualityErrorType.LEAKAGE_DETECTED,
            severity=QualitySeverity.HIGH,
            retryable=False,
            remediation="Remove leaking features, quarantine the contaminated records, and rerun semantic validation.",
            disposition=QualityDisposition.BLOCK,
        ):
            return self._detect_label_leakage(
                self._normalize_records(records),
                label_field=self._nonempty(label_field or self.default_label_field, "label_field"),
                feature_fields=feature_fields,
                source_id=source_id,
                batch_id=batch_id,
            ).to_dict()

    def validate_cross_field_consistency(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        label_field: Optional[str] = None,
        source_id: str = "unknown_source",
        batch_id: str = "unknown_batch",
    ) -> Dict[str, Any]:
        with self._boundary(
            operation="validate_cross_field_consistency",
            stage=QualityStage.VALIDATION,
            context={"source_id": source_id, "batch_id": batch_id},
            error_type=QualityErrorType.CROSS_FIELD_CONFLICT,
            severity=QualitySeverity.HIGH,
            retryable=False,
            remediation="Normalize conflicting fields, quarantine inconsistent records, and rerun semantic validation.",
            disposition=QualityDisposition.QUARANTINE,
        ):
            return self._validate_cross_field_consistency(
                self._normalize_records(records),
                label_field=self._nonempty(label_field or self.default_label_field, "label_field"),
                source_id=source_id,
                batch_id=batch_id,
            ).to_dict()

    def assess_provenance(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        source_id: str,
        batch_id: str,
        provenance: Optional[Mapping[str, Any]] = None,
        source_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._boundary(
            operation="assess_provenance",
            stage=QualityStage.VALIDATION,
            context={"source_id": source_id, "batch_id": batch_id},
            error_type=QualityErrorType.PROVENANCE_UNTRUSTED,
            severity=QualitySeverity.HIGH,
            retryable=False,
            remediation="Supply complete provenance metadata, verify lineage, and re-run semantic validation.",
            disposition=QualityDisposition.BLOCK,
        ):
            merged = self._merge_provenance_inputs(
                source_id=source_id,
                provenance=provenance,
                source_metadata=source_metadata,
                batch_id=batch_id,
                schema_version=None,
            )
            return self._assess_provenance(
                self._normalize_records(records),
                source_id=source_id,
                batch_id=batch_id,
                provenance=merged,
            ).to_dict()

    # ------------------------------------------------------------------
    # Internal semantic checks
    # ------------------------------------------------------------------
    def _detect_label_leakage(
        self,
        records: Sequence[Dict[str, Any]],
        *,
        label_field: str,
        feature_fields: Optional[Sequence[str]],
        source_id: str,
        batch_id: str,
    ) -> SemanticFinding:
        blocked_name_terms = [item.lower() for item in self._string_list(self.leakage_config.get("suspicious_name_block_terms", []))]
        warned_name_terms = [item.lower() for item in self._string_list(self.leakage_config.get("suspicious_name_warn_terms", []))]
        future_terms = [item.lower() for item in self._string_list(self.leakage_config.get("future_terms", []))]
        text_fields = set(self._string_list(self.leakage_config.get("suspicious_text_fields", [])))
        excluded_fields = set(self._string_list(self.leakage_config.get("fingerprint_fields_exclude", [])))
        exact_block_threshold = self._bounded_unit_interval(
            self.leakage_config.get("exact_match_block_threshold", 0.20),
            "semantic_quality.leakage_detection.exact_match_block_threshold",
        )
        exact_warn_threshold = self._bounded_unit_interval(
            self.leakage_config.get("exact_match_warn_threshold", 0.05),
            "semantic_quality.leakage_detection.exact_match_warn_threshold",
        )
        similarity_threshold = self._bounded_unit_interval(
            self.leakage_config.get("normalized_similarity_threshold", 0.97),
            "semantic_quality.leakage_detection.normalized_similarity_threshold",
        )
        high_risk_confidence = self._bounded_unit_interval(
            self.leakage_config.get("high_risk_confidence", 0.90),
            "semantic_quality.leakage_detection.high_risk_confidence",
        )

        candidate_fields = list(feature_fields) if feature_fields else self._discover_feature_fields(records, label_field, excluded_fields)
        if not candidate_fields:
            return SemanticFinding(
                finding_id=self._new_id("semantic_finding"),
                check_name="leakage",
                verdict="pass",
                severity="low",
                confidence=1.0,
                score=1.0,
                message="No candidate feature fields were available for leakage analysis.",
                flags=["semantic:no_candidate_features"],
                metrics={
                    "record_count": len(records),
                    "candidate_field_count": 0,
                    "contaminated_record_rate": 0.0,
                    "max_field_leakage_rate": 0.0,
                },
                remediation_actions=[],
                details={"candidate_fields": []},
            )

        field_stats: Dict[str, Dict[str, Any]] = {
            field: {
                "comparable": 0,
                "exact_matches": 0,
                "near_matches": 0,
                "suspicious_name": self._field_risk_level(field, blocked_name_terms, warned_name_terms, future_terms),
            }
            for field in candidate_fields
        }
        contaminated_record_ids: List[str] = []
        suspicious_fields: List[str] = []
        block_named_fields: List[str] = []
        warning_named_fields: List[str] = []

        for record in records:
            label_value = self._normalize_value(record.get(label_field))
            if not label_value:
                continue
            record_id = self._record_id(record)
            record_contaminated = False

            for candidate_field in candidate_fields:
                feature_value = self._normalize_value(record.get(candidate_field))
                if not feature_value:
                    continue

                stats = field_stats[candidate_field]
                stats["comparable"] += 1
                if feature_value == label_value:
                    stats["exact_matches"] += 1
                    record_contaminated = True
                elif candidate_field in text_fields and self._similarity(label_value, feature_value) >= similarity_threshold:
                    stats["near_matches"] += 1
                    record_contaminated = True

            if record_contaminated:
                contaminated_record_ids.append(record_id)

        field_summaries: List[Dict[str, Any]] = []
        max_field_leakage_rate = 0.0
        for field_name, stats in field_stats.items():
            comparable = int(stats["comparable"])
            exact_matches = int(stats["exact_matches"])
            near_matches = int(stats["near_matches"])
            leakage_rate = exact_matches / comparable if comparable else 0.0
            near_match_rate = near_matches / comparable if comparable else 0.0
            max_field_leakage_rate = max(max_field_leakage_rate, leakage_rate)
            risk_level = str(stats["suspicious_name"])

            if risk_level == "block":
                block_named_fields.append(field_name)
            elif risk_level == "warn":
                warning_named_fields.append(field_name)

            if leakage_rate >= exact_warn_threshold or near_match_rate > 0.0 or risk_level != "none":
                suspicious_fields.append(field_name)

            field_summaries.append(
                {
                    "field": field_name,
                    "risk_level": risk_level,
                    "comparable": comparable,
                    "exact_matches": exact_matches,
                    "near_matches": near_matches,
                    "leakage_rate": leakage_rate,
                    "near_match_rate": near_match_rate,
                }
            )

        contaminated_record_rate = len(set(contaminated_record_ids)) / len(records) if records else 0.0
        name_risk_present = bool(block_named_fields or warning_named_fields)

        verdict = "pass"
        severity = "low"
        flags = ["semantic:leakage_checked"]
        remediation_actions: List[str] = []
        error_payload: Optional[Dict[str, Any]] = None
        message = "No semantic leakage indicators crossed configured thresholds."

        if block_named_fields or contaminated_record_rate >= exact_block_threshold or max_field_leakage_rate >= exact_block_threshold:
            verdict = "block"
            severity = "critical" if block_named_fields else "high"
            flags.extend(["semantic:leakage_detected", "semantic:hard_block"])
            remediation_actions.extend(self._string_list(self.remediation_config.get("leakage_block_actions", [])))
            message = (
                "High-risk semantic leakage detected via exact label mirrors, near mirrors, or explicit target-like feature fields."
            )
            leakage_error = DataQualityError(
                message=message,
                error_type=QualityErrorType.LEAKAGE_DETECTED,
                severity=QualitySeverity.CRITICAL if severity == "critical" else QualitySeverity.HIGH,
                retryable=False,
                stage=QualityStage.VALIDATION,
                domain=QualityDomain.SEMANTIC,
                disposition=QualityDisposition.BLOCK,
                source_id=source_id,
                batch_id=batch_id,
                context={
                    "label_field": label_field,
                    "block_named_fields": block_named_fields,
                    "contaminated_record_rate": contaminated_record_rate,
                    "max_field_leakage_rate": max_field_leakage_rate,
                },
                remediation="Quarantine the contaminated records, remove leaking features, and re-run downstream selection.",
            )
            leakage_error.report()
            error_payload = leakage_error.to_dict()
        elif name_risk_present or contaminated_record_rate >= exact_warn_threshold or max_field_leakage_rate >= exact_warn_threshold:
            verdict = "warn"
            severity = "medium" if not name_risk_present else "high"
            flags.extend(["semantic:leakage_risk", "semantic:review_required"])
            remediation_actions.extend(self._string_list(self.remediation_config.get("leakage_warn_actions", [])))
            message = "Leakage risk indicators are elevated but not severe enough to force an immediate block."
            leakage_error = DataQualityError(
                message=message,
                error_type=QualityErrorType.LEAKAGE_DETECTED,
                severity=QualitySeverity.HIGH if severity == "high" else QualitySeverity.MEDIUM,
                retryable=False,
                stage=QualityStage.VALIDATION,
                domain=QualityDomain.SEMANTIC,
                disposition=QualityDisposition.QUARANTINE,
                source_id=source_id,
                batch_id=batch_id,
                context={
                    "label_field": label_field,
                    "warning_named_fields": warning_named_fields,
                    "contaminated_record_rate": contaminated_record_rate,
                    "max_field_leakage_rate": max_field_leakage_rate,
                },
                remediation="Review feature selection, mask leaking hints, and quarantine suspect rows before use.",
            )
            error_payload = leakage_error.to_dict()

        confidence = max(
            high_risk_confidence if block_named_fields else 0.0,
            min(1.0, max_field_leakage_rate + 0.25 if suspicious_fields else 0.0),
            min(1.0, contaminated_record_rate + 0.20),
        )
        if verdict == "pass":
            confidence = max(0.70, 1.0 - contaminated_record_rate)

        score = max(0.0, 1.0 - (0.80 * contaminated_record_rate) - (0.60 * max_field_leakage_rate) - (0.15 if name_risk_present else 0.0))
        if verdict == "block":
            score = min(score, 0.24)
        elif verdict == "warn":
            score = min(score, 0.74)

        return SemanticFinding(
            finding_id=self._new_id("semantic_finding"),
            check_name="leakage",
            verdict=verdict,
            severity=severity,
            confidence=round(min(max(confidence, 0.0), 1.0), 6),
            score=round(min(max(score, 0.0), 1.0), 6),
            message=message,
            flags=self._merge_string_sets(flags, [f"semantic:leakage_verdict:{verdict}"], [f"semantic:suspicious_field:{field}" for field in suspicious_fields]),
            affected_records=sorted(set(contaminated_record_ids)),
            metrics={
                "record_count": len(records),
                "candidate_field_count": len(candidate_fields),
                "suspicious_field_count": len(suspicious_fields),
                "contaminated_record_count": len(set(contaminated_record_ids)),
                "contaminated_record_rate": round(contaminated_record_rate, 6),
                "max_field_leakage_rate": round(max_field_leakage_rate, 6),
                "block_named_field_count": len(block_named_fields),
                "warning_named_field_count": len(warning_named_fields),
            },
            remediation_actions=self._merge_string_sets(remediation_actions),
            details={
                "label_field": label_field,
                "candidate_fields": candidate_fields,
                "field_summaries": field_summaries,
                "block_named_fields": block_named_fields,
                "warning_named_fields": warning_named_fields,
            },
            error=error_payload,
        )

    def _validate_cross_field_consistency(
        self,
        records: Sequence[Dict[str, Any]],
        *,
        label_field: str,
        source_id: str,
        batch_id: str,
    ) -> SemanticFinding:
        rules = self.consistency_config.get("rules", [])
        entity_fields = self._string_list(self.consistency_config.get("duplicate_entity_fields", []))
        warn_rate = self._bounded_unit_interval(
            self.consistency_config.get("max_record_inconsistency_rate_warn", 0.05),
            "semantic_quality.cross_field_consistency.max_record_inconsistency_rate_warn",
        )
        block_rate = self._bounded_unit_interval(
            self.consistency_config.get("max_record_inconsistency_rate_block", 0.15),
            "semantic_quality.cross_field_consistency.max_record_inconsistency_rate_block",
        )
        entity_conflict_block_threshold = self._bounded_unit_interval(
            self.consistency_config.get("label_conflict_block_threshold", 0.10),
            "semantic_quality.cross_field_consistency.label_conflict_block_threshold",
        )

        rule_violations: List[Dict[str, Any]] = []
        inconsistent_record_ids: List[str] = []
        rule_failure_counts: Dict[str, int] = defaultdict(int)

        input_refs = {"input.source_id": source_id, "input.batch_id": batch_id, "input.label_field": label_field}

        for record in records:
            record_id = self._record_id(record)
            record_failed = False
            for rule in rules:
                violation = self._evaluate_consistency_rule(record, rule, input_refs)
                if violation is None:
                    continue
                rule_violations.append({"record_id": record_id, **violation})
                rule_failure_counts[str(violation["rule_id"])] += 1
                record_failed = True
            if record_failed:
                inconsistent_record_ids.append(record_id)

        entity_label_conflicts: List[Dict[str, Any]] = []
        conflicting_entity_record_ids: List[str] = []
        for entity_field in entity_fields:
            grouped: Dict[str, Dict[str, Any]] = {}
            for record in records:
                entity_value = self._normalize_value(record.get(entity_field))
                label_value = self._normalize_value(record.get(label_field))
                if not entity_value or not label_value:
                    continue
                bucket = grouped.setdefault(entity_value, {"labels": set(), "record_ids": []})
                bucket["labels"].add(label_value)
                bucket["record_ids"].append(self._record_id(record))

            for entity_value, bucket in grouped.items():
                labels = sorted(bucket["labels"])
                if len(labels) > 1:
                    entity_label_conflicts.append(
                        {
                            "entity_field": entity_field,
                            "entity_value": entity_value,
                            "labels": labels,
                            "record_ids": sorted(set(bucket["record_ids"])),
                        }
                    )
                    conflicting_entity_record_ids.extend(bucket["record_ids"])

        affected_record_ids = sorted(set(inconsistent_record_ids) | set(conflicting_entity_record_ids))
        record_inconsistency_rate = len(affected_record_ids) / len(records) if records else 0.0
        entity_label_conflict_rate = len(entity_label_conflicts) / len(records) if records else 0.0

        verdict = "pass"
        severity = "low"
        flags = ["semantic:consistency_checked"]
        remediation_actions: List[str] = []
        message = "Cross-field consistency checks passed within configured tolerance."
        error_payload: Optional[Dict[str, Any]] = None

        if entity_label_conflict_rate >= entity_conflict_block_threshold or record_inconsistency_rate >= block_rate:
            verdict = "block"
            severity = "high"
            flags.extend(["semantic:cross_field_conflict", "semantic:hard_block"])
            remediation_actions.extend(self._string_list(self.remediation_config.get("consistency_actions", [])))
            message = "Cross-field conflicts or duplicate-entity label disagreements exceeded blocking thresholds."
            consistency_error = DataQualityError(
                message=message,
                error_type=(
                    QualityErrorType.INCONSISTENT_LABELS if entity_label_conflicts else QualityErrorType.CROSS_FIELD_CONFLICT
                ),
                severity=QualitySeverity.HIGH,
                retryable=False,
                stage=QualityStage.VALIDATION,
                domain=QualityDomain.SEMANTIC,
                disposition=QualityDisposition.QUARANTINE,
                source_id=source_id,
                batch_id=batch_id,
                context={
                    "record_inconsistency_rate": record_inconsistency_rate,
                    "entity_label_conflict_rate": entity_label_conflict_rate,
                    "entity_label_conflict_count": len(entity_label_conflicts),
                },
                remediation="Normalize conflicting fields, quarantine disagreeing records, and rerun semantic validation.",
            )
            consistency_error.report()
            error_payload = consistency_error.to_dict()
        elif entity_label_conflicts or record_inconsistency_rate >= warn_rate:
            verdict = "warn"
            severity = "medium"
            flags.extend(["semantic:cross_field_warning", "semantic:review_required"])
            remediation_actions.extend(self._string_list(self.remediation_config.get("consistency_actions", [])))
            message = "Cross-field inconsistencies were detected but remain below the hard-block threshold."
            consistency_error = DataQualityError(
                message=message,
                error_type=(
                    QualityErrorType.INCONSISTENT_LABELS if entity_label_conflicts else QualityErrorType.CROSS_FIELD_CONFLICT
                ),
                severity=QualitySeverity.MEDIUM,
                retryable=False,
                stage=QualityStage.VALIDATION,
                domain=QualityDomain.SEMANTIC,
                disposition=QualityDisposition.QUARANTINE,
                source_id=source_id,
                batch_id=batch_id,
                context={
                    "record_inconsistency_rate": record_inconsistency_rate,
                    "entity_label_conflict_rate": entity_label_conflict_rate,
                    "entity_label_conflict_count": len(entity_label_conflicts),
                },
                remediation="Review and normalize inconsistent field relationships before downstream use.",
            )
            error_payload = consistency_error.to_dict()

        confidence = min(1.0, 0.60 + record_inconsistency_rate + entity_label_conflict_rate)
        if verdict == "pass":
            confidence = max(0.72, 1.0 - (record_inconsistency_rate * 0.5))

        score = max(0.0, 1.0 - (0.75 * record_inconsistency_rate) - (0.55 * entity_label_conflict_rate))
        if verdict == "block":
            score = min(score, 0.30)
        elif verdict == "warn":
            score = min(score, 0.79)

        return SemanticFinding(
            finding_id=self._new_id("semantic_finding"),
            check_name="consistency",
            verdict=verdict,
            severity=severity,
            confidence=round(min(max(confidence, 0.0), 1.0), 6),
            score=round(min(max(score, 0.0), 1.0), 6),
            message=message,
            flags=self._merge_string_sets(flags, [f"semantic:consistency_verdict:{verdict}"], list(rule_failure_counts.keys())),
            affected_records=affected_record_ids,
            metrics={
                "record_count": len(records),
                "rule_violation_count": len(rule_violations),
                "record_inconsistency_count": len(affected_record_ids),
                "record_inconsistency_rate": round(record_inconsistency_rate, 6),
                "entity_label_conflict_count": len(entity_label_conflicts),
                "entity_label_conflict_rate": round(entity_label_conflict_rate, 6),
            },
            remediation_actions=self._merge_string_sets(remediation_actions),
            details={
                "rule_violations": rule_violations,
                "rule_failure_counts": dict(rule_failure_counts),
                "entity_label_conflicts": entity_label_conflicts,
                "configured_rules": self._normalized_mapping({"rules": rules, "duplicate_entity_fields": entity_fields}),
            },
            error=error_payload,
        )

    def _assess_provenance(
        self,
        records: Sequence[Dict[str, Any]],
        *,
        source_id: str,
        batch_id: str,
        provenance: Mapping[str, Any],
    ) -> SemanticFinding:
        required_fields = self._string_list(self.provenance_config.get("required_fields", []))
        optional_fields = self._string_list(self.provenance_config.get("optional_fields", []))
        min_trust_score = self._bounded_unit_interval(
            self.provenance_config.get("min_trust_score", 0.65),
            "semantic_quality.provenance.min_trust_score",
        )
        block_below_trust = self._bounded_unit_interval(
            self.provenance_config.get("block_below_trust", 0.45),
            "semantic_quality.provenance.block_below_trust",
        )
        block_on_missing_required = bool(self.provenance_config.get("block_on_missing_required", False))
        freshness_windows = self.provenance_config.get("freshness_windows_seconds", {})
        fresh_seconds = float(freshness_windows.get("fresh", 86400))
        acceptable_seconds = float(freshness_windows.get("acceptable", 604800))
        weights = self._normalized_weights(self.provenance_config.get("confidence_weights", {}), {
            "memory_reliability": 0.45,
            "metadata_completeness": 0.25,
            "recency": 0.15,
            "checksum": 0.10,
            "lineage": 0.05,
        })

        latest_reliability = self.memory.latest_source_reliability(source_id)
        memory_reliability = self._coerce_float(
            (latest_reliability or {}).get("reliability", self.memory.default_source_reliability),
            default=self.memory.default_source_reliability,
        )
        explicit_trust = self._coerce_float(
            provenance.get("trust_score", provenance.get("source_reliability", memory_reliability)),
            default=memory_reliability,
        )
        base_reliability = (memory_reliability + explicit_trust) / 2.0

        present_required = [field for field in required_fields if self._is_present(provenance.get(field))]
        missing_required = [field for field in required_fields if field not in present_required]
        present_optional = [field for field in optional_fields if self._is_present(provenance.get(field))]
        completeness_required = (len(present_required) / len(required_fields)) if required_fields else 1.0
        completeness_optional = (len(present_optional) / len(optional_fields)) if optional_fields else 1.0
        metadata_completeness = min(1.0, completeness_required + (0.20 * completeness_optional))

        timestamp_value = (
            provenance.get("collected_at")
            or provenance.get("fetched_at")
            or provenance.get("observed_at")
            or provenance.get("timestamp")
        )
        timestamp_epoch = self._parse_datetime(timestamp_value)
        recency_score = self._recency_score(timestamp_epoch, fresh_seconds=fresh_seconds, acceptable_seconds=acceptable_seconds)
        checksum_score = 1.0 if self._is_present(provenance.get("checksum")) else 0.0
        lineage_score = 1.0 if any(self._is_present(provenance.get(key)) for key in ["lineage_id", "uri", "source_uri", "upstream_source"]) else 0.0

        metadata_source_id = provenance.get("source_id")
        source_mismatch = self._is_present(metadata_source_id) and str(metadata_source_id).strip() != str(source_id).strip()
        if source_mismatch:
            base_reliability = min(base_reliability, 0.10)

        record_level_provenance_presence = 0
        for record in records:
            if self._is_present(record.get("provenance")) or self._is_present(record.get("source_id")):
                record_level_provenance_presence += 1
        record_provenance_coverage = record_level_provenance_presence / len(records) if records else 0.0

        source_trust_score = (
            weights["memory_reliability"] * base_reliability
            + weights["metadata_completeness"] * metadata_completeness
            + weights["recency"] * recency_score
            + weights["checksum"] * checksum_score
            + weights["lineage"] * lineage_score
        )
        provenance_confidence = min(1.0, max(0.0, (source_trust_score * 0.80) + (record_provenance_coverage * 0.20)))

        verdict = "pass"
        severity = "low"
        flags = ["semantic:provenance_checked"]
        remediation_actions: List[str] = []
        message = "Provenance metadata and source trust remain within configured tolerance."
        error_payload: Optional[Dict[str, Any]] = None

        if source_mismatch or source_trust_score < block_below_trust or (block_on_missing_required and missing_required):
            verdict = "block"
            severity = "critical" if source_mismatch or source_trust_score < block_below_trust else "high"
            flags.extend(["semantic:provenance_block", "semantic:review_required"])
            remediation_actions.extend(self._string_list(self.remediation_config.get("provenance_actions", [])))
            message = "Provenance trust failed blocking criteria due to low trust, missing required metadata, or source mismatch."
            trust_error = ProvenanceTrustError(
                source_id=source_id,
                trust_score=round(source_trust_score, 6),
                minimum_trust=round(block_below_trust, 6),
                context={
                    "batch_id": batch_id,
                    "missing_required": missing_required,
                    "source_mismatch": source_mismatch,
                    "provenance_confidence": provenance_confidence,
                },
            )
            trust_error.report()
            error_payload = trust_error.to_dict()
        elif missing_required or source_trust_score < min_trust_score:
            verdict = "warn"
            severity = "high" if missing_required else "medium"
            flags.extend(["semantic:provenance_warn", "semantic:review_required"])
            remediation_actions.extend(self._string_list(self.remediation_config.get("provenance_actions", [])))
            message = "Provenance metadata is incomplete or trust is lower than the preferred operating threshold."
            provenance_error = DataQualityError(
                message=message,
                error_type=(
                    QualityErrorType.PROVENANCE_MISSING if missing_required else QualityErrorType.PROVENANCE_UNTRUSTED
                ),
                severity=QualitySeverity.HIGH if missing_required else QualitySeverity.MEDIUM,
                retryable=False,
                stage=QualityStage.VALIDATION,
                domain=QualityDomain.SEMANTIC,
                disposition=QualityDisposition.BLOCK if missing_required else QualityDisposition.WARN,
                source_id=source_id,
                batch_id=batch_id,
                context={
                    "missing_required": missing_required,
                    "source_trust_score": source_trust_score,
                    "provenance_confidence": provenance_confidence,
                },
                remediation="Complete missing provenance metadata, verify source lineage, and re-run the semantic gate.",
            )
            error_payload = provenance_error.to_dict()

        score = max(
            0.0,
            min(
                1.0,
                (0.65 * source_trust_score)
                + (0.20 * metadata_completeness)
                + (0.10 * record_provenance_coverage)
                + (0.05 * (0.0 if source_mismatch else 1.0)),
            ),
        )
        if verdict == "block":
            score = min(score, 0.24)
        elif verdict == "warn":
            score = min(score, 0.74)

        return SemanticFinding(
            finding_id=self._new_id("semantic_finding"),
            check_name="provenance",
            verdict=verdict,
            severity=severity,
            confidence=round(min(max(provenance_confidence, 0.0), 1.0), 6),
            score=round(score, 6),
            message=message,
            flags=self._merge_string_sets(flags, [f"semantic:provenance_verdict:{verdict}"], [f"semantic:missing_required:{item}" for item in missing_required]),
            affected_records=[],
            metrics={
                "record_count": len(records),
                "source_trust_score": round(source_trust_score, 6),
                "provenance_confidence": round(provenance_confidence, 6),
                "metadata_completeness": round(metadata_completeness, 6),
                "record_provenance_coverage": round(record_provenance_coverage, 6),
                "memory_reliability": round(memory_reliability, 6),
                "explicit_trust": round(explicit_trust, 6),
                "recency_score": round(recency_score, 6),
                "source_mismatch": 1.0 if source_mismatch else 0.0,
            },
            remediation_actions=self._merge_string_sets(remediation_actions),
            details={
                "provenance": self._normalized_mapping(provenance),
                "required_fields": required_fields,
                "optional_fields": optional_fields,
                "present_required": present_required,
                "missing_required": missing_required,
                "present_optional": present_optional,
                "latest_memory_reliability": latest_reliability,
            },
            error=error_payload,
        )

    # ------------------------------------------------------------------
    # Rule engine and orchestration helpers
    # ------------------------------------------------------------------
    def _evaluate_consistency_rule(
        self,
        record: Mapping[str, Any],
        rule: Mapping[str, Any],
        input_refs: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        rule_type = str(rule.get("type", "")).strip().lower()
        rule_id = self._nonempty(rule.get("rule_id") or rule_type or "unnamed_rule", "rule_id")
        severity = str(rule.get("severity", "medium")).strip().lower()
        message = str(rule.get("message") or f"Consistency rule '{rule_id}' violated.")
        record_payload = dict(record)

        if rule_type == "temporal_order":
            earlier = self._parse_datetime(record_payload.get(rule.get("earlier_field")))
            later = self._parse_datetime(record_payload.get(rule.get("later_field")))
            if earlier is None or later is None or earlier <= later:
                return None
            return {
                "rule_id": rule_id,
                "rule_type": rule_type,
                "severity": severity,
                "message": message,
                "fields": [rule.get("earlier_field"), rule.get("later_field")],
                "observed": {
                    str(rule.get("earlier_field")): record_payload.get(rule.get("earlier_field")),
                    str(rule.get("later_field")): record_payload.get(rule.get("later_field")),
                },
            }

        if rule_type == "required_if":
            if_field = rule.get("if_field")
            expected = self._normalize_value(rule.get("if_equals"))
            required_field = rule.get("required_field")
            current = self._normalize_value(record_payload.get(if_field))
            if current != expected or self._is_present(record_payload.get(required_field)):
                return None
            return {
                "rule_id": rule_id,
                "rule_type": rule_type,
                "severity": severity,
                "message": message,
                "fields": [if_field, required_field],
                "observed": {str(if_field): record_payload.get(if_field), str(required_field): record_payload.get(required_field)},
            }

        if rule_type == "equals_if_present":
            field_name = rule.get("field")
            value = record_payload.get(field_name)
            if not self._is_present(value):
                return None
            expected = self._resolve_expected_value(rule, record_payload, input_refs)
            if self._normalize_value(value) == self._normalize_value(expected):
                return None
            return {
                "rule_id": rule_id,
                "rule_type": rule_type,
                "severity": severity,
                "message": message,
                "fields": [field_name],
                "observed": {str(field_name): value, "expected": expected},
            }

        if rule_type == "not_equals_if_present":
            field_name = rule.get("field")
            value = record_payload.get(field_name)
            if not self._is_present(value):
                return None
            disallowed = self._resolve_expected_value(rule, record_payload, input_refs)
            if self._normalize_value(value) != self._normalize_value(disallowed):
                return None
            return {
                "rule_id": rule_id,
                "rule_type": rule_type,
                "severity": severity,
                "message": message,
                "fields": [field_name],
                "observed": {str(field_name): value, "disallowed": disallowed},
            }

        if rule_type == "allowed_values":
            field_name = rule.get("field")
            value = record_payload.get(field_name)
            if not self._is_present(value):
                return None
            allowed_values = {self._normalize_value(item) for item in self._string_list(rule.get("allowed_values", []))}
            if self._normalize_value(value) in allowed_values:
                return None
            return {
                "rule_id": rule_id,
                "rule_type": rule_type,
                "severity": severity,
                "message": message,
                "fields": [field_name],
                "observed": {str(field_name): value, "allowed_values": sorted(allowed_values)},
            }

        if rule_type == "one_of_present":
            fields = self._string_list(rule.get("fields", []))
            if any(self._is_present(record_payload.get(field_name)) for field_name in fields):
                return None
            return {
                "rule_id": rule_id,
                "rule_type": rule_type,
                "severity": severity,
                "message": message,
                "fields": fields,
                "observed": {field_name: record_payload.get(field_name) for field_name in fields},
            }

        if rule_type == "regex":
            field_name = rule.get("field")
            value = record_payload.get(field_name)
            pattern = str(rule.get("pattern", ""))
            if not self._is_present(value) or not pattern:
                return None
            if re.search(pattern, str(value)):
                return None
            return {
                "rule_id": rule_id,
                "rule_type": rule_type,
                "severity": severity,
                "message": message,
                "fields": [field_name],
                "observed": {str(field_name): value, "pattern": pattern},
            }

        raise DataQualityError(
            message=f"Unsupported semantic consistency rule type '{rule_type}'",
            error_type=QualityErrorType.CONFIGURATION_INVALID,
            severity=QualitySeverity.HIGH,
            retryable=False,
            stage=QualityStage.VALIDATION,
            domain=QualityDomain.SYSTEM,
            disposition=QualityDisposition.ESCALATE,
            context={"rule_id": rule_id, "rule": self._normalized_mapping(rule)},
            remediation="Use a supported consistency rule type or extend the semantic rule engine.",
        )

    def _resolve_expected_value(
        self,
        rule: Mapping[str, Any],
        record: Mapping[str, Any],
        input_refs: Mapping[str, Any],
    ) -> Any:
        if "compare_to_field" in rule:
            return record.get(rule.get("compare_to_field"))
        if "compare_to_value" in rule:
            return rule.get("compare_to_value")
        if "compare_to_value_ref" in rule:
            return input_refs.get(str(rule.get("compare_to_value_ref")))
        return None

    def _aggregate_batch_score(self, findings: Sequence[SemanticFinding]) -> float:
        denominator = sum(self.component_weights.values())
        if denominator <= 0:
            raise DataQualityError(
                message="semantic_quality scoring weights must sum to a positive value",
                error_type=QualityErrorType.CONFIGURATION_INVALID,
                severity=QualitySeverity.HIGH,
                retryable=False,
                stage=QualityStage.SCORING,
                domain=QualityDomain.SYSTEM,
                disposition=QualityDisposition.ESCALATE,
                context={"weights": self.component_weights},
                remediation="Set positive semantic scoring weights for leakage, consistency, and provenance.",
            )

        indexed = {finding.check_name: finding for finding in findings}
        weighted_score = 0.0
        for key, weight in self.component_weights.items():
            weighted_score += weight * float(indexed[key].score)

        batch_score = weighted_score / denominator
        if any(finding.verdict == "block" and finding.severity in {"high", "critical"} for finding in findings):
            batch_score = min(batch_score, 0.24)
        elif any(finding.verdict == "warn" for finding in findings):
            batch_score = min(batch_score, 0.89)
        return round(min(max(batch_score, 0.0), 1.0), 6)

    def _resolve_final_verdict(
        self,
        batch_score: float,
        findings: Sequence[SemanticFinding],
        conflict_resolution: Mapping[str, Any],
    ) -> str:
        if any(finding.verdict == "block" for finding in findings):
            return "block"

        conflict_verdict = str(conflict_resolution.get("reconciled_verdict", "pass")).strip().lower()
        if conflict_verdict == "block":
            return "block"
        if conflict_verdict == "warn":
            return "warn"

        if batch_score >= self.pass_threshold:
            return "pass"
        if batch_score >= self.warn_threshold:
            return "warn"
        return "block"

    # ------------------------------------------------------------------
    # Normalization, parsing, config, and utility helpers
    # ------------------------------------------------------------------
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
        if self.pass_threshold <= 0 or self.pass_threshold > 1:
            raise self._configuration_error("semantic_quality.scoring.pass_threshold must be within (0, 1]")
        if self.warn_threshold < 0 or self.warn_threshold >= self.pass_threshold:
            raise self._configuration_error(
                "semantic_quality.scoring.warn_threshold must be within [0, pass_threshold)"
            )
        if any(weight < 0 for weight in self.component_weights.values()):
            raise self._configuration_error("semantic_quality scoring weights must be non-negative")
        if sum(self.component_weights.values()) <= 0:
            raise self._configuration_error("semantic_quality scoring weights must sum to a positive value")
        self._bounded_unit_interval(
            self.provenance_config.get("min_trust_score", 0.65),
            "semantic_quality.provenance.min_trust_score",
        )
        self._bounded_unit_interval(
            self.provenance_config.get("block_below_trust", 0.45),
            "semantic_quality.provenance.block_below_trust",
        )

    def _configuration_error(self, message: str) -> DataQualityError:
        return DataQualityError(
            message=message,
            error_type=QualityErrorType.CONFIGURATION_INVALID,
            severity=QualitySeverity.HIGH,
            retryable=False,
            stage=QualityStage.VALIDATION,
            domain=QualityDomain.SYSTEM,
            disposition=QualityDisposition.ESCALATE,
            remediation="Correct the semantic_quality configuration in quality_config.yaml before running the semantic gate.",
        )

    def _merge_provenance_inputs(
        self,
        *,
        source_id: str,
        provenance: Optional[Mapping[str, Any]],
        source_metadata: Optional[Mapping[str, Any]],
        schema_version: Optional[str],
        batch_id: str,
    ) -> Dict[str, Any]:
        merged = {
            "source_id": source_id,
            "batch_id": batch_id,
            **self._normalized_mapping(source_metadata),
            **self._normalized_mapping(provenance),
        }
        if schema_version is not None and not self._is_present(merged.get("schema_version")):
            merged["schema_version"] = schema_version
        return merged

    def _discover_feature_fields(
        self,
        records: Sequence[Mapping[str, Any]],
        label_field: str,
        excluded_fields: Iterable[str],
    ) -> List[str]:
        discovered: List[str] = []
        excluded = set(excluded_fields) | {label_field, "provenance"}
        excluded.update(self.record_id_candidates)
        for record in records:
            for key in record.keys():
                key_text = str(key)
                if key_text in excluded or key_text.startswith("_"):
                    continue
                if key_text not in discovered:
                    discovered.append(key_text)
        return discovered

    def _field_risk_level(
        self,
        field_name: str,
        blocked_terms: Sequence[str],
        warned_terms: Sequence[str],
        future_terms: Sequence[str],
    ) -> str:
        lowered = str(field_name).strip().lower()
        if any(term and term in lowered for term in blocked_terms):
            return "block"
        if any(term and term in lowered for term in future_terms):
            return "warn"
        if any(term and term in lowered for term in warned_terms):
            return "warn"
        return "none"

    def _normalize_records(self, records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise DataQualityError(
                    message="Semantic quality records must be mapping-like objects",
                    error_type=QualityErrorType.SCHEMA_VALIDATION_FAILED,
                    severity=QualitySeverity.HIGH,
                    retryable=False,
                    stage=QualityStage.VALIDATION,
                    domain=QualityDomain.SEMANTIC,
                    disposition=QualityDisposition.BLOCK,
                    context={"record_index": index, "record_type": type(record).__name__},
                    remediation="Convert records to dictionaries before semantic evaluation.",
                )
            normalized.append(self._normalized_mapping(record))
        return normalized

    def _record_id(self, record: Mapping[str, Any]) -> str:
        for candidate in self.record_id_candidates:
            if self._is_present(record.get(candidate)):
                return str(record.get(candidate))
        return self._normalize_value(record)[:32] or self._new_id("record")

    def _normalized_mapping(self, value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if value is None:
            return {}
        return json.loads(json.dumps(dict(value), default=str))

    def _normalize_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, (int, float)):
            if isinstance(value, float) and math.isnan(value):
                return ""
            return str(value).strip().lower()
        if isinstance(value, str):
            return re.sub(r"\s+", " ", value.strip().lower())
        if isinstance(value, Mapping):
            return json.dumps(value, sort_keys=True, default=str).strip().lower()
        if isinstance(value, (list, tuple, set)):
            return json.dumps(list(value), sort_keys=False, default=str).strip().lower()
        return re.sub(r"\s+", " ", str(value).strip().lower())

    def _is_present(self, value: Any) -> bool:
        normalized = self._normalize_value(value)
        return normalized not in {"", "none", "null", "nan"}

    def _similarity(self, left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        if left == right:
            return 1.0
        if len(left) > 256 or len(right) > 256:
            return 0.0
        return SequenceMatcher(a=left, b=right).ratio()

    def _parse_datetime(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            pass
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            parsed = datetime.fromisoformat(text)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
        except ValueError:
            return None

    def _recency_score(self, timestamp_epoch: Optional[float], *, fresh_seconds: float, acceptable_seconds: float) -> float:
        if timestamp_epoch is None:
            return 0.25
        age_seconds = max(0.0, time.time() - float(timestamp_epoch))
        if age_seconds <= fresh_seconds:
            return 1.0
        if age_seconds <= acceptable_seconds:
            span = max(acceptable_seconds - fresh_seconds, 1.0)
            return max(0.40, 1.0 - ((age_seconds - fresh_seconds) / span))
        stale_excess = age_seconds - acceptable_seconds
        return max(0.05, 0.40 - min(stale_excess / max(acceptable_seconds, 1.0), 0.35))

    def _coerce_float(self, value: Any, *, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _bounded_unit_interval(self, value: Any, field_name: str) -> float:
        try:
            numeric = float(value)
        except Exception as exc:
            raise DataQualityError(
                message=f"{field_name} must be numeric",
                error_type=QualityErrorType.CONFIGURATION_INVALID,
                severity=QualitySeverity.HIGH,
                retryable=False,
                stage=QualityStage.VALIDATION,
                domain=QualityDomain.SYSTEM,
                disposition=QualityDisposition.ESCALATE,
                context={"field_name": field_name, "value": value},
                remediation="Provide a numeric value between 0.0 and 1.0.",
                cause=exc,
            ) from exc
        if numeric < 0.0 or numeric > 1.0:
            raise DataQualityError(
                message=f"{field_name} must be within [0.0, 1.0]",
                error_type=QualityErrorType.CONFIGURATION_INVALID,
                severity=QualitySeverity.HIGH,
                retryable=False,
                stage=QualityStage.VALIDATION,
                domain=QualityDomain.SYSTEM,
                disposition=QualityDisposition.ESCALATE,
                context={"field_name": field_name, "value": numeric},
                remediation="Provide a numeric value between 0.0 and 1.0.",
            )
        return numeric

    def _normalized_weights(self, value: Mapping[str, Any], defaults: Mapping[str, float]) -> Dict[str, float]:
        weights = {str(key): float(value.get(key, defaults[key])) for key in defaults}
        total = sum(weights.values())
        if total <= 0:
            raise self._configuration_error("semantic_quality provenance confidence weights must sum to a positive value")
        return {key: numeric / total for key, numeric in weights.items()}

    def _string_list(self, values: Optional[Iterable[Any]]) -> List[str]:
        if values is None:
            return []
        return [str(item) for item in values]

    def _merge_string_sets(self, *groups: Iterable[str]) -> List[str]:
        ordered: List[str] = []
        seen = set()
        for group in groups:
            for item in group:
                text = str(item).strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                ordered.append(text)
        return ordered

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
                disposition=QualityDisposition.ESCALATE,
                context={"field_name": field_name, "value": value},
                remediation="Provide a non-empty semantic evaluation identifier or field name.",
            )
        return text

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}_{int(time.time() * 1000)}_{os.urandom(4).hex()}"


if __name__ == "__main__":
    print("\n=== Running Semantic Quality ===\n")
    printer.status("TEST", "Semantic Quality initialized", "info")

    semantic = SemanticQuality()
    printer.status("CONFIG", f"Loaded semantic_quality config from {semantic.config.get('__config_path__', 'unknown')}", "success")

    clean_records = [
        {
            "record_id": "clean_001",
            "entity_id": "doc_001",
            "label": "approved",
            "text": "Customer identity verified and documentation complete.",
            "status": "resolved",
            "resolved_at": "2026-04-08T11:30:00+00:00",
            "start_time": "2026-04-08T10:00:00+00:00",
            "end_time": "2026-04-08T11:00:00+00:00",
            "source_id": "source_semantic_alpha",
            "provenance": {"collector": "semantic_tester", "source_id": "source_semantic_alpha"},
        },
        {
            "record_id": "clean_002",
            "entity_id": "doc_002",
            "label": "rejected",
            "text": "Insufficient supporting evidence and failed verification checks.",
            "status": "resolved",
            "resolved_at": "2026-04-08T11:45:00+00:00",
            "start_time": "2026-04-08T10:05:00+00:00",
            "end_time": "2026-04-08T11:10:00+00:00",
            "source_id": "source_semantic_alpha",
            "provenance": {"collector": "semantic_tester", "source_id": "source_semantic_alpha"},
        },
    ]

    clean_provenance = {
        "source_id": "source_semantic_alpha",
        "source_type": "curated_dataset",
        "collected_at": "2026-04-08T11:50:00+00:00",
        "checksum": "sha256:6d2e0f9a1ec7ab7b",
        "lineage_id": "lineage_001",
        "collector": "semantic_quality_test",
        "uri": "internal://quality/semantic/source_semantic_alpha",
        "trust_score": 0.93,
        "schema_version": "v1.0.0",
    }

    clean_assessment = semantic.evaluate_batch(
        clean_records,
        source_id="source_semantic_alpha",
        batch_id="semantic_batch_clean_001",
        label_field="label",
        provenance=clean_provenance,
        schema_version="v1.0.0",
        context={"scenario": "clean_batch"},
    )
    printer.pretty("CLEAN_ASSESSMENT", clean_assessment, "success")

    risky_records = [
        {
            "record_id": "risk_001",
            "entity_id": "doc_900",
            "label": "approved",
            "target_label": "approved",
            "text": "approved",
            "status": "resolved",
            "resolved_at": None,
            "start_time": "2026-04-08T13:00:00+00:00",
            "end_time": "2026-04-08T12:00:00+00:00",
            "source_id": "unexpected_source",
        },
        {
            "record_id": "risk_002",
            "entity_id": "doc_900",
            "label": "rejected",
            "target_label": "rejected",
            "text": "rejected",
            "status": "resolved",
            "resolved_at": None,
            "start_time": "2026-04-08T14:00:00+00:00",
            "end_time": "2026-04-08T13:30:00+00:00",
            "source_id": "unexpected_source",
        },
    ]

    risky_provenance = {
        "source_id": "unexpected_source",
        "source_type": "external_feed",
        "collected_at": "2025-11-01T00:00:00+00:00",
        "trust_score": 0.20,
        "collector": "unknown_scraper",
    }

    leakage_result = semantic.detect_label_leakage(
        risky_records,
        label_field="label",
        source_id="source_semantic_beta",
        batch_id="semantic_batch_risk_001",
    )
    consistency_result = semantic.validate_cross_field_consistency(
        risky_records,
        label_field="label",
        source_id="source_semantic_beta",
        batch_id="semantic_batch_risk_001",
    )
    provenance_result = semantic.assess_provenance(
        risky_records,
        source_id="source_semantic_beta",
        batch_id="semantic_batch_risk_001",
        provenance=risky_provenance,
    )
    risky_assessment = semantic.evaluate_batch(
        risky_records,
        source_id="source_semantic_beta",
        batch_id="semantic_batch_risk_001",
        label_field="label",
        provenance=risky_provenance,
        context={"scenario": "risky_batch"},
    )

    latest_state = semantic.memory.latest_quality_state("source_semantic_beta")

    assert clean_assessment["verdict"] in {"pass", "warn"}, "Clean batch should not hard block"
    assert leakage_result["verdict"] in {"warn", "block"}, "Risky batch should surface leakage concerns"
    assert consistency_result["metrics"]["entity_label_conflict_count"] >= 1, "Duplicate entity label conflict should be detected"
    assert provenance_result["verdict"] == "block", "Low-trust provenance should block"
    assert risky_assessment["verdict"] == "block", "Combined risky semantic batch should block"
    assert latest_state is not None, "Semantic evaluation should record a quality snapshot to memory"
    assert latest_state["batch_id"] == "semantic_batch_risk_001", "Latest quality state should reference the latest batch"

    printer.pretty("LEAKAGE_RESULT", leakage_result, "warning")
    printer.pretty("CONSISTENCY_RESULT", consistency_result, "warning")
    printer.pretty("PROVENANCE_RESULT", provenance_result, "warning")
    printer.pretty("RISKY_ASSESSMENT", risky_assessment, "warning")
    printer.pretty("LATEST_MEMORY_STATE", latest_state, "success")

    semantic.memory.flush()
    printer.status("TEST", "Semantic Quality test flow completed", "success")
    print("\n=== Test ran successfully ===\n")
