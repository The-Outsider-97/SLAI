"""
- Distribution shift detection against baselines.
- Missingness profile and anomaly thresholds.
- Duplicate/outlier scoring.
"""

from __future__ import annotations

import hashlib
import json
import math
import time

from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from statistics import mean, median, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .utils.config_loader import load_global_config, get_config_section
from .utils.quality_error import (DataQualityError, DriftThresholdError,
                                  QualityErrorType, QualityMemoryError,
                                  QualitySeverity, normalize_quality_exception)
from .quality_memory import QualityMemory
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Statistical Quality")
printer = PrettyPrinter


@dataclass
class StatisticalFinding:
    check: str
    verdict: str
    severity: str
    score: float
    confidence: float
    flags: List[str] = field(default_factory=list)
    affected_records: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    field_details: Dict[str, Any] = field(default_factory=dict)
    remediation_actions: List[str] = field(default_factory=list)
    incidents: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StatisticalBatchResult:
    source_id: str
    batch_id: str
    verdict: str
    batch_score: float
    flags: List[str]
    quarantine_count: int
    shift_metrics: Dict[str, float]
    remediation_actions: List[str]
    findings: List[Dict[str, Any]]
    missingness_profile: Dict[str, Any]
    duplicate_profile: Dict[str, Any]
    outlier_profile: Dict[str, Any]
    drift_profile: Dict[str, Any]
    baseline_registered: bool
    context: Dict[str, Any] = field(default_factory=dict)
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["generated_at_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.generated_at))
        return payload


class StatisticalQuality:
    """Production-oriented statistical gate for batch quality evaluation.

    The Statistical Quality module evaluates whether a batch is numerically and
    distributionally fit for downstream use. It focuses on:
    - distribution shift against trusted baselines,
    - missingness profiles and threshold breaches,
    - duplicate concentration,
    - outlier burden across numeric fields.

    The module is designed to integrate with QualityMemory when richer memory APIs
    are available, while still remaining usable as a stateful runtime component.
    """

    def __init__(self) -> None:
        self.config = load_global_config()
        self.quality_config = get_config_section("statistical_quality")
        self.memory = QualityMemory()
        self._runtime_baselines: Dict[str, Dict[str, Any]] = {}
        self._validate_runtime_configuration()

        self.enabled = bool(self.quality_config.get("enabled", True))
        self.auto_record_to_memory = bool(self.quality_config.get("auto_record_to_memory", True))
        self.default_window = str(self.quality_config.get("default_window", "latest")).strip() or "latest"
        self.record_id_candidates = [str(item) for item in self.quality_config.get("ids", {}).get(
            "record_id_candidates", ["record_id", "id", "row_id", "sample_id", "uuid"]
        )]

        self.pass_threshold = float(self.quality_config.get("scoring", {}).get("pass_threshold", 0.90))
        self.warn_threshold = float(self.quality_config.get("scoring", {}).get("warn_threshold", 0.75))
        self.weights = {
            "drift": float(self.quality_config.get("scoring", {}).get("weights", {}).get("drift", 0.35)),
            "missingness": float(self.quality_config.get("scoring", {}).get("weights", {}).get("missingness", 0.25)),
            "duplicates": float(self.quality_config.get("scoring", {}).get("weights", {}).get("duplicates", 0.20)),
            "outliers": float(self.quality_config.get("scoring", {}).get("weights", {}).get("outliers", 0.20)),
        }

        self.baseline_config = self.quality_config.get("baseline", {})
        self.missingness_config = self.quality_config.get("missingness", {})
        self.duplicates_config = self.quality_config.get("duplicates", {})
        self.outliers_config = self.quality_config.get("outliers", {})
        self.drift_config = self.quality_config.get("drift", {})
        self.remediation_config = self.quality_config.get("remediation", {})
        self.memory_config = self.quality_config.get("memory", {})

        logger.info(
            "Statistical Quality initialized | enabled=%s | pass_threshold=%.3f | warn_threshold=%.3f",
            self.enabled,
            self.pass_threshold,
            self.warn_threshold,
        )

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------
    def assess_batch(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        source_id: str,
        batch_id: str,
        baseline: Optional[Mapping[str, Any]] = None,
        required_fields: Optional[Sequence[str]] = None,
        window: Optional[str] = None,
        schema_version: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._boundary(
            stage="statistical_assessment",
            context={"source_id": source_id, "batch_id": batch_id},
        ):
            source_key = self._nonempty(source_id, "source_id")
            batch_key = self._nonempty(batch_id, "batch_id")
            prepared_records = self._normalize_records(records)
            if not prepared_records:
                raise DataQualityError(
                    message="Statistical quality assessment requires at least one record",
                    error_type=QualityErrorType.SCORING_PIPELINE_FAILED,
                    severity=QualitySeverity.HIGH,
                    retryable=False,
                    context={"source_id": source_key, "batch_id": batch_key},
                    remediation="Provide a non-empty batch before running statistical quality checks.",
                )

            effective_window = str(window or self.default_window)
            profile = self._build_profile(prepared_records)
            baseline_profile, baseline_registered = self._resolve_baseline(
                source_id=source_key,
                current_profile=profile,
                explicit_baseline=baseline,
                window=effective_window,
            )

            missingness_finding = self.profile_missingness(
                prepared_records,
                required_fields=required_fields,
            )
            duplicate_finding = self.score_duplicates(prepared_records)
            outlier_finding = self.score_outliers(prepared_records)
            drift_finding = self.detect_distribution_shift(
                prepared_records,
                source_id=source_key,
                batch_id=batch_key,
                current_profile=profile,
                baseline_profile=baseline_profile,
                window=effective_window,
            )

            findings = [
                missingness_finding,
                duplicate_finding,
                outlier_finding,
                drift_finding,
            ]

            batch_score = self._weighted_score(findings)
            verdict = self._final_verdict(findings=findings, batch_score=batch_score)
            remediation_actions = self._collect_remediation_actions(findings)
            flags = self._collect_flags(findings)
            quarantine_count = max(
                missingness_finding.affected_records,
                duplicate_finding.affected_records,
                outlier_finding.affected_records,
                drift_finding.affected_records,
            ) if verdict == "block" else 0

            result = StatisticalBatchResult(
                source_id=source_key,
                batch_id=batch_key,
                verdict=verdict,
                batch_score=batch_score,
                flags=flags,
                quarantine_count=quarantine_count,
                shift_metrics=self._extract_shift_metrics(drift_finding),
                remediation_actions=remediation_actions,
                findings=[item.to_dict() for item in findings],
                missingness_profile=missingness_finding.metrics,
                duplicate_profile=duplicate_finding.metrics,
                outlier_profile=outlier_finding.metrics,
                drift_profile=drift_finding.metrics,
                baseline_registered=baseline_registered,
                context=self._normalized_mapping(context),
            )

            if self.auto_record_to_memory and self.memory_config.get("record_quality_snapshot", True):
                self._record_quality_snapshot(
                    source_id=source_key,
                    batch_id=batch_key,
                    batch_score=batch_score,
                    verdict=verdict,
                    findings=findings,
                    schema_version=schema_version,
                    window=effective_window,
                    quarantine_count=quarantine_count,
                    context=context,
                )

            return result.to_dict()

    def evaluate_batch(self, records: Sequence[Mapping[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        return self.assess_batch(records, **kwargs)

    def profile_missingness(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        required_fields: Optional[Sequence[str]] = None,
    ) -> StatisticalFinding:
        with self._boundary(stage="missingness_profile"):
            total_records = len(records)
            fields = self._field_union(records, required_fields=required_fields)
            required_set = {str(item) for item in (required_fields or self.missingness_config.get("required_fields", []))}

            field_missing_counts: Dict[str, int] = {field: 0 for field in fields}
            required_record_failures: List[str] = []
            record_level_missing = 0

            for record in records:
                record_missing = False
                record_id = self._resolve_record_id(record)
                for field in fields:
                    if self._is_missing(record.get(field)):
                        field_missing_counts[field] += 1
                        record_missing = True
                        if field in required_set:
                            required_record_failures.append(record_id)
                if record_missing:
                    record_level_missing += 1

            total_cells = max(total_records * max(len(fields), 1), 1)
            total_missing = sum(field_missing_counts.values())
            overall_missing_rate = total_missing / total_cells
            record_missing_rate = record_level_missing / max(total_records, 1)
            field_missing_rates = {
                field: field_missing_counts[field] / max(total_records, 1)
                for field in fields
            }

            warn_total_rate = float(self.missingness_config.get("warn_total_rate", 0.05))
            block_total_rate = float(self.missingness_config.get("block_total_rate", 0.12))
            warn_field_rate = float(self.missingness_config.get("warn_field_rate", 0.10))
            block_field_rate = float(self.missingness_config.get("block_field_rate", 0.25))
            anomaly_std_multiplier = float(self.missingness_config.get("anomaly_stddev_multiplier", 2.0))

            rates = list(field_missing_rates.values())
            mean_rate = mean(rates) if rates else 0.0
            std_rate = pstdev(rates) if len(rates) > 1 else 0.0
            anomaly_threshold = mean_rate + (std_rate * anomaly_std_multiplier)
            anomalous_fields = sorted(
                [field for field, rate in field_missing_rates.items() if rate >= anomaly_threshold and rate > 0],
                key=lambda item: field_missing_rates[item],
                reverse=True,
            )

            severity = "low"
            verdict = "pass"
            if overall_missing_rate >= block_total_rate or any(rate >= block_field_rate for rate in field_missing_rates.values()):
                verdict = "block"
                severity = "high"
            elif required_record_failures or overall_missing_rate >= warn_total_rate or any(rate >= warn_field_rate for rate in field_missing_rates.values()):
                verdict = "warn"
                severity = "medium"

            score = self._score_from_rate(
                observed=max(
                    overall_missing_rate,
                    max(field_missing_rates.values(), default=0.0),
                    warn_field_rate if required_record_failures else 0.0,
                ),
                warn_threshold=warn_total_rate,
                block_threshold=block_total_rate,
            )
            confidence = self._bounded_probability(0.60 + min(record_missing_rate, 0.35))
            flags = []
            if anomalous_fields:
                flags.append("missingness_anomaly")
            if required_record_failures:
                flags.append("required_field_missingness")
            if verdict == "block":
                flags.append("missingness_block")
            elif verdict == "warn":
                flags.append("missingness_warn")

            incidents: List[Dict[str, Any]] = []
            if verdict != "pass":
                err = DataQualityError(
                    message=(
                        f"Missingness threshold exceeded: overall_missing_rate={overall_missing_rate:.4f}, "
                        f"record_missing_rate={record_missing_rate:.4f}"
                    ),
                    error_type=QualityErrorType.REQUIRED_FIELD_MISSING if required_record_failures else QualityErrorType.SCORING_PIPELINE_FAILED,
                    severity=QualitySeverity.HIGH if verdict == "block" else QualitySeverity.MEDIUM,
                    retryable=False,
                    context={
                        "overall_missing_rate": overall_missing_rate,
                        "record_missing_rate": record_missing_rate,
                        "anomalous_fields": anomalous_fields,
                        "required_record_failures": required_record_failures[:100],
                    },
                    remediation="Apply imputation, re-fetch incomplete records, or quarantine records violating required completeness policy.",
                )
                err.report()
                incidents.append(err.to_dict())

            return StatisticalFinding(
                check="missingness",
                verdict=verdict,
                severity=severity,
                score=score,
                confidence=confidence,
                flags=flags,
                affected_records=len(set(required_record_failures)) or record_level_missing,
                metrics={
                    "record_count": total_records,
                    "field_count": len(fields),
                    "total_missing": total_missing,
                    "overall_missing_rate": overall_missing_rate,
                    "record_missing_rate": record_missing_rate,
                    "required_missing_records": len(set(required_record_failures)),
                    "anomaly_threshold": anomaly_threshold,
                    "anomalous_fields": anomalous_fields,
                },
                field_details={
                    "field_missing_rates": field_missing_rates,
                    "field_missing_counts": field_missing_counts,
                    "required_fields": sorted(required_set),
                },
                remediation_actions=self._string_list(self.remediation_config.get("missingness_actions", [
                    "impute_missing_values",
                    "re_fetch_incomplete_records",
                    "quarantine_incomplete_records",
                ])),
                incidents=incidents,
            )

    def score_duplicates(self, records: Sequence[Mapping[str, Any]]) -> StatisticalFinding:
        with self._boundary(stage="duplicate_scoring"):
            exclude_fields = set(self._string_list(
                self.duplicates_config.get("fingerprint_fields_exclude", [
                    "id", "record_id", "uuid", "created_at", "updated_at", "timestamp",
                ])
            ))
            key_fields = self._string_list(self.duplicates_config.get("key_fields", []))

            fingerprints: Dict[str, List[str]] = defaultdict(list)
            key_groups: Dict[str, List[str]] = defaultdict(list)
            for record in records:
                record_id = self._resolve_record_id(record)
                fingerprints[self._record_fingerprint(record, exclude_fields)].append(record_id)
                if key_fields:
                    key_payload = {field: record.get(field) for field in key_fields}
                    key_groups[self._stable_json(key_payload)].append(record_id)

            duplicate_groups = [group for group in fingerprints.values() if len(group) > 1]
            duplicate_records = sum(len(group) - 1 for group in duplicate_groups)
            duplicate_rate = duplicate_records / max(len(records), 1)

            duplicate_key_groups = [group for group in key_groups.values() if len(group) > 1]
            key_duplicate_records = sum(len(group) - 1 for group in duplicate_key_groups)
            key_duplicate_rate = key_duplicate_records / max(len(records), 1) if key_fields else 0.0

            warn_rate = float(self.duplicates_config.get("warn_rate", 0.03))
            block_rate = float(self.duplicates_config.get("block_rate", 0.10))
            effective_rate = max(duplicate_rate, key_duplicate_rate)

            verdict = "pass"
            severity = "low"
            if effective_rate >= block_rate:
                verdict = "block"
                severity = "high"
            elif effective_rate >= warn_rate:
                verdict = "warn"
                severity = "medium"

            incidents: List[Dict[str, Any]] = []
            if verdict != "pass":
                err = DataQualityError(
                    message=f"Duplicate rate exceeded: observed={effective_rate:.4f}",
                    error_type=QualityErrorType.DUPLICATE_RATE_EXCEEDED,
                    severity=QualitySeverity.HIGH if verdict == "block" else QualitySeverity.MEDIUM,
                    retryable=False,
                    context={
                        "duplicate_rate": duplicate_rate,
                        "key_duplicate_rate": key_duplicate_rate,
                        "duplicate_group_count": len(duplicate_groups),
                    },
                    remediation="Deduplicate the batch, enforce record identity constraints, or reduce source weight for repeated payloads.",
                )
                err.report()
                incidents.append(err.to_dict())

            return StatisticalFinding(
                check="duplicates",
                verdict=verdict,
                severity=severity,
                score=self._score_from_rate(effective_rate, warn_rate, block_rate),
                confidence=self._bounded_probability(0.55 + min(effective_rate * 5.0, 0.40)),
                flags=self._string_list([
                    "duplicate_rate_block" if verdict == "block" else "duplicate_rate_warn" if verdict == "warn" else "",
                ]),
                affected_records=max(duplicate_records, key_duplicate_records),
                metrics={
                    "record_count": len(records),
                    "duplicate_groups": len(duplicate_groups),
                    "duplicate_records": duplicate_records,
                    "duplicate_rate": duplicate_rate,
                    "key_duplicate_groups": len(duplicate_key_groups),
                    "key_duplicate_records": key_duplicate_records,
                    "key_duplicate_rate": key_duplicate_rate,
                },
                field_details={
                    "sample_duplicate_groups": duplicate_groups[:10],
                    "key_fields": key_fields,
                    "sample_key_duplicate_groups": duplicate_key_groups[:10],
                },
                remediation_actions=self._string_list(self.remediation_config.get("duplicate_actions", [
                    "drop_duplicate_records",
                    "merge_duplicate_records",
                    "enforce_identity_constraints",
                    "review_source_replay",
                ])),
                incidents=incidents,
            )

    def score_outliers(self, records: Sequence[Mapping[str, Any]]) -> StatisticalFinding:
        with self._boundary(stage="outlier_scoring"):
            numeric_fields = self._numeric_fields(records)
            min_observations = int(self.outliers_config.get("min_numeric_observations", 8))
            zscore_threshold = float(self.outliers_config.get("zscore_threshold", 3.0))
            iqr_multiplier = float(self.outliers_config.get("iqr_multiplier", 1.5))
            warn_rate = float(self.outliers_config.get("warn_rate", 0.04))
            block_rate = float(self.outliers_config.get("block_rate", 0.10))
            max_fields = int(self.outliers_config.get("max_fields", 100))

            field_outlier_counts: Dict[str, int] = {}
            outlier_records: set[str] = set()
            field_stats: Dict[str, Any] = {}

            for field in numeric_fields[:max_fields]:
                values_with_ids = [
                    (self._resolve_record_id(record), float(record[field]))
                    for record in records
                    if field in record and self._is_numeric(record[field])
                ]
                if len(values_with_ids) < min_observations:
                    continue

                values = [value for _, value in values_with_ids]
                center = mean(values)
                spread = pstdev(values) if len(values) > 1 else 0.0
                q1 = self._percentile(values, 0.25)
                q3 = self._percentile(values, 0.75)
                iqr = q3 - q1
                lower_fence = q1 - (iqr_multiplier * iqr)
                upper_fence = q3 + (iqr_multiplier * iqr)

                field_stats[field] = {
                    "count": len(values),
                    "mean": center,
                    "std": spread,
                    "median": median(values),
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "lower_fence": lower_fence,
                    "upper_fence": upper_fence,
                }

                field_count = 0
                for record_id, value in values_with_ids:
                    zscore = 0.0 if spread == 0.0 else abs((value - center) / spread)
                    iqr_outlier = value < lower_fence or value > upper_fence if iqr > 0.0 else False
                    zscore_outlier = zscore >= zscore_threshold if spread > 0.0 else False
                    if iqr_outlier or zscore_outlier:
                        outlier_records.add(record_id)
                        field_count += 1
                if field_count:
                    field_outlier_counts[field] = field_count

            outlier_rate = len(outlier_records) / max(len(records), 1)
            verdict = "pass"
            severity = "low"
            if outlier_rate >= block_rate:
                verdict = "block"
                severity = "high"
            elif outlier_rate >= warn_rate:
                verdict = "warn"
                severity = "medium"

            incidents: List[Dict[str, Any]] = []
            if verdict != "pass":
                err = DataQualityError(
                    message=f"Outlier rate exceeded: observed={outlier_rate:.4f}",
                    error_type=QualityErrorType.OUTLIER_RATE_EXCEEDED,
                    severity=QualitySeverity.HIGH if verdict == "block" else QualitySeverity.MEDIUM,
                    retryable=False,
                    context={
                        "outlier_rate": outlier_rate,
                        "field_outlier_counts": field_outlier_counts,
                    },
                    remediation="Review abnormal numeric fields, cap or remove extreme values, and verify whether the batch belongs to a shifted operating regime.",
                )
                err.report()
                incidents.append(err.to_dict())

            return StatisticalFinding(
                check="outliers",
                verdict=verdict,
                severity=severity,
                score=self._score_from_rate(outlier_rate, warn_rate, block_rate),
                confidence=self._bounded_probability(0.50 + min(outlier_rate * 6.0, 0.45)),
                flags=self._string_list([
                    "outlier_rate_block" if verdict == "block" else "outlier_rate_warn" if verdict == "warn" else "",
                ]),
                affected_records=len(outlier_records),
                metrics={
                    "numeric_fields_considered": len(field_stats),
                    "outlier_record_count": len(outlier_records),
                    "outlier_rate": outlier_rate,
                },
                field_details={
                    "field_outlier_counts": field_outlier_counts,
                    "field_statistics": field_stats,
                    "sample_outlier_record_ids": sorted(outlier_records)[:25],
                },
                remediation_actions=self._string_list(self.remediation_config.get("outlier_actions", [
                    "winsorize_extreme_values",
                    "quarantine_outlier_records",
                    "review_numeric_scaling",
                    "revalidate_sensor_or_source",
                ])),
                incidents=incidents,
            )

    def detect_distribution_shift(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        source_id: str,
        batch_id: str,
        current_profile: Optional[Mapping[str, Any]] = None,
        baseline_profile: Optional[Mapping[str, Any]] = None,
        window: Optional[str] = None,
    ) -> StatisticalFinding:
        with self._boundary(
            stage="distribution_shift_detection",
            context={"source_id": source_id, "batch_id": batch_id},
        ):
            current = dict(current_profile or self._build_profile(records))
            baseline = dict(baseline_profile or {})
            if not baseline:
                if not bool(self.baseline_config.get("auto_register_if_missing", True)):
                    raise DataQualityError(
                        message=f"No statistical baseline available for source '{source_id}'",
                        error_type=QualityErrorType.BASELINE_NOT_FOUND,
                        severity=QualitySeverity.MEDIUM,
                        retryable=False,
                        context={"source_id": source_id, "batch_id": batch_id},
                        remediation="Provide an explicit baseline or enable baseline auto-registration.",
                    )
                return StatisticalFinding(
                    check="drift",
                    verdict="pass",
                    severity="low",
                    score=1.0,
                    confidence=0.40,
                    flags=["baseline_registered"],
                    affected_records=0,
                    metrics={
                        "drift_score": 0.0,
                        "baseline_available": False,
                        "baseline_registered": True,
                        "window": str(window or self.default_window),
                    },
                    field_details={
                        "numeric_field_shift": {},
                        "categorical_field_shift": {},
                        "missing_rate_delta": {},
                    },
                    remediation_actions=self._string_list(self.remediation_config.get("drift_warn_actions", [
                        "monitor_next_batches",
                        "validate_baseline_quality",
                    ])),
                    incidents=[],
                )

            min_observations = int(self.drift_config.get("min_observations", 20))
            warn_score_threshold = float(self.drift_config.get("warn_score", 0.35))
            exclude_fields = set(self._string_list(self.drift_config.get("exclude_fields", [])))
            block_score_threshold = float(self.drift_config.get("block_score", 0.70))
            numeric_mean_warn = float(self.drift_config.get("numeric_mean_relative_warn", 0.20))
            numeric_mean_block = float(self.drift_config.get("numeric_mean_relative_block", 0.45))
            numeric_std_ratio_warn = float(self.drift_config.get("numeric_std_ratio_warn", 1.75))
            numeric_std_ratio_block = float(self.drift_config.get("numeric_std_ratio_block", 2.50))
            categorical_js_warn = float(self.drift_config.get("categorical_js_warn", 0.08))
            categorical_js_block = float(self.drift_config.get("categorical_js_block", 0.18))
            missing_delta_warn = float(self.drift_config.get("missing_rate_delta_warn", 0.05))
            missing_delta_block = float(self.drift_config.get("missing_rate_delta_block", 0.12))

            numeric_alerts: Dict[str, Any] = {}
            categorical_alerts: Dict[str, Any] = {}
            missing_delta_alerts: Dict[str, Any] = {}
            normalized_components: List[float] = []
            incidents: List[Dict[str, Any]] = []

            for field, field_profile in current.get("numeric_fields", {}).items():
                if field in exclude_fields:
                    continue
                base = baseline.get("numeric_fields", {}).get(field)
                if not base:
                    continue
                if min(field_profile.get("count", 0), base.get("count", 0)) < min_observations:
                    continue
                base_mean = float(base.get("mean", 0.0))
                curr_mean = float(field_profile.get("mean", 0.0))
                mean_relative_delta = abs(curr_mean - base_mean) / max(abs(base_mean), 1.0)
                base_std = abs(float(base.get("std", 0.0)))
                curr_std = abs(float(field_profile.get("std", 0.0)))
                if base_std == 0.0 and curr_std == 0.0:
                    std_ratio = 1.0
                elif min(base_std, curr_std) == 0.0:
                    std_ratio = max(base_std, curr_std, 1.0)
                else:
                    std_ratio = max(base_std, curr_std) / min(base_std, curr_std)

                component = max(
                    self._ratio_severity(mean_relative_delta, numeric_mean_warn, numeric_mean_block),
                    self._ratio_severity(std_ratio, numeric_std_ratio_warn, numeric_std_ratio_block),
                )
                normalized_components.append(component)
                if component > 0.0:
                    numeric_alerts[field] = {
                        "mean_relative_delta": mean_relative_delta,
                        "std_ratio": std_ratio,
                        "severity": component,
                    }
                    if component >= 1.0:
                        try:
                            drift_error = DriftThresholdError(
                                source_id=source_id,
                                metric=field,
                                observed=max(mean_relative_delta, std_ratio),
                                threshold=max(numeric_mean_block, numeric_std_ratio_block),
                                context={"batch_id": batch_id, "field": field},
                            )
                            drift_error.report()
                            incidents.append(drift_error.to_dict())
                        except Exception:
                            pass
                    self._record_drift_observation(
                        source_id=source_id,
                        metric=f"numeric::{field}",
                        observed=max(mean_relative_delta, std_ratio),
                        drift_score=component,
                        threshold=block_score_threshold,
                        batch_id=batch_id,
                        window=window,
                        baseline_value=base_mean,
                        context={"mean_relative_delta": mean_relative_delta, "std_ratio": std_ratio},
                    )

            for field, field_profile in current.get("categorical_fields", {}).items():
                if field in exclude_fields:
                    continue
                base = baseline.get("categorical_fields", {}).get(field)
                if not base:
                    continue
                divergence = self._jensen_shannon_divergence(
                    base.get("distribution", {}),
                    field_profile.get("distribution", {}),
                )
                component = self._ratio_severity(divergence, categorical_js_warn, categorical_js_block)
                normalized_components.append(component)
                if component > 0.0:
                    categorical_alerts[field] = {
                        "jensen_shannon_divergence": divergence,
                        "severity": component,
                    }
                    self._record_drift_observation(
                        source_id=source_id,
                        metric=f"categorical::{field}",
                        observed=divergence,
                        drift_score=component,
                        threshold=block_score_threshold,
                        batch_id=batch_id,
                        window=window,
                        baseline_value=None,
                        context={"field": field, "type": "categorical"},
                    )

            for field, field_missing_rate in current.get("missing_rate_by_field", {}).items():
                if field in exclude_fields:
                    continue
                base_rate = float(baseline.get("missing_rate_by_field", {}).get(field, 0.0))
                delta = abs(float(field_missing_rate) - base_rate)
                component = self._ratio_severity(delta, missing_delta_warn, missing_delta_block)
                normalized_components.append(component)
                if component > 0.0:
                    missing_delta_alerts[field] = {
                        "missing_rate_delta": delta,
                        "current_missing_rate": field_missing_rate,
                        "baseline_missing_rate": base_rate,
                        "severity": component,
                    }
                    self._record_drift_observation(
                        source_id=source_id,
                        metric=f"missingness::{field}",
                        observed=delta,
                        drift_score=component,
                        threshold=block_score_threshold,
                        batch_id=batch_id,
                        window=window,
                        baseline_value=base_rate,
                        context={"field": field, "type": "missingness_delta"},
                    )

            active_components = [component for component in normalized_components if component > 0.0]
            aggregate_drift_score = max(active_components) if active_components else 0.0
            verdict = "pass"
            severity = "low"
            if aggregate_drift_score >= block_score_threshold or any(
                detail.get("severity", 0.0) >= 1.0
                for bucket in (numeric_alerts, categorical_alerts, missing_delta_alerts)
                for detail in bucket.values()
            ):
                verdict = "block"
                severity = "high"
            elif aggregate_drift_score >= warn_score_threshold or active_components:
                verdict = "warn"
                severity = "medium"

            if verdict != "pass" and not incidents:
                err = DataQualityError(
                    message=f"Distribution drift detected for source '{source_id}' with score={aggregate_drift_score:.4f}",
                    error_type=QualityErrorType.DISTRIBUTION_DRIFT_DETECTED,
                    severity=QualitySeverity.HIGH if verdict == "block" else QualitySeverity.MEDIUM,
                    retryable=False,
                    context={
                        "source_id": source_id,
                        "batch_id": batch_id,
                        "drift_score": aggregate_drift_score,
                        "numeric_alerts": numeric_alerts,
                        "categorical_alerts": categorical_alerts,
                        "missing_delta_alerts": missing_delta_alerts,
                    },
                    remediation="Quarantine the shifted batch, compare with recent trusted baselines, and refresh or segment the upstream data source.",
                )
                err.report()
                incidents.append(err.to_dict())

            return StatisticalFinding(
                check="drift",
                verdict=verdict,
                severity=severity,
                score=max(0.0, 1.0 - min(aggregate_drift_score, 1.0)),
                confidence=self._bounded_probability(0.55 + min(aggregate_drift_score * 0.35, 0.35)),
                flags=self._string_list([
                    "distribution_shift_block" if verdict == "block" else "distribution_shift_warn" if verdict == "warn" else "",
                    "baseline_available",
                ]),
                affected_records=len(records) if verdict == "block" else 0,
                metrics={
                    "drift_score": aggregate_drift_score,
                    "baseline_available": True,
                    "window": str(window or self.default_window),
                    "numeric_alert_count": len(numeric_alerts),
                    "categorical_alert_count": len(categorical_alerts),
                    "missing_delta_alert_count": len(missing_delta_alerts),
                },
                field_details={
                    "numeric_field_shift": numeric_alerts,
                    "categorical_field_shift": categorical_alerts,
                    "missing_rate_delta": missing_delta_alerts,
                },
                remediation_actions=self._string_list(
                    self.remediation_config.get(
                        "drift_block_actions" if verdict == "block" else "drift_warn_actions",
                        [
                            "quarantine_shifted_batch",
                            "revalidate_baseline",
                            "segment_source_distribution",
                            "re_fetch_recent_data",
                        ],
                    )
                ),
                incidents=incidents,
            )

    # ------------------------------------------------------------------
    # Internal scoring and integration helpers
    # ------------------------------------------------------------------
    def _resolve_baseline(
        self,
        *,
        source_id: str,
        current_profile: Mapping[str, Any],
        explicit_baseline: Optional[Mapping[str, Any]],
        window: str,
    ) -> Tuple[Dict[str, Any], bool]:
        if explicit_baseline is not None:
            baseline_profile = self._normalized_mapping(explicit_baseline)
            self._runtime_baselines[source_id] = baseline_profile
            return baseline_profile, False

        if source_id in self._runtime_baselines:
            return self._runtime_baselines[source_id], False

        memory_read_enabled = bool(self.baseline_config.get("read_from_memory", True))
        if memory_read_enabled:
            baseline_profile = self._get_memory_baseline(source_id)
            if baseline_profile:
                self._runtime_baselines[source_id] = baseline_profile
                return baseline_profile, False

        auto_register = bool(self.baseline_config.get("auto_register_if_missing", True))
        if auto_register:
            baseline_profile = self._normalized_mapping(current_profile)
            self._runtime_baselines[source_id] = baseline_profile
            if bool(self.baseline_config.get("write_to_memory", True)):
                self._record_drift_baseline(
                    source_id=source_id,
                    metric="dataset_profile",
                    baseline=baseline_profile,
                    window=window,
                    context={"baseline_kind": "runtime_auto_registered"},
                )
            return baseline_profile, True

        return {}, False

    def _get_memory_baseline(self, source_id: str) -> Dict[str, Any]:
        method = getattr(self.memory, "latest_quality_state", None)
        if not callable(method):
            return {}
        try:
            state = method(source_id)
        except Exception as exc:
            normalized = normalize_quality_exception(
                exc,
                stage="memory_baseline_retrieval",
                context={"source_id": source_id},
            )
            normalized.report()
            logger.error("Failed to retrieve statistical baseline from memory: %s", normalized)
            return {}
        if not isinstance(state, Mapping):
            return {}
        baseline_profile = state.get("context", {}).get("statistical_profile")
        return self._normalized_mapping(baseline_profile) if isinstance(baseline_profile, Mapping) else {}

    def _record_quality_snapshot(
        self,
        *,
        source_id: str,
        batch_id: str,
        batch_score: float,
        verdict: str,
        findings: Sequence[StatisticalFinding],
        schema_version: Optional[str],
        window: str,
        quarantine_count: int,
        context: Optional[Mapping[str, Any]],
    ) -> None:
        method = getattr(self.memory, "record_quality_snapshot", None)
        if not callable(method):
            logger.warning("QualityMemory.record_quality_snapshot unavailable; skipping statistical snapshot persistence")
            return
        shift_metrics = self._extract_shift_metrics(next(item for item in findings if item.check == "drift"))
        try:
            method(
                source_id=source_id,
                batch_id=batch_id,
                batch_score=batch_score,
                verdict=verdict,
                flags=self._collect_flags(findings),
                quarantine_count=quarantine_count,
                shift_metrics=shift_metrics,
                remediation_actions=self._collect_remediation_actions(findings),
                source_reliability=float((self._normalized_mapping(context or {})).get("source_reliability", 0.5)),
                schema_version=schema_version,
                window=window,
                checker_findings=[
                    {
                        "checker": "statistical",
                        "check": item.check,
                        "verdict": item.verdict,
                        "severity": item.severity,
                        "confidence": item.confidence,
                        "score": item.score,
                    }
                    for item in findings
                ],
                context={
                    **self._normalized_mapping(context),
                    "statistical_profile": self._runtime_baselines.get(source_id),
                    "module": "StatisticalQuality",
                },
            )
        except Exception as exc:
            self._handle_memory_error("record_quality_snapshot", exc, {"source_id": source_id, "batch_id": batch_id})

    def _record_drift_baseline(
        self,
        *,
        source_id: str,
        metric: str,
        baseline: Mapping[str, Any],
        window: Optional[str],
        context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        method = getattr(self.memory, "record_drift_baseline", None)
        if not callable(method):
            return
        try:
            method(
                source_id=source_id,
                metric=metric,
                baseline=baseline,
                window=window,
                context=context,
            )
        except Exception as exc:
            self._handle_memory_error("record_drift_baseline", exc, {"source_id": source_id, "metric": metric})

    def _record_drift_observation(
        self,
        *,
        source_id: str,
        metric: str,
        observed: float,
        drift_score: float,
        threshold: Optional[float],
        batch_id: Optional[str],
        window: Optional[str],
        baseline_value: Optional[float],
        context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        method = getattr(self.memory, "record_drift_observation", None)
        if not callable(method):
            return
        try:
            method(
                source_id=source_id,
                metric=metric,
                observed=observed,
                drift_score=drift_score,
                threshold=threshold,
                baseline_value=baseline_value,
                window=window,
                batch_id=batch_id,
                is_alert=bool(threshold is not None and drift_score >= threshold),
                context=context,
            )
        except Exception as exc:
            self._handle_memory_error(
                "record_drift_observation",
                exc,
                {"source_id": source_id, "metric": metric, "batch_id": batch_id},
            )

    def _handle_memory_error(self, operation: str, exc: Exception, context: Mapping[str, Any]) -> None:
        if isinstance(exc, DataQualityError):
            exc.report()
            logger.error("Statistical memory integration error during %s: %s", operation, exc)
            return
        error = QualityMemoryError(operation, str(exc), context=dict(context))
        error.report()
        logger.error("Statistical memory integration error during %s: %s", operation, error)

    def _weighted_score(self, findings: Sequence[StatisticalFinding]) -> float:
        findings_by_check = {item.check: item for item in findings}
        total_weight = 0.0
        total_score = 0.0
        for check, weight in self.weights.items():
            if weight <= 0:
                continue
            item = findings_by_check.get(check)
            if item is None:
                continue
            total_weight += weight
            total_score += item.score * weight
        if total_weight == 0:
            raise DataQualityError(
                message="Statistical scoring weights must sum to a positive value",
                error_type=QualityErrorType.POLICY_THRESHOLD_INVALID,
                severity=QualitySeverity.HIGH,
                retryable=False,
                context={"weights": self.weights},
                remediation="Configure at least one positive scoring weight for the statistical quality module.",
            )
        return round(total_score / total_weight, 6)

    def _final_verdict(self, *, findings: Sequence[StatisticalFinding], batch_score: float) -> str:
        if any(item.verdict == "block" for item in findings):
            return "block"
        if batch_score >= self.pass_threshold and all(item.verdict == "pass" for item in findings):
            return "pass"
        if batch_score >= self.warn_threshold:
            return "warn"
        return "block"

    def _collect_flags(self, findings: Sequence[StatisticalFinding]) -> List[str]:
        seen: List[str] = []
        for finding in findings:
            for flag in finding.flags:
                if flag and flag not in seen:
                    seen.append(flag)
        return seen

    def _collect_remediation_actions(self, findings: Sequence[StatisticalFinding]) -> List[str]:
        ordered: List[str] = []
        for finding in findings:
            for action in finding.remediation_actions:
                if action not in ordered:
                    ordered.append(action)
        return ordered

    def _extract_shift_metrics(self, drift_finding: StatisticalFinding) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for key, value in drift_finding.metrics.items():
            if isinstance(value, (int, float)):
                metrics[str(key)] = float(value)
        metrics.setdefault("drift_score", float(drift_finding.metrics.get("drift_score", 0.0)))
        return metrics

    # ------------------------------------------------------------------
    # Profiling helpers
    # ------------------------------------------------------------------
    def _build_profile(self, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        field_names = self._field_union(records)
        numeric_fields: Dict[str, Any] = {}
        categorical_fields: Dict[str, Any] = {}
        missing_rate_by_field: Dict[str, float] = {}

        for field_name in field_names:
            column = [record.get(field_name) for record in records]
            missing_count = sum(1 for value in column if self._is_missing(value))
            missing_rate = missing_count / max(len(records), 1)
            missing_rate_by_field[field_name] = missing_rate

            numeric_values = [float(value) for value in column if self._is_numeric(value)]
            if len(numeric_values) >= int(self.drift_config.get("min_observations", 20)):
                numeric_fields[field_name] = {
                    "count": len(numeric_values),
                    "mean": mean(numeric_values),
                    "std": pstdev(numeric_values) if len(numeric_values) > 1 else 0.0,
                    "median": median(numeric_values),
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "p25": self._percentile(numeric_values, 0.25),
                    "p75": self._percentile(numeric_values, 0.75),
                }
                continue

            categorical_values = [self._normalize_categorical(value) for value in column if not self._is_missing(value)]
            if categorical_values:
                distribution = self._distribution(categorical_values)
                categorical_fields[field_name] = {
                    "count": len(categorical_values),
                    "unique_count": len(distribution),
                    "distribution": distribution,
                    "top_values": sorted(distribution.items(), key=lambda item: item[1], reverse=True)[:10],
                }

        return {
            "record_count": len(records),
            "field_count": len(field_names),
            "numeric_fields": numeric_fields,
            "categorical_fields": categorical_fields,
            "missing_rate_by_field": missing_rate_by_field,
            "generated_at": time.time(),
        }

    def _normalize_records(self, records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise DataQualityError(
                    message=f"Record at index {index} must be a mapping",
                    error_type=QualityErrorType.SCORING_PIPELINE_FAILED,
                    severity=QualitySeverity.HIGH,
                    retryable=False,
                    context={"index": index, "record_type": type(record).__name__},
                    remediation="Convert all records to dictionaries before statistical evaluation.",
                )
            normalized.append(dict(record))
        return normalized

    def _field_union(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        required_fields: Optional[Sequence[str]] = None,
    ) -> List[str]:
        fields: set[str] = set()
        for record in records:
            fields.update(str(key) for key in record.keys())
        for required_field in required_fields or []:
            fields.add(str(required_field))
        return sorted(fields)

    def _numeric_fields(self, records: Sequence[Mapping[str, Any]]) -> List[str]:
        candidates = Counter()
        for record in records:
            for required_field, value in record.items():
                if self._is_numeric(value):
                    candidates[str(required_field)] += 1
        return [field for field, _ in candidates.most_common()]

    # ------------------------------------------------------------------
    # Primitive helpers
    # ------------------------------------------------------------------
    def _validate_runtime_configuration(self) -> None:
        if not isinstance(self.quality_config, Mapping):
            raise DataQualityError(
                message="statistical_quality config section must be a mapping",
                error_type=QualityErrorType.CONFIGURATION_INVALID,
                severity=QualitySeverity.HIGH,
                retryable=False,
                context={"section": "statistical_quality"},
                remediation="Define statistical_quality as a dictionary in quality_config.yaml.",
            )
        scoring = self.quality_config.get("scoring", {})
        pass_threshold = float(scoring.get("pass_threshold", 0.90))
        warn_threshold = float(scoring.get("warn_threshold", 0.75))
        if not (0.0 <= warn_threshold <= pass_threshold <= 1.0):
            raise DataQualityError(
                message="Statistical quality thresholds must satisfy 0 <= warn_threshold <= pass_threshold <= 1",
                error_type=QualityErrorType.POLICY_THRESHOLD_INVALID,
                severity=QualitySeverity.HIGH,
                retryable=False,
                context={
                    "pass_threshold": pass_threshold,
                    "warn_threshold": warn_threshold,
                },
                remediation="Adjust scoring thresholds to valid ordered probability bounds.",
            )

    @contextmanager
    def _boundary(self, *, stage: str, context: Optional[Mapping[str, Any]] = None):
        try:
            yield
        except DataQualityError:
            raise
        except Exception as exc:
            raise normalize_quality_exception(exc, stage=stage, context=dict(context or {})) from exc

    def _nonempty(self, value: Any, field_name: str) -> str:
        text = str(value).strip() if value is not None else ""
        if not text:
            raise DataQualityError(
                message=f"{field_name} must not be empty",
                error_type=QualityErrorType.CONFIGURATION_INVALID,
                severity=QualitySeverity.MEDIUM,
                retryable=False,
                context={"field_name": field_name, "value": value},
                remediation="Provide a non-empty identifier.",
            )
        return text

    def _resolve_record_id(self, record: Mapping[str, Any]) -> str:
        for candidate in self.record_id_candidates:
            value = record.get(candidate)
            if value is not None and str(value).strip():
                return str(value)
        fallback = hashlib.sha256(self._stable_json(record).encode("utf-8")).hexdigest()[:12]
        return f"record_{fallback}"

    def _record_fingerprint(self, record: Mapping[str, Any], exclude_fields: Iterable[str]) -> str:
        payload = {str(k): v for k, v in record.items() if str(k) not in set(exclude_fields)}
        return hashlib.sha256(self._stable_json(payload).encode("utf-8")).hexdigest()

    def _stable_json(self, value: Any) -> str:
        return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))

    def _is_missing(self, value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() == ""
        if isinstance(value, (list, tuple, set, dict)):
            return len(value) == 0
        return False

    def _is_numeric(self, value: Any) -> bool:
        if isinstance(value, bool) or value is None:
            return False
        if isinstance(value, (int, float)):
            return math.isfinite(float(value))
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return False
            try:
                parsed = float(text)
            except Exception:
                return False
            return math.isfinite(parsed)
        return False

    def _normalize_categorical(self, value: Any) -> str:
        if isinstance(value, str):
            return value.strip().lower()
        return str(value)

    def _distribution(self, values: Sequence[str]) -> Dict[str, float]:
        counts = Counter(values)
        total = float(sum(counts.values())) or 1.0
        return {key: count / total for key, count in counts.items()}

    def _percentile(self, values: Sequence[float], q: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(float(item) for item in values)
        if len(sorted_values) == 1:
            return sorted_values[0]
        position = (len(sorted_values) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return sorted_values[int(position)]
        lower_value = sorted_values[lower]
        upper_value = sorted_values[upper]
        return lower_value + (upper_value - lower_value) * (position - lower)

    def _jensen_shannon_divergence(self, baseline: Mapping[str, float], current: Mapping[str, float]) -> float:
        labels = set(baseline.keys()) | set(current.keys())
        if not labels:
            return 0.0
        p = {label: float(baseline.get(label, 0.0)) for label in labels}
        q = {label: float(current.get(label, 0.0)) for label in labels}
        m = {label: (p[label] + q[label]) / 2.0 for label in labels}

        def _kl_divergence(left: Mapping[str, float], right: Mapping[str, float]) -> float:
            total = 0.0
            for label in labels:
                lv = left[label]
                rv = right[label]
                if lv > 0.0 and rv > 0.0:
                    total += lv * math.log(lv / rv, 2)
            return total

        divergence = (_kl_divergence(p, m) + _kl_divergence(q, m)) / 2.0
        return max(0.0, divergence)

    def _ratio_severity(self, observed: float, warn_threshold: float, block_threshold: float) -> float:
        if observed <= warn_threshold:
            return 0.0
        if block_threshold <= warn_threshold:
            return 1.0
        if observed >= block_threshold:
            return 1.0
        return (observed - warn_threshold) / (block_threshold - warn_threshold)

    def _score_from_rate(self, observed: float, warn_threshold: float, block_threshold: float) -> float:
        severity = self._ratio_severity(observed, warn_threshold, block_threshold)
        return round(max(0.0, 1.0 - severity), 6)

    def _bounded_probability(self, value: float) -> float:
        return round(min(max(float(value), 0.0), 1.0), 6)

    def _string_list(self, values: Iterable[Any]) -> List[str]:
        return [str(item) for item in values if str(item).strip()]

    def _normalized_mapping(self, mapping: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if mapping is None:
            return {}
        return json.loads(json.dumps(dict(mapping), default=str))


if __name__ == "__main__":
    print("\n=== Running Statistical Quality ===\n")
    printer.status("TEST", "Statistical Quality initialized", "info")

    stats = StatisticalQuality()
    printer.status("CONFIG", "Statistical Quality config loaded", "success")

    baseline_records = [
        {
            "record_id": f"baseline_{index}",
            "value": 10.0 + (index % 5),
            "score": 0.60 + ((index % 4) * 0.02),
            "category": "alpha" if index < 15 else "beta",
            "status": "ready",
            "sensor": 100 + (index % 3),
        }
        for index in range(30)
    ]

    baseline_result = stats.assess_batch(
        baseline_records,
        source_id="source_alpha",
        batch_id="baseline_batch_001",
        required_fields=["record_id", "value", "score", "category"],
        window="30d",
        schema_version="v1.0.0",
        context={"source_reliability": 0.95, "purpose": "baseline_establishment"},
    )
    printer.pretty("BASELINE_RESULT", baseline_result, "success")

    noisy_records: List[Dict[str, Any]] = []
    for index in range(30):
        noisy_records.append(
            {
                "record_id": f"batch_{index}",
                "value": 18.0 + (index * 0.7),
                "score": 0.55 + ((index % 5) * 0.03),
                "category": "gamma" if index < 24 else "beta",
                "status": "ready",
                "sensor": 100 + (index % 4),
            }
        )

    noisy_records[2]["score"] = None
    noisy_records[3]["category"] = ""
    noisy_records[4]["value"] = 250.0
    noisy_records[5] = dict(noisy_records[1])
    noisy_records[5]["record_id"] = "batch_5_duplicate"
    noisy_records[7]["sensor"] = 999.0
    noisy_records[9]["value"] = -75.0

    assessment = stats.assess_batch(
        noisy_records,
        source_id="source_alpha",
        batch_id="batch_2026_04_09_001",
        required_fields=["record_id", "value", "score", "category"],
        window="24h",
        schema_version="v1.0.0",
        context={
            "source_reliability": 0.88,
            "ingestion_path": "reader->training->statistical_quality",
            "trigger": "quality_gate",
        },
    )
    printer.pretty("ASSESSMENT", assessment, "success")

    printer.status("SUMMARY", f"Final verdict: {assessment['verdict']} | score={assessment['batch_score']}", "info")
    printer.status("SUMMARY", f"Flags: {assessment['flags']}", "info")

    print("\n=== Test ran successfully ===\n")
