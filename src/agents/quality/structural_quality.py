"""
- Schema/type validation.
- Required-field completeness checks.
- Range/domain constraints (e.g., timestamps, enum values).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .utils.config_loader import load_global_config, get_config_section
from .utils.quality_error import (DataQualityError, QualityErrorType, QualitySeverity,
                                  QualityMemoryError, SchemaValidationError, normalize_quality_exception)
from .quality_memory import QualityMemory
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Structural Quality")
printer = PrettyPrinter


@contextmanager
def _quality_boundary(stage: str, context: Optional[Mapping[str, Any]] = None):
    """Normalize arbitrary failures into the shared data quality error contract."""
    try:
        yield
    except Exception as exc:  # pragma: no cover - deliberate broad boundary.
        normalized = normalize_quality_exception(exc, stage=stage, context=dict(context or {}))
        raise normalized from exc


@dataclass(slots=True)
class StructuralFinding:
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
            "domain": "structural",
            "verdict": self.verdict,
            "severity": self.severity,
            "confidence": self.confidence,
            "flags": list(self.flags),
        }


@dataclass(slots=True)
class StructuralAssessment:
    assessment_id: str
    dataset_id: str
    source_id: str
    batch_id: str
    verdict: str
    batch_score: float
    record_count: int
    reviewed_record_count: int
    quarantine_count: int
    flags: List[str]
    remediation_actions: List[str]
    shift_metrics: Dict[str, float]
    findings: List[Dict[str, Any]]
    schema_version: Optional[str]
    window: Optional[str]
    context: Dict[str, Any]
    created_at: float
    schema_hash: Optional[str] = None
    memory_snapshot: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["created_at_iso"] = datetime.fromtimestamp(self.created_at, tz=timezone.utc).isoformat()
        return payload


class StructuralQuality:
    """Production-grade structural quality gate for schema, completeness, and constraint integrity.

    Structural quality is the earliest deterministic barrier in the data-quality stack. Its goal is
    to ensure that records are *well-formed enough* to be trusted by downstream semantic checks,
    learning pipelines, inference context builders, and memory writers.

    The module performs five tightly-related responsibilities:
    - schema/type validation,
    - required-field completeness analysis,
    - range constraint enforcement,
    - domain and pattern constraint enforcement,
    - schema registration / snapshot persistence into quality memory when available.
    """

    def __init__(self) -> None:
        self.config = load_global_config()
        self.quality_config = get_config_section("structural_quality")
        self.memory = QualityMemory()

        self.enabled = bool(self.quality_config.get("enabled", True))
        self.auto_record_to_memory = bool(self.quality_config.get("auto_record_to_memory", True))
        self.default_window = str(self.quality_config.get("default_window", "latest")).strip() or "latest"
        self.record_id_candidates = self._string_list(
            self.quality_config.get(
                "ids",
                {},
            ).get("record_id_candidates", ["record_id", "id", "row_id", "sample_id", "uuid"])
        )

        self.pass_threshold = float(self.quality_config.get("scoring", {}).get("pass_threshold", 0.90))
        self.warn_threshold = float(self.quality_config.get("scoring", {}).get("warn_threshold", 0.75))
        self.weights = self._normalized_weights(
            self.quality_config.get(
                "scoring",
                {},
            ).get(
                "weights",
                {
                    "schema": 0.35,
                    "required": 0.25,
                    "range": 0.15,
                    "domain": 0.15,
                    "coercion": 0.10,
                },
            ),
            expected_keys=["schema", "required", "range", "domain", "coercion"],
        )

        schema_cfg = self.quality_config.get("schema", {})
        self.strict_unknown_fields = bool(schema_cfg.get("strict_unknown_fields", False))
        self.allow_missing_schema = bool(schema_cfg.get("allow_missing_schema", False))
        self.read_expected_schema_from_memory = bool(schema_cfg.get("read_expected_schema_from_memory", True))
        self.write_schema_to_memory = bool(schema_cfg.get("write_schema_to_memory", True))
        self.enforce_schema_version_match = bool(schema_cfg.get("enforce_schema_version_match", True))
        self.infer_types_from_samples = bool(schema_cfg.get("infer_types_from_samples", True))
        self.max_schema_inference_records = self._positive_int(
            schema_cfg.get("max_schema_inference_records", 100),
            "structural_quality.schema.max_schema_inference_records",
        )

        completeness_cfg = self.quality_config.get("completeness", {})
        self.allow_empty_strings = bool(completeness_cfg.get("allow_empty_strings", False))
        self.warn_missing_record_rate = float(completeness_cfg.get("warn_missing_record_rate", 0.05))
        self.block_missing_record_rate = float(completeness_cfg.get("block_missing_record_rate", 0.15))
        self.warn_missing_field_rate = float(completeness_cfg.get("warn_missing_field_rate", 0.08))
        self.block_missing_field_rate = float(completeness_cfg.get("block_missing_field_rate", 0.20))

        coercion_cfg = self.quality_config.get("coercion", {})
        self.coercion_enabled = bool(coercion_cfg.get("enabled", True))
        self.apply_coerced_values = bool(coercion_cfg.get("apply_coerced_values", False))
        self.warn_coercion_rate = float(coercion_cfg.get("warn_rate", 0.05))
        self.block_coercion_rate = float(coercion_cfg.get("block_rate", 0.20))
        self.truthy_values = {str(v).strip().lower() for v in coercion_cfg.get("truthy_values", ["true", "1", "yes", "y", "on"])}
        self.falsy_values = {str(v).strip().lower() for v in coercion_cfg.get("falsy_values", ["false", "0", "no", "n", "off"])}
        self.datetime_formats = [str(v) for v in coercion_cfg.get(
            "datetime_formats",
            [
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
                "%Y/%m/%d",
            ],
        )]

        constraints_cfg = self.quality_config.get("constraints", {})
        self.warn_range_violation_rate = float(constraints_cfg.get("range_warn_violation_rate", 0.03))
        self.block_range_violation_rate = float(constraints_cfg.get("range_block_violation_rate", 0.10))
        self.warn_domain_violation_rate = float(constraints_cfg.get("domain_warn_violation_rate", 0.03))
        self.block_domain_violation_rate = float(constraints_cfg.get("domain_block_violation_rate", 0.10))
        self.warn_regex_violation_rate = float(constraints_cfg.get("regex_warn_violation_rate", 0.02))
        self.block_regex_violation_rate = float(constraints_cfg.get("regex_block_violation_rate", 0.08))
        self.warn_unknown_field_rate = float(constraints_cfg.get("unknown_field_warn_rate", 0.05))
        self.block_unknown_field_rate = float(constraints_cfg.get("unknown_field_block_rate", 0.15))

        remediation_cfg = self.quality_config.get("remediation", {})
        self.schema_actions = self._string_list(remediation_cfg.get("schema_actions", [
            "normalize_schema",
            "drop_invalid_records",
            "re_fetch_malformed_records",
            "align_schema_version",
        ]))
        self.required_actions = self._string_list(remediation_cfg.get("required_actions", [
            "impute_required_fields",
            "re_fetch_incomplete_records",
            "quarantine_incomplete_records",
            "enforce_required_field_contract",
        ]))
        self.range_actions = self._string_list(remediation_cfg.get("range_actions", [
            "clamp_numeric_range",
            "quarantine_out_of_range_records",
            "review_upstream_transformations",
        ]))
        self.domain_actions = self._string_list(remediation_cfg.get("domain_actions", [
            "normalize_enum_values",
            "quarantine_invalid_domain_records",
            "review_source_contract",
        ]))
        self.coercion_actions = self._string_list(remediation_cfg.get("coercion_actions", [
            "stabilize_type_casting",
            "review_upstream_serialization",
            "tighten_reader_normalization",
        ]))

        memory_cfg = self.quality_config.get("memory", {})
        self.memory_record_quality_snapshot = bool(memory_cfg.get("record_quality_snapshot", True))
        self.memory_register_schema_version = bool(memory_cfg.get("register_schema_version", True))
        self.memory_rule_id_prefix = str(memory_cfg.get("rule_id_prefix", "structural")).strip() or "structural"

        self._validate_runtime_configuration()

        logger.info(
            "Structural Quality initialized | enabled=%s | auto_record_to_memory=%s | strict_unknown_fields=%s",
            self.enabled,
            self.auto_record_to_memory,
            self.strict_unknown_fields,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate_batch(self, records: Sequence[Mapping[str, Any]], *, dataset_id: str,
        source_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        schema: Optional[Mapping[str, Any]] = None,
        window: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        with _quality_boundary(
            "validation",
            {
                "dataset_id": dataset_id,
                "source_id": source_id,
                "batch_id": batch_id,
                "operation": "evaluate_batch",
            },
        ):
            dataset_key = self._nonempty_text(dataset_id, "dataset_id")
            source_key = self._nonempty_text(source_id or dataset_id, "source_id")
            batch_key = self._nonempty_text(batch_id or self._new_id("batch"), "batch_id")
            evaluation_context = self._normalized_mapping(context)
            input_records = self._normalize_records(records)
            record_count = len(input_records)
            resolved_window = window or self.default_window

            if not self.enabled:
                return StructuralAssessment(
                    assessment_id=self._new_id("structural_assessment"),
                    dataset_id=dataset_key,
                    source_id=source_key,
                    batch_id=batch_key,
                    verdict="pass",
                    batch_score=1.0,
                    record_count=record_count,
                    reviewed_record_count=0,
                    quarantine_count=0,
                    flags=["structural_quality_disabled"],
                    remediation_actions=[],
                    shift_metrics={},
                    findings=[],
                    schema_version=None,
                    window=resolved_window,
                    context=evaluation_context,
                    created_at=time.time(),
                ).to_dict()

            if record_count == 0:
                return StructuralAssessment(
                    assessment_id=self._new_id("structural_assessment"),
                    dataset_id=dataset_key,
                    source_id=source_key,
                    batch_id=batch_key,
                    verdict="warn",
                    batch_score=0.75,
                    record_count=0,
                    reviewed_record_count=0,
                    quarantine_count=0,
                    flags=["empty_batch"],
                    remediation_actions=["verify_upstream_reader_output"],
                    shift_metrics={
                        "schema_violation_rate": 0.0,
                        "missing_record_rate": 0.0,
                        "range_violation_rate": 0.0,
                        "domain_violation_rate": 0.0,
                        "coercion_rate": 0.0,
                    },
                    findings=[],
                    schema_version=None,
                    window=resolved_window,
                    context=evaluation_context,
                    created_at=time.time(),
                ).to_dict()

            normalized_schema = self._resolve_schema(
                schema=schema,
                records=input_records,
                dataset_id=dataset_key,
                source_id=source_key,
            )
            schema_version = normalized_schema.get("schema_version")
            schema_hash = self._schema_hash(normalized_schema)

            schema_version_finding = self._schema_version_consistency_finding(
                source_id=source_key,
                dataset_id=dataset_key,
                schema_version=schema_version,
                schema_hash=schema_hash,
                normalized_schema=normalized_schema,
            )

            analysis = self._analyze_records(
                records=input_records,
                normalized_schema=normalized_schema,
                dataset_id=dataset_key,
                source_id=source_key,
            )

            findings: List[StructuralFinding] = []
            findings.append(self._build_schema_finding(analysis, normalized_schema, dataset_key))
            findings.append(self._build_required_finding(analysis, normalized_schema, dataset_key))
            findings.append(self._build_range_finding(analysis, normalized_schema, dataset_key))
            findings.append(self._build_domain_finding(analysis, normalized_schema, dataset_key))
            findings.append(self._build_coercion_finding(analysis, dataset_key))
            if schema_version_finding is not None:
                findings.append(schema_version_finding)

            aggregate = self._aggregate_findings(findings)
            memory_snapshot = self._record_to_memory(
                findings=findings,
                dataset_id=dataset_key,
                source_id=source_key,
                batch_id=batch_key,
                schema_version=schema_version,
                schema_hash=schema_hash,
                normalized_schema=normalized_schema,
                aggregate=aggregate,
                window=resolved_window,
                context=evaluation_context,
            )

            assessment = StructuralAssessment(
                assessment_id=self._new_id("structural_assessment"),
                dataset_id=dataset_key,
                source_id=source_key,
                batch_id=batch_key,
                verdict=aggregate["verdict"],
                batch_score=aggregate["score"],
                record_count=record_count,
                reviewed_record_count=analysis["reviewed_record_count"],
                quarantine_count=aggregate["quarantine_count"],
                flags=aggregate["flags"],
                remediation_actions=aggregate["remediation_actions"],
                shift_metrics=aggregate["shift_metrics"],
                findings=[finding.to_dict() for finding in findings],
                schema_version=schema_version,
                schema_hash=schema_hash,
                window=resolved_window,
                context=evaluation_context,
                created_at=time.time(),
                memory_snapshot=memory_snapshot,
            )
            return assessment.to_dict()

    def assess_batch(self, records: Sequence[Mapping[str, Any]], *, dataset_id: str,
        source_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        schema: Optional[Mapping[str, Any]] = None,
        window: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.evaluate_batch(
            records,
            dataset_id=dataset_id,
            source_id=source_id,
            batch_id=batch_id,
            schema=schema,
            window=window,
            context=context,
        )

    def validate_schema(self, schema: Mapping[str, Any], *, dataset_id: str = "unknown") -> Dict[str, Any]:
        with _quality_boundary("validation", {"dataset_id": dataset_id, "operation": "validate_schema"}):
            return self._resolve_schema(schema=schema, records=[], dataset_id=dataset_id, source_id=dataset_id)

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------
    def _resolve_schema(self, *,
        schema: Optional[Mapping[str, Any]],
        records: Sequence[Mapping[str, Any]],
        dataset_id: str, source_id: str,
    ) -> Dict[str, Any]:
        if schema is None and self.read_expected_schema_from_memory and hasattr(self.memory, "latest_schema_version"):
            latest = self.memory.latest_schema_version(source_id)
            if latest is not None:
                schema = {
                    "schema_version": latest.get("schema_version"),
                    "required_fields": latest.get("required_fields", []),
                    "field_types": latest.get("field_types", {}),
                }

        if schema is None:
            if not self.allow_missing_schema and not self.infer_types_from_samples:
                raise SchemaValidationError(
                    dataset_id=dataset_id,
                    details="No schema supplied and schema inference is disabled.",
                    context={"source_id": source_id},
                )
            if not self.infer_types_from_samples:
                raise SchemaValidationError(
                    dataset_id=dataset_id,
                    details="No schema supplied and missing schema is not allowed.",
                    context={"source_id": source_id},
                )
            schema = self._infer_schema(records)

        normalized = self._normalize_schema(schema)
        if not normalized["fields"]:
            raise SchemaValidationError(
                dataset_id=dataset_id,
                details="Normalized schema is empty and cannot validate records.",
                context={"source_id": source_id},
            )
        return normalized

    def _normalize_schema(self, schema: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(schema, Mapping):
            raise DataQualityError(
                message="Schema must be a mapping-like object",
                error_type=QualityErrorType.SCHEMA_VALIDATION_FAILED,
                severity=QualitySeverity.HIGH,
                retryable=False,
                context={"received_type": type(schema).__name__},
                remediation="Provide a mapping with fields, required_fields, and optional constraints.",
            )

        raw_fields = schema.get("fields", {})
        raw_required = set(str(v) for v in schema.get("required_fields", []))
        raw_field_types = dict(schema.get("field_types", {}))
        raw_constraints = dict(schema.get("constraints", {}))

        for field_name, field_type in raw_field_types.items():
            raw_fields.setdefault(field_name, {"type": field_type})
        for field_name in raw_required:
            raw_fields.setdefault(field_name, {})
        for field_name, field_constraints in raw_constraints.items():
            entry = raw_fields.setdefault(field_name, {})
            if isinstance(field_constraints, Mapping):
                entry = {**dict(entry), **dict(field_constraints)}
            else:
                entry = {**dict(entry), "constraints": field_constraints}
            raw_fields[field_name] = entry

        normalized_fields: Dict[str, Dict[str, Any]] = {}
        for field_name, raw_definition in dict(raw_fields).items():
            if isinstance(raw_definition, str):
                definition: Dict[str, Any] = {"type": raw_definition}
            elif isinstance(raw_definition, Mapping):
                definition = dict(raw_definition)
            else:
                raise SchemaValidationError(
                    dataset_id=str(schema.get("schema_version", "unknown_schema")),
                    details=f"Field definition for '{field_name}' must be a mapping or string type alias.",
                    context={"field_name": field_name, "definition_type": type(raw_definition).__name__},
                )

            normalized_type = self._canonical_type(definition.get("type", "any"))
            enum_values = definition.get("enum", definition.get("allowed_values"))
            normalized_fields[str(field_name)] = {
                "type": normalized_type,
                "required": bool(definition.get("required", str(field_name) in raw_required)),
                "nullable": bool(definition.get("nullable", not bool(definition.get("required", str(field_name) in raw_required)))),
                "coerce": bool(definition.get("coerce", self.coercion_enabled)),
                "enum": self._string_list(enum_values) if enum_values is not None else [],
                "pattern": None if definition.get("pattern") is None else str(definition.get("pattern")),
                "min": definition.get("min"),
                "max": definition.get("max"),
                "min_length": definition.get("min_length"),
                "max_length": definition.get("max_length"),
                "min_items": definition.get("min_items"),
                "max_items": definition.get("max_items"),
                "description": None if definition.get("description") is None else str(definition.get("description")),
            }

        return {
            "schema_version": None if schema.get("schema_version") is None else str(schema.get("schema_version")),
            "fields": normalized_fields,
            "required_fields": sorted([name for name, spec in normalized_fields.items() if spec["required"]]),
        }

    def _infer_schema(self, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        sampled = list(records[: self.max_schema_inference_records])
        field_profiles: Dict[str, Dict[str, Any]] = {}
        for record in sampled:
            for key, value in record.items():
                key_text = str(key)
                bucket = field_profiles.setdefault(
                    key_text,
                    {
                        "count": 0,
                        "nonnull_count": 0,
                        "types": {},
                        "max_length": 0,
                    },
                )
                bucket["count"] += 1
                if value is None:
                    continue
                bucket["nonnull_count"] += 1
                inferred_type = self._infer_value_type(value)
                bucket["types"][inferred_type] = bucket["types"].get(inferred_type, 0) + 1
                if isinstance(value, (str, list, tuple, dict)):
                    bucket["max_length"] = max(bucket["max_length"], len(value))

        inferred_fields: Dict[str, Dict[str, Any]] = {}
        record_count = len(sampled)
        for field_name, profile in field_profiles.items():
            dominant_type = "any"
            if profile["types"]:
                dominant_type = max(profile["types"].items(), key=lambda item: (item[1], item[0]))[0]
            presence_ratio = profile["count"] / record_count if record_count else 0.0
            inferred_fields[field_name] = {
                "type": dominant_type,
                "required": presence_ratio >= 0.95,
                "nullable": profile["nonnull_count"] < profile["count"],
                "coerce": self.coercion_enabled,
                "max_length": profile["max_length"] if profile["max_length"] else None,
            }

        return {
            "schema_version": None,
            "fields": inferred_fields,
            "required_fields": [name for name, spec in inferred_fields.items() if spec.get("required")],
        }

    def _schema_version_consistency_finding(self, *, source_id: str, dataset_id: str, schema_version: Optional[str],
                                            schema_hash: Optional[str], normalized_schema: Mapping[str, Any],
                                            ) -> Optional[StructuralFinding]:
        if not self.read_expected_schema_from_memory or not hasattr(self.memory, "latest_schema_version"):
            return None
        latest = self.memory.latest_schema_version(source_id)
        if latest is None or not latest.get("schema_version"):
            return None

        latest_version = str(latest.get("schema_version"))
        if schema_version is None:
            return StructuralFinding(
                finding_id=self._new_id("structural_finding"),
                check_name="schema_version_consistency",
                verdict="warn",
                severity="medium",
                confidence=0.75,
                score=0.5,
                message=f"No schema_version supplied for dataset '{dataset_id}' while quality memory expects '{latest_version}'.",
                flags=["missing_schema_version"],
                affected_records=[],
                metrics={"expected_schema_version": latest_version, "supplied_schema_version": None},
                remediation_actions=self.schema_actions,
                details={"expected": latest, "schema_hash": schema_hash, "normalized_schema": normalized_schema},
            )

        if schema_version == latest_version:
            return None

        verdict = "block" if self.enforce_schema_version_match else "warn"
        severity = "high" if verdict == "block" else "medium"
        score = 0.0 if verdict == "block" else 0.45
        return StructuralFinding(
            finding_id=self._new_id("structural_finding"),
            check_name="schema_version_consistency",
            verdict=verdict,
            severity=severity,
            confidence=0.95,
            score=score,
            message=(
                f"Schema version mismatch for source '{source_id}': supplied='{schema_version}', "
                f"expected='{latest_version}'."
            ),
            flags=["schema_version_mismatch"],
            affected_records=[],
            metrics={"expected_schema_version": latest_version, "supplied_schema_version": schema_version},
            remediation_actions=self.schema_actions,
            details={"expected": latest, "schema_hash": schema_hash, "normalized_schema": normalized_schema},
            error={
                "error_type": QualityErrorType.SCHEMA_VERSION_MISMATCH.value,
                "message": f"Schema version mismatch for source '{source_id}'",
            },
        )

    def _analyze_records(self, *,
        records: Sequence[Mapping[str, Any]],
        normalized_schema: Mapping[str, Any],
        dataset_id: str,
        source_id: str,
    ) -> Dict[str, Any]:
        fields = dict(normalized_schema.get("fields", {}))
        total_records = len(records)
        field_count = max(len(fields), 1)

        analysis: Dict[str, Any] = {
            "reviewed_record_count": total_records,
            "required_missing_records": set(),
            "required_missing_by_field": {},
            "type_violation_records": set(),
            "type_violation_by_field": {},
            "range_violation_records": set(),
            "range_violation_by_field": {},
            "domain_violation_records": set(),
            "domain_violation_by_field": {},
            "regex_violation_records": set(),
            "regex_violation_by_field": {},
            "unknown_field_records": set(),
            "unknown_field_names": {},
            "coercion_records": set(),
            "coercion_by_field": {},
            "record_errors": [],
            "normalized_examples": [],
            "field_count": field_count,
        }

        for index, raw_record in enumerate(records):
            record = dict(raw_record)
            record_id = self._resolve_record_id(record, index)
            normalized_record = dict(record)

            unknown_fields = [name for name in record.keys() if name not in fields]
            if unknown_fields:
                analysis["unknown_field_records"].add(record_id)
                analysis["unknown_field_names"][record_id] = unknown_fields

            for field_name, spec in fields.items():
                exists = field_name in record
                value = record.get(field_name)
                is_empty_string = isinstance(value, str) and value.strip() == ""

                if spec["required"]:
                    if not exists or value is None or (is_empty_string and not self.allow_empty_strings):
                        analysis["required_missing_records"].add(record_id)
                        analysis["required_missing_by_field"].setdefault(field_name, []).append(record_id)
                        continue

                if not exists or value is None:
                    continue
                if is_empty_string and not self.allow_empty_strings and spec["required"]:
                    analysis["required_missing_records"].add(record_id)
                    analysis["required_missing_by_field"].setdefault(field_name, []).append(record_id)
                    continue
                if is_empty_string and self.allow_empty_strings:
                    continue

                coerced_value = value
                was_coerced = False
                if spec.get("coerce", False):
                    coerced_value, was_coerced = self._attempt_coercion(value, spec["type"])
                    if was_coerced:
                        analysis["coercion_records"].add(record_id)
                        analysis["coercion_by_field"].setdefault(field_name, []).append(record_id)
                        if self.apply_coerced_values:
                            normalized_record[field_name] = coerced_value

                if not self._matches_type(coerced_value, spec["type"]):
                    analysis["type_violation_records"].add(record_id)
                    analysis["type_violation_by_field"].setdefault(field_name, []).append(record_id)
                    analysis["record_errors"].append(
                        {
                            "record_id": record_id,
                            "field": field_name,
                            "issue": "type_violation",
                            "expected_type": spec["type"],
                            "value_preview": self._preview_value(value),
                        }
                    )
                    continue

                range_result = self._evaluate_range_constraints(coerced_value, spec)
                if range_result is not None:
                    analysis["range_violation_records"].add(record_id)
                    analysis["range_violation_by_field"].setdefault(field_name, []).append(record_id)
                    analysis["record_errors"].append(
                        {
                            "record_id": record_id,
                            "field": field_name,
                            "issue": "range_violation",
                            "details": range_result,
                        }
                    )

                domain_result = self._evaluate_domain_constraints(coerced_value, spec)
                if domain_result is not None:
                    if domain_result["issue"] == "regex_violation":
                        analysis["regex_violation_records"].add(record_id)
                        analysis["regex_violation_by_field"].setdefault(field_name, []).append(record_id)
                    else:
                        analysis["domain_violation_records"].add(record_id)
                        analysis["domain_violation_by_field"].setdefault(field_name, []).append(record_id)
                    analysis["record_errors"].append(
                        {
                            "record_id": record_id,
                            "field": field_name,
                            "issue": domain_result["issue"],
                            "details": domain_result,
                        }
                    )

            if len(analysis["normalized_examples"]) < 5:
                analysis["normalized_examples"].append({"record_id": record_id, "record": normalized_record})

        analysis["required_missing_rate"] = len(analysis["required_missing_records"]) / total_records if total_records else 0.0
        analysis["type_violation_rate"] = len(analysis["type_violation_records"]) / total_records if total_records else 0.0
        analysis["unknown_field_rate"] = len(analysis["unknown_field_records"]) / total_records if total_records else 0.0
        analysis["coercion_rate"] = len(analysis["coercion_records"]) / total_records if total_records else 0.0
        analysis["range_violation_rate"] = len(analysis["range_violation_records"]) / total_records if total_records else 0.0
        analysis["domain_violation_rate"] = len(analysis["domain_violation_records"]) / total_records if total_records else 0.0
        analysis["regex_violation_rate"] = len(analysis["regex_violation_records"]) / total_records if total_records else 0.0
        analysis["max_missing_field_rate"] = self._max_field_rate(analysis["required_missing_by_field"], total_records)
        analysis["max_type_field_rate"] = self._max_field_rate(analysis["type_violation_by_field"], total_records)
        analysis["max_range_field_rate"] = self._max_field_rate(analysis["range_violation_by_field"], total_records)
        analysis["max_domain_field_rate"] = self._max_field_rate(analysis["domain_violation_by_field"], total_records)
        analysis["max_regex_field_rate"] = self._max_field_rate(analysis["regex_violation_by_field"], total_records)
        return analysis

    # ------------------------------------------------------------------
    # Finding builders
    # ------------------------------------------------------------------
    def _build_schema_finding(self,
        analysis: Mapping[str, Any],
        normalized_schema: Mapping[str, Any],
        dataset_id: str,
    ) -> StructuralFinding:
        total_rate = max(float(analysis["type_violation_rate"]), float(analysis["unknown_field_rate"]))
        warn_rate = max(self.warn_unknown_field_rate, self.warn_missing_field_rate)
        block_rate = max(self.block_unknown_field_rate, self.block_missing_field_rate)
        verdict, severity = self._verdict_from_rate(total_rate, warn_rate, block_rate)
        message = (
            f"Schema integrity evaluation completed for dataset '{dataset_id}'. "
            f"type_violation_rate={analysis['type_violation_rate']:.4f}, "
            f"unknown_field_rate={analysis['unknown_field_rate']:.4f}."
        )
        flags: List[str] = []
        if analysis["type_violation_records"]:
            flags.append("type_violations_detected")
        if analysis["unknown_field_records"]:
            flags.append("unknown_fields_detected")
        if not flags:
            flags.append("schema_integrity_ok")

        return StructuralFinding(
            finding_id=self._new_id("structural_finding"),
            check_name="schema_integrity",
            verdict=verdict,
            severity=severity,
            confidence=self._confidence_from_rate(total_rate),
            score=self._score_from_rate(total_rate, warn_rate, block_rate),
            message=message,
            flags=flags,
            affected_records=self._sorted_union(
                analysis["type_violation_records"],
                analysis["unknown_field_records"],
            ),
            metrics={
                "type_violation_rate": analysis["type_violation_rate"],
                "unknown_field_rate": analysis["unknown_field_rate"],
                "max_type_field_rate": analysis["max_type_field_rate"],
                "schema_field_count": len(normalized_schema.get("fields", {})),
            },
            remediation_actions=self.schema_actions,
            details={
                "type_violation_by_field": analysis["type_violation_by_field"],
                "unknown_field_names": analysis["unknown_field_names"],
                "record_errors": [error for error in analysis["record_errors"] if error["issue"] == "type_violation"],
            },
        )

    def _build_required_finding(self,
        analysis: Mapping[str, Any],
        normalized_schema: Mapping[str, Any],
        dataset_id: str,
    ) -> StructuralFinding:
        total_rate = max(float(analysis["required_missing_rate"]), float(analysis["max_missing_field_rate"]))
        verdict, severity = self._verdict_from_rate(
            total_rate,
            self.warn_missing_record_rate,
            self.block_missing_record_rate,
        )
        message = (
            f"Required-field completeness evaluated for dataset '{dataset_id}'. "
            f"missing_record_rate={analysis['required_missing_rate']:.4f}, "
            f"max_missing_field_rate={analysis['max_missing_field_rate']:.4f}."
        )
        flags = ["required_fields_complete"]
        if analysis["required_missing_records"]:
            flags = ["required_fields_missing"]

        return StructuralFinding(
            finding_id=self._new_id("structural_finding"),
            check_name="required_completeness",
            verdict=verdict,
            severity=severity,
            confidence=self._confidence_from_rate(total_rate),
            score=self._score_from_rate(total_rate, self.warn_missing_record_rate, self.block_missing_record_rate),
            message=message,
            flags=flags,
            affected_records=self._sorted_list(analysis["required_missing_records"]),
            metrics={
                "missing_record_rate": analysis["required_missing_rate"],
                "max_missing_field_rate": analysis["max_missing_field_rate"],
                "required_field_count": len(normalized_schema.get("required_fields", [])),
            },
            remediation_actions=self.required_actions,
            details={"required_missing_by_field": analysis["required_missing_by_field"]},
            error=(
                {
                    "error_type": QualityErrorType.REQUIRED_FIELD_MISSING.value,
                    "message": f"Required-field completeness violations detected in dataset '{dataset_id}'.",
                }
                if analysis["required_missing_records"]
                else None
            ),
        )

    def _build_range_finding(self,
        analysis: Mapping[str, Any],
        normalized_schema: Mapping[str, Any],
        dataset_id: str,
    ) -> StructuralFinding:
        total_rate = float(analysis["range_violation_rate"])
        verdict, severity = self._verdict_from_rate(
            total_rate,
            self.warn_range_violation_rate,
            self.block_range_violation_rate,
        )
        message = (
            f"Range constraints evaluated for dataset '{dataset_id}'. "
            f"range_violation_rate={analysis['range_violation_rate']:.4f}."
        )
        flags = ["range_constraints_ok"]
        if analysis["range_violation_records"]:
            flags = ["range_violations_detected"]
        return StructuralFinding(
            finding_id=self._new_id("structural_finding"),
            check_name="range_constraints",
            verdict=verdict,
            severity=severity,
            confidence=self._confidence_from_rate(total_rate),
            score=self._score_from_rate(total_rate, self.warn_range_violation_rate, self.block_range_violation_rate),
            message=message,
            flags=flags,
            affected_records=self._sorted_list(analysis["range_violation_records"]),
            metrics={
                "range_violation_rate": analysis["range_violation_rate"],
                "max_range_field_rate": analysis["max_range_field_rate"],
            },
            remediation_actions=self.range_actions,
            details={
                "range_violation_by_field": analysis["range_violation_by_field"],
                "record_errors": [error for error in analysis["record_errors"] if error["issue"] == "range_violation"],
            },
        )

    def _build_domain_finding(self,
        analysis: Mapping[str, Any],
        normalized_schema: Mapping[str, Any],
        dataset_id: str,
    ) -> StructuralFinding:
        total_rate = max(float(analysis["domain_violation_rate"]), float(analysis["regex_violation_rate"]))
        warn_rate = max(self.warn_domain_violation_rate, self.warn_regex_violation_rate)
        block_rate = max(self.block_domain_violation_rate, self.block_regex_violation_rate)
        verdict, severity = self._verdict_from_rate(total_rate, warn_rate, block_rate)
        message = (
            f"Domain and pattern constraints evaluated for dataset '{dataset_id}'. "
            f"domain_violation_rate={analysis['domain_violation_rate']:.4f}, "
            f"regex_violation_rate={analysis['regex_violation_rate']:.4f}."
        )
        flags = ["domain_constraints_ok"]
        if analysis["domain_violation_records"] or analysis["regex_violation_records"]:
            flags = ["domain_or_pattern_violations_detected"]
        return StructuralFinding(
            finding_id=self._new_id("structural_finding"),
            check_name="domain_constraints",
            verdict=verdict,
            severity=severity,
            confidence=self._confidence_from_rate(total_rate),
            score=self._score_from_rate(total_rate, warn_rate, block_rate),
            message=message,
            flags=flags,
            affected_records=self._sorted_union(
                analysis["domain_violation_records"],
                analysis["regex_violation_records"],
            ),
            metrics={
                "domain_violation_rate": analysis["domain_violation_rate"],
                "regex_violation_rate": analysis["regex_violation_rate"],
                "max_domain_field_rate": analysis["max_domain_field_rate"],
                "max_regex_field_rate": analysis["max_regex_field_rate"],
            },
            remediation_actions=self.domain_actions,
            details={
                "domain_violation_by_field": analysis["domain_violation_by_field"],
                "regex_violation_by_field": analysis["regex_violation_by_field"],
                "record_errors": [
                    error
                    for error in analysis["record_errors"]
                    if error["issue"] in {"domain_violation", "regex_violation"}
                ],
            },
        )

    def _build_coercion_finding(self, analysis: Mapping[str, Any], dataset_id: str) -> StructuralFinding:
        total_rate = float(analysis["coercion_rate"])
        verdict, severity = self._verdict_from_rate(total_rate, self.warn_coercion_rate, self.block_coercion_rate)
        message = (
            f"Type coercion pressure evaluated for dataset '{dataset_id}'. "
            f"coercion_rate={analysis['coercion_rate']:.4f}."
        )
        flags = ["coercion_not_required"]
        if analysis["coercion_records"]:
            flags = ["type_coercion_applied"]
        return StructuralFinding(
            finding_id=self._new_id("structural_finding"),
            check_name="coercion_health",
            verdict=verdict,
            severity=severity,
            confidence=self._confidence_from_rate(total_rate),
            score=self._score_from_rate(total_rate, self.warn_coercion_rate, self.block_coercion_rate),
            message=message,
            flags=flags,
            affected_records=self._sorted_list(analysis["coercion_records"]),
            metrics={"coercion_rate": analysis["coercion_rate"]},
            remediation_actions=self.coercion_actions,
            details={"coercion_by_field": analysis["coercion_by_field"]},
            error=(
                {
                    "error_type": QualityErrorType.TYPE_COERCION_FAILED.value,
                    "message": f"Excessive type coercion pressure detected in dataset '{dataset_id}'.",
                }
                if verdict in {"warn", "block"}
                else None
            ),
        )

    def _aggregate_findings(self, findings: Sequence[StructuralFinding]) -> Dict[str, Any]:
        weighted_score = 0.0
        flags: List[str] = []
        remediation_actions: List[str] = []
        quarantine_count = 0

        finding_map = {
            "schema": self._finding_by_name(findings, "schema_integrity"),
            "required": self._finding_by_name(findings, "required_completeness"),
            "range": self._finding_by_name(findings, "range_constraints"),
            "domain": self._finding_by_name(findings, "domain_constraints"),
            "coercion": self._finding_by_name(findings, "coercion_health"),
        }

        for weight_key, weight in self.weights.items():
            finding = finding_map.get(weight_key)
            if finding is not None:
                weighted_score += float(weight) * float(finding.score)

        for finding in findings:
            flags.extend(finding.flags)
            remediation_actions.extend(finding.remediation_actions)
            if finding.verdict == "block":
                quarantine_count = max(quarantine_count, len(finding.affected_records))
            elif finding.verdict == "warn":
                quarantine_count = max(quarantine_count, math.ceil(len(finding.affected_records) * 0.5))

        final_verdict = self._verdict_from_score(weighted_score)
        if any(finding.verdict == "block" for finding in findings):
            final_verdict = "block"
        elif final_verdict == "pass" and any(finding.verdict == "warn" for finding in findings):
            final_verdict = "warn"

        shift_metrics = {
            "schema_violation_rate": self._metric_from_findings(findings, "schema_integrity", "type_violation_rate"),
            "missing_record_rate": self._metric_from_findings(findings, "required_completeness", "missing_record_rate"),
            "range_violation_rate": self._metric_from_findings(findings, "range_constraints", "range_violation_rate"),
            "domain_violation_rate": self._metric_from_findings(findings, "domain_constraints", "domain_violation_rate"),
            "regex_violation_rate": self._metric_from_findings(findings, "domain_constraints", "regex_violation_rate"),
            "coercion_rate": self._metric_from_findings(findings, "coercion_health", "coercion_rate"),
        }

        return {
            "score": round(weighted_score, 6),
            "verdict": final_verdict,
            "flags": self._dedupe_keep_order(flags),
            "remediation_actions": self._dedupe_keep_order(remediation_actions),
            "quarantine_count": int(quarantine_count),
            "shift_metrics": shift_metrics,
        }

    def _record_to_memory(self, *, findings: Sequence[StructuralFinding],
        dataset_id: str, source_id: str, batch_id: str,
        schema_version: Optional[str],
        schema_hash: Optional[str],
        normalized_schema: Mapping[str, Any],
        aggregate: Mapping[str, Any],
        window: Optional[str],
        context: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self.auto_record_to_memory:
            return None

        memory_snapshot: Optional[Dict[str, Any]] = None

        if (
            self.memory_register_schema_version
            and hasattr(self.memory, "register_schema_version")
            and schema_version is not None
            and schema_hash is not None
        ):
            try:
                self.memory.register_schema_version(
                    source_id=source_id,
                    schema_version=schema_version,
                    schema_hash=schema_hash,
                    required_fields=normalized_schema.get("required_fields", []),
                    field_types={
                        field_name: spec.get("type", "any")
                        for field_name, spec in dict(normalized_schema.get("fields", {})).items()
                    },
                    is_active=True,
                    context={
                        "dataset_id": dataset_id,
                        "checker": "structural",
                        "field_count": len(normalized_schema.get("fields", {})),
                    },
                )
            except Exception as exc:
                raise QualityMemoryError(
                    "register_schema_version",
                    str(exc),
                    context={"dataset_id": dataset_id, "source_id": source_id, "schema_version": schema_version},
                ) from exc

        if self.memory_record_quality_snapshot and hasattr(self.memory, "record_quality_snapshot"):
            try:
                memory_snapshot = self.memory.record_quality_snapshot(
                    source_id=source_id,
                    batch_id=batch_id,
                    batch_score=float(aggregate["score"]),
                    verdict=str(aggregate["verdict"]),
                    flags=list(aggregate["flags"]),
                    quarantine_count=int(aggregate["quarantine_count"]),
                    shift_metrics=dict(aggregate["shift_metrics"]),
                    remediation_actions=list(aggregate["remediation_actions"]),
                    source_reliability=1.0 if aggregate["verdict"] == "pass" else (0.75 if aggregate["verdict"] == "warn" else 0.45),
                    schema_version=schema_version,
                    window=window,
                    checker_findings=[finding.to_conflict_input() for finding in findings],
                    context={
                        "dataset_id": dataset_id,
                        "checker": "structural",
                        "schema_hash": schema_hash,
                        **dict(context),
                    },
                )
            except Exception as exc:
                raise QualityMemoryError(
                    "record_quality_snapshot",
                    str(exc),
                    context={"dataset_id": dataset_id, "source_id": source_id, "batch_id": batch_id},
                ) from exc

        return memory_snapshot

    # ------------------------------------------------------------------
    # Constraint evaluation helpers
    # ------------------------------------------------------------------
    def _attempt_coercion(self, value: Any, expected_type: str) -> Tuple[Any, bool]:
        if value is None:
            return value, False
        if expected_type in {"any", "object"}:
            return value, False

        if expected_type == "string":
            if isinstance(value, str):
                return value, False
            return str(value), True

        if expected_type == "integer":
            if isinstance(value, bool):
                return value, False
            if isinstance(value, int):
                return value, False
            if isinstance(value, float) and float(value).is_integer():
                return int(value), True
            if isinstance(value, str):
                text = value.strip()
                if re.fullmatch(r"[+-]?\d+", text):
                    return int(text), True
            return value, False

        if expected_type == "number":
            if isinstance(value, bool):
                return value, False
            if isinstance(value, (int, float)):
                return float(value) if isinstance(value, int) else value, False
            if isinstance(value, str):
                text = value.strip()
                try:
                    return float(text), True
                except ValueError:
                    return value, False
            return value, False

        if expected_type == "boolean":
            if isinstance(value, bool):
                return value, False
            if isinstance(value, str):
                text = value.strip().lower()
                if text in self.truthy_values:
                    return True, True
                if text in self.falsy_values:
                    return False, True
            if isinstance(value, (int, float)) and value in {0, 1}:
                return bool(value), True
            return value, False

        if expected_type == "datetime":
            parsed = self._parse_datetime(value)
            if parsed is None:
                return value, False
            return parsed, not isinstance(value, datetime)

        if expected_type == "array":
            if isinstance(value, list):
                return value, False
            if isinstance(value, tuple):
                return list(value), True
            if isinstance(value, str):
                text = value.strip()
                if text.startswith("[") and text.endswith("]"):
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, list):
                            return parsed, True
                    except Exception:
                        return value, False
            return value, False

        if expected_type == "object":
            if isinstance(value, dict):
                return value, False
            if isinstance(value, str):
                text = value.strip()
                if text.startswith("{") and text.endswith("}"):
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, dict):
                            return parsed, True
                    except Exception:
                        return value, False
            return value, False

        return value, False

    def _matches_type(self, value: Any, expected_type: str) -> bool:
        if expected_type == "any":
            return True
        if expected_type == "string":
            return isinstance(value, str)
        if expected_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected_type == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if expected_type == "boolean":
            return isinstance(value, bool)
        if expected_type == "array":
            return isinstance(value, list)
        if expected_type == "object":
            return isinstance(value, dict)
        if expected_type == "datetime":
            return isinstance(value, datetime)
        return True

    def _evaluate_range_constraints(self, value: Any, spec: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        expected_type = str(spec.get("type", "any"))
        minimum = spec.get("min")
        maximum = spec.get("max")
        min_length = spec.get("min_length")
        max_length = spec.get("max_length")
        min_items = spec.get("min_items")
        max_items = spec.get("max_items")

        if expected_type in {"integer", "number"} and isinstance(value, (int, float)) and not isinstance(value, bool):
            if minimum is not None and float(value) < float(minimum):
                return {"issue": "range_violation", "kind": "min", "min": minimum, "observed": value}
            if maximum is not None and float(value) > float(maximum):
                return {"issue": "range_violation", "kind": "max", "max": maximum, "observed": value}

        if expected_type == "datetime" and isinstance(value, datetime):
            min_dt = self._parse_datetime(minimum) if minimum is not None else None
            max_dt = self._parse_datetime(maximum) if maximum is not None else None
            normalized_value = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
            if min_dt is not None and normalized_value < min_dt:
                return {"issue": "range_violation", "kind": "min_datetime", "min": min_dt.isoformat(), "observed": normalized_value.isoformat()}
            if max_dt is not None and normalized_value > max_dt:
                return {"issue": "range_violation", "kind": "max_datetime", "max": max_dt.isoformat(), "observed": normalized_value.isoformat()}

        if isinstance(value, str):
            length = len(value)
            if min_length is not None and length < int(min_length):
                return {"issue": "range_violation", "kind": "min_length", "min_length": min_length, "observed_length": length}
            if max_length is not None and length > int(max_length):
                return {"issue": "range_violation", "kind": "max_length", "max_length": max_length, "observed_length": length}

        if isinstance(value, list):
            length = len(value)
            if min_items is not None and length < int(min_items):
                return {"issue": "range_violation", "kind": "min_items", "min_items": min_items, "observed_length": length}
            if max_items is not None and length > int(max_items):
                return {"issue": "range_violation", "kind": "max_items", "max_items": max_items, "observed_length": length}

        return None

    def _evaluate_domain_constraints(self, value: Any, spec: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        enum_values = list(spec.get("enum", []))
        if enum_values:
            normalized_value = str(value)
            if normalized_value not in enum_values:
                return {
                    "issue": "domain_violation",
                    "allowed_values": enum_values,
                    "observed": normalized_value,
                }

        pattern = spec.get("pattern")
        if pattern and isinstance(value, str):
            if re.fullmatch(str(pattern), value) is None:
                return {
                    "issue": "regex_violation",
                    "pattern": str(pattern),
                    "observed": value,
                }

        return None

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _validate_runtime_configuration(self) -> None:
        if not (0.0 <= self.warn_threshold <= self.pass_threshold <= 1.0):
            raise DataQualityError(
                message="Structural scoring thresholds must satisfy 0 <= warn_threshold <= pass_threshold <= 1",
                error_type=QualityErrorType.POLICY_THRESHOLD_INVALID,
                severity=QualitySeverity.HIGH,
                retryable=False,
                context={
                    "warn_threshold": self.warn_threshold,
                    "pass_threshold": self.pass_threshold,
                },
                remediation="Correct the structural_quality.scoring thresholds in quality_config.yaml.",
            )

        threshold_pairs = [
            (self.warn_missing_record_rate, self.block_missing_record_rate, "completeness"),
            (self.warn_coercion_rate, self.block_coercion_rate, "coercion"),
            (self.warn_range_violation_rate, self.block_range_violation_rate, "range"),
            (self.warn_domain_violation_rate, self.block_domain_violation_rate, "domain"),
            (self.warn_regex_violation_rate, self.block_regex_violation_rate, "regex"),
            (self.warn_unknown_field_rate, self.block_unknown_field_rate, "unknown_fields"),
        ]
        for warn_value, block_value, label in threshold_pairs:
            if warn_value < 0 or block_value < 0 or warn_value > block_value:
                raise DataQualityError(
                    message=f"Invalid structural_quality threshold pair for {label}",
                    error_type=QualityErrorType.POLICY_THRESHOLD_INVALID,
                    severity=QualitySeverity.HIGH,
                    retryable=False,
                    context={"label": label, "warn": warn_value, "block": block_value},
                    remediation="Ensure every warn threshold is non-negative and less than or equal to its block threshold.",
                )

    def _normalize_records(self, records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(records, Sequence):
            raise DataQualityError(
                message="records must be a sequence of mapping-like objects",
                error_type=QualityErrorType.SCHEMA_VALIDATION_FAILED,
                severity=QualitySeverity.HIGH,
                retryable=False,
                context={"received_type": type(records).__name__},
                remediation="Pass a list or tuple of dictionaries to StructuralQuality.evaluate_batch.",
            )
        normalized: List[Dict[str, Any]] = []
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise DataQualityError(
                    message=f"Record at index {index} must be mapping-like",
                    error_type=QualityErrorType.SCHEMA_VALIDATION_FAILED,
                    severity=QualitySeverity.HIGH,
                    retryable=False,
                    context={"index": index, "received_type": type(record).__name__},
                    remediation="Ensure every batch element is a dictionary-like record.",
                )
            normalized.append(dict(record))
        return normalized

    def _canonical_type(self, raw_type: Any) -> str:
        value = str(raw_type).strip().lower() if raw_type is not None else "any"
        aliases = {
            "str": "string",
            "string": "string",
            "text": "string",
            "int": "integer",
            "integer": "integer",
            "long": "integer",
            "float": "number",
            "double": "number",
            "decimal": "number",
            "number": "number",
            "bool": "boolean",
            "boolean": "boolean",
            "list": "array",
            "array": "array",
            "dict": "object",
            "map": "object",
            "object": "object",
            "datetime": "datetime",
            "timestamp": "datetime",
            "date": "datetime",
            "any": "any",
            "unknown": "any",
        }
        return aliases.get(value, "any")

    def _infer_value_type(self, value: Any) -> str:
        if value is None:
            return "any"
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "number"
        if isinstance(value, datetime):
            return "datetime"
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        return "string"

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
                return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                pass
            for fmt in self.datetime_formats:
                try:
                    parsed = datetime.strptime(text, fmt)
                    return parsed.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
        return None

    def _schema_hash(self, normalized_schema: Mapping[str, Any]) -> str:
        encoded = json.dumps(dict(normalized_schema), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}_{int(time.time() * 1000)}_{hashlib.sha1(str(time.time_ns()).encode('utf-8')).hexdigest()[:8]}"

    def _resolve_record_id(self, record: Mapping[str, Any], index: int) -> str:
        for candidate in self.record_id_candidates:
            value = record.get(candidate)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return f"record_{index}"

    def _finding_by_name(self, findings: Sequence[StructuralFinding], name: str) -> Optional[StructuralFinding]:
        for finding in findings:
            if finding.check_name == name:
                return finding
        return None

    def _metric_from_findings(self, findings: Sequence[StructuralFinding], name: str, metric: str) -> float:
        finding = self._finding_by_name(findings, name)
        if finding is None:
            return 0.0
        return float(finding.metrics.get(metric, 0.0))

    def _verdict_from_rate(self, rate: float, warn_threshold: float, block_threshold: float) -> Tuple[str, str]:
        if rate >= block_threshold and block_threshold > 0:
            return "block", "high"
        if rate >= warn_threshold and warn_threshold > 0:
            return "warn", "medium"
        return "pass", "low"

    def _verdict_from_score(self, score: float) -> str:
        if score >= self.pass_threshold:
            return "pass"
        if score >= self.warn_threshold:
            return "warn"
        return "block"

    def _score_from_rate(self, rate: float, warn_threshold: float, block_threshold: float) -> float:
        if rate <= 0:
            return 1.0
        if warn_threshold <= 0:
            return 0.0 if rate > 0 else 1.0
        if rate <= warn_threshold:
            return max(0.75, 1.0 - 0.25 * (rate / warn_threshold))
        if block_threshold <= warn_threshold:
            return 0.0
        if rate >= block_threshold:
            return 0.0
        fraction = (rate - warn_threshold) / (block_threshold - warn_threshold)
        return max(0.0, 0.75 * (1.0 - fraction))

    def _confidence_from_rate(self, rate: float) -> float:
        return max(0.5, min(0.99, 0.99 - min(rate, 1.0) * 0.49))

    def _max_field_rate(self, grouped_ids: Mapping[str, Sequence[str]], total_records: int) -> float:
        if total_records <= 0 or not grouped_ids:
            return 0.0
        return max((len(set(values)) / total_records for values in grouped_ids.values()), default=0.0)

    def _normalized_weights(self, weights: Mapping[str, Any], expected_keys: Sequence[str]) -> Dict[str, float]:
        normalized: Dict[str, float] = {}
        total = 0.0
        for key in expected_keys:
            value = float(dict(weights).get(key, 0.0))
            if value < 0:
                raise DataQualityError(
                    message=f"Weight '{key}' must be non-negative",
                    error_type=QualityErrorType.POLICY_THRESHOLD_INVALID,
                    severity=QualitySeverity.HIGH,
                    retryable=False,
                    context={"key": key, "value": value},
                    remediation="Use non-negative structural scoring weights and ensure their sum is positive.",
                )
            normalized[key] = value
            total += value
        if total <= 0:
            raise DataQualityError(
                message="Structural scoring weights must sum to a positive value",
                error_type=QualityErrorType.POLICY_THRESHOLD_INVALID,
                severity=QualitySeverity.HIGH,
                retryable=False,
                context={"weights": dict(weights)},
                remediation="Configure positive scoring weights for structural_quality.scoring.weights.",
            )
        for key in normalized:
            normalized[key] = normalized[key] / total
        return normalized

    def _positive_int(self, value: Any, field_name: str) -> int:
        try:
            resolved = int(value)
        except Exception as exc:
            raise DataQualityError(
                message=f"{field_name} must be a positive integer",
                error_type=QualityErrorType.POLICY_THRESHOLD_INVALID,
                severity=QualitySeverity.HIGH,
                retryable=False,
                context={"field_name": field_name, "value": value},
                remediation="Use a positive integer in quality_config.yaml.",
            ) from exc
        if resolved <= 0:
            raise DataQualityError(
                message=f"{field_name} must be a positive integer",
                error_type=QualityErrorType.POLICY_THRESHOLD_INVALID,
                severity=QualitySeverity.HIGH,
                retryable=False,
                context={"field_name": field_name, "value": value},
                remediation="Use a positive integer in quality_config.yaml.",
            )
        return resolved

    def _nonempty_text(self, value: Any, field_name: str) -> str:
        text = str(value).strip() if value is not None else ""
        if not text:
            raise DataQualityError(
                message=f"{field_name} must not be empty",
                error_type=QualityErrorType.SCHEMA_VALIDATION_FAILED,
                severity=QualitySeverity.HIGH,
                retryable=False,
                context={"field_name": field_name, "value": value},
                remediation="Provide a non-empty identifier for structural evaluation.",
            )
        return text

    def _string_list(self, values: Optional[Iterable[Any]]) -> List[str]:
        if values is None:
            return []
        return [str(value) for value in values]

    def _dedupe_keep_order(self, values: Iterable[Any]) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for value in values:
            text = str(value)
            if text in seen:
                continue
            seen.add(text)
            ordered.append(text)
        return ordered

    def _sorted_list(self, values: Iterable[Any]) -> List[str]:
        return sorted(str(v) for v in values)

    def _sorted_union(self, left: Iterable[Any], right: Iterable[Any]) -> List[str]:
        return sorted({str(v) for v in left}.union({str(v) for v in right}))

    def _normalized_mapping(self, value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if value is None:
            return {}
        return json.loads(json.dumps(dict(value), default=str))

    def _preview_value(self, value: Any, limit: int = 120) -> str:
        text = repr(value)
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."


if __name__ == "__main__":
    print("\n=== Running Structural Quality ===\n")
    printer.status("TEST", "Structural Quality initialized", "info")

    structural = StructuralQuality()
    printer.status(
        "CONFIG",
        f"Loaded structural_quality config from {structural.config.get('__config_path__', 'unknown')}",
        "success",
    )

    schema = {
        "schema_version": "v2.1.0",
        "required_fields": ["id", "source_id", "event_time", "amount", "status", "email"],
        "fields": {
            "id": {"type": "str", "required": True},
            "source_id": {"type": "str", "required": True},
            "event_time": {"type": "datetime", "required": True},
            "amount": {"type": "float", "required": True, "min": 0.0, "max": 1000.0},
            "status": {"type": "str", "required": True, "enum": ["new", "processing", "resolved"]},
            "priority": {"type": "int", "required": False, "min": 1, "max": 5, "coerce": True},
            "email": {
                "type": "str",
                "required": True,
                "pattern": r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$",
            },
            "metadata": {"type": "dict", "required": False, "nullable": True},
        },
    }

    records = [
        {
            "id": "r1",
            "source_id": "source_alpha",
            "event_time": "2026-04-08T10:30:00Z",
            "amount": 125.5,
            "status": "new",
            "priority": "3",
            "email": "alpha@example.com",
            "metadata": {"region": "eu"},
        },
        {
            "id": "r2",
            "source_id": "source_alpha",
            "event_time": "2026-04-08T10:40:00Z",
            "amount": -10.0,
            "status": "processing",
            "priority": 2,
            "email": "beta@example.com",
        },
        {
            "id": "r3",
            "source_id": "source_alpha",
            "event_time": "2026-04-08 10:45:00",
            "amount": "87.4",
            "status": "resolved",
            "priority": "2",
            "email": "invalid-email-format",
        },
        {
            "id": "r4",
            "source_id": "source_alpha",
            "event_time": "2026-04-08T11:00:00Z",
            "amount": 220.0,
            "priority": 5,
            "email": "delta@example.com",
            "unexpected_field": "surprise",
        },
        {
            "id": "r5",
            "source_id": "source_alpha",
            "event_time": "not-a-timestamp",
            "amount": "bad-number",
            "status": "archived",
            "priority": 7,
            "email": "epsilon@example.com",
        },
    ]

    assessment = structural.evaluate_batch(
        records,
        dataset_id="orders_dataset",
        source_id="source_alpha",
        batch_id="batch_2026_04_08_structural_001",
        schema=schema,
        context={"pipeline": "reader->quality_gate->knowledge_ingestion"},
    )

    assert assessment["record_count"] == 5, "Structural assessment should count all records"
    assert assessment["reviewed_record_count"] == 5, "Structural assessment should review all records"
    assert assessment["findings"], "Structural assessment should emit detailed findings"
    assert assessment["verdict"] in {"warn", "block"}, "Test dataset should trigger a non-pass verdict"
    assert any(flag in assessment["flags"] for flag in ["required_fields_missing", "range_violations_detected", "domain_or_pattern_violations_detected", "type_violations_detected"]), (
        "Structural flags should reflect the injected violations"
    )

    printer.pretty("ASSESSMENT", assessment, "success")

    print("\n=== Test ran successfully ===\n")
