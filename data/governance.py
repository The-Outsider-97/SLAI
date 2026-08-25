from __future__ import annotations

import hashlib
import json
import re

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Sequence

from .utils.config_loader import get_config_section, load_global_config
from .utils.data_error import *
from .utils.data_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Data Governance")
printer = PrettyPrinter()


# =============================================================================
# Data contracts
# =============================================================================

@dataclass(frozen=True)
class DatasetField:
    """Single governed field definition for a dataset schema.

    The class intentionally stays lightweight and explicit so existing call
    sites can keep constructing fields directly in Python while governance
    remains responsible for validating the definition before records are used.
    """

    name: str
    expected_type: type | tuple[type, ...]
    nullable: bool = False
    max_items: int | None = None
    min_items: int | None = None
    min_value: int | float | None = None
    max_value: int | float | None = None
    allowed_values: Sequence[Any] | None = None
    regex: str | None = None
    description: str | None = None

    def type_names(self) -> tuple[str, ...]:
        types = self.expected_type if isinstance(self.expected_type, tuple) else (self.expected_type,)
        return tuple(t.__name__ for t in types)


@dataclass(frozen=True)
class DatasetSchema:
    """Governed schema for one modality.

    One schema maps to one modality (`vision`, `text`, `audio`, etc.). Multiple
    schemas with the same modality are rejected by `DatasetValidator` because
    ambiguous modality contracts create non-deterministic validation behaviour.
    """

    name: str
    version: str
    modality: str
    fields: tuple[DatasetField, ...]
    description: str | None = None

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)


@dataclass
class DatasetLineage:
    dataset_name: str
    dataset_version: str
    source_uri: str
    source_commit: str
    transform_id: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    producer: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        required = {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "source_uri": self.source_uri,
            "source_commit": self.source_commit,
            "transform_id": self.transform_id,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise DataVersioningError(
                "Dataset lineage is missing required fields",
                context={"missing": missing},
            )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "source_uri": self.source_uri,
            "source_commit": self.source_commit,
            "transform_id": self.transform_id,
            "created_at": self.created_at,
        }
        if self.producer is not None:
            payload["producer"] = self.producer
        if self.notes is not None:
            payload["notes"] = self.notes
        return payload


@dataclass(frozen=True)
class DatasetGovernanceReport:
    """Serializable outcome of a complete governance assessment."""

    quality: dict[str, Any]
    modality_stats: dict[str, Any]
    lineage_record: dict[str, Any] | None = None
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality": self.quality,
            "modality_stats": self.modality_stats,
            "lineage_record": self.lineage_record,
            "generated_at": self.generated_at,
        }


# =============================================================================
# Dataset validation and quality gates
# =============================================================================

class DatasetValidator:
    """Production validation layer for multimodal ingestion payloads.

    Responsibilities
    ----------------
    * Validate schema definitions at construction time.
    * Enforce modality alignment and expected modality sets.
    * Validate records against field contracts.
    * Run aggregate quality gates using canonical helper utilities.
    """

    def __init__(self, schemas: Sequence[DatasetSchema], max_records: int | None = None):
        dataset_cfg = get_config_section("dataset")
        validation_cfg = get_config_section("validation")

        self.max_records = _positive_int(
            max_records if max_records is not None else dataset_cfg.get("max_records", 1_000_000),
            name="dataset.max_records",
        )
        self.enforce_alignment = bool(dataset_cfg.get("enforce_alignment", True))
        self.allow_empty_payload = bool(dataset_cfg.get("allow_empty_payload", False))
        self.required_modalities = tuple(str(m) for m in dataset_cfg.get("required_modalities", ()) or ())

        self.strict_types = bool(validation_cfg.get("strict_types", True))
        self.coerce_types = bool(validation_cfg.get("coerce_types", False))
        self.fail_on_unknown_modality = bool(validation_cfg.get("fail_on_unknown_modality", True))
        self.allow_extra_fields = bool(validation_cfg.get("allow_extra_fields", True))
        self.default_max_items = int(validation_cfg.get("max_field_items_default", 100_000))

        self.schemas = self._build_schema_map(tuple(schemas))
        logger.info(
            {
                "event": "dataset_validator_initialized",
                "modalities": sorted(self.schemas.keys()),
                "max_records": self.max_records,
                "strict_types": self.strict_types,
                "coerce_types": self.coerce_types,
            }
        )

    def _build_schema_map(self, schemas: tuple[DatasetSchema, ...]) -> dict[str, DatasetSchema]:
        if not schemas:
            raise DataSchemaError("At least one DatasetSchema is required", context={})

        schema_map: dict[str, DatasetSchema] = {}
        for schema in schemas:
            self._validate_schema_definition(schema)
            if schema.modality in schema_map:
                raise DataSchemaError(
                    "Duplicate schema modality detected",
                    context={"modality": schema.modality, "schema": schema.name},
                )
            schema_map[schema.modality] = schema
        return schema_map

    def _validate_schema_definition(self, schema: DatasetSchema) -> None:
        if not isinstance(schema, DatasetSchema):
            raise DataSchemaError(
                "Expected DatasetSchema instance",
                context={"type": type(schema).__name__},
            )
        if not schema.name.strip() or not schema.version.strip() or not schema.modality.strip():
            raise DataSchemaError(
                "DatasetSchema name, version, and modality are required",
                context={"schema": repr(schema)},
            )
        if not schema.fields:
            raise DataSchemaError(
                "DatasetSchema must define at least one field",
                context={"schema": schema.name, "modality": schema.modality},
            )

        seen: set[str] = set()
        for field_def in schema.fields:
            if not isinstance(field_def, DatasetField):
                raise DataSchemaError(
                    "Schema field must be a DatasetField instance",
                    context={"schema": schema.name, "type": type(field_def).__name__},
                )
            if not field_def.name.strip():
                raise DataSchemaError("DatasetField name is required", context={"schema": schema.name})
            if field_def.name in seen:
                raise DataSchemaError(
                    "Duplicate field name in schema",
                    context={"schema": schema.name, "field": field_def.name},
                )
            seen.add(field_def.name)
            self._validate_field_definition(schema, field_def)

    def _validate_field_definition(self, schema: DatasetSchema, field_def: DatasetField) -> None:
        expected = field_def.expected_type if isinstance(field_def.expected_type, tuple) else (field_def.expected_type,)
        if not expected or any(not isinstance(t, type) for t in expected):
            raise DataSchemaError(
                "DatasetField.expected_type must be a type or tuple of types",
                context={"schema": schema.name, "field": field_def.name},
            )
        if field_def.min_items is not None and field_def.min_items < 0:
            raise DataSchemaError(
                "DatasetField.min_items cannot be negative",
                context={"schema": schema.name, "field": field_def.name, "min_items": field_def.min_items},
            )
        if field_def.max_items is not None and field_def.max_items < 1:
            raise DataSchemaError(
                "DatasetField.max_items must be positive",
                context={"schema": schema.name, "field": field_def.name, "max_items": field_def.max_items},
            )
        if (
            field_def.min_items is not None
            and field_def.max_items is not None
            and field_def.min_items > field_def.max_items
        ):
            raise DataSchemaError(
                "DatasetField.min_items cannot exceed max_items",
                context={"schema": schema.name, "field": field_def.name},
            )
        if (
            field_def.min_value is not None
            and field_def.max_value is not None
            and field_def.min_value > field_def.max_value
        ):
            raise DataSchemaError(
                "DatasetField.min_value cannot exceed max_value",
                context={"schema": schema.name, "field": field_def.name},
            )
        if field_def.regex is not None:
            try:
                re.compile(field_def.regex)
            except re.error as exc:
                raise DataSchemaError(
                    "DatasetField.regex is invalid",
                    context={"schema": schema.name, "field": field_def.name, "regex": field_def.regex},
                    cause=exc,
                ) from exc

    def validate_records(self, modality: str, records: Sequence[Mapping[str, Any]]) -> None:
        """Validate all records for one modality against the registered schema."""
        if modality not in self.schemas:
            if self.fail_on_unknown_modality:
                raise DataValidationError(
                    f"Unknown modality: {modality}",
                    context={"modality": modality, "known_modalities": sorted(self.schemas.keys())},
                )
            logger.warning({"event": "unknown_modality_skipped", "modality": modality})
            return

        self._validate_record_collection(modality, records)
        schema = self.schemas[modality]
        expected_fields = set(schema.field_names)

        for row_idx, row in enumerate(records):
            self._validate_record_shape(modality, row_idx, row)
            if not self.allow_extra_fields:
                extra_fields = set(row.keys()) - expected_fields
                if extra_fields:
                    raise DataValidationError(
                        f"{modality}[{row_idx}] contains fields not declared in schema",
                        context={"modality": modality, "row_idx": row_idx, "extra_fields": sorted(extra_fields)},
                    )
            for field_def in schema.fields:
                self._validate_field_value(modality, row_idx, row, field_def)

    def _validate_record_collection(self, modality: str, records: Sequence[Mapping[str, Any]]) -> None:
        if records is None:  # type: ignore[comparison-overlap]
            raise DataValidationError(f"{modality}: records cannot be None", context={"modality": modality})
        if len(records) == 0:
            raise DataValidationError(f"{modality}: no records provided", context={"modality": modality})
        if len(records) > self.max_records:
            raise DataValidationError(
                f"{modality}: record count {len(records)} exceeds max_records={self.max_records}",
                context={"modality": modality, "record_count": len(records), "max_records": self.max_records},
            )

    @staticmethod
    def _validate_record_shape(modality: str, row_idx: int, row: Mapping[str, Any]) -> None:
        if not isinstance(row, Mapping):
            raise DataValidationError(
                f"{modality}[{row_idx}] must be a mapping",
                context={"modality": modality, "row_idx": row_idx, "type": type(row).__name__},
            )

    def _validate_field_value(
        self,
        modality: str,
        row_idx: int,
        row: Mapping[str, Any],
        field_def: DatasetField,
    ) -> None:
        if field_def.name not in row:
            raise DataValidationError(
                f"{modality}[{row_idx}] missing required field '{field_def.name}'",
                context={"modality": modality, "row_idx": row_idx, "field": field_def.name},
            )

        value = row[field_def.name]
        if value is None:
            if field_def.nullable:
                return
            raise DataValidationError(
                f"{modality}[{row_idx}] field '{field_def.name}' cannot be null",
                context={"modality": modality, "row_idx": row_idx, "field": field_def.name},
            )

        self._validate_type(modality, row_idx, field_def, value)
        self._validate_item_count(modality, row_idx, field_def, value)
        self._validate_numeric_range(modality, row_idx, field_def, value)
        self._validate_allowed_values(modality, row_idx, field_def, value)
        self._validate_regex(modality, row_idx, field_def, value)

    def _validate_type(self, modality: str, row_idx: int, field_def: DatasetField, value: Any) -> None:
        if isinstance(value, field_def.expected_type):
            return

        expected_types = field_def.expected_type if isinstance(field_def.expected_type, tuple) else (field_def.expected_type,)
        can_coerce = self.coerce_types and not self.strict_types and len(expected_types) == 1
        if can_coerce:
            safe_cast(value, expected_types[0], field_name=field_def.name, modality=modality, row_idx=row_idx)
            return

        raise DataValidationError(
            f"{modality}[{row_idx}] field '{field_def.name}' expected {field_def.type_names()}, "
            f"got {type(value).__name__}",
            context={
                "modality": modality,
                "row_idx": row_idx,
                "field": field_def.name,
                "expected_type": field_def.type_names(),
                "actual_type": type(value).__name__,
            },
        )

    def _validate_item_count(self, modality: str, row_idx: int, field_def: DatasetField, value: Any) -> None:
        if not hasattr(value, "__len__") or isinstance(value, (str, bytes)):
            return

        actual_size = len(value)
        max_items = field_def.max_items if field_def.max_items is not None else self.default_max_items
        if field_def.min_items is not None and actual_size < field_def.min_items:
            raise DataValidationError(
                f"{modality}[{row_idx}] field '{field_def.name}' has fewer than min_items={field_def.min_items}",
                context={
                    "modality": modality,
                    "row_idx": row_idx,
                    "field": field_def.name,
                    "min_items": field_def.min_items,
                    "actual_size": actual_size,
                },
            )
        if max_items is not None and actual_size > max_items:
            raise DataValidationError(
                f"{modality}[{row_idx}] field '{field_def.name}' exceeds max_items={max_items}",
                context={
                    "modality": modality,
                    "row_idx": row_idx,
                    "field": field_def.name,
                    "max_items": max_items,
                    "actual_size": actual_size,
                },
            )

    @staticmethod
    def _validate_numeric_range(modality: str, row_idx: int, field_def: DatasetField, value: Any) -> None:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return
        if field_def.min_value is not None and value < field_def.min_value:
            raise DataValidationError(
                f"{modality}[{row_idx}] field '{field_def.name}' is below minimum {field_def.min_value}",
                context={
                    "modality": modality,
                    "row_idx": row_idx,
                    "field": field_def.name,
                    "value": value,
                    "min_value": field_def.min_value,
                },
            )
        if field_def.max_value is not None and value > field_def.max_value:
            raise DataValidationError(
                f"{modality}[{row_idx}] field '{field_def.name}' exceeds maximum {field_def.max_value}",
                context={
                    "modality": modality,
                    "row_idx": row_idx,
                    "field": field_def.name,
                    "value": value,
                    "max_value": field_def.max_value,
                },
            )

    @staticmethod
    def _validate_allowed_values(modality: str, row_idx: int, field_def: DatasetField, value: Any) -> None:
        if field_def.allowed_values is None:
            return
        if value not in field_def.allowed_values:
            raise DataValidationError(
                f"{modality}[{row_idx}] field '{field_def.name}' is outside allowed values",
                context={
                    "modality": modality,
                    "row_idx": row_idx,
                    "field": field_def.name,
                    "value": value,
                    "allowed_values": list(field_def.allowed_values),
                },
            )

    @staticmethod
    def _validate_regex(modality: str, row_idx: int, field_def: DatasetField, value: Any) -> None:
        if field_def.regex is None or not isinstance(value, str):
            return
        if re.fullmatch(field_def.regex, value) is None:
            raise DataValidationError(
                f"{modality}[{row_idx}] field '{field_def.name}' failed regex validation",
                context={
                    "modality": modality,
                    "row_idx": row_idx,
                    "field": field_def.name,
                    "regex": field_def.regex,
                },
            )

    def enforce_multimodal_alignment(self, payload: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
        """Validate payload-level ingestion contracts and then per-modality records."""
        if not payload:
            if self.allow_empty_payload:
                return
            raise DataIngestionContractError("Payload is empty", context={"payload": payload})

        expected = self.required_modalities or tuple(self.schemas.keys())
        if self.enforce_alignment:
            assert_modalities_aligned(payload, expected_modalities=expected)
        else:
            self._assert_expected_modalities_without_alignment(payload, expected)

        for modality, rows in payload.items():
            self.validate_records(modality, rows)

    def _assert_expected_modalities_without_alignment(
        self,
        payload: Mapping[str, Sequence[Mapping[str, Any]]],
        expected_modalities: Sequence[str],
    ) -> None:
        expected_set = set(expected_modalities)
        actual_set = set(payload.keys())
        missing = expected_set - actual_set
        unexpected = actual_set - expected_set
        if missing or (unexpected and self.fail_on_unknown_modality):
            raise DataIngestionContractError(
                "Payload modality mismatch",
                context={
                    "expected": sorted(expected_set),
                    "actual": sorted(actual_set),
                    "missing": sorted(missing),
                    "unexpected": sorted(unexpected),
                },
            )

    def quality_gate(self, payload: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
        """Return quality metrics and raise on configured hard failures."""
        quality_cfg = get_config_section("quality_gate")
        self.enforce_multimodal_alignment(payload)

        if not payload:
            return {
                "total_records": 0,
                "total_cells": 0,
                "total_nulls": 0,
                "null_ratio": 0.0,
                "modalities": [],
                "validated_at": datetime.now(timezone.utc).isoformat(),
            }

        null_audit = audit_nulls(payload)
        modality_stats = compute_modality_stats(payload)
        duplicate_audit = self._audit_duplicate_ids(payload, str(quality_cfg.get("id_field", "id")))

        max_null_ratio = float(quality_cfg.get("max_null_ratio", 0.0))
        if bool(quality_cfg.get("fail_on_nulls", True)) and null_audit["null_ratio"] > max_null_ratio:
            raise DataQualityGateError(
                f"Null ratio {null_audit['null_ratio']:.8f} exceeds threshold {max_null_ratio:.8f}",
                context={"null_audit": null_audit, "max_null_ratio": max_null_ratio},
            )

        ragged_modalities = [m for m, stats in modality_stats.items() if stats.get("is_ragged")]
        if bool(quality_cfg.get("fail_on_ragged_fields", False)) and ragged_modalities:
            raise DataQualityGateError(
                "Payload contains ragged records",
                context={"ragged_modalities": ragged_modalities, "modality_stats": modality_stats},
            )

        max_duplicate_ratio = float(quality_cfg.get("max_duplicate_id_ratio", 0.0))
        if bool(quality_cfg.get("fail_on_duplicate_ids", False)):
            offenders = {
                modality: stats
                for modality, stats in duplicate_audit.items()
                if stats["duplicate_ratio"] > max_duplicate_ratio
            }
            if offenders:
                raise DataQualityGateError(
                    "Duplicate identifier ratio exceeds configured threshold",
                    context={"duplicates": offenders, "max_duplicate_id_ratio": max_duplicate_ratio},
                )

        return {
            "total_records": null_audit["total_records"],
            "total_cells": null_audit["total_cells"],
            "total_nulls": null_audit["total_nulls"],
            "null_ratio": null_audit["null_ratio"],
            "per_modality_nulls": null_audit["per_modality"],
            "duplicate_ids": duplicate_audit,
            "modalities": sorted(payload.keys()),
            "validated_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _audit_duplicate_ids(
        payload: Mapping[str, Sequence[Mapping[str, Any]]],
        id_field: str,
    ) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        if not id_field:
            return result

        for modality, rows in payload.items():
            seen: set[Any] = set()
            duplicates: set[Any] = set()
            present = 0
            for row in rows:
                if id_field not in row or row[id_field] is None:
                    continue
                present += 1
                value = row[id_field]
                if value in seen:
                    duplicates.add(value)
                seen.add(value)

            result[modality] = {
                "id_field": id_field,
                "records_with_id": present,
                "duplicate_count": len(duplicates),
                "duplicate_ratio": round(len(duplicates) / max(present, 1), 8),
                "duplicates": sorted(str(value) for value in duplicates),
            }
        return result

    def modality_stats(self, payload: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
        self.enforce_multimodal_alignment(payload)
        return compute_modality_stats(payload) if payload else {}


# =============================================================================
# Dataset registry / lineage
# =============================================================================

class DatasetVersionRegistry:
    """Atomic versioning and lineage registry for reproducible datasets."""

    def __init__(self, registry_path: str | Path | None = None):
        version_cfg = get_config_section("versioning")
        path_value = registry_path or version_cfg.get("registry_path")
        if not path_value:
            raise DataConfigError("Missing versioning.registry_path in config")

        self.registry_path = Path(path_value).expanduser().resolve()
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            atomic_write_json([], self.registry_path)

        self.hash_algorithm = str(version_cfg.get("hash_algorithm", "sha256"))
        self.duplicate_policy = str(version_cfg.get("duplicate_policy", "raise")).lower().strip()
        if self.duplicate_policy not in {"raise", "return_existing"}:
            raise DataConfigError(
                "Unsupported versioning.duplicate_policy",
                context={"duplicate_policy": self.duplicate_policy, "supported": ["raise", "return_existing"]},
            )
        self._assert_hash_algorithm_supported(self.hash_algorithm)
        self._lock = RLock()

        logger.info(
            {
                "event": "dataset_version_registry_initialized",
                "registry_path": str(self.registry_path),
                "hash_algorithm": self.hash_algorithm,
                "duplicate_policy": self.duplicate_policy,
            }
        )

    def _load(self) -> list[dict[str, Any]]:
        with self._lock:
            try:
                raw = self.registry_path.read_text(encoding="utf-8")
                rows = json.loads(raw or "[]")
            except (OSError, json.JSONDecodeError) as exc:
                raise DataVersioningError(
                    "Failed to read dataset registry",
                    context={"registry_path": str(self.registry_path)},
                    cause=exc,
                ) from exc

            if not isinstance(rows, list):
                raise DataVersioningError(
                    "Dataset registry must contain a JSON list",
                    context={"registry_path": str(self.registry_path), "type": type(rows).__name__},
                )
            return rows

    def _save(self, rows: list[dict[str, Any]]) -> None:
        with self._lock:
            atomic_write_json(rows, self.registry_path)

    def register(
        self,
        lineage: DatasetLineage,
        payload: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        quality_report: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload_hash = self._compute_payload_hash(payload, self.hash_algorithm)
        record = {
            **lineage.to_dict(),
            "payload_hash": payload_hash,
            "hash_algorithm": self.hash_algorithm,
            "modalities": sorted(payload.keys()),
            "record_count": sum(len(rows) for rows in payload.values()),
            "record_count_by_modality": {modality: len(rows) for modality, rows in payload.items()},
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        if quality_report is not None and bool(get_config_section("versioning").get("include_quality_report", True)):
            record["quality_report"] = dict(quality_report)

        records = self._load()
        existing = self.get(lineage.dataset_name, lineage.dataset_version, records=records)
        if existing is not None:
            if existing.get("payload_hash") == payload_hash and self.duplicate_policy == "return_existing":
                return existing
            raise DataVersioningError(
                "Dataset name/version already exists in registry",
                context={
                    "dataset_name": lineage.dataset_name,
                    "dataset_version": lineage.dataset_version,
                    "existing_payload_hash": existing.get("payload_hash"),
                    "incoming_payload_hash": payload_hash,
                },
            )

        records.append(record)
        self._save(records)
        logger.info(
            {
                "event": "dataset_version_registered",
                "dataset_name": lineage.dataset_name,
                "dataset_version": lineage.dataset_version,
                "payload_hash": payload_hash,
            }
        )
        return record

    def get(
        self,
        dataset_name: str,
        dataset_version: str,
        *,
        records: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        rows = records if records is not None else self._load()
        for row in rows:
            if row.get("dataset_name") == dataset_name and row.get("dataset_version") == dataset_version:
                return dict(row)
        return None

    def list_records(self, dataset_name: str | None = None) -> list[dict[str, Any]]:
        records = self._load()
        if dataset_name is None:
            return [dict(row) for row in records]
        return [dict(row) for row in records if row.get("dataset_name") == dataset_name]

    def latest(self, dataset_name: str) -> dict[str, Any] | None:
        records = self.list_records(dataset_name)
        if not records:
            return None
        return max(records, key=lambda row: str(row.get("registered_at") or row.get("created_at") or ""))

    def verify_payload(
        self,
        dataset_name: str,
        dataset_version: str,
        payload: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> bool:
        record = self.get(dataset_name, dataset_version)
        if record is None:
            raise DataVersioningError(
                "Dataset version not found in registry",
                context={"dataset_name": dataset_name, "dataset_version": dataset_version},
            )

        actual_hash = self._compute_payload_hash(payload, str(record.get("hash_algorithm", self.hash_algorithm)))
        expected_hash = str(record.get("payload_hash"))
        if actual_hash != expected_hash:
            raise SecurityError(
                "Payload hash mismatch — possible data tampering",
                context={
                    "dataset_name": dataset_name,
                    "dataset_version": dataset_version,
                    "expected_hash": expected_hash,
                    "actual_hash": actual_hash,
                },
            )
        return True

    @staticmethod
    def _assert_hash_algorithm_supported(algorithm: str) -> None:
        try:
            hashlib.new(algorithm)
        except ValueError as exc:
            raise DataConfigError(
                "Unsupported hash algorithm",
                context={"algorithm": algorithm, "available": sorted(hashlib.algorithms_available)},
                cause=exc,
            ) from exc

    @staticmethod
    def _compute_payload_hash(payload: Mapping[str, Sequence[Mapping[str, Any]]], algorithm: str = "sha256") -> str:
        DatasetVersionRegistry._assert_hash_algorithm_supported(algorithm)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        hash_func = hashlib.new(algorithm)
        hash_func.update(canonical.encode("utf-8"))
        return hash_func.hexdigest()


# =============================================================================
# Orchestration facade
# =============================================================================

class DataGovernance:
    """Facade that combines validation, quality gates, statistics, and lineage."""

    def __init__(
        self,
        schemas: Sequence[DatasetSchema] | None = None,
        *,
        validator: DatasetValidator | None = None,
        registry: DatasetVersionRegistry | None = None,
    ):
        if validator is None and schemas is None:
            raise DataSchemaError("DataGovernance requires schemas or a DatasetValidator", context={})
        self.validator = validator or DatasetValidator(schemas or ())
        self.registry = registry or DatasetVersionRegistry()
        self.config = load_global_config()

    def assess(
        self,
        payload: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        lineage: DatasetLineage | None = None,
        register: bool = False,
    ) -> DatasetGovernanceReport:
        quality = self.validator.quality_gate(payload)
        stats = self.validator.modality_stats(payload)
        lineage_record: dict[str, Any] | None = None

        if register:
            if lineage is None:
                raise DataVersioningError("lineage is required when register=True", context={})
            lineage_record = self.registry.register(lineage, payload, quality_report=quality)

        return DatasetGovernanceReport(
            quality=quality,
            modality_stats=stats,
            lineage_record=lineage_record,
        )

    def verify_registered_payload(
        self,
        dataset_name: str,
        dataset_version: str,
        payload: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> bool:
        return self.registry.verify_payload(dataset_name, dataset_version, payload)


# =============================================================================
# Internal config helpers
# =============================================================================

def _positive_int(value: Any, *, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise DataConfigError(
            f"{name} must be an integer",
            context={"name": name, "value": value, "type": type(value).__name__},
            cause=exc,
        ) from exc
    if parsed < 1:
        raise DataConfigError(f"{name} must be >= 1", context={"name": name, "value": parsed})
    return parsed


if __name__ == "__main__":
    import tempfile

    print("\n=== Running Data Governance ===\n")
    printer.status("TEST", "Data Governance initialized", "info")

    schemas = (
        DatasetSchema("vision_schema", "1.0", "vision", (DatasetField("id", str), DatasetField("image_tokens", list, max_items=8),)),
        DatasetSchema("text_schema", "1.0", "text", (DatasetField("id", str), DatasetField("text", str, max_items=512),)),
        DatasetSchema("audio_schema", "1.0", "audio", (DatasetField("id", str), DatasetField("audio_features", list, max_items=8),)),
    )
    payload = {
        "vision": [{"id": "r1", "image_tokens": [1, 2]}, {"id": "r2", "image_tokens": [3]}],
        "text": [{"id": "r1", "text": "hello"}, {"id": "r2", "text": "world"}],
        "audio": [{"id": "r1", "audio_features": [0.1]}, {"id": "r2", "audio_features": [0.2]}],
    }

    validator = DatasetValidator(schemas)
    quality = validator.quality_gate(payload)
    assert quality["total_records"] == 6
    assert validator.modality_stats(payload)["vision"]["record_count"] == 2

    with tempfile.TemporaryDirectory() as tmp:
        registry = DatasetVersionRegistry(Path(tmp) / "dataset_registry.json")
        governance = DataGovernance(validator=validator, registry=registry)
        lineage = DatasetLineage("multimodal_test", "1.0.0", "memory://unit-test", "local", "identity")
        report = governance.assess(payload, lineage=lineage, register=True)
        assert report.lineage_record is not None
        assert governance.verify_registered_payload("multimodal_test", "1.0.0", payload) is True

    try:
        validator.validate_records("text", [{"id": "bad", "text": None}])
    except DataValidationError:
        printer.status("TEST", "Expected validation failure captured", "warning")
    else:
        raise AssertionError("Expected DataValidationError for null text")

    print("\n=== Test ran successfully ===\n")
