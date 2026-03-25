from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from data.utils.config_loader import get_config_section
from data.utils.data_error import (
    DataConfigError,
    DataIngestionContractError,
    DataQualityGateError,
    DataValidationError,
    DataVersioningError,
)
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Data Governance")
printer = PrettyPrinter


@dataclass(frozen=True)
class DatasetField:
    name: str
    expected_type: type
    nullable: bool = False
    max_items: int | None = None


@dataclass(frozen=True)
class DatasetSchema:
    name: str
    version: str
    modality: str
    fields: tuple[DatasetField, ...]


@dataclass
class DatasetLineage:
    dataset_name: str
    dataset_version: str
    source_uri: str
    source_commit: str
    transform_id: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "source_uri": self.source_uri,
            "source_commit": self.source_commit,
            "transform_id": self.transform_id,
            "created_at": self.created_at,
        }


class DatasetValidator:
    """Validation layer with schema and quality gates for ingestion."""

    def __init__(self, schemas: Sequence[DatasetSchema], max_records: int | None = None):
        self.schemas = {s.modality: s for s in schemas}
        dataset_cfg = get_config_section("dataset")
        validation_cfg = get_config_section("validation")
        self.max_records = max_records or int(dataset_cfg.get("max_records", 1_000_000))
        self.fail_on_unknown_modality = bool(validation_cfg.get("fail_on_unknown_modality", True))

        logger.info(f"Dataset Validator initialized with {self.max_records}")

    def validate_records(self, modality: str, records: Sequence[Mapping[str, Any]]) -> None:
        if modality not in self.schemas:
            if self.fail_on_unknown_modality:
                raise DataValidationError(
                    f"Unknown modality: {modality}",
                    context={"modality": modality, "known_modalities": list(self.schemas.keys())},
                )
            return

        schema = self.schemas[modality]
        if len(records) == 0:
            raise DataValidationError(f"{modality}: no records provided", context={"modality": modality})
        if len(records) > self.max_records:
            raise DataValidationError(
                f"{modality}: record count {len(records)} exceeds max_records={self.max_records}",
                context={"modality": modality, "record_count": len(records), "max_records": self.max_records},
            )

        for row_idx, row in enumerate(records):
            for field in schema.fields:
                if field.name not in row:
                    raise DataValidationError(
                        f"{modality}[{row_idx}] missing required field '{field.name}'",
                        context={"modality": modality, "row_idx": row_idx, "field": field.name},
                    )
                value = row[field.name]
                if value is None and not field.nullable:
                    raise DataValidationError(
                        f"{modality}[{row_idx}] field '{field.name}' cannot be null",
                        context={"modality": modality, "row_idx": row_idx, "field": field.name},
                    )
                if value is not None and not isinstance(value, field.expected_type):
                    raise DataValidationError(
                        f"{modality}[{row_idx}] field '{field.name}' expected {field.expected_type.__name__}, "
                        f"got {type(value).__name__}",
                        context={
                            "modality": modality,
                            "row_idx": row_idx,
                            "field": field.name,
                            "expected_type": field.expected_type.__name__,
                            "actual_type": type(value).__name__,
                        },
                    )
                if field.max_items is not None and hasattr(value, "__len__") and len(value) > field.max_items:
                    raise DataValidationError(
                        f"{modality}[{row_idx}] field '{field.name}' exceeds max_items={field.max_items}",
                        context={
                            "modality": modality,
                            "row_idx": row_idx,
                            "field": field.name,
                            "max_items": field.max_items,
                            "actual_size": len(value),
                        },
                    )

    def enforce_multimodal_alignment(self, payload: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
        dataset_cfg = get_config_section("dataset")
        if not payload and not bool(dataset_cfg.get("allow_empty_payload", False)):
            raise DataIngestionContractError("Payload is empty", context={"payload": payload})

        lengths = {modality: len(rows) for modality, rows in payload.items()}
        if bool(dataset_cfg.get("enforce_alignment", True)) and len(set(lengths.values())) > 1:
            raise DataIngestionContractError(
                f"Modality alignment failed; expected equal sizes but got {lengths}",
                context={"lengths": lengths},
            )
        for modality, rows in payload.items():
            self.validate_records(modality, rows)

    def quality_gate(self, payload: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
        """Returns dataset quality metrics and raises on hard failures."""
        quality_cfg = get_config_section("quality_gate")
        self.enforce_multimodal_alignment(payload)

        total = sum(len(rows) for rows in payload.values())
        null_counts: dict[str, int] = {}
        total_nulls = 0

        for modality, rows in payload.items():
            nulls = 0
            for row in rows:
                nulls += sum(1 for v in row.values() if v is None)
            total_nulls += nulls
            null_counts[modality] = nulls

        null_ratio = (total_nulls / max(total, 1))
        max_null_ratio = float(quality_cfg.get("max_null_ratio", 0.0))
        if bool(quality_cfg.get("fail_on_nulls", True)) and null_ratio > max_null_ratio:
            raise DataQualityGateError(
                f"Null ratio {null_ratio:.6f} exceeds threshold {max_null_ratio:.6f}",
                context={"null_ratio": null_ratio, "max_null_ratio": max_null_ratio, "null_counts": null_counts},
            )

        return {
            "total_records": total,
            "modalities": sorted(payload.keys()),
            "null_counts": null_counts,
            "null_ratio": round(null_ratio, 6),
            "validated_at": datetime.now(timezone.utc).isoformat(),
        }


class DatasetVersionRegistry:
    """Versioning/lineage metadata for reproducible datasets."""

    def __init__(self, registry_path: str | Path | None = None):
        version_cfg = get_config_section("versioning")
        path_value = registry_path or version_cfg.get("registry_path")
        if not path_value:
            raise DataConfigError("Missing versioning.registry_path in config")

        self.registry_path = Path(path_value)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self.registry_path.write_text("[]", encoding="utf-8")
        self.hash_algorithm = str(version_cfg.get("hash_algorithm", "sha256"))

        logger.info(f"Dataset Version Registry initialized with registry from {self.registry_path}")

    def _load(self) -> list[dict[str, Any]]:
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _save(self, rows: list[dict[str, Any]]) -> None:
        self.registry_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def register(self, lineage: DatasetLineage, payload: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
        payload_hash = self._compute_payload_hash(payload, self.hash_algorithm)
        record = {
            **lineage.to_dict(),
            "payload_hash": payload_hash,
            "hash_algorithm": self.hash_algorithm,
            "modalities": sorted(payload.keys()),
            "record_count": sum(len(rows) for rows in payload.values()),
        }

        records = self._load()
        if any(
            r.get("dataset_name") == lineage.dataset_name
            and r.get("dataset_version") == lineage.dataset_version
            and r.get("payload_hash") == payload_hash
            for r in records
        ):
            raise DataVersioningError(
                "Duplicate dataset version and payload hash detected",
                context={
                    "dataset_name": lineage.dataset_name,
                    "dataset_version": lineage.dataset_version,
                    "payload_hash": payload_hash,
                },
            )

        records.append(record)
        self._save(records)
        return record

    @staticmethod
    def _compute_payload_hash(payload: Mapping[str, Sequence[Mapping[str, Any]]], algorithm: str = "sha256") -> str:
        canonical = json.dumps(payload, sort_keys=True, default=str)
        try:
            hash_func = getattr(hashlib, algorithm)
        except AttributeError as exc:
            raise DataConfigError("Unsupported hash algorithm", context={"algorithm": algorithm}) from exc
        return hash_func(canonical.encode("utf-8")).hexdigest()