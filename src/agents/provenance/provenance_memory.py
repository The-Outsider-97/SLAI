"""
Checkpoint-specific provenance memory.

This module is intentionally not another general SLAI memory system.  It keeps
checkpoint manifests and ancestry metadata required to reconstruct model state,
following reproducibility practice, ReproZip-style environment capture, and
ModelDB-style model/checkpoint lineage.
"""
from __future__ import annotations

__version__ = "2.3.0"

import threading

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Optional

from .utils.config_loader import get_config_section, load_global_config
from .utils.provenance_errors import *
from .utils.provenance_helpers import *
from .provenance_store import ProvenanceStore
from .provenance_types import CheckpointRecord
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Provenance Memory")
printer = PrettyPrinter()


class ProvenanceMemory:
    CHECKPOINT_PREFIX = "checkpoint"
    MANIFEST_FILE = "checkpoints_manifest.json"
    MANIFEST_SCHEMA = "slai.provenance.checkpoint-manifest.v1"

    def __init__(
        self,
        store: Optional[ProvenanceStore] = None,
        manifest_path: Optional[str | Path] = None,
        *,
        persist: Optional[bool] = None,
    ) -> None:
        self.config = load_global_config()
        self.memory_config = get_config_section("provenance_memory", config=self.config, default={})
        self.provenance_store = store or ProvenanceStore()
        configured = manifest_path or self.memory_config.get("path") or Path("data/provenance") / self.MANIFEST_FILE
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[3] / path
        self.manifest_path = path.resolve()
        self.persist = bool(self.memory_config.get("persist", True) if persist is None else persist)
        self._lock = threading.RLock()
        self._manifest = self._load_manifest()

    def _empty_manifest(self) -> dict[str, Any]:
        return {"schema_version": self.MANIFEST_SCHEMA, "updated_at": None, "checkpoints": {}}

    def _load_manifest(self) -> dict[str, Any]:
        if not self.persist or not self.manifest_path.exists():
            return self._empty_manifest()
        raw = load_json_mapping(self.manifest_path)
        if raw.get("schema_version") != self.MANIFEST_SCHEMA:
            raise ProvenanceStorageError(
                "unsupported checkpoint manifest schema",
                context={"path": str(self.manifest_path), "schema": raw.get("schema_version")},
            )
        checkpoints = raw.get("checkpoints", {})
        if not isinstance(checkpoints, Mapping):
            raise ProvenanceStorageError("checkpoint manifest entries must be a mapping")
        return {"schema_version": self.MANIFEST_SCHEMA, "updated_at": raw.get("updated_at"), "checkpoints": dict(checkpoints)}

    def _commit(self) -> None:
        self._manifest["updated_at"] = get_current_timestamp()
        if self.persist:
            atomic_write_provenance_json(self.manifest_path, self._manifest)

    def _would_cycle(self, checkpoint_id: str, parent_checkpoint_id: Optional[str]) -> bool:
        if parent_checkpoint_id is None:
            return False
        cursor: Optional[str] = parent_checkpoint_id
        seen: set[str] = set()
        while cursor:
            if cursor == checkpoint_id:
                return True
            if cursor in seen:
                return True
            seen.add(cursor)
            item = self._manifest["checkpoints"].get(cursor)
            cursor = str(item.get("parent_checkpoint_id")) if isinstance(item, Mapping) and item.get("parent_checkpoint_id") else None
        return False

    def save_checkpoint(
        self,
        checkpoint: CheckpointRecord | Mapping[str, Any] | str,
        *,
        model_id: Optional[str] = None,
        parent_checkpoint_id: Optional[str] = None,
        training_run_id: Optional[str] = None,
        dataset_ids: Optional[Iterable[str]] = None,
        configuration_id: Optional[str] = None,
        code_version: Optional[str] = None,
        framework_versions: Optional[Mapping[str, Any]] = None,
        artifact_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> dict[str, Any]:
        if isinstance(checkpoint, CheckpointRecord):
            record = checkpoint
        elif isinstance(checkpoint, Mapping):
            record = CheckpointRecord.from_dict(checkpoint)
        else:
            record = CheckpointRecord(
                checkpoint_id=require_identifier(checkpoint, field_name="checkpoint_id"),
                model_id=model_id,
                parent_checkpoint_id=parent_checkpoint_id,
                training_run_id=training_run_id,
                dataset_ids=normalize_id_sequence(dataset_ids, field_name="dataset_ids"),
                configuration_id=configuration_id,
                code_version=code_version,
                framework_versions=dict(framework_versions or {}),
                artifact_id=artifact_id,
                created_at=timestamp or get_current_timestamp(),
                metadata=normalize_metadata(metadata),
            )
        payload = record.to_dict()
        with self._lock:
            if self._would_cycle(record.checkpoint_id, record.parent_checkpoint_id):
                raise ProvenanceGraphError(
                    "checkpoint ancestry would introduce a cycle",
                    context={"checkpoint_id": record.checkpoint_id, "parent_checkpoint_id": record.parent_checkpoint_id},
                )
            existing = self._manifest["checkpoints"].get(record.checkpoint_id)
            if existing is not None and existing != payload:
                raise ProvenanceConflictError(
                    "checkpoint identity is already registered with different provenance",
                    context={"checkpoint_id": record.checkpoint_id},
                )
            self._manifest["checkpoints"][record.checkpoint_id] = payload
            self.provenance_store.save_checkpoint_record(record)
            self._commit()
            return dict(payload)

    def get_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        identifier = require_identifier(checkpoint_id, field_name="checkpoint_id")
        with self._lock:
            item = self._manifest["checkpoints"].get(identifier)
            if item is None:
                stored = self.provenance_store.get_checkpoint(identifier, strict=False)
                if stored is None:
                    raise ProvenanceNotFoundError("checkpoint is not known", context={"checkpoint_id": identifier})
                return stored
            return dict(item)

    def list_checkpoints(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(self._manifest["checkpoints"][key]) for key in sorted(self._manifest["checkpoints"])]

    def checkpoint_ancestry(self, checkpoint_id: str) -> list[str]:
        current = self.get_checkpoint(checkpoint_id)
        result: list[str] = []
        seen = {str(current["checkpoint_id"])}
        parent = current.get("parent_checkpoint_id")
        while parent:
            identifier = str(parent)
            if identifier in seen:
                raise ProvenanceGraphError("cycle detected in checkpoint ancestry", context={"checkpoint_id": identifier})
            seen.add(identifier)
            result.append(identifier)
            parent = self.get_checkpoint(identifier).get("parent_checkpoint_id")
        return result


__all__ = ["ProvenanceMemory"]

if __name__ == "__main__":
    configure_logging()
    printer.status("SMOKE", "ProvenanceMemory module loaded", "success")
