"""
Checkpoint-local provenance memory for SLAI v2.3.

``ProvenanceMemory`` is deliberately narrower than both SLAI SharedMemory and
``ProvenanceStore``:

- SharedMemory coordinates transient references/events between agents.
- ProvenanceStore owns durable provenance facts and graph evidence.
- ProvenanceMemory maintains a bounded local manifest/index for checkpoint
  provenance and Provenance-Agent lifecycle recovery.
- CheckpointManager remains responsible for creating/restoring physical SLAI
  checkpoints; this module records provenance *about* those checkpoints.

The design follows W3C PROV's immutable evidence model, ModelDB/PROV-ML model
lineage, and Sandve/ReproZip reproducibility practice.  It does not store
conversation history, semantic knowledge, runtime telemetry, or quality scores.
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
    """Bounded checkpoint-provenance manifest and restart/recovery index.

    The manifest mirrors only checkpoint references required for fast local
    lifecycle recovery.  ``ProvenanceStore`` remains authoritative: pruning a
    local manifest entry never deletes the durable checkpoint provenance fact.
    """

    CHECKPOINT_PREFIX = "checkpoint"
    MANIFEST_FILE = "checkpoints_manifest.json"
    MANIFEST_SCHEMA = "slai.provenance.checkpoint-manifest.v2"
    LEGACY_SCHEMAS = frozenset({"slai.provenance.checkpoint-manifest.v1"})

    def __init__(
        self,
        store: Optional[ProvenanceStore] = None,
        manifest_path: Optional[str | Path] = None,
        *,
        persist: Optional[bool] = None,
        max_checkpoints: Optional[int] = None,
    ) -> None:
        self.config = load_global_config()
        self.memory_config = get_config_section("provenance_memory", config=self.config, default={})
        self.provenance_store = store or ProvenanceStore()

        configured = (
            manifest_path
            or self.memory_config.get("path")
            or Path("data/provenance") / self.MANIFEST_FILE
        )
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[3] / path
        self.manifest_path = path.resolve()

        self.persist = bool(self.memory_config.get("persist", True) if persist is None else persist)
        configured_limit = (
            self.memory_config.get("max_checkpoints", 1000)
            if max_checkpoints is None
            else max_checkpoints
        )
        if isinstance(configured_limit, bool):
            raise ProvenanceValidationError("max_checkpoints must be a positive integer")
        try:
            self.max_checkpoints = int(configured_limit)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProvenanceValidationError("max_checkpoints must be a positive integer", cause=exc) from exc
        if self.max_checkpoints < 1:
            raise ProvenanceValidationError("max_checkpoints must be >= 1")

        self._lock = threading.RLock()
        self._closed = False
        self._dirty = False
        self._manifest = self._load_manifest()

    # ------------------------------------------------------------------
    # Manifest lifecycle
    # ------------------------------------------------------------------
    def _empty_manifest(self) -> dict[str, Any]:
        now = get_current_timestamp()
        return {
            "schema_version": self.MANIFEST_SCHEMA,
            "created_at": now,
            "updated_at": None,
            "revision": 0,
            "checkpoints": {},
        }

    def _ensure_open(self) -> None:
        if self._closed:
            raise ProvenanceStorageError(
                "ProvenanceMemory is closed",
                context={"path": str(self.manifest_path)},
            )

    def _normalize_manifest(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        schema = raw.get("schema_version")
        if schema != self.MANIFEST_SCHEMA and schema not in self.LEGACY_SCHEMAS:
            raise ProvenanceStorageError(
                "unsupported checkpoint manifest schema",
                context={"path": str(self.manifest_path), "schema": schema},
            )

        checkpoints = raw.get("checkpoints", {})
        if not isinstance(checkpoints, Mapping):
            raise ProvenanceStorageError("checkpoint manifest entries must be a mapping")

        normalized: dict[str, Any] = {}
        for raw_id, raw_record in checkpoints.items():
            checkpoint_id = require_identifier(raw_id, field_name="checkpoint_id")
            if not isinstance(raw_record, Mapping):
                raise ProvenanceStorageError(
                    "checkpoint manifest entry must be a mapping",
                    context={"checkpoint_id": checkpoint_id},
                )
            record = CheckpointRecord.from_dict(raw_record)
            if record.checkpoint_id != checkpoint_id:
                raise ProvenanceConflictError(
                    "checkpoint manifest key conflicts with record identity",
                    context={
                        "manifest_key": checkpoint_id,
                        "record_checkpoint_id": record.checkpoint_id,
                    },
                )
            normalized[checkpoint_id] = record.to_dict()

        revision = raw.get("revision", 0)
        if isinstance(revision, bool):
            raise ProvenanceStorageError("checkpoint manifest revision must be an integer")
        try:
            revision_value = int(revision)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProvenanceStorageError(
                "checkpoint manifest revision must be an integer", cause=exc
            ) from exc
        if revision_value < 0:
            raise ProvenanceStorageError("checkpoint manifest revision cannot be negative")

        return {
            "schema_version": self.MANIFEST_SCHEMA,
            "created_at": raw.get("created_at") or get_current_timestamp(),
            "updated_at": raw.get("updated_at"),
            "revision": revision_value,
            "checkpoints": normalized,
        }

    def _load_manifest(self) -> dict[str, Any]:
        if not self.persist or not self.manifest_path.exists():
            return self._empty_manifest()
        raw = load_json_mapping(self.manifest_path)
        return self._normalize_manifest(raw)

    def _commit_locked(self, *, force: bool = False) -> None:
        self._ensure_open()
        if not force and not self._dirty:
            return
        self._manifest["updated_at"] = get_current_timestamp()
        if self.persist:
            atomic_write_provenance_json(self.manifest_path, self._manifest)
        self._dirty = False

    def flush(self) -> None:
        """Persist pending local manifest state atomically."""

        with self._lock:
            self._commit_locked(force=self._dirty)

    def restore(self) -> dict[str, Any]:
        """Reload local checkpoint lifecycle state from its manifest.

        Durable provenance facts are not restored from this file; they remain in
        ``ProvenanceStore``.  The returned snapshot is intentionally compact.
        """

        with self._lock:
            self._ensure_open()
            if not self.persist:
                # With persistence disabled there is no restart state to load;
                # keep the current in-memory index intact.
                return self.snapshot(include_checkpoints=False)
            self._manifest = self._load_manifest()
            self._dirty = False
            return self.snapshot(include_checkpoints=False)

    def close(self) -> None:
        """Flush local state and make subsequent mutation explicit failures."""

        with self._lock:
            if self._closed:
                return
            self._commit_locked(force=self._dirty)
            self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def revision(self) -> int:
        with self._lock:
            return int(self._manifest.get("revision", 0))

    def snapshot(self, *, include_checkpoints: bool = False) -> dict[str, Any]:
        """Return deterministic local-memory lifecycle state.

        ``include_checkpoints=False`` is the intended BaseAgent-checkpoint form:
        it records only a reference to the local provenance manifest rather than
        duplicating every durable checkpoint fact.
        """

        with self._lock:
            # Read-only inspection remains available after ``close()`` so the
            # Agent can publish a final lifecycle state without reopening or
            # mutating provenance-local memory.
            checkpoints = self._manifest["checkpoints"]
            payload: dict[str, Any] = {
                "schema_version": self.MANIFEST_SCHEMA,
                "manifest_path": str(self.manifest_path),
                "persist": self.persist,
                "revision": int(self._manifest.get("revision", 0)),
                "checkpoint_count": len(checkpoints),
                "latest_checkpoint_id": self._latest_checkpoint_id_locked(),
                "updated_at": self._manifest.get("updated_at"),
            }
            if include_checkpoints:
                payload["checkpoints"] = {
                    key: dict(checkpoints[key]) for key in sorted(checkpoints)
                }
            return canonicalize_value(payload)

    # ------------------------------------------------------------------
    # Checkpoint indexing / ancestry
    # ------------------------------------------------------------------
    def _record_for_id_locked(self, checkpoint_id: str) -> Optional[dict[str, Any]]:
        local = self._manifest["checkpoints"].get(checkpoint_id)
        if isinstance(local, Mapping):
            return dict(local)
        stored = self.provenance_store.get_checkpoint(checkpoint_id, strict=False)
        return dict(stored) if isinstance(stored, Mapping) else None

    def _parent_of_locked(self, checkpoint_id: str) -> Optional[str]:
        item = self._record_for_id_locked(checkpoint_id)
        if not item:
            return None
        parent = item.get("parent_checkpoint_id")
        return str(parent) if parent else None

    def _would_cycle(self, checkpoint_id: str, parent_checkpoint_id: Optional[str]) -> bool:
        if parent_checkpoint_id is None:
            return False
        cursor: Optional[str] = parent_checkpoint_id
        seen: set[str] = set()
        while cursor:
            if cursor == checkpoint_id or cursor in seen:
                return True
            seen.add(cursor)
            cursor = self._parent_of_locked(cursor)
        return False

    def _latest_checkpoint_id_locked(self, model_id: Optional[str] = None) -> Optional[str]:
        records = []
        for checkpoint_id, raw in self._manifest["checkpoints"].items():
            if not isinstance(raw, Mapping):
                continue
            if model_id is not None and raw.get("model_id") != model_id:
                continue
            records.append((str(raw.get("created_at") or ""), checkpoint_id))
        if not records:
            return None
        records.sort(key=lambda item: (item[0], item[1]))
        return records[-1][1]

    def _prune_locked(self) -> list[str]:
        checkpoints = self._manifest["checkpoints"]
        overflow = len(checkpoints) - self.max_checkpoints
        if overflow <= 0:
            return []
        ordered = sorted(
            checkpoints.items(),
            key=lambda item: (str(item[1].get("created_at") or ""), item[0]),
        )
        removed: list[str] = []
        for checkpoint_id, _ in ordered[:overflow]:
            checkpoints.pop(checkpoint_id, None)
            removed.append(checkpoint_id)
        return removed

    def prune_checkpoints(self, *, max_entries: Optional[int] = None) -> list[str]:
        """Prune only the local manifest index; durable store facts are retained."""

        with self._lock:
            self._ensure_open()
            if max_entries is not None:
                if isinstance(max_entries, bool) or int(max_entries) < 1:
                    raise ProvenanceValidationError("max_entries must be >= 1")
                original = self.max_checkpoints
                self.max_checkpoints = int(max_entries)
                try:
                    removed = self._prune_locked()
                finally:
                    self.max_checkpoints = original
            else:
                removed = self._prune_locked()
            if removed:
                self._manifest["revision"] = int(self._manifest.get("revision", 0)) + 1
                self._dirty = True
                self._commit_locked()
            return removed

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
        """Persist/index one immutable checkpoint-provenance record.

        Repeating the identical record is idempotent.  Reusing the checkpoint
        identity with different provenance raises ``ProvenanceConflictError``.
        """

        if isinstance(checkpoint, CheckpointRecord):
            record = checkpoint
        elif isinstance(checkpoint, Mapping):
            record = CheckpointRecord.from_dict(checkpoint)
        else:
            checkpoint_id = require_identifier(checkpoint, field_name="checkpoint_id")
            effective_timestamp = timestamp
            if effective_timestamp is None:
                # Creation time is immutable identity metadata, not a retry
                # nonce. Reuse it when the stable checkpoint already exists so
                # an otherwise identical retried capture remains idempotent.
                with self._lock:
                    self._ensure_open()
                    existing = self._record_for_id_locked(checkpoint_id)
                    if existing is not None:
                        effective_timestamp = existing.get("created_at")
            record = CheckpointRecord(
                checkpoint_id=checkpoint_id,
                model_id=model_id,
                parent_checkpoint_id=parent_checkpoint_id,
                training_run_id=training_run_id,
                dataset_ids=normalize_id_sequence(dataset_ids, field_name="dataset_ids"),
                configuration_id=configuration_id,
                code_version=code_version,
                framework_versions=dict(framework_versions or {}),
                artifact_id=artifact_id,
                created_at=effective_timestamp or get_current_timestamp(),
                metadata=normalize_metadata(metadata),
            )
        payload = record.to_dict()

        with self._lock:
            self._ensure_open()
            if self._would_cycle(record.checkpoint_id, record.parent_checkpoint_id):
                raise ProvenanceGraphError(
                    "checkpoint ancestry would introduce a cycle",
                    context={
                        "checkpoint_id": record.checkpoint_id,
                        "parent_checkpoint_id": record.parent_checkpoint_id,
                    },
                )

            existing_local = self._manifest["checkpoints"].get(record.checkpoint_id)
            if existing_local is not None and existing_local != payload:
                raise ProvenanceConflictError(
                    "checkpoint identity is already registered with different provenance",
                    context={"checkpoint_id": record.checkpoint_id},
                )

            # Store first: it is authoritative and enforces stable-ID conflicts.
            persisted = self.provenance_store.save_checkpoint_record(record)
            if persisted != payload:
                raise ProvenanceConflictError(
                    "persisted checkpoint provenance differs from requested record",
                    context={"checkpoint_id": record.checkpoint_id},
                )

            changed = existing_local is None
            self._manifest["checkpoints"][record.checkpoint_id] = payload
            removed = self._prune_locked()
            if changed or removed:
                self._manifest["revision"] = int(self._manifest.get("revision", 0)) + 1
                self._dirty = True
                self._commit_locked()
            return dict(payload)

    def get_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        identifier = require_identifier(checkpoint_id, field_name="checkpoint_id")
        with self._lock:
            self._ensure_open()
            item = self._record_for_id_locked(identifier)
            if item is None:
                raise ProvenanceNotFoundError(
                    "checkpoint is not known",
                    context={"checkpoint_id": identifier},
                )
            return item

    def list_checkpoints(self, *, model_id: Optional[str] = None, limit: Optional[int] = None) -> list[dict[str, Any]]:
        with self._lock:
            self._ensure_open()
            records = [
                dict(raw)
                for raw in self._manifest["checkpoints"].values()
                if isinstance(raw, Mapping)
                and (model_id is None or raw.get("model_id") == model_id)
            ]
            records.sort(
                key=lambda raw: (
                    str(raw.get("created_at") or ""),
                    str(raw.get("checkpoint_id") or ""),
                ),
                reverse=True,
            )
            if limit is not None:
                if isinstance(limit, bool) or int(limit) < 0:
                    raise ProvenanceValidationError("limit must be >= 0")
                records = records[: int(limit)]
            return records

    def latest_checkpoint(self, *, model_id: Optional[str] = None) -> Optional[dict[str, Any]]:
        with self._lock:
            self._ensure_open()
            checkpoint_id = self._latest_checkpoint_id_locked(model_id=model_id)
            return self._record_for_id_locked(checkpoint_id) if checkpoint_id else None

    def checkpoint_ancestry(self, checkpoint_id: str, *, max_depth: int = 1024) -> list[str]:
        if isinstance(max_depth, bool) or max_depth < 1:
            raise ProvenanceValidationError("max_depth must be >= 1")
        with self._lock:
            self._ensure_open()
            current = self.get_checkpoint(checkpoint_id)
            result: list[str] = []
            seen = {str(current["checkpoint_id"])}
            parent = current.get("parent_checkpoint_id")
            while parent:
                if len(result) >= max_depth:
                    raise ProvenanceGraphError(
                        "checkpoint ancestry exceeds configured traversal bound",
                        context={"checkpoint_id": checkpoint_id, "max_depth": max_depth},
                    )
                identifier = str(parent)
                if identifier in seen:
                    raise ProvenanceGraphError(
                        "cycle detected in checkpoint ancestry",
                        context={"checkpoint_id": identifier},
                    )
                seen.add(identifier)
                result.append(identifier)
                parent_record = self._record_for_id_locked(identifier)
                if parent_record is None:
                    raise ProvenanceNotFoundError(
                        "checkpoint ancestry references an unknown parent",
                        context={
                            "checkpoint_id": checkpoint_id,
                            "missing_parent_checkpoint_id": identifier,
                        },
                    )
                parent = parent_record.get("parent_checkpoint_id")
            return result


__all__ = ["ProvenanceMemory"]

if __name__ == "__main__":
    configure_logging()
    printer.status("SMOKE", "ProvenanceMemory module loaded", "success")
