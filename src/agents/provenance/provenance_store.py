"""
Lightweight durable storage for SLAI provenance records.

The store follows the infrastructure principles of provenance-aware storage
(Muniswamy-Reddy et al., 2006): provenance metadata is persisted alongside
stable artifact identities and remains retrievable independently of callers.
The logical representation stays PROV-compatible while the physical backend is
an intentionally small deterministic JSON document suitable for SLAI v2.3.

The store does not score sources, evaluate outputs, or capture runtime traces.
"""

from __future__ import annotations

__version__ = "2.3.0"

import threading

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from .provenance_types import (
    CheckpointRecord,
    CustodyRecord,
    DependencyRecord,
    LineageRecord,
    ProvenanceActivity,
    ProvenanceAgentRef,
    ProvenanceEntity,
    ProvenanceRecord,
    ProvenanceRelation,
    SourceRecord,
    TransformationRecord,
)
from .utils.config_loader import get_config_section, load_global_config
from .utils.provenance_errors import *
from .utils.provenance_helpers import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("Provenance Store")
printer = PrettyPrinter()


class ProvenanceStore:
    """Thread-safe JSON-backed persistence for immutable provenance records."""

    DEFAULT_RELATIVE_PATH = Path("data/provenance/provenance_store.json")
    DEFAULT_SCHEMA_VERSION = "slai.provenance.store.v1"
    _TABLES = (
        "entities",
        "activities",
        "agents",
        "relations",
        "lineage_records",
        "custody_records",
        "sources",
        "checkpoints",
        "dependencies",
        "transformations",
    )

    _registry_guard = threading.RLock()
    _path_locks: Dict[str, threading.RLock] = {}

    def __init__(self, storage_path: Optional[str | Path] = None, *, persist: Optional[bool] = None) -> None:
        self.config = load_global_config()
        self.store_config = get_config_section("provenance_store", config=self.config, default={})
        configured_path = storage_path or self.store_config.get("path") or self.DEFAULT_RELATIVE_PATH
        self.storage_path = self._resolve_storage_path(configured_path)
        self.persist = bool(self.store_config.get("persist", True) if persist is None else persist)
        self.schema_version = str(self.store_config.get("schema_version", self.DEFAULT_SCHEMA_VERSION))
        self._lock = self._lock_for_path(self.storage_path)
        self._mtime_ns: Optional[int] = None
        self._state = self._empty_state()
        self._indexes: Dict[str, Any] = {}
        self._load_initial_state()

    @classmethod
    def _lock_for_path(cls, path: Path) -> threading.RLock:
        key = str(path)
        with cls._registry_guard:
            lock = cls._path_locks.get(key)
            if lock is None:
                lock = threading.RLock()
                cls._path_locks[key] = lock
            return lock

    @staticmethod
    def _repository_root() -> Path:
        return Path(__file__).resolve().parents[3]

    @classmethod
    def _resolve_storage_path(cls, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = cls._repository_root() / path
        return path.resolve()

    def _empty_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "updated_at": None,
        }
        for table in self._TABLES:
            state[table] = {}
        return state

    def _normalize_state(self, raw: Mapping[str, Any]) -> Dict[str, Any]:
        schema = str(raw.get("schema_version") or self.schema_version)
        if schema != self.schema_version:
            raise ProvenanceStorageError(
                "unsupported provenance store schema",
                context={
                    "expected": self.schema_version,
                    "observed": schema,
                    "path": str(self.storage_path),
                },
            )
        state = self._empty_state()
        state["updated_at"] = raw.get("updated_at")
        for table in self._TABLES:
            value = raw.get(table, {})
            if not isinstance(value, Mapping):
                raise ProvenanceStorageError(
                    "provenance store table must be a mapping",
                    context={"table": table, "type": type(value).__name__},
                )
            state[table] = {str(key): canonicalize_value(item) for key, item in value.items()}
        return state

    def _load_initial_state(self) -> None:
        with self._lock:
            if self.persist and self.storage_path.exists():
                self._state = self._normalize_state(load_json_mapping(self.storage_path))
                self._mtime_ns = self.storage_path.stat().st_mtime_ns
            else:
                self._state = self._empty_state()
            self._rebuild_indexes()

    def _refresh_if_changed(self) -> None:
        if not self.persist or not self.storage_path.exists():
            return
        try:
            mtime_ns = self.storage_path.stat().st_mtime_ns
        except OSError as exc:
            raise ProvenanceStorageError(
                "failed to stat provenance store",
                context={"path": str(self.storage_path)},
                cause=exc,
            ) from exc
        if self._mtime_ns == mtime_ns:
            return
        self._state = self._normalize_state(load_json_mapping(self.storage_path))
        self._mtime_ns = mtime_ns
        self._rebuild_indexes()

    def _commit(self) -> None:
        self._state["updated_at"] = get_current_timestamp()
        if not self.persist:
            return
        atomic_write_provenance_json(self.storage_path, self._state)
        try:
            self._mtime_ns = self.storage_path.stat().st_mtime_ns
        except OSError as exc:
            raise ProvenanceStorageError(
                "provenance store was written but its metadata could not be read",
                context={"path": str(self.storage_path)},
                cause=exc,
            ) from exc

    def _rebuild_indexes(self) -> None:
        lineage_by_artifact: Dict[str, List[str]] = defaultdict(list)
        children_by_parent: Dict[str, Set[str]] = defaultdict(set)
        custody_by_artifact: Dict[str, List[str]] = defaultdict(list)
        dependencies_by_artifact: Dict[str, List[str]] = defaultdict(list)
        transformations_by_output: Dict[str, List[str]] = defaultdict(list)
        checkpoints_by_artifact: Dict[str, List[str]] = defaultdict(list)
        checkpoints_by_model: Dict[str, List[str]] = defaultdict(list)
        relations_by_subject: Dict[str, List[str]] = defaultdict(list)
        relations_by_object: Dict[str, List[str]] = defaultdict(list)

        for record_id, record in self._state["lineage_records"].items():
            artifact_id = record.get("artifact_id")
            if not artifact_id:
                continue
            lineage_by_artifact[str(artifact_id)].append(record_id)
            parents = record.get("parent_artifact_ids") or ()
            for parent in parents:
                children_by_parent[str(parent)].add(str(artifact_id))

        for event_id, record in self._state["custody_records"].items():
            artifact_id = record.get("artifact_id")
            if artifact_id:
                custody_by_artifact[str(artifact_id)].append(event_id)

        for record_id, record in self._state["dependencies"].items():
            artifact_id = record.get("artifact_id")
            if artifact_id:
                dependencies_by_artifact[str(artifact_id)].append(record_id)

        for transformation_id, record in self._state["transformations"].items():
            for output_id in record.get("output_ids") or ():
                transformations_by_output[str(output_id)].append(transformation_id)

        for checkpoint_id, record in self._state["checkpoints"].items():
            artifact_id = record.get("artifact_id")
            model_id = record.get("model_id")
            if artifact_id:
                checkpoints_by_artifact[str(artifact_id)].append(checkpoint_id)
            if model_id:
                checkpoints_by_model[str(model_id)].append(checkpoint_id)

        for relation_id, record in self._state["relations"].items():
            subject_id = record.get("subject_id")
            object_id = record.get("object_id")
            if subject_id:
                relations_by_subject[str(subject_id)].append(relation_id)
            if object_id:
                relations_by_object[str(object_id)].append(relation_id)

        for mapping in (
            lineage_by_artifact,
            custody_by_artifact,
            dependencies_by_artifact,
            transformations_by_output,
            checkpoints_by_artifact,
            checkpoints_by_model,
            relations_by_subject,
            relations_by_object,
        ):
            for values in mapping.values():
                values.sort()

        self._indexes = {
            "lineage_by_artifact": dict(lineage_by_artifact),
            "children_by_parent": {key: set(value) for key, value in children_by_parent.items()},
            "custody_by_artifact": dict(custody_by_artifact),
            "dependencies_by_artifact": dict(dependencies_by_artifact),
            "transformations_by_output": dict(transformations_by_output),
            "checkpoints_by_artifact": dict(checkpoints_by_artifact),
            "checkpoints_by_model": dict(checkpoints_by_model),
            "relations_by_subject": dict(relations_by_subject),
            "relations_by_object": dict(relations_by_object),
        }

    @staticmethod
    def _record_payload(record: ProvenanceRecord | Mapping[str, Any]) -> Dict[str, Any]:
        if isinstance(record, ProvenanceRecord):
            return record.to_dict()
        if not isinstance(record, Mapping):
            raise ProvenanceValidationError(
                "provenance record must be a typed record or mapping",
                context={"type": type(record).__name__},
            )
        normalized = canonicalize_value(record)
        if not isinstance(normalized, dict):
            raise ProvenanceValidationError("provenance record did not normalize to a mapping")
        return normalized

    def _upsert(self, table: str, key: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        if table not in self._TABLES:
            raise ProvenanceStorageError("unknown provenance storage table", context={"table": table})
        stable_key = require_identifier(key, field_name=f"{table}_key")
        normalized = self._record_payload(payload)
        with self._lock:
            self._refresh_if_changed()
            existing = self._state[table].get(stable_key)
            if existing is not None:
                if canonical_json_dumps(existing) == canonical_json_dumps(normalized):
                    return dict(existing)
                raise ProvenanceConflictError(
                    "stable provenance identity already exists with different content",
                    context={"table": table, "key": stable_key},
                )
            self._state[table][stable_key] = normalized
            self._commit()
            self._rebuild_indexes()
            logger.debug("provenance record persisted | table=%s | key=%s", table, stable_key)
            return dict(normalized)

    def _get(self, table: str, key: str, *, strict: bool = True) -> Optional[Dict[str, Any]]:
        stable_key = require_identifier(key, field_name=f"{table}_key")
        with self._lock:
            self._refresh_if_changed()
            value = self._state[table].get(stable_key)
            if value is None:
                if strict:
                    raise ProvenanceNotFoundError(
                        "provenance record was not found",
                        context={"table": table, "key": stable_key},
                    )
                return None
            return dict(value)

    def _list(self, table: str) -> List[Dict[str, Any]]:
        with self._lock:
            self._refresh_if_changed()
            return [dict(self._state[table][key]) for key in sorted(self._state[table])]

    # ------------------------------------------------------------------
    # Canonical PROV entities / activities / agents / relations
    # ------------------------------------------------------------------
    def save_entity(self, record: ProvenanceEntity | Mapping[str, Any]) -> Dict[str, Any]:
        typed = record if isinstance(record, ProvenanceEntity) else ProvenanceEntity.from_dict(record)
        return self._upsert("entities", typed.entity_id, typed.to_dict())

    def get_entity(self, entity_id: str, *, strict: bool = True) -> Optional[Dict[str, Any]]:
        return self._get("entities", entity_id, strict=strict)

    def list_entities(self) -> List[Dict[str, Any]]:
        return self._list("entities")

    def save_activity(self, record: ProvenanceActivity | Mapping[str, Any]) -> Dict[str, Any]:
        typed = record if isinstance(record, ProvenanceActivity) else ProvenanceActivity.from_dict(record)
        return self._upsert("activities", typed.activity_id, typed.to_dict())

    def save_agent(self, record: ProvenanceAgentRef | Mapping[str, Any]) -> Dict[str, Any]:
        typed = record if isinstance(record, ProvenanceAgentRef) else ProvenanceAgentRef.from_dict(record)
        return self._upsert("agents", typed.agent_id, typed.to_dict())

    def save_relation(self, record: ProvenanceRelation | Mapping[str, Any]) -> Dict[str, Any]:
        typed = record if isinstance(record, ProvenanceRelation) else ProvenanceRelation.from_dict(record)
        return self._upsert("relations", typed.relation_id, typed.to_dict())

    # ------------------------------------------------------------------
    # Lineage
    # ------------------------------------------------------------------
    def _coerce_lineage_record(self, record: LineageRecord | Mapping[str, Any]) -> LineageRecord:
        if isinstance(record, LineageRecord):
            return record
        payload = dict(record)
        if "parent_artifact_ids" not in payload:
            singular_parent = payload.pop("parent_artifact_id", None)
            payload["parent_artifact_ids"] = () if singular_parent is None else (singular_parent,)
        payload.setdefault("timestamp", get_current_timestamp())
        payload.setdefault("lineage_type", "artifact")
        if not payload.get("record_id"):
            payload["record_id"] = stable_provenance_id(
                "lineage",
                payload.get("artifact_id"),
                payload.get("parent_artifact_ids"),
                payload.get("transformation_id"),
                payload.get("transformation"),
                payload.get("timestamp"),
                payload.get("agent_id"),
            )
        return LineageRecord.from_dict(payload)

    def save_lineage_record(self, record: LineageRecord | Mapping[str, Any]) -> Dict[str, Any]:
        typed = self._coerce_lineage_record(record)
        with self._lock:
            self._refresh_if_changed()
            if lineage_would_create_cycle(
                self._state["lineage_records"].values(),
                child_id=typed.artifact_id,
                parent_ids=typed.parent_artifact_ids,
            ):
                raise ProvenanceGraphError(
                    "lineage record would introduce a derivation cycle",
                    context={
                        "artifact_id": typed.artifact_id,
                        "parent_artifact_ids": list(typed.parent_artifact_ids),
                    },
                )
            return self._upsert("lineage_records", typed.record_id, typed.to_dict())

    def get_lineage_records(self, artifact_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            self._refresh_if_changed()
            if artifact_id is None:
                keys = sorted(self._state["lineage_records"])
            else:
                artifact = require_identifier(artifact_id, field_name="artifact_id")
                keys = list(self._indexes["lineage_by_artifact"].get(artifact, ()))
            return [dict(self._state["lineage_records"][key]) for key in keys]

    def children_of(self, artifact_id: str) -> List[str]:
        artifact = require_identifier(artifact_id, field_name="artifact_id")
        with self._lock:
            self._refresh_if_changed()
            return sorted(self._indexes["children_by_parent"].get(artifact, set()))

    # ------------------------------------------------------------------
    # Custody
    # ------------------------------------------------------------------
    def _coerce_custody_record(self, record: CustodyRecord | Mapping[str, Any]) -> CustodyRecord:
        if isinstance(record, CustodyRecord):
            return record
        payload = dict(record)
        if "new_custodian" not in payload and "owner" in payload:
            payload["new_custodian"] = payload.pop("owner")
        payload.setdefault("previous_custodian", None)
        payload.setdefault("timestamp", get_current_timestamp())
        if not payload.get("event_id"):
            payload["event_id"] = stable_provenance_id(
                "custody",
                payload.get("artifact_id"),
                payload.get("previous_custodian"),
                payload.get("new_custodian"),
                payload.get("timestamp"),
            )
        return CustodyRecord.from_dict(payload)

    def save_custody_record(self, record: CustodyRecord | Mapping[str, Any]) -> Dict[str, Any]:
        typed = self._coerce_custody_record(record)
        return self._upsert("custody_records", typed.event_id, typed.to_dict())

    def get_custody_history(self, artifact_id: str) -> List[Dict[str, Any]]:
        artifact = require_identifier(artifact_id, field_name="artifact_id")
        with self._lock:
            self._refresh_if_changed()
            keys = self._indexes["custody_by_artifact"].get(artifact, ())
            records = [dict(self._state["custody_records"][key]) for key in keys]
        records.sort(key=lambda item: (item.get("timestamp", ""), item.get("event_id", "")))
        return records

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------
    def save_source_record(self, record: SourceRecord | Mapping[str, Any]) -> Dict[str, Any]:
        typed = record if isinstance(record, SourceRecord) else SourceRecord.from_dict(record)
        return self._upsert("sources", typed.source_id, typed.to_dict())

    def get_source(self, source_id: str, *, strict: bool = True) -> Optional[Dict[str, Any]]:
        return self._get("sources", source_id, strict=strict)

    def list_sources(self) -> List[Dict[str, Any]]:
        return self._list("sources")

    # ------------------------------------------------------------------
    # Checkpoints / dependencies / transformations
    # ------------------------------------------------------------------
    def save_checkpoint_record(self, record: CheckpointRecord | Mapping[str, Any]) -> Dict[str, Any]:
        typed = record if isinstance(record, CheckpointRecord) else CheckpointRecord.from_dict(record)
        with self._lock:
            self._refresh_if_changed()
            parent = typed.parent_checkpoint_id
            if parent == typed.checkpoint_id:
                raise ProvenanceGraphError(
                    "checkpoint cannot be its own parent",
                    context={"checkpoint_id": typed.checkpoint_id},
                )
            seen: Set[str] = set()
            cursor = parent
            while cursor:
                if cursor == typed.checkpoint_id:
                    raise ProvenanceGraphError(
                        "checkpoint ancestry would introduce a cycle",
                        context={"checkpoint_id": typed.checkpoint_id, "parent_checkpoint_id": parent},
                    )
                if cursor in seen:
                    raise ProvenanceGraphError(
                        "cycle detected in existing checkpoint ancestry",
                        context={"checkpoint_id": cursor},
                    )
                seen.add(cursor)
                current = self._state["checkpoints"].get(cursor)
                cursor = str(current.get("parent_checkpoint_id")) if isinstance(current, Mapping) and current.get("parent_checkpoint_id") else None
            return self._upsert("checkpoints", typed.checkpoint_id, typed.to_dict())

    def get_checkpoint(self, checkpoint_id: str, *, strict: bool = True) -> Optional[Dict[str, Any]]:
        return self._get("checkpoints", checkpoint_id, strict=strict)

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        return self._list("checkpoints")

    def save_dependency_record(self, record: DependencyRecord | Mapping[str, Any]) -> Dict[str, Any]:
        typed = record if isinstance(record, DependencyRecord) else DependencyRecord.from_dict(record)
        return self._upsert("dependencies", typed.record_id, typed.to_dict())

    def get_dependencies(self, artifact_id: str) -> List[Dict[str, Any]]:
        artifact = require_identifier(artifact_id, field_name="artifact_id")
        with self._lock:
            self._refresh_if_changed()
            keys = self._indexes["dependencies_by_artifact"].get(artifact, ())
            return [dict(self._state["dependencies"][key]) for key in keys]

    def save_transformation_record(self, record: TransformationRecord | Mapping[str, Any]) -> Dict[str, Any]:
        typed = record if isinstance(record, TransformationRecord) else TransformationRecord.from_dict(record)
        with self._lock:
            self._refresh_if_changed()
            for parent in typed.parent_transformation_ids:
                if parent == typed.transformation_id:
                    raise ProvenanceGraphError(
                        "transformation cannot be its own parent",
                        context={"transformation_id": typed.transformation_id},
                    )
                seen: Set[str] = set()
                stack: List[str] = [parent]
                while stack:
                    cursor = stack.pop()
                    if cursor == typed.transformation_id:
                        raise ProvenanceGraphError(
                            "transformation ancestry would introduce a cycle",
                            context={"transformation_id": typed.transformation_id, "parent_transformation_id": parent},
                        )
                    if cursor in seen:
                        continue
                    seen.add(cursor)
                    current = self._state["transformations"].get(cursor)
                    existing_parents = current.get("parent_transformation_ids") if isinstance(current, Mapping) else None
                    if existing_parents:
                        stack.extend(str(item) for item in existing_parents)
            return self._upsert("transformations", typed.transformation_id, typed.to_dict())

    def get_transformation(self, transformation_id: str, *, strict: bool = True) -> Optional[Dict[str, Any]]:
        return self._get("transformations", transformation_id, strict=strict)

    def get_transformations_for_output(self, artifact_id: str) -> List[Dict[str, Any]]:
        artifact = require_identifier(artifact_id, field_name="artifact_id")
        with self._lock:
            self._refresh_if_changed()
            keys = self._indexes["transformations_by_output"].get(artifact, ())
            return [dict(self._state["transformations"][key]) for key in keys]

    # ------------------------------------------------------------------
    # Aggregate retrieval
    # ------------------------------------------------------------------
    def get_provenance(self, artifact_id: str) -> Dict[str, Any]:
        """Return all stored derivation evidence directly associated with an artifact."""

        artifact = require_identifier(artifact_id, field_name="artifact_id")
        with self._lock:
            self._refresh_if_changed()
            entity = self._state["entities"].get(artifact)
            lineage_keys = self._indexes["lineage_by_artifact"].get(artifact, ())
            lineage = [dict(self._state["lineage_records"][key]) for key in lineage_keys]
            children = sorted(self._indexes["children_by_parent"].get(artifact, set()))
            custody_keys = self._indexes["custody_by_artifact"].get(artifact, ())
            custody = [dict(self._state["custody_records"][key]) for key in custody_keys]
            dependency_keys = self._indexes["dependencies_by_artifact"].get(artifact, ())
            dependencies = [dict(self._state["dependencies"][key]) for key in dependency_keys]
            transformation_keys = self._indexes["transformations_by_output"].get(artifact, ())
            transformations = [dict(self._state["transformations"][key]) for key in transformation_keys]
            checkpoint_keys = set(self._indexes["checkpoints_by_artifact"].get(artifact, ()))
            checkpoint_keys.update(self._indexes["checkpoints_by_model"].get(artifact, ()))
            checkpoints = [dict(self._state["checkpoints"][key]) for key in sorted(checkpoint_keys)]
            relation_keys = set(self._indexes["relations_by_subject"].get(artifact, ()))
            relation_keys.update(self._indexes["relations_by_object"].get(artifact, ()))
            relations = [dict(self._state["relations"][key]) for key in sorted(relation_keys)]

            source_ids: Set[str] = set()
            parent_ids: Set[str] = set()
            agent_ids: Set[str] = set()
            activity_ids: Set[str] = set()
            for record in lineage:
                source_ids.update(str(value) for value in record.get("source_ids") or ())
                parent_ids.update(str(value) for value in record.get("parent_artifact_ids") or ())
                if record.get("agent_id"):
                    agent_ids.add(str(record["agent_id"]))
                if record.get("transformation_id"):
                    activity_ids.add(str(record["transformation_id"]))

            sources = [
                dict(self._state["sources"][source_id])
                for source_id in sorted(source_ids)
                if source_id in self._state["sources"]
            ]
            agents = [
                dict(self._state["agents"][agent_id])
                for agent_id in sorted(agent_ids)
                if agent_id in self._state["agents"]
            ]
            activities = [
                dict(self._state["activities"][activity_id])
                for activity_id in sorted(activity_ids)
                if activity_id in self._state["activities"]
            ]

            has_evidence = any(
                (
                    entity,
                    lineage,
                    children,
                    custody,
                    dependencies,
                    transformations,
                    checkpoints,
                    relations,
                )
            )
            if not has_evidence:
                raise ProvenanceNotFoundError(
                    "no provenance evidence is known for artifact",
                    context={"artifact_id": artifact},
                )

            custody.sort(key=lambda item: (item.get("timestamp", ""), item.get("event_id", "")))
            lineage.sort(key=lambda item: (item.get("timestamp", ""), item.get("record_id", "")))

            return {
                "artifact_id": artifact,
                "entity": dict(entity) if isinstance(entity, Mapping) else None,
                "parents": sorted(parent_ids),
                "children": children,
                "lineage": lineage,
                "custody": custody,
                "dependencies": dependencies,
                "transformations": transformations,
                "checkpoints": checkpoints,
                "sources": sources,
                "agents": agents,
                "activities": activities,
                "relations": relations,
            }

    def snapshot(self) -> Dict[str, Any]:
        """Return a detached deterministic view of the full provenance store."""

        with self._lock:
            self._refresh_if_changed()
            return canonicalize_value(self._state)


__all__ = ["ProvenanceStore"]


if __name__ == "__main__":
    configure_logging()
    smoke = ProvenanceStore(persist=False)
    printer.status("PROVENANCE", f"store ready ({smoke.schema_version})", "success")
