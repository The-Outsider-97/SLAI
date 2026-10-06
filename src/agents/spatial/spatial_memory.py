"""Subsystem-local spatial state, cache, bounded history, and artifacts.

SpatialMemory is intentionally internal infrastructure. The future SpatialAgent
need not know it exists; subsystem services obtain the shared default instance
when one is not explicitly injected.
"""
from __future__ import annotations

__version__ = "2.3.0"

from collections import OrderedDict, deque
from copy import deepcopy
from threading import RLock
from time import monotonic
from typing import Any, Mapping

from .spatial_types import SpatialEntity, SpatialRelationship
from .utils.config_loader import get_config_section, load_global_config
from .utils.spatial_errors import SpatialValidationError
from .utils.spatial_helpers import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Spatial Memory")
printer = PrettyPrinter()


class SpatialMemory:
    def __init__(self) -> None:
        self.config = load_global_config()
        self.memory_config = get_config_section("spatial_memory", config=self.config, default={})
        self.max_history = max(1, int(self.memory_config.get("max_history", 1000)))
        self.max_query_cache = max(1, int(self.memory_config.get("max_query_cache", 256)))
        self.default_cache_ttl = max(0.0, float(self.memory_config.get("query_cache_ttl_seconds", 30.0)))
        self._lock = RLock()
        self._entities: dict[str, SpatialEntity] = {}
        self._relationships: dict[tuple[str, str, str], SpatialRelationship] = {}
        self._map_versions: dict[str, list[dict[str, Any]]] = {}
        self._transforms: dict[tuple[str, str], dict[str, Any]] = {}
        self._artifacts: dict[str, Any] = {}
        self._query_cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._history: deque[dict[str, Any]] = deque(maxlen=self.max_history)
        self._entity_revision = 0
        logger.info("Spatial Memory successfully initialized")

    @property
    def entity_revision(self) -> int:
        with self._lock:
            return self._entity_revision

    def upsert_entity(self, entity: SpatialEntity) -> SpatialEntity:
        if not isinstance(entity, SpatialEntity):
            raise SpatialValidationError("entity must be a SpatialEntity")
        with self._lock:
            self._entities[entity.entity_id] = entity
            self._entity_revision += 1
            self._invalidate_query_cache_locked()
            self._record_locked("entity_upsert", {"entity_id": entity.entity_id, "revision": entity.revision})
            return entity

    def get_entity(self, entity_id: str) -> SpatialEntity | None:
        key = validate_identifier(entity_id, name="entity_id")
        with self._lock:
            return self._entities.get(key)

    def require_entity(self, entity_id: str) -> SpatialEntity:
        entity = self.get_entity(entity_id)
        if entity is None:
            raise SpatialValidationError("unknown spatial entity", context={"entity_id": entity_id})
        return entity

    def remove_entity(self, entity_id: str) -> SpatialEntity | None:
        key = validate_identifier(entity_id, name="entity_id")
        with self._lock:
            entity = self._entities.pop(key, None)
            if entity is not None:
                self._entity_revision += 1
                self._relationships = {
                    relation_key: relation
                    for relation_key, relation in self._relationships.items()
                    if relation.subject_id != key and relation.object_id != key
                }
                self._invalidate_query_cache_locked()
                self._record_locked("entity_remove", {"entity_id": key})
            return entity

    def list_entities(self, *, frame_id: str | None = None) -> tuple[SpatialEntity, ...]:
        with self._lock:
            values = self._entities.values()
            if frame_id is not None:
                values = (entity for entity in values if entity.frame_id == frame_id)
            return tuple(values)

    def record_relationship(self, relationship: SpatialRelationship) -> SpatialRelationship:
        if not isinstance(relationship, SpatialRelationship):
            raise SpatialValidationError("relationship must be a SpatialRelationship")
        key = (relationship.subject_id, relationship.object_id, relationship.relation.value)
        with self._lock:
            self._relationships[key] = relationship
            self._record_locked("relationship", relationship.to_dict())
            return relationship

    def relationships_for(self, entity_id: str) -> tuple[SpatialRelationship, ...]:
        key = validate_identifier(entity_id, name="entity_id")
        with self._lock:
            return tuple(
                relation
                for relation in self._relationships.values()
                if relation.subject_id == key or relation.object_id == key
            )

    def record_map_version(self, map_id: str, descriptor: Mapping[str, Any]) -> int:
        key = validate_identifier(map_id, name="map_id")
        with self._lock:
            versions = self._map_versions.setdefault(key, [])
            payload = {"version": len(versions) + 1, "timestamp": utc_now_iso(), **to_json_safe(dict(descriptor))}
            versions.append(payload)
            self._record_locked("map_version", {"map_id": key, "version": payload["version"]})
            return int(payload["version"])

    def map_versions(self, map_id: str) -> tuple[dict[str, Any], ...]:
        key = validate_identifier(map_id, name="map_id")
        with self._lock:
            return tuple(deepcopy(self._map_versions.get(key, [])))

    def store_transform(self, source_frame: str, target_frame: str, transform: Any) -> None:
        source = validate_identifier(source_frame, name="source_frame")
        target = validate_identifier(target_frame, name="target_frame")
        with self._lock:
            self._transforms[(source, target)] = {
                "timestamp": utc_now_iso(),
                "value": transform,
            }
            self._record_locked("transform", {"source_frame": source, "target_frame": target})

    def get_transform(self, source_frame: str, target_frame: str) -> Any | None:
        with self._lock:
            payload = self._transforms.get((source_frame, target_frame))
            return None if payload is None else payload["value"]

    def store_artifact(self, artifact_id: str, artifact: Any) -> None:
        key = validate_identifier(artifact_id, name="artifact_id")
        with self._lock:
            self._artifacts[key] = artifact
            self._record_locked("artifact", {"artifact_id": key, "type": type(artifact).__name__})

    def get_artifact(self, artifact_id: str) -> Any | None:
        with self._lock:
            return self._artifacts.get(validate_identifier(artifact_id, name="artifact_id"))

    def cache_query(self, key: str, value: Any, *, ttl_seconds: float | None = None) -> None:
        cache_key = validate_identifier(key, name="cache_key")
        ttl = self.default_cache_ttl if ttl_seconds is None else max(0.0, float(ttl_seconds))
        expires_at = float("inf") if ttl == 0.0 else monotonic() + ttl
        with self._lock:
            self._query_cache.pop(cache_key, None)
            self._query_cache[cache_key] = (expires_at, value)
            while len(self._query_cache) > self.max_query_cache:
                self._query_cache.popitem(last=False)

    def get_cached_query(self, key: str) -> Any | None:
        cache_key = validate_identifier(key, name="cache_key")
        with self._lock:
            item = self._query_cache.get(cache_key)
            if item is None:
                return None
            expires_at, value = item
            if monotonic() > expires_at:
                self._query_cache.pop(cache_key, None)
                return None
            self._query_cache.move_to_end(cache_key)
            return value

    def clear_query_cache(self) -> None:
        with self._lock:
            self._invalidate_query_cache_locked()

    def recent_history(self, limit: int = 50) -> tuple[dict[str, Any], ...]:
        if limit < 1:
            raise SpatialValidationError("history limit must be >= 1")
        with self._lock:
            return tuple(deepcopy(list(self._history)[-limit:]))

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "entities": len(self._entities),
                "entity_revision": self._entity_revision,
                "relationships": len(self._relationships),
                "maps": len(self._map_versions),
                "transforms": len(self._transforms),
                "artifacts": len(self._artifacts),
                "cached_queries": len(self._query_cache),
                "history": len(self._history),
            }

    def export_state(self) -> dict[str, Any]:
        """Return checkpoint-compatible JSON-safe subsystem state."""
        with self._lock:
            return {
                "schema": "slai.spatial-memory.state.v1",
                "entity_revision": self._entity_revision,
                "entities": [entity.to_dict() for entity in self._entities.values()],
                "relationships": [relationship.to_dict() for relationship in self._relationships.values()],
                "map_versions": to_json_safe(self._map_versions),
                "transforms": [
                    {
                        "source_frame": source,
                        "target_frame": target,
                        "timestamp": payload["timestamp"],
                        "value": to_json_safe(payload["value"]),
                    }
                    for (source, target), payload in self._transforms.items()
                ],
                "history": to_json_safe(list(self._history)),
            }

    def _record_locked(self, event: str, payload: Mapping[str, Any]) -> None:
        self._history.append({"timestamp": utc_now_iso(), "event": event, "payload": to_json_safe(payload)})

    def _invalidate_query_cache_locked(self) -> None:
        self._query_cache.clear()


_DEFAULT_MEMORY: SpatialMemory | None = None
_DEFAULT_MEMORY_LOCK = RLock()


def get_default_spatial_memory() -> SpatialMemory:
    global _DEFAULT_MEMORY
    with _DEFAULT_MEMORY_LOCK:
        if _DEFAULT_MEMORY is None:
            _DEFAULT_MEMORY = SpatialMemory()
        return _DEFAULT_MEMORY


def reset_default_spatial_memory() -> SpatialMemory:
    """Replace the process-local subsystem memory; intended for tests/lifecycle reset."""
    global _DEFAULT_MEMORY
    with _DEFAULT_MEMORY_LOCK:
        _DEFAULT_MEMORY = SpatialMemory()
        return _DEFAULT_MEMORY


__all__ = ["SpatialMemory", "get_default_spatial_memory", "reset_default_spatial_memory"]


if __name__ == "__main__":
    configure_logging()
    memory = SpatialMemory()
    printer.status("TEST", f"stats={memory.stats()}", "info")
