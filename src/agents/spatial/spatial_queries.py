"""Spatial query interface built on memory-backed indexing and predicates."""
from __future__ import annotations

__version__ = "2.3.0"

from typing import Any, Iterable

from .utils.config_loader import get_config_section, load_global_config
from .utils.spatial_errors import SpatialError, SpatialQueryError, SpatialValidationError
from .world.geometry import AABB
from .world.visibility import visible_from as geometry_visible_from
from .spatial_index import SpatialIndex
from .spatial_memory import SpatialMemory, get_default_spatial_memory
from .spatial_relations import SpatialRelations
from .spatial_types import RelationKind, SpatialEntity, SpatialQueryResult
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Spatial Queries")
printer = PrettyPrinter()


class SpatialQueries:
    def __init__(
        self,
        memory: SpatialMemory | None = None,
        *,
        index: SpatialIndex | None = None,
        relations: SpatialRelations | None = None,
    ) -> None:
        self.config = load_global_config()
        self.queries_config = get_config_section("spatial_queries", config=self.config, default={})
        self.memory = memory or get_default_spatial_memory()
        self.index = index or SpatialIndex(self.memory)
        self.relations = relations or SpatialRelations(self.memory)
        logger.info("Spatial queries successfully initialized")

    def nearest(self, point: Any, *, frame_id: str | None = None, exclude_ids: set[str] | None = None) -> SpatialQueryResult:
        matches = self.index.nearest(point, k=1, frame_id=frame_id, exclude_ids=exclude_ids)
        return self._distance_result("nearest", matches, {"frame_id": frame_id})

    def k_nearest(self, point: Any, k: int, *, frame_id: str | None = None) -> SpatialQueryResult:
        if k < 1:
            raise SpatialQueryError("k must be >= 1")
        matches = self.index.k_nearest(point, k, frame_id=frame_id)
        return self._distance_result("k_nearest", matches, {"frame_id": frame_id, "k": k})

    def within_radius(self, point: Any, radius: float, *, frame_id: str | None = None) -> SpatialQueryResult:
        matches = self.index.within_radius(point, radius, frame_id=frame_id)
        return self._distance_result("within_radius", matches, {"frame_id": frame_id, "radius": radius})

    def within_bounds(self, bounds: AABB, *, frame_id: str | None = None) -> SpatialQueryResult:
        entities = self.index.within_bounds(bounds, frame_id=frame_id)
        return SpatialQueryResult("within_bounds", tuple(entity.entity_id for entity in entities), metadata={"frame_id": frame_id})

    def intersecting_bounds(self, bounds: AABB, *, frame_id: str | None = None) -> SpatialQueryResult:
        entities = self.index.intersecting_bounds(bounds, frame_id=frame_id)
        return SpatialQueryResult("intersecting_bounds", tuple(entity.entity_id for entity in entities), metadata={"frame_id": frame_id})

    def contains(self, container: SpatialEntity | str, *, candidates: Iterable[SpatialEntity | str] | None = None) -> SpatialQueryResult:
        container_entity = self._entity(container)
        pool = tuple(candidates) if candidates is not None else self.memory.list_entities(frame_id=container_entity.frame_id)
        matches: list[str] = []
        skipped: list[str] = []
        for candidate in pool:
            entity = self._entity(candidate)
            if entity.entity_id == container_entity.entity_id:
                continue
            try:
                if self.relations.contains(container_entity, entity):
                    matches.append(entity.entity_id)
            except SpatialError:
                skipped.append(entity.entity_id)
        return SpatialQueryResult(
            "contains",
            tuple(matches),
            metadata={"container": container_entity.entity_id, "unsupported_candidates": tuple(skipped)},
        )

    def intersects(self, entity: SpatialEntity | str, *, candidates: Iterable[SpatialEntity | str] | None = None) -> SpatialQueryResult:
        subject = self._entity(entity)
        pool = tuple(candidates) if candidates is not None else self.memory.list_entities(frame_id=subject.frame_id)
        matches: list[str] = []
        skipped: list[str] = []
        for candidate in pool:
            other = self._entity(candidate)
            if other.entity_id == subject.entity_id:
                continue
            try:
                if self.relations.intersects(subject, other):
                    matches.append(other.entity_id)
            except SpatialError:
                skipped.append(other.entity_id)
        return SpatialQueryResult(
            "intersects",
            tuple(matches),
            metadata={"entity_id": subject.entity_id, "unsupported_candidates": tuple(skipped)},
        )

    def query_relation(self, entity: SpatialEntity | str, relation: RelationKind | str, *, candidates: Iterable[SpatialEntity | str] | None = None) -> SpatialQueryResult:
        subject = self._entity(entity)
        relation_kind = relation if isinstance(relation, RelationKind) else RelationKind(str(relation))
        method_name = relation_kind.value
        method = getattr(self.relations, method_name, None)
        if method is None or not callable(method):
            raise SpatialQueryError("relation is not directly queryable", context={"relation": relation_kind.value})
        pool = tuple(candidates) if candidates is not None else self.memory.list_entities(frame_id=subject.frame_id)
        ids: list[str] = []
        relationships = []
        for candidate in pool:
            other = self._entity(candidate)
            if other.entity_id == subject.entity_id:
                continue
            try:
                if method(subject, other):
                    ids.append(other.entity_id)
                    relationships.append(
                        self._relationship(subject, other, relation_kind)
                    )
            except SpatialError:
                continue
        return SpatialQueryResult("relation", tuple(ids), relationships=tuple(relationships), metadata={"subject": subject.entity_id, "relation": relation_kind.value})

    def visible_from(
        self,
        observer: SpatialEntity | str,
        target: SpatialEntity | str,
        *,
        obstacles: Iterable[SpatialEntity | Any] | None = None,
    ) -> SpatialQueryResult:
        source = self._entity(observer)
        destination = self._entity(target)
        if source.frame_id != destination.frame_id or source.dimension != 3 or destination.dimension != 3:
            raise SpatialQueryError("visibility requires 3D entities in the same frame")
        if obstacles is None:
            obstacle_geometries = [
                entity.geometry
                for entity in self.memory.list_entities(frame_id=source.frame_id)
                if entity.entity_id not in {source.entity_id, destination.entity_id} and entity.geometry is not None
            ]
        else:
            obstacle_geometries = [item.geometry if isinstance(item, SpatialEntity) else item for item in obstacles]
        result = geometry_visible_from(source.position, destination.position, obstacle_geometries)
        ids = (destination.entity_id,) if result.visible else ()
        return SpatialQueryResult(
            "visible_from",
            ids,
            distances=(result.distance,) if result.visible else (),
            metadata={"observer": source.entity_id, "target": destination.entity_id, "visible": result.visible, "blocker_index": result.blocker_index},
        )

    def _entity(self, value: SpatialEntity | str) -> SpatialEntity:
        if isinstance(value, SpatialEntity):
            return value
        if isinstance(value, str):
            return self.memory.require_entity(value)
        raise SpatialValidationError("query entity must be SpatialEntity or entity id")

    @staticmethod
    def _relationship(subject: SpatialEntity, other: SpatialEntity, relation: RelationKind):
        from .spatial_types import SpatialRelationship
        return SpatialRelationship(subject.entity_id, other.entity_id, relation, value=True, frame_id=subject.frame_id)

    @staticmethod
    def _distance_result(query: str, matches: tuple[tuple[SpatialEntity, float], ...], metadata: dict[str, Any]) -> SpatialQueryResult:
        return SpatialQueryResult(
            query,
            tuple(entity.entity_id for entity, _ in matches),
            distances=tuple(distance for _, distance in matches),
            metadata=metadata,
        )


__all__ = ["SpatialQueries"]


if __name__ == "__main__":
    configure_logging()
    memory = SpatialMemory()
    memory.upsert_entity(SpatialEntity("a", (0.0, 0.0, 0.0)))
    memory.upsert_entity(SpatialEntity("b", (2.0, 0.0, 0.0)))
    queries = SpatialQueries(memory)
    printer.status("TEST", f"nearest={queries.nearest((0.25, 0, 0)).entity_ids}", "info")
