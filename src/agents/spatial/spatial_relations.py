"""Spatial-relation semantics over canonical entities and geometry.

Topological predicates delegate to ``world.topology``; metric and directional
relations use entity coordinates. This module does not reimplement low-level
intersection algorithms.
"""
from __future__ import annotations

__version__ = "2.3.0"

from typing import Any

import numpy as np # type: ignore

from .spatial_memory import SpatialMemory, get_default_spatial_memory
from .spatial_types import RelationKind, SpatialEntity, SpatialRelationship
from .utils.config_loader import get_config_section, load_global_config
from .utils.spatial_errors import SpatialGeometryError, SpatialTopologyError, SpatialValidationError
from .utils.spatial_helpers import DEFAULT_ABS_TOL, all_close
from .world.geometry import AABB, intersects as geometry_intersects
from .world.topology import Topology, contains as topology_contains, crosses as topology_crosses, disjoint as topology_disjoint, equals as topology_equals, inside as topology_inside, overlaps as topology_overlaps, touches as topology_touches
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Spatial Relations")
printer = PrettyPrinter()


class SpatialRelations:
    def __init__(self, memory: SpatialMemory | None = None) -> None:
        self.config = load_global_config()
        self.relations_config = get_config_section("spatial_relations", config=self.config, default={})
        self.memory = memory or get_default_spatial_memory()
        self.tolerance = float(self.relations_config.get("tolerance", DEFAULT_ABS_TOL))
        self.near_threshold = float(self.relations_config.get("near_threshold", 1.0))
        self.far_threshold = float(self.relations_config.get("far_threshold", 10.0))
        if self.tolerance <= 0.0 or self.near_threshold < 0.0 or self.far_threshold < self.near_threshold:
            raise SpatialValidationError("invalid spatial relation thresholds")
        self.topology = Topology()
        logger.info("Spatial relations successfully initialized")

    def relate(self, a: SpatialEntity | str, b: SpatialEntity | str, *, record: bool = True) -> tuple[SpatialRelationship, ...]:
        first, second = self._entities(a, b)
        self._require_same_frame(first, second)
        relationships: list[SpatialRelationship] = []

        predicates = (
            (RelationKind.INTERSECTS, self.intersects),
            (RelationKind.DISJOINT, self.disjoint),
            (RelationKind.CONTAINS, self.contains),
            (RelationKind.INSIDE, self.inside),
            (RelationKind.TOUCHES, self.touches),
            (RelationKind.OVERLAPS, self.overlaps),
            (RelationKind.CROSSES, self.crosses),
            (RelationKind.CONNECTED, self.connected),
            (RelationKind.EQUALS, self.equals),
            (RelationKind.NEAR, self.near),
            (RelationKind.FAR, self.far),
            (RelationKind.LEFT_OF, self.left_of),
            (RelationKind.RIGHT_OF, self.right_of),
            (RelationKind.ABOVE, self.above),
            (RelationKind.BELOW, self.below),
            (RelationKind.IN_FRONT_OF, self.in_front_of),
            (RelationKind.BEHIND, self.behind),
        )
        for relation, predicate in predicates:
            try:
                if predicate(first, second):
                    relationships.append(
                        SpatialRelationship(
                            first.entity_id,
                            second.entity_id,
                            relation,
                            value=True,
                            frame_id=first.frame_id,
                        )
                    )
            except (SpatialTopologyError, SpatialValidationError):
                logger.debug(
                    "Relation %s is not defined for %s/%s",
                    relation.value,
                    first.entity_id,
                    second.entity_id,
                )

        distance = self.distance(first, second)
        relationships.append(
            SpatialRelationship(
                first.entity_id,
                second.entity_id,
                RelationKind.DISTANCE,
                value=distance,
                frame_id=first.frame_id,
                metadata={"metric": "euclidean"},
            )
        )
        if record:
            for relationship in relationships:
                self.memory.record_relationship(relationship)
        return tuple(relationships)

    def distance(self, a: SpatialEntity | str, b: SpatialEntity | str) -> float:
        first, second = self._entities(a, b)
        self._require_same_frame(first, second)
        if first.dimension != second.dimension:
            raise SpatialValidationError("entity dimensions must match for distance")
        return float(np.linalg.norm(np.asarray(first.position) - np.asarray(second.position)))

    def intersects(self, a: SpatialEntity | str, b: SpatialEntity | str) -> bool:
        first, second = self._entities(a, b)
        self._require_same_frame(first, second)
        if first.geometry is None or second.geometry is None:
            return all_close(first.position, second.position, abs_tol=self.tolerance, rel_tol=0.0)
        try:
            return geometry_intersects(first.geometry, second.geometry, tolerance=self.tolerance)
        except SpatialGeometryError:
            if first.bounds is not None and second.bounds is not None:
                return geometry_intersects(
                    AABB(np.asarray(first.bounds.lower), np.asarray(first.bounds.upper)),
                    AABB(np.asarray(second.bounds.lower), np.asarray(second.bounds.upper)),
                    tolerance=self.tolerance,
                )
            raise

    def contains(self, a: SpatialEntity | str, b: SpatialEntity | str) -> bool:
        first, second = self._entities(a, b)
        return self._topological_binary(topology_contains, first, second)

    def inside(self, a: SpatialEntity | str, b: SpatialEntity | str) -> bool:
        first, second = self._entities(a, b)
        return self._topological_binary(topology_inside, first, second)

    def touches(self, a: SpatialEntity | str, b: SpatialEntity | str) -> bool:
        first, second = self._entities(a, b)
        return self._topological_binary(topology_touches, first, second)

    def overlaps(self, a: SpatialEntity | str, b: SpatialEntity | str) -> bool:
        first, second = self._entities(a, b)
        return self._topological_binary(topology_overlaps, first, second)

    def crosses(self, a: SpatialEntity | str, b: SpatialEntity | str) -> bool:
        first, second = self._entities(a, b)
        return self._topological_binary(topology_crosses, first, second)

    def connected(self, a: SpatialEntity | str, b: SpatialEntity | str) -> bool:
        return not self.disjoint(a, b)

    def disjoint(self, a: SpatialEntity | str, b: SpatialEntity | str) -> bool:
        first, second = self._entities(a, b)
        if first.geometry is None or second.geometry is None:
            return not self.intersects(first, second)
        try:
            return self._topological_binary(topology_disjoint, first, second)
        except SpatialTopologyError:
            return not self.intersects(first, second)

    disconnected = disjoint

    def equals(self, a: SpatialEntity | str, b: SpatialEntity | str) -> bool:
        first, second = self._entities(a, b)
        if first.geometry is None or second.geometry is None:
            return first.frame_id == second.frame_id and all_close(first.position, second.position, abs_tol=self.tolerance, rel_tol=0.0)
        return self._topological_binary(topology_equals, first, second)

    def near(self, a: SpatialEntity | str, b: SpatialEntity | str, *, threshold: float | None = None) -> bool:
        limit = self.near_threshold if threshold is None else float(threshold)
        if limit < 0.0:
            raise SpatialValidationError("near threshold must be non-negative")
        return self.distance(a, b) <= limit + self.tolerance

    def far(self, a: SpatialEntity | str, b: SpatialEntity | str, *, threshold: float | None = None) -> bool:
        limit = self.far_threshold if threshold is None else float(threshold)
        if limit < 0.0:
            raise SpatialValidationError("far threshold must be non-negative")
        return self.distance(a, b) >= limit - self.tolerance

    def left_of(self, a: SpatialEntity | str, b: SpatialEntity | str) -> bool:
        first, second = self._directional_pair(a, b, minimum_dimension=2)
        return first.position[0] < second.position[0] - self.tolerance

    def right_of(self, a: SpatialEntity | str, b: SpatialEntity | str) -> bool:
        first, second = self._directional_pair(a, b, minimum_dimension=2)
        return first.position[0] > second.position[0] + self.tolerance

    def above(self, a: SpatialEntity | str, b: SpatialEntity | str) -> bool:
        first, second = self._directional_pair(a, b, minimum_dimension=2)
        axis = 2 if first.dimension >= 3 else 1
        return first.position[axis] > second.position[axis] + self.tolerance

    def below(self, a: SpatialEntity | str, b: SpatialEntity | str) -> bool:
        first, second = self._directional_pair(a, b, minimum_dimension=2)
        axis = 2 if first.dimension >= 3 else 1
        return first.position[axis] < second.position[axis] - self.tolerance

    def in_front_of(self, a: SpatialEntity | str, b: SpatialEntity | str) -> bool:
        first, second = self._directional_pair(a, b, minimum_dimension=3)
        return first.position[1] > second.position[1] + self.tolerance

    def behind(self, a: SpatialEntity | str, b: SpatialEntity | str) -> bool:
        first, second = self._directional_pair(a, b, minimum_dimension=3)
        return first.position[1] < second.position[1] - self.tolerance

    def _topological_binary(self, function: Any, first: SpatialEntity, second: SpatialEntity) -> bool:
        self._require_same_frame(first, second)
        if first.geometry is None or second.geometry is None:
            raise SpatialTopologyError(
                "topological relation requires geometry on both entities",
                context={"first": first.entity_id, "second": second.entity_id},
            )
        return bool(function(first.geometry, second.geometry, tolerance=self.tolerance))

    def _directional_pair(self, a: SpatialEntity | str, b: SpatialEntity | str, *, minimum_dimension: int) -> tuple[SpatialEntity, SpatialEntity]:
        first, second = self._entities(a, b)
        self._require_same_frame(first, second)
        if first.dimension != second.dimension or first.dimension < minimum_dimension:
            raise SpatialValidationError("directional relation requires matching coordinate dimensions")
        return first, second

    def _entities(self, a: SpatialEntity | str, b: SpatialEntity | str) -> tuple[SpatialEntity, SpatialEntity]:
        return self._entity(a), self._entity(b)

    def _entity(self, value: SpatialEntity | str) -> SpatialEntity:
        if isinstance(value, SpatialEntity):
            return value
        if isinstance(value, str):
            return self.memory.require_entity(value)
        raise SpatialValidationError("relation operand must be SpatialEntity or entity id")

    @staticmethod
    def _require_same_frame(first: SpatialEntity, second: SpatialEntity) -> None:
        if first.frame_id != second.frame_id:
            raise SpatialValidationError(
                "spatial relation operands must be expressed in the same frame",
                context={"first_frame": first.frame_id, "second_frame": second.frame_id},
            )


__all__ = ["SpatialRelations"]


if __name__ == "__main__":
    configure_logging()
    memory = SpatialMemory()
    memory.upsert_entity(SpatialEntity("a", (0.0, 0.0, 0.0)))
    memory.upsert_entity(SpatialEntity("b", (1.0, 0.0, 0.0)))
    relations = SpatialRelations(memory)
    printer.status("TEST", f"near={relations.near('a', 'b')}", "info")
