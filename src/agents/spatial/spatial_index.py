"""Memory-backed spatial acceleration structures.

The primary point index is an exact balanced k-d tree (Bentley, 1975). Bounds
queries use vectorized AABB broad-phase arrays. The authoritative entity state
remains in SpatialMemory, so independent SpatialIndex views stay coherent and
rebuild lazily only when the entity revision changes.
"""
from __future__ import annotations

__version__ = "2.3.0"

import numpy as np # type: ignore

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Any

from .utils.config_loader import get_config_section, load_global_config
from .utils.spatial_errors import SpatialGeometryError, SpatialIndexError
from .utils.spatial_helpers import finite_vector, validate_non_negative
from .world.geometry import AABB, geometry_bounds
from .spatial_memory import SpatialMemory, get_default_spatial_memory
from .spatial_types import SpatialBounds, SpatialEntity
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Spatial Index")
printer = PrettyPrinter()


@dataclass(slots=True)
class _KDNode:
    point_index: int
    axis: int
    left: "_KDNode | None" = None
    right: "_KDNode | None" = None


class KDTree:
    """Exact immutable k-d tree over a numeric point matrix."""

    def __init__(self, points: Any, ids: tuple[str, ...] | list[str]) -> None:
        array = np.asarray(points, dtype=float)
        if array.ndim != 2:
            raise SpatialIndexError("KDTree points must have shape (N, D)")
        if array.shape[0] != len(ids):
            raise SpatialIndexError("KDTree ids must align with points")
        if array.size and not np.all(np.isfinite(array)):
            raise SpatialIndexError("KDTree points must be finite")
        self.points = array.copy()
        self.ids = tuple(ids)
        self.dimension = int(array.shape[1]) if array.ndim == 2 and array.shape[0] else (int(array.shape[1]) if array.ndim == 2 else 0)
        indices = np.arange(len(self.points), dtype=int)
        self.root = self._build(indices, depth=0) if len(indices) else None

    def _build(self, indices: np.ndarray, depth: int) -> _KDNode | None:
        if len(indices) == 0:
            return None
        axis = depth % self.points.shape[1]
        ordered = indices[np.argsort(self.points[indices, axis], kind="stable")]
        middle = len(ordered) // 2
        index = int(ordered[middle])
        return _KDNode(
            point_index=index,
            axis=axis,
            left=self._build(ordered[:middle], depth + 1),
            right=self._build(ordered[middle + 1 :], depth + 1),
        )

    def nearest(self, point: Any, *, k: int = 1) -> tuple[tuple[str, float], ...]:
        if k < 1:
            raise SpatialIndexError("k must be >= 1")
        if self.root is None:
            return ()
        query = finite_vector(point, name="point", dimension=self.points.shape[1])
        heap: list[tuple[float, int]] = []  # (-squared_distance, -index)

        def visit(node: _KDNode | None) -> None:
            if node is None:
                return
            delta_vector = self.points[node.point_index] - query
            squared = float(delta_vector @ delta_vector)
            item = (-squared, -node.point_index)
            if len(heap) < k:
                heappush(heap, item)
            elif item > heap[0]:
                heappop(heap)
                heappush(heap, item)

            axis_delta = float(query[node.axis] - self.points[node.point_index, node.axis])
            near = node.left if axis_delta < 0.0 else node.right
            far = node.right if axis_delta < 0.0 else node.left
            visit(near)
            worst_squared = float("inf") if len(heap) < k else -heap[0][0]
            if axis_delta * axis_delta <= worst_squared:
                visit(far)

        visit(self.root)
        ordered = sorted(((-distance_sq, -index) for distance_sq, index in heap), key=lambda item: (item[0], item[1]))
        return tuple((self.ids[index], float(np.sqrt(distance_sq))) for distance_sq, index in ordered)

    def within_radius(self, point: Any, radius: float) -> tuple[tuple[str, float], ...]:
        if self.root is None:
            return ()
        query = finite_vector(point, name="point", dimension=self.points.shape[1])
        r = validate_non_negative(radius, name="radius")
        r2 = r * r
        matches: list[tuple[float, int]] = []

        def visit(node: _KDNode | None) -> None:
            if node is None:
                return
            delta = self.points[node.point_index] - query
            squared = float(delta @ delta)
            if squared <= r2:
                matches.append((squared, node.point_index))
            axis_delta = float(query[node.axis] - self.points[node.point_index, node.axis])
            if axis_delta <= 0.0:
                visit(node.left)
                if axis_delta * axis_delta <= r2:
                    visit(node.right)
            else:
                visit(node.right)
                if axis_delta * axis_delta <= r2:
                    visit(node.left)

        visit(self.root)
        matches.sort(key=lambda item: (item[0], item[1]))
        return tuple((self.ids[index], float(np.sqrt(squared))) for squared, index in matches)


class SpatialIndex:
    def __init__(self, memory: SpatialMemory | None = None) -> None:
        self.config = load_global_config()
        self.index_config = get_config_section("spatial_index", config=self.config, default={})
        self.memory = memory or get_default_spatial_memory()
        self._revision = -1
        self._active_frame: str | None = None
        self._dimension: int | None = None
        self._tree = KDTree(np.empty((0, 3)), ())
        self._bounds_ids: tuple[str, ...] = ()
        self._bounds_min = np.empty((0, 3), dtype=float)
        self._bounds_max = np.empty((0, 3), dtype=float)
        logger.info("Spatial index successfully initialized")

    def insert(self, entity: SpatialEntity) -> SpatialEntity:
        return self.memory.upsert_entity(entity)

    upsert = insert

    def remove(self, entity_id: str) -> SpatialEntity | None:
        return self.memory.remove_entity(entity_id)

    def get(self, entity_id: str) -> SpatialEntity | None:
        return self.memory.get_entity(entity_id)

    def k_nearest(self, point: Any, k: int, *, frame_id: str | None = None) -> tuple[tuple[SpatialEntity, float], ...]:
        return self.nearest(point, k=k, frame_id=frame_id)

    def _ensure_rebuilt(self, *, frame_id: str | None = None) -> None:
        # Frame-filtered views are rebuilt on demand because the authoritative
        # memory revision alone cannot identify which frame changed.
        revision = self.memory.entity_revision
        if revision == self._revision and frame_id == self._active_frame:
            return
        entities = self.memory.list_entities(frame_id=frame_id)
        if not entities:
            dimension = self._dimension or 3
            self._tree = KDTree(np.empty((0, dimension)), ())
            self._bounds_ids = ()
            self._bounds_min = np.empty((0, dimension))
            self._bounds_max = np.empty((0, dimension))
            self._revision = revision
            self._active_frame = frame_id
            return
        dimensions = {entity.dimension for entity in entities}
        if len(dimensions) != 1:
            raise SpatialIndexError("all indexed entities in a view must share dimensionality", context={"dimensions": sorted(dimensions)})
        dimension = dimensions.pop()
        points = np.asarray([entity.position for entity in entities], dtype=float)
        ids = tuple(entity.entity_id for entity in entities)
        self._tree = KDTree(points, ids)
        self._dimension = dimension

        bounded: list[tuple[str, AABB]] = []
        for entity in entities:
            box: AABB | None = None
            if entity.bounds is not None:
                box = self._coerce_bounds(entity.bounds)
            elif entity.geometry is not None:
                try:
                    box = geometry_bounds(entity.geometry)
                except SpatialGeometryError:
                    box = None
            if box is not None and box.dimension == dimension:
                bounded.append((entity.entity_id, box))
        self._bounds_ids = tuple(item[0] for item in bounded)
        self._bounds_min = np.asarray([item[1].minimum for item in bounded], dtype=float) if bounded else np.empty((0, dimension))
        self._bounds_max = np.asarray([item[1].maximum for item in bounded], dtype=float) if bounded else np.empty((0, dimension))
        self._revision = revision
        self._active_frame = frame_id

    def nearest(self, point: Any, *, k: int = 1, frame_id: str | None = None, exclude_ids: set[str] | None = None) -> tuple[tuple[SpatialEntity, float], ...]:
        self._ensure_rebuilt(frame_id=frame_id)
        exclusions = exclude_ids or set()
        if not exclusions:
            raw = self._tree.nearest(point, k=k)
        else:
            raw = self._tree.nearest(point, k=min(len(self._tree.ids), k + len(exclusions)))
            raw = tuple(item for item in raw if item[0] not in exclusions)[:k]
        return tuple((self.memory.require_entity(entity_id), distance) for entity_id, distance in raw)

    def within_radius(self, point: Any, radius: float, *, frame_id: str | None = None) -> tuple[tuple[SpatialEntity, float], ...]:
        self._ensure_rebuilt(frame_id=frame_id)
        return tuple((self.memory.require_entity(entity_id), distance) for entity_id, distance in self._tree.within_radius(point, radius))

    def within_bounds(self, bounds: AABB | SpatialBounds, *, frame_id: str | None = None) -> tuple[SpatialEntity, ...]:
        box = self._coerce_bounds(bounds)
        self._ensure_rebuilt(frame_id=frame_id)
        if self._tree.root is None:
            return ()
        points = self._tree.points
        mask = np.all(points >= box.minimum, axis=1) & np.all(points <= box.maximum, axis=1)
        return tuple(self.memory.require_entity(self._tree.ids[index]) for index in np.flatnonzero(mask))

    def intersecting_bounds(self, bounds: AABB | SpatialBounds, *, frame_id: str | None = None) -> tuple[SpatialEntity, ...]:
        box = self._coerce_bounds(bounds)
        self._ensure_rebuilt(frame_id=frame_id)
        if len(self._bounds_ids) == 0:
            return ()
        if box.dimension != self._bounds_min.shape[1]:
            raise SpatialIndexError("query bounds dimension does not match index dimension")
        mask = np.all(self._bounds_max >= box.minimum, axis=1) & np.all(self._bounds_min <= box.maximum, axis=1)
        return tuple(self.memory.require_entity(self._bounds_ids[index]) for index in np.flatnonzero(mask))

    def snapshot(self) -> dict[str, Any]:
        self._ensure_rebuilt()
        return {
            "entity_revision": self._revision,
            "dimension": self._dimension,
            "point_entries": len(self._tree.ids),
            "bounded_entries": len(self._bounds_ids),
        }

    @staticmethod
    def _coerce_bounds(bounds: AABB | SpatialBounds) -> AABB:
        if isinstance(bounds, AABB):
            return bounds
        if isinstance(bounds, SpatialBounds):
            return AABB(np.asarray(bounds.lower, dtype=float), np.asarray(bounds.upper, dtype=float))
        raise SpatialIndexError("bounds must be AABB or SpatialBounds")


__all__ = ["KDTree", "SpatialIndex"]


if __name__ == "__main__":
    configure_logging()
    memory = SpatialMemory()
    index = SpatialIndex(memory)
    index.insert(SpatialEntity("a", (0.0, 0.0, 0.0)))
    index.insert(SpatialEntity("b", (2.0, 0.0, 0.0)))
    printer.status("TEST", f"nearest={index.nearest((0.5, 0, 0))[0][0].entity_id}", "info")
