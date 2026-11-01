"""High-level deterministic spatial-computation façade.

``SpatialCompute`` coordinates existing low-level Spatial modules. It does not
own geometry algorithms, planning, perception, persistence, or subsystem
configuration beyond delegating to its existing services.

The small frame-management methods in this façade intentionally expose the
already-implemented ``CoordinateFrames``/``FrameGraph`` capability so callers
such as ``SpatialAgent`` do not need to reach through subsystem internals.
"""
from __future__ import annotations

__version__ = "2.3.0"

import numpy as np  # type: ignore

from typing import Any, Mapping

from .modules.calculations import Calculations
from .modules.coordinate_frames import CoordinateFrames
from .modules.transform import RigidTransform
from .spatial_memory import SpatialMemory, get_default_spatial_memory
from .spatial_types import SpatialEntity
from .utils.config_loader import get_config_section, load_global_config
from .utils.spatial_errors import SpatialGeometryError, SpatialValidationError
from .world.collision import CollisionEngine, CollisionResult
from .world.geometry import geometry_bounds
from .world.visibility import VisibilityResult, visible_from
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Spatial Compute")
printer = PrettyPrinter()


class SpatialCompute:
    """Orchestrate deterministic Spatial calculations and frame operations."""

    def __init__(
        self,
        calculations: Calculations | None = None,
        memory: SpatialMemory | None = None,
        *,
        frames: CoordinateFrames | None = None,
        collision: CollisionEngine | None = None,
    ) -> None:
        self.config = load_global_config()
        self.compute_config = get_config_section("spatial_calculations", config=self.config, default={})
        self.calculator = calculations or Calculations()
        self.memory = memory or get_default_spatial_memory()
        self.frames = frames or CoordinateFrames(root_frame=str(self.compute_config.get("root_frame", "world")))
        self.collision = collision or CollisionEngine()
        logger.info("Spatial calculations successfully initialized")

    # ------------------------------------------------------------------
    # Metric / geometry façade
    # ------------------------------------------------------------------

    def distance(self, a: Any, b: Any, *, metric: str = "euclidean") -> float:
        first = self._position(a)
        second = self._position(b)
        if metric == "euclidean":
            return self.calculator.euclidean_distance(first, second)
        if metric == "squared_euclidean":
            return self.calculator.squared_distance(first, second)
        if metric == "manhattan":
            return self.calculator.manhattan_distance(first, second)
        raise SpatialValidationError("unsupported spatial metric", context={"metric": metric})

    def collision_state(self, a: Any, b: Any) -> CollisionResult:
        first = self._geometry(a)
        second = self._geometry(b)
        return self.collision.test(first, second)

    def intersects(self, a: Any, b: Any) -> bool:
        result = self.collision_state(a, b)
        if result.broad_phase_only and result.collides:
            raise SpatialGeometryError("only conservative broad-phase collision evidence is available for this geometry pair")
        return result.collides

    def bounds(self, value: Any):
        return geometry_bounds(self._geometry(value))

    def line_of_sight(self, a: Any, b: Any, obstacles: tuple[Any, ...] | list[Any] = ()) -> VisibilityResult:
        origin = self._position(a)
        target = self._position(b)
        if len(origin) != 3 or len(target) != 3:
            raise SpatialValidationError("line_of_sight requires 3D points")
        geometry_obstacles = [self._geometry(item) if isinstance(item, SpatialEntity) else item for item in obstacles]
        return visible_from(origin, target, geometry_obstacles)

    # ------------------------------------------------------------------
    # Coordinate-frame façade
    # ------------------------------------------------------------------

    def register_frame(
        self,
        frame_id: str,
        *,
        parent_id: str | None = None,
        transform_to_parent: RigidTransform | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Register a local frame in the subsystem frame graph."""
        self.frames.frames.add_frame(
            frame_id,
            parent_id=parent_id,
            transform_to_parent=transform_to_parent,
            metadata=dict(metadata or {}),
        )

    def set_frame_parent(self, frame_id: str, parent_id: str, transform_to_parent: RigidTransform) -> None:
        """Re-parent an existing frame using a validated rigid transform."""
        self.frames.frames.set_parent(frame_id, parent_id, transform_to_parent)

    def set_frame_transform(self, transform_to_parent: RigidTransform) -> None:
        """Update the transform from an existing child frame to its parent."""
        self.frames.frames.set_transform(transform_to_parent)

    def remove_frame(self, frame_id: str, *, recursive: bool = False) -> None:
        """Remove a non-root frame from the subsystem frame graph."""
        self.frames.frames.remove_frame(frame_id, recursive=recursive)

    def resolve_transform(self, source_frame: str, target_frame: str) -> RigidTransform:
        """Resolve a rigid transform across the current frame chain."""
        return self.frames.frames.lookup_transform(source_frame, target_frame)

    def transform_point(self, point: Any, source_frame: str, target_frame: str) -> np.ndarray:
        return self.frames.transform_point(point, source_frame, target_frame)

    def transform_points(self, points: Any, source_frame: str, target_frame: str) -> np.ndarray:
        return self.frames.transform_points(points, source_frame, target_frame)

    def transform_vector(self, vector: Any, source_frame: str, target_frame: str) -> np.ndarray:
        return self.resolve_transform(source_frame, target_frame).apply_vector(vector)

    def transform_pose(self, pose: RigidTransform, target_frame: str) -> RigidTransform:
        """Re-express a pose whose target is a known frame in ``target_frame``.

        ``pose`` maps an object/local frame into ``pose.target_frame``. The
        resolved frame-chain transform is composed after that pose.
        """
        if not isinstance(pose, RigidTransform):
            raise SpatialValidationError("pose must be a RigidTransform")
        if pose.target_frame == target_frame:
            return pose
        return pose.then(self.resolve_transform(pose.target_frame, target_frame))

    def transform_crs(self, coordinates: Any, source_crs: str, target_crs: str) -> np.ndarray:
        """Delegate an explicitly registered CRS conversion to the CRS registry."""
        return self.frames.crs.transform(coordinates, source_crs, target_crs)

    def frame_snapshot(self) -> tuple[dict[str, Any], ...]:
        """Return serializable frame metadata without exposing graph internals."""
        return tuple(
            {
                "frame_id": frame.frame_id,
                "parent_id": frame.parent_id,
                "metadata": dict(frame.metadata or {}),
            }
            for frame in self.frames.frames.frames()
        )

    # ------------------------------------------------------------------
    # Entity façade
    # ------------------------------------------------------------------

    def transform_entity(self, entity: SpatialEntity, target_frame: str) -> SpatialEntity:
        """Transform entity position between frames.

        Geometry is deliberately not transformed here. Therefore an entity
        carrying non-point geometry may only be safely transformed for
        position-only calculations. Geometry/topology callers must keep such
        entities in a common frame or explicitly transform their geometry in
        the owning geometry module.
        """
        if entity.dimension != 3:
            raise SpatialValidationError("rigid frame transforms require 3D entity coordinates")
        position = self.transform_point(entity.position, entity.frame_id, target_frame)
        return entity.moved(position, frame_id=target_frame)

    def _position(self, value: Any) -> Any:
        if isinstance(value, SpatialEntity):
            return value.position
        if isinstance(value, str):
            return self.memory.require_entity(value).position
        return value

    def _geometry(self, value: Any) -> Any:
        if isinstance(value, SpatialEntity):
            if value.geometry is None:
                raise SpatialValidationError("SpatialEntity has no geometry", context={"entity_id": value.entity_id})
            return value.geometry
        if isinstance(value, str):
            return self._geometry(self.memory.require_entity(value))
        return value


__all__ = ["SpatialCompute"]


if __name__ == "__main__":
    configure_logging()
    compute = SpatialCompute()
    printer.status(
        "TEST",
        f"distance={compute.distance((0, 0, 0), (1, 1, 1)):.6f}",
        "info",
    )
