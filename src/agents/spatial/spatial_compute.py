"""High-level deterministic spatial-computation façade.

SpatialCompute coordinates existing low-level modules; it does not duplicate
geometry, transform, collision, visibility or planning logic.
"""
from __future__ import annotations

__version__ = "2.3.0"

import numpy as np # type: ignore

from typing import Any

from .utils.config_loader import get_config_section, load_global_config
from .utils.spatial_errors import SpatialGeometryError, SpatialValidationError
from .world.collision import CollisionEngine, CollisionResult
from .world.geometry import geometry_bounds
from .world.visibility import VisibilityResult, visible_from
from .modules.calculations import Calculations
from .modules.coordinate_frames import CoordinateFrames
from .spatial_memory import SpatialMemory, get_default_spatial_memory
from .spatial_types import SpatialEntity
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Spatial Compute")
printer = PrettyPrinter()


class SpatialCompute:
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

    def transform_point(self, point: Any, source_frame: str, target_frame: str) -> np.ndarray:
        return self.frames.transform_point(point, source_frame, target_frame)

    def transform_entity(self, entity: SpatialEntity, target_frame: str) -> SpatialEntity:
        if entity.dimension != 3:
            raise SpatialValidationError("rigid frame transforms require 3D entity coordinates")
        position = self.transform_point(entity.position, entity.frame_id, target_frame)
        return entity.moved(position, frame_id=target_frame)

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
    printer.status("TEST", f"distance={compute.distance((0, 0, 0), (1, 1, 1)):.6f}", "info")
