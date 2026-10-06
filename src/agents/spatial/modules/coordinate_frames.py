"""Coordinate frames, rigid transforms, and lightweight CRS support.

The frame graph follows the transform-tree idea popularised by ROS tf: each
frame has at most one parent and stores a rigid transform from child to parent.
This guarantees an unambiguous chain while preventing cycles. Geographic
support is deliberately small: named CRS metadata plus WGS84 geodetic/ECEF
conversion and explicit transform registration. It is not a replacement for a
full GIS projection library.
"""
from __future__ import annotations

__version__ = "2.3.0"

import numpy as np # type: ignore

from dataclasses import dataclass
from math import atan2, cos, radians, sin, sqrt
from threading import RLock
from typing import Any, Callable

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.spatial_errors import SpatialFrameError, SpatialValidationError
from ..utils.spatial_helpers import *
from .transform import RigidTransform
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Coordinate Frames")
printer = PrettyPrinter()

# WGS84 constants (Snyder / EPSG conventional values)
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)


@dataclass(frozen=True, slots=True)
class CoordinateReferenceSystem:
    crs_id: str
    name: str
    kind: str = "cartesian"
    authority: str | None = None
    code: str | None = None
    units: tuple[str, ...] = ("m", "m", "m")
    axis_order: tuple[str, ...] = ("x", "y", "z")

    def __post_init__(self) -> None:
        object.__setattr__(self, "crs_id", validate_identifier(self.crs_id, name="crs_id"))
        if not self.name.strip():
            raise SpatialValidationError("CRS name must not be empty")


@dataclass(frozen=True, slots=True)
class CoordinateFrame:
    frame_id: str
    parent_id: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", validate_identifier(self.frame_id, name="frame_id"))
        if self.parent_id is not None:
            object.__setattr__(self, "parent_id", validate_identifier(self.parent_id, name="parent_id"))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


class FrameGraph:
    """Thread-safe tree of local rigid coordinate frames."""

    def __init__(self, root_frame: str = "world") -> None:
        root = validate_identifier(root_frame, name="root_frame")
        self._lock = RLock()
        self._frames: dict[str, CoordinateFrame] = {root: CoordinateFrame(root)}
        self._to_parent: dict[str, RigidTransform] = {}
        self.root_frame = root

    def has_frame(self, frame_id: str) -> bool:
        with self._lock:
            return frame_id in self._frames

    def add_frame(
        self,
        frame_id: str,
        *,
        parent_id: str | None = None,
        transform_to_parent: RigidTransform | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        frame = validate_identifier(frame_id, name="frame_id")
        parent = self.root_frame if parent_id is None and frame != self.root_frame else parent_id
        with self._lock:
            if frame in self._frames:
                raise SpatialFrameError("frame already exists", context={"frame_id": frame})
            if parent is not None and parent not in self._frames:
                raise SpatialFrameError("parent frame is unknown", context={"parent_id": parent})
            if parent == frame:
                raise SpatialFrameError("a frame cannot be its own parent", context={"frame_id": frame})
            if parent is not None:
                if transform_to_parent is None:
                    transform_to_parent = RigidTransform(frame, parent, np.eye(3), np.zeros(3))
                self._validate_edge(frame, parent, transform_to_parent)
            self._frames[frame] = CoordinateFrame(frame, parent, metadata)
            if transform_to_parent is not None:
                self._to_parent[frame] = transform_to_parent

    def set_parent(self, frame_id: str, parent_id: str, transform_to_parent: RigidTransform) -> None:
        frame = validate_identifier(frame_id, name="frame_id")
        parent = validate_identifier(parent_id, name="parent_id")
        with self._lock:
            self._require_frame(frame)
            self._require_frame(parent)
            if frame == self.root_frame:
                raise SpatialFrameError("root frame cannot be re-parented")
            if frame == parent or self._is_descendant(parent, frame):
                raise SpatialFrameError(
                    "frame re-parenting would create a cycle",
                    context={"frame_id": frame, "parent_id": parent},
                )
            self._validate_edge(frame, parent, transform_to_parent)
            current = self._frames[frame]
            self._frames[frame] = CoordinateFrame(frame, parent, current.metadata)
            self._to_parent[frame] = transform_to_parent

    def set_transform(self, transform_to_parent: RigidTransform) -> None:
        with self._lock:
            child = transform_to_parent.source_frame
            self._require_frame(child)
            frame = self._frames[child]
            if frame.parent_id is None:
                raise SpatialFrameError("root frame has no parent transform", context={"frame_id": child})
            self._validate_edge(child, frame.parent_id, transform_to_parent)
            self._to_parent[child] = transform_to_parent

    def frame(self, frame_id: str) -> CoordinateFrame:
        with self._lock:
            self._require_frame(frame_id)
            return self._frames[frame_id]

    def remove_frame(self, frame_id: str, *, recursive: bool = False) -> None:
        frame = validate_identifier(frame_id, name="frame_id")
        with self._lock:
            self._require_frame(frame)
            if frame == self.root_frame:
                raise SpatialFrameError("root frame cannot be removed")
            children = [item.frame_id for item in self._frames.values() if item.parent_id == frame]
            if children and not recursive:
                raise SpatialFrameError("frame has children", context={"frame_id": frame, "children": children})
            for child in children:
                self.remove_frame(child, recursive=True)
            self._frames.pop(frame, None)
            self._to_parent.pop(frame, None)

    def lookup_transform(self, source_frame: str, target_frame: str) -> RigidTransform:
        source = validate_identifier(source_frame, name="source_frame")
        target = validate_identifier(target_frame, name="target_frame")
        with self._lock:
            self._require_frame(source)
            self._require_frame(target)
            if source == target:
                return RigidTransform.identity(source)

            source_chain = self._chain_to_root(source)
            target_chain = self._chain_to_root(target)
            target_set = set(target_chain)
            common = next((frame for frame in source_chain if frame in target_set), None)
            if common is None:
                raise SpatialFrameError(
                    "frames do not share a common ancestor",
                    context={"source_frame": source, "target_frame": target},
                )
            source_to_common = self._to_ancestor(source, common)
            target_to_common = self._to_ancestor(target, common)
            return source_to_common.then(target_to_common.inverse())

    def transform_point(self, point: Any, source_frame: str, target_frame: str) -> np.ndarray:
        return self.lookup_transform(source_frame, target_frame).apply_point(point)

    def transform_points(self, points: Any, source_frame: str, target_frame: str) -> np.ndarray:
        return self.lookup_transform(source_frame, target_frame).apply_points(points)

    def frames(self) -> tuple[CoordinateFrame, ...]:
        with self._lock:
            return tuple(self._frames.values())

    def _require_frame(self, frame_id: str) -> None:
        if frame_id not in self._frames:
            raise SpatialFrameError("unknown coordinate frame", context={"frame_id": frame_id})

    @staticmethod
    def _validate_edge(child: str, parent: str, transform: RigidTransform) -> None:
        if transform.source_frame != child or transform.target_frame != parent:
            raise SpatialFrameError(
                "transform must map child coordinates to parent coordinates",
                context={
                    "child": child,
                    "parent": parent,
                    "transform": f"{transform.source_frame}->{transform.target_frame}",
                },
            )

    def _is_descendant(self, candidate: str, ancestor: str) -> bool:
        current: str | None = candidate
        while current is not None:
            if current == ancestor:
                return True
            frame = self._frames.get(current)
            current = frame.parent_id if frame else None
        return False

    def _chain_to_root(self, frame_id: str) -> list[str]:
        chain: list[str] = []
        current: str | None = frame_id
        visited: set[str] = set()
        while current is not None:
            if current in visited:
                raise SpatialFrameError("cycle detected in frame graph", context={"frame_id": current})
            visited.add(current)
            chain.append(current)
            current = self._frames[current].parent_id
        return chain

    def _to_ancestor(self, frame_id: str, ancestor: str) -> RigidTransform:
        transform = RigidTransform.identity(frame_id)
        current = frame_id
        while current != ancestor:
            edge = self._to_parent.get(current)
            if edge is None:
                raise SpatialFrameError("missing parent transform", context={"frame_id": current})
            transform = transform.then(edge)
            current = edge.target_frame
        return transform


class CRSRegistry:
    """Small registry for explicit CRS conversions plus WGS84 geodetic/ECEF."""

    WGS84_GEODETIC = "EPSG:4979"
    WGS84_ECEF = "EPSG:4978"

    def __init__(self) -> None:
        self._crs: dict[str, CoordinateReferenceSystem] = {}
        self._transforms: dict[tuple[str, str], Callable[[np.ndarray], np.ndarray]] = {}
        self.register_crs(CoordinateReferenceSystem(self.WGS84_GEODETIC, "WGS 84 3D", "geographic", "EPSG", "4979", ("deg", "deg", "m"), ("lat", "lon", "h")))
        self.register_crs(CoordinateReferenceSystem(self.WGS84_ECEF, "WGS 84 geocentric", "geocentric", "EPSG", "4978"))
        self.register_transform(self.WGS84_GEODETIC, self.WGS84_ECEF, self._geodetic_array_to_ecef)
        self.register_transform(self.WGS84_ECEF, self.WGS84_GEODETIC, self._ecef_array_to_geodetic)

    def register_crs(self, crs: CoordinateReferenceSystem) -> None:
        self._crs[crs.crs_id] = crs

    def register_transform(
        self,
        source_crs: str,
        target_crs: str,
        transform: Callable[[np.ndarray], np.ndarray],
        *,
        inverse: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        if source_crs not in self._crs or target_crs not in self._crs:
            raise SpatialFrameError("both CRS definitions must be registered before their transform")
        if not callable(transform):
            raise SpatialValidationError("CRS transform must be callable")
        self._transforms[(source_crs, target_crs)] = transform
        if inverse is not None:
            self._transforms[(target_crs, source_crs)] = inverse

    def transform(self, coordinates: Any, source_crs: str, target_crs: str) -> np.ndarray:
        array = finite_array(coordinates, name="coordinates")
        if array.shape[-1] != 3:
            raise SpatialFrameError("CRS coordinates must have a final dimension of 3")
        if source_crs == target_crs:
            return array.copy()
        function = self._transforms.get((source_crs, target_crs))
        if function is None:
            raise SpatialFrameError(
                "no registered CRS transform",
                context={"source_crs": source_crs, "target_crs": target_crs},
            )
        result = np.asarray(function(array), dtype=float)
        if result.shape != array.shape or not np.all(np.isfinite(result)):
            raise SpatialFrameError("CRS transform returned malformed coordinates")
        return result

    @staticmethod
    def geodetic_to_ecef(latitude_deg: float, longitude_deg: float, height_m: float = 0.0) -> np.ndarray:
        lat = radians(float(latitude_deg))
        lon = radians(float(longitude_deg))
        h = float(height_m)
        n = _WGS84_A / sqrt(1.0 - _WGS84_E2 * sin(lat) ** 2)
        x = (n + h) * cos(lat) * cos(lon)
        y = (n + h) * cos(lat) * sin(lon)
        z = (n * (1.0 - _WGS84_E2) + h) * sin(lat)
        return np.array([x, y, z], dtype=float)

    @staticmethod
    def ecef_to_geodetic(x: float, y: float, z: float) -> np.ndarray:
        x, y, z = float(x), float(y), float(z)
        p = sqrt(x * x + y * y)
        if p < 1.0e-12:
            latitude = np.pi / 2.0 if z >= 0.0 else -np.pi / 2.0
            longitude = 0.0
            b = _WGS84_A * (1.0 - _WGS84_F)
            height = abs(z) - b
            return np.degrees([latitude, longitude, 0.0]) + np.array([0.0, 0.0, height])
        longitude = atan2(y, x)
        latitude = atan2(z, p * (1.0 - _WGS84_E2))
        height = 0.0
        for _ in range(10):
            n = _WGS84_A / sqrt(1.0 - _WGS84_E2 * sin(latitude) ** 2)
            height = p / cos(latitude) - n
            new_latitude = atan2(z, p * (1.0 - _WGS84_E2 * n / (n + height)))
            if abs(new_latitude - latitude) < 1.0e-13:
                latitude = new_latitude
                break
            latitude = new_latitude
        return np.array([np.degrees(latitude), np.degrees(longitude), height], dtype=float)

    @classmethod
    def _geodetic_array_to_ecef(cls, coordinates: np.ndarray) -> np.ndarray:
        flat = coordinates.reshape(-1, 3)
        result = np.vstack([cls.geodetic_to_ecef(*row) for row in flat])
        return result.reshape(coordinates.shape)

    @classmethod
    def _ecef_array_to_geodetic(cls, coordinates: np.ndarray) -> np.ndarray:
        flat = coordinates.reshape(-1, 3)
        result = np.vstack([cls.ecef_to_geodetic(*row) for row in flat])
        return result.reshape(coordinates.shape)


class CoordinateFrames:
    """Spatial-facing façade combining local frames and CRS transformations."""

    def __init__(self, root_frame: str = "world") -> None:
        self.config = load_global_config()
        self.frame_config = get_config_section("coordinate_frames", config=self.config, default={})
        self.frames = FrameGraph(root_frame=root_frame)
        self.crs = CRSRegistry()

    def transform_point(self, point: Any, source_frame: str, target_frame: str) -> np.ndarray:
        return self.frames.transform_point(point, source_frame, target_frame)

    def transform_points(self, points: Any, source_frame: str, target_frame: str) -> np.ndarray:
        return self.frames.transform_points(points, source_frame, target_frame)


__all__ = [
    "CoordinateReferenceSystem",
    "CoordinateFrame",
    "FrameGraph",
    "CRSRegistry",
    "CoordinateFrames",
]


if __name__ == "__main__":
    configure_logging()
    graph = FrameGraph("world")
    graph.add_frame("camera", parent_id="world", transform_to_parent=RigidTransform("camera", "world", np.eye(3), np.array([1.0, 0.0, 0.0])))
    printer.status("TEST", f"camera origin in world={graph.transform_point((0, 0, 0), 'camera', 'world').tolist()}", "info")
