"""Spatial map lifecycle, rigid registration, and map alignment.

This module deliberately stops at spatial representation/alignment. It does
not interpret raw sensors, choose navigation actions, perform exploration, or
own a full SLAM policy. Registration follows the rigid point-set foundations
of Horn (1987), Besl & McKay (1992), and Rusinkiewicz & Levoy (2001).
"""
from __future__ import annotations

__version__ = "2.3.0"

import numpy as np # type: ignore

from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Mapping as MappingType

from .transform import RigidTransform
from ..utils.config_loader import get_config_section, load_global_config
from ..utils.spatial_errors import SpatialMappingError, SpatialValidationError
from ..utils.spatial_helpers import DEFAULT_ABS_TOL, finite_array, to_json_safe, utc_now_iso, validate_identifier
from ..world.geometry import PointCloud, aabb_from_points
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Mapping")
printer = PrettyPrinter()


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    transform: RigidTransform
    rmse: float
    iterations: int
    converged: bool
    correspondences: int
    source_points: int
    target_points: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform": self.transform.to_dict(),
            "rmse": self.rmse,
            "iterations": self.iterations,
            "converged": self.converged,
            "correspondences": self.correspondences,
            "source_points": self.source_points,
            "target_points": self.target_points,
        }


@dataclass(frozen=True, slots=True)
class MapRecord:
    map_id: str
    frame_id: str
    representation: Any
    version: int = 1
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "map_id", validate_identifier(self.map_id, name="map_id"))
        object.__setattr__(self, "frame_id", validate_identifier(self.frame_id, name="frame_id"))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.version < 1:
            raise SpatialValidationError("map version must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "map_id": self.map_id,
            "frame_id": self.frame_id,
            "version": self.version,
            "created_at": self.created_at,
            "metadata": to_json_safe(self.metadata),
            "representation_type": type(self.representation).__name__,
        }


def best_fit_transform(
    source: Any,
    target: Any,
    *,
    source_frame: str = "source",
    target_frame: str = "target",
    weights: Any | None = None,
) -> RigidTransform:
    """Least-squares rigid transform using weighted Kabsch/SVD."""
    src = finite_array(source, name="source", ndim=2)
    dst = finite_array(target, name="target", ndim=2)
    if src.shape != dst.shape or src.shape[1] != 3:
        raise SpatialMappingError(
            "source and target must have matching shape (N, 3)",
            context={"source_shape": src.shape, "target_shape": dst.shape},
        )
    if src.shape[0] < 3:
        raise SpatialMappingError("at least three correspondences are required for rigid 3D registration")

    if weights is None:
        w = np.ones(src.shape[0], dtype=float)
    else:
        w = finite_array(weights, name="weights", ndim=1)
        if w.shape[0] != src.shape[0] or np.any(w < 0.0) or float(w.sum()) <= DEFAULT_ABS_TOL:
            raise SpatialMappingError("weights must be non-negative, non-zero, and match point count")
    w = w / float(w.sum())
    src_centroid = np.sum(src * w[:, None], axis=0)
    dst_centroid = np.sum(dst * w[:, None], axis=0)
    src_centered = src - src_centroid
    dst_centered = dst - dst_centroid
    covariance = (src_centered * w[:, None]).T @ dst_centered
    u, singular_values, vt = np.linalg.svd(covariance)
    if np.count_nonzero(singular_values > DEFAULT_ABS_TOL) < 2:
        raise SpatialMappingError("point configuration is degenerate for stable rigid registration")
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = dst_centroid - rotation @ src_centroid
    return RigidTransform(source_frame, target_frame, rotation, translation)


def _nearest_neighbors(source: np.ndarray, target: np.ndarray, *, chunk_size: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Exact nearest neighbours with bounded temporary memory."""
    if len(target) == 0:
        raise SpatialMappingError("target point set is empty")
    indices = np.empty(len(source), dtype=int)
    distances = np.empty(len(source), dtype=float)
    for start in range(0, len(source), chunk_size):
        stop = min(start + chunk_size, len(source))
        block = source[start:stop]
        squared = np.sum((block[:, None, :] - target[None, :, :]) ** 2, axis=2)
        local = np.argmin(squared, axis=1)
        indices[start:stop] = local
        distances[start:stop] = np.sqrt(squared[np.arange(len(block)), local])
    return indices, distances


def icp(
    source: Any,
    target: Any,
    *,
    source_frame: str = "source",
    target_frame: str = "target",
    initial: RigidTransform | None = None,
    max_iterations: int = 50,
    tolerance: float = 1.0e-6,
    max_correspondence_distance: float | None = None,
    trim_fraction: float = 1.0,
) -> AlignmentResult:
    """Deterministic point-to-point ICP with trimming and convergence guards."""
    src = finite_array(source, name="source", ndim=2)
    dst = finite_array(target, name="target", ndim=2)
    if src.shape[1:] != (3,) or dst.shape[1:] != (3,):
        raise SpatialMappingError("ICP point sets must have shape (N, 3)")
    if len(src) < 3 or len(dst) < 3:
        raise SpatialMappingError("ICP requires at least three source and three target points")
    if max_iterations < 1:
        raise SpatialMappingError("max_iterations must be >= 1")
    if tolerance <= 0.0:
        raise SpatialMappingError("tolerance must be positive")
    if not 0.0 < trim_fraction <= 1.0:
        raise SpatialMappingError("trim_fraction must lie in (0, 1]")
    if max_correspondence_distance is not None and max_correspondence_distance <= 0.0:
        raise SpatialMappingError("max_correspondence_distance must be positive")

    total = initial or RigidTransform.identity(source_frame)
    if total.source_frame != source_frame:
        raise SpatialMappingError("initial transform source frame does not match source_frame")
    if total.target_frame != target_frame:
        # Identity source-frame transform may be relabelled for initialisation.
        if initial is None:
            total = RigidTransform(source_frame, target_frame, np.eye(3), np.zeros(3))
        else:
            raise SpatialMappingError("initial transform target frame does not match target_frame")

    transformed = total.apply_points(src)
    previous_rmse = float("inf")
    converged = False
    used_correspondences = 0
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        neighbor_indices, distances = _nearest_neighbors(transformed, dst)
        mask = np.ones(len(transformed), dtype=bool)
        if max_correspondence_distance is not None:
            mask &= distances <= max_correspondence_distance
        candidate_indices = np.flatnonzero(mask)
        if candidate_indices.size < 3:
            raise SpatialMappingError(
                "insufficient ICP correspondences after distance filtering",
                context={"correspondences": int(candidate_indices.size)},
            )
        if trim_fraction < 1.0:
            keep = max(3, int(np.floor(candidate_indices.size * trim_fraction)))
            order = candidate_indices[np.argsort(distances[candidate_indices], kind="stable")[:keep]]
            mask[:] = False
            mask[order] = True

        matched_source = transformed[mask]
        matched_target = dst[neighbor_indices[mask]]
        used_correspondences = len(matched_source)
        incremental = best_fit_transform(
            matched_source,
            matched_target,
            source_frame=target_frame,
            target_frame=target_frame,
        )
        transformed = incremental.apply_points(transformed)
        total = total.then(incremental)

        residual = transformed[mask] - matched_target
        rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        if abs(previous_rmse - rmse) <= tolerance:
            converged = True
            previous_rmse = rmse
            break
        previous_rmse = rmse

    return AlignmentResult(
        transform=total,
        rmse=previous_rmse,
        iterations=iteration,
        converged=converged,
        correspondences=used_correspondences,
        source_points=len(src),
        target_points=len(dst),
    )


class Mapping:
    """Thread-safe lifecycle manager for spatial map representations."""

    def __init__(self) -> None:
        self.config = load_global_config()
        self.mapping_config = get_config_section("mapping", config=self.config, default={})
        self._lock = RLock()
        self._maps: dict[str, MapRecord] = {}

    def register(self, map_id: str, representation: Any, *, frame_id: str = "world", metadata: MappingType[str, Any] | None = None) -> MapRecord:
        key = validate_identifier(map_id, name="map_id")
        with self._lock:
            existing = self._maps.get(key)
            version = 1 if existing is None else existing.version + 1
            record = MapRecord(key, frame_id, representation, version=version, metadata=dict(metadata or {}))
            self._maps[key] = record
            return record

    def get(self, map_id: str) -> MapRecord:
        key = validate_identifier(map_id, name="map_id")
        with self._lock:
            try:
                return self._maps[key]
            except KeyError as exc:
                raise SpatialMappingError("unknown map", context={"map_id": key}) from exc

    def remove(self, map_id: str) -> MapRecord:
        key = validate_identifier(map_id, name="map_id")
        with self._lock:
            try:
                return self._maps.pop(key)
            except KeyError as exc:
                raise SpatialMappingError("unknown map", context={"map_id": key}) from exc

    def align_point_clouds(self, source: PointCloud, target: PointCloud, **kwargs: Any) -> AlignmentResult:
        return icp(
            source.points,
            target.points,
            source_frame=source.frame_id,
            target_frame=target.frame_id,
            **kwargs,
        )

    @staticmethod
    def merge_point_clouds(source: PointCloud, target: PointCloud, transform: RigidTransform, *, deduplicate_tolerance: float = 0.0) -> PointCloud:
        if transform.source_frame != source.frame_id or transform.target_frame != target.frame_id:
            raise SpatialMappingError("transform frames do not match point-cloud frames")
        transformed = transform.apply_points(source.points)
        points = np.vstack((target.points, transformed))
        if deduplicate_tolerance > 0.0 and len(points):
            scale = 1.0 / deduplicate_tolerance
            quantized = np.rint(points * scale).astype(np.int64)
            _, unique_indices = np.unique(quantized, axis=0, return_index=True)
            points = points[np.sort(unique_indices)]
        return PointCloud(points, frame_id=target.frame_id)

    def summary(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(record.to_dict() for record in self._maps.values())


__all__ = [
    "AlignmentResult",
    "MapRecord",
    "best_fit_transform",
    "icp",
    "Mapping",
]


if __name__ == "__main__":
    configure_logging()
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    result = icp(points, points + np.array([1.0, 2.0, 3.0]), source_frame="a", target_frame="b")
    printer.status("TEST", f"ICP rmse={result.rmse:.8f}", "info")
