"""Occupancy and voxelized world representations.

The module consumes already interpreted spatial observations. It implements
probabilistic occupancy state (Elfes), sparse voxel/octree storage (Hornung et
al.) and lightweight TSDF/ESDF containers (Curless & Levoy; Oleynikova et al.)
without performing sensor perception or navigation.
"""
from __future__ import annotations

__version__ = "2.3.0"

import numpy as np # type: ignore

from dataclasses import dataclass
from enum import Enum
from math import log
from typing import Any, Iterable

from ..utils.spatial_errors import SpatialOccupancyError
from ..utils.spatial_helpers import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Occupancy")
printer = PrettyPrinter()


class OccupancyState(str, Enum):
    UNKNOWN = "unknown"
    FREE = "free"
    OCCUPIED = "occupied"
    UNCERTAIN = "uncertain"


def probability_to_log_odds(probability: float) -> float:
    p = clamp(float(probability), 1.0e-9, 1.0 - 1.0e-9)
    return log(p / (1.0 - p))


def log_odds_to_probability(log_odds: float) -> float:
    value = float(log_odds)
    if value >= 0.0:
        exp_neg = np.exp(-value)
        return float(1.0 / (1.0 + exp_neg))
    exp_pos = np.exp(value)
    return float(exp_pos / (1.0 + exp_pos))


class OccupancyGrid2D:
    def __init__(
        self,
        width: int,
        height: int,
        *,
        resolution: float = 1.0,
        origin: Any = (0.0, 0.0),
        free_threshold: float = 0.35,
        occupied_threshold: float = 0.65,
    ) -> None:
        if width <= 0 or height <= 0:
            raise SpatialOccupancyError("width and height must be positive")
        self.width = int(width)
        self.height = int(height)
        self.resolution = validate_positive(resolution, name="resolution")
        self.origin = finite_vector(origin, name="origin", dimension=2)
        self.free_threshold = float(free_threshold)
        self.occupied_threshold = float(occupied_threshold)
        if not 0.0 <= self.free_threshold < self.occupied_threshold <= 1.0:
            raise SpatialOccupancyError("occupancy thresholds must satisfy 0 <= free < occupied <= 1")
        self._log_odds = np.full((self.height, self.width), np.nan, dtype=float)

    def world_to_grid(self, point: Any) -> tuple[int, int]:
        p = finite_vector(point, name="point", dimension=2)
        xy = np.floor((p - self.origin) / self.resolution).astype(int)
        x, y = int(xy[0]), int(xy[1])
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise SpatialOccupancyError("world point lies outside occupancy grid", context={"point": p.tolist()})
        return x, y

    def grid_to_world(self, x: int, y: int, *, center: bool = True) -> np.ndarray:
        self._validate_index(x, y)
        offset = 0.5 if center else 0.0
        return self.origin + np.array([x + offset, y + offset], dtype=float) * self.resolution

    def probability(self, x: int, y: int) -> float | None:
        self._validate_index(x, y)
        value = self._log_odds[y, x]
        return None if np.isnan(value) else log_odds_to_probability(float(value))

    def set_probability(self, x: int, y: int, probability: float) -> None:
        self._validate_index(x, y)
        self._log_odds[y, x] = probability_to_log_odds(probability)

    def update_probability(self, x: int, y: int, observation_probability: float, *, prior_probability: float = 0.5) -> float:
        """Log-odds Bayesian occupancy update using an interpreted observation."""
        self._validate_index(x, y)
        observation = probability_to_log_odds(observation_probability)
        prior = probability_to_log_odds(prior_probability)
        current = self._log_odds[y, x]
        current = prior if np.isnan(current) else float(current)
        updated = float(np.clip(current + observation - prior, -30.0, 30.0))
        self._log_odds[y, x] = updated
        return log_odds_to_probability(updated)

    def state(self, x: int, y: int) -> OccupancyState:
        probability = self.probability(x, y)
        if probability is None:
            return OccupancyState.UNKNOWN
        if probability <= self.free_threshold:
            return OccupancyState.FREE
        if probability >= self.occupied_threshold:
            return OccupancyState.OCCUPIED
        return OccupancyState.UNCERTAIN

    def occupied_cells(self) -> np.ndarray:
        probabilities = np.full_like(self._log_odds, np.nan)
        mask = ~np.isnan(self._log_odds)
        probabilities[mask] = 1.0 / (1.0 + np.exp(-self._log_odds[mask]))
        return np.argwhere(probabilities >= self.occupied_threshold)[:, ::-1]

    def clear(self) -> None:
        self._log_odds.fill(np.nan)

    def _validate_index(self, x: int, y: int) -> None:
        if not (0 <= int(x) < self.width and 0 <= int(y) < self.height):
            raise SpatialOccupancyError("grid index is out of bounds", context={"x": x, "y": y})


class VoxelGrid:
    """Sparse probabilistic 3D voxel grid."""

    def __init__(
        self,
        *,
        resolution: float = 1.0,
        origin: Any = (0.0, 0.0, 0.0),
        free_threshold: float = 0.35,
        occupied_threshold: float = 0.65,
    ) -> None:
        self.resolution = validate_positive(resolution, name="resolution")
        self.origin = finite_vector(origin, name="origin", dimension=3)
        self.free_threshold = float(free_threshold)
        self.occupied_threshold = float(occupied_threshold)
        if not 0.0 <= self.free_threshold < self.occupied_threshold <= 1.0:
            raise SpatialOccupancyError("occupancy thresholds must satisfy 0 <= free < occupied <= 1")
        self._log_odds: dict[tuple[int, int, int], float] = {}

    def world_to_voxel(self, point: Any) -> tuple[int, int, int]:
        p = finite_vector(point, name="point", dimension=3)
        index = np.floor((p - self.origin) / self.resolution).astype(int)
        return int(index[0]), int(index[1]), int(index[2])

    def voxel_to_world(self, index: tuple[int, int, int], *, center: bool = True) -> np.ndarray:
        if len(index) != 3:
            raise SpatialOccupancyError("voxel index must have three coordinates")
        offset = 0.5 if center else 0.0
        return self.origin + (np.asarray(index, dtype=float) + offset) * self.resolution

    def probability(self, index: tuple[int, int, int]) -> float | None:
        key = (int(index[0]), int(index[1]), int(index[2]))
        value = self._log_odds.get(key)
        return None if value is None else log_odds_to_probability(value)

    def set_probability(self, index: tuple[int, int, int], probability: float) -> None:
        key = (int(index[0]), int(index[1]), int(index[2]))
        self._log_odds[key] = probability_to_log_odds(probability)

    def update_probability(self, index: tuple[int, int, int], observation_probability: float, *, prior_probability: float = 0.5) -> float:
        key: tuple[int, int, int] = (int(index[0]), int(index[1]), int(index[2]))
        observation = probability_to_log_odds(observation_probability)
        prior = probability_to_log_odds(prior_probability)
        current = self._log_odds.get(key, prior)
        updated = float(np.clip(current + observation - prior, -30.0, 30.0))
        self._log_odds[key] = updated
        return log_odds_to_probability(updated)

    def state(self, index: tuple[int, int, int]) -> OccupancyState:
        probability = self.probability(index)
        if probability is None:
            return OccupancyState.UNKNOWN
        if probability <= self.free_threshold:
            return OccupancyState.FREE
        if probability >= self.occupied_threshold:
            return OccupancyState.OCCUPIED
        return OccupancyState.UNCERTAIN

    def occupied_voxels(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(index for index, value in self._log_odds.items() if log_odds_to_probability(value) >= self.occupied_threshold)

    def __len__(self) -> int:
        return len(self._log_odds)


OccupancyGrid3D = VoxelGrid


class OctreeOccupancy:
    """Sparse fixed-depth octree addressing over a bounded cubic world volume.

    Leaves are stored sparsely by Morton-style path tuples. The class provides
    octree spatial semantics without pretending to be a full OctoMap clone.
    """

    def __init__(self, minimum: Any, maximum: Any, *, max_depth: int = 8) -> None:
        self.minimum = finite_vector(minimum, name="minimum", dimension=3)
        self.maximum = finite_vector(maximum, name="maximum", dimension=3)
        if np.any(self.minimum >= self.maximum):
            raise SpatialOccupancyError("octree minimum must be strictly smaller than maximum")
        if max_depth < 1 or max_depth > 24:
            raise SpatialOccupancyError("max_depth must lie in [1, 24]")
        self.max_depth = int(max_depth)
        self._leaves: dict[tuple[int, ...], float] = {}

    def path_for_point(self, point: Any) -> tuple[int, ...]:
        p = finite_vector(point, name="point", dimension=3)
        if np.any(p < self.minimum) or np.any(p > self.maximum):
            raise SpatialOccupancyError("point lies outside octree bounds")
        lo = self.minimum.copy()
        hi = self.maximum.copy()
        path: list[int] = []
        for _ in range(self.max_depth):
            mid = (lo + hi) * 0.5
            bits = (p >= mid).astype(int)
            child = int(bits[0] | (bits[1] << 1) | (bits[2] << 2))
            path.append(child)
            lo = np.where(bits, mid, lo)
            hi = np.where(bits, hi, mid)
        return tuple(path)

    def set_probability(self, point: Any, probability: float) -> tuple[int, ...]:
        path = self.path_for_point(point)
        self._leaves[path] = probability_to_log_odds(probability)
        return path

    def update_probability(self, point: Any, observation_probability: float, *, prior_probability: float = 0.5) -> float:
        path = self.path_for_point(point)
        prior = probability_to_log_odds(prior_probability)
        current = self._leaves.get(path, prior)
        updated = float(np.clip(current + probability_to_log_odds(observation_probability) - prior, -30.0, 30.0))
        self._leaves[path] = updated
        return log_odds_to_probability(updated)

    def probability(self, point: Any) -> float | None:
        value = self._leaves.get(self.path_for_point(point))
        return None if value is None else log_odds_to_probability(value)

    def __len__(self) -> int:
        return len(self._leaves)


@dataclass(slots=True)
class TSDFVoxel:
    distance: float
    weight: float


class TSDFVolume:
    def __init__(self, *, voxel_size: float = 0.1, truncation_distance: float = 0.3, origin: Any = (0.0, 0.0, 0.0)) -> None:
        self.voxel_size = validate_positive(voxel_size, name="voxel_size")
        self.truncation_distance = validate_positive(truncation_distance, name="truncation_distance")
        self.origin = finite_vector(origin, name="origin", dimension=3)
        self._voxels: dict[tuple[int, int, int], TSDFVoxel] = {}

    def world_to_voxel(self, point: Any) -> tuple[int, int, int]:
        index = np.floor((finite_vector(point, name="point", dimension=3) - self.origin) / self.voxel_size).astype(int)
        return (int(index[0]), int(index[1]), int(index[2]))

    def integrate(self, index: tuple[int, int, int], signed_distance: float, weight: float = 1.0) -> TSDFVoxel:
        if weight <= 0.0:
            raise SpatialOccupancyError("TSDF integration weight must be positive")
        distance = float(np.clip(float(signed_distance), -self.truncation_distance, self.truncation_distance))
        previous = self._voxels.get(index)
        if previous is None:
            voxel = TSDFVoxel(distance, float(weight))
        else:
            total = previous.weight + float(weight)
            voxel = TSDFVoxel((previous.distance * previous.weight + distance * weight) / total, total)
        self._voxels[index] = voxel
        return voxel

    def get(self, index: tuple[int, int, int]) -> TSDFVoxel | None:
        return self._voxels.get(index)


class ESDFVolume:
    def __init__(self, *, voxel_size: float = 0.1, origin: Any = (0.0, 0.0, 0.0)) -> None:
        self.voxel_size = validate_positive(voxel_size, name="voxel_size")
        self.origin = finite_vector(origin, name="origin", dimension=3)
        self._distance: dict[tuple[int, int, int], float] = {}

    def set_distance(self, index: tuple[int, int, int], distance: float) -> None:
        value = float(distance)
        if not np.isfinite(value):
            raise SpatialOccupancyError("ESDF distance must be finite")
        key = (int(index[0]), int(index[1]), int(index[2]))
        self._distance[key] = value

    def distance(self, index: tuple[int, int, int]) -> float | None:
        key = (int(index[0]), int(index[1]), int(index[2]))
        return self._distance.get(key)


__all__ = [
    "OccupancyState",
    "probability_to_log_odds",
    "log_odds_to_probability",
    "OccupancyGrid2D",
    "VoxelGrid",
    "OccupancyGrid3D",
    "OctreeOccupancy",
    "TSDFVoxel",
    "TSDFVolume",
    "ESDFVolume",
]


if __name__ == "__main__":
    configure_logging()
    grid = OccupancyGrid2D(10, 10, resolution=0.5)
    grid.update_probability(1, 1, 0.8)
    printer.status("TEST", f"state={grid.state(1, 1).value}", "info")
