"""Time-aware conservative spatial queries over an external evaluated scene.

Spatial consumes a read-only evaluation protocol; it never imports SGEO or
owns animation/project state. AABB separation is a proven *lower bound* for
shape distance. Overlapping AABBs are only potential collisions, including
for meshes and analytic curved geometry. Interval queries sample discrete
times and never purport to be continuous collision detection.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

import numpy as np

from .world.collision import CollisionEngine, aabb_separation
from .world.geometry import AABB, Sphere
from .utils.spatial_errors import SpatialValidationError


class EvaluatedSceneProtocol(Protocol):
    time: float
    revision: int

    def ids(self, *, include_hidden: bool = False) -> tuple[str, ...]: ...
    def geometric_ids(self) -> tuple[str, ...]: ...
    def get_object_state(self, object_id: str) -> dict: ...
    def world_bounds(self, object_id: str, *, include_hidden: bool = False): ...


class SceneEvaluatorProtocol(Protocol):
    def evaluate(self, time: float) -> EvaluatedSceneProtocol: ...


@dataclass(frozen=True, slots=True)
class TemporalProximity:
    object_a: str
    object_b: str
    time: float
    revision: int
    separation_lower_bound: float
    classification: str

    @property
    def potential_collision(self) -> bool:
        return self.classification == "potential_overlap"


@dataclass(frozen=True, slots=True)
class TemporalIntervalResult:
    object_a: str
    object_b: str
    start: float
    end: float
    revision: int
    samples: tuple[TemporalProximity, ...]
    continuous_verified: bool = False

    @property
    def status(self) -> str:
        return ("potential_at_samples" if any(s.potential_collision for s in self.samples)
                else "clear_at_samples_only")


@dataclass(frozen=True, slots=True)
class TemporalCollision:
    object_a: str
    object_b: str
    time: float
    revision: int
    classification: str
    method: str
    separation_lower_bound: float


class TemporalSpatialQueries:
    """Bounded, typed temporal broad-phase operations for animated scenes."""

    def __init__(self, evaluator: SceneEvaluatorProtocol, *, collision: CollisionEngine | None = None) -> None:
        if not callable(getattr(evaluator, "evaluate", None)):
            raise SpatialValidationError("temporal queries require a scene evaluator")
        self.evaluator = evaluator
        self.collision = collision or CollisionEngine()

    @staticmethod
    def _time(value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not isfinite(value) or value < 0 or value > 1e9:
            raise SpatialValidationError("time must be finite, nonnegative seconds <= 1e9")
        return float(value)

    @staticmethod
    def _box(state: EvaluatedSceneProtocol, oid: str) -> AABB:
        lo, hi = state.world_bounds(oid)
        return AABB(np.asarray(lo, dtype=float), np.asarray(hi, dtype=float))

    def get_object_state(self, object_id: str, time: float) -> dict:
        return self.evaluator.evaluate(self._time(time)).get_object_state(object_id)

    def get_bounds_at(self, object_id: str, time: float):
        return self.evaluator.evaluate(self._time(time)).world_bounds(object_id)

    def _proximity(self, state: EvaluatedSceneProtocol, a: str, b: str) -> TemporalProximity:
        if a == b:
            raise SpatialValidationError("temporal proximity requires distinct objects")
        distance = aabb_separation(self._box(state, a), self._box(state, b))
        return TemporalProximity(a, b, state.time, state.revision, distance,
                                 "potential_overlap" if distance == 0.0 else "verified_separation")

    def get_distance_at(self, a: str, b: str, time: float) -> TemporalProximity:
        """Return a conservative lower bound, NOT exact surface distance."""
        return self._proximity(self.evaluator.evaluate(self._time(time)), a, b)

    @staticmethod
    def _supported_primitive(state: EvaluatedSceneProtocol, oid: str):
        """Produce an exact Spatial primitive only when SGEO transform permits it."""
        record = state.get_object_state(oid)
        geo = record["object"].get("geometry")
        if not isinstance(geo, dict):
            return None
        matrix = np.asarray(record["world_transform"], dtype=float).reshape(4, 4)
        if geo.get("kind") == "box":
            if not np.allclose(matrix[:3, :3], np.eye(3), rtol=0, atol=1e-12):
                return None
            return TemporalSpatialQueries._box(state, oid)
        if geo.get("kind") == "sphere":
            rotation = matrix[:3, :3]
            if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-12):
                return None
            if not np.isclose(np.linalg.det(rotation), 1.0, rtol=0, atol=1e-12):
                return None
            center = rotation @ np.asarray(geo["center"], dtype=float) + matrix[:3, 3]
            return Sphere(center, float(geo["radius"]))
        return None

    def collision_at(self, a: str, b: str, time: float) -> TemporalCollision:
        """Use existing exact primitive collision engine when representation permits.

        Rotated boxes, meshes and arbitrary B-reps remain conservative only.
        """
        state = self.evaluator.evaluate(self._time(time))
        broad = self._proximity(state, a, b)
        if not broad.potential_collision:
            return TemporalCollision(a, b, state.time, state.revision,
                                     "verified_separation", "aabb_lower_bound", broad.separation_lower_bound)
        first = self._supported_primitive(state, a)
        second = self._supported_primitive(state, b)
        if first is None or second is None:
            return TemporalCollision(a, b, state.time, state.revision,
                                     "broad_phase_potential", "world_aabb", 0.0)
        result = self.collision.test(first, second)
        if result.broad_phase_only:
            return TemporalCollision(a, b, state.time, state.revision,
                                     "broad_phase_potential", "collision_engine_broad_phase", 0.0)
        return TemporalCollision(a, b, state.time, state.revision,
                                 "verified_collision" if result.collides else "verified_separation",
                                 "collision_engine_primitive", max(0.0, result.distance))

    def find_intersections_at(self, time: float, *, max_objects: int = 256) -> tuple[TemporalProximity, ...]:
        """Return broad-phase *candidates*; never claim exact intersections."""
        if type(max_objects) is not int or not 1 <= max_objects <= 256:
            raise SpatialValidationError("max_objects must be 1..256")
        state = self.evaluator.evaluate(self._time(time))
        ids = state.geometric_ids()
        if len(ids) > max_objects:
            raise SpatialValidationError("temporal broad-phase object budget exceeded")
        boxes = {oid: self._box(state, oid) for oid in ids}
        result = []
        for i, a in enumerate(ids):
            for b in ids[i+1:]:
                if aabb_separation(boxes[a], boxes[b]) == 0.0:
                    result.append(TemporalProximity(a, b, state.time, state.revision, 0.0,
                                                    "potential_overlap"))
        return tuple(result)

    def query_collision_interval(self, a: str, b: str, start: float, end: float,
                                 *, samples: int = 33) -> TemporalIntervalResult:
        """Discrete AABB sampling, never continuous collision verification."""
        start, end = self._time(start), self._time(end)
        if end < start:
            raise SpatialValidationError("interval end precedes start")
        if type(samples) is not int or not 2 <= samples <= 1025:
            raise SpatialValidationError("samples must be 2..1025")
        checked = tuple(self.get_distance_at(a, b, start + (end-start)*i/(samples-1))
                        for i in range(samples))
        if len({sample.revision for sample in checked}) != 1:
            raise SpatialValidationError("scene revision changed during temporal interval evaluation")
        return TemporalIntervalResult(a, b, start, end, checked[0].revision, checked)


__all__ = ["EvaluatedSceneProtocol", "SceneEvaluatorProtocol", "TemporalProximity",
           "TemporalIntervalResult", "TemporalCollision", "TemporalSpatialQueries"]
