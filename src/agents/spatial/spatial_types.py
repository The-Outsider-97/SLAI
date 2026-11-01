"""Canonical data contracts for the SLAI Spatial subsystem.

The type model follows the separation encouraged by ISO 19107 / OGC Simple
Features: identity, reference, geometry and relation metadata are explicit,
while concrete geometric algorithms remain in ``world.geometry``.
"""
from __future__ import annotations

__version__ = "2.3.0"

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Sequence

from .utils.spatial_errors import SpatialValidationError
from .utils.spatial_helpers import to_json_safe, utc_now_iso, validate_identifier
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Spatial Types")
printer = PrettyPrinter()


class GeometryKind(str, Enum):
    POINT = "point"
    LINE = "line"
    SEGMENT = "segment"
    RAY = "ray"
    PLANE = "plane"
    TRIANGLE = "triangle"
    POLYGON = "polygon"
    POLYHEDRON = "polyhedron"
    SPHERE = "sphere"
    AABB = "aabb"
    OBB = "obb"
    MESH = "mesh"
    POINT_CLOUD = "point_cloud"
    OCCUPANCY = "occupancy"
    UNKNOWN = "unknown"


class RelationKind(str, Enum):
    DISTANCE = "distance"
    INTERSECTS = "intersects"
    DISJOINT = "disjoint"
    TOUCHES = "touches"
    OVERLAPS = "overlaps"
    CONTAINS = "contains"
    INSIDE = "inside"
    EQUALS = "equals"
    CROSSES = "crosses"
    CONNECTED = "connected"
    NEAR = "near"
    FAR = "far"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"
    VISIBLE_FROM = "visible_from"


@dataclass(frozen=True, slots=True)
class SpatialReference:
    """Reference metadata for local frames or named coordinate systems."""

    reference_id: str
    kind: str = "local_frame"
    authority: str | None = None
    code: str | None = None
    units: str = "m"
    axis_order: tuple[str, ...] = ("x", "y", "z")
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reference_id", validate_identifier(self.reference_id, name="reference_id"))
        if not self.kind.strip():
            raise SpatialValidationError("SpatialReference.kind must not be empty")
        if not self.units.strip():
            raise SpatialValidationError("SpatialReference.units must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return to_json_safe(self)


@dataclass(frozen=True, slots=True)
class SpatialBounds:
    """Axis-aligned bounds represented by finite lower and upper coordinates."""

    lower: tuple[float, ...]
    upper: tuple[float, ...]
    frame_id: str = "world"

    def __post_init__(self) -> None:
        if not self.lower or len(self.lower) != len(self.upper):
            raise SpatialValidationError(
                "SpatialBounds lower/upper dimensions must match and be non-empty",
                context={"lower": self.lower, "upper": self.upper},
            )
        lower = tuple(float(v) for v in self.lower)
        upper = tuple(float(v) for v in self.upper)
        if any(lo > hi for lo, hi in zip(lower, upper)):
            raise SpatialValidationError(
                "SpatialBounds lower values must not exceed upper values",
                context={"lower": lower, "upper": upper},
            )
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "frame_id", validate_identifier(self.frame_id, name="frame_id"))

    @property
    def dimension(self) -> int:
        return len(self.lower)

    @property
    def center(self) -> tuple[float, ...]:
        return tuple((lo + hi) * 0.5 for lo, hi in zip(self.lower, self.upper))

    def to_dict(self) -> dict[str, Any]:
        return {"lower": list(self.lower), "upper": list(self.upper), "frame_id": self.frame_id}


@dataclass(frozen=True, slots=True)
class SpatialEntity:
    """Canonical world-space entity consumed by Spatial after perception."""

    entity_id: str
    position: tuple[float, ...]
    frame_id: str = "world"
    geometry: Any = None
    geometry_kind: GeometryKind = GeometryKind.UNKNOWN
    bounds: SpatialBounds | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_now_iso)
    revision: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_id", validate_identifier(self.entity_id, name="entity_id"))
        object.__setattr__(self, "frame_id", validate_identifier(self.frame_id, name="frame_id"))
        if not self.position:
            raise SpatialValidationError("SpatialEntity.position must not be empty")
        position = tuple(float(value) for value in self.position)
        if any(value != value or value in (float("inf"), float("-inf")) for value in position):
            raise SpatialValidationError("SpatialEntity.position must contain finite values")
        object.__setattr__(self, "position", position)
        if self.revision < 0:
            raise SpatialValidationError("SpatialEntity.revision must be non-negative")

    @property
    def dimension(self) -> int:
        return len(self.position)

    def moved(self, position: Sequence[float], *, frame_id: str | None = None) -> "SpatialEntity":
        return replace(
            self,
            position=tuple(float(v) for v in position),
            frame_id=self.frame_id if frame_id is None else frame_id,
            timestamp=utc_now_iso(),
            revision=self.revision + 1,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "entity_id": self.entity_id,
            "position": list(self.position),
            "frame_id": self.frame_id,
            "geometry_kind": self.geometry_kind.value,
            "bounds": self.bounds.to_dict() if self.bounds else None,
            "metadata": to_json_safe(self.metadata),
            "timestamp": self.timestamp,
            "revision": self.revision,
        }
        if self.geometry is not None:
            payload["geometry"] = to_json_safe(self.geometry)
        return payload


@dataclass(frozen=True, slots=True)
class SpatialRelationship:
    subject_id: str
    object_id: str
    relation: RelationKind
    value: float | bool | str | None = None
    confidence: float = 1.0
    frame_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_id", validate_identifier(self.subject_id, name="subject_id"))
        object.__setattr__(self, "object_id", validate_identifier(self.object_id, name="object_id"))
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise SpatialValidationError("confidence must lie in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return to_json_safe(self)


@dataclass(frozen=True, slots=True)
class SpatialRecord:
    record_id: str
    kind: str
    payload: Mapping[str, Any]
    timestamp: str = field(default_factory=utc_now_iso)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", validate_identifier(self.record_id, name="record_id"))
        if not self.kind.strip():
            raise SpatialValidationError("SpatialRecord.kind must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return to_json_safe(self)


@dataclass(frozen=True, slots=True)
class SpatialActivity:
    activity_id: str
    operation: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        object.__setattr__(self, "activity_id", validate_identifier(self.activity_id, name="activity_id"))
        if not self.operation.strip():
            raise SpatialValidationError("SpatialActivity.operation must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return to_json_safe(self)


@dataclass(frozen=True, slots=True)
class SpatialQueryResult:
    query: str
    entity_ids: tuple[str, ...]
    distances: tuple[float, ...] = ()
    relationships: tuple[SpatialRelationship, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.distances and len(self.distances) != len(self.entity_ids):
            raise SpatialValidationError("distances must align with entity_ids")

    def to_dict(self) -> dict[str, Any]:
        return to_json_safe(self)


@dataclass(frozen=True, slots=True)
class SpatialBundle:
    entities: tuple[SpatialEntity, ...] = ()
    relationships: tuple[SpatialRelationship, ...] = ()
    records: tuple[SpatialRecord, ...] = ()
    reference: SpatialReference | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_json_safe(self)


# Backward-compatible plural alias retained from the original subsystem stub.
SpatialRecords = SpatialRecord


class SpatialTypes:
    """Namespace exposing canonical Spatial domain types for introspection."""

    entity = SpatialEntity
    activity = SpatialActivity
    record = SpatialRecord
    relationship = SpatialRelationship
    bundle = SpatialBundle
    reference = SpatialReference
    bounds = SpatialBounds
    query_result = SpatialQueryResult


__all__ = [
    "GeometryKind",
    "RelationKind",
    "SpatialReference",
    "SpatialBounds",
    "SpatialEntity",
    "SpatialActivity",
    "SpatialRecord",
    "SpatialRecords",
    "SpatialRelationship",
    "SpatialQueryResult",
    "SpatialBundle",
    "SpatialTypes",
]


if __name__ == "__main__":
    configure_logging()
    printer.status("TEST", "Spatial Types initialized", "info")
