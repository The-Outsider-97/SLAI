"""Controlled DTO bridge to the existing Spatial Agent, imported only on demand.

Agents should exchange a *reference* to a document (id, revision, digest, units,
selected object ids) and fetch geometry on demand rather than pass scene bytes
around (feasibility report §4.2/§11.1). :func:`spatial_reference` builds that
reference; :func:`spatial_records` returns JSON-safe per-object DTOs, optionally
converted to another length unit and restricted to selected objects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from ..common.errors import ValidationError
from ..common.ids import require_id
from ..common.units import convert_length, require_unit
from ..sgeo.scene import Scene
from ..sdrw.drawing import Drawing


class SpatialMemoryService(Protocol):
    """Minimum canonical SpatialMemory interface needed by the bridge."""

    def require_entity(self, entity_id: str) -> Any: ...


class SpatialComputeService(Protocol):
    """The Spatial Agent, not the format package, owns metric algorithms."""

    def distance(self, a: Any, b: Any, *, metric: str = "euclidean") -> float: ...


class SpatialQueryService(Protocol):
    """Subset of SpatialQueries consumed by a document-scoped session."""

    def nearest(self, point: Any, *, frame_id: str | None = None,
                exclude_ids: set[str] | None = None) -> Any: ...

    def k_nearest(self, point: Any, k: int, *, frame_id: str | None = None) -> Any: ...

    def within_radius(self, point: Any, radius: float, *, frame_id: str | None = None) -> Any: ...

    def intersecting_bounds(self, bounds: Any, *, frame_id: str | None = None) -> Any: ...

    def query_relation(self, entity: Any, relation: str, *, candidates: Any = None) -> Any: ...


_CENTER_RELATIONS = frozenset({
    "near", "far", "left_of", "right_of", "above", "below", "in_front_of", "behind",
})


@dataclass(frozen=True, slots=True)
class SpatialSceneAdapter:
    """Document-scoped, SI-normalized access to real Spatial services.

    An adapter is a *snapshot of one document revision*. Its mutable agent
    memory is private to the session; create another adapter after editing.
    All native query distances use metres. Bounding-box hits are broad-phase
    candidates, **not** claims of exact solid intersection.
    """

    reference: Mapping[str, Any]
    entity_ids: frozenset[str]
    memory: SpatialMemoryService
    compute: SpatialComputeService
    queries: SpatialQueryService

    def entity(self, entity_id: str) -> Any:
        key = require_id(entity_id)
        if key not in self.entity_ids:
            raise ValidationError("object is not present in the spatial session", context={"id": key})
        return self.memory.require_entity(key)

    def distance(self, first_id: str, second_id: str, *, unit: str = "m") -> float:
        """Return centre-to-centre Euclidean distance in the requested unit."""
        measured_m = self.compute.distance(self.entity(first_id), self.entity(second_id))
        return convert_length(measured_m, "m", require_unit(unit))

    def nearest(self, position: tuple[float, ...], *, unit: str = "m", k: int = 1) -> Any:
        """Return the Spatial Agent's result; distances in it are metres."""
        point_m = tuple(convert_length(value, unit, "m") for value in position)
        if k == 1:
            return self.queries.nearest(point_m, frame_id="world")
        return self.queries.k_nearest(point_m, k, frame_id="world")

    def within_radius(self, position: tuple[float, ...], radius: float,
                      *, unit: str = "m") -> Any:
        """Perform the real Spatial Agent's centre-point radius query."""
        point_m = tuple(convert_length(value, unit, "m") for value in position)
        radius_m = convert_length(radius, unit, "m")
        return self.queries.within_radius(point_m, radius_m, frame_id="world")

    def intersecting_bounds(self, entity_id: str) -> Any:
        """AABB broad-phase candidates only; excludes the queried object."""
        entity = self.entity(entity_id)
        if entity.bounds is None:
            raise ValidationError("spatial entity has no world bounds", context={"id": entity_id})
        result = self.queries.intersecting_bounds(entity.bounds, frame_id="world")
        return tuple(other_id for other_id in result.entity_ids if other_id != entity_id)

    def centre_relation(self, first_id: str, second_id: str, relation: str) -> bool:
        """Use Spatial's centre-based metric/directional relations.

        Geometry predicates (contains/intersects/etc.) are intentionally not
        advertised: this DTO bridge supplies bounds, not exact agent geometry.
        """
        if relation not in _CENTER_RELATIONS:
            raise ValidationError("unsupported centre relation", context={"relation": relation})
        first, second = self.entity(first_id), self.entity(second_id)
        result = self.queries.query_relation(first, relation, candidates=(second,))
        return second.entity_id in result.entity_ids


def spatial_reference(scene: Scene | Drawing, *, sha256: str | None = None, uri: str | None = None,
                      object_ids=None) -> dict:
    """Return the small, typed handle an agent message should carry.

    ``sha256`` is the digest of the saved package the revision came from (known
    to a document opened from disk); ``uri`` is an opaque caller-supplied
    logical identifier — this library never resolves or fetches it.
    """
    scene.validate()
    if sha256 is not None and (not isinstance(sha256, str) or len(sha256) != 64
                               or any(c not in "0123456789abcdef" for c in sha256)):
        raise ValidationError("sha256 must be a lowercase hex digest")
    if uri is not None and (not isinstance(uri, str) or not uri or len(uri) > 2048):
        raise ValidationError("uri must be a bounded non-empty string")
    ids = None if object_ids is None else [require_id(i) for i in object_ids]
    if ids is not None:
        known = {obj["id"] for obj in (scene.objects if isinstance(scene, Scene) else scene.entities)}
        missing = [entity_id for entity_id in ids if entity_id not in known]
        if missing:
            raise ValidationError("unknown spatial reference selection", context={"ids": missing[:10]})
    return {"document_id": scene.document_id, "revision": scene.revision, "units": scene.unit,
            "format_id": "slai.sgeo" if isinstance(scene, Scene) else "slai.sdrw",
            "sha256": sha256, "uri": uri, "object_ids": ids}


def spatial_records(scene: Scene, *, object_ids=None, unit: str | None = None) -> list[dict]:
    """Return JSON-safe DTOs; do not mutate SpatialMemory or import agent runtime.

    Coordinates are expressed in ``unit`` (default: the scene's own unit) and the
    unit is stated in every record, so a consumer can never mistake metres for
    millimetres. ``object_ids`` restricts the result (unknown ids are an error).
    """
    scene.validate()
    target = require_unit(unit) if unit is not None else scene.unit
    factor = 1.0 if target == scene.unit else convert_length(1.0, scene.unit, target)
    wanted = None
    if object_ids is not None:
        wanted = [require_id(i) for i in object_ids]
        known = {o["id"] for o in scene.objects}
        missing = [i for i in wanted if i not in known]
        if missing:
            raise ValidationError("unknown object id in selection", context={"ids": missing[:10]})
        wanted = set(wanted)
    bounds = scene.bounds_by_object()  # one world-transform pass for the whole scene
    results = []
    for obj in scene.objects:
        geometry = obj.get("geometry")
        if geometry is None or (wanted is not None and obj["id"] not in wanted):
            continue
        lo, hi = ([c * factor for c in corner] for corner in bounds[obj["id"]])
        results.append({"entity_id": obj["id"], "frame_id": "world", "unit": target,
                        "geometry_kind": geometry["kind"],
                        "position": [(a + b) / 2 for a, b in zip(lo, hi)],
                        "bounds": {"lower": lo, "upper": hi, "frame_id": "world"},
                        "metadata": {"slai.source_document": scene.document_id,
                                     "slai.source_revision": scene.revision,
                                     "slai.source_unit": scene.unit,
                                     "slai.record_unit": target}})
    return results


def to_agent_bundle(scene: Scene, *, object_ids=None, unit: str | None = None):
    """Hydrate actual Spatial types after caller opts into loading the agent runtime.

    ``unit`` defaults to metres: the Spatial Agent works in SI units, so a scene
    authored in millimetres is converted rather than handed over unlabeled.
    """
    from src.agents.spatial.spatial_types import SpatialEntity, SpatialBounds, SpatialBundle, GeometryKind # type: ignore
    items = []
    for record in spatial_records(scene, object_ids=object_ids, unit=unit or "m"):
        enum = (GeometryKind(record["geometry_kind"]) if record["geometry_kind"] in GeometryKind._value2member_map_
                else GeometryKind.UNKNOWN)
        bounds = record["bounds"]
        items.append(SpatialEntity(entity_id=record["entity_id"], position=tuple(record["position"]),
                                   geometry_kind=enum,
                                   bounds=SpatialBounds(tuple(bounds["lower"]), tuple(bounds["upper"])),
                                   metadata=record["metadata"]))
    return SpatialBundle(entities=tuple(items))


def drawing_records(drawing: Drawing, *, entity_ids=None, unit: str | None = None) -> list[dict]:
    """SDRW model-space entities as 2D centre/bounds DTOs, not exact solids.

    Reuses the drawing subsystem's analytic bound evaluator (including block
    inserts). Text and dimension bounds inherit its documented approximation.
    """
    from ..sdrw.operations import entity_bounds

    drawing.validate()
    target = require_unit(unit) if unit is not None else drawing.unit
    wanted = None if entity_ids is None else {require_id(value) for value in entity_ids}
    if wanted is not None:
        missing = wanted - {entity["id"] for entity in drawing.entities}
        if missing:
            raise ValidationError("unknown drawing entity in selection", context={"ids": sorted(missing)[:10]})
    blocks = {block["name"]: block for block in drawing.blocks}
    records = []
    for entity in drawing.entities:
        if wanted is not None and entity["id"] not in wanted:
            continue
        lower, upper = entity_bounds(entity, blocks)
        lower = [convert_length(value, drawing.unit, target) for value in lower]
        upper = [convert_length(value, drawing.unit, target) for value in upper]
        records.append({
            "entity_id": entity["id"], "frame_id": "world", "unit": target,
            "geometry_kind": entity["kind"],
            "position": [(a + b) * 0.5 for a, b in zip(lower, upper)],
            "bounds": {"lower": lower, "upper": upper, "frame_id": "world"},
            "metadata": {
                "slai.source_document": drawing.document_id,
                "slai.source_revision": drawing.revision,
                "slai.source_unit": drawing.unit,
                "slai.record_unit": target,
                "slai.bounds_are_approximate": entity["kind"] in {"text", "dimension", "insert"},
            },
        })
    return records


def to_drawing_bundle(drawing: Drawing, *, entity_ids=None, unit: str | None = None):
    """Hydrate 2D Spatial Agent entities without importing it on package load."""
    from src.agents.spatial.spatial_types import SpatialBounds, SpatialBundle, SpatialEntity, GeometryKind

    items = []
    for record in drawing_records(drawing, entity_ids=entity_ids, unit=unit or "m"):
        kind = record["geometry_kind"]
        enum = GeometryKind(kind) if kind in GeometryKind._value2member_map_ else GeometryKind.UNKNOWN
        bounds = record["bounds"]
        items.append(SpatialEntity(
            entity_id=record["entity_id"],
            position=tuple(record["position"]),
            geometry_kind=enum,
            bounds=SpatialBounds(tuple(bounds["lower"]), tuple(bounds["upper"])),
            metadata=record["metadata"],
        ))
    return SpatialBundle(entities=tuple(items))
