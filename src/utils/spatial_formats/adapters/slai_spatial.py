"""Controlled DTO bridge to the existing Spatial Agent, imported only on demand.

Agents should exchange a *reference* to a document (id, revision, digest, units,
selected object ids) and fetch geometry on demand rather than pass scene bytes
around (feasibility report §4.2/§11.1). :func:`spatial_reference` builds that
reference; :func:`spatial_records` returns JSON-safe per-object DTOs, optionally
converted to another length unit and restricted to selected objects.
"""
from __future__ import annotations

from ..common.errors import ValidationError
from ..common.ids import require_id
from ..common.units import convert_length, require_unit
from ..sgeo.scene import Scene


def spatial_reference(scene: Scene, *, sha256: str | None = None, uri: str | None = None,
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
    return {"document_id": scene.document_id, "revision": scene.revision, "units": scene.unit,
            "format_id": "slai.sgeo", "sha256": sha256, "uri": uri, "object_ids": ids}


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
