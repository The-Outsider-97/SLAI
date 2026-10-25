"""Strict native SDRW primitive representation, coordinates in declared drawing units."""
from __future__ import annotations
from math import pi
from ..common import ValidationError, finite, vector, require_id, new_id, require_metadata

KINDS = frozenset(("point", "line", "polyline", "circle", "arc"))
MAX_POLYLINE_POINTS = 100_000

def validate_entity(entity: dict, layers: set[str] | None = None) -> dict:
    if not isinstance(entity, dict):
        raise ValidationError("drawing entity must be an object")
    require_id(entity.get("id"))
    kind = entity.get("kind")
    if kind not in KINDS:
        raise ValidationError("unsupported SDRW primitive kind")
    layer = entity.get("layer", "0")
    if not isinstance(layer, str) or not layer or len(layer) > 255:
        raise ValidationError("invalid drawing layer")
    if layers is not None and layer not in layers:
        raise ValidationError("entity references nonexistent layer: " + layer)
    data = entity.get("geometry")
    if not isinstance(data, dict):
        raise ValidationError("entity geometry must be an object")
    if kind == "point":
        vector(data.get("position"), 2)
    elif kind == "line":
        a, b = vector(data.get("start"), 2), vector(data.get("end"), 2)
        if a == b:
            raise ValidationError("zero-length line")
    elif kind == "polyline":
        pts = data.get("points")
        if not isinstance(pts, list) or not 2 <= len(pts) <= MAX_POLYLINE_POINTS:
            raise ValidationError("polyline requires 2..100000 points")
        points = [vector(p, 2) for p in pts]
        if all(p == points[0] for p in points):
            raise ValidationError("degenerate polyline")
        if type(data.get("closed", False)) is not bool:
            raise ValidationError("polyline closed flag must be Boolean")
        if data.get("closed", False) and len(set(points)) < 3:
            raise ValidationError("closed polyline requires three unique points")
    else:
        vector(data.get("center"), 2)
        radius = finite(data.get("radius"), "radius")
        if radius <= 0:
            raise ValidationError("circle/arc radius must be positive")
        if kind == "arc":
            start = finite(data.get("start_angle"), "start_angle", limit=1e8)
            end = finite(data.get("end_angle"), "end_angle", limit=1e8)
            if not 0 < end - start <= 2*pi + 1e-10:
                raise ValidationError("arc sweep must lie in (0, 2π]")
    if "metadata" in entity:
        require_metadata(entity["metadata"])
    if "name" in entity and (not isinstance(entity["name"], str) or len(entity["name"]) > 255):
        raise ValidationError("invalid entity name")
    for key in entity:
        if key not in {"id", "kind", "layer", "geometry", "name", "metadata", "style"}:
            raise ValidationError("unknown entity field: " + key)
    if "style" in entity:
        style = entity["style"]
        if not isinstance(style, dict):
            raise ValidationError("style must be an object")
        if set(style) - {"stroke", "stroke_width"}:
            raise ValidationError("unsupported entity style property")
        if "stroke" in style and (not isinstance(style["stroke"], str) or len(style["stroke"]) > 32):
            raise ValidationError("invalid stroke")
        if "stroke_width" in style and finite(style["stroke_width"], "stroke_width") < 0:
            raise ValidationError("stroke width cannot be negative")
    return entity

def make_entity(kind: str, geometry: dict, *, layer="0", name=None, metadata=None, style=None, entity_id=None) -> dict:
    entity = {"id": entity_id or new_id(), "kind": kind, "layer": layer, "geometry": geometry}
    if name is not None:
        entity["name"] = name
    if metadata is not None:
        entity["metadata"] = metadata
    if style is not None:
        entity["style"] = style
    return validate_entity(entity)

def entity_bounds(entity: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    """Tight analytic bounds for drawing entities, including arc extrema."""
    from .operations import entity_bounds as evaluate_bounds
    return evaluate_bounds(entity)
