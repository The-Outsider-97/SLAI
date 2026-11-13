"""Strict native SDRW entity representation, coordinates in declared drawing units.

Kinds: ``point``, ``line``, ``polyline``, ``circle``, ``arc`` (0.1) and ``text``,
``dimension``, ``hatch``, ``insert`` (0.2).
"""
from __future__ import annotations
from math import pi
from copy import deepcopy
from ..common.errors import GeometryValidationError, ReferenceError
from ..common import ValidationError, finite, vector, require_id, new_id, require_metadata
from . import annotations as _ann
from .blocks import INSERT_KEYS, validate_insert
from .linetypes import check_linetype_ref

PRIMITIVE_KINDS = frozenset(("point", "line", "polyline", "circle", "arc"))
ANNOTATION_KINDS = frozenset(("text", "dimension", "hatch"))
KINDS = PRIMITIVE_KINDS | ANNOTATION_KINDS | {"insert"}
MAX_POLYLINE_POINTS = 100_000
_GEOMETRY_KEYS = {
    "point": frozenset({"position"}),
    "line": frozenset({"start", "end"}),
    "polyline": frozenset({"points", "closed"}),
    "circle": frozenset({"center", "radius"}),
    "arc": frozenset({"center", "radius", "start_angle", "end_angle"}),
    "text": _ann.TEXT_KEYS,
    "hatch": _ann.HATCH_KEYS,
    "insert": INSERT_KEYS,
}
STYLE_KEYS = frozenset({"stroke", "stroke_width", "fill", "linetype", "linetype_scale"})


def validate_style(style, linetypes=None) -> dict:
    """Validate an entity or layer style dict (``stroke``, ``stroke_width``, ``fill``,
    ``linetype``, ``linetype_scale``). ``linetypes`` is the set of defined names, or None."""
    if not isinstance(style, dict):
        raise ValidationError("style must be an object")
    if set(style) - STYLE_KEYS:
        raise ValidationError("unsupported entity style property")
    for key in ("stroke", "fill"):
        if key in style and (not isinstance(style[key], str) or not 1 <= len(style[key]) <= 32):
            raise ValidationError("invalid " + key)
    if "stroke_width" in style and finite(style["stroke_width"], "stroke_width") < 0:
        raise ValidationError("stroke width cannot be negative")
    if "linetype" in style:
        check_linetype_ref(style["linetype"], linetypes)
    if "linetype_scale" in style and finite(style["linetype_scale"], "linetype_scale") <= 0:
        raise ValidationError("linetype_scale must be positive")
    return style


def validate_entity(entity: dict, layers: set[str] | None = None, *, linetypes=None, blocks=None) -> dict:
    """Validate one entity.

    ``layers``, ``linetypes`` and ``blocks`` are the sets of names defined by the
    drawing; pass ``None`` to skip the corresponding reference check (a lone
    entity can be validated without its drawing).
    """
    if not isinstance(entity, dict):
        raise ValidationError("drawing entity must be an object")
    require_id(entity.get("id"))
    kind = entity.get("kind")
    if not isinstance(kind, str) or kind not in KINDS:
        raise ValidationError("unsupported SDRW primitive kind")
    layer = entity.get("layer", "0")
    if not isinstance(layer, str) or not layer or len(layer) > 255:
        raise ValidationError("invalid drawing layer")
    if layers is not None and layer not in layers:
        raise ReferenceError("entity references nonexistent layer: " + layer, context={"layer": layer})
    data = entity.get("geometry")
    if not isinstance(data, dict):
        raise ValidationError("entity geometry must be an object")
    if kind == "dimension":
        _ann.validate_dimension(data)
    else:
        unknown = set(data) - _GEOMETRY_KEYS[kind]
        if unknown:
            raise GeometryValidationError("unknown fields in SDRW geometry: " + ", ".join(sorted(map(str, unknown))))
    if kind == "point":
        vector(data.get("position"), 2)
    elif kind == "line":
        a, b = vector(data.get("start"), 2), vector(data.get("end"), 2)
        if a == b:
            raise GeometryValidationError("zero-length line")
    elif kind == "polyline":
        pts = data.get("points")
        if not isinstance(pts, list) or not 2 <= len(pts) <= MAX_POLYLINE_POINTS:
            raise GeometryValidationError("polyline requires 2..100000 points")
        points = [vector(p, 2) for p in pts]
        if all(p == points[0] for p in points):
            raise GeometryValidationError("degenerate polyline")
        if type(data.get("closed", False)) is not bool:
            raise GeometryValidationError("polyline closed flag must be Boolean")
        if data.get("closed", False) and len(set(points)) < 3:
            raise GeometryValidationError("closed polyline requires three unique points")
    elif kind in ("circle", "arc"):
        vector(data.get("center"), 2)
        radius = finite(data.get("radius"), "radius")
        if radius <= 0:
            raise GeometryValidationError("circle/arc radius must be positive")
        if kind == "arc":
            start = finite(data.get("start_angle"), "start_angle", limit=1e8)
            end = finite(data.get("end_angle"), "end_angle", limit=1e8)
            if not 0 < end - start <= 2*pi + 1e-10:
                raise GeometryValidationError("arc sweep must lie in (0, 2π]")
    elif kind == "text":
        _ann.validate_text(data)
    elif kind == "hatch":
        _ann.validate_hatch(data)
    elif kind == "insert":
        validate_insert(data)
        if blocks is not None and data["block"] not in blocks:
            raise ReferenceError("insert references unknown block: " + data["block"], context={"block": data["block"]})
    if "metadata" in entity:
        require_metadata(entity["metadata"])
    if "name" in entity and (not isinstance(entity["name"], str) or len(entity["name"]) > 255):
        raise ValidationError("invalid entity name")
    for key in entity:
        if key not in {"id", "kind", "layer", "geometry", "name", "metadata", "style"}:
            raise ValidationError("unknown entity field: " + key)
    if "style" in entity:
        validate_style(entity["style"], linetypes)
    return entity

def make_entity(kind: str, geometry: dict, *, layer="0", name=None, metadata=None, style=None, entity_id=None) -> dict:
    entity = {"id": entity_id or new_id(), "kind": kind, "layer": layer, "geometry": geometry}
    if name is not None:
        entity["name"] = name
    if metadata is not None:
        entity["metadata"] = metadata
    if style is not None:
        entity["style"] = style
    return validate_entity(deepcopy(entity))

def entity_bounds(entity: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    """Tight analytic bounds for drawing entities, including arc extrema."""
    from .operations import entity_bounds as evaluate_bounds
    return evaluate_bounds(entity)
