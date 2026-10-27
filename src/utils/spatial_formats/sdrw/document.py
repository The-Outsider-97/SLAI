"""Editable SDRW document: the object behind ``src.utils.sdrw.SDRWDocument``.

Wraps one validated :class:`Drawing` with a task-oriented API: typed entity
constructors, atomic edits (every call is exactly one revision), queries,
measurements, layer management, unit conversion and loss-reported interchange.
All geometry is planar, in the document's declared length unit, right-handed,
with angles in **radians** unless a method says ``degrees=``.
"""
from __future__ import annotations

from copy import deepcopy
from math import atan2, cos, hypot, isfinite, pi, radians, sin, tau

from ..common.errors import ReferenceError, ValidationError
from ..common.helpers import finite, vector
from ..common.ids import is_valid_id, require_id
from ..common.units import convert_area, convert_length, require_unit
from ..documents import SpatialDocument, _checked_metadata
from ..verification.report import ValidationReport, check_drawing
from . import annotations as _ann
from .constraints import ConstraintResult, check_constraints
from .drawing import Drawing
from .entities import KINDS
from .operations import enclosed_area, entity_bounds, entity_length, expand_insert, transform_entity
from .transactions import apply_drawing_operations

__all__ = ["DrawingDocument", "MAX_QUERY_LIMIT"]

MAX_QUERY_LIMIT = 10_000
_OPTIONS = frozenset({"layer", "name", "metadata", "style", "entity_id", "create_layer"})
_DIM_OPTIONS = frozenset({"text", "text_height", "text_position", "precision", "suffix", "arrow_size"})


def _pt(value, label="point") -> list:
    return [float(v) for v in vector(value, 2, label)]


def _translation(dx: float, dy: float) -> list:
    return [1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0, 1.0]


def _rotation_about(angle: float, cx: float, cy: float) -> list:
    c, s = cos(angle), sin(angle)
    return [c, -s, cx - (c * cx - s * cy), s, c, cy - (s * cx + c * cy), 0.0, 0.0, 1.0]


def _scale_about(factor: float, cx: float, cy: float) -> list:
    return [factor, 0.0, cx * (1 - factor), 0.0, factor, cy * (1 - factor), 0.0, 0.0, 1.0]


def _angle(value, label, degrees: bool) -> float:
    number = finite(value, label, limit=1e8)
    return radians(number) if degrees else number


def _as_ids(entity_ids) -> list:
    if isinstance(entity_ids, str):
        entity_ids = [entity_ids]
    try:
        ids = [require_id(i) for i in entity_ids]
    except TypeError as exc:
        raise ValidationError("entity ids must be an id or an iterable of ids") from exc
    if not ids:
        raise ValidationError("at least one entity id is required")
    if len(set(ids)) != len(ids):
        raise ValidationError("duplicate entity id in selection")
    return ids


class DrawingDocument(SpatialDocument):
    """A validated 2D drawing with an atomic, queryable editing API."""

    FORMAT_ID = "slai.sdrw"
    FILE_SUFFIX = ".sdrw"
    KIND_LABEL = "drawing"

    def __init__(self, drawing: Drawing | None = None, *, name: str = "Untitled", unit: str = "mm",
                 path=None, source_sha256=None, read_only: bool = False, source_version=None):
        if drawing is None:
            drawing = Drawing(name, unit=unit)
        elif not isinstance(drawing, Drawing):
            raise ValidationError("expected a Drawing model")
        super().__init__(drawing, path=path, source_sha256=source_sha256, read_only=read_only, source_version=source_version)

    # -- construction ---------------------------------------------------------
    @classmethod
    def new(cls, name: str = "Untitled", *, unit: str = "mm", layers=None, metadata=None,
            document_id=None) -> "DrawingDocument":
        """Create an empty, writable, never-saved drawing (layer ``"0"`` always exists)."""
        return cls(Drawing(name, unit=unit, layers=layers, metadata=metadata, document_id=document_id))

    @classmethod
    def from_drawing(cls, drawing: Drawing, *, copy: bool = True) -> "DrawingDocument":
        """Wrap an existing :class:`Drawing` (deep-copied unless ``copy=False``)."""
        if not isinstance(drawing, Drawing):
            raise ValidationError("expected a Drawing model")
        with drawing._mutation_lock:
            drawing.validate()
            return cls(deepcopy(drawing) if copy else drawing)

    @classmethod
    def _model_from_package(cls, package):
        return Drawing.from_package(package)

    # -- hooks ------------------------------------------------------------------
    def _check(self, mode: str, max_issues: int) -> ValidationReport:
        return check_drawing(self._model, mode=mode, max_issues=max_issues)

    def _apply(self, operations, expected_revision):
        return apply_drawing_operations(self._model, operations, expected_revision=expected_revision)

    def _counts(self) -> dict:
        with self._model._mutation_lock:
            kinds: dict = {}
            for e in self._model.entities:
                kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
            return {"layer_count": len(self._model.layers), "layers": [l["name"] for l in self._model.layers],
                    "entity_count": len(self._model.entities), "entity_kinds": dict(sorted(kinds.items())),
                    "linetype_count": len(self._model.linetypes), "block_count": len(self._model.blocks),
                    "bounds": self.bounds()}

    # -- layers -------------------------------------------------------------------
    def layers(self) -> list:
        """Layers in order: ``{"name", "visible", "entity_count"}`` plus any of ``stroke``,
        ``stroke_width``, ``linetype`` that are set (detached copies)."""
        with self._model._mutation_lock:
            use: dict = {}
            for e in self._model.entities:
                use[e.get("layer", "0")] = use.get(e.get("layer", "0"), 0) + 1
            return [{**deepcopy(l), "visible": l.get("visible", True), "entity_count": use.get(l["name"], 0)}
                    for l in self._model.layers]

    def add_layer(self, name: str, *, visible: bool = True, stroke: str | None = None,
                  stroke_width: float | None = None, linetype: str | None = None) -> str:
        """Add an empty layer (one revision); duplicates are rejected.

        ``stroke`` (a colour string), ``stroke_width`` and ``linetype`` are the layer's defaults
        for entities that do not override them.
        """
        style = {k: v for k, v in (("stroke", stroke), ("stroke_width", stroke_width),
                                   ("linetype", linetype)) if v is not None}
        with self._edit() as model:
            return model.add_layer(name, visible=visible, **style)

    def update_layer(self, name: str, *, visible: bool | None = None, stroke=..., stroke_width=...,
                     linetype=...) -> str:
        """Change layer properties in one revision. Pass ``None`` to clear ``stroke``,
        ``stroke_width`` or ``linetype``; omit an argument to leave it unchanged."""
        value = {}
        if visible is not None:
            value["visible"] = visible
        for key, item in (("stroke", stroke), ("stroke_width", stroke_width), ("linetype", linetype)):
            if item is not ...:
                value[key] = item
        if not value:
            raise ValidationError("nothing to update")
        return self.apply([{"op": "update_layer", "name": name, "value": value}])[0]

    def remove_layer(self, name: str) -> str:
        """Remove an *unused* layer; layer ``"0"`` may be removed only if another layer remains."""
        return self.apply([{"op": "remove_layer", "name": name}])[0]

    def rename_layer(self, name: str, new_name: str) -> str:
        """Rename a layer and every entity on it atomically."""
        self.apply([{"op": "rename_layer", "name": name, "value": new_name}])
        return new_name

    def set_layer_visibility(self, name: str, visible: bool) -> str:
        self.apply([{"op": "layer_visibility", "name": name, "value": visible}])
        return name

    # -- entity construction ----------------------------------------------------------
    def _add(self, kind: str, geometry: dict, options: dict) -> str:
        unknown = set(options) - _OPTIONS
        if unknown:
            raise ValidationError("unknown entity option(s): " + ", ".join(sorted(unknown)))
        with self._edit() as model:
            return model.add(kind, geometry, **options)

    def add_point(self, position, **options) -> str:
        """Add a point. Options: layer, name, metadata, style, entity_id, create_layer."""
        return self._add("point", {"position": _pt(position)}, options)

    def add_line(self, start, end, **options) -> str:
        return self._add("line", {"start": _pt(start, "start"), "end": _pt(end, "end")}, options)

    def add_polyline(self, points, *, closed: bool = False, **options) -> str:
        """Open or closed polyline of at least two points (closed needs three distinct)."""
        if isinstance(points, (str, bytes)) or not hasattr(points, "__iter__"):
            raise ValidationError("points must be a sequence of (x, y) pairs")
        return self._add("polyline", {"points": [_pt(p) for p in points], "closed": bool(closed)}, options)

    def add_polygon(self, points, **options) -> str:
        """Closed polyline (alias of ``add_polyline(points, closed=True)``)."""
        return self.add_polyline(points, closed=True, **options)

    def add_rectangle(self, corner, width: float, height: float, *, centered: bool = False, **options) -> str:
        """Axis-aligned rectangle as a closed 4-point polyline (counter-clockwise).

        ``corner`` is the lower-left corner, or the centre with ``centered=True``.
        """
        x, y = _pt(corner, "corner")
        w, h = finite(width, "width"), finite(height, "height")
        if w <= 0 or h <= 0:
            raise ValidationError("rectangle width and height must be positive")
        if centered:
            x, y = x - w / 2, y - h / 2
        return self.add_polyline([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], closed=True, **options)

    def add_circle(self, center, radius: float, **options) -> str:
        return self._add("circle", {"center": _pt(center, "center"), "radius": radius}, options)

    def add_arc(self, center, radius: float, start_angle: float, end_angle: float, *, degrees: bool = False,
                **options) -> str:
        """Counter-clockwise arc from ``start_angle`` to ``end_angle`` (sweep in (0, 2π])."""
        geometry = {"center": _pt(center, "center"), "radius": radius,
                    "start_angle": _angle(start_angle, "start_angle", degrees),
                    "end_angle": _angle(end_angle, "end_angle", degrees)}
        return self._add("arc", geometry, options)

    def add_arc_through(self, start, through, end, **options) -> str:
        """Arc passing through three points, directed start -> through -> end.

        The stored arc is always counter-clockwise, so a clockwise request is
        stored with its endpoints swapped (the geometry is identical).
        """
        (x1, y1), (x2, y2), (x3, y3) = _pt(start), _pt(through), _pt(end)
        d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        scale = max(abs(x1), abs(y1), abs(x2), abs(y2), abs(x3), abs(y3), 1.0)
        if abs(d) <= 1e-12 * scale * scale:
            raise ValidationError("the three points are collinear: no arc passes through them")
        s1, s2, s3 = x1 * x1 + y1 * y1, x2 * x2 + y2 * y2, x3 * x3 + y3 * y3
        cx = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d
        cy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d
        radius = hypot(x1 - cx, y1 - cy)
        a1, a2, a3 = (atan2(y - cy, x - cx) for y, x in ((y1, x1), (y2, x2), (y3, x3)))
        ccw_sweep = (a3 - a1) % tau
        if (a2 - a1) % tau < ccw_sweep:     # `through` lies on the counter-clockwise arc
            start_angle, sweep = a1, ccw_sweep
        else:                                # clockwise: store end -> start counter-clockwise
            start_angle, sweep = a3, (a1 - a3) % tau
        if sweep <= 0:
            raise ValidationError("degenerate three-point arc")
        return self._add("arc", {"center": [cx, cy], "radius": radius, "start_angle": start_angle,
                                 "end_angle": start_angle + sweep}, options)

    # -- annotations ----------------------------------------------------------------------
    def add_text(self, position, text: str, height: float, *, rotation: float = 0.0, degrees: bool = False,
                 halign: str = "left", valign: str = "baseline", width: float | None = None,
                 line_spacing: float | None = None, font: str | None = None, **options) -> str:
        """Add text anchored at ``position``.

        ``halign`` is left/center/right and ``valign`` baseline/bottom/middle/top relative to the
        anchor. ``width`` wraps long lines like MTEXT; ``\\n`` starts a new line. No font is
        rendered here, so extents (and therefore ``bounds``) are estimated at 0.6 x height per character.
        """
        geometry = {"position": _pt(position), "text": text, "height": height,
                    "rotation": _angle(rotation, "rotation", degrees), "halign": halign, "valign": valign}
        for key, value in (("width", width), ("line_spacing", line_spacing), ("font", font)):
            if value is not None:
                geometry[key] = value
        return self._add("text", geometry, options)

    def set_text(self, entity_id: str, text: str) -> str:
        """Replace only the string of a text entity."""
        entity = self.get(entity_id)
        if entity["kind"] != "text":
            raise ValidationError("entity is not text", context={"kind": entity["kind"]})
        return self.update(entity_id, geometry={**entity["geometry"], "text": text})

    def _loop(self, item) -> dict:
        """Accept a loop dict, a list of points (polygon), or the id of a closed polyline / circle."""
        if isinstance(item, dict):
            return deepcopy(item)
        if isinstance(item, str):
            entity = self.get(item)
            g = entity["geometry"]
            if entity["kind"] == "circle":
                return {"kind": "circle", "center": list(g["center"]), "radius": g["radius"]}
            if entity["kind"] == "polyline" and g.get("closed"):
                return {"kind": "polygon", "points": [list(p) for p in g["points"]]}
            raise ValidationError("a hatch boundary entity must be a circle or closed polyline",
                                  context={"id": item, "kind": entity["kind"]})
        return {"kind": "polygon", "points": [_pt(p) for p in item]}

    def add_hatch(self, boundary, *, pattern: str = "solid", angle: float = 0.0, degrees: bool = False,
                  spacing: float | None = None, **options) -> str:
        """Add a hatch filling ``boundary`` (even-odd: nested loops alternate fill and hole).

        Each boundary item is a loop dict (``polygon``, ``circle`` or ``path`` of line/arc
        segments), a list of points (polygon), or the id of an existing circle or closed
        polyline. ``pattern`` is ``solid``, ``lines`` or ``cross`` (the latter two need
        ``spacing``). Use ``style={"fill": "#cccccc"}`` / ``{"stroke": ...}`` for colour.
        """
        if isinstance(boundary, (str, dict)):
            boundary = [boundary]
        geometry = {"boundary": [self._loop(item) for item in boundary], "pattern": pattern}
        if pattern != "solid":
            geometry["angle"] = _angle(angle, "angle", degrees)
            if spacing is not None:
                geometry["spacing"] = spacing
        elif spacing is not None:
            raise ValidationError("a solid hatch has no spacing")
        return self._add("hatch", geometry, options)

    def _dimension(self, geometry: dict, options: dict) -> str:
        dim = {k: options.pop(k) for k in list(options) if k in _DIM_OPTIONS and options[k] is not None}
        for k in [k for k in options if k in _DIM_OPTIONS]:
            options.pop(k)
        return self._add("dimension", {**geometry, **dim}, options)

    def add_dimension_linear(self, p1, p2, dimline_point, *, angle: float = 0.0, degrees: bool = False,
                             **options) -> str:
        """Horizontal (``angle=0``) or vertical (``90``) / rotated dimension of the distance between two points
        projected on the dimension-line direction; the line passes through ``dimline_point``.

        Options: ``text`` (``<>`` stands for the value), ``text_height``, ``text_position``, ``precision``,
        ``suffix``, ``arrow_size`` plus the usual entity options.
        """
        geometry = {"dim_type": "linear", "p1": _pt(p1, "p1"), "p2": _pt(p2, "p2"),
                    "dimline_point": _pt(dimline_point, "dimline_point"), "angle": _angle(angle, "angle", degrees)}
        return self._dimension(geometry, options)

    def add_dimension_aligned(self, p1, p2, dimline_point, **options) -> str:
        """Dimension of the true distance between two points; ``dimline_point`` sets the offset side."""
        geometry = {"dim_type": "aligned", "p1": _pt(p1, "p1"), "p2": _pt(p2, "p2"),
                    "dimline_point": _pt(dimline_point, "dimline_point")}
        return self._dimension(geometry, options)

    def add_dimension_angular(self, center, p1, p2, arc_point, **options) -> str:
        """Angle from the ray centre->p1 counter-clockwise to centre->p2; ``arc_point`` sets the arc radius."""
        geometry = {"dim_type": "angular", "center": _pt(center, "center"), "p1": _pt(p1, "p1"),
                    "p2": _pt(p2, "p2"), "arc_point": _pt(arc_point, "arc_point")}
        return self._dimension(geometry, options)

    def add_dimension_radial(self, center, point, **options) -> str:
        """Radius dimension (``R``) from ``center`` to a ``point`` on the circle."""
        return self._dimension({"dim_type": "radial", "center": _pt(center, "center"), "point": _pt(point, "point")},
                               options)

    def add_dimension_diameter(self, center, point, **options) -> str:
        """Diameter dimension (the diameter sign) through ``center`` and a ``point`` on the circle."""
        return self._dimension({"dim_type": "diameter", "center": _pt(center, "center"),
                                "point": _pt(point, "point")}, options)

    def _of_kind(self, entity_id: str, kind: str) -> dict:
        entity = self.get(entity_id)
        if entity["kind"] != kind:
            raise ValidationError("entity is not a %s" % kind, context={"kind": entity["kind"]})
        return entity["geometry"]

    def dimension_value(self, entity_id: str, *, degrees: bool = False) -> float:
        """Measured value: a length in the document unit, or an angle (radians, or degrees with ``degrees``)."""
        g = self._of_kind(entity_id, "dimension")
        value = _ann.dimension_value(g)
        return value * 180.0 / pi if (degrees and g["dim_type"] == "angular") else value

    def dimension_text(self, entity_id: str) -> str:
        """The string a viewer should show (override applied, ``<>`` substituted)."""
        return _ann.dimension_text(self._of_kind(entity_id, "dimension"))

    def dimension_primitives(self, entity_id: str) -> list:
        """The dimension drawn as plain ``line`` / ``arc`` / ``text`` geometry dicts."""
        return _ann.dimension_primitives(self._of_kind(entity_id, "dimension"))

    # -- linetypes ----------------------------------------------------------------------------
    def linetypes(self) -> list:
        """Defined linetypes (the built-in ``continuous`` is implicit): ``{"name", "pattern", ...}``."""
        with self._model._mutation_lock:
            return deepcopy(self._model.linetypes)

    def define_linetype(self, name: str, pattern, *, description: str | None = None, replace: bool = False) -> str:
        """Define a linetype. ``pattern`` is in drawing units: positive = dash, negative = gap, 0 = dot,
        e.g. ``[6, -3]`` dashed, ``[6, -2, 0, -2]`` dash-dot."""
        op = {"op": "define_linetype", "name": name, "pattern": [float(v) for v in pattern], "replace": replace}
        if description is not None:
            op["description"] = description
        return self.apply([op])[0]

    def remove_linetype(self, name: str) -> str:
        """Remove an unused linetype (layers and entities that use it make this fail)."""
        return self.apply([{"op": "remove_linetype", "name": name}])[0]

    # -- blocks -------------------------------------------------------------------------------
    def blocks(self) -> list:
        """Block summaries: ``name``, ``base_point``, ``entity_count``, ``used_by`` (number of inserts)."""
        with self._model._mutation_lock:
            uses: dict = {}
            for e in list(self._model.entities) + [x for b in self._model.blocks for x in b["entities"]]:
                if e["kind"] == "insert":
                    uses[e["geometry"]["block"]] = uses.get(e["geometry"]["block"], 0) + 1
            return [{"name": b["name"], "base_point": list(b["base_point"]), "entity_count": len(b["entities"]),
                     "used_by": uses.get(b["name"], 0), **({"description": b["description"]} if "description" in b else {})}
                    for b in self._model.blocks]

    def get_block(self, name: str) -> dict:
        """A detached copy of one block definition (entities included)."""
        with self._model._mutation_lock:
            for block in self._model.blocks:
                if block["name"] == name:
                    return deepcopy(block)
        raise ReferenceError("unknown block: " + str(name), context={"block": name})

    def define_block(self, name: str, entities, *, base_point=(0.0, 0.0), description: str | None = None,
                     replace: bool = False) -> str:
        """Define a block from entity specs (``{"kind", "geometry"[, "layer", "style", ...]}``).

        ``base_point`` is the point of the block that lands on an insert's position. Entities on layer
        ``"0"`` take the insert's layer when exploded. ``replace=True`` redefines an existing block
        (all its inserts update). Blocks may contain inserts of other blocks, never of themselves.
        """
        op = {"op": "define_block", "name": name, "entities": [deepcopy(spec) for spec in entities],
              "base_point": _pt(base_point, "base_point"), "replace": replace}
        if description is not None:
            op["description"] = description
        return self.apply([op])[0]

    def remove_block(self, name: str) -> str:
        """Remove a block that nothing inserts."""
        return self.apply([{"op": "remove_block", "name": name}])[0]

    def rename_block(self, name: str, new_name: str) -> str:
        """Rename a block and retarget every insert of it."""
        self.apply([{"op": "rename_block", "name": name, "value": new_name}])
        return new_name

    def add_insert(self, block: str, position, *, rotation: float = 0.0, scale: float = 1.0,
                   degrees: bool = False, **options) -> str:
        """Place one instance of ``block`` (uniform positive ``scale``; its base point lands on ``position``)."""
        return self._add("insert", {"block": block, "position": _pt(position),
                                    "rotation": _angle(rotation, "rotation", degrees), "scale": scale}, options)

    def make_block(self, name: str, entity_ids, *, base_point=None, replace_with_insert: bool = True,
                   description: str | None = None) -> dict:
        """Turn existing entities into a block (one revision).

        The block keeps the entities' coordinates; ``base_point`` defaults to the lower-left corner of
        their bounds. With ``replace_with_insert`` (default) the originals are removed and one insert at
        the base point takes their place on the first entity's layer, so the drawing looks unchanged.
        Returns ``{"block": name, "insert": id or None}``.
        """
        ids = _as_ids(entity_ids)
        with self._edit() as model:
            by_id = {e["id"]: e for e in model.entities}
            missing = [i for i in ids if i not in by_id]
            if missing:
                raise ReferenceError("unknown drawing entity in selection", context={"ids": missing[:10]})
            chosen = [by_id[i] for i in ids]
            blocks = {b["name"]: b for b in model.blocks}
            boxes = [entity_bounds(e, blocks) for e in chosen]
            base = _pt(base_point, "base_point") if base_point is not None else [min(b[0][0] for b in boxes),
                                                                                min(b[0][1] for b in boxes)]
            specs = []
            for e in chosen:
                spec = {k: deepcopy(e[k]) for k in ("kind", "geometry", "layer", "name", "metadata", "style") if k in e}
                specs.append(spec)
            operations = [{"op": "define_block", "name": name, "entities": specs, "base_point": base,
                           **({"description": description} if description is not None else {})}]
            if replace_with_insert:
                operations += [{"op": "remove", "id": i} for i in ids]
                operations.append({"op": "add", "kind": "insert", "layer": chosen[0].get("layer", "0"),
                                   "geometry": {"block": name, "position": base, "rotation": 0.0, "scale": 1.0}})
            results = apply_drawing_operations(model, operations)
            return {"block": name, "insert": results[-1] if replace_with_insert else None}

    def explode(self, entity_ids, *, recursive: bool = False) -> list:
        """Replace inserts and dimensions by plain entities (one revision); returns the new ids.

        An insert becomes copies of its block's entities (transformed, new ids; layer ``"0"`` content
        moves to the insert's layer). With ``recursive`` nested inserts are exploded too. A dimension
        becomes its lines, arcs and text.
        """
        ids = _as_ids(entity_ids)
        with self._edit() as model:
            by_id = {e["id"]: e for e in model.entities}
            blocks = {b["name"]: b for b in model.blocks}
            operations, new_count = [], 0

            for i in ids:
                entity = by_id.get(i)
                if entity is None:
                    raise ReferenceError("unknown drawing entity", context={"id": i})
                if entity["kind"] == "insert":
                    produced = expand_insert(entity, blocks, recursive=recursive)
                elif entity["kind"] == "dimension":
                    produced = [{"kind": p["kind"], "geometry": {k: v for k, v in p.items() if k != "kind"},
                                 "layer": entity.get("layer", "0"),
                                 **({"style": deepcopy(entity["style"])} if "style" in entity else {})}
                                for p in _ann.dimension_primitives(entity["geometry"])]
                else:
                    raise ValidationError("only inserts and dimensions can be exploded",
                                          context={"id": i, "kind": entity["kind"]})
                operations.append({"op": "remove", "id": i})
                for p in produced:
                    op = {"op": "add", "kind": p["kind"], "geometry": p["geometry"], "layer": p.get("layer", "0")}
                    for key in ("name", "metadata", "style"):
                        if key in p:
                            op[key] = p[key]
                    operations.append(op)
                    new_count += 1
            if new_count + len(model.entities) > 100_000:
                raise ValidationError("exploding would exceed the entity budget")
            results = apply_drawing_operations(model, operations)
            return [r for op, r in zip(operations, results) if op["op"] == "add"]

    # -- editing ------------------------------------------------------------------------
    def remove(self, entity_ids) -> list:
        """Remove one or more entities in one revision; returns the removed ids."""
        ids = _as_ids(entity_ids)
        return self.apply([{"op": "remove", "id": i} for i in ids])

    def update(self, entity_id: str, *, geometry=None, layer=None, name=None, metadata=None, style=None) -> str:
        """Replace the given fields of an entity (each given field is replaced wholesale).

        ``metadata`` here *replaces* the entity's metadata; use
        :meth:`attach_semantics` to merge. The entity id and kind never change.
        """
        value = {k: v for k, v in (("geometry", geometry), ("layer", layer), ("name", name),
                                   ("metadata", metadata), ("style", style)) if v is not None}
        if not value:
            raise ValidationError("nothing to update")
        if "metadata" in value:
            value["metadata"] = _checked_metadata(value["metadata"])
        return self.apply([{"op": "update", "id": require_id(entity_id), "value": value}])[0]

    def transform(self, entity_ids, matrix) -> list:
        """Apply a planar similarity ``[a,b,tx,c,d,ty,0,0,1]`` (rotation, uniform scale, translation).

        Reflections and non-uniform scales are refused because circles and arcs
        must stay exact. Ids are preserved; all entities change in one revision.
        """
        m = [float(v) for v in vector(matrix, 9, "2D homogeneous matrix")]
        return self.apply([{"op": "transform", "id": i, "value": m} for i in _as_ids(entity_ids)])

    def move(self, entity_ids, dx: float, dy: float) -> list:
        return self.transform(entity_ids, _translation(finite(dx, "dx"), finite(dy, "dy")))

    def rotate(self, entity_ids, angle: float, *, about=(0.0, 0.0), degrees: bool = False) -> list:
        """Rotate counter-clockwise about a point (default the origin)."""
        cx, cy = _pt(about, "about")
        return self.transform(entity_ids, _rotation_about(_angle(angle, "angle", degrees), cx, cy))

    def scale(self, entity_ids, factor: float, *, about=(0.0, 0.0)) -> list:
        """Uniformly scale about a point; ``factor`` must be positive."""
        f = finite(factor, "factor")
        if f <= 0:
            raise ValidationError("scale factor must be positive")
        cx, cy = _pt(about, "about")
        return self.transform(entity_ids, _scale_about(f, cx, cy))

    def duplicate(self, entity_ids, *, dx: float = 0.0, dy: float = 0.0, layer: str | None = None,
                  create_layer: bool = False) -> list:
        """Copy entities (new ids), optionally offset and/or moved to another layer."""
        from .operations import transform_entity
        ids = _as_ids(entity_ids)
        matrix = _translation(finite(dx, "dx"), finite(dy, "dy"))
        with self._edit() as model:
            by_id = {e["id"]: e for e in model.entities}
            operations = []
            for i in ids:
                if i not in by_id:
                    raise ReferenceError("unknown drawing entity", context={"id": i})
                copy = transform_entity(by_id[i], matrix)
                op = {"op": "add", "kind": copy["kind"], "geometry": copy["geometry"],
                      "layer": layer if layer is not None else copy.get("layer", "0")}
                for key in ("name", "metadata", "style"):
                    if key in copy:
                        op[key] = copy[key]
                if create_layer:
                    op["create_layer"] = True
                operations.append(op)
            return apply_drawing_operations(model, operations)

    def attach_semantics(self, entity_id: str, values: dict, *, replace: bool = False,
                         namespace: str | None = None) -> dict:
        """Merge (or replace) semantic metadata on an entity; returns the resulting metadata.

        Keys are ``namespace:name`` or ``slai.*``. ``namespace="aec"`` prefixes
        bare keys, so ``{"category": "wall"}`` is stored as ``aec:category``.
        """
        safe = _checked_metadata(_namespaced(values, namespace))
        current = self._item_metadata(entity_id)
        merged = safe if replace else {**current, **safe}
        self.apply([{"op": "update", "id": entity_id, "value": {"metadata": merged}}])
        return merged

    # -- queries ----------------------------------------------------------------------------
    def _item_metadata(self, entity_id: str) -> dict:
        return deepcopy(self.get(entity_id).get("metadata", {}))

    def get(self, entity_id: str) -> dict:
        """A detached copy of one entity, or ``ReferenceError`` if the id is unknown."""
        require_id(entity_id)
        with self._model._mutation_lock:
            for entity in self._model.entities:
                if entity["id"] == entity_id:
                    return deepcopy(entity)
        raise ReferenceError("unknown drawing entity", context={"id": entity_id})

    def __contains__(self, entity_id) -> bool:
        return isinstance(entity_id, str) and is_valid_id(entity_id) and any(
            e["id"] == entity_id for e in self._model.entities)

    def __len__(self) -> int:
        return len(self._model.entities)

    def _block_map(self) -> dict:
        return {b["name"]: b for b in self._model.blocks}

    def _matching(self, kind, layer, name, metadata, within):
        if kind is not None and kind not in KINDS:
            raise ValidationError("unknown entity kind: " + str(kind), context={"allowed": sorted(KINDS)})
        box = None
        if within is not None:
            lo, hi = (_pt(within[0], "within minimum"), _pt(within[1], "within maximum")) if len(within) == 2 \
                else (None, None)
            if lo is None or lo[0] > hi[0] or lo[1] > hi[1]:
                raise ValidationError("within must be ((xmin, ymin), (xmax, ymax)) with min <= max")
            box = (lo, hi)
        if metadata is not None:
            _checked_metadata(metadata)
        blocks = self._block_map() if within is not None else None
        for entity in self._model.entities:
            if kind is not None and entity["kind"] != kind:
                continue
            if layer is not None and entity.get("layer", "0") != layer:
                continue
            if name is not None and entity.get("name") != name:
                continue
            if metadata is not None:
                have = entity.get("metadata", {})
                if any(k not in have or have[k] != v for k, v in metadata.items()):
                    continue
            if box is not None:
                (x0, y0), (x1, y1) = entity_bounds(entity, blocks)
                if x1 < box[0][0] or x0 > box[1][0] or y1 < box[0][1] or y0 > box[1][1]:
                    continue
            yield entity

    def query_ids(self, *, kind: str | None = None, layer: str | None = None, name: str | None = None,
                  metadata: dict | None = None, within=None, limit: int = 100, offset: int = 0) -> list:
        """Ids of matching entities in document order.

        ``within=((xmin, ymin), (xmax, ymax))`` keeps entities whose tight
        bounds overlap the box (a broad-phase test, not exact intersection).
        ``metadata`` requires every given key to be present with an equal value.
        """
        if type(limit) is not int or not 1 <= limit <= MAX_QUERY_LIMIT:
            raise ValidationError(f"limit must be an integer in [1, {MAX_QUERY_LIMIT}]")
        if type(offset) is not int or offset < 0:
            raise ValidationError("offset must be a non-negative integer")
        with self._model._mutation_lock:
            ids = []
            skipped = 0
            for entity in self._matching(kind, layer, name, metadata, within):
                if skipped < offset:
                    skipped += 1
                    continue
                ids.append(entity["id"])
                if len(ids) >= limit:
                    break
            return ids

    def find(self, **criteria) -> list:
        """Like :meth:`query_ids` but returns detached entity copies."""
        return [self.get(i) for i in self.query_ids(**criteria)]

    def count(self, *, kind: str | None = None, layer: str | None = None) -> int:
        with self._model._mutation_lock:
            return sum(1 for _ in self._matching(kind, layer, None, None, None))

    # -- measurement ------------------------------------------------------------------------------
    def bounds(self, entity_ids=None):
        """Tight ``((xmin, ymin), (xmax, ymax))`` of all (or the given) entities; ``None`` if empty."""
        with self._model._mutation_lock:
            if entity_ids is None:
                entities = self._model.entities
            else:
                wanted = set(_as_ids(entity_ids))
                entities = [e for e in self._model.entities if e["id"] in wanted]
                if len(entities) != len(wanted):
                    raise ReferenceError("unknown drawing entity in selection")
            blocks = self._block_map()
            boxes = [entity_bounds(e, blocks) for e in entities]
        if not boxes:
            return None
        return ((min(b[0][0] for b in boxes), min(b[0][1] for b in boxes)),
                (max(b[1][0] for b in boxes), max(b[1][1] for b in boxes)))

    def length(self, entity_id: str, *, unit: str | None = None) -> float:
        """Length / circumference / perimeter, in the document unit or ``unit``."""
        value = entity_length(self.get(entity_id))
        return value if unit is None or unit == self.unit else convert_length(value, self.unit, require_unit(unit))

    def area(self, entity_id: str, *, unit: str | None = None) -> float:
        """Enclosed area of a circle, simple closed polyline or hatch (even-odd), in unit² (document unit or ``unit``)."""
        value = enclosed_area(self.get(entity_id))
        return value if unit is None or unit == self.unit else convert_area(value, self.unit, require_unit(unit))

    def measure(self, entity_id: str) -> dict:
        """Kind, length, area (when defined), bounds and unit for one entity."""
        entity = self.get(entity_id)
        try:
            area = enclosed_area(entity)
        except ValidationError:
            area = None
        try:
            length = entity_length(entity)
        except ValidationError:
            length = None          # text, dimensions, hatches and inserts have no length
        result = {"id": entity_id, "kind": entity["kind"], "unit": self.unit, "length": length,
                  "area": area, "bounds": entity_bounds(entity, self._block_map())}
        if entity["kind"] == "dimension":
            result["value"] = _ann.dimension_value(entity["geometry"])
            result["text"] = _ann.dimension_text(entity["geometry"])
        return result

    def total_length(self, *, layer: str | None = None, unit: str | None = None) -> float:
        with self._model._mutation_lock:
            value = sum(entity_length(e) for e in self._matching(None, layer, None, None, None)
                        if e["kind"] in ("point", "line", "polyline", "circle", "arc"))
        return value if unit is None or unit == self.unit else convert_length(value, self.unit, require_unit(unit))

    def check_constraints(self, constraints: list, *, absolute_tolerance: float = 1e-8,
                          relative_tolerance: float = 1e-8) -> list[ConstraintResult]:
        """Verify geometric constraints (parallel, perpendicular, horizontal, vertical,
        equal_length, distance, coincident, angle) without modifying the drawing."""
        with self._model._mutation_lock:
            return check_constraints(deepcopy(self._model.entities), constraints,
                                     absolute_tolerance=absolute_tolerance,
                                     relative_tolerance=relative_tolerance)

    # -- units --------------------------------------------------------------------------------------
    def convert_unit(self, unit: str) -> float:
        """Re-express the whole drawing in another length unit (one revision).

        All coordinates and radii are scaled; angles, ids and layers are unchanged.
        Returns the scale factor applied.
        """
        target = require_unit(unit)
        with self._edit() as model:
            if target == model.unit:
                return 1.0
            factor = convert_length(1.0, model.unit, target)
            candidate = model._detached_copy()
            matrix = _scale_about(factor, 0.0, 0.0)

            def rescale(entity):
                if entity["kind"] == "insert":      # block content is rescaled below, so the insert only moves
                    g = entity["geometry"]
                    moved = deepcopy(entity)
                    moved["geometry"]["position"] = [g["position"][0] * factor, g["position"][1] * factor]
                    return moved
                return transform_entity(entity, matrix, preserve_id=True)

            candidate.entities = [rescale(e) for e in candidate.entities]
            for block in candidate.blocks:
                block["entities"] = [rescale(e) for e in block["entities"]]
                block["base_point"] = [block["base_point"][0] * factor, block["base_point"][1] * factor]
            for record in candidate.linetypes:
                record["pattern"] = [v * factor for v in record["pattern"]]
            candidate.unit = target
            candidate.revision = model.revision + 1
            candidate.validate()
            model.entities, model.blocks, model.linetypes = candidate.entities, candidate.blocks, candidate.linetypes
            model.unit, model.revision = target, candidate.revision
            return factor

    # -- copies & identity ---------------------------------------------------------------------------
    def copy(self, *, fresh_id: bool = True, name: str | None = None) -> "DrawingDocument":
        """A writable, unsaved deep copy (fresh document id and revision 1 unless ``fresh_id=False``)."""
        from ..common.ids import new_id
        snapshot = self.snapshot()
        if fresh_id:
            snapshot.document_id = new_id()
            snapshot.revision = 1
        if name is not None:
            snapshot.name = name
        return DrawingDocument(snapshot.validate())

    # -- interchange (see adapters.exchange for the loss-report contract) --------------------------------
    def to_svg(self, *, policy="best_effort"):
        """``(svg_markup, ConversionReport)``."""
        from ..adapters.exchange import drawing_to_svg
        return drawing_to_svg(self.snapshot(), policy=policy)

    def export_svg(self, path, *, overwrite: bool = False, policy="best_effort"):
        """Write an SVG file atomically; returns the :class:`ConversionReport`."""
        from ..common.atomic import write_bytes_atomic
        markup, report = self.to_svg(policy=policy)
        write_bytes_atomic(path, markup.encode("utf-8"), overwrite=overwrite)
        return report

    def export_dxf(self, path, *, overwrite: bool = False, policy="best_effort"):
        """Write DXF R2010 atomically; returns the :class:`ConversionReport`."""
        from ..adapters.exchange import drawing_to_dxf
        return drawing_to_dxf(self.snapshot(), path, policy=policy, overwrite=overwrite)

    def export_dwg(self, path, *, executable, overwrite: bool = False, policy="best_effort",
                   version="ACAD2018", timeout=90):
        """Write DWG through an explicitly configured ODA converter."""
        from ..adapters.exchange import drawing_to_dwg
        return drawing_to_dwg(self.snapshot(), path, executable=executable, policy=policy,
                              overwrite=overwrite, version=version, timeout=timeout)

    @classmethod
    def from_svg(cls, markup, *, unit: str = "mm", name: str = "Imported SVG", preserve_ids: bool = False,
                 policy="best_effort"):
        """``(document, ConversionReport)`` from SVG text/bytes (supported subset only)."""
        from ..adapters.exchange import svg_to_drawing
        drawing, report = svg_to_drawing(markup, unit=unit, policy=policy, preserve_ids=preserve_ids)
        drawing.name = name
        return cls(drawing), report

    @classmethod
    def import_svg(cls, path, **kwargs):
        """Read an SVG file (bounded) and delegate to :meth:`from_svg`."""
        from ..adapters.svg import MAX_SVG_BYTES
        from ..common.errors import ResourceLimitError
        from pathlib import Path
        source = Path(path)
        if not source.is_file() or source.stat().st_size > MAX_SVG_BYTES:
            raise ResourceLimitError("SVG file is absent or exceeds the input budget")
        return cls.from_svg(source.read_bytes(), **kwargs)

    @classmethod
    def import_dxf(cls, path, *, unit: str | None = None, name: str | None = None, policy="best_effort",
                   skip_unsupported: bool = False):
        """``(document, ConversionReport)`` from a DXF file (supported entity subset).

        Unsupported entities raise unless ``skip_unsupported`` is true; then they are omitted and
        listed in ``report.dropped``.
        """
        from ..adapters.exchange import dxf_to_drawing
        drawing, report = dxf_to_drawing(path, unit=unit, policy=policy, skip_unsupported=skip_unsupported)
        drawing.name = name or "Imported DXF"
        return cls(drawing), report

    @classmethod
    def import_dwg(cls, path, *, executable, unit: str = "mm", name: str | None = None, policy="best_effort",
                   version="ACAD2018", timeout=90):
        """``(document, ConversionReport)`` from DWG via an explicitly configured ODA converter."""
        from ..adapters.exchange import dwg_to_drawing
        drawing, report = dwg_to_drawing(path, executable=executable, unit=unit, policy=policy,
                                         version=version, timeout=timeout)
        drawing.name = name or "Imported DWG"
        return cls(drawing), report


def _namespaced(values, namespace):
    if namespace is None:
        return values
    if not isinstance(namespace, str) or not namespace or ":" in namespace or namespace.startswith("slai."):
        raise ValidationError("namespace must be a non-empty string without ':'")
    if not isinstance(values, dict):
        raise ValidationError("metadata must be a mapping")
    return {(k if (isinstance(k, str) and (":" in k or k.startswith("slai."))) else f"{namespace}:{k}"): v
            for k, v in values.items()}
