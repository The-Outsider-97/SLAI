"""DXF exchange through ezdxf (an explicitly optional backend).

SDRW 0.2 content maps as follows (full detail in the README):

=============  ==========================================================================
SDRW           DXF
=============  ==========================================================================
point, line,   POINT, LINE, LWPOLYLINE, CIRCLE, ARC (as before)
polyline,
circle, arc
text           TEXT (one line) or MTEXT (several lines / wrapping width)
dimension      DIMENSION (linear, aligned, angular 3-point, radius, diameter) with its
               rendered block, so any viewer shows the drawing
hatch          HATCH: polyline / edge boundary paths, solid fill or user-defined lines
insert, block  INSERT / BLOCK (uniform positive scale; rotation)
linetype       LTYPE table entries (dash / gap / dot elements)
layer, entity  colour (true colour), lineweight, linetype, linetype scale
style
=============  ==========================================================================

Import is the reverse. What cannot be represented is either refused (the default,
fail-closed) or -- with ``skip_unsupported=True`` -- skipped; either way every
approximation and omission is recorded in a :class:`~.dxf_map.Notes` log that the
conversion layer turns into the report and checks against the caller's policy.
"""
from __future__ import annotations

import os
import re
import tempfile
from math import atan2, cos, degrees, hypot, radians, sin, tau
from pathlib import Path

from ..common import DependencyUnavailableError, UnsupportedFeatureError, ValidationError
from ..common.errors import ResourceLimitError
from ..sdrw import Drawing
from ..sdrw import annotations as ann
from ..sdrw.entities import make_entity
from .contracts import Capabilities
from .dxf_map import Notes, hex_color, mtext_factor, plan_export, spacing_from_factor, stroke_width_for

_FULL_SWEEP = tau - 1e-9
_ANGLE_EPS = 1e-9  # degrees
_UNIT_CODES = {"in": 1, "ft": 2, "mm": 4, "cm": 5, "m": 6, "km": 7}
_MAX_DXF_BYTES = 64 * 1024 * 1024
_MAX_EXPLODE_DEPTH = 8
_ARROW_BLOCKS = frozenset({"_DOT", "_DOTSMALL", "_DOTBLANK", "_ORIGIN", "_ORIGIN2", "_OPEN", "_OPEN90", "_OPEN30",
                           "_CLOSED", "_SMALL", "_NONE", "_OBLIQUE", "_BOXFILLED", "_BOXBLANK", "_CLOSEDBLANK",
                           "_CLOSEDFILLED", "_DATUMFILLED", "_DATUMBLANK", "_INTEGRAL", "_ARCHTICK"})
_SUPPORTED = ("POINT", "LINE", "LWPOLYLINE", "CIRCLE", "ARC", "TEXT", "MTEXT", "DIMENSION", "HATCH", "INSERT")


def capabilities():
    try:
        _ezdxf()
        available = True
    except DependencyUnavailableError:
        available = False
    return Capabilities("dxf", "ezdxf", available, ("R2010",) if available else (), ("R2010",) if available else (),
                        _SUPPORTED,
                        ("Only model-space planar entities; paper space refused",
                         "Bulges, OCS extrusion and elevated entities refused",
                         "DIMENSION: linear/aligned/angular-3p/radius/diameter map natively, other types are "
                         "exploded into lines and text",
                         "HATCH: solid and parallel-line / cross patterns; other patterns are approximated",
                         "INSERT: uniform scale only, others are exploded",
                         "TEXT/MTEXT: plain text only, inline formatting is removed",
                         "Entity handles, XDATA, plot styles and transparency are not preserved"))


def _ezdxf():
    try:
        import ezdxf
    except ImportError as exc:
        raise DependencyUnavailableError("install ezdxf for DXF exchange") from exc
    return ezdxf


def _arc_angles(g):
    """DXF arc angles (degrees, CCW, start in [0,360)); None for a full circle."""
    sweep = g["end_angle"] - g["start_angle"]
    if sweep >= _FULL_SWEEP:
        return None
    start = degrees(g["start_angle"]) % 360.0
    return start, (start + degrees(sweep)) % 360.0


# ===================================================================================== export
_ATTACH = {("top", "left"): 1, ("top", "center"): 2, ("top", "right"): 3,
           ("middle", "left"): 4, ("middle", "center"): 5, ("middle", "right"): 6,
           ("bottom", "left"): 7, ("bottom", "center"): 8, ("bottom", "right"): 9}


def _text_alignment(valign: str, halign: str):
    from ezdxf.enums import TextEntityAlignment as A
    row = {"baseline": "", "bottom": "BOTTOM_", "middle": "MIDDLE_", "top": "TOP_"}[valign]
    return getattr(A, row + halign.upper())


def _mtext_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", "\\P")


class _Writer:
    def __init__(self, ezdxf, doc, plan):
        self.ezdxf, self.doc, self.plan = ezdxf, doc, plan

    # -- one entity -------------------------------------------------------------------------
    def emit(self, layout, entity):
        kind, g = entity["kind"], entity["geometry"]
        attrs = {"layer": self.plan.layers[entity.get("layer", "0")], **self.plan.attribs.get(entity["id"], {})}
        if kind == "point":
            layout.add_point((*g["position"], 0), dxfattribs=attrs)
        elif kind == "line":
            layout.add_line((*g["start"], 0), (*g["end"], 0), dxfattribs=attrs)
        elif kind == "polyline":
            layout.add_lwpolyline(g["points"], close=g.get("closed", False), dxfattribs=attrs)
        elif kind == "circle":
            layout.add_circle((*g["center"], 0), g["radius"], dxfattribs=attrs)
        elif kind == "arc":
            angles = _arc_angles(g)
            if angles is None:
                layout.add_circle((*g["center"], 0), g["radius"], dxfattribs=attrs)
            else:
                layout.add_arc((*g["center"], 0), g["radius"], angles[0], angles[1], dxfattribs=attrs)
        elif kind == "text":
            self.text(layout, g, attrs)
        elif kind == "hatch":
            self.hatch(layout, g, attrs)
        elif kind == "dimension":
            self.dimension(layout, g, attrs)
        elif kind == "insert":
            s = float(g.get("scale", 1.0))
            layout.add_blockref(self.plan.blocks[g["block"]], tuple(g["position"]),
                                dxfattribs={**attrs, "xscale": s, "yscale": s, "zscale": s,
                                            "rotation": degrees(g.get("rotation", 0.0))})
        else:  # pragma: no cover - the entity kinds are closed by the schema
            raise UnsupportedFeatureError("entity kind has no DXF mapping: " + kind)

    # -- text --------------------------------------------------------------------------------
    def text(self, layout, g, attrs):
        height, rotation = g["height"], g.get("rotation", 0.0)
        text = g["text"].replace("\t", "    ")
        halign, valign = g.get("halign", "left"), g.get("valign", "baseline")
        if "font" in g:
            attrs = {**attrs, "style": self.plan.fonts[g["font"]][0]}
        px, py = g["position"]
        if "\n" not in text and not g.get("width"):
            entity = layout.add_text(text, height=height, rotation=degrees(rotation), dxfattribs=attrs)
            entity.set_placement((px, py), align=_text_alignment(valign, halign))
            return
        layout_info = ann.text_layout(g)
        if valign == "baseline":
            # MTEXT cannot anchor on the first baseline: anchor the bottom line, shifted so the text stays put
            drop = layout_info["height"] - height
            px, py = px + sin(rotation) * drop, py - cos(rotation) * drop
            valign = "bottom"
        options = {**attrs, "insert": (px, py), "char_height": height, "rotation": degrees(rotation),
                   "attachment_point": _ATTACH[(valign, halign)]}
        if g.get("width"):
            options["width"] = g["width"]
        if len(layout_info["lines"]) > 1:
            options["line_spacing_factor"] = mtext_factor(g.get("line_spacing", 1.0))
        layout.add_mtext(_mtext_escape(text), dxfattribs=options)

    # -- hatch -------------------------------------------------------------------------------
    def hatch(self, layout, g, attrs):
        hatch = layout.add_hatch(dxfattribs=attrs)
        for loop in g["boundary"]:
            kind = loop["kind"]
            if kind == "polygon":
                hatch.paths.add_polyline_path([tuple(p) for p in loop["points"]], is_closed=True)
            elif kind == "circle":
                hatch.paths.add_edge_path().add_arc(tuple(loop["center"]), loop["radius"], 0, 360)
            else:
                edges = hatch.paths.add_edge_path()
                for seg, reverse in ann.chain_segments(loop["segments"]):
                    if seg["kind"] == "line":
                        a, b = (seg["end"], seg["start"]) if reverse else (seg["start"], seg["end"])
                        edges.add_line(tuple(a), tuple(b))
                    else:
                        edges.add_arc(tuple(seg["center"]), seg["radius"], degrees(seg["start_angle"]) % 360.0,
                                      degrees(seg["end_angle"]) % 360.0 or 360.0, ccw=not reverse)
        pattern = g.get("pattern", "solid")
        if pattern == "solid":
            hatch.set_solid_fill(style=0)
            return
        angle, spacing = g.get("angle", 0.0), g["spacing"]
        angles = [angle] if pattern == "lines" else [angle, angle + tau / 4]
        definition = [(degrees(a), (0.0, 0.0), (-spacing * sin(a), spacing * cos(a)), []) for a in angles]
        hatch.set_pattern_fill("SLAI_" + pattern.upper(), angle=0.0, scale=1.0, style=0, definition=definition)

    # -- dimensions --------------------------------------------------------------------------
    def dimension(self, layout, g, attrs):
        kind = g["dim_type"]
        height = ann.dimension_text_height(g)
        arrow = g.get("arrow_size", 0.6 * height)
        precision = g.get("precision", 2)
        # ezdxf's stock dimension styles carry DIMLFAC=100 and suppress trailing zeros: pin everything the
        # displayed text depends on, so the DXF shows exactly what SDRW's dimension_text() says.
        override = {"dimtxt": height, "dimasz": arrow, "dimdec": precision, "dimexe": 0.0, "dimexo": 0.0,
                    "dimgap": 0.2 * height, "dimscale": 1.0, "dimlfac": 1.0, "dimzin": 0}
        if kind == "angular":
            override.update({"dimadec": precision, "dimazin": 0})
        elif g.get("suffix"):
            override["dimpost"] = "<>" + g["suffix"]
        text = g.get("text", "<>")
        common = {"text": text if text else " ", "override": override, "dxfattribs": attrs}
        if kind == "linear":
            dim = layout.add_linear_dim(base=tuple(g["dimline_point"]), p1=tuple(g["p1"]), p2=tuple(g["p2"]),
                                        angle=degrees(g.get("angle", 0.0)), **common)
        elif kind == "aligned":
            p1, p2, q = g["p1"], g["p2"], g["dimline_point"]
            length = hypot(p2[0] - p1[0], p2[1] - p1[1])
            dx, dy = (p2[0] - p1[0]) / length, (p2[1] - p1[1]) / length
            distance = dx * (q[1] - p1[1]) - dy * (q[0] - p1[0])          # signed: positive on the left of p1->p2
            dim = layout.add_aligned_dim(p1=tuple(p1), p2=tuple(p2), distance=distance, **common)
        elif kind == "angular":
            dim = layout.add_angular_dim_3p(base=tuple(g["arc_point"]), center=tuple(g["center"]),
                                            p1=tuple(g["p1"]), p2=tuple(g["p2"]), **common)
        elif kind == "radial":
            dim = layout.add_radius_dim(center=tuple(g["center"]), mpoint=tuple(g["point"]), **common)
        else:
            dim = layout.add_diameter_dim(center=tuple(g["center"]), mpoint=tuple(g["point"]), **common)
        if "text_position" in g:
            dim.set_location(tuple(g["text_position"]))
        dim.render()


def build_dxf(drawing: Drawing, plan=None):
    """Build the ezdxf document for ``drawing``; returns ``(document, plan)``. Nothing is written."""
    ezdxf = _ezdxf()
    drawing.validate()
    plan = plan or plan_export(drawing)
    doc = ezdxf.new("R2010", setup=["styles", "dimstyles"] if plan.has_dimensions else False)
    doc.units = _UNIT_CODES[drawing.unit]
    writer = _Writer(ezdxf, doc, plan)
    try:
        for record in drawing.linetypes:
            elements = [float(v) for v in record["pattern"]]
            doc.linetypes.add(plan.linetypes[record["name"]], pattern=[sum(abs(v) for v in elements), *elements],
                              description=record.get("description", record["name"]))
        for _font, (style, file) in plan.fonts.items():
            doc.styles.new(style, dxfattribs={"font": file})
        for layer in drawing.layers:
            name = plan.layers[layer["name"]]
            attribs = plan.layer_attribs.get(layer["name"], {})
            entry = doc.layers.get("0") if name == "0" else doc.layers.new(name)
            for key, value in attribs.items():
                if key != "ltscale":
                    entry.dxf.set(key, value)
            if not layer.get("visible", True):
                entry.off()
        for block in drawing.blocks:
            layout = doc.blocks.new(plan.blocks[block["name"]], base_point=tuple(block["base_point"]))
            for inner in block["entities"]:
                writer.emit(layout, inner)
        msp = doc.modelspace()
        for entity in drawing.entities:
            writer.emit(msp, entity)
    except (ezdxf.DXFError, ValueError, ZeroDivisionError) as exc:
        raise UnsupportedFeatureError("drawing cannot be represented as DXF: " + str(exc)) from exc
    return doc, plan


def save_dxf(doc, path) -> None:
    """Atomically write an ezdxf document to ``path``."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=".slai-dxf-", suffix=".tmp", dir=destination.parent)
    os.close(handle)
    try:
        doc.saveas(name)
        os.replace(name, destination)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def export_dxf(drawing: Drawing, path) -> Notes:
    """Write the drawing as a DXF R2010 file (atomic replace of ``path``) and return the loss notes.

    A full-sweep SDRW arc is exported as an exactly equivalent CIRCLE because a
    DXF ARC with equal start and end angles is ambiguous.
    """
    doc, plan = build_dxf(drawing)
    save_dxf(doc, path)
    return plan.notes


# ===================================================================================== import
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_PERCENT = re.compile(r"%%([dDpPcC%uUoO])")
_FORMATTING = re.compile(r"\\[A-Za-z~]|[{}]")
_PERCENT_CHARS = {"d": "\N{DEGREE SIGN}", "p": "\N{PLUS-MINUS SIGN}", "c": "\N{DIAMETER SIGN}", "%": "%"}
_TEXT_ALIGNS = {"LEFT": ("left", "baseline"), "CENTER": ("center", "baseline"), "RIGHT": ("right", "baseline"),
                "BOTTOM_LEFT": ("left", "bottom"), "BOTTOM_CENTER": ("center", "bottom"),
                "BOTTOM_RIGHT": ("right", "bottom"), "MIDDLE_LEFT": ("left", "middle"),
                "MIDDLE_CENTER": ("center", "middle"), "MIDDLE_RIGHT": ("right", "middle"),
                "TOP_LEFT": ("left", "top"), "TOP_CENTER": ("center", "top"), "TOP_RIGHT": ("right", "top")}
_ATTACH_BACK = {n: k for k, n in _ATTACH.items()}


def _clean_text(text: str) -> str:
    return _CTRL.sub(" ", text.replace("\r\n", "\n").replace("\r", "\n"))


def _vec(point) -> list:
    return [float(point[0]), float(point[1])]


def _dedupe(points) -> list:
    out = []
    for p in points:
        if not out or hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > 1e-12 * max(1.0, abs(p[0]), abs(p[1])):
            out.append(p)
    if len(out) > 1 and hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) <= 1e-12 * max(1.0, abs(out[0][0])):
        out.pop()
    return out


class _Importer:
    def __init__(self, ezdxf, doc, unit, skip_unsupported):
        self.ezdxf, self.doc, self.skip = ezdxf, doc, skip_unsupported
        self.unit = unit
        self.notes = Notes()
        self.lt_names: dict = {}            # lower-case DXF linetype name -> SDRW name
        self.layer_colored: set = set()     # lower-case layer names with an explicit colour
        self.layer_linetype: dict = {}      # lower-case layer name -> SDRW linetype
        self.dxf_blocks = {b.name.lower(): b for b in doc.blocks if not b.name.startswith("*")}
        self.block_state: dict = {}         # canonical name -> "pending" | "ok" | "bad"
        self.block_specs: dict = {}
        self.block_order: list = []
        self.skipped_blocks: list = []
        self.solid_ok = 0                   # >0 while exploding a dimension (SOLID arrowheads become hatches)

    # -- loss handling ------------------------------------------------------------------------
    def refuse(self, message: str, label: str):
        """Fail closed by default; with ``skip_unsupported`` record the omission and carry on."""
        if not self.skip:
            raise UnsupportedFeatureError(message)
        self.notes.add("dropped", "%s: %s" % (label, message))
        return []

    def _finish(self, e, kind, geometry, *, style=None, fill=False, label=None):
        """Validate one converted entity; returns ``[spec]`` or (skipping) ``[]``."""
        spec = {"kind": kind, "geometry": geometry, "layer": e.dxf.get("layer", "0") or "0"}
        style = style if style is not None else self.style(e, fill=fill)
        if style:
            spec["style"] = style
        try:
            make_entity(kind, geometry, layer=spec["layer"], style=style or None)
        except ValidationError as exc:
            if not self.skip:
                raise
            self.notes.add("dropped", "%s: %s" % (label or e.dxftype(), exc))
            return []
        return [spec]

    # -- styles -------------------------------------------------------------------------------
    def _rgb(self, true_color, aci, *, layer_name=None):
        from ezdxf import colors
        if true_color is not None:
            return tuple(colors.int2rgb(true_color))[:3]
        if aci is not None and 1 <= aci <= 255 and (aci != 7 or (layer_name or "").lower() in self.layer_colored):
            return tuple(colors.aci2rgb(aci))[:3]
        return None

    def style(self, e, *, fill=False) -> dict:
        style = {}
        layer = (e.dxf.get("layer", "0") or "0").lower()
        aci = e.dxf.get("color", 256)
        if aci == 0:
            self.notes.add("approximations", "BYBLOCK colours not preserved (entities take the layer colour)")
        rgb = self._rgb(e.dxf.get("true_color"), aci if aci != 256 else None, layer_name=layer)
        if rgb is not None:
            style["fill" if fill else "stroke"] = hex_color(rgb)
        weight = e.dxf.get("lineweight", -1)
        if weight is not None and weight >= 0:
            style["stroke_width"] = stroke_width_for(weight, self.unit)
        name = e.dxf.get("linetype", "BYLAYER") or "BYLAYER"
        low = name.lower()
        if low == "byblock":
            self.notes.add("approximations", "BYBLOCK linetypes not preserved (entities take the layer linetype)")
        elif low == "continuous":
            if self.layer_linetype.get(layer):
                style["linetype"] = "continuous"
        elif low != "bylayer":
            if low in self.lt_names:
                style["linetype"] = self.lt_names[low]
            else:
                self.notes.add("approximations", "reference to undefined linetype %r ignored" % name)
        scale = e.dxf.get("ltscale", 1.0)
        if scale is not None and scale > 0 and abs(scale - 1.0) > 1e-9:
            style["linetype_scale"] = float(scale)
        return style

    # -- planarity ---------------------------------------------------------------------------
    def _extrusion_ok(self, e) -> bool:
        return tuple(float(v) for v in e.dxf.get("extrusion", (0, 0, 1))) == (0.0, 0.0, 1.0)

    # -- dispatch ----------------------------------------------------------------------------
    def convert(self, e, depth: int = 0) -> list:
        kind = e.dxftype()
        handler = getattr(self, "_c_" + kind.lower(), None)
        if handler is None or (kind == "SOLID" and not self.solid_ok):
            return self.refuse("unsupported DXF entity: " + kind, kind)
        if kind not in ("DIMENSION",) and not self._extrusion_ok(e):
            return self.refuse("DXF OCS extrusion requires explicit conversion", kind)
        return handler(e, depth)

    # -- the original five -------------------------------------------------------------------
    def _c_point(self, e, depth):
        if abs(float(e.dxf.location[2])) > 1e-9:
            return self.refuse("3D/elevated DXF entity cannot be silently projected", "POINT")
        return self._finish(e, "point", {"position": _vec(e.dxf.location)})

    def _c_line(self, e, depth):
        if any(abs(float(p[2])) > 1e-9 for p in (e.dxf.start, e.dxf.end)):
            return self.refuse("3D/elevated DXF entity cannot be silently projected", "LINE")
        return self._finish(e, "line", {"start": _vec(e.dxf.start), "end": _vec(e.dxf.end)})

    def _c_lwpolyline(self, e, depth):
        if abs(float(e.dxf.get("elevation", 0.0))) > 1e-9:
            return self.refuse("elevated DXF polylines require projection", "LWPOLYLINE")
        if any(abs(float(p[2])) > 1e-12 for p in e.get_points("xyb")):
            return self.refuse("DXF polyline bulge arcs require exact curve decomposition", "LWPOLYLINE")
        return self._finish(e, "polyline", {"points": [list(p[:2]) for p in e.get_points()],
                                            "closed": bool(e.closed)})

    def _c_circle(self, e, depth):
        if abs(float(e.dxf.center[2])) > 1e-9:
            return self.refuse("3D/elevated DXF entity cannot be silently projected", "CIRCLE")
        return self._finish(e, "circle", {"center": _vec(e.dxf.center), "radius": e.dxf.radius})

    def _c_arc(self, e, depth):
        if abs(float(e.dxf.center[2])) > 1e-9:
            return self.refuse("3D/elevated DXF entity cannot be silently projected", "ARC")
        start, end = float(e.dxf.start_angle) % 360.0, float(e.dxf.end_angle) % 360.0
        if abs(end - start) <= _ANGLE_EPS or abs(abs(end - start) - 360.0) <= _ANGLE_EPS:
            # Equal angles are ambiguous (empty arc vs. full circle): never guess.
            return self.refuse("DXF ARC with equal start and end angles is ambiguous", "ARC")
        if end < start:
            end += 360.0
        return self._finish(e, "arc", {"center": _vec(e.dxf.center), "radius": e.dxf.radius,
                                       "start_angle": radians(start), "end_angle": radians(end)})

    # -- text --------------------------------------------------------------------------------
    def _font_of(self, e):
        name = e.dxf.get("style", "Standard") or "Standard"
        if name.lower() == "standard":
            return None
        try:
            font = self.doc.styles.get(name).dxf.get("font", "") or ""
        except (self.ezdxf.DXFError, KeyError):
            return None
        stem = re.sub(r"\.[A-Za-z0-9]{2,4}$", "", font)
        return stem[:255] or None

    def _text_entity(self, e, text, height, position, rotation, halign, valign, *, width=None, spacing=None):
        text = _clean_text(text)
        if len(text) > ann.MAX_TEXT_LENGTH:
            return self.refuse("text longer than %d characters" % ann.MAX_TEXT_LENGTH, e.dxftype())
        if not text.strip():
            self.notes.add("dropped", "empty text entities")
            return []
        if not height or height <= 0:
            return self.refuse("text height must be positive", e.dxftype())
        geometry = {"position": _vec(position), "text": text, "height": float(height), "rotation": rotation,
                    "halign": halign, "valign": valign}
        if width:
            geometry["width"] = float(width)
        if spacing is not None:
            geometry["line_spacing"] = spacing
        font = self._font_of(e)
        if font:
            geometry["font"] = font
        return self._finish(e, "text", geometry)

    def _c_text(self, e, depth):
        if abs(float(e.dxf.insert[2])) > 1e-9:
            return self.refuse("3D/elevated DXF entity cannot be silently projected", "TEXT")
        align, p1, p2 = e.get_placement()
        rotation = radians(float(e.dxf.get("rotation", 0.0)))
        if align.name in ("ALIGNED", "FIT"):
            if p2 is not None:
                rotation = atan2(p2[1] - p1[1], p2[0] - p1[0])
            self.notes.add("approximations", "ALIGNED/FIT text imported as left-aligned text of its own height")
            halign, valign = "left", "baseline"
        else:
            halign, valign = _TEXT_ALIGNS[align.name]
        if abs(float(e.dxf.get("width", 1.0)) - 1.0) > 1e-9 or abs(float(e.dxf.get("oblique", 0.0))) > 1e-9:
            self.notes.add("approximations", "text width factor and oblique angle ignored")
        if e.dxf.get("text_generation_flag", 0):
            self.notes.add("approximations", "mirrored text imported unmirrored")
        text = _PERCENT.sub(lambda m: _PERCENT_CHARS.get(m.group(1).lower(), ""), e.dxf.text)
        return self._text_entity(e, text, e.dxf.height, p1, rotation, halign, valign)

    def _c_mtext(self, e, depth):
        if abs(float(e.dxf.insert[2])) > 1e-9:
            return self.refuse("3D/elevated DXF entity cannot be silently projected", "MTEXT")
        raw = e.text
        core = raw.replace("\\\\", "").replace("\\P", "").replace("\\{", "").replace("\\}", "")
        if _FORMATTING.search(core):
            self.notes.add("approximations", "inline MTEXT formatting (fonts, colours, stacking) removed")
        text = e.plain_text(split=False)
        valign_name, halign_name = _ATTACH_BACK[int(e.dxf.get("attachment_point", 1))]
        width = e.dxf.get("width") or None
        spacing = None
        if "\n" in text or width:
            spacing = spacing_from_factor(float(e.dxf.get("line_spacing_factor", 1.0)))
        return self._text_entity(e, text, e.dxf.get("char_height", 0.0), e.dxf.insert, radians(e.get_rotation()),
                                 halign_name, valign_name, width=width, spacing=spacing)

    # -- inserts and blocks -------------------------------------------------------------------
    def convert_block(self, name: str) -> bool:
        """Convert a block (once); True when it is usable. Dependencies finish first."""
        state = self.block_state.get(name)
        if state is not None:
            return state == "ok"
        self.block_state[name] = "pending"
        block = self.dxf_blocks[name.lower()]
        try:
            if block.block is not None and int(block.block.dxf.get("flags", 0)) & 4:
                raise UnsupportedFeatureError("external-reference blocks are not supported")
            specs = []
            for inner in block:
                specs.extend(self.convert(inner, 0))
        except BaseException:
            self.block_state[name] = "bad"
            raise
        if not specs:
            self.block_state[name] = "bad"
            return False
        base = block.block.dxf.get("base_point", (0, 0, 0)) if block.block is not None else (0, 0, 0)
        self.block_specs[name] = (specs, _vec(base))
        self.block_state[name] = "ok"
        self.block_order.append(name)
        return True

    def _c_insert(self, e, depth):
        if abs(float(e.dxf.insert[2])) > 1e-9:
            return self.refuse("3D/elevated DXF entity cannot be silently projected", "INSERT")
        if int(e.dxf.get("column_count", 1)) > 1 or int(e.dxf.get("row_count", 1)) > 1:
            parts = list(e.multi_insert())
            self.notes.add("approximations", "MINSERT arrays expanded into individual inserts")
            return [spec for part in parts for spec in self._c_insert(part, depth)]
        if len(e.attribs):
            self.notes.add("dropped", "block attribute values (ATTRIB) on inserts")
        name = e.dxf.name
        block = self.dxf_blocks.get(name.lower())
        xs, ys = float(e.dxf.get("xscale", 1.0)), float(e.dxf.get("yscale", 1.0))
        uniform = xs != 0 and abs(xs - ys) <= 1e-9 * max(abs(xs), abs(ys))
        if block is not None and uniform and self.convert_block(block.name):
            rotation = radians(float(e.dxf.get("rotation", 0.0)))
            if xs < 0:                      # equal negative scales are a half turn
                rotation += tau / 2
            geometry = {"block": block.name, "position": _vec(e.dxf.insert), "rotation": rotation,
                        "scale": abs(xs)}
            return self._finish(e, "insert", geometry)
        if depth >= _MAX_EXPLODE_DEPTH:
            return self.refuse("blocks nest too deeply to explode", "INSERT")
        reason = ("mirrored or non-uniformly scaled" if block is not None and not uniform else
                  "anonymous or unusable block")
        try:
            parts = list(e.virtual_entities())
        except (self.ezdxf.DXFError, ValueError, KeyError) as exc:
            return self.refuse("INSERT of %r cannot be imported (%s)" % (name, exc), "INSERT")
        self.notes.add("approximations", "INSERT of %s exploded into plain entities (%s)" % ("a block", reason))
        return [spec for part in parts for spec in self.convert(part, depth + 1)]

    # -- hatch -------------------------------------------------------------------------------
    def _flatten(self, hatch, path):
        import ezdxf.path as epath
        from ezdxf.math import BoundingBox
        built = epath.from_hatch_boundary_path(path, ocs=hatch.ocs(),
                                               elevation=float(hatch.dxf.get("elevation", (0, 0, 0))[2]))
        box = BoundingBox(list(built.control_vertices()))
        size = max(box.size.x, box.size.y) if box.has_data else 1.0
        points = _dedupe([[v.x, v.y] for v in built.flattening(max(size * 1e-3, 1e-9))])
        return {"kind": "polygon", "points": points} if len(points) >= 3 else None

    def _loop(self, hatch, path):
        kind = type(path).__name__
        segments, flat_only = [], False
        if kind == "PolylinePath":
            from ezdxf.math import bulge_to_arc
            verts = [(float(v[0]), float(v[1]), float(v[2]) if len(v) > 2 else 0.0) for v in path.vertices]
            if len(verts) < 2:
                return None
            if all(abs(v[2]) < 1e-12 for v in verts):
                points = _dedupe([[v[0], v[1]] for v in verts])
                return {"kind": "polygon", "points": points} if len(points) >= 3 else None
            for i, (x, y, bulge) in enumerate(verts):
                nx, ny, _ = verts[(i + 1) % len(verts)]
                if hypot(nx - x, ny - y) < 1e-12:
                    continue
                if abs(bulge) < 1e-12:
                    segments.append({"kind": "line", "start": [x, y], "end": [nx, ny]})
                else:
                    center, a0, a1, radius = bulge_to_arc((x, y), (nx, ny), bulge)
                    if a1 <= a0:
                        a1 += tau
                    segments.append({"kind": "arc", "center": [center[0], center[1]], "radius": float(radius),
                                     "start_angle": a0, "end_angle": a1})
        elif kind == "EdgePath":
            for edge in path.edges:
                name = type(edge).__name__
                if name == "LineEdge":
                    if hypot(edge.end[0] - edge.start[0], edge.end[1] - edge.start[1]) >= 1e-12:
                        segments.append({"kind": "line", "start": _vec(edge.start), "end": _vec(edge.end)})
                elif name == "ArcEdge":
                    sweep = (float(edge.end_angle) - float(edge.start_angle)) % 360.0
                    if sweep < 1e-9:
                        sweep = 360.0
                    a0 = radians(float(edge.start_angle))
                    segments.append({"kind": "arc", "center": _vec(edge.center), "radius": float(edge.radius),
                                     "start_angle": a0, "end_angle": a0 + radians(sweep)})
                else:
                    flat_only = True
        else:
            flat_only = True
        if not flat_only and segments:
            if len(segments) == 1 and segments[0]["kind"] == "arc" and \
                    abs(segments[0]["end_angle"] - segments[0]["start_angle"] - tau) < 1e-9:
                loop = {"kind": "circle", "center": segments[0]["center"], "radius": segments[0]["radius"]}
            else:
                loop = {"kind": "path", "segments": segments}
            try:
                ann.validate_hatch_loop(loop)
                return loop
            except ValidationError:
                pass
        loop = self._flatten(hatch, path)
        if loop is not None:
            self.notes.add("approximations", "hatch boundaries with curves other than lines/arcs (or not exactly "
                                             "closed) flattened to polygons")
        return loop

    def _pattern(self, e):
        """``(pattern, angle, spacing)`` from the pattern definition lines."""
        lines = list(e.pattern.lines) if e.pattern is not None else []
        if not lines:
            self.notes.add("approximations", "hatch pattern without a definition imported as solid")
            return "solid", None, None
        if any(line.dash_length_items for line in lines):
            self.notes.add("approximations", "dashed hatch pattern lines drawn continuous")
        spacings = []
        for line in lines:
            a = radians(float(line.angle))
            spacings.append(abs(-sin(a) * float(line.offset[0]) + cos(a) * float(line.offset[1])))
        usable = [(float(line.angle), s) for line, s in zip(lines, spacings) if s > 1e-12]
        if not usable:
            self.notes.add("approximations", "hatch pattern without line spacing imported as solid")
            return "solid", None, None
        if len(usable) == 2 and abs(((usable[1][0] - usable[0][0]) % 180.0) - 90.0) < 1e-6 and \
                abs(usable[0][1] - usable[1][1]) <= 1e-6 * usable[0][1]:
            return "cross", radians(min(usable[0][0], usable[1][0])), usable[0][1]
        if len(usable) > 1 or len(lines) > len(usable):
            self.notes.add("approximations", "hatch patterns with several line families approximated by their "
                                             "first line family")
        return "lines", radians(usable[0][0]), usable[0][1]

    def _c_hatch(self, e, depth):
        if abs(float(e.dxf.get("elevation", (0, 0, 0))[2])) > 1e-9:
            return self.refuse("elevated DXF hatches require projection", "HATCH")
        loops = [loop for loop in (self._loop(e, path) for path in e.paths) if loop is not None]
        if not loops:
            self.notes.add("dropped", "hatches without a usable boundary")
            return []
        if int(e.dxf.get("hatch_style", 0)) != 0:
            self.notes.add("approximations", "hatch styles 'outer' and 'ignore' treated as normal (even-odd)")
        if e.dxf.get("associative", 0):
            self.notes.add("approximations", "hatch associativity with its boundary entities is not preserved")
        if e.has_gradient_data:
            self.notes.add("approximations", "gradient fills replaced by a solid colour")
        geometry = {"boundary": loops}
        solid = e.has_solid_fill or e.has_gradient_data
        if solid:
            geometry["pattern"] = "solid"
        else:
            pattern, angle, spacing = self._pattern(e)
            geometry["pattern"] = pattern
            if pattern != "solid":
                geometry["angle"], geometry["spacing"] = angle, spacing
        return self._finish(e, "hatch", geometry, fill=geometry["pattern"] == "solid")

    def _c_solid(self, e, depth):          # only reached while exploding a dimension (arrowheads)
        pts = [e.dxf.get("vtx%d" % i) for i in range(4) if e.dxf.get("vtx%d" % i) is not None]
        if len(pts) == 4:
            pts = [pts[0], pts[1], pts[3], pts[2]]
        points = _dedupe([_vec(p) for p in pts])
        if len(points) < 3:
            return []
        return self._finish(e, "hatch", {"boundary": [{"kind": "polygon", "points": points}], "pattern": "solid"},
                            fill=True)

    # -- dimensions --------------------------------------------------------------------------
    def _dimvar(self, override, name, default=None):
        try:
            value = override.get(name, default)
        except (self.ezdxf.DXFError, KeyError, ValueError, AttributeError):
            return default
        return default if value is None else value

    def _explode_dimension(self, e, depth, reason):
        try:
            parts = list(e.virtual_entities())
        except (self.ezdxf.DXFError, ValueError, KeyError, AttributeError):
            parts = []
        if not parts:
            return self.refuse("unsupported DXF dimension type (%s)" % reason, "DIMENSION")
        self.notes.add("approximations", "DIMENSION (%s) exploded into lines and text; measurement lost" % reason)
        out = []
        self.solid_ok += 1
        try:
            for part in parts:
                if part.dxftype() == "POINT" and (part.dxf.get("layer", "") or "").lower() == "defpoints":
                    continue
                out.extend(self.convert(part, depth + 1))
        finally:
            self.solid_ok -= 1
        return out

    def _c_dimension(self, e, depth):
        dimtype = int(e.dxf.get("dimtype", 0))
        base = dimtype & 7
        if not self._extrusion_ok(e):
            return self.refuse("DXF OCS extrusion requires explicit conversion", "DIMENSION")
        d = e.dxf
        if base in (0, 1):
            p1, p2, q = d.defpoint2, d.defpoint3, d.defpoint
            direction = atan2(p2[1] - p1[1], p2[0] - p1[0])
            angle = radians(float(d.get("angle", 0.0))) if base == 0 else direction
            if abs(sin(angle - direction)) < 1e-9:
                geometry = {"dim_type": "aligned", "p1": _vec(p1), "p2": _vec(p2), "dimline_point": _vec(q)}
            else:
                geometry = {"dim_type": "linear", "p1": _vec(p1), "p2": _vec(p2), "dimline_point": _vec(q),
                            "angle": angle}
        elif base == 5:
            geometry = {"dim_type": "angular", "center": _vec(d.defpoint5), "p1": _vec(d.defpoint2),
                        "p2": _vec(d.defpoint3), "arc_point": _vec(d.defpoint)}
        elif base == 4:
            geometry = {"dim_type": "radial", "center": _vec(d.defpoint), "point": _vec(d.defpoint4)}
        elif base == 3:
            a, b = d.defpoint, d.defpoint4
            geometry = {"dim_type": "diameter", "center": [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2],
                        "point": _vec(a)}
        else:
            return self._explode_dimension(e, depth, "type %d" % base)

        override = e.override()
        scale = float(self._dimvar(override, "dimscale", 1.0)) or 1.0
        height = float(self._dimvar(override, "dimtxt", 0.0)) * scale
        if height > 0:
            geometry["text_height"] = height
        arrow = float(self._dimvar(override, "dimasz", 0.0)) * scale
        if arrow > 0:
            geometry["arrow_size"] = arrow
        angular = geometry["dim_type"] == "angular"
        digits = self._dimvar(override, "dimadec", -1) if angular else None
        if digits is None or digits < 0:
            digits = self._dimvar(override, "dimdec", None)
        if digits is not None and 0 <= int(digits) <= 8:
            geometry["precision"] = int(digits)
        if not angular:
            post = str(self._dimvar(override, "dimpost", "") or "")
            if "<>" in post:
                prefix, post = post.split("<>", 1)
                if prefix:
                    self.notes.add("approximations", "dimension text prefixes ignored (only suffixes are kept)")
            if post:
                geometry["suffix"] = post[:32]
        text = d.get("text", "") or ""
        if text == " ":
            geometry["text"] = ""
        elif text not in ("", "<>"):
            if "\\" in text or "{" in text:
                from ezdxf.tools.text import plain_mtext
                text = plain_mtext(text, split=False)
                self.notes.add("approximations", "inline formatting removed from dimension text overrides")
            geometry["text"] = text[:255]
        if dimtype & 128 and d.hasattr("text_midpoint"):
            geometry["text_position"] = _vec(d.text_midpoint)
        if float(self._dimvar(override, "dimlfac", 1.0)) != 1.0:
            self.notes.add("approximations", "dimension linear scale factor (DIMLFAC) ignored")
        if int(self._dimvar(override, "dimlunit", 2)) != 2 and not angular:
            self.notes.add("approximations", "non-decimal dimension units ignored")
        if int(self._dimvar(override, "dimtol", 0)) or int(self._dimvar(override, "dimlim", 0)):
            self.notes.add("approximations", "dimension tolerances ignored")
        result = self._finish(e, "dimension", geometry)
        stored = d.get("actual_measurement")
        if result and stored:
            derived = ann.dimension_value(geometry)
            if abs(derived - float(stored)) > 1e-6 * max(1.0, abs(derived)):
                self.notes.add("warnings", "dimensions whose stored measurement differs from their geometry "
                                           "(the geometry was used)")
        return result

    # -- tables ------------------------------------------------------------------------------
    def read_linetypes(self) -> list:
        records = []
        from .dxf_map import RESERVED_LINETYPES
        for lt in self.doc.linetypes:
            name = lt.dxf.name
            if name.lower() in RESERVED_LINETYPES:
                continue
            tags = list(lt.pattern_tags.tags)
            elements = [float(t.value) for t in tags if t.code == 49]
            if any(t.code == 74 and int(t.value) != 0 for t in tags):
                self.notes.add("approximations", "complex linetype elements (text, shapes) removed")
                elements = [float(tags[i].value) for i in range(len(tags))
                            if tags[i].code == 49 and (i + 1 >= len(tags) or tags[i + 1].code != 74
                                                       or int(tags[i + 1].value) == 0)]
            usable = (1 <= len(elements) <= 32 and any(v >= 0 for v in elements) and sum(abs(v) for v in elements) > 0
                      and len(name) <= 255)
            if not usable:
                self.notes.add("dropped", "linetypes without a usable dash pattern")
                continue
            record = {"name": name, "pattern": elements}
            description = lt.dxf.get("description", "") or ""
            if description and description != name and len(description) <= 255:   # the writer repeats the name
                record["description"] = description
            records.append(record)
            self.lt_names[name.lower()] = name
        if abs(float(self.doc.header.get("$LTSCALE", 1.0)) - 1.0) > 1e-9:
            self.notes.add("warnings", "the drawing's global linetype scale ($LTSCALE) is not applied to patterns")
        return records

    def read_layers(self) -> list:
        records = []
        for layer in self.doc.layers:
            name = layer.dxf.name
            if name == "Defpoints":
                continue
            record = {"name": name, "visible": not layer.is_off() and not layer.is_frozen()}
            aci = abs(int(layer.dxf.get("color", 7)))
            true_color = layer.dxf.get("true_color")
            if true_color is not None or aci not in (0, 7):
                self.layer_colored.add(name.lower())
            rgb = self._rgb(true_color, aci if aci not in (0, 7) else None)
            if rgb is not None:
                record["stroke"] = hex_color(rgb)
            weight = layer.dxf.get("lineweight", -3)
            if weight is not None and weight >= 0:
                record["stroke_width"] = stroke_width_for(weight, self.unit)
            low = (layer.dxf.get("linetype", "Continuous") or "Continuous").lower()
            if low in self.lt_names:
                record["linetype"] = self.lt_names[low]
                self.layer_linetype[name.lower()] = record["linetype"]
            records.append(record)
        return records

    # -- driver ------------------------------------------------------------------------------
    def run(self) -> Drawing:
        linetypes = self.read_linetypes()
        layers = self.read_layers()
        model_specs = []
        for entity in self.doc.modelspace():
            model_specs.extend(self.convert(entity))
        for block in list(self.dxf_blocks.values()):
            if block.name in self.block_state or block.name.upper() in _ARROW_BLOCKS:
                continue            # arrowhead blocks of dimension styles are not user content
            try:
                self.convert_block(block.name)
            except (UnsupportedFeatureError, ValidationError):
                self.skipped_blocks.append(block.name)
        if self.skipped_blocks:
            shown = ", ".join(self.skipped_blocks[:5]) + (" ..." if len(self.skipped_blocks) > 5 else "")
            self.notes.add("dropped", "%d unreferenced block(s) with unsupported content not imported: %s"
                           % (len(self.skipped_blocks), shown))
        drawing = Drawing(unit=self.unit, layers=layers, linetypes=linetypes)
        known_layers = {x["name"] for x in drawing.layers}
        for name in self.block_order:
            specs, base = self.block_specs[name]
            for spec in specs:
                if spec["layer"] not in known_layers:
                    drawing.add_layer(spec["layer"])
                    known_layers.add(spec["layer"])
            drawing.add_block(name, specs, base_point=base)
        for spec in model_specs:
            drawing.add(spec["kind"], spec["geometry"], layer=spec["layer"], style=spec.get("style"),
                        create_layer=True)
        return drawing


def import_dxf_with_report(path, *, unit=None, skip_unsupported: bool = False):
    """Read a DXF file; returns ``(Drawing, Notes)``.

    Entities outside the supported subset raise :class:`UnsupportedFeatureError` unless
    ``skip_unsupported`` is true, in which case they are omitted and listed in the notes.
    """
    ezdxf = _ezdxf()
    source = Path(path)
    if not source.is_file() or source.is_symlink() or not 0 < source.stat().st_size <= _MAX_DXF_BYTES:
        raise ResourceLimitError("invalid or oversized DXF file")
    try:
        doc = ezdxf.readfile(str(path))
    except (ezdxf.DXFError, UnicodeDecodeError, OSError, ValueError, IndexError) as exc:
        raise ValidationError("DXF file could not be parsed: " + str(exc)) from exc
    if any(len(layout) > 0 for layout in doc.layouts if layout.name.lower() != "model"):
        # Paper space is not part of SDRW and must not vanish silently.
        raise UnsupportedFeatureError("DXF paper-space layouts are not supported")
    declared = next((name for name, code in _UNIT_CODES.items() if code == doc.units), None)
    if unit is None:
        if declared is None:
            raise UnsupportedFeatureError("DXF units require an explicit supported unit override")
        unit = declared
    importer = _Importer(ezdxf, doc, unit, skip_unsupported)
    return importer.run(), importer.notes


def import_dxf(path, *, unit=None, skip_unsupported: bool = False) -> Drawing:
    """Read a DXF file as a :class:`Drawing` (see :func:`import_dxf_with_report`)."""
    return import_dxf_with_report(path, unit=unit, skip_unsupported=skip_unsupported)[0]
