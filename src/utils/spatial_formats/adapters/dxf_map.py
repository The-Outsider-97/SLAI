"""Backend-free mapping rules between SDRW and DXF.

Everything that decides *what a DXF file will say* about an SDRW drawing lives
here and needs no ``ezdxf``: safe names, colours, lineweights, fonts and the
loss notes. :func:`plan_export` is used twice from the same code path -- by the
DXF writer to build the file, and by the conversion layer to enforce the
caller's policy *before* a byte is written -- so the report cannot disagree
with the file.

Conventions (documented in the README):

* ``stroke`` / ``fill`` are ``#rgb``, ``#rrggbb`` or a basic CSS colour name and
  become true colours; ``stroke_width`` is in drawing units and becomes the
  nearest DXF lineweight (hundredths of a millimetre).
* Linetype patterns are DXF style (dash > 0, gap < 0, dot = 0) in drawing units.
* Text height, dimension sizes and hatch spacing are in drawing units.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..sdrw.annotations import text_layout

UNIT_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "km": 1_000_000.0, "in": 25.4, "ft": 304.8}
VALID_LINEWEIGHTS = (0, 5, 9, 13, 15, 18, 20, 25, 30, 35, 40, 50, 53, 60, 70, 80, 90, 100, 106, 120, 140, 158,
                     200, 211)
MTEXT_PITCH = 5.0 / 3.0          # CAD single line spacing, as a multiple of the character height
MTEXT_MIN_FACTOR = 0.25          # DXF limits for the MTEXT line-spacing factor
MTEXT_MAX_FACTOR = 4.0
RESERVED_LINETYPES = ("continuous", "bylayer", "byblock")

_NAMED_COLORS = {
    "black": (0, 0, 0), "white": (255, 255, 255), "red": (255, 0, 0), "lime": (0, 255, 0),
    "green": (0, 128, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0), "cyan": (0, 255, 255),
    "aqua": (0, 255, 255), "magenta": (255, 0, 255), "fuchsia": (255, 0, 255), "gray": (128, 128, 128),
    "grey": (128, 128, 128), "silver": (192, 192, 192), "maroon": (128, 0, 0), "olive": (128, 128, 0),
    "navy": (0, 0, 128), "teal": (0, 128, 128), "purple": (128, 0, 128), "orange": (255, 165, 0),
}
_HEX = re.compile(r"#([0-9a-f]{3}|[0-9a-f]{6})")
_BAD_NAME_CHARS = re.compile(r'[<>/\\":;?*|=`\x00-\x1f\x7f]')


# ------------------------------------------------------------------------------------------ notes
class Notes:
    """Ordered loss notes; a message that repeats is reported once with its count."""

    CATEGORIES = ("dropped", "approximations", "assumptions", "warnings")

    def __init__(self):
        self._items = {name: {} for name in self.CATEGORIES}

    def add(self, category: str, message: str, count: int = 1) -> None:
        bucket = self._items[category]
        bucket[message] = bucket.get(message, 0) + count

    def lines(self, category: str) -> list[str]:
        return [m if n == 1 else "%d x %s" % (n, m) for m, n in self._items[category].items()]

    dropped = property(lambda self: self.lines("dropped"))
    approximations = property(lambda self: self.lines("approximations"))
    assumptions = property(lambda self: self.lines("assumptions"))
    warnings = property(lambda self: self.lines("warnings"))

    def __bool__(self) -> bool:
        return any(self._items[name] for name in self.CATEGORIES)


# ---------------------------------------------------------------------------------------- colours
def parse_color(value) -> tuple[int, int, int] | None:
    """``(r, g, b)`` for ``#rgb``, ``#rrggbb`` or a basic CSS colour name; ``None`` otherwise."""
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if text in _NAMED_COLORS:
        return _NAMED_COLORS[text]
    match = _HEX.fullmatch(text)
    if not match:
        return None
    digits = match.group(1)
    if len(digits) == 3:
        digits = "".join(c * 2 for c in digits)
    return tuple(int(digits[i:i + 2], 16) for i in (0, 2, 4))


def rgb_int(rgb) -> int:
    return (int(rgb[0]) << 16) | (int(rgb[1]) << 8) | int(rgb[2])


def hex_color(rgb) -> str:
    return "#%02x%02x%02x" % (int(rgb[0]), int(rgb[1]), int(rgb[2]))


# ------------------------------------------------------------------------------------ lineweights
def lineweight_for(width: float, unit: str) -> tuple[int, bool]:
    """Nearest DXF lineweight (1/100 mm) for a stroke width in drawing units, and whether it is exact."""
    hundredths = max(0.0, float(width)) * UNIT_MM[unit] * 100.0
    best = min(VALID_LINEWEIGHTS, key=lambda v: abs(v - hundredths))
    return best, abs(best - hundredths) <= 0.5


def stroke_width_for(code: int, unit: str) -> float:
    """Stroke width in drawing units for a DXF lineweight (1/100 mm)."""
    return round(code / 100.0 / UNIT_MM[unit], 12)


# ---------------------------------------------------------------------------------------- names
def safe_name(name: str, taken: set, *, reserved=()) -> str:
    """A DXF-legal table name for ``name``, unique case-insensitively within ``taken`` (which is updated)."""
    base = _BAD_NAME_CHARS.sub("_", name)[:250] or "_"
    candidate, n = base, 1
    while candidate.lower() in taken or candidate.lower() in reserved:
        n += 1
        candidate = "%s_%d" % (base, n)
    taken.add(candidate.lower())
    return candidate


def font_file(font: str) -> str:
    """DXF style font file for an SDRW font name (``Arial`` -> ``Arial.ttf``)."""
    return font if re.search(r"\.[A-Za-z0-9]{2,4}$", font) else font + ".ttf"


# ----------------------------------------------------------------------------------------- plan
@dataclass
class ExportPlan:
    layers: dict = field(default_factory=dict)          # SDRW layer name -> DXF layer name
    linetypes: dict = field(default_factory=dict)       # SDRW linetype name -> DXF name
    blocks: dict = field(default_factory=dict)          # SDRW block name -> DXF block name
    fonts: dict = field(default_factory=dict)           # SDRW font -> (style name, font file)
    layer_attribs: dict = field(default_factory=dict)   # SDRW layer name -> DXF layer attributes
    attribs: dict = field(default_factory=dict)         # entity id -> DXF entity attributes (without layer)
    notes: Notes = field(default_factory=Notes)
    has_dimensions: bool = False


def _all_entities(drawing):
    yield from drawing.entities
    for block in drawing.blocks:
        yield from block["entities"]


def _style_attribs(style: dict, *, fill_first: bool, unit: str, plan: ExportPlan) -> dict:
    """DXF attributes for an SDRW style dict (an entity's or a layer's)."""
    attribs: dict = {}
    notes = plan.notes
    # a solid hatch is coloured by its fill, everything else by its stroke
    colour_key = next((k for k in (("fill", "stroke") if fill_first else ("stroke",)) if k in style), None)
    if colour_key is not None:
        rgb = parse_color(style[colour_key])
        if rgb is None:
            notes.add("dropped", "unrecognised colour %r (use #rgb, #rrggbb or a basic CSS colour name)"
                      % style[colour_key])
        else:
            attribs["true_color"] = rgb_int(rgb)
    if "stroke_width" in style:
        code, exact = lineweight_for(style["stroke_width"], unit)
        attribs["lineweight"] = code
        if not exact:
            notes.add("approximations", "stroke widths snapped to the nearest DXF lineweight")
    if "linetype" in style:
        name = style["linetype"]
        low = name.lower()
        if low == "continuous":
            attribs["linetype"] = "Continuous"
        elif low == "bylayer":
            pass
        else:
            attribs["linetype"] = plan.linetypes[name]
    if "linetype_scale" in style:
        attribs["ltscale"] = style["linetype_scale"]
    return attribs


def plan_export(drawing) -> ExportPlan:
    """Decide names, attributes and loss notes for writing ``drawing`` as DXF."""
    plan = ExportPlan()
    notes, unit = plan.notes, drawing.unit

    taken_layers: set = {"0"}
    for layer in drawing.layers:
        name = layer["name"]
        if name == "0":
            plan.layers[name] = "0"
            continue
        plan.layers[name] = safe_name(name, taken_layers)
    taken_lt: set = set()
    for record in drawing.linetypes:
        plan.linetypes[record["name"]] = safe_name(record["name"], taken_lt, reserved=RESERVED_LINETYPES)
    taken_blocks: set = set()
    for block in drawing.blocks:
        plan.blocks[block["name"]] = safe_name(block["name"], taken_blocks)
    for table, label in ((plan.layers, "layer"), (plan.linetypes, "linetype"), (plan.blocks, "block")):
        for old, new in table.items():
            if old != new:
                notes.add("approximations", "%s name %r written as %r (not valid in DXF)" % (label, old, new))

    for layer in drawing.layers:
        style = {k: layer[k] for k in ("stroke", "stroke_width", "linetype") if k in layer}
        if style:
            plan.layer_attribs[layer["name"]] = _style_attribs(style, fill_first=False, unit=unit, plan=plan)

    taken_styles: set = set()
    multiline = dimension_seen = False
    for entity in _all_entities(drawing):
        kind, g = entity["kind"], entity["geometry"]
        style = entity.get("style") or {}
        solid_hatch = kind == "hatch" and g.get("pattern", "solid") == "solid"
        attribs = _style_attribs(style, fill_first=solid_hatch, unit=unit, plan=plan)
        if "fill" in style and not solid_hatch:
            notes.add("dropped", "fill colour on entities other than solid hatches")
        if attribs:
            plan.attribs[entity["id"]] = attribs
        if kind == "text":
            if "font" in g and g["font"] not in plan.fonts:
                plan.fonts[g["font"]] = (safe_name(g["font"].rsplit(".", 1)[0] if "." in g["font"] else g["font"],
                                                   taken_styles, reserved=("standard",)), font_file(g["font"]))
            lines = text_layout(g)["lines"]
            if len(lines) > 1 or g.get("width"):
                multiline = True
                spacing = g.get("line_spacing", 1.0)
                if len(lines) > 1 and spacing / MTEXT_PITCH < MTEXT_MIN_FACTOR:
                    notes.add("approximations", "line spacing below %.3f raised to the DXF minimum"
                              % (MTEXT_MIN_FACTOR * MTEXT_PITCH))
        elif kind == "dimension":
            dimension_seen = True
    plan.has_dimensions = dimension_seen
    if multiline:
        notes.add("warnings", "multi-line text spacing uses the CAD single-line pitch of 5/3 x text height; "
                              "wrapping and extents follow the DXF reader's font")
    if dimension_seen:
        notes.add("warnings", "DXF dimensions are drawn by the reader's dimension style, so arrowheads and "
                              "text placement may differ slightly from SDRW's own drawing")
    return plan


def mtext_factor(spacing: float) -> float:
    """MTEXT line-spacing factor reproducing SDRW's ``line_spacing`` (clamped to the DXF range)."""
    return min(MTEXT_MAX_FACTOR, max(MTEXT_MIN_FACTOR, spacing / MTEXT_PITCH))


def spacing_from_factor(factor: float) -> float:
    """Inverse of :func:`mtext_factor`, rounded so exported values come back exactly."""
    return round(min(4.0, max(0.25, factor * MTEXT_PITCH)), 9)
