"""Geometry of the annotation entities: ``text``, ``hatch`` and ``dimension``.

Validation, measurement and derived primitives live here so entities, transforms
and interchange all agree. Everything is planar, in drawing units; angles are
radians, counter-clockwise positive.

Derived data is never stored: a dimension's measured value and display string
are computed from its defining points (so they cannot go stale), and a hatch's
area is computed from its boundary.
"""
from __future__ import annotations

from math import atan2, ceil, cos, hypot, pi, sin, sqrt

from ..common import ValidationError, finite, vector
from ..common.errors import GeometryValidationError

_TAU = 2.0 * pi

# --------------------------------------------------------------------------------------------- text
TEXT_KEYS = frozenset({"position", "text", "height", "rotation", "halign", "valign", "width", "line_spacing", "font"})
HALIGN = ("left", "center", "right")
VALIGN = ("baseline", "bottom", "middle", "top")
MAX_TEXT_LENGTH = 10_000
CHAR_WIDTH = 0.6  # average advance as a fraction of the height; used for extents only


def validate_text(g: dict) -> None:
    vector(g.get("position"), 2)
    text = g.get("text")
    if not isinstance(text, str) or not 1 <= len(text) <= MAX_TEXT_LENGTH:
        raise GeometryValidationError("text requires 1..%d characters" % MAX_TEXT_LENGTH)
    if any((c < " " and c not in "\n\t") or c == "\x7f" for c in text):
        raise GeometryValidationError("text contains control characters")
    if finite(g.get("height"), "text height") <= 0:
        raise GeometryValidationError("text height must be positive")
    if "rotation" in g:
        finite(g["rotation"], "text rotation", limit=1e8)
    if g.get("halign", "left") not in HALIGN:
        raise GeometryValidationError("halign must be one of: " + ", ".join(HALIGN))
    if g.get("valign", "baseline") not in VALIGN:
        raise GeometryValidationError("valign must be one of: " + ", ".join(VALIGN))
    if "width" in g and finite(g["width"], "text width") <= 0:
        raise GeometryValidationError("text width must be positive")
    if "line_spacing" in g and not 0.25 <= finite(g["line_spacing"], "line spacing") <= 4.0:
        raise GeometryValidationError("line_spacing must lie in [0.25, 4]")
    if "font" in g and (not isinstance(g["font"], str) or not 1 <= len(g["font"]) <= 255):
        raise GeometryValidationError("invalid font name")


def text_layout(g: dict) -> dict:
    """Estimated layout ``{lines, width, height}`` in drawing units.

    No font is consulted: the advance is ``0.6 * height`` per character, so the
    box is an estimate that is right to within a font's variation. A ``width``
    wraps long lines (greedy by characters) like MTEXT.
    """
    h = g["height"]
    per_line = None
    if g.get("width"):
        per_line = max(1, int(g["width"] / (CHAR_WIDTH * h)))
    lines = []
    for raw in g["text"].split("\n"):
        raw = raw.replace("\t", "    ")
        if per_line is None or len(raw) <= per_line:
            lines.append(raw)
            continue
        current = ""
        for word in raw.split(" "):
            candidate = word if not current else current + " " + word
            if len(candidate) <= per_line or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    longest = max(len(x) for x in lines)
    width = g["width"] if g.get("width") and per_line is not None and longest > 0 and len(lines) > 1 else longest * CHAR_WIDTH * h
    spacing = g.get("line_spacing", 1.0)
    return {"lines": lines, "width": max(width, CHAR_WIDTH * h * 0.1), "height": h * (1 + (len(lines) - 1) * spacing)}


def text_corners(g: dict) -> list[tuple[float, float]]:
    """Corners of the estimated text box in world coordinates (counter-clockwise)."""
    layout = text_layout(g)
    w, hh, h = layout["width"], layout["height"], g["height"]
    halign, valign = g.get("halign", "left"), g.get("valign", "baseline")
    x0 = {"left": 0.0, "center": -w / 2, "right": -w}[halign]
    y0 = {"baseline": -(hh - h), "bottom": 0.0, "middle": -hh / 2, "top": -hh}[valign]
    theta = g.get("rotation", 0.0)
    c, s = cos(theta), sin(theta)
    px, py = g["position"]
    return [(px + c * x - s * y, py + s * x + c * y) for x, y in
            ((x0, y0), (x0 + w, y0), (x0 + w, y0 + hh), (x0, y0 + hh))]


# ------------------------------------------------------------------------------------------- hatch
HATCH_KEYS = frozenset({"boundary", "pattern", "angle", "spacing"})
HATCH_PATTERNS = ("solid", "lines", "cross")
MAX_HATCH_LOOPS = 256
MAX_LOOP_SEGMENTS = 1_024
MAX_LOOP_POINTS = 100_000


def _arc_ok(seg):
    vector(seg.get("center"), 2)
    if finite(seg.get("radius"), "arc radius") <= 0:
        raise GeometryValidationError("hatch arc radius must be positive")
    a = finite(seg.get("start_angle"), "start_angle", limit=1e8)
    b = finite(seg.get("end_angle"), "end_angle", limit=1e8)
    if not 0 < b - a <= _TAU + 1e-10:
        raise GeometryValidationError("hatch arc sweep must lie in (0, 2π]")


def arc_point(seg, angle):
    cx, cy = seg["center"]
    return (cx + seg["radius"] * cos(angle), cy + seg["radius"] * sin(angle))


def _chain(segments, tolerance=None):
    """Walk an undirected chain of line/arc segments.

    Returns ``[(segment, reversed)]`` in traversal order, or raises when the
    segments do not form exactly one closed loop in the given order.
    """
    def ends(seg):
        if seg["kind"] == "line":
            return tuple(seg["start"]), tuple(seg["end"])
        return arc_point(seg, seg["start_angle"]), arc_point(seg, seg["end_angle"])

    scale = max([1.0] + [abs(v) for seg in segments for p in ends(seg) for v in p])
    tol = tolerance if tolerance is not None else 1e-9 * scale

    def same(p, q):
        return hypot(p[0] - q[0], p[1] - q[1]) <= tol

    first, second = ends(segments[0]), ends(segments[1])
    candidates = ((first[1], False), (first[0], True))
    reversed_first = None
    for joint, rev in candidates:
        if same(joint, second[0]) or same(joint, second[1]):
            reversed_first = rev
            break
    if reversed_first is None:
        raise GeometryValidationError("hatch path segments do not connect")
    order = []
    cursor = ends(segments[0])[0] if reversed_first else ends(segments[0])[1]
    start = ends(segments[0])[1] if reversed_first else ends(segments[0])[0]
    order.append((segments[0], reversed_first))
    for seg in segments[1:]:
        a, b = ends(seg)
        if same(cursor, a):
            order.append((seg, False))
            cursor = b
        elif same(cursor, b):
            order.append((seg, True))
            cursor = a
        else:
            raise GeometryValidationError("hatch path segments do not connect in order")
    if not same(cursor, start):
        raise GeometryValidationError("hatch path is not closed")
    return order


def chain_segments(segments):
    """Public form of the closed-chain walk: ``[(segment, reversed)]`` in traversal order."""
    return _chain(segments)


def validate_hatch_loop(loop) -> None:
    if not isinstance(loop, dict):
        raise GeometryValidationError("hatch loop must be an object")
    kind = loop.get("kind")
    if kind == "polygon":
        if set(loop) - {"kind", "points"}:
            raise GeometryValidationError("unknown fields in hatch polygon loop")
        pts = loop.get("points")
        if not isinstance(pts, list) or not 3 <= len(pts) <= MAX_LOOP_POINTS:
            raise GeometryValidationError("hatch polygon requires 3..%d points" % MAX_LOOP_POINTS)
        points = [vector(p, 2) for p in pts]
        if len(set(points)) < 3:
            raise GeometryValidationError("hatch polygon requires three unique points")
        if any(points[i] == points[(i + 1) % len(points)] for i in range(len(points))):
            raise GeometryValidationError("hatch polygon has a zero-length edge")
        if abs(_shoelace(points)) <= 1e-14 * max(1.0, max(abs(v) for p in points for v in p) ** 2):
            raise GeometryValidationError("hatch polygon has zero area")
    elif kind == "circle":
        if set(loop) - {"kind", "center", "radius"}:
            raise GeometryValidationError("unknown fields in hatch circle loop")
        vector(loop.get("center"), 2)
        if finite(loop.get("radius"), "radius") <= 0:
            raise GeometryValidationError("hatch circle radius must be positive")
    elif kind == "path":
        if set(loop) - {"kind", "segments"}:
            raise GeometryValidationError("unknown fields in hatch path loop")
        segs = loop.get("segments")
        if not isinstance(segs, list) or not 1 <= len(segs) <= MAX_LOOP_SEGMENTS:
            raise GeometryValidationError("hatch path requires 1..%d segments" % MAX_LOOP_SEGMENTS)
        for seg in segs:
            if not isinstance(seg, dict):
                raise GeometryValidationError("hatch segment must be an object")
            if seg.get("kind") == "line":
                if set(seg) - {"kind", "start", "end"}:
                    raise GeometryValidationError("unknown fields in hatch line segment")
                if vector(seg.get("start"), 2) == vector(seg.get("end"), 2):
                    raise GeometryValidationError("zero-length hatch segment")
            elif seg.get("kind") == "arc":
                if set(seg) - {"kind", "center", "radius", "start_angle", "end_angle"}:
                    raise GeometryValidationError("unknown fields in hatch arc segment")
                _arc_ok(seg)
            else:
                raise GeometryValidationError("hatch segment kind must be 'line' or 'arc'")
        if len(segs) == 1:
            seg = segs[0]
            if seg["kind"] != "arc" or abs(seg["end_angle"] - seg["start_angle"] - _TAU) > 1e-9:
                raise GeometryValidationError("a one-segment hatch path must be a full circle")
        else:
            _chain(segs)
    else:
        raise GeometryValidationError("hatch loop kind must be polygon, circle or path")


def validate_hatch(g: dict) -> None:
    loops = g.get("boundary")
    if not isinstance(loops, list) or not 1 <= len(loops) <= MAX_HATCH_LOOPS:
        raise GeometryValidationError("hatch requires 1..%d boundary loops" % MAX_HATCH_LOOPS)
    for loop in loops:
        validate_hatch_loop(loop)
    pattern = g.get("pattern", "solid")
    if pattern not in HATCH_PATTERNS:
        raise GeometryValidationError("hatch pattern must be one of: " + ", ".join(HATCH_PATTERNS))
    if "angle" in g:
        finite(g["angle"], "hatch angle", limit=1e8)
    if pattern == "solid":
        if "spacing" in g:
            raise GeometryValidationError("a solid hatch has no spacing")
    elif "spacing" not in g or finite(g["spacing"], "hatch spacing") <= 0:
        raise GeometryValidationError("patterned hatch requires a positive spacing")
    # Loop containment is the only relation we can check cheaply and exactly enough.
    polygons = [loop_polygon(loop) for loop in loops]
    for i, poly in enumerate(polygons):
        for j, other in enumerate(polygons):
            if i != j and _loops_cross(poly, other):
                raise GeometryValidationError("hatch boundary loops intersect each other")


def _shoelace(points):
    return 0.5 * sum(points[i][0] * points[(i + 1) % len(points)][1] - points[(i + 1) % len(points)][0] * points[i][1]
                     for i in range(len(points)))


def _arc_samples(seg, forward=True, step=pi / 90):
    a, b = seg["start_angle"], seg["end_angle"]
    n = max(2, int(ceil((b - a) / step)))
    angles = [a + (b - a) * i / n for i in range(n + 1)]
    pts = [arc_point(seg, t) for t in angles]
    return pts if forward else pts[::-1]


def loop_polygon(loop, *, step=pi / 90):
    """Closed polygon approximating one hatch loop (no repeated end point).

    Circles and arcs are sampled every ``step`` radians (2 degrees by default),
    which keeps chord error under 0.06% of the radius.
    """
    kind = loop["kind"]
    if kind == "polygon":
        return [tuple(map(float, p)) for p in loop["points"]]
    if kind == "circle":
        n = int(ceil(_TAU / step))
        cx, cy = loop["center"]
        r = loop["radius"]
        return [(cx + r * cos(_TAU * i / n), cy + r * sin(_TAU * i / n)) for i in range(n)]
    segs = loop["segments"]
    if len(segs) == 1:
        return _arc_samples(segs[0], step=step)[:-1]
    points = []
    for seg, rev in _chain(segs):
        if seg["kind"] == "line":
            a, b = tuple(seg["start"]), tuple(seg["end"])
            pts = [b, a] if rev else [a, b]
        else:
            pts = _arc_samples(seg, not rev, step)
        points.extend(pts[:-1])
    return points


def loop_signed_area(loop) -> float:
    """Exact signed area (counter-clockwise positive) via Green's theorem."""
    kind = loop["kind"]
    if kind == "polygon":
        return _shoelace([tuple(p) for p in loop["points"]])
    if kind == "circle":
        return pi * loop["radius"] ** 2
    segs = loop["segments"]
    if len(segs) == 1:
        return pi * segs[0]["radius"] ** 2
    total = 0.0
    for seg, rev in _chain(segs):
        if seg["kind"] == "line":
            a, b = seg["start"], seg["end"]
            if rev:
                a, b = b, a
            total += 0.5 * (a[0] * b[1] - b[0] * a[1])
        else:
            cx, cy = seg["center"]
            r, a, b = seg["radius"], seg["start_angle"], seg["end_angle"]
            part = 0.5 * (r * r * (b - a) + cx * r * (sin(b) - sin(a)) - cy * r * (cos(b) - cos(a)))
            total += -part if rev else part
    return total


def point_in_polygon(point, polygon) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def _loops_cross(p, q) -> bool:
    # Cheap rejection: disjoint boxes cannot cross.
    if (max(x for x, _ in p) < min(x for x, _ in q) or max(x for x, _ in q) < min(x for x, _ in p)
            or max(y for _, y in p) < min(y for _, y in q) or max(y for _, y in q) < min(y for _, y in p)):
        return False
    if len(p) * len(q) > 4_000_000:
        return False  # too large to audit exactly; strict reports note this separately
    for i in range(len(p)):
        a, b = p[i], p[(i + 1) % len(p)]
        for j in range(len(q)):
            c, d = q[j], q[(j + 1) % len(q)]
            if _proper_cross(a, b, c, d):
                return True
    return False


def _proper_cross(a, b, c, d) -> bool:
    def cross(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    v1, v2, v3, v4 = cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b)
    return (v1 > 0 > v2 or v1 < 0 < v2) and (v3 > 0 > v4 or v3 < 0 < v4)


def hatch_area(g: dict) -> float:
    """Filled area under the even-odd rule: nested loops alternate fill and hole."""
    loops = g["boundary"]
    polygons = [loop_polygon(loop) for loop in loops]
    total = 0.0
    for i, loop in enumerate(loops):
        probe = polygons[i][0]
        depth = sum(1 for j, other in enumerate(polygons) if j != i and point_in_polygon(probe, other))
        total += (-1) ** depth * abs(loop_signed_area(loop))
    return total


def arc_extent(center, radius, a, b):
    """Tight bounds of the arc from angle ``a`` to ``b`` (counter-clockwise)."""
    cx, cy = center
    if b - a >= _TAU - 1e-12:
        return (cx - radius, cy - radius), (cx + radius, cy + radius)
    angles = [a, b]
    for cardinal in (0.0, pi / 2, pi, 3 * pi / 2):
        offset = (cardinal - a) % _TAU
        if offset <= b - a + 1e-12:
            angles.append(a + offset)
    pts = [(cx + radius * cos(t), cy + radius * sin(t)) for t in angles]
    return (min(p[0] for p in pts), min(p[1] for p in pts)), (max(p[0] for p in pts), max(p[1] for p in pts))


def hatch_bounds(g: dict):
    boxes = []
    for loop in g["boundary"]:
        kind = loop["kind"]
        if kind == "polygon":
            pts = loop["points"]
            boxes.append(((min(p[0] for p in pts), min(p[1] for p in pts)), (max(p[0] for p in pts), max(p[1] for p in pts))))
        elif kind == "circle":
            boxes.append(arc_extent(loop["center"], loop["radius"], 0.0, _TAU))
        else:
            for seg in loop["segments"]:
                if seg["kind"] == "line":
                    a, b = seg["start"], seg["end"]
                    boxes.append(((min(a[0], b[0]), min(a[1], b[1])), (max(a[0], b[0]), max(a[1], b[1]))))
                else:
                    boxes.append(arc_extent(seg["center"], seg["radius"], seg["start_angle"], seg["end_angle"]))
    return ((min(b[0][0] for b in boxes), min(b[0][1] for b in boxes)),
            (max(b[1][0] for b in boxes), max(b[1][1] for b in boxes)))


def hatch_lines(g: dict, *, max_lines: int = 20_000):
    """Pattern strokes clipped to the (even-odd) boundary as ``[(p, q), ...]``.

    Used by SVG export of patterned hatches. ``solid`` hatches have no strokes.
    ``cross`` adds the perpendicular family. Raises ``ResourceLimitError`` when
    the spacing would create more than ``max_lines`` strokes.
    """
    from ..common.errors import ResourceLimitError
    if g.get("pattern", "solid") == "solid":
        return []
    polygons = [loop_polygon(loop) for loop in g["boundary"]]
    (x0, y0), (x1, y1) = hatch_bounds(g)
    spacing = g["spacing"]
    base = g.get("angle", 0.0)
    angles = [base] if g["pattern"] == "lines" else [base, base + pi / 2]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    radius = hypot(x1 - x0, y1 - y0) / 2 + spacing
    lines = []
    for theta in angles:
        dx, dy = cos(theta), sin(theta)
        nx, ny = -dy, dx
        count = int(2 * radius / spacing) + 1
        if len(lines) + count > max_lines:
            raise ResourceLimitError("hatch spacing is too small for the boundary size")
        for k in range(count):
            offset = -radius + (k + 1e-7) * spacing  # nudge off exact vertex alignment (keeps the parity rule sound)
            ox, oy = cx + nx * offset, cy + ny * offset
            hits = []
            for poly in polygons:
                for i in range(len(poly)):
                    (ax, ay), (bx, by) = poly[i], poly[(i + 1) % len(poly)]
                    # Solve o + t*d = a + s*(b-a)
                    ex, ey = bx - ax, by - ay
                    det = dx * ey - dy * ex
                    if abs(det) < 1e-14:
                        continue
                    s = ((ax - ox) * dy - (ay - oy) * dx) / det
                    t = ((ax - ox) * ey - (ay - oy) * ex) / det
                    if 0 <= s < 1:
                        hits.append(t)
            hits.sort()
            for i in range(0, len(hits) - 1, 2):
                if hits[i + 1] - hits[i] > 1e-12:
                    lines.append(((ox + dx * hits[i], oy + dy * hits[i]), (ox + dx * hits[i + 1], oy + dy * hits[i + 1])))
    return lines


# ------------------------------------------------------------------------------------- dimension
DIM_TYPES = ("linear", "aligned", "angular", "radial", "diameter")
_DIM_COMMON = frozenset({"dim_type", "text", "text_height", "text_position", "precision", "suffix", "arrow_size"})
DIM_KEYS = {
    "linear": _DIM_COMMON | {"p1", "p2", "dimline_point", "angle"},
    "aligned": _DIM_COMMON | {"p1", "p2", "dimline_point"},
    "angular": _DIM_COMMON | {"center", "p1", "p2", "arc_point"},
    "radial": _DIM_COMMON | {"center", "point"},
    "diameter": _DIM_COMMON | {"center", "point"},
}
MIN_MEASUREMENT = 1e-12


def validate_dimension(g: dict) -> None:
    kind = g.get("dim_type")
    if kind not in DIM_TYPES:
        raise GeometryValidationError("dim_type must be one of: " + ", ".join(DIM_TYPES))
    unknown = set(g) - DIM_KEYS[kind]
    if unknown:
        raise GeometryValidationError("unknown fields in %s dimension: %s" % (kind, ", ".join(sorted(map(str, unknown)))))
    if "text" in g and (not isinstance(g["text"], str) or len(g["text"]) > 255):
        raise GeometryValidationError("dimension text override must be a string of at most 255 characters")
    if "text_height" in g and finite(g["text_height"], "text_height") <= 0:
        raise GeometryValidationError("text_height must be positive")
    if "arrow_size" in g and finite(g["arrow_size"], "arrow_size") <= 0:
        raise GeometryValidationError("arrow_size must be positive")
    if "text_position" in g:
        vector(g["text_position"], 2)
    if "precision" in g and (type(g["precision"]) is not int or not 0 <= g["precision"] <= 8):
        raise GeometryValidationError("precision must be an integer in [0, 8]")
    if "suffix" in g and (not isinstance(g["suffix"], str) or len(g["suffix"]) > 32):
        raise GeometryValidationError("suffix must be a string of at most 32 characters")
    if kind in ("linear", "aligned"):
        p1, p2, q = vector(g.get("p1"), 2), vector(g.get("p2"), 2), vector(g.get("dimline_point"), 2)
        if kind == "linear":
            finite(g.get("angle", 0.0), "angle", limit=1e8)
        if dimension_value(g) <= MIN_MEASUREMENT:
            raise GeometryValidationError("dimension measures a zero distance")
    elif kind == "angular":
        c, p1, p2, q = (vector(g.get(k), 2) for k in ("center", "p1", "p2", "arc_point"))
        if c == p1 or c == p2 or c == q:
            raise GeometryValidationError("angular dimension points must differ from the centre")
        if dimension_value(g) <= 1e-12:
            raise GeometryValidationError("angular dimension rays coincide")
    else:
        c, p = vector(g.get("center"), 2), vector(g.get("point"), 2)
        if c == p:
            raise GeometryValidationError("radial/diameter dimension point equals its centre")


def dimension_value(g: dict) -> float:
    """Measured value: a length in drawing units, or radians for ``angular``."""
    kind = g["dim_type"]
    if kind == "linear":
        th = g.get("angle", 0.0)
        return abs((g["p2"][0] - g["p1"][0]) * cos(th) + (g["p2"][1] - g["p1"][1]) * sin(th))
    if kind == "aligned":
        return hypot(g["p2"][0] - g["p1"][0], g["p2"][1] - g["p1"][1])
    if kind == "angular":
        c = g["center"]
        a1 = atan2(g["p1"][1] - c[1], g["p1"][0] - c[0])
        a2 = atan2(g["p2"][1] - c[1], g["p2"][0] - c[0])
        return (a2 - a1) % _TAU
    r = hypot(g["point"][0] - g["center"][0], g["point"][1] - g["center"][1])
    return 2 * r if kind == "diameter" else r


def dimension_text(g: dict) -> str:
    """Display string. ``text`` overrides it; ``<>`` inside the override is the measured value."""
    kind = g["dim_type"]
    value = dimension_value(g)
    precision = g.get("precision", 2)
    if kind == "angular":
        shown = ("%.*f" % (precision, value * 180.0 / pi)) + "\N{DEGREE SIGN}"
    else:
        shown = {"radial": "R", "diameter": "\N{DIAMETER SIGN}"}.get(kind, "") + ("%.*f" % (precision, value)) + g.get("suffix", "")
    if "text" in g:
        return g["text"].replace("<>", shown) if g["text"] else ""
    return shown


def _dim_scale(g):
    kind = g["dim_type"]
    if kind == "angular":
        return hypot(g["arc_point"][0] - g["center"][0], g["arc_point"][1] - g["center"][1])
    return max(dimension_value(g), 1e-9)


def dimension_text_height(g: dict) -> float:
    return g.get("text_height") or 0.04 * _dim_scale(g)


def _arrow(tip, direction, size):
    """Open arrowhead: two strokes of length ``size`` splayed 15 degrees either side."""
    ang = atan2(direction[1], direction[0])
    out = []
    for side in (-1, 1):
        t = ang + pi + side * (pi / 12)
        out.append({"kind": "line", "start": [tip[0], tip[1]], "end": [tip[0] + size * cos(t), tip[1] + size * sin(t)]})
    return out


def dimension_primitives(g: dict) -> list[dict]:
    """The drawing of a dimension as plain primitives (``line``/``arc``/``text``).

    Linear/aligned: two extension lines, the dimension line and arrowheads.
    Angular: two extension lines, an arc and arrowheads. Radial: leader to the
    circle. Diameter: leader across the circle. The text is centred on the
    dimension line (or at ``text_position``) and kept upright.
    """
    kind = g["dim_type"]
    h = dimension_text_height(g)
    arrow = g.get("arrow_size", 0.6 * h)
    prims = []
    text_pos = None
    rotation = 0.0
    if kind in ("linear", "aligned"):
        p1, p2, q = g["p1"], g["p2"], g["dimline_point"]
        if kind == "linear":
            th = g.get("angle", 0.0)
        else:
            th = atan2(p2[1] - p1[1], p2[0] - p1[0])
        d, n = (cos(th), sin(th)), (-sin(th), cos(th))
        a = (p1[0] + n[0] * ((q[0] - p1[0]) * n[0] + (q[1] - p1[1]) * n[1]),
             p1[1] + n[1] * ((q[0] - p1[0]) * n[0] + (q[1] - p1[1]) * n[1]))
        b = (p2[0] + n[0] * ((q[0] - p2[0]) * n[0] + (q[1] - p2[1]) * n[1]),
             p2[1] + n[1] * ((q[0] - p2[0]) * n[0] + (q[1] - p2[1]) * n[1]))
        for src, dst in ((p1, a), (p2, b)):
            if hypot(dst[0] - src[0], dst[1] - src[1]) > 1e-12:
                prims.append({"kind": "line", "start": list(src), "end": list(dst)})
        if hypot(b[0] - a[0], b[1] - a[1]) > 1e-12:
            prims.append({"kind": "line", "start": list(a), "end": list(b)})
            toward = (b[0] - a[0], b[1] - a[1])
            prims += _arrow(a, toward, arrow) + _arrow(b, (-toward[0], -toward[1]), arrow)
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        text_pos = (mid[0] + n[0] * 0.6 * h, mid[1] + n[1] * 0.6 * h)
        rotation = th
    elif kind == "angular":
        c = g["center"]
        r = hypot(g["arc_point"][0] - c[0], g["arc_point"][1] - c[1])
        a1 = atan2(g["p1"][1] - c[1], g["p1"][0] - c[0])
        sweep = dimension_value(g)
        a2 = a1 + sweep
        for ang, src in ((a1, g["p1"]), (a2, g["p2"])):
            tip = (c[0] + r * cos(ang), c[1] + r * sin(ang))
            if hypot(tip[0] - src[0], tip[1] - src[1]) > 1e-12:
                prims.append({"kind": "line", "start": list(src), "end": list(tip)})
        prims.append({"kind": "arc", "center": list(c), "radius": r, "start_angle": a1, "end_angle": a2})
        for ang, sign in ((a1, 1), (a2, -1)):
            tip = (c[0] + r * cos(ang), c[1] + r * sin(ang))
            tangent = (-sin(ang) * sign, cos(ang) * sign)
            prims += _arrow(tip, tangent, arrow)
        mid = a1 + sweep / 2
        text_pos = (c[0] + (r + 0.8 * h) * cos(mid), c[1] + (r + 0.8 * h) * sin(mid))
        rotation = mid - pi / 2
    else:
        c, p = g["center"], g["point"]
        direction = (p[0] - c[0], p[1] - c[1])
        length = hypot(*direction)
        unit = (direction[0] / length, direction[1] / length)
        rotation = atan2(direction[1], direction[0])
        if kind == "radial":
            prims.append({"kind": "line", "start": list(c), "end": list(p)})
            prims += _arrow(p, (-unit[0], -unit[1]), arrow)
            text_pos = (p[0] + unit[0] * 1.2 * h, p[1] + unit[1] * 1.2 * h)
        else:
            far = (2 * c[0] - p[0], 2 * c[1] - p[1])
            prims.append({"kind": "line", "start": list(far), "end": list(p)})
            prims += _arrow(p, (-unit[0], -unit[1]), arrow) + _arrow(far, unit, arrow)
            text_pos = (c[0] - unit[1] * 0.6 * h, c[1] + unit[0] * 0.6 * h)
    if "text_position" in g:
        text_pos = tuple(g["text_position"])
    while rotation > pi / 2 + 1e-9:
        rotation -= pi
    while rotation <= -pi / 2 + 1e-9:
        rotation += pi
    label = dimension_text(g)
    if label:
        prims.append({"kind": "text", "position": list(text_pos), "text": label, "height": h, "rotation": rotation,
                      "halign": "center", "valign": "bottom"})
    return prims


def primitive_bounds(prim):
    kind = prim["kind"]
    if kind == "line":
        a, b = prim["start"], prim["end"]
        return ((min(a[0], b[0]), min(a[1], b[1])), (max(a[0], b[0]), max(a[1], b[1])))
    if kind == "arc":
        return arc_extent(prim["center"], prim["radius"], prim["start_angle"], prim["end_angle"])
    corners = text_corners(prim)
    return ((min(p[0] for p in corners), min(p[1] for p in corners)), (max(p[0] for p in corners), max(p[1] for p in corners)))


def dimension_bounds(g: dict):
    boxes = [primitive_bounds(p) for p in dimension_primitives(g)]
    extra = [g[k] for k in ("p1", "p2", "center", "point", "dimline_point", "arc_point") if k in g]
    xs = [b[0][0] for b in boxes] + [b[1][0] for b in boxes] + [p[0] for p in extra]
    ys = [b[0][1] for b in boxes] + [b[1][1] for b in boxes] + [p[1] for p in extra]
    return ((min(xs), min(ys)), (max(xs), max(ys)))
