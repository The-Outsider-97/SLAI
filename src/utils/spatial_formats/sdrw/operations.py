"""Analytic, unit-preserving 2D measurements and planar similarity transforms.

Coordinates are model-space positions in the document's declared length unit.
Angles are radians; counter-clockwise rotation is positive. No raster units or
implicit page/view transforms are involved.
"""
from __future__ import annotations

from copy import deepcopy
from math import atan2, cos, hypot, pi, sin, sqrt

from ..common import ValidationError, finite, new_id, vector
from . import annotations as _ann
from .blocks import MAX_BLOCK_DEPTH, insert_matrix
from .entities import validate_entity

_TAU = 2.0 * pi


def entity_bounds(entity: dict, blocks=None) -> tuple[tuple[float, float], tuple[float, float]]:
    """Tight axis-aligned bounds, including cardinal extrema of partial arcs.

    ``text`` and ``dimension`` text use an *estimated* box (no font metrics are
    available). ``insert`` bounds need ``blocks``: a ``{name: block record}`` mapping.
    """
    validate_entity(entity)
    kind, geometry = entity["kind"], entity["geometry"]
    if kind == "text":
        corners = _ann.text_corners(geometry)
        return ((min(p[0] for p in corners), min(p[1] for p in corners)),
                (max(p[0] for p in corners), max(p[1] for p in corners)))
    if kind == "hatch":
        return _ann.hatch_bounds(geometry)
    if kind == "dimension":
        return _ann.dimension_bounds(geometry)
    if kind == "insert":
        if blocks is None or geometry["block"] not in blocks:
            raise ValidationError("insert bounds require the block definitions")
        block = blocks[geometry["block"]]
        matrix = insert_matrix(geometry, block["base_point"])
        boxes = [entity_bounds(transform_entity(e, matrix, preserve_id=True), blocks) for e in block["entities"]]
        return ((min(b[0][0] for b in boxes), min(b[0][1] for b in boxes)),
                (max(b[1][0] for b in boxes), max(b[1][1] for b in boxes)))
    if kind == "point":
        points = [geometry["position"]]
    elif kind == "line":
        points = [geometry["start"], geometry["end"]]
    elif kind == "polyline":
        points = geometry["points"]
    else:
        cx, cy = geometry["center"]
        radius = geometry["radius"]
        if kind == "circle":
            return (cx - radius, cy - radius), (cx + radius, cy + radius)
        start, end = geometry["start_angle"], geometry["end_angle"]
        if end - start >= _TAU - 1e-12:
            return (cx - radius, cy - radius), (cx + radius, cy + radius)
        angles = [start, end]
        for cardinal in (0.0, pi/2, pi, 3*pi/2):
            offset = (cardinal - start) % _TAU
            if offset <= end - start + 1e-12:
                angles.append(start + offset)
        points = [(cx + radius*cos(angle), cy + radius*sin(angle)) for angle in angles]
    return (tuple(min(p[i] for p in points) for i in range(2)),
            tuple(max(p[i] for p in points) for i in range(2)))


def entity_length(entity: dict) -> float:
    """Distance, circumference or perimeter in declared document length units."""
    validate_entity(entity)
    kind, g = entity["kind"], entity["geometry"]
    if kind in ("text", "dimension", "hatch", "insert"):
        raise ValidationError("length is not defined for %s entities" % kind)
    if kind == "point":
        return 0.0
    if kind == "line":
        a, b = g["start"], g["end"]
        return finite(hypot(b[0]-a[0], b[1]-a[1]), "line length", limit=1e18)
    if kind == "circle":
        return finite(_TAU*g["radius"], "circumference", limit=1e18)
    if kind == "arc":
        return finite((g["end_angle"]-g["start_angle"])*g["radius"], "arc length", limit=1e18)
    points = g["points"]
    pairs = list(zip(points, points[1:]))
    if g.get("closed", False):
        pairs.append((points[-1], points[0]))
    return finite(sum(hypot(b[0]-a[0], b[1]-a[1]) for a, b in pairs),
                  "polyline length", limit=1e18)


def enclosed_area(entity: dict) -> float:
    """Positive filled area of a closed polyline or circle, in squared document units.

    Open entities, arcs and lines are not regions. Reject instead of guessing.
    A self-intersecting polygon has ambiguous filled area and is not supported.
    """
    validate_entity(entity)
    kind, g = entity["kind"], entity["geometry"]
    if kind == "hatch":
        return finite(_ann.hatch_area(g), "hatch area", limit=1e18)
    if kind == "circle":
        return finite(pi*g["radius"]**2, "circle area", limit=1e18)
    if kind != "polyline" or not g.get("closed"):
        raise ValidationError("enclosed area requires a closed polyline or circle")
    points = [tuple(p) for p in g["points"]]
    if points[0] == points[-1]:
        points.pop()
    if len(points) < 3:
        raise ValidationError("polygon requires three distinct corners")
    for i in range(len(points)):
        a,b = points[i],points[(i+1)%len(points)]
        for j in range(i+1,len(points)):
            if j == i+1 or (i==0 and j==len(points)-1):
                continue
            c,d=points[j],points[(j+1)%len(points)]
            if _segments_intersect(a,b,c,d):
                raise ValidationError("self-intersecting polygon has ambiguous area")
    twice = sum(points[i][0]*points[(i+1)%len(points)][1] -
                points[(i+1)%len(points)][0]*points[i][1] for i in range(len(points)))
    area = abs(twice)*0.5
    if area <= 1e-14:
        raise ValidationError("zero-area polyline")
    return finite(area, "polygon area", limit=1e18)


def _segments_intersect(a,b,c,d) -> bool:
    def cross(p,q,r):
        return (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
    def on(p,q,r):
        return min(p[0],q[0])-1e-12 <= r[0] <= max(p[0],q[0])+1e-12 and min(p[1],q[1])-1e-12 <= r[1] <= max(p[1],q[1])+1e-12
    v1,v2,v3,v4=cross(a,b,c),cross(a,b,d),cross(c,d,a),cross(c,d,b)
    if (v1>0>v2 or v1<0<v2) and (v3>0>v4 or v3<0<v4):
        return True
    return any(abs(v)<=1e-12 and on(p,q,r) for v,p,q,r in
               ((v1,a,b,c),(v2,a,b,d),(v3,c,d,a),(v4,c,d,b)))


def expand_insert(entity: dict, blocks, *, recursive: bool = True, _depth: int = 0) -> list:
    """The entities an ``insert`` stands for: transformed copies (fresh ids) of its block content.

    ``blocks`` is a ``{name: block record}`` mapping. Content on layer ``"0"`` takes the insert's
    layer; with ``recursive`` nested inserts are expanded too.
    """
    if _depth > MAX_BLOCK_DEPTH + 1:
        raise ValidationError("blocks nest too deeply to expand")
    g = entity["geometry"]
    block = blocks[g["block"]]
    matrix = insert_matrix(g, block["base_point"])
    layer = entity.get("layer", "0")
    out = []
    for inner in block["entities"]:
        moved = transform_entity(inner, matrix)
        if moved.get("layer", "0") == "0":
            moved["layer"] = layer
        if moved["kind"] == "insert" and recursive:
            out.extend(expand_insert(moved, blocks, recursive=True, _depth=_depth + 1))
        else:
            out.append(moved)
    return out


def transform_entity(entity: dict, matrix, *, preserve_id=False) -> dict:
    """Transform SDRW geometry by a proper (non-reflective) 2D similarity.

    matrix: row-major [a,b,tx,c,d,ty,0,0,1], acting on column [x,y,1].
    Shear, nonuniform scale, singular and reflective transforms are rejected to
    preserve circles and arcs analytically rather than silently approximating.
    """
    validate_entity(entity)
    m=vector(matrix,9,"2D homogeneous matrix")
    if any(abs(m[i]-target)>1e-10 for i,target in ((6,0),(7,0),(8,1))):
        raise ValidationError("2D transform must be affine")
    a,b,tx,c,d,ty,_,_,_=m
    scale2=a*a+c*c
    if scale2<=1e-24 or abs(scale2-(b*b+d*d))>1e-9*max(1,scale2) or abs(a*b+c*d)>1e-9*max(1,scale2):
        raise ValidationError("only uniform scaling and rotation are supported")
    if a*d-b*c<=0:
        raise ValidationError("reflections and singular transforms are unsupported")
    scale=sqrt(scale2)
    rotation=atan2(c,a)
    def move(point):
        x,y=vector(point,2)
        return [finite(a*x+b*y+tx,"transformed x"),finite(c*x+d*y+ty,"transformed y")]
    out=deepcopy(entity)
    if not preserve_id:
        out["id"]=new_id()
    g=out["geometry"]
    kind=out["kind"]
    if kind=="point":
        g["position"]=move(g["position"])
    elif kind=="text":
        g["position"]=move(g["position"])
        g["height"]=finite(g["height"]*scale,"scaled text height")
        g["rotation"]=finite(g.get("rotation",0.0)+rotation,"text rotation",limit=1e8)
        if "width" in g:
            g["width"]=finite(g["width"]*scale,"scaled text width")
    elif kind=="insert":
        g["position"]=move(g["position"])
        g["rotation"]=finite(g.get("rotation",0.0)+rotation,"insert rotation",limit=1e8)
        g["scale"]=finite(g.get("scale",1.0)*scale,"insert scale")
    elif kind=="hatch":
        for index,loop in enumerate(g["boundary"]):
            g["boundary"][index]=_transform_loop(loop,move,scale,rotation)
        if g.get("pattern","solid")!="solid":
            g["angle"]=finite(g.get("angle",0.0)+rotation,"hatch angle",limit=1e8)
            g["spacing"]=finite(g["spacing"]*scale,"scaled hatch spacing")
    elif kind=="dimension":
        for key in ("p1","p2","center","point","dimline_point","arc_point","text_position"):
            if key in g:
                g[key]=move(g[key])
        if g["dim_type"]=="linear":
            g["angle"]=finite(g.get("angle",0.0)+rotation,"dimension angle",limit=1e8)
        for key in ("text_height","arrow_size"):
            if key in g:
                g[key]=finite(g[key]*scale,"scaled "+key)
    elif kind=="line":
        g["start"],g["end"]=move(g["start"]),move(g["end"])
    elif kind=="polyline":
        g["points"]=[move(p) for p in g["points"]]
    else:
        g["center"]=move(g["center"])
        g["radius"]=finite(g["radius"]*scale,"scaled radius")
        if kind=="arc":
            g["start_angle"]=finite(g["start_angle"]+rotation, "arc start", limit=1e8)
            g["end_angle"]=finite(g["end_angle"]+rotation, "arc end", limit=1e8)
    return validate_entity(out)


def _transform_loop(loop, move, scale, rotation):
    """Similarity transform of one hatch loop (circles stay circles, arcs stay arcs)."""
    kind = loop["kind"]
    if kind == "polygon":
        return {"kind": "polygon", "points": [move(p) for p in loop["points"]]}
    if kind == "circle":
        return {"kind": "circle", "center": move(loop["center"]),
                "radius": finite(loop["radius"] * scale, "scaled radius")}
    segments = []
    for seg in loop["segments"]:
        if seg["kind"] == "line":
            segments.append({"kind": "line", "start": move(seg["start"]), "end": move(seg["end"])})
        else:
            segments.append({"kind": "arc", "center": move(seg["center"]),
                             "radius": finite(seg["radius"] * scale, "scaled radius"),
                             "start_angle": finite(seg["start_angle"] + rotation, "arc start", limit=1e8),
                             "end_angle": finite(seg["end_angle"] + rotation, "arc end", limit=1e8)})
    return {"kind": "path", "segments": segments}
