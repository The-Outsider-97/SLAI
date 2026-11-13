"""SGEO source geometry: analytic primitives, indexed triangle meshes (optionally with UVs), NURBS curves and
surfaces, and boundary representations (B-rep).

``nurbs_curve`` / ``nurbs_surface`` carry the fields of :mod:`.nurbs`; ``brep`` is the topology model of
:mod:`.brep`. A mesh may carry ``uvs`` (one ``[u, v]`` per vertex; origin bottom-left, v up)."""
from __future__ import annotations
from math import sqrt
from ..common import finite, vector, ValidationError
from ..common.errors import GeometryValidationError
from . import brep as _brep
from . import nurbs as _nurbs

GEOMETRIES = frozenset({"point", "segment", "box", "sphere", "cylinder", "mesh", "nurbs_curve", "nurbs_surface", "brep"})
#: kinds introduced by SGEO 0.2.0 (rejected inside legacy 0.1.0 documents)
V02_GEOMETRIES = frozenset({"nurbs_curve", "nurbs_surface", "brep"})
MAX_UV_ABS = 1e6
MAX_VERTICES = 1_000_000
MAX_TRIANGLES = 2_000_000

def validate_geometry(geometry: dict) -> dict:
    if not isinstance(geometry, dict) or not isinstance(geometry.get("kind"), str) or geometry["kind"] not in GEOMETRIES:
        raise GeometryValidationError("invalid or unsupported SGEO geometry kind")
    kind = geometry["kind"]
    if kind == "point":
        vector(geometry.get("position"), 3)
    elif kind == "segment":
        a, b = vector(geometry.get("start"),3), vector(geometry.get("end"),3)
        if a == b:
            raise GeometryValidationError("zero-length segment")
    elif kind == "box":
        vector(geometry.get("center"),3)
        dims = vector(geometry.get("size"),3)
        if any(v <= 0 for v in dims):
            raise GeometryValidationError("box dimensions must be positive")
    elif kind in {"sphere","cylinder"}:
        vector(geometry.get("center"),3)
        if finite(geometry.get("radius"),"radius") <= 0:
            raise GeometryValidationError("radius must be positive")
        if kind == "cylinder" and finite(geometry.get("height"),"height") <= 0:
            raise GeometryValidationError("cylinder height must be positive")
    elif kind == "mesh":
        vertices = geometry.get("vertices")
        faces = geometry.get("faces")
        if not isinstance(vertices,list) or not 3 <= len(vertices) <= MAX_VERTICES:
            raise GeometryValidationError("mesh requires 3..1,000,000 vertices")
        if not isinstance(faces,list) or not 1 <= len(faces) <= MAX_TRIANGLES:
            raise GeometryValidationError("mesh requires 1..2,000,000 triangles")
        for p in vertices:
            vector(p,3,"vertex")
        for face in faces:
            if not isinstance(face,list) or len(face)!=3 or any(type(i) is not int or i<0 or i>=len(vertices) for i in face):
                raise GeometryValidationError("mesh face indices must refer to existing vertices")
            if len(set(face)) != 3:
                raise GeometryValidationError("repeated mesh vertex index")
            a, b, c = (vertices[i] for i in face)
            u = [b[i]-a[i] for i in range(3)]
            v = [c[i]-a[i] for i in range(3)]
            normal = (u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0])
            if not all(float("-inf") < n < float("inf") for n in normal) or sum(n*n for n in normal) <= 1e-30:
                raise GeometryValidationError("degenerate or overflowing mesh triangle")
        uvs = geometry.get("uvs")
        if uvs is not None:
            if not isinstance(uvs, list) or len(uvs) != len(vertices):
                raise GeometryValidationError("mesh uvs need exactly one [u, v] per vertex")
            for uv in uvs:
                if not isinstance(uv, list) or len(uv) != 2 or any(
                        isinstance(c, bool) or not isinstance(c, (int, float)) or not -MAX_UV_ABS <= c <= MAX_UV_ABS
                        for c in uv):
                    raise GeometryValidationError("mesh uvs must be finite [u, v] pairs within +-1e6")
    elif kind == "nurbs_curve":
        _nurbs.validate_curve({k: v for k, v in geometry.items() if k != "kind"})
    elif kind == "nurbs_surface":
        _nurbs.validate_surface({k: v for k, v in geometry.items() if k != "kind"})
    elif kind == "brep":
        _brep.validate_brep(geometry)
    allowed = {
        "point":{"kind","position"}, "segment":{"kind","start","end"},
        "box":{"kind","center","size"}, "sphere":{"kind","center","radius"},
        "cylinder":{"kind","center","radius","height"},
        "mesh":{"kind","vertices","faces","uvs"},
        "nurbs_curve":{"kind","degree","knots","control_points","weights"},
        "nurbs_surface":{"kind","degree_u","degree_v","knots_u","knots_v","control_points","weights"},
        "brep":set(_brep.BREP_KEYS)}
    if set(geometry) - allowed[kind]:
        raise GeometryValidationError("unknown fields in SGEO geometry")
    return geometry

def local_bounds(geometry):
    """Axis-aligned bounds in local scene coordinates; cylinder axis is local +Z."""
    validate_geometry(geometry)
    g,k=geometry,geometry["kind"]
    if k=="point":
        return tuple(g["position"]),tuple(g["position"])
    if k=="segment":
        pts=(g["start"],g["end"])
        return tuple(min(p[i] for p in pts) for i in range(3)),tuple(max(p[i] for p in pts) for i in range(3))
    if k=="mesh":
        pts=g["vertices"]
        return tuple(min(p[i] for p in pts) for i in range(3)),tuple(max(p[i] for p in pts) for i in range(3))
    if k=="nurbs_curve":
        return _nurbs.curve_bounds({x:y for x,y in g.items() if x!="kind"})
    if k=="nurbs_surface":
        return _nurbs.surface_bounds({x:y for x,y in g.items() if x!="kind"})
    if k=="brep":
        from .brep_mesh import brep_bounds
        return brep_bounds(g)
    c=g["center"]
    if k=="box":
        half=[d/2 for d in g["size"]]
    elif k=="sphere":
        half=[g["radius"]]*3
    else:
        half=[g["radius"],g["radius"],g["height"]/2]
    return tuple(c[i]-half[i] for i in range(3)),tuple(c[i]+half[i] for i in range(3))

def mesh_surface_area(geometry) -> float:
    validate_geometry(geometry)
    if geometry["kind"] != "mesh":
        raise GeometryValidationError("surface-area operation requires a triangle mesh")
    area=0.0
    for face in geometry["faces"]:
        a,b,c=(geometry["vertices"][i] for i in face)
        u=[b[i]-a[i] for i in range(3)]
        v=[c[i]-a[i] for i in range(3)]
        cross=(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0])
        area+=sqrt(sum(x*x for x in cross))/2
    return finite(area,"mesh surface area",limit=1e18)
