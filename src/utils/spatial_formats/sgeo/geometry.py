"""SGEO v0.1 analytic primitives and indexed triangle meshes (source geometry)."""
from __future__ import annotations
from math import sqrt
from ..common import finite, vector, ValidationError

GEOMETRIES = frozenset({"point", "segment", "box", "sphere", "cylinder", "mesh"})
MAX_VERTICES = 1_000_000
MAX_TRIANGLES = 2_000_000

def validate_geometry(geometry: dict) -> dict:
    if not isinstance(geometry, dict) or not isinstance(geometry.get("kind"), str) or geometry["kind"] not in GEOMETRIES:
        raise ValidationError("invalid or unsupported SGEO geometry kind")
    kind = geometry["kind"]
    if kind == "point":
        vector(geometry.get("position"), 3)
    elif kind == "segment":
        a, b = vector(geometry.get("start"),3), vector(geometry.get("end"),3)
        if a == b:
            raise ValidationError("zero-length segment")
    elif kind == "box":
        vector(geometry.get("center"),3)
        dims = vector(geometry.get("size"),3)
        if any(v <= 0 for v in dims):
            raise ValidationError("box dimensions must be positive")
    elif kind in {"sphere","cylinder"}:
        vector(geometry.get("center"),3)
        if finite(geometry.get("radius"),"radius") <= 0:
            raise ValidationError("radius must be positive")
        if kind == "cylinder" and finite(geometry.get("height"),"height") <= 0:
            raise ValidationError("cylinder height must be positive")
    elif kind == "mesh":
        vertices = geometry.get("vertices")
        faces = geometry.get("faces")
        if not isinstance(vertices,list) or not 3 <= len(vertices) <= MAX_VERTICES:
            raise ValidationError("mesh requires 3..1,000,000 vertices")
        if not isinstance(faces,list) or not 1 <= len(faces) <= MAX_TRIANGLES:
            raise ValidationError("mesh requires 1..2,000,000 triangles")
        for p in vertices:
            vector(p,3,"vertex")
        for face in faces:
            if not isinstance(face,list) or len(face)!=3 or any(type(i) is not int or i<0 or i>=len(vertices) for i in face):
                raise ValidationError("mesh face indices must refer to existing vertices")
            if len(set(face)) != 3:
                raise ValidationError("repeated mesh vertex index")
            a, b, c = (vertices[i] for i in face)
            u = [b[i]-a[i] for i in range(3)]
            v = [c[i]-a[i] for i in range(3)]
            normal = (u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0])
            if not all(float("-inf") < n < float("inf") for n in normal) or sum(n*n for n in normal) <= 1e-30:
                raise ValidationError("degenerate or overflowing mesh triangle")
    allowed = {
        "point":{"kind","position"}, "segment":{"kind","start","end"},
        "box":{"kind","center","size"}, "sphere":{"kind","center","radius"},
        "cylinder":{"kind","center","radius","height"},
        "mesh":{"kind","vertices","faces"}}
    if set(geometry) - allowed[kind]:
        raise ValidationError("unknown fields in SGEO geometry")
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
        raise ValidationError("surface-area operation requires a triangle mesh")
    area=0.0
    for face in geometry["faces"]:
        a,b,c=(geometry["vertices"][i] for i in face)
        u=[b[i]-a[i] for i in range(3)]
        v=[c[i]-a[i] for i in range(3)]
        cross=(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0])
        area+=sqrt(sum(x*x for x in cross))/2
    return finite(area,"mesh surface area",limit=1e18)
