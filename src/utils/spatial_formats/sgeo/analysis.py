"""Deterministic mesh topology and mass-property diagnostics (CPU-only).

Topological watertightness is necessary but not sufficient for a valid solid:
self-intersection, inward winding and disconnected shells must be resolved by
the caller before using a mesh for manufacturing or structural calculations.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from math import fsum, sqrt
from ..common import ValidationError, finite
from .geometry import validate_geometry


@dataclass(frozen=True)
class MeshDiagnostics:
    vertex_count: int
    triangle_count: int
    edge_count: int
    boundary_edge_count: int
    non_manifold_edge_count: int
    inconsistent_winding_edge_count: int
    isolated_vertex_count: int
    connected_components: int
    is_closed_oriented_surface: bool
    surface_area: float
    signed_volume: float | None

    def as_dict(self):
        return asdict(self)


def analyze_mesh(geometry: dict) -> MeshDiagnostics:
    validate_geometry(geometry)
    if geometry["kind"] != "mesh":
        raise ValidationError("mesh diagnostics require triangle mesh geometry")
    vertices, faces=geometry["vertices"], geometry["faces"]
    adjacency={i:[] for i in range(len(vertices))}
    used=set()
    edges={}
    areas=[]
    volumes=[]
    for face in faces:
        a,b,c=face
        used.update(face)
        p,q,r=(vertices[i] for i in face)
        u=(q[0]-p[0],q[1]-p[1],q[2]-p[2])
        v=(r[0]-p[0],r[1]-p[1],r[2]-p[2])
        n=(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0])
        areas.append(sqrt(sum(x*x for x in n))/2)
        volumes.append((p[0]*(q[1]*r[2]-q[2]*r[1])
                        +p[1]*(q[2]*r[0]-q[0]*r[2])
                        +p[2]*(q[0]*r[1]-q[1]*r[0]))/6)
        for start,end in ((a,b),(b,c),(c,a)):
            key=(min(start,end),max(start,end))
            edges.setdefault(key,[]).append((start,end))
            adjacency[start].append(end)
            adjacency[end].append(start)
    boundary=sum(len(uses)==1 for uses in edges.values())
    nonmanifold=sum(len(uses)>2 for uses in edges.values())
    winding=sum(len(uses)==2 and uses[0]==uses[1] for uses in edges.values())
    unvisited=set(used)
    components=0
    while unvisited:
        components+=1
        stack=[unvisited.pop()]
        while stack:
            for neighbour in adjacency[stack.pop()]:
                if neighbour in unvisited:
                    unvisited.remove(neighbour)
                    stack.append(neighbour)
    closed=(bool(faces) and boundary==0 and nonmanifold==0 and winding==0)
    area=finite(fsum(areas),"surface area",limit=1e18)
    volume=finite(fsum(volumes),"signed volume",limit=1e18) if closed else None
    return MeshDiagnostics(len(vertices),len(faces),len(edges),boundary,nonmanifold,
                           winding,len(vertices)-len(used),components,closed,area,volume)


def transformed_bounds(bounds, transform):
    """Transform all eight AABB corners; conservative world-space bounds."""
    from itertools import product
    from ..common.transforms import transform_point
    minimum,maximum=bounds
    if len(minimum)!=3 or len(maximum)!=3:
        raise ValidationError("bounds require 3D coordinate tuples")
    corners=(transform_point(transform,p) for p in product(*zip(minimum,maximum)))
    xyz=list(corners)
    return (tuple(min(p[i] for p in xyz) for i in range(3)),
            tuple(max(p[i] for p in xyz) for i in range(3)))
