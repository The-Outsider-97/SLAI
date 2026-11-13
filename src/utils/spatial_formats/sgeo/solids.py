"""Controlled convex-planar extrusion and revolution to validated triangle meshes.

Outputs are deterministic tessellated approximations, not exact B-reps. Only
simple convex closed profiles are supported. This explicit restriction avoids
silently triangulating concave/self-intersecting profiles incorrectly.
"""
from __future__ import annotations
from math import cos, hypot, pi, sin, sqrt
from ..common.errors import ValidationError
from ..common.helpers import finite, vector
from .geometry import validate_geometry
from .analysis import analyze_mesh


def _convex_profile(points, *, axis_radius=False):
    if not isinstance(points,(list,tuple)) or not 3<=len(points)<=4096:
        raise ValidationError("profile requires 3..4096 vertices")
    vertices=[vector(p,2,"profile coordinate") for p in points]
    if vertices[0]==vertices[-1]:
        vertices.pop()
    if len(vertices)<3 or len(set(vertices))!=len(vertices):
        raise ValidationError("profile contains duplicate corners")
    # Convexity checks also prohibit self-intersection for a simple polygon.
    cross=[]
    for i in range(len(vertices)):
        a,b,c=vertices[i-1],vertices[i],vertices[(i+1)%len(vertices)]
        z=(b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0])
        if abs(z)<=1e-12:
            raise ValidationError("collinear or near-degenerate profile edges")
        cross.append(z)
    if not (all(x>0 for x in cross) or all(x<0 for x in cross)):
        raise ValidationError("only simple convex profiles are supported")
    if axis_radius and any(r<=0 for r,z in vertices):
        raise ValidationError("revolution profile must have strictly positive radii")
    return vertices


def extrude(profile, depth, *, direction=(0,0,1)):
    """Generate a closed, consistently oriented convex extruded prism.

    The original planar sketch lies on z=0. Direction must have nonzero z
    component; a horizontal sweep produces no volume and is rejected.
    """
    pts=_convex_profile(profile)
    distance=finite(depth,"extrusion depth")
    if distance<=0:
        raise ValidationError("extrusion depth must be positive")
    dx,dy,dz=vector(direction,3,"extrusion direction")
    norm=hypot(hypot(dx,dy),dz)
    if norm<=1e-12 or abs(dz)/norm < 1e-9:
        raise ValidationError("extrusion direction must not be parallel to sketch")
    shift=tuple(distance*x/norm for x in (dx,dy,dz))
    n=len(pts)
    vertices=[[x,y,0.0] for x,y in pts]
    vertices.extend([[x+shift[0],y+shift[1],shift[2]] for x,y in pts])
    faces=[]
    for i in range(1,n-1):
        faces.extend([[0,i+1,i],[n,n+i,n+i+1]])
    for i in range(n):
        j=(i+1)%n
        faces.extend([[i,j,n+j],[i,n+j,n+i]])
    geom={"kind":"mesh","vertices":vertices,"faces":faces}
    validate_geometry(geom)
    diag=analyze_mesh(geom)
    if not diag.is_closed_oriented_surface:
        raise ValidationError("extrusion output did not produce closed manifold")
    if diag.signed_volume<0:
        geom["faces"]=[[a,c,b] for a,b,c in geom["faces"]]
    return validate_geometry(geom)


def revolve(profile, *, sweep=2*pi, segments=64):
    """Rotate a convex radial/elevation section about +Z.

    0 < sweep <= 2π; partial sweeps are closed by planar end caps.
    No profile may cross or touch the rotation axis.
    """
    pts=_convex_profile(profile,axis_radius=True)
    angle=finite(sweep,"sweep angle")
    if not 0<angle<=2*pi:
        raise ValidationError("sweep must lie in (0, 2π]")
    if type(segments) is not int or not 8<=segments<=256:
        raise ValidationError("revolution segments must lie in [8,256]")
    if angle==2*pi:
        rings=segments
        closed=True
    else:
        rings=segments+1
        closed=False
    n=len(pts)
    vertices=[]
    for i in range(rings):
        theta=angle*i/segments
        vertices.extend([[r*cos(theta),r*sin(theta),z] for r,z in pts])
    faces=[]
    for j in range(segments):
        if closed:
            right=(j+1)%rings
        else:
            right=j+1
        for i in range(n):
            k=(i+1)%n
            a,b=j*n+i,j*n+k
            c,d=right*n+i,right*n+k
            faces.extend([[a,b,d],[a,d,c]])
    if not closed:
        for i in range(1,n-1):
            faces.extend([[0,i+1,i],
                          [segments*n,segments*n+i,segments*n+i+1]])
    result={"kind":"mesh","vertices":vertices,"faces":faces}
    validate_geometry(result)
    diag=analyze_mesh(result)
    if not diag.is_closed_oriented_surface:
        raise ValidationError("revolution output did not produce closed manifold")
    if diag.signed_volume is None or abs(diag.signed_volume)<=1e-12:
        raise ValidationError("revolution produced degenerate solid")
    if diag.signed_volume<0:
        result["faces"]=[[a,c,b] for a,b,c in result["faces"]]
    return validate_geometry(result)
