"""Bounded deterministic display tessellation of SGEO analytic primitives.

Output meshes are approximations used for viewing, interchange, and spatial
queries; they do not replace exact primitive parameters as authoritative data.
"""
from __future__ import annotations
from math import cos, pi, sin
from ..common.errors import ValidationError, ResourceLimitError
from .geometry import validate_geometry, MAX_TRIANGLES, MAX_VERTICES

def tessellate(geometry: dict, *, radial_segments=32, latitude_bands=16) -> dict:
    validate_geometry(geometry)
    if type(radial_segments) is not int or not 8<=radial_segments<=256:
        raise ValidationError("radial_segments must be an integer in [8,256]")
    if type(latitude_bands) is not int or not 4<=latitude_bands<=128:
        raise ValidationError("latitude_bands must be an integer in [4,128]")
    kind=geometry["kind"]
    if kind=="mesh":
        return {"kind":"mesh","vertices":[list(v) for v in geometry["vertices"]],
                "faces":[list(face) for face in geometry["faces"]]}
    if kind in {"point","segment"}:
        raise ValidationError("point/segment does not define a triangle surface")
    cx,cy,cz=geometry["center"]
    vertices=[];faces=[]
    if kind=="box":
        sx,sy,sz=(s/2 for s in geometry["size"])
        vertices=[[cx+dx*sx,cy+dy*sy,cz+dz*sz]
                  for dx,dy,dz in ((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
                                    (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1))]
        faces=[[0,2,1],[0,3,2],[4,5,6],[4,6,7],
               [0,1,5],[0,5,4],[3,7,6],[3,6,2],
               [0,4,7],[0,7,3],[1,2,6],[1,6,5]]
    elif kind=="cylinder":
        n=radial_segments
        radius=geometry["radius"]
        half=geometry["height"]/2
        for z in (cz-half,cz+half):
            vertices.extend([[cx+radius*cos(2*pi*i/n),cy+radius*sin(2*pi*i/n),z]
                             for i in range(n)])
        bottom,top=len(vertices),len(vertices)+1
        vertices.extend([[cx,cy,cz-half],[cx,cy,cz+half]])
        for i in range(n):
            j=(i+1)%n
            faces.extend([[i,j,i+n],[j,j+n,i+n],
                          [bottom,j,i],[top,i+n,j+n]])
    else:
        n,k=radial_segments,latitude_bands
        radius=geometry["radius"]
        vertices.append([cx,cy,cz+radius])
        for band in range(1,k):
            theta=pi*band/k
            for i in range(n):
                phi=2*pi*i/n
                vertices.append([cx+radius*sin(theta)*cos(phi),
                                 cy+radius*sin(theta)*sin(phi),
                                 cz+radius*cos(theta)])
        south=len(vertices)
        vertices.append([cx,cy,cz-radius])
        for i in range(n):
            j=(i+1)%n
            faces.append([0,1+i,1+j])
            for band in range(k-2):
                a=1+band*n+i
                b=1+band*n+j
                c=a+n
                d=b+n
                faces.extend([[a,c,d],[a,d,b]])
            faces.append([south,1+(k-2)*n+j,1+(k-2)*n+i])
    if len(vertices)>MAX_VERTICES or len(faces)>MAX_TRIANGLES:
        raise ResourceLimitError("tessellation resource budget exceeded")
    result={"kind":"mesh","vertices":vertices,"faces":faces}
    return validate_geometry(result)
