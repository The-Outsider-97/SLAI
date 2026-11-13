"""Bounded deterministic display tessellation of SGEO geometry.

Output meshes are approximations used for viewing, interchange, and spatial
queries; they do not replace exact primitive / NURBS / B-rep parameters as authoritative data.

``uvs=True`` adds texture coordinates (origin bottom-left, v up, ``[0, 1]`` per face or per surface):

* box     - every face is its own unit square (24 vertices);
* cylinder- ``u`` runs once around (seam duplicated), ``v`` from bottom (0) to top (1); each cap is a unit disc;
* sphere  - ``u`` is longitude, ``v`` latitude from the south pole (0) to the north pole (1);
* nurbs_surface / brep - parameter space normalised per face.
"""
from __future__ import annotations
from math import cos, pi, sin
from ..common.errors import ValidationError, ResourceLimitError
from .geometry import validate_geometry, MAX_TRIANGLES, MAX_VERTICES

# (normal, u axis, v axis) with u x v = normal, so the quad (-u-v, +u-v, +u+v, -u+v) is counter-clockwise outside
_BOX_FACES = (
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    ((0, 1, 0), (0, 0, 1), (1, 0, 0)), ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    ((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (0, 1, 0), (1, 0, 0)))


def tessellate(geometry: dict, *, radial_segments=32, latitude_bands=16, uvs=False) -> dict:
    validate_geometry(geometry)
    if type(radial_segments) is not int or not 8<=radial_segments<=256:
        raise ValidationError("radial_segments must be an integer in [8,256]")
    if type(latitude_bands) is not int or not 4<=latitude_bands<=128:
        raise ValidationError("latitude_bands must be an integer in [4,128]")
    kind=geometry["kind"]
    if kind=="mesh":
        mesh={"kind":"mesh","vertices":[list(v) for v in geometry["vertices"]],
              "faces":[list(face) for face in geometry["faces"]]}
        if "uvs" in geometry:
            mesh["uvs"]=[list(uv) for uv in geometry["uvs"]]
        return mesh
    if kind in {"point","segment","nurbs_curve"}:
        raise ValidationError("%s does not define a triangle surface" % kind)
    if kind in {"nurbs_surface","brep"}:
        from .brep_build import brep_from_nurbs_surface
        from .brep_mesh import tessellate_brep
        brep=brep_from_nurbs_surface(geometry) if kind=="nurbs_surface" else geometry
        return validate_geometry(tessellate_brep(brep,radial_segments=radial_segments,uvs=uvs))
    cx,cy,cz=geometry["center"]
    vertices=[];faces=[];tex=[]
    if kind=="box":
        sx,sy,sz=(s/2 for s in geometry["size"])
        if uvs:
            half=(sx,sy,sz)
            for normal,a,b in _BOX_FACES:
                base=len(vertices)
                for du,dv,uv in ((-1,-1,(0.0,0.0)),(1,-1,(1.0,0.0)),(1,1,(1.0,1.0)),(-1,1,(0.0,1.0))):
                    vertices.append([c+(normal[i]+du*a[i]+dv*b[i])*h for i,(c,h) in enumerate(zip((cx,cy,cz),half))])
                    tex.append(list(uv))
                faces.extend([[base,base+1,base+2],[base,base+2,base+3]])
        else:
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
        if uvs:
            for z,v in ((cz-half,0.0),(cz+half,1.0)):                       # side: seam duplicated
                for i in range(n+1):
                    angle=2*pi*(i%n)/n
                    vertices.append([cx+radius*cos(angle),cy+radius*sin(angle),z])
                    tex.append([i/n,v])
            for i in range(n):
                j=i+1
                faces.extend([[i,j,i+n+1],[j,j+n+1,i+n+1]])
            for z,sign in ((cz-half,-1),(cz+half,1)):                       # caps: own ring plus centre
                centre=len(vertices)
                vertices.append([cx,cy,z]);tex.append([0.5,0.5])
                for i in range(n):
                    angle=2*pi*i/n
                    vertices.append([cx+radius*cos(angle),cy+radius*sin(angle),z])
                    tex.append([0.5+0.5*cos(angle),0.5+0.5*sin(angle)])
                for i in range(n):
                    a,b=centre+1+i,centre+1+(i+1)%n
                    faces.append([centre,b,a] if sign<0 else [centre,a,b])
        else:
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
        if uvs:
            for band in range(k+1):                                         # poles duplicated per column
                theta=pi*band/k
                for i in range(n+1):
                    phi=2*pi*(i%n)/n
                    vertices.append([cx+radius*sin(theta)*cos(phi),cy+radius*sin(theta)*sin(phi),cz+radius*cos(theta)])
                    tex.append([i/n,1.0-band/k])
            for band in range(k):
                for i in range(n):
                    a=band*(n+1)+i
                    b=a+1
                    c=a+n+1
                    d=c+1
                    if band!=k-1:
                        faces.append([a,c,d])
                    if band!=0:
                        faces.append([a,d,b])
        else:
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
    if uvs:
        result["uvs"]=tex
    return validate_geometry(result)
