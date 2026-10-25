"""glTF/GLB mesh exchange via optional trimesh backend.

Export deliberately supports a single mesh and rejects non-mesh SGEO semantics.
GLB is a presentation mesh, not a lossless scene/family interchange.
"""
from __future__ import annotations
from ..common.errors import DependencyUnavailableError, UnsupportedFeatureError, ValidationError
from ..sgeo.geometry import validate_geometry

def _trimesh():
    try: import trimesh
    except ImportError as exc: raise DependencyUnavailableError("trimesh is required for glTF/GLB exchange") from exc
    return trimesh

def export_glb(geometry:dict) -> bytes:
    validate_geometry(geometry)
    if geometry["kind"]!="mesh":
        raise UnsupportedFeatureError("only triangle meshes can be converted to GLB")
    lib=_trimesh()
    mesh=lib.Trimesh(vertices=geometry["vertices"],faces=geometry["faces"],process=False)
    return mesh.export(file_type="glb")

def import_glb(data:bytes)->dict:
    if not isinstance(data,bytes) or not 0 < len(data) <= 128*1024*1024:
        raise ValidationError("GLB data exceeds byte budget")
    if data[:4]!=b"glTF":
        raise ValidationError("GLB magic invalid")
    lib=_trimesh()
    from io import BytesIO
    asset=lib.load(BytesIO(data),file_type="glb",force="scene")
    meshes=list(asset.geometry.values())
    if len(meshes)!=1:
        raise UnsupportedFeatureError("GLB importer accepts exactly one uninstanced mesh")
    # Transform semantics are intentionally not applied: reject nonidentity nodes.
    import numpy as np
    for node in asset.graph.nodes_geometry:
        transform,_=asset.graph[node]
        if not np.allclose(transform,np.eye(4),atol=1e-9,rtol=0):
            raise UnsupportedFeatureError("GLB node transformations require scene conversion")
    mesh=meshes[0]
    result={"kind":"mesh","vertices":mesh.vertices.tolist(),"faces":mesh.faces.tolist()}
    return validate_geometry(result)
