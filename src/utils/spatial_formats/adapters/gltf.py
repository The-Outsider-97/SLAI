"""glTF 2.0 scene exchange via an optional trimesh backend.

The SGEO scene unit is meters, right-handed +Z up. glTF is right-handed +Y
up. Coordinates and node transforms are converted by a proper +90 degree X
rotation; a non-identity root transform is never silently discarded.
This scoped adapter explicitly refuses animated, externally-resourced, or
multi-primitive material-rich scenes where the selected backend cannot make
preservation guarantees. Mesh exports are presentation geometry only.
"""
from __future__ import annotations
from io import BytesIO
from ..common.errors import DependencyUnavailableError, UnsupportedFeatureError, ValidationError, ResourceLimitError
from ..common.transforms import IDENTITY, compose, invert_transform, require_transform
from ..sgeo.geometry import validate_geometry
from ..sgeo.scene import Scene
from ..common.helpers import digest
from .contracts import Capabilities

MAX_GLTF_BYTES=128*1024*1024
# Map glTF (x,y,z) to SGEO (x,-z,y), determinant +1.
Y_UP_TO_Z=(1,0,0,0, 0,0,-1,0, 0,1,0,0, 0,0,0,1)
Z_UP_TO_Y=(1,0,0,0, 0,0,1,0, 0,-1,0,0, 0,0,0,1)

def _trimesh():
    try: import trimesh
    except ImportError as exc: raise DependencyUnavailableError("trimesh is required for glTF/GLB interchange") from exc
    return trimesh

def capabilities():
    try: backend=_trimesh();available=True
    except DependencyUnavailableError: available=False
    return Capabilities("glTF","trimesh",available,("2.0",) if available else (),
                        ("2.0",) if available else (),("triangle-mesh","node-hierarchy","instances"),
                        ("No animation/skin round-trip","No exact CAD/BIM or texture semantics"))

def _matrix_to_row_major(matrix):
    import numpy as np
    m=np.asarray(matrix,dtype=float)
    if m.shape!=(4,4): raise ValidationError("invalid glTF node transform shape")
    return require_transform(m.reshape(16).tolist())

def _conjugate(transform,to_z=True):
    basis=Y_UP_TO_Z if to_z else Z_UP_TO_Y
    inverse=Z_UP_TO_Y if to_z else Y_UP_TO_Z
    return compose(compose(basis,transform),inverse)

def _map_vertices(vertices, *, to_z: bool):
    """Apply the exact proper basis change to mesh-local position vectors.

    Conjugating node matrices alone does not change mesh-local coordinates.
    Transform both positions and local matrices to preserve world geometry.
    """
    import numpy as np
    points=np.asarray(vertices,dtype=np.float64)
    if points.ndim!=2 or points.shape[1]!=3:
        raise ValidationError("invalid glTF vertex array")
    if to_z:
        converted=np.column_stack((points[:,0],-points[:,2],points[:,1]))
    else:
        converted=np.column_stack((points[:,0],points[:,2],-points[:,1]))
    if not np.isfinite(converted).all():
        raise ValidationError("nonfinite mesh vertices")
    return converted


def _validate_embedded_glb(data):
    """Reject external resources, extensions and non-triangle primitives.

    This is an inexpensive structural preflight, not a replacement for
    trimesh's geometry validation.
    """
    import json
    offset=12
    model=None
    binary_chunk_found=False
    while offset < len(data):
        if len(data)-offset < 8:
            raise ValidationError("truncated GLB chunk header")
        size=int.from_bytes(data[offset:offset+4],"little")
        chunk_type=data[offset+4:offset+8]
        offset+=8
        if size > len(data)-offset:
            raise ValidationError("GLB chunk exceeds file boundaries")
        payload=data[offset:offset+size]
        offset+=size
        if model is None:
            if chunk_type!=b"JSON":
                raise ValidationError("GLB JSON chunk must be first")
            try:
                model=json.loads(payload.decode("utf-8").rstrip(" \x00"))
            except (UnicodeError,ValueError) as exc:
                raise ValidationError("invalid GLB JSON") from exc
        elif chunk_type==b"BIN\x00":
            if binary_chunk_found:
                raise ValidationError("duplicate GLB binary chunk")
            binary_chunk_found=True
        else:
            raise UnsupportedFeatureError("unsupported GLB chunk type")
    if not isinstance(model,dict) or not isinstance(model.get("asset"),dict) or model["asset"].get("version")!="2.0":
        raise ValidationError("not a glTF 2.0 asset")
    if model.get("extensionsRequired") or model.get("extensionsUsed") or model.get("animations") or model.get("skins"):
        raise UnsupportedFeatureError("GLB extensions, animations and skins are not supported")
    for buffer in model.get("buffers",[]):
        if not isinstance(buffer,dict) or "uri" in buffer:
            raise UnsupportedFeatureError("GLB external or data URI buffers are prohibited")
    for image in model.get("images",[]):
        if not isinstance(image,dict) or "uri" in image:
            raise UnsupportedFeatureError("GLB external image references are prohibited")
    for mesh in model.get("meshes",[]):
        for primitive in mesh.get("primitives",[]):
            if primitive.get("mode",4)!=4:
                raise UnsupportedFeatureError("only glTF TRIANGLES primitive mode is supported")
    if len(model.get("nodes",[]))>10000 or len(model.get("meshes",[]))>10000:
        raise ResourceLimitError("glTF scene node/mesh count exceeds budget")
    if model.get("materials") or model.get("textures") or model.get("images"):
        raise UnsupportedFeatureError("glTF material and texture semantics are not yet preserved")
    return model


def import_glb(data:bytes)->dict:
    """Legacy single-mesh import. Rejects scenes that would lose hierarchy."""
    scene=import_glb_scene(data)
    objects=[o for o in scene.objects if o.get("geometry") is not None]
    if len(objects)!=1 or len(scene.objects)!=1:
        raise UnsupportedFeatureError("single-mesh API cannot preserve multi-node GLB")
    if tuple(objects[0]["transform"])!=IDENTITY:
        raise UnsupportedFeatureError("single-mesh API cannot discard node transform")
    return objects[0]["geometry"]

def import_glb_scene(data:bytes)->Scene:
    if not isinstance(data,bytes) or not 20<=len(data)<=MAX_GLTF_BYTES or data[:4]!=b"glTF":
        raise ValidationError("invalid or oversized GLB")
    if int.from_bytes(data[4:8],"little")!=2 or int.from_bytes(data[8:12],"little")!=len(data):
        raise ValidationError("GLB header/version/length mismatch")
    _validate_embedded_glb(data)
    trimesh=_trimesh()
    import numpy as np
    try: scene=trimesh.load(BytesIO(data),file_type="glb",force="scene",process=False)
    except Exception as exc: raise ValidationError("GLB backend rejected content") from exc
    if len(scene.graph.nodes)>10000 or len(scene.geometry)>10000:
        raise ResourceLimitError("GLB node/mesh budget exceeded")
    result=Scene(name="Imported glTF",unit="m")
    # trimesh returns world matrices; reconstruct local matrices relative to parent.
    nodes=list(scene.graph.nodes)
    if len(nodes)>10000: raise ResourceLimitError("excessive node count")
    entries={}
    for name in nodes:
        if name==scene.graph.base_frame: continue
        world,geometry_key=scene.graph.get(name)
        local=_matrix_to_row_major(world)
        parents=scene.graph.to_edgelist()
        parent=next((source for source,target,_ in parents if target==name),None)
        if parent and parent!=scene.graph.base_frame:
            parent_world,_=scene.graph.get(parent)
            local=compose(invert_transform(_matrix_to_row_major(parent_world)),local)
        mesh=None
        if geometry_key is not None:
            m=scene.geometry[geometry_key]
            if not isinstance(m,trimesh.Trimesh) or len(m.faces)>2_000_000:
                raise UnsupportedFeatureError("GLB contains unsupported mesh or face budget")
            mesh=validate_geometry({"kind":"mesh","vertices":_map_vertices(m.vertices,to_z=True).tolist(),"faces":m.faces.tolist()})
        entries[name]=(parent,local,mesh)
    if len(entries)>10000: raise ResourceLimitError("GLB hierarchy too large")
    pending=dict(entries);mapped={}
    while pending:
        progressed=False
        for name,(parent,local,mesh) in list(pending.items()):
            if parent not in (None,scene.graph.base_frame) and parent not in mapped: continue
            oid=result.add_object(name=str(name)[:255],geometry=mesh,
                parent=mapped.get(parent),transform=_conjugate(local))
            mapped[name]=oid;del pending[name];progressed=True
        if not progressed: raise ValidationError("cyclic or invalid glTF graph")
    return result

def export_glb(geometry:dict)->bytes:
    validate_geometry(geometry)
    if geometry["kind"]!="mesh":
        raise UnsupportedFeatureError("GLB mesh export requires triangles")
    trimesh=_trimesh()
    mesh=trimesh.Trimesh(vertices=_map_vertices(geometry["vertices"],to_z=False),faces=geometry["faces"],process=False)
    return mesh.export(file_type="glb")

def export_glb_scene(scene:Scene)->bytes:
    """Export triangle meshes and parent transforms; refuse unsupported semantics."""
    scene.validate()
    if scene.families or scene.metadata:
        raise UnsupportedFeatureError("GLB cannot preserve parametric families or scene semantic metadata")
    trimesh=_trimesh()
    import numpy as np
    target=trimesh.Scene()
    existing={}
    pending={o["id"]:o for o in scene.objects}
    while pending:
        progressed=False
        for oid,obj in list(pending.items()):
            parent=obj.get("parent")
            if parent is not None and parent not in existing: continue
            geometry=obj.get("geometry")
            if geometry is not None and geometry["kind"]!="mesh":
                raise UnsupportedFeatureError("GLB scene export requires triangle meshes")
            if obj.get("family_id") or obj.get("metadata") or obj.get("parameters"):
                raise UnsupportedFeatureError("GLB cannot preserve SGEO object semantics or families")
            local=np.array(_conjugate(obj.get("transform",IDENTITY),False)).reshape(4,4)
            if geometry is not None:
                mesh=trimesh.Trimesh(vertices=_map_vertices(geometry["vertices"],to_z=False),faces=geometry["faces"],process=False)
                target.add_geometry(mesh,node_name=oid,geom_name=oid,parent_node_name=existing.get(parent),
                                    transform=local)
            else:
                target.graph.update(frame_to=oid,frame_from=existing.get(parent,target.graph.base_frame),
                                    matrix=local)
            existing[oid]=oid
            del pending[oid];progressed=True
        if not progressed: raise ValidationError("invalid scene hierarchy")
    return target.export(file_type="glb")
