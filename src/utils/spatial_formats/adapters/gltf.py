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


def _validate_glb_arrays(model, binary_size):
    """Validate references and allocation sizes before calling the mesh backend."""
    from ..common.helpers import MAX_NODES
    for key in ("buffers","bufferViews","accessors","meshes","nodes","scenes"):
        values=model.get(key,[])
        if not isinstance(values,list) or any(not isinstance(v,dict) for v in values):
            raise ValidationError("invalid glTF " + key)
        if len(values)>10000:
            raise ResourceLimitError("glTF table exceeds budget")
    def integer(value, label, minimum=0):
        if type(value) is not int or value<minimum:
            raise ValidationError("invalid glTF " + label)
        return value
    def reference(value, values, label):
        index=integer(value,label)
        if index>=len(values):
            raise ValidationError("glTF reference outside " + label)
        return values[index]
    buffers=model.get("buffers",[])
    if len(buffers)>1:
        raise UnsupportedFeatureError("GLB supports one embedded buffer")
    if buffers:
        size=integer(buffers[0].get("byteLength"),"buffer size",1)
        if not size<=binary_size<=size+3:
            raise ValidationError("GLB binary size disagrees with buffer")
    elif binary_size:
        raise ValidationError("undeclared GLB binary buffer")
    views=model.get("bufferViews",[])
    for view in views:
        buffer=reference(view.get("buffer"),buffers,"buffer")
        offset=integer(view.get("byteOffset",0),"view offset")
        length=integer(view.get("byteLength"),"view length",1)
        if offset+length>buffer["byteLength"]:
            raise ValidationError("GLB buffer view exceeds buffer")
        if "byteStride" in view:
            stride=integer(view["byteStride"],"stride",4)
            if stride>252 or stride%4:
                raise ValidationError("invalid glTF stride")
    accessors=model.get("accessors",[])
    components={5120:1,5121:1,5122:2,5123:2,5125:4,5126:4}
    widths={"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4}
    expanded=0
    for accessor in accessors:
        if "sparse" in accessor:
            raise UnsupportedFeatureError("sparse glTF accessors are unsupported")
        view=reference(accessor.get("bufferView"),views,"buffer view")
        component=accessor.get("componentType")
        typ=accessor.get("type")
        if type(component) is not int or component not in components or not isinstance(typ,str) or typ not in widths:
            raise UnsupportedFeatureError("unsupported glTF accessor element type")
        count=integer(accessor.get("count"),"accessor count",1)
        expanded+=count*(widths[typ]+1)
        if expanded>MAX_NODES:
            raise ResourceLimitError("glTF decoded array budget exceeded")
        offset=integer(accessor.get("byteOffset",0),"accessor offset")
        width=components[component]*widths[typ]
        stride=view.get("byteStride",width)
        if stride<width or offset%components[component] or (view.get("byteOffset",0)+offset)%components[component]:
            raise ValidationError("misaligned glTF accessor")
        if offset+(count-1)*stride+width>view["byteLength"]:
            raise ValidationError("glTF accessor exceeds buffer view")
    meshes=model.get("meshes",[])
    for mesh in meshes:
        primitives=mesh.get("primitives")
        if not isinstance(primitives,list) or len(primitives)!=1:
            raise UnsupportedFeatureError("one triangle primitive per mesh is supported")
        primitive=primitives[0]
        if not isinstance(primitive,dict) or not isinstance(primitive.get("attributes"),dict):
            raise ValidationError("invalid glTF primitive")
        if primitive.get("targets") or primitive.get("extensions"):
            raise UnsupportedFeatureError("glTF morph targets and primitive extensions are unsupported")
        position=reference(primitive["attributes"].get("POSITION"),accessors,"position accessor")
        if position["type"]!="VEC3" or position["componentType"]!=5126:
            raise ValidationError("positions must be float32 VEC3")
        for index in primitive["attributes"].values():
            attribute=reference(index,accessors,"attribute accessor")
            if attribute["count"]!=position["count"]:
                raise ValidationError("glTF attribute counts disagree")
        count=position["count"]
        if "indices" in primitive:
            indices=reference(primitive["indices"],accessors,"index accessor")
            if indices["type"]!="SCALAR" or indices["componentType"] not in (5121,5123,5125):
                raise ValidationError("invalid glTF triangle index type")
            count=indices["count"]
        if count%3:
            raise ValidationError("incomplete glTF triangle")
    nodes=model.get("nodes",[])
    parents={}
    instance_budget=0
    for index,node in enumerate(nodes):
        if "mesh" in node:
            mesh=reference(node["mesh"],meshes,"mesh")
            primitive=mesh["primitives"][0]
            position=accessors[primitive["attributes"]["POSITION"]]
            indices=accessors[primitive["indices"]] if "indices" in primitive else position
            instance_budget += position["count"]*4 + indices["count"]*2
            if instance_budget>MAX_NODES:
                raise ResourceLimitError("glTF instantiated geometry exceeds budget")
        children=node.get("children",[])
        if not isinstance(children,list):
            raise ValidationError("invalid glTF children")
        for child in children:
            reference(child,nodes,"child node")
            if child in parents:
                raise ValidationError("glTF node has multiple parent references")
            parents[child]=index
    visited=set()
    for node in range(len(nodes)):
        chain=set()
        cursor=node
        while cursor is not None and cursor not in visited:
            if cursor in chain:
                raise ValidationError("cyclic glTF hierarchy")
            chain.add(cursor)
            cursor=parents.get(cursor)
        visited.update(chain)
    scenes=model.get("scenes",[])
    for scene in scenes:
        roots=scene.get("nodes",[])
        if not isinstance(roots,list):
            raise ValidationError("invalid glTF scene roots")
        for index in roots:
            reference(index,nodes,"scene root")
            if index in parents:
                raise ValidationError("glTF scene root has a parent")
    if "scene" in model:
        reference(model["scene"],scenes,"default scene")


def _validate_embedded_glb(data):
    """Reject external resources, extensions and non-triangle primitives.

    This is an inexpensive structural preflight, not a replacement for
    trimesh's geometry validation.
    """
    from ..common.helpers import parse_json
    offset=12
    model=None
    binary_chunk_found=False
    binary_size=0
    while offset < len(data):
        if len(data)-offset < 8:
            raise ValidationError("truncated GLB chunk header")
        size=int.from_bytes(data[offset:offset+4],"little")
        chunk_type=data[offset+4:offset+8]
        offset+=8
        if size % 4 or size > len(data)-offset:
            raise ValidationError("GLB chunk exceeds file boundaries")
        payload=data[offset:offset+size]
        offset+=size
        if model is None:
            if chunk_type!=b"JSON":
                raise ValidationError("GLB JSON chunk must be first")
            try:
                model=parse_json(payload.rstrip(b" "))
            except (UnicodeError,ValueError) as exc:
                raise ValidationError("invalid GLB JSON") from exc
        elif chunk_type==b"BIN\x00":
            if binary_chunk_found:
                raise ValidationError("duplicate GLB binary chunk")
            binary_chunk_found=True
            binary_size=size
        else:
            raise UnsupportedFeatureError("unsupported GLB chunk type")
    if not isinstance(model,dict) or not isinstance(model.get("asset"),dict) or model["asset"].get("version")!="2.0":
        raise ValidationError("not a glTF 2.0 asset")
    if model.get("extensionsRequired") or model.get("extensionsUsed") or model.get("animations") or model.get("skins"):
        raise UnsupportedFeatureError("GLB extensions, animations and skins are not supported")
    _validate_glb_arrays(model,binary_size)
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
    parents={target:source for source,target,_ in scene.graph.to_edgelist()}
    for name in nodes:
        if name==scene.graph.base_frame: continue
        world,geometry_key=scene.graph.get(name)
        local=_matrix_to_row_major(world)
        parent=parents.get(name)
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

def _export_mesh(geometry, scale=1.0):
    vertices=(_map_vertices(geometry["vertices"],to_z=False)*scale).astype("float32")
    validate_geometry({"kind":"mesh","vertices":vertices.tolist(),"faces":geometry["faces"]})
    return _trimesh().Trimesh(vertices=vertices,faces=geometry["faces"],process=False)


def export_glb(geometry:dict)->bytes:
    validate_geometry(geometry)
    if geometry["kind"]!="mesh":
        raise UnsupportedFeatureError("GLB mesh export requires triangles")
    mesh=_export_mesh(geometry)
    return mesh.export(file_type="glb")

def export_glb_scene(scene:Scene)->bytes:
    """Export triangle meshes and parent transforms; refuse unsupported semantics."""
    scene.validate()
    if scene.families or scene.metadata:
        raise UnsupportedFeatureError("GLB cannot preserve parametric families or scene semantic metadata")
    trimesh=_trimesh()
    import numpy as np
    from ..common.units import convert_length
    scale=convert_length(1,scene.unit,"m")
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
            local[:3,3] *= scale
            if geometry is not None:
                mesh=_export_mesh(geometry,scale)
                target.add_geometry(mesh,node_name=oid,geom_name=oid,parent_node_name=existing.get(parent),
                                    transform=local)
            else:
                target.graph.update(frame_to=oid,frame_from=existing.get(parent,target.graph.base_frame),
                                    matrix=local)
            existing[oid]=oid
            del pending[oid];progressed=True
        if not progressed: raise ValidationError("invalid scene hierarchy")
    return target.export(file_type="glb")
