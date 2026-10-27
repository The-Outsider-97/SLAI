"""Binary SGEO scene transport with transparent validated mesh hydration.

The archive stores mesh array descriptors in scene.json and content-addressed
buffers in geometry/. In-memory Scene objects always use ordinary indexed meshes.
Embedded texture images are stored verbatim as ``textures/sha256-<hex>.png|jpg``; the image records in
scene.json gain a ``path`` there (stripped again on load) and every image is re-verified against its record.
"""
from __future__ import annotations
from copy import deepcopy
from ..common.errors import ValidationError, IntegrityError, ResourceLimitError
from .buffers import encode_mesh_buffers, decode_mesh_buffers, validate_descriptor

_MESH_REFS=({"vertices","faces"},{"vertices","faces","uvs"})

def externalize_scene(document: dict, textures: dict | None = None) -> tuple[dict, dict[str,bytes]]:
    """Return a detached serializable document and deduplicated binary assets.

    ``textures`` maps image id -> encoded bytes and must cover exactly the document's image registry.
    """
    if not isinstance(document,dict):
        raise ValidationError("scene document must be an object")
    result=deepcopy(document)
    assets={}
    images=result.get("images",[])
    textures={} if textures is None else textures
    if not isinstance(images,list) or not isinstance(textures,dict):
        raise ValidationError("images must be a list and textures a mapping")
    from ..sgeo.images import texture_path
    from ..sgeo.materials import verify_image_bytes
    ids={record.get("id") if isinstance(record,dict) else None for record in images}
    if set(textures)!=ids:
        raise ValidationError("texture bytes must be supplied for exactly the registered images",
                              context={"missing":sorted(map(str,ids-set(textures)))[:10],
                                       "unknown":sorted(map(str,set(textures)-ids))[:10]})
    for record in images:
        raw=textures[record["id"]]
        if not isinstance(raw,bytes):
            raise ValidationError("texture data must be bytes")
        verify_image_bytes(record,raw)
        path=texture_path(record["sha256"],record["mime"])
        record["path"]=path
        assets[path]=raw
    for obj in result.get("objects",[]):
        if not isinstance(obj,dict):
            raise ValidationError("scene object must be an object")
        geometry=obj.get("geometry")
        if isinstance(geometry,dict) and geometry.get("kind")=="mesh":
            descriptors,payloads=encode_mesh_buffers(geometry)
            for path,raw in payloads.items():
                if path in assets and assets[path]!=raw:
                    raise IntegrityError("content-addressed asset collision")
                assets[path]=raw
            obj["geometry"]={"kind":"mesh_buffer","buffers":descriptors}
    return result,assets

def hydrate_scene(document: dict, assets: dict[str,bytes]) -> dict:
    """Reconstruct in-memory mesh objects from verified, local buffer references."""
    if not isinstance(document,dict) or not isinstance(assets,dict):
        raise ValidationError("scene and assets must be mappings")
    validate_buffer_references(document, assets)
    result=deepcopy(document)
    for record in result.get("images",[]):
        record.pop("path",None)
    decoded={}
    for obj in result.get("objects",[]):
        if not isinstance(obj,dict):
            raise ValidationError("scene object must be an object")
        geom=obj.get("geometry")
        if isinstance(geom,dict) and geom.get("kind")=="mesh_buffer":
            if set(geom)!={"kind","buffers"}:
                raise ValidationError("unknown binary geometry fields")
            from ..common.helpers import canonical_json
            key=canonical_json(geom["buffers"])
            if key not in decoded:
                decoded[key]=decode_mesh_buffers(geom["buffers"],assets)
            # Keep independently editable object geometry while avoiding
            # redundant binary buffer decoding for repeated mesh instances.
            obj["geometry"]=deepcopy(decoded[key])
    return result

def validate_buffer_references(document:dict, assets:dict[str,bytes])->None:
    """Reject orphaned/undeclared assets and invalid geometry references."""
    from ..common.helpers import MAX_DOCUMENT_NODES, json_safe
    if not isinstance(document,dict) or not isinstance(document.get("objects"),list) or not isinstance(assets,dict):
        raise ValidationError("scene objects must be a list and assets a mapping")
    json_safe(document, max_nodes=MAX_DOCUMENT_NODES)
    expected=set()
    expanded=0
    for obj in document["objects"]:
        if not isinstance(obj,dict):
            raise ValidationError("scene object must be an object")
        geometry=obj.get("geometry")
        if isinstance(geometry,dict) and geometry.get("kind")=="mesh_buffer":
            refs=geometry.get("buffers")
            if not isinstance(refs,dict) or set(refs) not in _MESH_REFS:
                raise ValidationError("invalid mesh buffer references")
            for descriptor in refs.values():
                desc=validate_descriptor(descriptor)
                expected.add(desc.path)
                # Each row and each scalar becomes a JSON node on hydration.
                from math import prod
                expanded += prod(desc.shape) + desc.shape[0]
                if expanded > MAX_DOCUMENT_NODES:
                    raise ResourceLimitError("hydrated scene exceeds structural budget")
    images=document.get("images",[])
    if not isinstance(images,list):
        raise ValidationError("images must be a list")
    from ..sgeo.images import texture_path
    from ..sgeo.materials import verify_image_bytes
    for record in images:
        if not isinstance(record,dict) or not isinstance(record.get("path"),str):
            raise ValidationError("every stored image needs a texture path")
        bare={k:v for k,v in record.items() if k!="path"}
        from ..sgeo.materials import validate_image_record
        validate_image_record(bare)
        if record["path"]!=texture_path(record["sha256"],record["mime"]):
            raise ValidationError("content-addressed texture path mismatch")
        raw=assets.get(record["path"])
        if not isinstance(raw,bytes):
            raise IntegrityError("missing referenced texture resource")
        verify_image_bytes(bare,raw)
        expected.add(record["path"])
    if set(assets)!=expected:
        raise IntegrityError("unreferenced or missing scene binary asset")
