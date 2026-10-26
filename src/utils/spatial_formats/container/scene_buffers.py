"""Binary SGEO scene transport with transparent validated mesh hydration.

The archive stores mesh array descriptors in scene.json and content-addressed
buffers in geometry/. In-memory Scene objects always use ordinary indexed meshes.
"""
from __future__ import annotations
from copy import deepcopy
from ..common.errors import ValidationError, IntegrityError
from .buffers import encode_mesh_buffers, decode_mesh_buffers, validate_descriptor

def externalize_scene(document: dict) -> tuple[dict, dict[str,bytes]]:
    """Return a detached serializable document and deduplicated binary assets."""
    if not isinstance(document,dict):
        raise ValidationError("scene document must be an object")
    result=deepcopy(document)
    assets={}
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
    result=deepcopy(document)
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
    if not isinstance(document,dict) or not isinstance(assets,dict):
        raise ValidationError("scene document and assets must be mappings")
    objects=document.get("objects",[])
    if not isinstance(objects,list):
        raise ValidationError("scene objects must be an array")
    expected=set()
    for obj in objects:
        if not isinstance(obj,dict):
            raise ValidationError("scene object must be a mapping")
        geometry=obj.get("geometry")
        if isinstance(geometry,dict) and geometry.get("kind")=="mesh_buffer":
            refs=geometry.get("buffers")
            if not isinstance(refs,dict) or set(refs)!={"vertices","faces"}:
                raise ValidationError("invalid mesh buffer references")
            for descriptor in refs.values():
                expected.add(validate_descriptor(descriptor).path)
    if set(assets)!=expected:
        raise IntegrityError("unreferenced or missing scene binary asset")
