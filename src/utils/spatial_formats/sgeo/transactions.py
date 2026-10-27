"""Transactional SGEO scene editing with optimistic revision checks.

Edits are applied to a detached candidate and validated before committing.
All changes are rolled back on failure; no caller callback runs on live state.

Besides object and family edits, a batch may change the material and image registries
(``add_material``, ``update_material``, ``remove_material``, ``add_image``, ``remove_image``) and assign a
material (``set_material``; ``add`` and ``instantiate`` also take ``material``). Removing a material that an
object uses, or an image that a material samples, is refused.
"""
from __future__ import annotations
from copy import deepcopy
from ..common.errors import ConflictError, ValidationError, ResourceLimitError, ReferenceError
from ..common.ids import new_id, require_id
from ..common.metadata import require_metadata
from ..common.transforms import IDENTITY, require_transform
from .geometry import validate_geometry
from .families import generate_geometry
from .materials import MAX_MATERIALS, describe_image, validate_material
from .images import MAX_IMAGES

def apply_scene_operations(scene, operations: list[dict], *, expected_revision: int | None = None):
    """Serialize optimistic edits against the Scene's private mutation lock."""
    from .scene import Scene
    if not isinstance(scene,Scene):
        raise ValidationError("expected an SGEO Scene")
    with scene._mutation_lock:
        return _apply_scene_operations_locked(scene,operations,expected_revision=expected_revision)


def _new_object_id(op, objects):
    """Caller-chosen ids make batches idempotent and let one batch reference its own new objects."""
    if "id" not in op or op["id"] is None:
        return new_id()
    oid=require_id(op["id"])
    if oid in objects:
        raise ValidationError("duplicate object identifier", context={"id": oid})
    return oid


def _apply_scene_operations_locked(scene, operations: list[dict], *, expected_revision: int | None = None):
    from .scene import Scene, MAX_OBJECTS, MAX_FAMILIES
    if not isinstance(scene, Scene) or not isinstance(operations,list) or not 1 <= len(operations) <= 1000:
        raise ValidationError("operations must be a bounded nonempty list for a Scene")
    scene.validate()
    if expected_revision is not None and scene.revision != expected_revision:
        raise ConflictError("scene revision does not match expected revision")
    candidate=scene._detached_copy()
    # Each step is checked against local indexes (O(1)); the cross-object
    # invariants (cycles, family/geometry agreement, id uniqueness) are audited
    # once by the full validate() before the candidate replaces live state.
    # Insertion-ordered dicts preserve document order for the final list.
    objects={o["id"]:o for o in candidate.objects}
    families={f["id"]:f for f in candidate.families}
    materials={m["id"]:m for m in candidate.materials}
    images={i["id"]:i for i in candidate.images}
    image_data=candidate.image_data
    children={}
    instances={}
    for o in objects.values():
        if o.get("parent") is not None:
            children[o["parent"]]=children.get(o["parent"],0)+1
        if "family_id" in o:
            instances[o["family_id"]]=instances.get(o["family_id"],0)+1
    def link(obj, sign):
        parent=obj.get("parent")
        if parent is not None:
            children[parent]=children.get(parent,0)+sign
        if "family_id" in obj:
            instances[obj["family_id"]]=instances.get(obj["family_id"],0)+sign
    def check_parent(parent, oid=None):
        if parent is None:
            return
        require_id(parent)
        if parent not in objects or parent==oid:
            raise ReferenceError("invalid scene parent reference")
    results=[]
    for op in operations:
        if not isinstance(op,dict) or not isinstance(op.get("op"),str) or op.get("op") not in {
            "add", "remove", "rename", "transform", "geometry", "reparent", "parameters", "metadata",
            "add_family", "update_family", "remove_family", "instantiate",
            "add_material", "update_material", "remove_material", "set_material", "add_image", "remove_image"
        }:
            raise ValidationError("unknown scene mutation operation")
        kind=op["op"]
        if kind in {"add_family","update_family","remove_family"}:
            if set(op)-{"op","id","value"}:
                raise ValidationError("unexpected family operation arguments")
            from .families import validate_family
            if kind=="add_family":
                family=deepcopy(op.get("value"))
                validate_family(family)
                if family["id"] in families:
                    raise ValidationError("duplicate family identifier")
                if len(families)>=MAX_FAMILIES:
                    raise ResourceLimitError("too many families")
                families[family["id"]]=family
                fid=family["id"]
            else:
                fid=require_id(op.get("id"))
                family=families.get(fid)
                if family is None:
                    raise ReferenceError("unknown family identifier")
                if kind=="remove_family":
                    if instances.get(fid,0):
                        raise ValidationError("cannot remove an instantiated family")
                    del families[fid]
                else:
                    replacement=deepcopy(op.get("value"))
                    if not isinstance(replacement,dict) or replacement.get("id")!=fid:
                        raise ValidationError("replacement must preserve family identifier")
                    validate_family(replacement)
                    families[fid]=replacement
                    if instances.get(fid,0):
                        for obj in objects.values():
                            if obj.get("family_id")==fid:
                                obj["geometry"]=generate_geometry(replacement,obj.get("parameters",{}))
            results.append(fid)
            continue
        if kind in {"add_material","update_material","remove_material","add_image","remove_image"}:
            results.append(_registry_operation(kind,op,materials,images,image_data,objects))
            continue
        if kind=="set_material":
            if set(op)-{"op","id","value"}:
                raise ValidationError("unexpected scene operation arguments")
            oid=require_id(op.get("id"))
            obj=objects.get(oid)
            if obj is None:
                raise ReferenceError("unknown scene object")
            value=op.get("value")
            if value is None:
                obj.pop("material",None)
            else:
                if require_id(value) not in materials:
                    raise ReferenceError("unknown material identifier")
                if obj.get("geometry") is None:
                    raise ValidationError("only an object with geometry can have a material")
                obj["material"]=value
            results.append(oid)
            continue
        if kind=="instantiate":
            if set(op)-{"op","id","family_id","parameters","name","parent","transform","metadata","material"}:
                raise ValidationError("unexpected instantiate arguments")
            fid=require_id(op.get("family_id"))
            family=families.get(fid)
            if family is None:
                raise ReferenceError("unknown family identifier")
            if len(objects)>=MAX_OBJECTS:
                raise ResourceLimitError("invalid scene object count")
            parameters=deepcopy(op.get("parameters",{}))
            obj={"id":_new_object_id(op,objects),"family_id":fid,"parameters":parameters,
                 "name":op.get("name",family["name"]),"parent":op.get("parent"),
                 "transform":deepcopy(op.get("transform",list(IDENTITY))),
                 "geometry":generate_geometry(family,parameters),"metadata":deepcopy(op.get("metadata",{}))}
            _assign_material(obj,op.get("material"),materials)
            Scene._check_object(obj)
            check_parent(obj["parent"])
            objects[obj["id"]]=obj
            link(obj,1)
            results.append(obj["id"])
            continue
        if kind=="add":
            if set(op)-{"op","id","name","parent","geometry","transform","metadata","material"}:
                raise ValidationError("unexpected add arguments")
            if len(objects)>=MAX_OBJECTS:
                raise ResourceLimitError("invalid scene object count")
            obj={"id":_new_object_id(op,objects),"name":op.get("name",""),"parent":op.get("parent"),
                 "transform":list(op["transform"] if op.get("transform") is not None else IDENTITY),
                 "geometry":deepcopy(op.get("geometry")),"metadata":deepcopy(op.get("metadata") or {})}
            _assign_material(obj,op.get("material"),materials)
            Scene._check_object(obj)
            check_parent(obj["parent"])
            objects[obj["id"]]=obj
            link(obj,1)
            results.append(obj["id"])
            continue
        if set(op)-{"op","id","value"}:
            raise ValidationError("unexpected scene operation arguments")
        oid=require_id(op.get("id"))
        obj=objects.get(oid)
        if obj is None:
            raise ReferenceError("unknown scene object")
        value=op.get("value")
        if kind=="remove":
            if children.get(oid,0):
                raise ValidationError("cannot remove object with children")
            link(obj,-1)
            del objects[oid]
        elif kind=="rename":
            if not isinstance(value,str) or len(value)>255:
                raise ValidationError("invalid new name")
            obj["name"]=value
        elif kind=="transform":
            obj["transform"]=list(require_transform(value))
        elif kind=="geometry":
            if value is not None:
                validate_geometry(value)
            obj["geometry"]=deepcopy(value)
        elif kind=="reparent":
            check_parent(value,oid)
            if obj.get("parent") is not None:
                children[obj["parent"]]-=1
            obj["parent"]=value
            if value is not None:
                children[value]=children.get(value,0)+1
        elif kind=="metadata":
            obj["metadata"]=require_metadata(value)
        elif kind=="parameters":
            if not isinstance(value,dict) or "family_id" not in obj:
                raise ValidationError("parameters require an instantiated family")
            family=families.get(obj["family_id"])
            if family is None:
                raise ValidationError("family definition missing")
            derived=generate_geometry(family,value)
            obj["parameters"]=deepcopy(value)
            obj["geometry"]=derived
        results.append(oid)
    # Treat a batch as one document revision, never one revision per step.
    candidate.objects=list(objects.values())
    candidate.families=list(families.values())
    candidate.materials=list(materials.values())
    candidate.images=list(images.values())
    candidate.revision=scene.revision+1
    candidate.validate()
    scene.objects=candidate.objects
    scene.families=candidate.families
    scene.materials=candidate.materials
    scene.images=candidate.images
    scene.image_data=candidate.image_data
    scene.metadata=candidate.metadata
    scene.revision=candidate.revision
    return results

def _assign_material(obj, material, materials):
    if material is None:
        return
    if require_id(material) not in materials:
        raise ReferenceError("unknown material identifier")
    obj["material"]=material


def _registry_operation(kind, op, materials, images, image_data, objects):
    """Material and image registry edits on the candidate's indexes; returns the affected id."""
    if kind=="add_image":
        if set(op)-{"op","id","name","data"}:
            raise ValidationError("unexpected add_image arguments")
        data=op.get("data")
        if not isinstance(data,(bytes,bytearray)):
            raise ValidationError("add_image needs the encoded image bytes as 'data'")
        raw=bytes(data)
        facts=describe_image(raw)
        if "id" not in op or op["id"] is None:
            for record in images.values():
                if record["sha256"]==facts["sha256"] and record["mime"]==facts["mime"]:
                    return record["id"]              # identical content is stored once
            ident=new_id()
        else:
            ident=require_id(op["id"])
            if ident in images:
                raise ValidationError("duplicate image identifier",context={"id":ident})
        if len(images)>=MAX_IMAGES:
            raise ResourceLimitError("too many images")
        name=op.get("name","")
        if not isinstance(name,str) or len(name)>255:
            raise ValidationError("invalid image name")
        images[ident]={"id":ident,"name":name,**facts}
        image_data[ident]=raw
        return ident
    if kind=="remove_image":
        if set(op)-{"op","id"}:
            raise ValidationError("unexpected remove_image arguments")
        ident=require_id(op.get("id"))
        if ident not in images:
            raise ReferenceError("unknown image identifier")
        for material in materials.values():
            for slot,spec in material.get("textures",{}).items():
                if spec.get("image")==ident:
                    raise ValidationError("image is used by a material texture",
                                          context={"material":material["id"],"slot":slot})
        del images[ident]
        image_data.pop(ident,None)
        return ident
    if kind=="remove_material":
        if set(op)-{"op","id"}:
            raise ValidationError("unexpected remove_material arguments")
        ident=require_id(op.get("id"))
        if ident not in materials:
            raise ReferenceError("unknown material identifier")
        if any(o.get("material")==ident for o in objects.values()):
            raise ValidationError("material is assigned to an object",context={"id":ident})
        del materials[ident]
        return ident
    if set(op)-{"op","id","value"}:
        raise ValidationError("unexpected %s arguments"%kind)
    value=deepcopy(op.get("value"))
    if not isinstance(value,dict):
        raise ValidationError("%s needs a material record as 'value'"%kind)
    if kind=="add_material":
        if "id" not in value:
            value["id"]=op["id"] if op.get("id") is not None else new_id()
        validate_material(value,image_ids=images)
        if value["id"] in materials:
            raise ValidationError("duplicate material identifier",context={"id":value["id"]})
        if len(materials)>=MAX_MATERIALS:
            raise ResourceLimitError("too many materials")
        materials[value["id"]]=value
        return value["id"]
    ident=require_id(op.get("id"))                  # update_material: replace the whole record
    if ident not in materials:
        raise ReferenceError("unknown material identifier")
    if value.get("id",ident)!=ident:
        raise ValidationError("replacement must preserve the material identifier")
    value["id"]=ident
    validate_material(value,image_ids=images)
    materials[ident]=value
    return ident


def objects_within_bounds(scene, minimum, maximum):
    """Broad-phase AABB overlap only; not an exact solids-intersection predicate."""
    from ..common.helpers import vector
    lo,hi=vector(minimum,3),vector(maximum,3)
    if any(a>b for a,b in zip(lo,hi)):
        raise ValidationError("invalid query bounds")
    matches=[]
    for oid,(a,b) in scene.bounds_by_object().items():
        if all(a[i]<=hi[i] and b[i]>=lo[i] for i in range(3)):
            matches.append(oid)
    return matches
