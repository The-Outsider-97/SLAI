"""Transactional SGEO scene editing with optimistic revision checks.

Edits are applied to a detached candidate and validated before committing.
All changes are rolled back on failure; no caller callback runs on live state.
"""
from __future__ import annotations
from copy import deepcopy
from ..common.errors import ConflictError, ValidationError
from ..common.ids import new_id, require_id
from ..common.transforms import IDENTITY
from .geometry import validate_geometry
from .families import generate_geometry

def apply_scene_operations(scene, operations: list[dict], *, expected_revision: int | None = None):
    """Serialize optimistic edits against the Scene's private mutation lock."""
    from .scene import Scene
    if not isinstance(scene,Scene):
        raise ValidationError("expected an SGEO Scene")
    with scene._mutation_lock:
        return _apply_scene_operations_locked(scene,operations,expected_revision=expected_revision)


def _apply_scene_operations_locked(scene, operations: list[dict], *, expected_revision: int | None = None):
    from .scene import Scene
    if not isinstance(scene, Scene) or not isinstance(operations,list) or not 1 <= len(operations) <= 1000:
        raise ValidationError("operations must be a bounded nonempty list for a Scene")
    scene.validate()
    if expected_revision is not None and scene.revision != expected_revision:
        raise ConflictError("scene revision does not match expected revision")
    candidate=Scene(**scene.as_dict(),unit=scene.unit,document_kind=scene.document_kind)
    results=[]
    for op in operations:
        if not isinstance(op,dict) or op.get("op") not in {
            "add", "remove", "rename", "transform", "geometry", "reparent", "parameters"
        }:
            raise ValidationError("unknown scene mutation operation")
        kind=op["op"]
        if kind=="add":
            if set(op)-{"op","name","parent","geometry","transform","metadata"}:
                raise ValidationError("unexpected add arguments")
            oid=candidate.add_object(name=op.get("name",""),parent=op.get("parent"),
                                    geometry=op.get("geometry"),transform=op.get("transform"),
                                    metadata=op.get("metadata"))
            results.append(oid)
            continue
        if set(op)-{"op","id","value"}:
            raise ValidationError("unexpected scene operation arguments")
        oid=require_id(op.get("id"))
        matches=[x for x in candidate.objects if x["id"]==oid]
        if len(matches)!=1:
            raise ValidationError("unknown scene object")
        obj=matches[0]
        value=op.get("value")
        if kind=="remove":
            if any(other.get("parent")==oid for other in candidate.objects):
                raise ValidationError("cannot remove object with children")
            candidate.objects.remove(obj)
        elif kind=="rename":
            if not isinstance(value,str) or len(value)>255:
                raise ValidationError("invalid new name")
            obj["name"]=value
        elif kind=="transform":
            from ..common.transforms import require_transform
            obj["transform"]=list(require_transform(value))
        elif kind=="geometry":
            if value is not None:
                validate_geometry(value)
            obj["geometry"]=deepcopy(value)
        elif kind=="reparent":
            if value is not None:
                require_id(value)
            obj["parent"]=value
        elif kind=="parameters":
            if not isinstance(value,dict) or "family_id" not in obj:
                raise ValidationError("parameters require an instantiated family")
            family=next((x for x in candidate.families if x["id"]==obj["family_id"]),None)
            if family is None:
                raise ValidationError("family definition missing")
            derived=generate_geometry(family,value)
            obj["parameters"]=deepcopy(value)
            obj["geometry"]=derived
        candidate.validate()
        results.append(oid)
    # Treat a batch as one document revision, never one revision per step.
    candidate.revision=scene.revision+1
    candidate.validate()
    scene.objects=deepcopy(candidate.objects)
    scene.families=deepcopy(candidate.families)
    scene.metadata=deepcopy(candidate.metadata)
    scene.revision=candidate.revision
    return results

def objects_within_bounds(scene, minimum, maximum):
    """Broad-phase AABB overlap only; not an exact solids-intersection predicate."""
    from ..common.helpers import vector
    lo,hi=vector(minimum,3),vector(maximum,3)
    if any(a>b for a,b in zip(lo,hi)):
        raise ValidationError("invalid query bounds")
    scene.validate()
    matches=[]
    for obj in scene.objects:
        if obj.get("geometry") is None:
            continue
        a,b=scene.world_bounds(obj["id"])
        if all(a[i]<=hi[i] and b[i]>=lo[i] for i in range(3)):
            matches.append(obj["id"])
    return matches
