"""SGEO document scene graph: one parent per object, acyclic, stable object IDs."""
from __future__ import annotations
from copy import deepcopy
from itertools import product
from ..common import ValidationError, new_id, require_id, assert_unique, require_unit, require_metadata
from ..common.transforms import IDENTITY, compose, require_transform, transform_point
from .geometry import validate_geometry, local_bounds

class Scene:
    def __init__(self, name="Untitled", *, unit="m", document_id=None, revision=1,
                 objects=None, metadata=None, document_kind="model", families=None):
        self.name=name
        self.unit=require_unit(unit)
        self.document_id=document_id or new_id()
        self.revision=revision
        self.objects=deepcopy(objects) if objects is not None else []
        self.metadata=deepcopy(metadata) if metadata is not None else {}
        self.document_kind=document_kind
        self.families=deepcopy(families) if families is not None else []
        self.validate()

    def validate(self):
        require_id(self.document_id)
        if type(self.revision) is not int or self.revision<1:
            raise ValidationError("scene revision must be positive")
        if not isinstance(self.name,str) or len(self.name)>255:
            raise ValidationError("invalid scene name")
        if self.document_kind not in ("model","family"):
            raise ValidationError("invalid SGEO document kind")
        require_metadata(self.metadata)
        if not isinstance(self.objects,list) or len(self.objects)>100_000:
            raise ValidationError("invalid scene object count")
        assert_unique(o.get("id") if isinstance(o,dict) else None for o in self.objects)
        index={o["id"]:o for o in self.objects}
        for obj in self.objects:
            if set(obj) - {"id","name","parent","transform","geometry","metadata","family_id","parameters"}:
                raise ValidationError("unknown scene object field")
            if not isinstance(obj.get("name",""),str) or len(obj.get("name",""))>255:
                raise ValidationError("invalid object name")
            require_transform(obj.get("transform",IDENTITY))
            if obj.get("geometry") is not None:
                validate_geometry(obj["geometry"])
            require_metadata(obj.get("metadata",{}))
            parent=obj.get("parent")
            if parent is not None and (parent not in index or parent==obj["id"]):
                raise ValidationError("invalid scene parent reference")
        # Iterative three-colour DFS avoids recursion and detects every cycle.
        state={}
        for oid in index:
            cursor=oid
            chain=[]
            while cursor is not None and state.get(cursor) is None:
                state[cursor]=1
                chain.append(cursor)
                cursor=index[cursor].get("parent")
            if cursor is not None and state.get(cursor)==1:
                raise ValidationError("scene graph contains a parent cycle")
            for key in chain:
                state[key]=2
        if not isinstance(self.families,list) or len(self.families)>10_000:
            raise ValidationError("families must be a list")
        from .families import validate_family, resolve_parameters
        assert_unique(f.get("id") if isinstance(f,dict) else None for f in self.families)
        fam_index={f["id"]:f for f in self.families}
        for family in self.families:
            validate_family(family)
        for obj in self.objects:
            if "family_id" in obj:
                if obj["family_id"] not in fam_index:
                    raise ValidationError("unknown family reference")
                resolve_parameters(fam_index[obj["family_id"]],obj.get("parameters",{}))
            elif "parameters" in obj:
                raise ValidationError("parameters require a family reference")
        return self

    def add_object(self, *, name="", geometry=None, parent=None, transform=None, metadata=None):
        obj={"id":new_id(),"name":name,"parent":parent,
             "transform":list(transform if transform is not None else IDENTITY),
             "geometry":deepcopy(geometry),"metadata":deepcopy(metadata or {})}
        self.objects.append(obj)
        try:
            self.validate()
        except Exception:
            self.objects.pop()
            raise
        self.revision+=1
        return obj["id"]

    def remove_object(self, object_id):
        require_id(object_id)
        if any(obj.get("parent")==object_id for obj in self.objects):
            raise ValidationError("cannot remove scene object with children")
        pos=next((i for i,obj in enumerate(self.objects) if obj["id"]==object_id),None)
        if pos is None:
            raise ValidationError("object ID not found")
        self.objects.pop(pos)
        self.revision+=1

    def world_transform(self, object_id):
        self.validate()
        nodes={obj["id"]:obj for obj in self.objects}
        if object_id not in nodes:
            raise ValidationError("unknown object identifier")
        chain=[]
        cursor=object_id
        while cursor is not None:
            node=nodes[cursor]
            chain.append(node)
            cursor=node.get("parent")
        matrix=IDENTITY
        for node in reversed(chain):
            matrix=compose(matrix,node.get("transform",IDENTITY))
        return matrix

    def world_bounds(self, object_id):
        nodes={obj["id"]:obj for obj in self.objects}
        if object_id not in nodes or nodes[object_id].get("geometry") is None:
            raise ValidationError("object has no geometric representation")
        minimum,maximum=local_bounds(nodes[object_id]["geometry"])
        matrix=self.world_transform(object_id)
        corners=[transform_point(matrix,p) for p in product(*zip(minimum,maximum))]
        return (tuple(min(p[i] for p in corners) for i in range(3)),
                tuple(max(p[i] for p in corners) for i in range(3)))

    def as_dict(self):
        self.validate()
        return deepcopy({"document_id":self.document_id,"revision":self.revision,"name":self.name,
                         "objects":self.objects,"families":self.families,"metadata":self.metadata})

    def manifest(self):
        self.validate()
        return {"format_id":"slai.sgeo","format_version":"0.1.0","schema":"slai.sgeo/0.1",
                "document_kind":self.document_kind,"spatial_dimension":3,"document_id":self.document_id,
                "revision":self.revision,"units":self.unit,"axis_up":"+Z","handedness":"right",
                "requires":["scene-graph","analytic-primitives","triangle-mesh"] + (["families"] if self.families else [])}

    @classmethod
    def from_package(cls,package):
        from ..verification import validate_document
        validate_document(package.manifest,package.document)
        if package.manifest["format_id"]!="slai.sgeo":
            raise ValidationError("package does not contain SGEO scene")
        return cls(**package.document,unit=package.manifest["units"],
                   document_kind=package.manifest["document_kind"])
