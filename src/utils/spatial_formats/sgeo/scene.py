"""SGEO document scene graph: one parent per object, acyclic, stable object IDs.

Since 0.2.0 a scene also owns a material registry (``materials``), an image registry (``images``, records only)
and the encoded image bytes (``image_data``: image id -> PNG/JPEG bytes). The bytes never appear in
:meth:`Scene.as_dict`; the container stores them as ``textures/`` members."""
from __future__ import annotations
from copy import deepcopy
from itertools import product
from threading import RLock
from ..common.errors import ReferenceError
from ..common import ValidationError, ResourceLimitError, new_id, require_id, assert_unique, require_unit, require_metadata
from ..common.transforms import IDENTITY, compose, require_transform, transform_point
from ..container.versions import CURRENT_VERSION, schema_id
from .geometry import validate_geometry, local_bounds
from .images import texture_path
from .materials import validate_registries, verify_image_bytes

MAX_OBJECTS = 100_000
MAX_FAMILIES = 10_000

class Scene:
    def __init__(self, name="Untitled", *, unit="m", document_id=None, revision=1,
                 objects=None, metadata=None, document_kind="model", families=None, materials=None, images=None,
                 image_data=None):
        self._mutation_lock=RLock()
        self.name=name
        self.unit=require_unit(unit)
        self.document_id=document_id or new_id()
        self.revision=revision
        self.objects=deepcopy(objects) if objects is not None else []
        self.metadata=deepcopy(metadata) if metadata is not None else {}
        self.document_kind=document_kind
        self.families=deepcopy(families) if families is not None else []
        self.materials=deepcopy(materials) if materials is not None else []
        self.images=deepcopy(images) if images is not None else []
        self.image_data=dict(image_data) if image_data else {}
        self.validate()

    @staticmethod
    def _check_object(obj):
        """Validate everything about one object record that needs no other object."""
        if not isinstance(obj,dict):
            raise ValidationError("scene object must be an object")
        if set(obj) - {"id","name","parent","transform","geometry","metadata","family_id","parameters","material"}:
            raise ValidationError("unknown scene object field")
        if "material" in obj:
            require_id(obj["material"])
            if obj.get("geometry") is None:
                raise ValidationError("only an object with geometry can have a material")
        if not isinstance(obj.get("name",""),str) or len(obj.get("name",""))>255:
            raise ValidationError("invalid object name")
        require_transform(obj.get("transform",IDENTITY))
        if obj.get("geometry") is not None:
            validate_geometry(obj["geometry"])
        require_metadata(obj.get("metadata",{}))
        if obj.get("parent") is not None:
            require_id(obj["parent"])

    def _detached_copy(self):
        """Deep copy for transactional candidates; the caller validates before commit."""
        clone=object.__new__(type(self))
        clone._mutation_lock=RLock()
        clone.name=self.name
        clone.unit=self.unit
        clone.document_id=self.document_id
        clone.revision=self.revision
        clone.objects=deepcopy(self.objects)
        clone.metadata=deepcopy(self.metadata)
        clone.document_kind=self.document_kind
        clone.families=deepcopy(self.families)
        clone.materials=deepcopy(self.materials)
        clone.images=deepcopy(self.images)
        clone.image_data=dict(self.image_data)
        return clone

    def validate(self):
        require_id(self.document_id)
        if type(self.revision) is not int or self.revision<1:
            raise ValidationError("scene revision must be positive")
        if not isinstance(self.name,str) or len(self.name)>255:
            raise ValidationError("invalid scene name")
        if self.document_kind not in ("model","family"):
            raise ValidationError("invalid SGEO document kind")
        require_metadata(self.metadata)
        if not isinstance(self.objects,list) or len(self.objects)>MAX_OBJECTS:
            raise ValidationError("invalid scene object count")
        assert_unique(o.get("id") if isinstance(o,dict) else None for o in self.objects)
        index={o["id"]:o for o in self.objects}
        for obj in self.objects:
            self._check_object(obj)
            parent=obj.get("parent")
            if parent is not None and (parent not in index or parent==obj["id"]):
                raise ReferenceError("invalid scene parent reference")
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
        validate_registries(self.materials,self.images,image_data=self.image_data or None)
        material_ids={m["id"] for m in self.materials}
        for obj in self.objects:
            if "material" in obj and obj["material"] not in material_ids:
                raise ReferenceError("object references an unknown material",context={"material":obj["material"]})
        if not isinstance(self.families,list) or len(self.families)>MAX_FAMILIES:
            raise ValidationError("families must be a list")
        from .families import validate_family, generate_geometry
        assert_unique(f.get("id") if isinstance(f,dict) else None for f in self.families)
        fam_index={f["id"]:f for f in self.families}
        for family in self.families:
            validate_family(family)
        for obj in self.objects:
            if "family_id" in obj:
                require_id(obj["family_id"])
                if obj["family_id"] not in fam_index:
                    raise ReferenceError("unknown family reference")
                expected=generate_geometry(fam_index[obj["family_id"]],obj.get("parameters",{}))
                if obj.get("geometry") != expected:
                    raise ValidationError("family instance geometry does not match its parameters")
            elif "parameters" in obj:
                raise ValidationError("parameters require a family reference")
        from .animation import ANIMATION_KEY, ANIMATION_V2_KEY, validate_animation, validate_animation_v2
        if ANIMATION_KEY in self.metadata and ANIMATION_V2_KEY in self.metadata:
            raise ValidationError("scene cannot contain both v1 and v2 animation")
        if ANIMATION_KEY in self.metadata:
            validate_animation(self.metadata[ANIMATION_KEY], self)
        if ANIMATION_V2_KEY in self.metadata:
            validate_animation_v2(self.metadata[ANIMATION_V2_KEY], self)
        return self

    def set_animation(self, animation: dict | None, *, version: int | None = None) -> int:
        """Atomically replace animation, preserving the active version by default.

        Explicit version=2 upgrades v1 data only when the caller chooses it.
        Removing animation clears both supported versions. Validation precedes
        committing the detached candidate.
        """
        from .animation import (ANIMATION_KEY, ANIMATION_V2_KEY, validate_animation,
                                validate_animation_v2)
        if version is not None and (type(version) is not int or version not in (1, 2)):
            raise ValidationError("animation version must be 1 or 2")
        with self._mutation_lock:
            candidate = self._detached_copy()
            active_v2 = ANIMATION_V2_KEY in candidate.metadata
            chosen = (2 if active_v2 else 1) if version is None else version
            candidate.metadata.pop(ANIMATION_KEY, None)
            candidate.metadata.pop(ANIMATION_V2_KEY, None)
            if animation is not None:
                if chosen == 2:
                    candidate.metadata[ANIMATION_V2_KEY] = validate_animation_v2(animation, candidate)
                else:
                    candidate.metadata[ANIMATION_KEY] = validate_animation(animation, candidate)
            candidate.revision += 1
            candidate.validate()
            self.metadata = candidate.metadata
            self.revision = candidate.revision
            return self.revision

    def upgrade_animation(self, *, expected_revision: int | None = None) -> int:
        """Explicitly migrate valid legacy v1 animation to equivalent native v2.

        V1 schema is a strict subset of v2. Migration preserves keyframe data
        exactly; no implicit upgrade is performed by loading or evaluation.
        """
        from .animation import ANIMATION_KEY, ANIMATION_V2_KEY
        from ..common.errors import ConflictError
        with self._mutation_lock:
            if expected_revision is not None and (
                    type(expected_revision) is not int or expected_revision < 1):
                raise ValidationError("expected_revision must be positive integer")
            if expected_revision is not None and self.revision != expected_revision:
                raise ConflictError("scene revision does not match expected revision")
            if ANIMATION_V2_KEY in self.metadata:
                raise ValidationError("scene animation is already v2")
            spec = self.metadata.get(ANIMATION_KEY)
            if spec is None:
                raise ValidationError("scene has no legacy v1 animation to migrate")
            return self.set_animation(deepcopy(spec), version=2)

    def evaluate(self, time: float):
        """Evaluate read-only motion state without changing this scene."""
        from .animation import evaluate_scene
        return evaluate_scene(self, time)

    def forward_kinematics(self, values: dict | None = None):
        """Compute a detached articulated pose from native v2 joint defaults."""
        from .kinematics import forward_kinematics
        from .animation import ANIMATION_V2_KEY
        with self._mutation_lock:
            self.validate()
            spec = self.metadata.get(ANIMATION_V2_KEY)
            if spec is None:
                raise ValidationError("forward kinematics requires native v2 animation")
            candidate = self._detached_copy()
        return forward_kinematics(candidate, spec.get("joints", []), values)

    def solve_inverse_kinematics(self, end_effector, target, *,
                                 expected_revision=None, max_iterations=200,
                                 tolerance=1e-5):
        """Return validated joint proposals without editing authoritative state."""
        from .kinematics import solve_inverse_kinematics
        from .animation import ANIMATION_V2_KEY
        if expected_revision is not None and (
                type(expected_revision) is not int or expected_revision < 1):
            raise ValidationError("expected_revision must be a positive integer")
        with self._mutation_lock:
            if expected_revision is not None and self.revision != expected_revision:
                from ..common.errors import ConflictError
                raise ConflictError("scene revision does not match expected revision")
            self.validate()
            spec = self.metadata.get(ANIMATION_V2_KEY)
            if spec is None:
                raise ValidationError("inverse kinematics requires native v2 animation")
            candidate = self._detached_copy()
        result = solve_inverse_kinematics(
            candidate, spec.get("joints", []), end_effector, target,
            max_iterations=max_iterations, tolerance=tolerance,
        )
        return {"revision": candidate.revision, **result}

    def apply_inverse_kinematics(self, end_effector, target, *,
                                 expected_revision=None, max_iterations=200,
                                 tolerance=1e-5):
        """Solve and commit a supported static joint pose atomically.

        This intentionally refuses authored tracks/drivers/constraints so an
        IK proposal cannot be committed when the evaluated pose differs from
        its verified unanimated kinematic baseline. Animation-track authoring
        remains the responsibility of the existing animation editing API.
        """
        from .animation import ANIMATION_V2_KEY
        from ..common.errors import ConflictError
        solution = self.solve_inverse_kinematics(
            end_effector, target, expected_revision=expected_revision,
            max_iterations=max_iterations, tolerance=tolerance,
        )
        if solution["status"] != "solved":
            return {**solution, "committed": False}
        with self._mutation_lock:
            if self.revision != solution["revision"]:
                raise ConflictError("scene changed during IK solve; commit rejected")
            spec = deepcopy(self.metadata.get(ANIMATION_V2_KEY))
            if spec is None:
                raise ValidationError("scene has no v2 animation")
            if spec.get("tracks") or spec.get("drivers") or spec.get("constraints") or spec.get("instances"):
                raise ValidationError("IK pose commit requires an unanimated, unconstrained joint assembly")
            proposals = solution["joint_values"]
            for joint in spec.get("joints", []):
                if joint["object_id"] in proposals:
                    joint["value"] = proposals[joint["object_id"]]
            committed_revision = self.set_animation(spec, version=2)
        return {**solution, "revision": committed_revision, "committed": True}

    def add_object(self, *, name="", geometry=None, parent=None, transform=None, metadata=None, object_id=None,
                   material=None):
        """Commit one individually validated leaf object under the scene lock.

        A new object has no children and no family reference, so only the new
        record and its parent reference need checking; the full-scene audit in
        :meth:`validate` stays reserved for load/save paths. This keeps building
        a scene object by object linear rather than quadratic.
        """
        with self._mutation_lock:
            if len(self.objects) >= MAX_OBJECTS:
                raise ResourceLimitError("invalid scene object count")
            oid=new_id() if object_id is None else require_id(object_id)
            if object_id is not None and any(o.get("id")==oid for o in self.objects):
                raise ValidationError("duplicate object identifier")
            obj={"id":oid,"name":name,"parent":parent,
                 "transform":list(transform if transform is not None else IDENTITY),
                 "geometry":deepcopy(geometry),"metadata":deepcopy(metadata or {})}
            if not isinstance(name,str) or len(name)>255:
                raise ValidationError("invalid object name")
            require_transform(obj["transform"])
            if obj["geometry"] is not None:
                validate_geometry(obj["geometry"])
            require_metadata(obj["metadata"])
            if parent is not None:
                require_id(parent)
                if not any(o.get("id")==parent for o in self.objects):
                    raise ReferenceError("invalid scene parent reference")
            if material is not None:
                require_id(material)
                if obj["geometry"] is None:
                    raise ValidationError("only an object with geometry can have a material")
                if not any(m.get("id")==material for m in self.materials):
                    raise ReferenceError("object references an unknown material",context={"material":material})
                obj["material"]=material
            self.objects.append(obj)
            self.revision+=1
            return obj["id"]

    def remove_object(self, object_id):
        """Remove an unreferenced object without racing another scene mutation."""
        with self._mutation_lock:
            require_id(object_id)
            if any(obj.get("parent")==object_id for obj in self.objects):
                raise ValidationError("cannot remove scene object with children")
            for key in ("slai.animation.v1", "slai.animation.v2"):
                spec = self.metadata.get(key, {})
                if any(track["object_id"] == object_id for track in spec.get("tracks", [])):
                    raise ValidationError("cannot remove object referenced by animation")
                if key == "slai.animation.v2":
                    if any(object_id in (d["source_object_id"], d["target_object_id"])
                           for d in spec.get("drivers", [])):
                        raise ValidationError("cannot remove object referenced by driver")
                    if any(object_id in (c["object_id"], c.get("other_id"))
                           for c in spec.get("constraints", [])):
                        raise ValidationError("cannot remove object referenced by constraint")
                    if any(j["object_id"] == object_id for j in spec.get("joints", [])):
                        raise ValidationError("cannot remove object referenced by joint")
                    if any(i["object_id"] == object_id for i in spec.get("instances", [])):
                        raise ValidationError("cannot remove object referenced by animation clip")
            pos=next((i for i,obj in enumerate(self.objects) if obj["id"]==object_id),None)
            if pos is None:
                raise ValidationError("object ID not found")
            self.objects.pop(pos)
            self.revision+=1

    def world_transforms(self):
        """Return detached world matrices, computing each parent chain once."""
        with self._mutation_lock:
            self.validate()
            nodes={obj["id"]:obj for obj in self.objects}
            matrices={}
            for oid in nodes:
                chain=[]
                cursor=oid
                while cursor is not None and cursor not in matrices:
                    chain.append(cursor)
                    cursor=nodes[cursor].get("parent")
                matrix=matrices.get(cursor,IDENTITY)
                for key in reversed(chain):
                    matrix=compose(matrix,nodes[key].get("transform",IDENTITY))
                    matrices[key]=matrix
            return matrices

    def world_transform(self, object_id):
        require_id(object_id)
        matrices=self.world_transforms()
        if object_id not in matrices:
            raise ValidationError("unknown object identifier")
        return matrices[object_id]

    def bounds_by_object(self):
        """Snapshot broad-phase world AABBs for all geometric objects."""
        with self._mutation_lock:
            matrices=self.world_transforms()
            bounds={}
            for obj in self.objects:
                if obj.get("geometry") is None:
                    continue
                minimum,maximum=local_bounds(obj["geometry"])
                corners=[transform_point(matrices[obj["id"]],p) for p in product(*zip(minimum,maximum))]
                bounds[obj["id"]]=(tuple(min(p[i] for p in corners) for i in range(3)),
                                    tuple(max(p[i] for p in corners) for i in range(3)))
            return bounds

    def world_bounds(self, object_id):
        require_id(object_id)
        bounds=self.bounds_by_object()
        if object_id not in bounds:
            raise ValidationError("object has no geometric representation")
        return bounds[object_id]

    def as_dict(self):
        with self._mutation_lock:
            self.validate()
            return deepcopy({"document_id":self.document_id,"revision":self.revision,"name":self.name,
                             "objects":self.objects,"families":self.families,"metadata":self.metadata,
                             "materials":self.materials,"images":self.images})

    def manifest(self):
        with self._mutation_lock:
            self.validate()
            return {"format_id":"slai.sgeo","format_version":CURRENT_VERSION,"schema":schema_id("slai.sgeo",CURRENT_VERSION),
                    "document_kind":self.document_kind,"spatial_dimension":3,"document_id":self.document_id,
                    "revision":self.revision,"units":self.unit,"axis_up":"+Z","handedness":"right",
                    "requires":self._requires()}

    def _requires(self):
        kinds={o["geometry"]["kind"] for o in self.objects if o.get("geometry") is not None}
        requires=["scene-graph","analytic-primitives","triangle-mesh"]
        if self.families:
            requires.append("families")
        if kinds & {"nurbs_curve","nurbs_surface"}:
            requires.append("nurbs")
        if "brep" in kinds:
            requires.append("brep")
        if self.materials or any("material" in o for o in self.objects):
            requires.append("materials")
        if self.images:
            requires.append("textures")
        if "slai.animation.v1" in self.metadata:
            requires.append("animation-v1")
        if "slai.animation.v2" in self.metadata:
            requires.append("animation-v2")
        return requires

    def texture_bytes(self, image_id):
        """The encoded bytes (PNG or JPEG) of a registered image."""
        require_id(image_id)
        with self._mutation_lock:
            if image_id not in self.image_data:
                raise ReferenceError("no embedded bytes for this image",context={"image":image_id})
            return self.image_data[image_id]

    @classmethod
    def from_package(cls,package):
        from ..verification import validate_document
        validate_document(package.manifest,package.document)
        if package.manifest["format_id"]!="slai.sgeo":
            raise ValidationError("package does not contain SGEO scene")
        image_data={}
        for record in package.document.get("images",[]):
            raw=package.assets.get(texture_path(record["sha256"],record["mime"]))
            if raw is None:
                raise ValidationError("package is missing the bytes of an image",context={"image":record["id"]})
            verify_image_bytes(record,raw)
            image_data[record["id"]]=raw
        return cls(**package.document,unit=package.manifest["units"],
                   document_kind=package.manifest["document_kind"],image_data=image_data)
