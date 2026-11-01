"""Editable SGEO document: the object behind ``src.utils.sgeo.SGEODocument``.

Wraps one validated :class:`Scene` with a task-oriented API: typed primitive
constructors, a transform hierarchy, reusable parametric families, atomic edits
(every call is exactly one revision), queries, measurements, unit conversion and
loss-reported interchange.

Conventions: right-handed, +Z up, angles in radians unless ``degrees=`` says
otherwise, flat row-major 4x4 transforms, lengths in the document unit. Box
extents are ``width`` (X), ``depth`` (Y), ``height`` (Z); a cylinder's axis is
local +Z.
"""
from __future__ import annotations

from copy import deepcopy
from math import pi, radians

from ..common.errors import ReferenceError, ResourceLimitError, UnsupportedFeatureError, ValidationError
from ..common.helpers import finite, vector
from ..common.ids import is_valid_id, new_id, require_id
from ..common.transforms import (IDENTITY, compose, determinant, from_axis_angle, from_translation,
                                 invert_transform, is_rigid, is_similarity, require_transform, scale_factors)
from ..common.units import convert_area, convert_length, convert_volume, require_unit
from ..documents import SpatialDocument, _checked_metadata
from ..verification.report import ValidationReport, check_scene
from .analysis import analyze_mesh
from .families import generate_geometry, validate_family
from . import brep_build, nurbs
from .brep import scale_brep
from .brep_mesh import analyze_brep
from .geometry import mesh_surface_area, validate_geometry
from .materials import DEFAULT_MATERIAL, TEXTURE_SLOTS, resolve_material
from .meshing import extract_meshes, merge_meshes
from .scene import Scene
from .solids import extrude, revolve
from .tessellation import tessellate
from .transactions import apply_scene_operations
from .box_boolean import boolean_boxes
from .box_offset import offset_box_faces

__all__ = ["SceneDocument", "MAX_QUERY_LIMIT"]

MAX_QUERY_LIMIT = 10_000
_UNSET = ...
_OBJECT_OPTIONS = frozenset({"name", "parent", "transform", "position", "metadata", "object_id", "material"})
_MATERIAL_FIELDS = ("base_color", "metallic", "roughness", "emissive", "alpha_mode", "alpha_cutoff", "double_sided")
MAX_IMAGE_FILE_BYTES = 32 * 1024 * 1024


def _p3(value, label="point") -> list:
    return [float(v) for v in vector(value, 3, label)]


def _angle(value, label, degrees):
    number = finite(value, label, limit=1e8)
    return radians(number) if degrees else number


def _length_spec(value, label):
    """A bare number becomes a plain length parameter; a dict is a full parameter spec."""
    if isinstance(value, dict):
        return deepcopy(value)
    return {"type": "length", "default": finite(value, label)}


def _scale_geometry(geometry: dict, f: float) -> dict:
    g = deepcopy(geometry)
    kind = g["kind"]
    if kind == "point":
        g["position"] = [c * f for c in g["position"]]
    elif kind == "segment":
        g["start"], g["end"] = [c * f for c in g["start"]], [c * f for c in g["end"]]
    elif kind == "box":
        g["center"], g["size"] = [c * f for c in g["center"]], [c * f for c in g["size"]]
    elif kind in ("sphere", "cylinder"):
        g["center"] = [c * f for c in g["center"]]
        g["radius"] *= f
        if kind == "cylinder":
            g["height"] *= f
    elif kind == "nurbs_curve":
        g = nurbs.scale_curve(g, f)
    elif kind == "nurbs_surface":
        g = nurbs.scale_surface(g, f)
    elif kind == "brep":
        g = scale_brep(g, f)
    else:
        g["vertices"] = [[c * f for c in v] for v in g["vertices"]]      # uvs are dimensionless
    return validate_geometry(g)


class SceneDocument(SpatialDocument):
    """A validated 3D scene graph with an atomic, queryable editing API."""

    FORMAT_ID = "slai.sgeo"
    FILE_SUFFIX = ".sgeo"
    KIND_LABEL = "scene"

    def __init__(self, scene: Scene | None = None, *, name: str = "Untitled", unit: str = "m",
                 document_kind: str = "model", path=None, source_sha256=None, read_only: bool = False, source_version=None):
        if scene is None:
            scene = Scene(name, unit=unit, document_kind=document_kind)
        elif not isinstance(scene, Scene):
            raise ValidationError("expected a Scene model")
        super().__init__(scene, path=path, source_sha256=source_sha256, read_only=read_only, source_version=source_version)

    # -- construction ---------------------------------------------------------
    @classmethod
    def new(cls, name: str = "Untitled", *, unit: str = "m", document_kind: str = "model", metadata=None,
            document_id=None) -> "SceneDocument":
        """Create an empty, writable, never-saved scene (``document_kind``: ``model`` or ``family``)."""
        return cls(Scene(name, unit=unit, document_kind=document_kind, metadata=metadata,
                         document_id=document_id))

    @classmethod
    def from_scene(cls, scene: Scene, *, copy: bool = True) -> "SceneDocument":
        if not isinstance(scene, Scene):
            raise ValidationError("expected a Scene model")
        with scene._mutation_lock:
            scene.validate()
            return cls(scene._detached_copy() if copy else scene)

    @classmethod
    def _model_from_package(cls, package):
        return Scene.from_package(package)

    # -- hooks ----------------------------------------------------------------------
    def _check(self, mode: str, max_issues: int) -> ValidationReport:
        return check_scene(self._model, mode=mode, max_issues=max_issues)

    def _apply(self, operations, expected_revision):
        return apply_scene_operations(self._model, operations, expected_revision=expected_revision)

    def _counts(self) -> dict:
        with self._model._mutation_lock:
            kinds: dict = {}
            for o in self._model.objects:
                g = o.get("geometry")
                if g is not None:
                    kinds[g["kind"]] = kinds.get(g["kind"], 0) + 1
            return {"document_kind": self._model.document_kind, "object_count": len(self._model.objects),
                    "root_count": sum(1 for o in self._model.objects if o.get("parent") is None),
                    "family_count": len(self._model.families), "geometry_kinds": dict(sorted(kinds.items())),
                    "material_count": len(self._model.materials), "image_count": len(self._model.images),
                    "image_bytes": sum(len(b) for b in self._model.image_data.values()),
                    "bounds": self.bounds()}

    @property
    def document_kind(self) -> str:
        return self._model.document_kind

    def _item_metadata(self, object_id: str) -> dict:
        return deepcopy(self.get(object_id).get("metadata", {}))

    # -- object construction ------------------------------------------------------------
    def _object_args(self, options: dict) -> dict:
        unknown = set(options) - _OBJECT_OPTIONS
        if unknown:
            raise ValidationError("unknown object option(s): " + ", ".join(sorted(unknown)))
        args = {k: options[k] for k in ("name", "parent", "metadata", "material") if k in options}
        if options.get("transform") is not None and options.get("position") is not None:
            raise ValidationError("give either transform or position, not both")
        if options.get("position") is not None:
            args["transform"] = list(from_translation(*_p3(options["position"], "position")))
        elif options.get("transform") is not None:
            args["transform"] = [float(v) for v in require_transform(options["transform"])]
        if options.get("object_id") is not None:
            args["object_id"] = options["object_id"]
        return args

    def add_object(self, geometry=None, **options) -> str:
        """Add an object (``geometry=None`` makes an empty group node).

        Options: name, parent, transform (16 numbers) *or* position (x, y, z),
        metadata, object_id, material (id of a registered material).
        """
        args = self._object_args(options)
        with self._edit() as model:
            return model.add_object(geometry=geometry, **args)

    def add_group(self, name: str = "", **options) -> str:
        """An empty node for organising children and sharing a transform."""
        return self.add_object(None, name=name, **options)

    def add_point(self, position, **options) -> str:
        return self.add_object({"kind": "point", "position": _p3(position)}, **options)

    def add_segment(self, start, end, **options) -> str:
        return self.add_object({"kind": "segment", "start": _p3(start, "start"), "end": _p3(end, "end")}, **options)

    def add_box(self, *, width: float | None = None, depth: float | None = None, height: float | None = None,
                size=None, center=(0.0, 0.0, 0.0), **options) -> str:
        """Axis-aligned box centred at ``center`` in its local frame.

        Give ``width`` (X), ``depth`` (Y) and ``height`` (Z), or ``size=(x, y, z)``.
        """
        if size is not None:
            if any(v is not None for v in (width, depth, height)):
                raise ValidationError("give size or width/depth/height, not both")
            dims = _p3(size, "size")
        else:
            if width is None or depth is None or height is None:
                raise ValidationError("a box needs width, depth and height (or size)")
            dims = [float(width), float(depth), float(height)]
        return self.add_object({"kind": "box", "center": _p3(center, "center"), "size": dims}, **options)

    def add_sphere(self, radius: float, *, center=(0.0, 0.0, 0.0), **options) -> str:
        return self.add_object({"kind": "sphere", "center": _p3(center, "center"), "radius": radius}, **options)

    def add_cylinder(self, radius: float, height: float, *, center=(0.0, 0.0, 0.0), **options) -> str:
        """Cylinder with its axis along local +Z, centred at ``center``."""
        return self.add_object({"kind": "cylinder", "center": _p3(center, "center"), "radius": radius,
                                "height": height}, **options)

    def add_mesh(self, vertices, faces, *, uvs=None, **options) -> str:
        """Indexed triangle mesh (counter-clockwise faces seen from outside).

        ``uvs`` (optional) gives one ``(u, v)`` per vertex; the origin is the bottom-left of the image, v up.
        """
        if not isinstance(vertices, (list, tuple)) or not isinstance(faces, (list, tuple)):
            raise ValidationError("vertices and faces must be sequences")
        geometry = {"kind": "mesh", "vertices": [_p3(v, "vertex") for v in vertices],
                    "faces": [[int(i) if isinstance(i, int) and not isinstance(i, bool) else i
                               for i in f] for f in faces]}
        if uvs is not None:
            if not isinstance(uvs, (list, tuple)):
                raise ValidationError("uvs must be a sequence of (u, v) pairs")
            geometry["uvs"] = [[finite(c, "uv", limit=1e6) for c in vector(uv, 2, "uv")] for uv in uvs]
        return self.add_object(geometry, **options)

    def boolean_boxes(self, first_id: str, second_id: str, operation: str, *, name: str = "") -> str | None:
        """Apply a rectilinear Boolean to two *world-axis-aligned* SGEO boxes.

        The resulting mesh stores the exact boundary of the box set operation.
        This is deliberately not a general CAD/B-rep Boolean implementation.
        An empty intersection/difference returns ``None`` without an edit.
        """
        self._writable()
        with self._model._mutation_lock:
            objects = {obj["id"]: obj for obj in self._model.objects}
            first_id, second_id = require_id(first_id), require_id(second_id)
            try:
                a, b = objects[first_id], objects[second_id]
            except KeyError as exc:
                raise ReferenceError("Boolean operand object does not exist") from exc
            transforms = self._model.world_transforms()

            def world_box(obj):
                geometry = obj.get("geometry")
                if not isinstance(geometry, dict) or geometry.get("kind") != "box":
                    raise UnsupportedFeatureError("rectilinear Booleans require box operands")
                matrix = transforms[obj["id"]]
                if any(abs(matrix[index] - expected) > 1e-12 for index, expected in
                       ((0, 1), (1, 0), (2, 0), (4, 0), (5, 1), (6, 0),
                        (8, 0), (9, 0), (10, 1))):
                    raise UnsupportedFeatureError("rectilinear Booleans require translation-only world transforms")
                return {"kind": "box", "size": deepcopy(geometry["size"]),
                        "center": [geometry["center"][i] + matrix[4*i+3] for i in range(3)]}

            result = boolean_boxes(world_box(a), world_box(b), operation)
            if result is None:
                return None
            # Only commit after all geometry calculations and validation succeed.
            return self.add_object(result, name=name,
                                   metadata={"slai.geometry:boolean": {"operation": operation,
                                                                       "operands": [first_id, second_id],
                                                                       "representation": "rectilinear-mesh"}})

    def set_animation(self, animation: dict | None, *, version: int | None = None) -> int:
        """Persist native animation; explicit version=2 opts into the extended schema."""
        with self._edit() as model:
            return model.set_animation(animation, version=version)

    def offset_box_faces(self, object_id: str, distance: float, *, name: str = "") -> str:
        """Create a mitered six-face box offset as a separate SGEO object.

        This operation is valid for arbitrary rigidly oriented boxes; non-rigid
        inherited transforms are refused so distance remains in document units.
        """
        self._writable()
        with self._model._mutation_lock:
            obj = self.get(object_id)
            if not is_rigid(self._model.world_transform(object_id)):
                raise UnsupportedFeatureError("face offset requires a rigid world transformation")
            result = offset_box_faces(obj["geometry"], distance)
            return self.add_object(result, name=name, parent=obj.get("parent"),
                                   transform=obj["transform"],
                                   metadata={"slai.geometry:face_offset": {"source": object_id,
                                                                            "distance": float(distance),
                                                                            "join": "miter"}})

    def upgrade_animation(self, *, expected_revision=None) -> int:
        """Explicitly migrate an existing v1 animation to native v2."""
        with self._edit() as model:
            return model.upgrade_animation(expected_revision=expected_revision)

    def animation(self) -> dict | None:
        """Get a detached animation definition, not a live reference."""
        from .animation import ANIMATION_KEY, ANIMATION_V2_KEY
        metadata = self._model.metadata
        return deepcopy(metadata.get(ANIMATION_V2_KEY, metadata.get(ANIMATION_KEY)))

    def evaluate(self, time: float):
        """Return the detached evaluated SGEO scene at time in seconds."""
        self._live()
        return self._model.evaluate(time)

    def undo_animation(self, *, expected_revision=None) -> int:
        """Undo one SGEO animation revision without reverting unrelated state."""
        with self._edit() as model:
            return model.undo_animation(expected_revision=expected_revision)

    def redo_animation(self, *, expected_revision=None) -> int:
        """Redo one SGEO animation revision as an atomic new scene revision."""
        with self._edit() as model:
            return model.redo_animation(expected_revision=expected_revision)

    def generate_joint_motion(self, end_effector, target, duration, *,
                              expected_revision=None, max_iterations=200,
                              tolerance=1e-5):
        """Propose a verified native joint animation without modifying this document."""
        self._live()
        return self._model.generate_joint_motion(
            end_effector, target, duration, expected_revision=expected_revision,
            max_iterations=max_iterations, tolerance=tolerance,
        )

    def retarget_joint_clip(self, source_scene, source_clip_id, joint_map, *,
                            translation_scales=None):
        """Propose an explicit joint-channel clip conversion without committing it."""
        self._live()
        source = source_scene._model if isinstance(source_scene, SceneDocument) else source_scene
        return self._model.retarget_joint_clip(
            source, source_clip_id, joint_map, translation_scales=translation_scales,
        )

    def animation_events(self, start, end, *, include_start: bool = False):
        """Query native v2 animation markers without executing callbacks."""
        self._live()
        return self._model.animation_events(start, end, include_start=include_start)

    def forward_kinematics(self, values: dict | None = None):
        """Return a detached native articulated world-transform mapping."""
        self._live()
        return self._model.forward_kinematics(values)

    def solve_inverse_kinematics(self, end_effector, target, *,
                                 expected_revision=None, max_iterations=200,
                                 tolerance=1e-5):
        """Return a non-mutating native SGEO IK solution and diagnostics."""
        self._live()
        return self._model.solve_inverse_kinematics(
            end_effector, target, expected_revision=expected_revision,
            max_iterations=max_iterations, tolerance=tolerance,
        )

    def apply_inverse_kinematics(self, end_effector, target, *,
                                 expected_revision=None, max_iterations=200,
                                 tolerance=1e-5):
        """Commit a validated static IK pose through existing writable SGEO facade."""
        self._writable()
        # Do not hold the document edit lock across a bounded nonlinear solve.
        # Scene snapshots the source and rechecks its revision before commit.
        return self._model.apply_inverse_kinematics(
            end_effector, target, expected_revision=expected_revision,
            max_iterations=max_iterations, tolerance=tolerance,
        )

    # -- NURBS and B-rep -----------------------------------------------------------------------------
    def add_nurbs_curve(self, control_points, *, degree: int = 3, knots=None, weights=None, **options) -> str:
        """A (rational) B-spline curve. Without ``knots`` a clamped uniform vector is used."""
        if not isinstance(control_points, (list, tuple)):
            raise ValidationError("control_points must be a sequence of 3D points")
        points = [_p3(p, "control point") for p in control_points]
        geometry = {"kind": "nurbs_curve", "degree": degree, "control_points": points,
                    "knots": [float(k) for k in knots] if knots is not None
                    else nurbs.clamped_uniform_knots(len(points), degree)}
        if weights is not None:
            geometry["weights"] = [float(w) for w in weights]
        return self.add_object(geometry, **options)

    def add_nurbs_surface(self, control_points, *, degree_u: int = 3, degree_v: int = 3, knots_u=None, knots_v=None,
                          weights=None, **options) -> str:
        """A (rational) B-spline surface; ``control_points[i][j]`` runs along u then v.

        Its natural normal is dP/du x dP/dv; it is an open sheet (a face without volume).
        """
        if not isinstance(control_points, (list, tuple)) or not control_points or \
                not all(isinstance(r, (list, tuple)) for r in control_points):
            raise ValidationError("control_points must be a grid (list of rows) of 3D points")
        net = [[_p3(p, "control point") for p in row] for row in control_points]
        geometry = {"kind": "nurbs_surface", "degree_u": degree_u, "degree_v": degree_v, "control_points": net,
                    "knots_u": [float(k) for k in knots_u] if knots_u is not None
                    else nurbs.clamped_uniform_knots(len(net), degree_u),
                    "knots_v": [float(k) for k in knots_v] if knots_v is not None
                    else nurbs.clamped_uniform_knots(len(net[0]), degree_v)}
        if weights is not None:
            geometry["weights"] = [[float(w) for w in row] for row in weights]
        return self.add_object(geometry, **options)

    def add_brep(self, brep: dict, **options) -> str:
        """A boundary representation (see :mod:`.brep`); it is validated structurally and geometrically."""
        if not isinstance(brep, dict):
            raise ValidationError("a brep must be an object")
        return self.add_object({**deepcopy(brep), "kind": "brep"}, **options)

    def add_brep_box(self, *, width: float | None = None, depth: float | None = None, height: float | None = None,
                     size=None, center=(0.0, 0.0, 0.0), **options) -> str:
        """An exact six-face B-rep box (same arguments as :meth:`add_box`)."""
        if size is not None:
            dims = _p3(size, "size")
        elif width is None or depth is None or height is None:
            raise ValidationError("a box needs width, depth and height (or size)")
        else:
            dims = [float(width), float(depth), float(height)]
        return self.add_brep(brep_build.brep_box(dims, _p3(center, "center")), **options)

    def add_brep_cylinder(self, radius: float, height: float, *, center=(0.0, 0.0, 0.0), radius_top=None,
                          **options) -> str:
        """An exact B-rep cylinder (axis +Z, centred); a ``radius_top`` makes it a truncated cone."""
        return self.add_brep(brep_build.brep_cylinder(radius, height, _p3(center, "center"), radius_top), **options)

    def add_brep_sphere(self, radius: float, *, center=(0.0, 0.0, 0.0), **options) -> str:
        return self.add_brep(brep_build.brep_sphere(radius, _p3(center, "center")), **options)

    def add_brep_torus(self, major_radius: float, minor_radius: float, *, center=(0.0, 0.0, 0.0), **options) -> str:
        return self.add_brep(brep_build.brep_torus(major_radius, minor_radius, _p3(center, "center")), **options)

    def add_brep_extrusion(self, profile, depth: float, *, holes=None, **options) -> str:
        """Exact prism of a simple (possibly concave) XY profile with optional holes, extruded along +Z."""
        return self.add_brep(brep_build.brep_extrusion(profile, depth, holes), **options)

    def add_brep_sheet(self, control_points, *, degree_u: int = 3, degree_v: int = 3, knots_u=None, knots_v=None,
                       weights=None, **options) -> str:
        """An open one-face B-rep trimmed to a NURBS surface's boundary (a sheet, not a solid)."""
        surface = {"degree_u": degree_u, "degree_v": degree_v, "weights": weights,
                   "control_points": [[_p3(p, "control point") for p in row] for row in control_points]}
        surface["knots_u"] = [float(k) for k in knots_u] if knots_u is not None \
            else nurbs.clamped_uniform_knots(len(surface["control_points"]), degree_u)
        surface["knots_v"] = [float(k) for k in knots_v] if knots_v is not None \
            else nurbs.clamped_uniform_knots(len(surface["control_points"][0]), degree_v)
        if weights is None:
            del surface["weights"]
        return self.add_brep(brep_build.brep_from_nurbs_surface(surface), **options)

    def add_extrusion(self, profile, depth: float, *, direction=(0, 0, 1), **options) -> str:
        """Extrude a simple convex XY profile into a closed prism mesh (tessellated, not exact B-rep)."""
        return self.add_object(extrude(profile, depth, direction=direction), **options)

    def add_revolution(self, profile, *, sweep: float = 2 * pi, segments: int = 64, **options) -> str:
        """Revolve a convex radius/height profile about +Z into a closed mesh."""
        return self.add_object(revolve(profile, sweep=sweep, segments=segments), **options)

    # -- editing ------------------------------------------------------------------------------
    def remove(self, object_id: str) -> str:
        """Remove a childless object (children must be removed or reparented first)."""
        return self.apply([{"op": "remove", "id": require_id(object_id)}])[0]

    def remove_tree(self, object_id: str) -> list:
        """Remove an object and all its descendants in one revision; returns removed ids."""
        ids = self.descendants(object_id)[::-1] + [require_id(object_id)]
        return self.apply([{"op": "remove", "id": i} for i in ids])

    def rename_object(self, object_id: str, name: str) -> str:
        return self.apply([{"op": "rename", "id": require_id(object_id), "value": name}])[0]

    def set_geometry(self, object_id: str, geometry) -> str:
        """Replace an object's geometry (``None`` clears it). Family instances are derived: edit parameters."""
        return self.apply([{"op": "geometry", "id": require_id(object_id), "value": geometry}])[0]

    def set_transform(self, object_id: str, transform) -> str:
        """Set the local transform (relative to the parent)."""
        m = [float(v) for v in require_transform(transform)]
        return self.apply([{"op": "transform", "id": require_id(object_id), "value": m}])[0]

    def move(self, object_id: str, dx: float, dy: float, dz: float) -> str:
        """Translate in the *parent's* frame (adds to the current translation)."""
        delta = from_translation(finite(dx, "dx"), finite(dy, "dy"), finite(dz, "dz"))
        new = compose(delta, self.get(object_id)["transform"])
        return self.set_transform(object_id, new)

    def rotate(self, object_id: str, axis, angle: float, *, degrees: bool = False) -> str:
        """Rotate about the object's own origin; the axis is expressed in the parent's frame."""
        old = [float(v) for v in require_transform(self.get(object_id)["transform"])]
        rot = from_axis_angle(_p3(axis, "axis"), _angle(angle, "angle", degrees))
        new = list(compose(rot, old))
        new[3], new[7], new[11] = old[3], old[7], old[11]  # keep the origin where it is
        return self.set_transform(object_id, new)

    def reparent(self, object_id: str, parent: str | None, *, keep_world: bool = False) -> str:
        """Move an object under ``parent`` (``None`` = a root).

        By default the *local* transform is kept (the object moves with its new
        parent); ``keep_world=True`` recomputes it so the object stays put in
        world space.
        """
        oid = require_id(object_id)
        operations = []
        if keep_world:
            matrices = self._model.world_transforms()
            if oid not in matrices:
                raise ReferenceError("unknown scene object", context={"id": oid})
            if parent is not None and require_id(parent) not in matrices:
                raise ReferenceError("invalid scene parent reference", context={"parent": parent})
            local = matrices[oid] if parent is None else compose(invert_transform(matrices[parent]), matrices[oid])
            operations.append({"op": "transform", "id": oid, "value": [float(v) for v in local]})
        operations.append({"op": "reparent", "id": oid, "value": parent})
        return self.apply(operations)[-1]

    def attach_semantics(self, object_id: str, values: dict, *, replace: bool = False,
                         namespace: str | None = None) -> dict:
        """Merge (or replace) semantic metadata on an object; returns the resulting metadata.

        Keys are ``namespace:name`` or ``slai.*``; ``namespace="aec"`` prefixes
        bare keys (``{"category": "wall"}`` is stored as ``aec:category``).
        """
        from ..sdrw.document import _namespaced
        safe = _checked_metadata(_namespaced(values, namespace))
        merged = safe if replace else {**self._item_metadata(object_id), **safe}
        self.apply([{"op": "metadata", "id": require_id(object_id), "value": merged}])
        return merged

    def duplicate(self, object_id: str, *, with_children: bool = True, name_suffix: str = " copy") -> dict:
        """Copy an object (and, by default, its subtree) under the same parent; returns ``{old_id: new_id}``.

        Geometry, transforms, metadata and family instances are copied; the new
        ids are allocated up front so the whole copy is one revision.
        """
        root = require_id(object_id)
        with self._edit() as model:
            index = {o["id"]: o for o in model.objects}
            if root not in index:
                raise ReferenceError("unknown scene object", context={"id": root})
            order = [root]
            if with_children:
                order += self._descendants_locked(root)
            mapping = {old: new_id() for old in order}
            operations = []
            for old in order:
                src = index[old]
                parent = src.get("parent") if old == root else mapping[src["parent"]]
                name = src.get("name", "") + (name_suffix if old == root else "")
                if "family_id" in src:
                    operations.append({"op": "instantiate", "id": mapping[old], "family_id": src["family_id"],
                                       "parameters": deepcopy(src.get("parameters", {})), "name": name,
                                       "parent": parent, "transform": list(src.get("transform", IDENTITY)),
                                       "metadata": deepcopy(src.get("metadata", {})), "material": src.get("material")})
                else:
                    operations.append({"op": "add", "id": mapping[old], "name": name, "parent": parent,
                                       "geometry": deepcopy(src.get("geometry")),
                                       "transform": list(src.get("transform", IDENTITY)),
                                       "metadata": deepcopy(src.get("metadata", {})), "material": src.get("material")})
            apply_scene_operations(model, operations)
            return mapping

    # -- materials and images -------------------------------------------------------------------------------
    def materials(self) -> list:
        """Stored material records (detached) with ``use_count`` (objects using each)."""
        with self._model._mutation_lock:
            use: dict = {}
            for o in self._model.objects:
                if "material" in o:
                    use[o["material"]] = use.get(o["material"], 0) + 1
            return [{**deepcopy(m), "use_count": use.get(m["id"], 0)} for m in self._model.materials]

    def get_material(self, material_id: str, *, resolved: bool = False) -> dict:
        """One material record; ``resolved=True`` fills every default."""
        require_id(material_id)
        with self._model._mutation_lock:
            for m in self._model.materials:
                if m["id"] == material_id:
                    return resolve_material(m) if resolved else deepcopy(m)
        raise ReferenceError("unknown material", context={"id": material_id})

    def add_material(self, name: str = "", *, base_color=None, metallic=None, roughness=None, emissive=None,
                     alpha_mode=None, alpha_cutoff=None, double_sided=None, textures=None, metadata=None,
                     material_id: str | None = None) -> str:
        """Register a PBR (metallic-roughness) material and return its id.

        Colours are linear RGB(A) in [0, 1]; the stored record always has explicit values.
        ``textures`` maps a slot (``base_color``, ``metallic_roughness``, ``normal``, ``occlusion``,
        ``emissive``) to ``{"image": image_id, "wrap_u", "wrap_v", "filter", "scale" | "strength"}``.
        """
        given = {"base_color": base_color, "metallic": metallic, "roughness": roughness, "emissive": emissive,
                 "alpha_mode": alpha_mode, "alpha_cutoff": alpha_cutoff, "double_sided": double_sided}
        record = {"id": material_id or new_id(), "name": name}
        for key in _MATERIAL_FIELDS:
            value = given[key]
            if key == "alpha_cutoff":
                continue
            default = DEFAULT_MATERIAL[key]
            record[key] = deepcopy(default) if value is None else ([float(c) for c in value]
                                                                    if key in ("base_color", "emissive") else value)
        mode = record["alpha_mode"]
        if alpha_cutoff is not None:
            record["alpha_cutoff"] = alpha_cutoff
        elif mode == "mask":
            record["alpha_cutoff"] = DEFAULT_MATERIAL["alpha_cutoff"]
        record["textures"] = deepcopy(textures) if textures is not None else {}
        if metadata is not None:
            record["metadata"] = _checked_metadata(metadata)
        return self.apply([{"op": "add_material", "value": record}])[0]

    def update_material(self, material_id: str, **changes) -> str:
        """Change fields of a material in one revision (``name``, ``metadata``, ``textures`` (replaced as a
        whole) and any PBR field). Use :meth:`set_texture` / :meth:`clear_texture` to edit a single slot."""
        require_id(material_id)
        allowed = set(_MATERIAL_FIELDS) | {"name", "metadata", "textures"}
        unknown = set(changes) - allowed
        if unknown:
            raise ValidationError("unknown material field(s): " + ", ".join(sorted(unknown)))
        with self._edit() as model:
            current = next((deepcopy(m) for m in model.materials if m["id"] == material_id), None)
            if current is None:
                raise ReferenceError("unknown material", context={"id": material_id})
            for key, value in changes.items():
                current[key] = ([float(c) for c in value] if key in ("base_color", "emissive") and value is not None
                                else deepcopy(value))
            if "alpha_mode" in changes:
                if changes["alpha_mode"] != "mask" and "alpha_cutoff" not in changes:
                    current.pop("alpha_cutoff", None)
                elif changes["alpha_mode"] == "mask" and "alpha_cutoff" not in current:
                    current["alpha_cutoff"] = DEFAULT_MATERIAL["alpha_cutoff"]
            return apply_scene_operations(model, [{"op": "update_material", "id": material_id,
                                                   "value": current}])[0]

    def set_texture(self, material_id: str, slot: str, image_id: str, *, wrap_u=None, wrap_v=None, filter=None,
                    scale=None, strength=None) -> str:
        """Bind a registered image to one texture slot (``scale`` for normal, ``strength`` for occlusion)."""
        if slot not in TEXTURE_SLOTS:
            raise ValidationError("unknown texture slot; use one of: " + ", ".join(TEXTURE_SLOTS))
        spec = {"image": require_id(image_id)}
        for key, value in (("wrap_u", wrap_u), ("wrap_v", wrap_v), ("filter", filter), ("scale", scale),
                           ("strength", strength)):
            if value is not None:
                spec[key] = value
        with self._edit():
            textures = deepcopy(self.get_material(material_id).get("textures", {}))
            textures[slot] = spec
            return self.update_material(material_id, textures=textures)

    def clear_texture(self, material_id: str, slot: str) -> str:
        with self._edit():
            textures = deepcopy(self.get_material(material_id).get("textures", {}))
            if slot not in textures:
                raise ReferenceError("the material has no texture in this slot", context={"slot": slot})
            del textures[slot]
            return self.update_material(material_id, textures=textures)

    def remove_material(self, material_id: str) -> str:
        """Remove a material no object uses."""
        return self.apply([{"op": "remove_material", "id": require_id(material_id)}])[0]

    def set_material(self, object_id: str, material_id: str | None) -> str:
        """Assign a material to an object with geometry (``None`` removes the assignment)."""
        return self.apply([{"op": "set_material", "id": require_id(object_id), "value": material_id}])[0]

    def material_of(self, object_id: str, *, resolved: bool = True) -> dict | None:
        """The material assigned to an object, or ``None``."""
        material = self.get(object_id).get("material")
        return None if material is None else self.get_material(material, resolved=resolved)

    def images(self) -> list:
        """Image registry records (detached) with ``use_count`` (texture slots sampling each image)."""
        with self._model._mutation_lock:
            use: dict = {}
            for m in self._model.materials:
                for spec in m.get("textures", {}).values():
                    use[spec["image"]] = use.get(spec["image"], 0) + 1
            return [{**deepcopy(i), "use_count": use.get(i["id"], 0)} for i in self._model.images]

    def add_image(self, data, *, name: str = "", image_id: str | None = None) -> str:
        """Embed a PNG or JPEG (validated structurally, never decoded) and return its id.

        Identical bytes are stored once: adding them again returns the existing id.
        """
        operation = {"op": "add_image", "data": data, "name": name}
        if image_id is not None:
            operation["id"] = image_id
        return self.apply([operation])[0]

    def add_image_file(self, path, *, name: str | None = None) -> str:
        """Embed an image file (at most 32 MiB); the name defaults to the file name."""
        from pathlib import Path
        raw = _read_bounded(path, MAX_IMAGE_FILE_BYTES, "image")
        return self.add_image(raw, name=Path(path).name if name is None else name)

    def image_bytes(self, image_id: str) -> bytes:
        """The embedded bytes of an image exactly as stored."""
        return self._model.texture_bytes(image_id)

    def export_image(self, image_id: str, path, *, overwrite: bool = False) -> str:
        """Write an embedded image to a file atomically; returns its SHA-256."""
        from ..common.atomic import write_bytes_atomic
        raw = self.image_bytes(image_id)
        return write_bytes_atomic(path, raw, overwrite=overwrite)

    def remove_image(self, image_id: str) -> str:
        """Remove an image no material texture uses."""
        return self.apply([{"op": "remove_image", "id": require_id(image_id)}])[0]

    # -- families ---------------------------------------------------------------------------------
    def families(self) -> list:
        """Family definitions with their instance counts (detached copies)."""
        with self._model._mutation_lock:
            use: dict = {}
            for o in self._model.objects:
                if "family_id" in o:
                    use[o["family_id"]] = use.get(o["family_id"], 0) + 1
            return [{**deepcopy(f), "instance_count": use.get(f["id"], 0)} for f in self._model.families]

    def define_family(self, name: str, parameters: dict, generator: str = "builtin.box_v1", *,
                      family_id: str | None = None) -> str:
        """Register a parametric family. ``generator``: builtin.box_v1 (width, depth, height),
        builtin.sphere_v1 (radius) or builtin.cylinder_v1 (radius, height).

        ``parameters`` maps names to full parameter specs (``type``, ``default`` |
        ``expression``, ``min``, ``max``...). Only allowlisted generators and the
        safe expression language exist: nothing in a family is executable.
        """
        family = {"id": family_id or new_id(), "name": name, "parameters": deepcopy(parameters),
                  "generator": {"kind": generator}}
        validate_family(family)
        return self.apply([{"op": "add_family", "value": family}])[0]

    def define_box_family(self, name: str, *, width, depth, height, family_id: str | None = None) -> str:
        """Box family; each dimension is a number (default length) or a full parameter spec."""
        return self.define_family(name, {"width": _length_spec(width, "width"), "depth": _length_spec(depth, "depth"),
                                         "height": _length_spec(height, "height")}, "builtin.box_v1",
                                  family_id=family_id)

    def define_sphere_family(self, name: str, *, radius, family_id: str | None = None) -> str:
        return self.define_family(name, {"radius": _length_spec(radius, "radius")}, "builtin.sphere_v1",
                                  family_id=family_id)

    def define_cylinder_family(self, name: str, *, radius, height, family_id: str | None = None) -> str:
        return self.define_family(name, {"radius": _length_spec(radius, "radius"),
                                         "height": _length_spec(height, "height")}, "builtin.cylinder_v1",
                                  family_id=family_id)

    def update_family(self, family: dict) -> str:
        """Replace a family definition (same id) and regenerate every instance atomically."""
        validate_family(family)
        return self.apply([{"op": "update_family", "id": family["id"], "value": deepcopy(family)}])[0]

    def remove_family(self, family_id: str) -> str:
        """Remove a family that has no instances."""
        return self.apply([{"op": "remove_family", "id": require_id(family_id)}])[0]

    def instantiate(self, family_id: str, parameters: dict | None = None, **options) -> str:
        """Place a family instance; its geometry is generated from the (bounded) parameters."""
        args = self._object_args(options)
        operation = {"op": "instantiate", "family_id": require_id(family_id), "parameters": deepcopy(parameters or {})}
        if "object_id" in args:
            operation["id"] = args.pop("object_id")
        operation.update(args)
        return self.apply([operation])[0]

    def set_parameters(self, object_id: str, parameters: dict) -> str:
        """Replace an instance's parameter overrides and regenerate its geometry (ids are stable)."""
        return self.apply([{"op": "parameters", "id": require_id(object_id), "value": deepcopy(parameters)}])[0]

    def update_parameters(self, object_id: str, **changes) -> str:
        """Merge ``changes`` into an instance's existing overrides."""
        current = deepcopy(self.get(object_id).get("parameters", {}))
        current.update(changes)
        return self.set_parameters(object_id, current)

    def parameter_values(self, object_id: str) -> dict:
        """Fully resolved parameter values of a family instance (defaults, overrides, expressions)."""
        from .families import resolve_parameters
        obj = self.get(object_id)
        if "family_id" not in obj:
            raise ValidationError("object is not a family instance")
        family = next(f for f in self._model.families if f["id"] == obj["family_id"])
        return {k: (v.value if hasattr(v, "value") else v)
                for k, v in resolve_parameters(family, obj.get("parameters", {})).items()}

    # -- queries ------------------------------------------------------------------------------------
    def get(self, object_id: str) -> dict:
        """A detached copy of one object, or ``ReferenceError`` for an unknown id."""
        require_id(object_id)
        with self._model._mutation_lock:
            for obj in self._model.objects:
                if obj["id"] == object_id:
                    return deepcopy(obj)
        raise ReferenceError("unknown scene object", context={"id": object_id})

    def __contains__(self, object_id) -> bool:
        return isinstance(object_id, str) and is_valid_id(object_id) and any(
            o["id"] == object_id for o in self._model.objects)

    def __len__(self) -> int:
        return len(self._model.objects)

    def roots(self) -> list:
        return [o["id"] for o in self._model.objects if o.get("parent") is None]

    def children(self, object_id: str) -> list:
        require_id(object_id)
        with self._model._mutation_lock:
            if not any(o["id"] == object_id for o in self._model.objects):
                raise ReferenceError("unknown scene object", context={"id": object_id})
            return [o["id"] for o in self._model.objects if o.get("parent") == object_id]

    def _descendants_locked(self, object_id: str) -> list:
        kids: dict = {}
        for o in self._model.objects:
            if o.get("parent") is not None:
                kids.setdefault(o["parent"], []).append(o["id"])
        order, queue = [], [object_id]
        while queue:
            nxt = queue.pop(0)
            for child in kids.get(nxt, ()):
                order.append(child)
                queue.append(child)
        return order

    def descendants(self, object_id: str) -> list:
        """All descendants, breadth first (parents before children)."""
        require_id(object_id)
        with self._model._mutation_lock:
            if not any(o["id"] == object_id for o in self._model.objects):
                raise ReferenceError("unknown scene object", context={"id": object_id})
            return self._descendants_locked(object_id)

    def ancestors(self, object_id: str) -> list:
        """Parent chain from the immediate parent up to the root."""
        require_id(object_id)
        with self._model._mutation_lock:
            index = {o["id"]: o for o in self._model.objects}
            if object_id not in index:
                raise ReferenceError("unknown scene object", context={"id": object_id})
            chain, cursor = [], index[object_id].get("parent")
            while cursor is not None:
                chain.append(cursor)
                cursor = index[cursor].get("parent")
            return chain

    def query_ids(self, *, type_name: str | None = None, parent=_UNSET, name: str | None = None,
                  family_id: str | None = None, metadata: dict | None = None, within=None,
                  has_geometry: bool | None = None, material: str | None = None, limit: int = 100,
                  offset: int = 0) -> list:
        """Ids of matching objects in document order.

        ``type_name`` is the geometry kind (point, segment, box, sphere, cylinder,
        mesh, nurbs_curve, nurbs_surface, brep); ``material`` keeps objects using that material id. ``parent=None`` selects roots; omit it for no filter. ``within`` is
        a world-space AABB ``((x0, y0, z0), (x1, y1, z1))`` and keeps objects whose
        bounds overlap it (broad phase, not exact intersection).
        """
        if type(limit) is not int or not 1 <= limit <= MAX_QUERY_LIMIT:
            raise ValidationError(f"limit must be an integer in [1, {MAX_QUERY_LIMIT}]")
        if type(offset) is not int or offset < 0:
            raise ValidationError("offset must be a non-negative integer")
        if parent is not _UNSET and parent is not None:
            require_id(parent)
        if family_id is not None:
            require_id(family_id)
        if material is not None:
            require_id(material)
        if metadata is not None:
            _checked_metadata(metadata)
        box = None
        if within is not None:
            lo, hi = vector(within[0], 3, "within minimum"), vector(within[1], 3, "within maximum")
            if any(a > b for a, b in zip(lo, hi)):
                raise ValidationError("within minimum must not exceed maximum")
            box = (lo, hi)
        with self._model._mutation_lock:
            world = self._model.bounds_by_object() if box is not None else None
            ids, skipped = [], 0
            for obj in self._model.objects:
                geometry = obj.get("geometry")
                if type_name is not None and (geometry is None or geometry["kind"] != type_name):
                    continue
                if has_geometry is not None and (geometry is not None) != has_geometry:
                    continue
                if parent is not _UNSET and obj.get("parent") != parent:
                    continue
                if name is not None and obj.get("name") != name:
                    continue
                if family_id is not None and obj.get("family_id") != family_id:
                    continue
                if material is not None and obj.get("material") != material:
                    continue
                if metadata is not None:
                    have = obj.get("metadata", {})
                    if any(k not in have or have[k] != v for k, v in metadata.items()):
                        continue
                if box is not None:
                    bounds = world.get(obj["id"])
                    if bounds is None or any(bounds[1][i] < box[0][i] or bounds[0][i] > box[1][i] for i in range(3)):
                        continue
                if skipped < offset:
                    skipped += 1
                    continue
                ids.append(obj["id"])
                if len(ids) >= limit:
                    break
            return ids

    def find(self, **criteria) -> list:
        """Like :meth:`query_ids` but returns detached object copies."""
        return [self.get(i) for i in self.query_ids(**criteria)]

    def count(self, **criteria) -> int:
        criteria.setdefault("limit", MAX_QUERY_LIMIT)
        return len(self.query_ids(**criteria))

    # -- measurement ----------------------------------------------------------------------------------
    def world_transform(self, object_id: str) -> tuple:
        return self._model.world_transform(require_id(object_id))

    def world_bounds(self, object_id: str):
        """World-space AABB ``((x0, y0, z0), (x1, y1, z1))`` of one object's geometry."""
        return self._model.world_bounds(require_id(object_id))

    def bounds(self, object_ids=None):
        """Union world AABB of all (or the given) geometric objects; ``None`` if there is none."""
        with self._model._mutation_lock:
            boxes = self._model.bounds_by_object()
            if object_ids is not None:
                wanted = [require_id(i) for i in ([object_ids] if isinstance(object_ids, str) else object_ids)]
                missing = [i for i in wanted if i not in boxes]
                if missing:
                    raise ReferenceError("selection contains unknown or geometry-less objects",
                                         context={"ids": missing[:10]})
                boxes = {i: boxes[i] for i in wanted}
        if not boxes:
            return None
        values = list(boxes.values())
        return (tuple(min(b[0][i] for b in values) for i in range(3)),
                tuple(max(b[1][i] for b in values) for i in range(3)))

    def _geometry_and_world(self, object_id: str):
        obj = self.get(object_id)
        if obj.get("geometry") is None:
            raise ValidationError("object has no geometry")
        return obj["geometry"], self._model.world_transform(object_id)

    def volume(self, object_id: str, *, unit: str | None = None, resolution: int = 64) -> float:
        """World-space volume in unit³ (document unit or ``unit``).

        Analytic for box/sphere/cylinder; for meshes the mesh must be a closed
        oriented surface. A ``brep`` must be closed and outward oriented; its volume is measured on a
        tessellation at ``resolution`` segments per circle, so it is exact for flat-faced solids and
        approximate (converging as ``resolution`` grows) for curved ones.
        The world transform's determinant scales the result.
        """
        geometry, world = self._geometry_and_world(object_id)
        kind = geometry["kind"]
        if kind == "box":
            a, b, c = geometry["size"]
            local = a * b * c
        elif kind == "sphere":
            local = 4.0 / 3.0 * pi * geometry["radius"] ** 3
        elif kind == "cylinder":
            local = pi * geometry["radius"] ** 2 * geometry["height"]
        elif kind == "mesh":
            diagnostics = analyze_mesh(geometry)
            if not diagnostics.is_closed_oriented_surface:
                raise ValidationError("mesh is not a closed oriented surface; volume is undefined")
            local = abs(diagnostics.signed_volume)
        elif kind == "brep":
            facts = analyze_brep(geometry, radial_segments=resolution)
            if facts["volume"] is None:
                raise ValidationError("brep is not a closed watertight solid; volume is undefined")
            if facts["volume"] <= 0:
                raise ValidationError("brep faces point inward (negative volume); check face orientation")
            local = facts["volume"]
        else:
            raise ValidationError(f"{kind} has no volume")
        value = local * abs(determinant(world))
        return value if unit is None or unit == self.unit else convert_volume(value, self.unit, require_unit(unit))

    def surface_area(self, object_id: str, *, unit: str | None = None, resolution: int = 64) -> float:
        """World-space surface area in unit² (needs a similarity world transform: uniform scale).

        B-rep and NURBS surfaces are measured on a tessellation at ``resolution`` (approximate, converging)."""
        geometry, world = self._geometry_and_world(object_id)
        if not is_similarity(world):
            raise ValidationError("surface area under non-uniform scale or shear is not supported")
        s = scale_factors(world)[0]
        kind = geometry["kind"]
        if kind == "box":
            a, b, c = geometry["size"]
            local = 2 * (a * b + b * c + a * c)
        elif kind == "sphere":
            local = 4 * pi * geometry["radius"] ** 2
        elif kind == "cylinder":
            r, h = geometry["radius"], geometry["height"]
            local = 2 * pi * r * (r + h)
        elif kind == "mesh":
            local = mesh_surface_area(geometry)
        elif kind in ("brep", "nurbs_surface"):
            source = geometry if kind == "brep" else brep_build.brep_from_nurbs_surface(geometry)
            local = analyze_brep(source, radial_segments=resolution)["area"]
        else:
            raise ValidationError(f"{kind} has no surface")
        value = local * s * s
        return value if unit is None or unit == self.unit else convert_area(value, self.unit, require_unit(unit))

    def analyze(self, object_id: str, *, resolution: int = 64) -> dict:
        """Topology and mass diagnostics as a dict: mesh checks (watertight, components, winding...), or for a
        brep / NURBS surface its Euler characteristic, closedness and tessellation based area and volume."""
        obj = self.get(object_id)
        geometry = obj.get("geometry")
        if geometry is not None and geometry["kind"] in ("brep", "nurbs_surface"):
            source = geometry if geometry["kind"] == "brep" else brep_build.brep_from_nurbs_surface(geometry)
            return analyze_brep(source, radial_segments=resolution)
        if geometry is None or geometry["kind"] != "mesh":
            raise ValidationError("diagnostics require a triangle mesh, brep or NURBS surface object")
        return analyze_mesh(geometry).as_dict()

    def tessellate(self, object_id: str, *, radial_segments: int = 32, latitude_bands: int = 16,
                   uvs: bool = False) -> dict:
        """A triangle-mesh approximation of one object's geometry in its local frame (``uvs=True`` adds
        texture coordinates)."""
        geometry = self.get(object_id).get("geometry")
        if geometry is None:
            raise ValidationError("object has no geometry")
        return tessellate(geometry, radial_segments=radial_segments, latitude_bands=latitude_bands, uvs=uvs)

    def to_mesh(self, object_ids=None, *, radial_segments: int = 32, latitude_bands: int = 16) -> dict:
        """One merged world-space triangle mesh of the selection (default: every surface)."""
        ids = None if object_ids is None else ([object_ids] if isinstance(object_ids, str) else list(object_ids))
        extraction = extract_meshes(self._model, ids, radial_segments=radial_segments,
                                    latitude_bands=latitude_bands)
        return merge_meshes(extraction.meshes.values())

    # -- units ------------------------------------------------------------------------------------------
    def convert_unit(self, unit: str) -> float:
        """Re-express the scene in another length unit (one revision); returns the scale factor.

        Geometry and transform translations are scaled; rotations and ids are
        unchanged. Scenes with parametric families are refused because their
        parameter lengths would need dimension-aware rewriting.
        """
        target = require_unit(unit)
        with self._edit() as model:
            if target == model.unit:
                return 1.0
            if model.families:
                raise UnsupportedFeatureError("unit conversion of scenes with parametric families is not supported")
            factor = convert_length(1.0, model.unit, target)
            candidate = model._detached_copy()
            for obj in candidate.objects:
                if obj.get("geometry") is not None:
                    obj["geometry"] = _scale_geometry(obj["geometry"], factor)
                t = list(obj.get("transform", IDENTITY))
                t[3], t[7], t[11] = t[3] * factor, t[7] * factor, t[11] * factor
                obj["transform"] = t
            candidate.unit = target
            candidate.revision = model.revision + 1
            candidate.validate()
            model.objects, model.unit, model.revision = candidate.objects, target, candidate.revision
            return factor

    # -- copies ----------------------------------------------------------------------------------------------
    def copy(self, *, fresh_id: bool = True, name: str | None = None) -> "SceneDocument":
        """A writable, unsaved deep copy (fresh document id and revision 1 unless ``fresh_id=False``)."""
        snapshot = self.snapshot()
        if fresh_id:
            snapshot.document_id = new_id()
            snapshot.revision = 1
        if name is not None:
            snapshot.name = name
        return SceneDocument(snapshot.validate())

    # -- agent integration ----------------------------------------------------------------------------------
    def spatial_records(self, object_ids=None, *, unit: str | None = None) -> list:
        """JSON-safe Spatial-agent DTOs (world bounds, explicit unit, source revision)."""
        from ..adapters.slai_spatial import spatial_records
        return spatial_records(self.snapshot(), object_ids=object_ids, unit=unit)

    def reference(self, *, uri: str | None = None, object_ids=None) -> dict:
        """The small typed handle agents should pass instead of scene bytes (id, revision, digest, units)."""
        from ..adapters.slai_spatial import spatial_reference
        return spatial_reference(self.snapshot(), sha256=self._source_sha256, uri=uri, object_ids=object_ids)

    def to_agent_bundle(self, *, object_ids=None, unit: str | None = None):
        """Hydrate ``SpatialBundle`` (loads the Spatial Agent runtime on demand; metres by default)."""
        from ..adapters.slai_spatial import to_agent_bundle
        return to_agent_bundle(self.snapshot(), object_ids=object_ids, unit=unit)

    # -- interchange (see adapters.exchange for the loss-report contract) ---------------------------------------
    def to_glb(self, *, policy="best_effort", radial_segments: int = 32, latitude_bands: int = 16):
        """``(glb_bytes, ConversionReport)``: tessellated meshes + hierarchy, metres, +Y up."""
        from ..adapters.exchange import scene_to_glb
        return scene_to_glb(self.snapshot(), policy=policy, radial_segments=radial_segments,
                            latitude_bands=latitude_bands)

    def export_glb(self, path, *, overwrite: bool = False, **kwargs):
        """Write a GLB file atomically; returns the :class:`ConversionReport`."""
        from ..common.atomic import write_bytes_atomic
        data, report = self.to_glb(**kwargs)
        write_bytes_atomic(path, data, overwrite=overwrite)
        return report

    def to_stl(self, object_ids=None, *, policy="best_effort", radial_segments: int = 32,
               latitude_bands: int = 16):
        """``(binary_stl_bytes, ConversionReport)``: selected objects merged in the world frame."""
        from ..adapters.exchange import scene_to_stl
        ids = None if object_ids is None else ([object_ids] if isinstance(object_ids, str) else list(object_ids))
        return scene_to_stl(self.snapshot(), object_ids=ids, policy=policy, radial_segments=radial_segments,
                            latitude_bands=latitude_bands)

    def export_stl(self, path, object_ids=None, *, overwrite: bool = False, **kwargs):
        from ..common.atomic import write_bytes_atomic
        data, report = self.to_stl(object_ids, **kwargs)
        write_bytes_atomic(path, data, overwrite=overwrite)
        return report

    @classmethod
    def from_glb(cls, data: bytes, *, name: str | None = None, policy="best_effort"):
        """``(document, ConversionReport)`` from GLB bytes (meshes + node hierarchy)."""
        from ..adapters.exchange import glb_to_scene
        scene, report = glb_to_scene(data, policy=policy, name=name)
        return cls(scene), report

    @classmethod
    def import_glb(cls, path, **kwargs):
        from ..adapters.gltf import MAX_GLTF_BYTES
        return cls.from_glb(_read_bounded(path, MAX_GLTF_BYTES, "GLB"), **kwargs)

    @classmethod
    def from_stl(cls, data: bytes, *, name: str = "Imported STL", unit: str = "mm", weld: bool = True,
                 policy="best_effort"):
        """``(document, ConversionReport)`` from binary STL bytes (``unit`` is your declaration)."""
        from ..adapters.exchange import stl_to_scene
        scene, report = stl_to_scene(data, name=name, unit=unit, weld=weld, policy=policy)
        return cls(scene), report

    @classmethod
    def import_stl(cls, path, **kwargs):
        return cls.from_stl(_read_bounded(path, 84 + 50 * 333_333, "STL"), **kwargs)

    @classmethod
    def import_ifc(cls, path, *, strict: bool = True, max_products: int = 10000, name: str | None = None,
                   policy="best_effort"):
        """``(document, ConversionReport)`` from an IFC file (needs IfcOpenShell; geometry only)."""
        from ..adapters.exchange import ifc_to_scene
        scene, report = ifc_to_scene(path, strict=strict, max_products=max_products, policy=policy)
        if name is not None:
            scene.name = name
        return cls(scene), report


def _read_bounded(path, limit: int, label: str) -> bytes:
    from pathlib import Path
    source = Path(path)
    if not source.is_file() or source.stat().st_size > limit:
        raise ResourceLimitError(f"{label} file is absent or exceeds the input budget")
    with source.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ResourceLimitError(f"{label} file exceeds the input budget")
    return data
