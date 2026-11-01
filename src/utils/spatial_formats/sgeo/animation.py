"""Versioned SGEO animation metadata and deterministic read-only scene evaluation.

The ``slai.animation.v1`` metadata extension stores bounded *declarative*
tracks, never executable expressions or full per-frame geometry duplicates.
Time is seconds, not frame number. Translational/axis-angle tracks are local
deltas composed after a static object transform. Interpolation is clamped to
the first/last keyframe; playback/looping and IK are not implied.

Ownership
---------
This is the authoritative SGEO v1 animation schema and scene evaluator.
SLAI's Animation Agent may orchestrate calls but must not implement a second
copy of its interpolation, family evaluation, or scene state. Resource ceilings
below are SGEO format limits, not application configuration defaults.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType

from ..common.errors import ConflictError, ReferenceError, ResourceLimitError, ValidationError
from ..common.transforms import (compose, from_axis_angle, from_translation, from_scale,
                                 from_quaternion, unit_quaternion, slerp_quaternion)
from ..common.helpers import cubic_hermite_value
from ..common.helpers import finite, vector
from ..common.ids import require_id
from .families import generate_geometry

ANIMATION_KEY = "slai.animation.v1"
ANIMATION_V2_KEY = "slai.animation.v2"
ANIMATION_KEYS = (ANIMATION_KEY, ANIMATION_V2_KEY)
MAX_TRACKS = 4096
MAX_KEYFRAMES = 100_000
MAX_TIME = 1e9


def validate_animation(spec: dict, scene) -> dict:
    """Validate tracks, targets and native-family parameter compatibility."""
    if not isinstance(spec, dict) or set(spec) != {"duration", "tracks"}:
        raise ValidationError("animation must contain exactly duration and tracks")
    duration = finite(spec["duration"], "animation duration", limit=MAX_TIME)
    if duration <= 0:
        raise ValidationError("animation duration must be positive")
    tracks = spec["tracks"]
    if not isinstance(tracks, list):
        raise ValidationError("animation tracks must be a list")
    if len(tracks) > MAX_TRACKS:
        raise ResourceLimitError("too many animation tracks")
    objects = {obj["id"]: obj for obj in scene.objects}
    families = {fam["id"]: fam for fam in scene.families}
    targets = set()
    total = 0
    for track in tracks:
        if not isinstance(track, dict) or set(track) - {"object_id", "property", "keyframes", "interpolation", "axis"}:
            raise ValidationError("invalid animation track fields")
        oid = require_id(track.get("object_id"))
        obj = objects.get(oid)
        if obj is None:
            raise ReferenceError("animation refers to a missing scene object", context={"object_id": oid})
        prop = track.get("property")
        if not isinstance(prop, str) or (prop not in {"translation", "rotation_angle", "visibility"}
                                         and not prop.startswith("parameter.")):
            raise ValidationError("unsupported animation track property")
        key = (oid, prop)
        if key in targets:
            raise ValidationError("duplicate animation target/property")
        targets.add(key)
        if prop == "rotation_angle":
            if set(track) - {"object_id", "property", "keyframes", "interpolation", "axis"} or "axis" not in track:
                raise ValidationError("rotation_angle track requires an axis")
            axis = vector(track["axis"], 3, "rotation axis")
            if sum(x*x for x in axis) < 1e-24:
                raise ValidationError("rotation axis is zero")
        elif "axis" in track:
            raise ValidationError("only rotation_angle tracks accept axis")
        if prop.startswith("parameter."):
            name = prop.removeprefix("parameter.")
            family = families.get(obj.get("family_id"))
            if family is None or name not in family["parameters"]:
                raise ValidationError("animated parameter must exist in target family")
            definition = family["parameters"][name]
            if (definition["type"] not in {"length", "angle", "number", "integer"}
                    or "expression" in definition or definition.get("read_only", False)):
                raise ValidationError("animated parameter must be an independent numeric family parameter")
        interpolation = track.get("interpolation", "linear")
        if interpolation not in {"step", "linear"}:
            raise ValidationError("unsupported animation interpolation")
        if prop == "visibility" and interpolation != "step":
            raise ValidationError("visibility requires step interpolation")
        frames = track.get("keyframes")
        if not isinstance(frames, list) or not frames:
            raise ValidationError("animation track must contain keyframes")
        total += len(frames)
        if total > MAX_KEYFRAMES:
            raise ResourceLimitError("animation keyframe budget exceeded")
        previous = -1.0
        for frame in frames:
            if not isinstance(frame, dict) or set(frame) != {"time", "value"}:
                raise ValidationError("keyframe must contain exactly time and value")
            timestamp = finite(frame["time"], "keyframe time", limit=MAX_TIME)
            if timestamp < 0 or timestamp > duration or timestamp <= previous:
                raise ValidationError("keyframe timestamps must be strictly increasing within duration")
            previous = timestamp
            value = frame["value"]
            if prop == "translation":
                vector(value, 3, "translation")
            elif prop == "visibility":
                if type(value) is not bool:
                    raise ValidationError("visibility keyframes must be booleans")
            else:
                finite(value, "animation scalar", limit=1e9)
                if prop.startswith("parameter.") and definition["type"] == "integer" and interpolation != "step":
                    raise ValidationError("integer parameter animation requires step interpolation")
                if prop.startswith("parameter."):
                    if (value < definition.get("min", 0 if definition["type"] == "length" else -1e12)
                            or value > definition.get("max", 1e12)):
                        raise ValidationError("animated parameter keyframe is outside its declared bounds")
                    if definition["type"] == "integer" and not float(value).is_integer():
                        raise ValidationError("integer animation keyframes require integral values")
    return deepcopy(spec)



def validate_animation_v2(spec: dict, scene) -> dict:
    """Validate SGEO v2 typed transform tracks without changing v1 semantics.

    New v2 fields intentionally remain constrained to features with a native
    SGEO transform representation. Other requested properties require entity
    models and must not be stored as unevaluable tracks.
    """
    if not isinstance(spec, dict) or set(spec) - {"duration", "tracks", "drivers", "constraints"} or not {"duration", "tracks"} <= set(spec):
        raise ValidationError("v2 animation requires duration/tracks and optional drivers/constraints")
    duration = finite(spec["duration"], "animation duration", limit=MAX_TIME)
    if duration <= 0:
        raise ValidationError("animation duration must be positive")
    tracks = spec["tracks"]
    if not isinstance(tracks, list):
        raise ValidationError("animation tracks must be a list")
    if len(tracks) > MAX_TRACKS:
        raise ResourceLimitError("too many animation tracks")
    objects = {obj["id"]: obj for obj in scene.objects}
    families = {fam["id"]: fam for fam in scene.families}
    seen = set()
    count = 0
    for track in tracks:
        if not isinstance(track, dict) or set(track) - {
            "object_id", "property", "keyframes", "interpolation", "axis"
        }:
            raise ValidationError("invalid v2 animation track fields")
        oid = require_id(track.get("object_id"))
        obj = objects.get(oid)
        if obj is None:
            raise ReferenceError("animation refers to a missing scene object", context={"object_id": oid})
        prop = track.get("property")
        if not isinstance(prop, str) or (prop not in {
            "translation", "rotation_angle", "rotation_quaternion",
            "scale", "visibility"
        } and not prop.startswith("parameter.")):
            raise ValidationError("unsupported v2 animation property")
        if (oid, prop) in seen:
            raise ValidationError("duplicate animation target/property")
        if prop == "rotation_quaternion" and (oid, "rotation_angle") in seen:
            raise ValidationError("multiple rotational tracks on one target")
        if prop == "rotation_angle" and (oid, "rotation_quaternion") in seen:
            raise ValidationError("multiple rotational tracks on one target")
        seen.add((oid, prop))
        if prop == "rotation_angle":
            axis = vector(track.get("axis"), 3, "rotation axis")
            if sum(x*x for x in axis) < 1e-24:
                raise ValidationError("rotation axis is zero")
        elif "axis" in track:
            raise ValidationError("axis only supported for axis-angle tracks")
        definition = None
        if prop.startswith("parameter."):
            fam = families.get(obj.get("family_id"))
            name = prop.removeprefix("parameter.")
            if fam is None or name not in fam["parameters"]:
                raise ValidationError("animated parameter must exist in target family")
            definition = fam["parameters"][name]
            if (definition["type"] not in {"length", "angle", "number", "integer"}
                    or "expression" in definition or definition.get("read_only", False)):
                raise ValidationError("animated parameter must be independent and numeric")
        interpolation = track.get("interpolation", "linear")
        allowed = {"step"} if prop == "visibility" or (
            definition is not None and definition["type"] == "integer") else (
            {"step", "slerp"} if prop == "rotation_quaternion"
            else {"step", "linear", "cubic"}
        )
        if interpolation not in allowed:
            raise ValidationError("unsupported interpolation for v2 property type")
        frames = track.get("keyframes")
        if not isinstance(frames, list) or not frames:
            raise ValidationError("animation track must contain keyframes")
        count += len(frames)
        if count > MAX_KEYFRAMES:
            raise ResourceLimitError("animation keyframe budget exceeded")
        previous = -1.0
        for frame in frames:
            if not isinstance(frame, dict) or set(frame) - {"time", "value", "in_tangent", "out_tangent"}:
                raise ValidationError("invalid v2 keyframe fields")
            if set(frame) < {"time", "value"}:
                raise ValidationError("v2 keyframe requires time and value")
            timestamp = finite(frame["time"], "keyframe time", limit=MAX_TIME)
            if timestamp < 0 or timestamp > duration or timestamp <= previous:
                raise ValidationError("keyframe times must increase strictly within duration")
            previous = timestamp
            value = frame["value"]
            if prop in {"translation", "scale"}:
                values = vector(value, 3, prop)
                if prop == "scale" and any(abs(x) < 1e-9 for x in values):
                    raise ValidationError("v2 scale components must be nonsingular")
            elif prop == "rotation_quaternion":
                unit_quaternion(value)
            elif prop == "visibility":
                if type(value) is not bool:
                    raise ValidationError("visibility keyframes must be booleans")
            else:
                num = finite(value, "animation scalar", limit=1e9)
                if definition is not None:
                    if (num < definition.get("min", 0 if definition["type"] == "length" else -1e12)
                            or num > definition.get("max", 1e12)):
                        raise ValidationError("animated parameter is outside declared bounds")
                    if definition["type"] == "integer" and not num.is_integer():
                        raise ValidationError("integer animation requires integral values")
            tangent_fields = {"in_tangent", "out_tangent"} & set(frame)
            if interpolation != "cubic" and tangent_fields:
                raise ValidationError("tangents require cubic interpolation")
            if interpolation == "cubic":
                if tangent_fields != {"in_tangent", "out_tangent"}:
                    raise ValidationError("cubic keyframes require both tangents")
                for direction in ("in_tangent", "out_tangent"):
                    if prop in {"translation", "scale"}:
                        vector(frame[direction], 3, direction)
                    else:
                        finite(frame[direction], direction, limit=1e9)
    _validate_drivers(spec.get("drivers", []), objects, seen)
    _validate_constraints(spec.get("constraints", []), objects)
    return deepcopy(spec)



def _validate_drivers(drivers, objects, track_targets) -> tuple:
    """Validate bounded directed scalar mechanical couplings and detect cycles."""
    if not isinstance(drivers, list) or len(drivers) > MAX_TRACKS:
        raise ValidationError("drivers must be a bounded list")
    graph = {}
    destinations = set()
    for driver in drivers:
        if not isinstance(driver, dict) or set(driver) - {
            "source_object_id", "source_property", "target_object_id",
            "target_property", "factor", "offset", "axis"
        } or not {"source_object_id", "source_property", "target_object_id",
                  "target_property", "factor"} <= set(driver):
            raise ValidationError("invalid mechanical driver fields")
        src = require_id(driver["source_object_id"])
        dst = require_id(driver["target_object_id"])
        if src not in objects or dst not in objects:
            raise ReferenceError("driver references a missing object")
        sp, tp = driver["source_property"], driver["target_property"]
        if sp != "rotation_angle" and not (isinstance(sp, str) and sp.startswith("parameter.")):
            raise ValidationError("driver source must be an animated scalar")
        if tp not in {"rotation_angle", "translation"}:
            raise ValidationError("driver target must be rotation_angle or translation")
        source = (src, sp)
        target = (dst, tp)
        if target in track_targets or target in destinations:
            raise ValidationError("driver target conflicts with a track or another driver")
        if sp.startswith("parameter."):
            # Native family parameter dependency must have an evaluated source track.
            if source not in track_targets and source not in destinations:
                raise ValidationError("driver source must be an existing animated property")
        if source not in track_targets and source not in destinations:
            raise ValidationError("driver source must be an existing animated property")
        destinations.add(target)
        finite(driver["factor"], "driver factor")
        finite(driver.get("offset", 0), "driver offset")
        if tp in {"rotation_angle", "translation"}:
            axis = vector(driver.get("axis"), 3, "driver axis")
            if sum(component*component for component in axis) < 1e-24:
                raise ValidationError("driver axis must be nonzero")
        graph.setdefault(source, set()).add(target)
    # Kahn's algorithm: a coupled relationship requires a solver, not silent
    # evaluation-order-dependent feedback.
    nodes = set(graph) | {x for targets in graph.values() for x in targets}
    indegree = {node: 0 for node in nodes}
    for targets in graph.values():
        for target in targets:
            indegree[target] += 1
    ready = sorted(node for node, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        node = ready.pop(0)
        visited += 1
        for target in sorted(graph.get(node, ())):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
        ready.sort()
    if visited != len(nodes):
        raise ValidationError("mechanical driver dependency cycle requires simultaneous solving")
    return tuple(sorted(nodes))


def _evaluate_drivers(drivers: list, grouped: dict):
    pending = list(drivers)
    while pending:
        remaining = []
        progress = False
        for driver in pending:
            source = grouped.get(driver["source_object_id"], {}).get(driver["source_property"])
            if source is None:
                remaining.append(driver)
                continue
            value = source[1]
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise ValidationError("mechanical driver source is not scalar")
            scalar = finite(value * driver["factor"] + driver.get("offset", 0), "driven motion")
            target_prop = driver["target_property"]
            axis = vector(driver["axis"], 3, "driver axis")
            if target_prop == "translation":
                driven = [component * scalar for component in axis]
            else:
                driven = scalar
            grouped.setdefault(driver["target_object_id"], {})[target_prop] = (
                {"axis": list(axis), "property": target_prop}, driven
            )
            progress = True
        if not progress:
            raise ValidationError("unresolved mechanical driver dependency")
        pending = remaining


def _validate_constraints(constraints, objects) -> None:
    """Validate supported time-independent world-position constraints."""
    if not isinstance(constraints, list) or len(constraints) > 128:
        raise ValidationError("constraints must be a bounded list (max 128)")
    identifiers = set()
    for item in constraints:
        if not isinstance(item, dict) or set(item) - {
            "id", "kind", "object_id", "other_id", "target", "distance",
            "tolerance", "movable"
        } or not {"id", "kind", "object_id", "movable"} <= set(item):
            raise ValidationError("invalid constraint record")
        ident = require_id(item["id"])
        if ident in identifiers:
            raise ValidationError("duplicate constraint id")
        identifiers.add(ident)
        oid = require_id(item["object_id"])
        if oid not in objects:
            raise ReferenceError("constraint refers to a missing object")
        movable = item["movable"]
        if type(movable) is not bool:
            raise ValidationError("constraint movable must be boolean")
        if movable and objects[oid].get("parent") is not None:
            raise ValidationError("currently movable constraints require root objects")
        tolerance = finite(item.get("tolerance", 1e-6), "constraint tolerance", limit=1e6)
        if tolerance <= 0:
            raise ValidationError("constraint tolerance must be positive")
        if item["kind"] == "position":
            if "other_id" in item or "distance" in item:
                raise ValidationError("position constraint cannot have distance fields")
            vector(item.get("target"), 3, "position target")
        elif item["kind"] == "distance":
            if "target" in item:
                raise ValidationError("distance constraint cannot have a position target")
            other = require_id(item.get("other_id"))
            if other not in objects or other == oid:
                raise ReferenceError("invalid distance constraint endpoint")
            if finite(item.get("distance"), "constraint distance", limit=1e9) < 0:
                raise ValidationError("distance must be nonnegative")
        else:
            raise ValidationError("unsupported animation constraint kind")


def _sample_v2(track: dict, time: float):
    frames = track["keyframes"]
    if time <= frames[0]["time"]:
        return deepcopy(frames[0]["value"])
    if time >= frames[-1]["time"]:
        return deepcopy(frames[-1]["value"])
    low, high = 0, len(frames) - 1
    while high - low > 1:
        mid = (low + high) // 2
        if frames[mid]["time"] <= time:
            low = mid
        else:
            high = mid
    before, after = frames[low], frames[high]
    method = track.get("interpolation", "linear")
    if method == "step":
        return deepcopy(before["value"])
    dt = after["time"] - before["time"]
    alpha = (time - before["time"]) / dt
    a, b = before["value"], after["value"]
    if method == "slerp":
        return list(slerp_quaternion(a, b, alpha))
    if method == "cubic":
        def interpolate(x, y, m, n):
            return cubic_hermite_value(x, dt*m, y, dt*n, alpha)
        if isinstance(a, (tuple, list)):
            values = [interpolate(x, y, m, n) for x, y, m, n in zip(
                a, b, before["out_tangent"], after["in_tangent"]
            )]
        else:
            values = interpolate(a, b, before["out_tangent"], after["in_tangent"])
    elif isinstance(a, (tuple, list)):
        values = [x + (y-x)*alpha for x,y in zip(a,b)]
    else:
        values = a + (b-a)*alpha
    if track["property"] == "scale" and any(abs(x) < 1e-9 for x in values):
        raise ValidationError("interpolated scale is singular")
    return values


def _sample(track: dict, time: float):
    frames = track["keyframes"]
    if time <= frames[0]["time"]:
        return deepcopy(frames[0]["value"])
    if time >= frames[-1]["time"]:
        return deepcopy(frames[-1]["value"])
    # Binary search avoids traversing long animation tracks on every query.
    low, high = 0, len(frames) - 1
    while high - low > 1:
        mid = (low + high) // 2
        if frames[mid]["time"] <= time:
            low = mid
        else:
            high = mid
    first, second = frames[low], frames[high]
    if track.get("interpolation", "linear") == "step":
        return deepcopy(first["value"])
    fraction = (time - first["time"]) / (second["time"] - first["time"])
    a, b = first["value"], second["value"]
    if track["property"] == "translation":
        return [a[i] + (b[i] - a[i])*fraction for i in range(3)]
    return a + (b - a)*fraction


@dataclass(frozen=True, slots=True)
class EvaluatedScene:
    """Detached scene state with guarded read/query access; never the live scene."""

    time: float
    revision: int
    _scene: object
    _hidden: frozenset[str]
    _index: object = field(init=False, repr=False)
    _transforms: object = field(init=False, repr=False)
    _bounds: object = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_index", MappingProxyType({obj["id"]: obj for obj in self._scene.objects}))
        object.__setattr__(self, "_transforms", MappingProxyType(self._scene.world_transforms()))
        object.__setattr__(self, "_bounds", MappingProxyType(self._scene.bounds_by_object()))

    def ids(self, *, include_hidden: bool = False) -> tuple[str, ...]:
        return tuple(oid for oid in self._index if include_hidden or oid not in self._hidden)

    def geometric_ids(self) -> tuple[str, ...]:
        """Return visible geometric objects without copying any mesh payloads."""
        return tuple(oid for oid in self._bounds if oid not in self._hidden)

    def visible(self, object_id: str) -> bool:
        if object_id not in self._index:
            raise ReferenceError("unknown evaluated object")
        return object_id not in self._hidden

    def get_object_state(self, object_id: str) -> dict:
        require_id(object_id)
        found = self._index.get(object_id)
        if found is None:
            raise ReferenceError("unknown evaluated object")
        return {"time": self.time, "revision": self.revision, "visible": object_id not in self._hidden,
                "object": deepcopy(found), "world_transform": self.world_transform(object_id)}

    def world_transform(self, object_id: str) -> tuple:
        if object_id not in self._transforms:
            raise ReferenceError("unknown evaluated object")
        return self._transforms[object_id]

    def world_bounds(self, object_id: str, *, include_hidden: bool = False):
        if not include_hidden and not self.visible(object_id):
            raise ValidationError("object is hidden at the requested time")
        if object_id not in self._bounds:
            raise ValidationError("object has no geometric representation")
        return self._bounds[object_id]

    def snapshot(self):
        """Return an independent mutable copy for downstream computation."""
        return self._scene._detached_copy()


def evaluate_scene(scene, time: float, *, expected_revision: int | None = None) -> EvaluatedScene:
    """Read a consistent, detached scene snapshot and evaluate at ``time``.

    ``expected_revision`` is an optional optimistic check performed under the
    same SGEO mutation lock as the copy. A mismatched revision raises SGEO's
    native ``ConflictError``. This intentionally does not change the existing
    ``Scene.evaluate(time)`` call or the ``slai.animation.v1`` schema.
    """
    timestamp = finite(time, "evaluation time", limit=MAX_TIME)
    if timestamp < 0:
        raise ValidationError("evaluation time must be nonnegative")
    if expected_revision is not None and (type(expected_revision) is not int or expected_revision < 1):
        raise ValidationError("expected_revision must be a positive integer")
    with scene._mutation_lock:
        if expected_revision is not None and scene.revision != expected_revision:
            raise ConflictError("scene revision does not match expected revision")
        scene.validate()
        candidate = scene._detached_copy()
    active = ANIMATION_V2_KEY if ANIMATION_V2_KEY in candidate.metadata else ANIMATION_KEY
    spec = candidate.metadata.get(active)
    if spec is None:
        return EvaluatedScene(timestamp, candidate.revision, candidate, frozenset())
    index = {obj["id"]: obj for obj in candidate.objects}
    families = {f["id"]: f for f in candidate.families}
    grouped: dict[str, dict[str, tuple[dict, object]]] = {}
    for track in spec["tracks"]:
        grouped.setdefault(track["object_id"], {})[track["property"]] = (track, _sample_v2(track, timestamp) if active == ANIMATION_V2_KEY else _sample(track, timestamp))
    if active == ANIMATION_V2_KEY and spec.get("drivers"):
        _evaluate_drivers(spec["drivers"], grouped)
    invisible: set[str] = set()
    for oid, tracks in grouped.items():
        obj = index[oid]
        if "visibility" in tracks and not tracks["visibility"][1]:
            invisible.add(oid)
        transform = obj["transform"]
        if "translation" in tracks:
            transform = compose(transform, from_translation(*tracks["translation"][1]))
        if "rotation_angle" in tracks:
            track, angle = tracks["rotation_angle"]
            transform = compose(transform, from_axis_angle(track["axis"], angle))
        if "rotation_quaternion" in tracks:
            transform = compose(transform, from_quaternion(tracks["rotation_quaternion"][1]))
        if "scale" in tracks:
            transform = compose(transform, from_scale(*tracks["scale"][1]))
        obj["transform"] = list(transform)
        parameters = {name.removeprefix("parameter."): value for name, (_, value) in tracks.items()
                      if name.startswith("parameter.")}
        if parameters:
            overrides = {**obj.get("parameters", {}), **parameters}
            obj["geometry"] = generate_geometry(families[obj["family_id"]], overrides)
            obj["parameters"] = overrides
    # Visibility inherits through the hierarchy without mutating object data.
    for oid, obj in index.items():
        parent = obj.get("parent")
        while parent is not None:
            if parent in invisible:
                invisible.add(oid)
                break
            parent = index[parent].get("parent")
    candidate.validate()
    return EvaluatedScene(timestamp, candidate.revision, candidate, frozenset(invisible))


__all__ = ["ANIMATION_KEY", "ANIMATION_V2_KEY", "ANIMATION_KEYS", "EvaluatedScene",
           "validate_animation", "validate_animation_v2", "evaluate_scene"]
