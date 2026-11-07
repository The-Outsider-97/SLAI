"""Evaluators for non-destructive SGEO geometry modifiers.

An evaluator only returns a detached geometric result and diagnostics. It never
edits source objects or commits a geometry change. Numerical mesh operations
are explicitly distinguished from the exact axis-aligned box fast path.
"""
from __future__ import annotations

from copy import deepcopy
from math import cos, pi, sin, sqrt

from ..common.errors import (DependencyUnavailableError, GeometryValidationError,
                             ResourceLimitError, UnsupportedFeatureError)
from ..common.transforms import IDENTITY
from .box_boolean import boolean_boxes
from .geometry import MAX_TRIANGLES, MAX_VERTICES, validate_geometry


def _dependencies():
    try:
        import numpy as np
        import trimesh
    except ImportError as exc:
        raise DependencyUnavailableError("SGEO mesh modifiers require numpy and trimesh") from exc
    return np, trimesh


def _manifold():
    np, trimesh = _dependencies()
    try:
        import manifold3d  # noqa: F401 - declared optional CSG backend
    except ImportError as exc:
        raise DependencyUnavailableError("general 3D Boolean requires manifold3d") from exc
    return np, trimesh


def _as_mesh(record, *, tessellate_analytic=True):
    from .tessellation import tessellate
    geometry = record.get("geometry")
    if geometry is None:
        raise GeometryValidationError("modifier requires nonempty source geometry")
    validate_geometry(geometry)
    if geometry["kind"] == "mesh":
        return deepcopy(geometry), False
    if not tessellate_analytic or geometry["kind"] not in {"box", "sphere", "cylinder", "brep", "nurbs_surface"}:
        raise UnsupportedFeatureError("modifier requires a solid surface or supported analytic primitive")
    return tessellate(geometry, radial_segments=32, latitude_bands=16), True


def _to_trimesh(record):
    _, trimesh = _dependencies()
    mesh, tessellated = _as_mesh(record)
    if len(mesh["faces"]) > 250_000:
        raise ResourceLimitError("numerical mesh modifier exceeds its 250,000-triangle input budget")
    return trimesh.Trimesh(vertices=mesh["vertices"], faces=mesh["faces"], process=False), tessellated


def _from_trimesh(mesh):
    if mesh is None or mesh.is_empty or len(mesh.faces) == 0:
        return None
    if len(mesh.vertices) > MAX_VERTICES or len(mesh.faces) > MAX_TRIANGLES:
        raise ResourceLimitError("modifier generated too many mesh vertices or faces")
    if not mesh.is_watertight or not mesh.is_winding_consistent or not mesh.is_volume:
        raise GeometryValidationError("modifier generated an open or inconsistently oriented mesh")
    result = {"kind": "mesh", "vertices": mesh.vertices.tolist(), "faces": mesh.faces.tolist()}
    return validate_geometry(result)


def _policy(modifier, allowed):
    mode = modifier["evaluation_policy"]["mode"]
    if mode not in allowed:
        raise UnsupportedFeatureError("operation cannot guarantee the requested evaluation policy",
                                      context={"mode": mode, "supported": sorted(allowed)})


def _boolean(source, operands, modifier, source_world, operand_worlds):
    params = modifier["parameters"]
    op = {"union": "union", "intersect": "intersection", "subtract": "difference"}[params["operation"]]
    first, second = source["geometry"], operands[0]["geometry"]
    exact_boxes = (first is not None and second is not None and first["kind"] == second["kind"] == "box"
                   and tuple(source_world) == tuple(operand_worlds[0]))
    if exact_boxes:
        _policy(modifier, {"exact", "tolerance_controlled", "display_only"})
        geometry = boolean_boxes(first, second, op)
        return geometry, {"backend": "axis-aligned-box", "approximate": False,
                          "empty": geometry is None}

    _policy(modifier, {"tolerance_controlled", "display_only"})
    np, trimesh = _manifold()
    target = np.asarray(source_world, dtype=float).reshape(4, 4)
    other = np.asarray(operand_worlds[0], dtype=float).reshape(4, 4)
    try:
        relative = np.linalg.solve(target, other)
    except np.linalg.LinAlgError as exc:
        raise GeometryValidationError("non-invertible target object transform") from exc
    first_mesh, tessellated_a = _to_trimesh(source)
    second_mesh, tessellated_b = _to_trimesh(operands[0])
    second_mesh.apply_transform(relative)
    if not first_mesh.is_volume or not second_mesh.is_volume:
        raise GeometryValidationError("general mesh Boolean requires watertight positive-volume operands")
    try:
        if op == "union":
            result = trimesh.boolean.union([first_mesh, second_mesh], engine="manifold")
        elif op == "intersection":
            result = trimesh.boolean.intersection([first_mesh, second_mesh], engine="manifold")
        else:
            result = trimesh.boolean.difference([first_mesh, second_mesh], engine="manifold")
    except (ValueError, RuntimeError) as exc:
        raise GeometryValidationError("manifold Boolean could not construct a valid result") from exc
    geometry = _from_trimesh(result)
    return geometry, {"backend": "manifold3d", "approximate": True,
                      "tessellated_inputs": [tessellated_a, tessellated_b],
                      "uvs_dropped": any("uvs" in (record.get("geometry") or {}) for record in (source, operands[0])),
                      "error_bound": None, "empty": geometry is None}


def _split(source, modifier):
    params = modifier["parameters"]
    geometry = source.get("geometry")
    magnitude = sqrt(sum(v*v for v in params["normal"]))
    normal = [v / magnitude for v in params["normal"]]
    origin = params["origin"]
    if params["side"] == "negative":
        normal = [-v for v in normal]
    # Exact analytic clipping of a box by an axis-aligned halfspace.
    if geometry is not None and geometry["kind"] == "box" and sum(v != 0 for v in normal) == 1:
        _policy(modifier, {"exact", "tolerance_controlled", "display_only"})
        axis = next(i for i in range(3) if normal[i] != 0)
        lo = geometry["center"][axis] - geometry["size"][axis] / 2
        hi = geometry["center"][axis] + geometry["size"][axis] / 2
        if normal[axis] > 0:
            lo = max(lo, float(origin[axis]))
        else:
            hi = min(hi, float(origin[axis]))
        if hi <= lo:
            return None, {"backend": "analytic-box-split", "approximate": False, "empty": True}
        result = deepcopy(geometry)
        result["center"][axis] = (lo + hi) / 2
        result["size"][axis] = hi - lo
        return validate_geometry(result), {"backend": "analytic-box-split", "approximate": False, "empty": False}
    _policy(modifier, {"tolerance_controlled", "display_only"})
    np, trimesh = _dependencies()
    mesh, tessellated = _to_trimesh(source)
    if not mesh.is_volume:
        raise GeometryValidationError("solid split requires a watertight positive-volume input")
    try:
        result = mesh.slice_plane(plane_origin=origin, plane_normal=normal, cap=True)
    except (ValueError, RuntimeError, ImportError) as exc:
        raise GeometryValidationError("mesh split and cap failed") from exc
    geometry = _from_trimesh(result)
    return geometry, {"backend": "mesh-plane-split", "approximate": True,
                      "tessellated_input": tessellated, "error_bound": None,
                      "uvs_dropped": "uvs" in (source.get("geometry") or {}),
                      "empty": geometry is None}


def _bend(source, modifier):
    _policy(modifier, {"display_only"})
    np, trimesh = _dependencies()
    params = modifier["parameters"]
    mesh, tessellated = _as_mesh(source)
    original = trimesh.Trimesh(vertices=mesh["vertices"], faces=mesh["faces"], process=False)
    span = params["end"] - params["start"]
    max_edge = span / params["segments"]
    # Subdivide triangles before deformation so curves are not represented by
    # only the original box corner vertices. Hard limits prevent mesh explosion.
    if len(original.faces) > 100_000:
        raise ResourceLimitError("bend input is too large for adaptive subdivision")
    iteration_budget = 0
    while iteration_budget < 7 and len(original.faces) * 4 ** (iteration_budget + 1) <= MAX_TRIANGLES:
        iteration_budget += 1
    try:
        vertices, faces = trimesh.remesh.subdivide_to_size(
            original.vertices, original.faces, max_edge=max_edge, max_iter=iteration_budget)
    except (ValueError, RuntimeError) as exc:
        raise GeometryValidationError("bend subdivision could not be completed") from exc
    if len(vertices) > MAX_VERTICES or len(faces) > MAX_TRIANGLES:
        raise ResourceLimitError("bend would exceed the mesh budget")
    if max(sqrt(sum((vertices[a][i] - vertices[b][i])**2 for i in range(3)))
           for face in faces for a,b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))) > max_edge * (1 + 1e-8):
        raise ResourceLimitError("bend cannot reach requested subdivision within its resource budget")
    axis = {"x": 0, "y": 1, "z": 2}[params["axis"]]
    lateral = (axis + 1) % 3
    radius = span / params["angle"]
    offsets = [radius + float(v[lateral]) for v in vertices]
    if min(offsets) <= 1e-8 <= max(offsets) or min(abs(v) for v in offsets) <= 1e-8:
        raise GeometryValidationError("bend crosses its own center of curvature")
    output = np.asarray(vertices).copy()
    for v in output:
        s, t = float(v[axis]), float(v[lateral])
        if s <= params["start"]:
            continue
        bend_s = min(s, params["end"])
        theta = params["angle"] * (bend_s - params["start"]) / span
        extra = max(0.0, s - params["end"])
        v[axis] = params["start"] + (radius + t) * sin(theta) + extra * cos(theta)
        v[lateral] = (radius + t) * cos(theta) - radius - extra * sin(theta)
    result = trimesh.Trimesh(vertices=output, faces=faces, process=False)
    geometry = _from_trimesh(result) if original.is_volume else validate_geometry(
        {"kind": "mesh", "vertices": result.vertices.tolist(), "faces": result.faces.tolist()})
    return geometry, {"backend": "subdivided-mesh-bend", "approximate": True,
                      "error_bound": None, "segments": params["segments"],
                      "tessellated_input": tessellated,
                      "uvs_dropped": "uvs" in (source.get("geometry") or {}),
                      "watertight": bool(result.is_watertight)}


def _chamfer(source, modifier):
    _policy(modifier, {"tolerance_controlled", "display_only"})
    geometry = source.get("geometry")
    if geometry is None or geometry["kind"] != "box":
        raise UnsupportedFeatureError("chamfer currently requires an analytic axis-aligned box")
    np, trimesh = _dependencies()
    amount, depth, segments = (modifier["parameters"][k] for k in ("amount", "depth", "segments"))
    halves = [v / 2 for v in geometry["size"]]
    if 2 * max(amount, depth) >= min(geometry["size"]):
        raise GeometryValidationError("chamfer exceeds the available box cross-section")

    def margin(face, neighbor):
        return amount if neighbor == (face + 1) % 3 else depth

    points = []
    for face in range(3):
        other, last = (face + 1) % 3, (face + 2) % 3
        for sf in (-1, 1):
            for so in (-1, 1):
                for sl in (-1, 1):
                    p = [0.0] * 3
                    p[face], p[other], p[last] = (sf * halves[face],
                                                 so * (halves[other] - margin(face, other)),
                                                 sl * (halves[last] - margin(face, last)))
                    points.append(p)
    for free in range(3):
        face, other = (free + 1) % 3, (free + 2) % 3
        for sf in (-1, 1):
            for so in (-1, 1):
                for sk in (-1, 1):
                    for i in range(1, segments):
                        t = i / segments
                        theta = t * pi / 2
                        p = [0.0] * 3
                        p[face] = sf * (halves[face] - margin(other, face) * (1 - cos(theta)))
                        p[other] = so * (halves[other] - margin(face, other) * (1 - sin(theta)))
                        p[free] = sk * (halves[free] - ((1 - t) * margin(face, free)
                                                       + t * margin(other, free)))
                        points.append(p)
    hull = trimesh.convex.convex_hull(np.asarray(points))
    hull.apply_translation(geometry["center"])
    result = _from_trimesh(hull)
    return result, {"backend": "convex-box-chamfer", "approximate": segments > 1,
                    "error_bound": None if segments > 1 else 0.0, "segments": segments,
                    "watertight": True}


def evaluate(source, operands, modifier, *, source_world=IDENTITY, operand_worlds=()):
    """Return ``(detached_record, diagnostics)`` or fail without changing source."""
    operation = modifier["kind"].split(".", 1)[1]
    if operation == "boolean":
        geometry, details = _boolean(source, operands, modifier, source_world, operand_worlds)
    elif operation == "split":
        geometry, details = _split(source, modifier)
    elif operation == "bend":
        geometry, details = _bend(source, modifier)
    elif operation == "chamfer":
        geometry, details = _chamfer(source, modifier)
    else:
        raise UnsupportedFeatureError("unimplemented SGEO modifier kind")
    output = deepcopy(source)
    output["geometry"] = geometry
    return output, details
