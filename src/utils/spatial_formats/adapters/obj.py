"""Wavefront OBJ triangle-surface interchange (text, geometry only); materials, units, hierarchy and semantics are lost.

OBJ is read and written as plain UTF-8 text. Only what OBJ states for certain is used: positions (``v``), faces (``f``) and the ``o``/``g``
names that split a file into objects. Everything else is either *skipped and counted* (normals, texture coordinates, materials, smoothing, points
and lines: reported as dropped) or *refused* (free-form curves and surfaces), never guessed. Polygons with more than three corners are split into
a fan of triangles, which is exact for convex polygons only; the import says how many it split so the approximation is visible.
"""
from __future__ import annotations

from collections import Counter

from ..common import ResourceLimitError, UnsupportedFeatureError, ValidationError
from ..sgeo.geometry import validate_geometry

MAX_BYTES = 64 * 1024 * 1024
MAX_FACES = 2_000_000
MAX_VERTICES = 4_000_000
MAX_GROUPS = 10_000
_FREE_FORM = {"curv", "curv2", "surf", "cstype", "deg", "bmat", "step", "parm", "trim", "hole", "scrv", "sp", "end", "con", "bevel", "c_interp", "d_interp", "lod", "shadow_obj", "trace_obj", "ctech", "stech"}
_LABELS = {"vt": "texture coordinates", "vn": "vertex normals", "vp": "parameter-space vertices", "mtllib": "material libraries", "usemtl": "material assignments",
           "s": "smoothing groups", "l": "polylines", "p": "points", "mg": "merging groups"}


def _name(value: str, fallback: str) -> str:
    cleaned = "_".join(str(value or "").split())
    return cleaned or fallback


def export_obj(objects: list[tuple[str, dict]], *, unit_comment: str = "") -> bytes:
    """``objects`` is a list of ``(name, mesh geometry)``; each becomes one ``o`` block. Indices are global and 1-based."""
    if not objects:
        raise UnsupportedFeatureError("OBJ export needs at least one triangle mesh")
    lines = ["# SLAI SGEO export", "# Geometry only: no materials, hierarchy or semantics."]
    if unit_comment:
        lines.append(f"# {unit_comment}")
    base, faces_total = 0, 0
    for index, (name, geometry) in enumerate(objects, 1):
        validate_geometry(geometry)
        if geometry["kind"] != "mesh":
            raise UnsupportedFeatureError("OBJ only represents triangle meshes")
        vertices, faces = geometry["vertices"], geometry["faces"]
        faces_total += len(faces)
        if faces_total > MAX_FACES:
            raise ResourceLimitError("too many faces for OBJ")
        lines.append(f"o {_name(name, f'object_{index}')}")
        lines.extend("v %r %r %r" % (float(x), float(y), float(z)) for x, y, z in vertices)
        lines.extend(f"f {a + 1 + base} {b + 1 + base} {c + 1 + base}" for a, b, c in faces)
        base += len(vertices)
    return ("\n".join(lines) + "\n").encode("utf-8")


def import_obj(data: bytes) -> tuple[list[tuple[str, dict]], dict]:
    """Parse OBJ bytes into ``[(group name, mesh geometry)]`` plus ``{"skipped": {directive: count}, "triangulated": n, "unnamed": bool}``.

    A face belongs to the object named by the latest ``o`` or ``g`` before it; faces before any name go to an unnamed first object. Objects
    keep only the vertices their faces use, so a file whose objects share one vertex pool splits cleanly."""
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise ValidationError("invalid OBJ payload")
    if len(data) > MAX_BYTES:
        raise ResourceLimitError("OBJ file is too large")
    try:
        text = bytes(data).decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValidationError("OBJ must be UTF-8 text") from exc
    positions: list[tuple[float, float, float]] = []
    groups: list[tuple[str, list[tuple[int, int, int]]]] = []
    current: list[tuple[int, int, int]] | None = None
    skipped: Counter[str] = Counter()
    triangulated = 0
    logical = ""
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if line.endswith("\\"):
            logical += line[:-1] + " "
            continue
        line, logical = logical + line, ""
        parts = line.split()
        if not parts:
            continue
        key, args = parts[0], parts[1:]
        if key == "v":
            if len(args) not in (3, 4, 6, 7):
                raise ValidationError("an OBJ vertex needs three coordinates")
            try:
                x, y, z = (float(a) for a in args[:3])
            except ValueError as exc:
                raise ValidationError("an OBJ vertex coordinate is not a number") from exc
            if not all(abs(c) < 1e300 and c == c for c in (x, y, z)):
                raise ValidationError("an OBJ vertex coordinate is not finite")
            positions.append((x, y, z))
            if len(positions) > MAX_VERTICES:
                raise ResourceLimitError("too many vertices for OBJ")
            if len(args) in (6, 7):
                skipped["vertex colours"] += 1
        elif key in ("o", "g"):
            if len(groups) >= MAX_GROUPS:
                raise ResourceLimitError("too many objects for OBJ")
            current = []
            groups.append((" ".join(args) or f"object {len(groups) + 1}", current))
        elif key == "f":
            if len(args) < 3:
                raise ValidationError("an OBJ face needs at least three vertices")
            ring = []
            for token in args:
                head = token.split("/", 1)[0]
                try:
                    i = int(head)
                except ValueError as exc:
                    raise ValidationError("an OBJ face index is not an integer") from exc
                if i == 0 or abs(i) > len(positions):
                    raise ValidationError("an OBJ face refers to a vertex that does not exist")
                ring.append(i - 1 if i > 0 else len(positions) + i)
            if current is None:
                current = []
                groups.append(("", current))
            if len(ring) > 3:
                triangulated += 1
            for k in range(1, len(ring) - 1):
                tri = (ring[0], ring[k], ring[k + 1])
                if len(set(tri)) == 3:
                    current.append(tri)
                else:
                    skipped["degenerate triangles"] += 1
            if sum(len(g[1]) for g in groups) > MAX_FACES:
                raise ResourceLimitError("too many faces for OBJ")
        elif key in _FREE_FORM:
            raise UnsupportedFeatureError(f"OBJ free-form geometry ('{key}') is not supported; only polygon meshes are read")
        elif key in _LABELS:
            skipped[_LABELS[key]] += 1
        else:
            skipped[f"unknown directive '{key}'"] += 1
    meshes: list[tuple[str, dict]] = []
    for name, faces in groups:
        if not faces:
            continue
        used = sorted({i for tri in faces for i in tri})
        remap = {old: new for new, old in enumerate(used)}
        geometry = validate_geometry({"kind": "mesh", "vertices": [list(positions[i]) for i in used], "faces": [[remap[a], remap[b], remap[c]] for a, b, c in faces]})
        meshes.append((name, geometry))
    if not meshes:
        raise ValidationError("the OBJ file contains no faces")
    return meshes, {"skipped": dict(skipped), "triangulated": triangulated, "unnamed": any(not n for n, _ in groups if _)}
