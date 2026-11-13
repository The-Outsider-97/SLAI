"""Scene-level mesh extraction and flattening for triangle-only interchange.

STL and glTF carry triangles, not analytic primitives, families or metadata.
These helpers turn an SGEO scene into plain triangle data *deterministically*
and say exactly what they dropped, so adapters can report the loss instead of
implying preservation:

* :func:`extract_meshes` – world-space triangle meshes per object (analytic
  primitives are tessellated; mirroring transforms keep outward winding).
* :func:`merge_meshes` – concatenate meshes into one indexed mesh.
* :func:`flatten_scene` – a *detached* scene with the same hierarchy whose
  geometry is triangle-only, without families or semantic metadata; materials
  and their texture images are kept (surfaces that need them get UVs).
* :func:`weld_vertices` – exact-coordinate vertex welding for unindexed soups
  such as STL.

Tessellation is a display approximation; the authoritative geometry stays the
analytic / NURBS / B-rep parameters in the source scene.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..common.errors import ResourceLimitError, ValidationError
from ..common.ids import require_id
from ..common.transforms import IDENTITY, determinant, require_transform
from .geometry import MAX_TRIANGLES, MAX_VERTICES, validate_geometry
from .scene import Scene
from .tessellation import tessellate

__all__ = ["MeshExtraction", "extract_meshes", "merge_meshes", "flatten_scene", "weld_vertices",
           "transform_mesh", "TRIANGLE_KINDS"]

TRIANGLE_KINDS = frozenset({"mesh", "box", "sphere", "cylinder", "nurbs_surface", "brep"})
_ANALYTIC = frozenset({"box", "sphere", "cylinder", "nurbs_surface", "brep"})   # tessellated, so approximate


@dataclass(frozen=True)
class MeshExtraction:
    """World-space meshes plus a precise account of what was not triangulated."""

    meshes: dict[str, dict]                      # object id -> mesh (document order)
    tessellated: dict[str, str] = field(default_factory=dict)   # id -> original analytic kind
    skipped: dict[str, str] = field(default_factory=dict)       # id -> reason (point/segment...)
    mirrored: tuple[str, ...] = ()               # ids whose winding was corrected

    @property
    def triangle_count(self) -> int:
        return sum(len(m["faces"]) for m in self.meshes.values())


def transform_mesh(mesh: dict, matrix, *, fix_winding: bool = True) -> dict:
    """Apply a 4x4 affine transform to a mesh. A mirroring matrix swaps winding."""
    m = require_transform(matrix)
    vertices = [[m[0] * x + m[1] * y + m[2] * z + m[3],
                 m[4] * x + m[5] * y + m[6] * z + m[7],
                 m[8] * x + m[9] * y + m[10] * z + m[11]] for x, y, z in mesh["vertices"]]
    faces = [list(f) for f in mesh["faces"]]
    if fix_winding and determinant(m) < 0:
        faces = [[a, c, b] for a, b, c in faces]
    result = {"kind": "mesh", "vertices": vertices, "faces": faces}
    if "uvs" in mesh:
        result["uvs"] = [list(uv) for uv in mesh["uvs"]]
    return validate_geometry(result)


def _select(scene: Scene, object_ids):
    if object_ids is None:
        return list(scene.objects)
    wanted = [require_id(i) for i in object_ids]
    if len(set(wanted)) != len(wanted):
        raise ValidationError("duplicate object id in selection")
    index = {o["id"]: o for o in scene.objects}
    missing = [i for i in wanted if i not in index]
    if missing:
        raise ValidationError("unknown object id in selection", context={"ids": missing[:10]})
    chosen = set(wanted)
    return [o for o in scene.objects if o["id"] in chosen]


def extract_meshes(scene: Scene, object_ids=None, *, radial_segments: int = 32,
                   latitude_bands: int = 16, world: bool = True, uvs: bool = False) -> MeshExtraction:
    """Triangle meshes for every geometric object (or the selected ones).

    ``world=True`` applies each object's *world* transform (composition of its
    ancestors); ``world=False`` returns local-frame meshes. ``uvs=True`` makes generated meshes carry texture
    coordinates (a mesh object keeps its own ``uvs`` either way).
    """
    with scene._mutation_lock:
        matrices = scene.world_transforms() if world else {}
        selected = _select(scene, object_ids)
        meshes, tessellated, skipped, mirrored = {}, {}, {}, []
        for obj in selected:
            geometry = obj.get("geometry")
            if geometry is None:
                continue
            kind = geometry["kind"]
            if kind not in TRIANGLE_KINDS:
                skipped[obj["id"]] = kind + " has no triangle surface"
                continue
            mesh = tessellate(geometry, radial_segments=radial_segments, latitude_bands=latitude_bands, uvs=uvs)
            if kind in _ANALYTIC:
                tessellated[obj["id"]] = kind
            if world:
                matrix = matrices[obj["id"]]
                if determinant(matrix) < 0:
                    mirrored.append(obj["id"])
                if tuple(matrix) != IDENTITY:
                    mesh = transform_mesh(mesh, matrix)
            meshes[obj["id"]] = mesh
        return MeshExtraction(meshes, tessellated, skipped, tuple(mirrored))


def merge_meshes(meshes) -> dict:
    """Concatenate meshes (iterable of mesh dicts) into one indexed mesh.

    The result has ``uvs`` only when every input mesh does."""
    vertices: list = []
    faces: list = []
    tex: list = []
    all_uvs = True
    for mesh in meshes:
        offset = len(vertices)
        if "uvs" in mesh:
            tex.extend(list(uv) for uv in mesh["uvs"])
        else:
            all_uvs = False
        vertices.extend(list(v) for v in mesh["vertices"])
        faces.extend([a + offset, b + offset, c + offset] for a, b, c in mesh["faces"])
        if len(vertices) > MAX_VERTICES or len(faces) > MAX_TRIANGLES:
            raise ResourceLimitError("merged mesh exceeds vertex/triangle budget")
    if not faces:
        raise ValidationError("no triangle geometry to merge")
    result = {"kind": "mesh", "vertices": vertices, "faces": faces}
    if all_uvs and len(tex) == len(vertices):
        result["uvs"] = tex
    return validate_geometry(result)


def weld_vertices(mesh: dict, *, decimals: int | None = None) -> tuple[dict, int]:
    """Merge vertices with identical coordinates; drop faces that collapse.

    ``decimals`` rounds the *comparison key* only (vertex coordinates stay
    exact); ``None`` welds bit-identical vertices. Returns ``(mesh, dropped)``
    where ``dropped`` counts degenerate faces removed. Vertices no longer
    referenced are removed.
    """
    if decimals is not None and (type(decimals) is not int or not 0 <= decimals <= 15):
        raise ValidationError("decimals must be an integer in [0, 15] or None")
    validate_geometry(mesh)
    if mesh["kind"] != "mesh":
        raise ValidationError("welding requires a triangle mesh")
    lookup: dict = {}
    remap: list[int] = []
    vertices: list = []
    source_uvs = mesh.get("uvs")
    uvs: list = []
    for index, vertex in enumerate(mesh["vertices"]):
        key = tuple(vertex) if decimals is None else tuple(round(c, decimals) + 0.0 for c in vertex)
        if source_uvs is not None:
            key = key + tuple(source_uvs[index])         # a texture seam keeps its split vertices
        found = lookup.get(key)
        if found is None:
            found = lookup[key] = len(vertices)
            vertices.append(list(vertex))
            if source_uvs is not None:
                uvs.append(list(source_uvs[index]))
        remap.append(found)
    faces, dropped, seen = [], 0, {}
    for a, b, c in mesh["faces"]:
        face = [remap[a], remap[b], remap[c]]
        if len(set(face)) < 3:
            dropped += 1
            continue
        faces.append(face)
    if not faces:
        raise ValidationError("welding removed every triangle")
    used = sorted({i for f in faces for i in f})
    if len(used) != len(vertices):
        compact = {old: new for new, old in enumerate(used)}
        vertices = [vertices[i] for i in used]
        if source_uvs is not None:
            uvs = [uvs[i] for i in used]
        faces = [[compact[i] for i in f] for f in faces]
    result = {"kind": "mesh", "vertices": vertices, "faces": faces}
    if source_uvs is not None:
        result["uvs"] = uvs
    return validate_geometry(result), dropped


def flatten_scene(scene: Scene, *, radial_segments: int = 32, latitude_bands: int = 16,
                  keep_metadata: bool = False) -> tuple[Scene, dict]:
    """Detached triangle-only copy of ``scene`` plus a loss summary.

    Hierarchy, names, ids and local transforms are preserved. Analytic
    primitives become meshes (tessellated, so approximate); points and segments
    have no surface and are removed from their object (the object remains as
    an empty node so children keep their parent); family instances become
    ordinary objects; families are removed; metadata is dropped unless
    ``keep_metadata``. NURBS surfaces and B-reps are tessellated like the
    primitives. Materials, images and object-material assignments are kept; a
    tessellated surface whose material has textures gets generated UVs.

    The loss summary has keys ``tessellated`` (id -> kind), ``geometry_removed``
    (id -> reason), ``instances_flattened`` (ids), ``families_removed`` (ids),
    ``metadata_dropped`` (bool), ``scene_metadata_dropped`` (bool),
    ``uvs_generated`` (ids) and ``material_dropped`` (ids of objects whose geometry vanished).
    """
    with scene._mutation_lock:
        scene.validate()
        copy = Scene(name=scene.name, unit=scene.unit, document_id=scene.document_id,
                     revision=scene.revision, document_kind="model", materials=scene.materials,
                     images=scene.images, image_data=scene.image_data)
        textured = {m["id"] for m in scene.materials if m.get("textures")}
        summary = {"tessellated": {}, "geometry_removed": {}, "instances_flattened": [],
                   "uvs_generated": [], "material_dropped": [],
                   "families_removed": [f["id"] for f in scene.families],
                   "metadata_dropped": False,
                   "scene_metadata_dropped": bool(scene.metadata) and not keep_metadata}
        if keep_metadata:
            copy.metadata = dict(scene.metadata)
        for obj in scene.objects:
            geometry = obj.get("geometry")
            new_geometry = None
            if geometry is not None:
                kind = geometry["kind"]
                if kind == "mesh":
                    new_geometry = geometry
                elif kind in _ANALYTIC:
                    wants_uvs = obj.get("material") in textured
                    new_geometry = tessellate(geometry, radial_segments=radial_segments,
                                              latitude_bands=latitude_bands, uvs=wants_uvs)
                    summary["tessellated"][obj["id"]] = kind
                    if wants_uvs:
                        summary["uvs_generated"].append(obj["id"])
                else:
                    summary["geometry_removed"][obj["id"]] = kind + " has no triangle surface"
            record = {"id": obj["id"], "name": obj.get("name", ""), "parent": obj.get("parent"),
                      "transform": list(obj.get("transform", IDENTITY)),
                      "geometry": new_geometry,
                      "metadata": dict(obj.get("metadata", {})) if keep_metadata else {}}
            if "material" in obj:
                if new_geometry is not None:
                    record["material"] = obj["material"]
                else:
                    summary["material_dropped"].append(obj["id"])
            if obj.get("metadata") and not keep_metadata:
                summary["metadata_dropped"] = True
            if "family_id" in obj:
                summary["instances_flattened"].append(obj["id"])
            copy.objects.append(record)
        return copy.validate(), summary
