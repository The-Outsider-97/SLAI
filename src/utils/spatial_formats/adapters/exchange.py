"""Explicit conversion-fidelity contracts for every supported interchange format.

Each wrapper pairs a backend adapter with a :class:`ConversionReport` that says,
for *this* document, what was preserved, approximated or dropped. A
``policy`` (see :class:`~.contracts.Policy`) is enforced **before** any byte is
produced or file touched, so a strict caller never receives a silently lossy
result:

* ``strict`` / ``lossless_where_supported`` – refuse on any loss or approximation;
* ``allow_approximation`` – refuse only if semantics would be discarded;
* ``best_effort`` (default) – always convert, always report.

Losses are computed from the actual document: an entity style only counts as
dropped when some entity has one. Identity is treated separately: SLAI ids that
cannot be stored by a format are listed under ``assumptions`` (new ids are
allocated on reimport) rather than ``dropped``.
"""
from __future__ import annotations

from collections import Counter
from math import tau
from pathlib import Path

from ..common.errors import UnsupportedFeatureError
from ..common.helpers import MAX_DOCUMENT_NODES, canonical_json, digest
from ..verification.loss_report import ConversionReport
from .contracts import Policy, enforce_policy
from .stl import export_binary_stl, import_binary_stl
from .svg import export_svg, import_svg

__all__ = ["mesh_to_stl", "stl_to_mesh", "drawing_to_svg", "svg_to_drawing", "drawing_to_dxf",
           "dxf_to_drawing", "drawing_to_dwg", "dwg_to_drawing", "scene_to_glb", "glb_to_scene",
           "scene_to_stl", "stl_to_scene", "ifc_to_scene"]

_FULL_SWEEP = tau - 1e-9


def _doc_digest(document) -> str:
    return digest(canonical_json(document.as_dict(), max_nodes=MAX_DOCUMENT_NODES))


def _file_digest(path) -> str:
    from ..common.atomic import file_sha256
    return file_sha256(path)


def _accuracy(dropped, approximations) -> str:
    return "approximate" if (dropped or approximations) else "exact"


# ---------------------------------------------------------------------------
# Triangle meshes <-> STL (kept for the single-mesh API)
# ---------------------------------------------------------------------------

def mesh_to_stl(geometry):
    """Export a triangle mesh and a machine-readable lossy conversion report."""
    result = export_binary_stl(geometry)
    report = ConversionReport(source_digest=digest(canonical_json(geometry, max_nodes=MAX_DOCUMENT_NODES)),
        target_digest=digest(result), adapter="binary-stl", adapter_version="1",
        accuracy_class="approximate",
        dropped=["semantic object identifiers", "parametric family information",
                 "scene hierarchy", "units", "materials", "unreferenced vertices"],
        approximations=["vertex coordinates encoded as float32"],
        warnings=["STL has no standard physical unit declaration"])
    return result, report


def stl_to_mesh(data):
    geometry = import_binary_stl(data)
    report = ConversionReport(source_digest=digest(data),
        target_digest=digest(canonical_json(geometry, max_nodes=MAX_DOCUMENT_NODES)), adapter="binary-stl",
        adapter_version="1", accuracy_class="approximate",
        dropped=["STL face attributes", "original indexed-vertex topology"],
        assumptions=["triangle coordinates use the caller-selected length unit"],
        warnings=["triangle vertices are not welded during import"])
    return geometry, report


# ---------------------------------------------------------------------------
# SDRW drawings <-> SVG / DXF / DWG
# ---------------------------------------------------------------------------

def _all_entities(drawing):
    yield from drawing.entities
    for block in drawing.blocks:
        yield from block["entities"]


def _drawing_features(drawing) -> dict:
    entities = drawing.entities
    used = {e.get("layer", "0") for e in entities}
    return {
        "texts": sum(1 for e in _all_entities(drawing) if e["kind"] == "text"),
        "hatches": sum(1 for e in _all_entities(drawing) if e["kind"] == "hatch"),
        "curved_hatches": sum(1 for e in _all_entities(drawing) if e["kind"] == "hatch" and any(
            loop["kind"] != "polygon" for loop in e["geometry"]["boundary"])),
        "dimensions": sum(1 for e in _all_entities(drawing) if e["kind"] == "dimension"),
        "inserts": sum(1 for e in entities if e["kind"] == "insert"),
        "blocks": len(drawing.blocks),
        "dotted": sum(1 for r in drawing.linetypes if any(v == 0 for v in r["pattern"])),
        "names": sum(1 for e in entities if e.get("name")),
        "entity_metadata": sum(1 for e in entities if e.get("metadata")),
        "document_metadata": bool(drawing.metadata),
        "hidden_layers": [l["name"] for l in drawing.layers if not l.get("visible", True)],
        "empty_layers": [l["name"] for l in drawing.layers if l["name"] not in used and l["name"] != "0"],
        "full_arcs": sum(1 for e in entities if e["kind"] == "arc"
                         and e["geometry"]["end_angle"] - e["geometry"]["start_angle"] >= _FULL_SWEEP),
        "arcs": sum(1 for e in entities if e["kind"] == "arc"),
        "points": sum(1 for e in entities if e["kind"] == "point"),
    }


def _drawing_loss(drawing, target: str):
    f = _drawing_features(drawing)
    dropped, approximations, assumptions, warnings = [], [], [], []
    if f["names"]:
        dropped.append(f"entity names on {f['names']} entities")
    if f["entity_metadata"] or f["document_metadata"]:
        dropped.append("semantic metadata (entity and document level)")
    if f["full_arcs"]:
        approximations.append(f"{f['full_arcs']} full-sweep arc(s) written as exactly equivalent circle(s)")
    if target == "svg":
        if f["hidden_layers"]:
            dropped.append("layer visibility for layers: " + ", ".join(f["hidden_layers"][:10]))
        if f["empty_layers"]:
            dropped.append("empty layers: " + ", ".join(f["empty_layers"][:10]))
        dropped.append("document revision")
        dropped.append("stable drawing identities on reimport (data-id is written; import with preserve_ids=True to reuse it)")
        warnings.append("SVG display coordinates may use a different y-axis orientation")
        if f["points"]:
            approximations.append(f"{f['points']} point(s) drawn as tiny circles (data-kind=point restores them)")
        assumptions.append("entity identifiers are written to data-id attributes")
        assumptions.append("stroke, stroke-width, fill and stroke-dasharray are written for styled entities and "
                           "layers; reimport ignores them")
        if f["texts"]:
            approximations.append(f"{f['texts']} text entit(ies) drawn as SVG text (fonts, wrapping and extents "
                                  "are the viewer's)")
        if f["hatches"]:
            approximations.append(f"{f['hatches']} hatch(es) written as paths"
                                  + (" with curved boundaries flattened" if f["curved_hatches"] else ""))
        if f["dimensions"]:
            dropped.append(f"{f['dimensions']} dimension(s) flattened to lines and text (measurement and "
                           "associativity are lost)")
        if f["blocks"]:
            dropped.append(f"block structure: {f['blocks']} block definition(s); {f['inserts']} insert(s) "
                           "expanded into plain shapes")
        if f["dotted"]:
            approximations.append("dotted linetype elements drawn as tiny dashes")
        if f["texts"] or f["hatches"] or f["dimensions"] or f["inserts"]:
            warnings.append("text, hatches, dimensions and inserts are export-only: reimporting this SVG is "
                            "refused rather than guessed")
    else:
        from .dxf_map import plan_export
        notes = plan_export(drawing).notes
        dropped.extend(notes.dropped)
        approximations.extend(notes.approximations)
        warnings.extend(notes.warnings)
        dropped.append("document revision")
        assumptions.append("SLAI entity identifiers are not stored in DXF; reimport allocates new identifiers")
        if f["arcs"]:
            warnings.append("arc angles are converted between radians and degrees (about 1e-16 relative error)")
        assumptions.append("DXF is written as R2010 model space with the drawing unit in $INSUNITS")
    return dropped, approximations, assumptions, warnings


def drawing_to_svg(drawing, *, policy="best_effort"):
    """Export a drawing as an SVG-subset string with a loss report."""
    drawing.validate()
    dropped, approximations, assumptions, warnings = _drawing_loss(drawing, "svg")
    enforce_policy(policy, losses=_semantic(dropped), approximations=approximations)
    markup = export_svg(drawing)
    report = ConversionReport(source_digest=_doc_digest(drawing),
        target_digest=digest(markup.encode("utf-8")), adapter="svg-subset", adapter_version="3",
        accuracy_class=_accuracy(dropped, approximations),
        object_mapping={e["id"]: e["id"] for e in drawing.entities},
        dropped=dropped, approximations=approximations, assumptions=assumptions, warnings=warnings)
    return markup, report


def svg_to_drawing(data, *, unit="mm", policy="best_effort", preserve_ids=False):
    """Import the supported SVG subset as a drawing with a loss report.

    ``preserve_ids=True`` reuses valid, unique ``data-id`` values written by
    :func:`drawing_to_svg`; by default fresh identifiers are allocated so an
    untrusted file can never dictate identities.
    """
    raw = data.encode("utf-8") if isinstance(data, str) else data
    drawing = import_svg(raw, unit=unit, preserve_ids=preserve_ids)
    text = raw.decode("utf-8-sig", errors="ignore")
    preserved = {e["id"]: e["id"] for e in drawing.entities if e["id"] in text}
    dropped = ["source SVG styles and unknown semantic metadata"]
    assumptions = [f"SVG numbers interpreted as {unit}"]
    warnings = []
    if len(preserved) != len(drawing.entities):
        warnings.append(f"new SLAI entity identifiers were allocated for {len(drawing.entities) - len(preserved)} entities")
    enforce_policy(policy, losses=(), approximations=())
    report = ConversionReport(source_digest=digest(raw), target_digest=_doc_digest(drawing),
        adapter="svg-subset", adapter_version="3", accuracy_class="approximate",
        object_mapping=preserved, dropped=dropped, assumptions=assumptions, warnings=warnings)
    return drawing, report


def _semantic(dropped):
    """Dropped items that discard user content (everything except bookkeeping)."""
    return [d for d in dropped if d != "document revision"]


def drawing_to_dxf(drawing, path, *, policy="best_effort", overwrite=True):
    """Write a DXF file atomically; return the :class:`ConversionReport`."""
    from .dxf import export_dxf
    drawing.validate()
    dropped, approximations, assumptions, warnings = _drawing_loss(drawing, "dxf")
    enforce_policy(policy, losses=_semantic(dropped), approximations=approximations)
    destination = Path(path)
    if destination.exists() and not overwrite:
        from ..common.errors import ConflictError
        raise ConflictError("output file already exists", path=str(destination))
    export_dxf(drawing, destination)
    import ezdxf
    return ConversionReport(source_digest=_doc_digest(drawing), target_digest=_file_digest(destination),
        adapter="dxf-ezdxf", adapter_version="3+ezdxf-" + ezdxf.__version__,
        accuracy_class=_accuracy(_semantic(dropped), approximations),
        dropped=dropped, approximations=approximations, assumptions=assumptions, warnings=warnings)


def dxf_to_drawing(path, *, unit=None, policy="best_effort", skip_unsupported=False):
    """Read the supported DXF subset with a loss report.

    Entities outside the subset fail closed (``UnsupportedFeatureError``) unless
    ``skip_unsupported`` is true; then they are omitted and listed under ``dropped``.
    ``policy`` is checked against everything that was dropped or approximated.
    """
    from .dxf import import_dxf_with_report
    drawing, notes = import_dxf_with_report(path, unit=unit, skip_unsupported=skip_unsupported)
    enforce_policy(policy, losses=notes.dropped, approximations=notes.approximations)
    import ezdxf
    assumptions = ["model-space POINT, LINE, LWPOLYLINE, CIRCLE, ARC, TEXT, MTEXT, DIMENSION, HATCH and INSERT "
                   "entities, BLOCK definitions and LTYPE patterns are read",
                   "new SLAI entity identifiers were allocated"]
    if unit is not None:
        assumptions.append(f"unit overridden by caller: {unit}")
    warnings = list(notes.warnings)
    warnings.append("entities outside the supported subset, paper space, bulges and 3D data are "
                    + ("omitted and listed under dropped" if skip_unsupported else "refused, never skipped"))
    report = ConversionReport(source_digest=_file_digest(path), target_digest=_doc_digest(drawing),
        adapter="dxf-ezdxf", adapter_version="3+ezdxf-" + ezdxf.__version__, accuracy_class="approximate",
        dropped=notes.dropped + ["DXF extended data (XDATA), plot styles, transparency and entity handles"],
        approximations=notes.approximations,
        assumptions=assumptions + notes.assumptions, warnings=warnings)
    return drawing, report


def drawing_to_dwg(drawing, path, *, executable, policy="best_effort", overwrite=False, version="ACAD2018",
                   timeout=90):
    """Write DWG through the configured ODA converter (via DXF); return the report."""
    from .dwg import export_dwg
    drawing.validate()
    dropped, approximations, assumptions, warnings = _drawing_loss(drawing, "dxf")
    enforce_policy(policy, losses=_semantic(dropped), approximations=approximations)
    destination = export_dwg(drawing, path, executable=executable, version=version, timeout=timeout,
                             overwrite=overwrite)
    return ConversionReport(source_digest=_doc_digest(drawing), target_digest=_file_digest(destination),
        adapter="dwg-oda", adapter_version="2", accuracy_class=_accuracy(_semantic(dropped), approximations),
        dropped=dropped, approximations=approximations,
        assumptions=assumptions + [f"converted by the configured ODA converter to {version}"],
        warnings=warnings + ["the ODA converter is an external, separately licensed tool"])


def dwg_to_drawing(path, *, executable, unit="mm", policy="best_effort", version="ACAD2018", timeout=90):
    """Read DWG through the configured ODA converter (via DXF)."""
    from .dwg import import_dwg
    drawing = import_dwg(path, executable=executable, unit=unit, version=version, timeout=timeout)
    enforce_policy(policy, losses=(), approximations=())
    report = ConversionReport(source_digest=_file_digest(path), target_digest=_doc_digest(drawing),
        adapter="dwg-oda", adapter_version="2", accuracy_class="approximate",
        dropped=["everything outside the DXF-supported planar entity subset (refused, never skipped)"],
        assumptions=[f"DWG units are not trusted; unit '{unit}' applied", "new SLAI entity identifiers were allocated"],
        warnings=["the ODA converter is an external, separately licensed tool"])
    return drawing, report


# ---------------------------------------------------------------------------
# SGEO scenes <-> GLB / STL / IFC
# ---------------------------------------------------------------------------

def _scene_digest(scene) -> str:
    return _doc_digest(scene)


def _flatten_losses(scene, summary, *, target):
    dropped, approximations, assumptions, warnings = [], [], [], []
    if summary["families_removed"] or summary["instances_flattened"]:
        dropped.append(f"parametric families ({len(summary['families_removed'])}) and instance parameters "
                       f"({len(summary['instances_flattened'])} instances become plain objects)")
    if summary["metadata_dropped"] or summary["scene_metadata_dropped"]:
        dropped.append("semantic metadata (object and scene level)")
    if summary["geometry_removed"]:
        dropped.append(f"{len(summary['geometry_removed'])} geometr"
                       f"{'y' if len(summary['geometry_removed']) == 1 else 'ies'} without a triangle surface "
                       "(points, segments, NURBS curves)")
    if summary.get("material_dropped"):
        dropped.append(f"materials of {len(summary['material_dropped'])} object(s) whose geometry has no surface")
    if any(m.get("metadata") for m in scene.materials):
        dropped.append("material metadata")
    if summary["tessellated"]:
        kinds = Counter(summary["tessellated"].values())
        approximations.append("analytic, NURBS and B-rep geometry tessellated to triangles: "
                              + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())))
    if summary.get("uvs_generated"):
        assumptions.append(f"texture coordinates were generated for {len(summary['uvs_generated'])} tessellated "
                           "surface(s) (per-face unit squares for boxes and B-rep faces, angular for round shapes)")
    return dropped, approximations, assumptions, warnings


def scene_to_glb(scene, *, policy="best_effort", radial_segments=32, latitude_bands=16):
    """Export a scene as GLB (binary glTF 2.0); return ``(bytes, report)``.

    Analytic primitives, NURBS surfaces and B-reps are tessellated, families flattened, metadata dropped; each is
    declared in the report and gated by ``policy``. PBR materials, embedded PNG/JPEG textures and UVs are kept.
    Coordinates are converted to metres and to glTF's +Y-up basis (and back on import).
    """
    from ..sgeo.meshing import flatten_scene
    from . import glb
    scene.validate()
    flat, summary = flatten_scene(scene, radial_segments=radial_segments, latitude_bands=latitude_bands)
    dropped, approximations, assumptions, warnings = _flatten_losses(scene, summary, target="glb")
    approximations.append("vertex and texture coordinates stored as float32 (about 1e-7 relative precision)")
    assumptions += [f"coordinates converted from {scene.unit} to metres", "+Z-up source rebased to glTF +Y-up",
                    "object and material ids are kept in extras.slai.id (other readers ignore them); node names use "
                    "the object name when unique, else its id"]
    if flat.materials:
        assumptions.append("texture v axis flipped on export (SLAI origin bottom-left, glTF top-left)")
    textured = {m["id"] for m in flat.materials if m.get("textures")}
    bare = [o["id"] for o in flat.objects if o.get("material") in textured and "uvs" not in o["geometry"]]
    if bare:
        warnings.append(f"{len(bare)} textured object(s) are meshes without UVs; viewers cannot map their textures")
    enforce_policy(policy, losses=dropped, approximations=approximations)
    data = glb.write_glb(flat)
    report = ConversionReport(source_digest=_scene_digest(scene), target_digest=digest(data),
        adapter="glb-native", adapter_version="3", accuracy_class="approximate",
        object_mapping={o["id"]: o["id"] for o in flat.objects if o.get("geometry") is not None},
        dropped=dropped, approximations=approximations, assumptions=assumptions, warnings=warnings)
    return data, report


def glb_to_scene(data, *, policy="best_effort", name=None):
    """Import a GLB file as a scene: meshes, node hierarchy, PBR materials, textures and UVs."""
    from . import glb
    scene, notes = glb.read_glb(data)
    if name is not None:
        scene.name = name
    dropped, warnings = [], []
    if notes.get("cameras_dropped"):
        dropped.append(f"{notes['cameras_dropped']} camera node(s)")
    if notes.get("unreachable_nodes"):
        dropped.append(f"{notes['unreachable_nodes']} node(s) outside the default scene")
    ignored = [k[len("ignored_"):-len("_attributes")] for k in notes if k.startswith("ignored_")]
    if ignored:
        warnings.append("vertex " + ", ".join(sorted(ignored)) + " were not stored (viewers recompute normals)")
    if notes.get("degenerate_triangles_removed"):
        warnings.append(f"{notes['degenerate_triangles_removed']} degenerate triangle(s) removed")
    if notes.get("empty_primitives"):
        warnings.append(f"{notes['empty_primitives']} primitive(s) had no valid triangles and were left empty")
    assumptions = ["glTF metres imported as unit 'm'", "glTF +Y-up rebased to +Z-up",
                   "texture v axis flipped on import (glTF top-left origin to SLAI bottom-left)",
                   "SLAI object and material ids were restored from extras.slai.id where present, "
                   "otherwise new identifiers were allocated; node names kept as object names"]
    if notes.get("multi_primitive_meshes_split"):
        assumptions.append(f"{notes['multi_primitive_meshes_split']} multi-primitive mesh(es) became a group object "
                           "with one child object per primitive")
    enforce_policy(policy, losses=dropped, approximations=())
    report = ConversionReport(source_digest=digest(data), target_digest=_scene_digest(scene),
        adapter="glb-native", adapter_version="3", accuracy_class="approximate",
        dropped=dropped, assumptions=assumptions,
        warnings=warnings + ["only triangle meshes, materials and textures are read; animations, skins, extensions "
                             "and external resources are refused, never skipped"])
    return scene, report


def scene_to_stl(scene, *, object_ids=None, policy="best_effort", radial_segments=32, latitude_bands=16):
    """Export selected (default: all) objects as one binary STL in the world frame."""
    from ..sgeo.meshing import extract_meshes, merge_meshes
    scene.validate()
    extraction = extract_meshes(scene, object_ids, radial_segments=radial_segments, latitude_bands=latitude_bands)
    if not extraction.meshes:
        raise UnsupportedFeatureError("selection contains no triangle geometry to export")
    selected = None if object_ids is None else set(object_ids)
    selected_objects = [o for o in scene.objects if selected is None or o["id"] in selected]
    dropped = ["object identity, names and scene hierarchy (objects are merged into one mesh)", "units"]
    if any("material" in o for o in selected_objects):
        dropped.append("materials and textures")
    if any(o.get("geometry") is not None and o["geometry"].get("uvs") for o in selected_objects):
        dropped.append("texture coordinates")
    approximations = ["vertex coordinates encoded as float32"]
    if extraction.tessellated:
        kinds = Counter(extraction.tessellated.values())
        approximations.append("analytic, NURBS and B-rep geometry tessellated to triangles: "
                              + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())))
    if extraction.skipped:
        dropped.append(f"{len(extraction.skipped)} object(s) without a triangle surface "
                       "(points, segments, NURBS curves)")
    in_scope = [o for o in scene.objects if selected is None or o["id"] in selected]
    if any(o.get("metadata") for o in in_scope) or (scene.metadata and selected is None):
        dropped.append("semantic metadata")
    if any("family_id" in o for o in in_scope):
        dropped.append("parametric family information")
    warnings = ["STL has no standard physical unit declaration"]
    if extraction.mirrored:
        warnings.append(f"{len(extraction.mirrored)} mirrored object(s): triangle winding was corrected")
    enforce_policy(policy, losses=dropped, approximations=approximations)
    data = export_binary_stl(merge_meshes(extraction.meshes.values()))
    report = ConversionReport(source_digest=_scene_digest(scene), target_digest=digest(data),
        adapter="binary-stl", adapter_version="2", accuracy_class="approximate",
        object_mapping={oid: "merged-mesh" for oid in extraction.meshes},
        dropped=dropped, approximations=approximations,
        assumptions=[f"coordinates are expressed in the scene unit '{scene.unit}' in the world frame"],
        warnings=warnings)
    return data, report


def stl_to_scene(data, *, name="Imported STL", unit="mm", weld=True, policy="best_effort"):
    """Import a binary STL as a one-object scene (optionally welding duplicate vertices)."""
    from ..sgeo.meshing import weld_vertices
    from ..sgeo.scene import Scene
    mesh = import_binary_stl(data)
    warnings, dropped = [], ["STL face attributes", "original unindexed triangle-soup layout"]
    if weld:
        mesh, collapsed = weld_vertices(mesh)
        if collapsed:
            warnings.append(f"{collapsed} degenerate triangle(s) removed while welding")
    else:
        warnings.append("triangle vertices are not welded")
    enforce_policy(policy, losses=(), approximations=())
    scene = Scene(name=name, unit=unit)
    scene.add_object(name=name, geometry=mesh)
    report = ConversionReport(source_digest=digest(data), target_digest=_scene_digest(scene),
        adapter="binary-stl", adapter_version="2", accuracy_class="approximate", dropped=dropped,
        approximations=["vertex coordinates are float32 values widened to float64"],
        assumptions=[f"STL coordinates interpreted as {unit}",
                     "vertices welded by exact coordinate equality" if weld else "no welding applied"],
        warnings=warnings)
    return scene, report


def ifc_to_scene(path, *, strict=True, max_products=10000, policy="best_effort"):
    """Import selected IFC products as a triangulated scene (geometry-first profile)."""
    from .ifc import import_ifc
    from ..common.atomic import file_sha256
    scene = import_ifc(path, max_products=max_products, strict=strict)
    enforce_policy(policy, losses=(), approximations=())
    report = ConversionReport(source_digest=file_sha256(path), target_digest=_scene_digest(scene),
        adapter="ifcopenshell", adapter_version="2", accuracy_class="approximate",
        dropped=["property sets, materials, classifications and georeferencing",
                 "original B-rep / swept-solid representations (triangulated by IfcOpenShell)",
                 "spatial containment hierarchy (products become a flat list)"],
        approximations=["geometry triangulated by the IfcOpenShell engine"],
        assumptions=["IFC length unit is metres after IfcOpenShell world-coordinate evaluation",
                     "IFC GlobalIds are kept as metadata 'ifc:global_id' (they are not UUIDs)"],
        warnings=["strict import refuses unsupported product classes" if strict else
                  "non-strict import skips unsupported products"])
    return scene, report
