"""Structured, machine-readable validation of in-memory SDRW/SGEO models.

``Drawing.validate()`` / ``Scene.validate()`` stop at the first violation and
raise. That is right for mutation guards but wrong for tooling and agents, which
need *every* problem, a stable code per problem, a location, and clearly
separated advisory findings. :func:`check_drawing` and :func:`check_scene`
return a :class:`ValidationReport` instead of raising.

Modes
-----
``basic``   semantic checks of the in-memory model (fast, no serialisation).
``strict``  ``basic`` plus JSON-schema conformance, the resource budgets that
            apply on save, and a deterministic save -> load round trip of the
            exact bytes that would be written. A model that passes ``strict``
            can be persisted and reloaded identically.

Severities: ``error`` (document invalid / unsaveable), ``warning`` (valid but
suspicious or likely to disappoint), ``info`` (neutral facts). Only errors make
``report.ok`` false.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from ..common.errors import (ResourceLimitError, SchemaError, SpatialFormatError, ValidationError,
                             ReferenceError, error_for_code)
from ..common.helpers import json_safe
from ..common.ids import require_id
from ..common.metadata import require_metadata

__all__ = ["Issue", "ValidationReport", "check_drawing", "check_scene", "report_from_exception",
           "MODES", "SEVERITIES"]

MODES = ("basic", "strict")
SEVERITIES = ("error", "warning", "info")
DEFAULT_MAX_ISSUES = 200


@dataclass(frozen=True)
class Issue:
    """One finding. ``code`` is stable (``SpatialFormatError.code`` for errors)."""

    severity: str
    code: str
    message: str
    path: str | None = None
    context: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"severity": self.severity, "code": self.code, "message": self.message,
                "path": self.path, "context": json_safe(self.context)}

    @classmethod
    def from_dict(cls, value: dict) -> "Issue":
        if not isinstance(value, dict) or value.get("severity") not in SEVERITIES:
            raise ValidationError("invalid issue record")
        return cls(value["severity"], str(value.get("code", "spatial")), str(value.get("message", "")),
                   value.get("path"), dict(value.get("context") or {}))


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of validating one document. Immutable and JSON round-trippable."""

    document_kind: str
    document_id: str | None
    revision: int | None
    mode: str
    issues: tuple[Issue, ...] = ()
    checks: tuple[str, ...] = ()
    truncated: bool = False

    @property
    def errors(self) -> tuple[Issue, ...]:
        return tuple(i for i in self.issues if i.severity == "error")

    @property
    def warnings(self) -> tuple[Issue, ...]:
        return tuple(i for i in self.issues if i.severity == "warning")

    @property
    def infos(self) -> tuple[Issue, ...]:
        return tuple(i for i in self.issues if i.severity == "info")

    @property
    def ok(self) -> bool:
        """True when there are no errors (warnings and infos never fail a report)."""
        return not self.errors

    def __bool__(self) -> bool:  # `if report:` means "valid"
        return self.ok

    def codes(self) -> tuple[str, ...]:
        return tuple(sorted({i.code for i in self.issues}))

    def raise_for_errors(self) -> "ValidationReport":
        """Raise the first error as its typed exception; return ``self`` if clean.

        The exception carries ``context={"error_count", "issues"}`` so callers
        who catch it still see every problem. Warnings never raise.
        """
        errors = self.errors
        if not errors:
            return self
        first = errors[0]
        cls = error_for_code(first.code)
        context = dict(first.context)
        context.update({"error_count": len(errors), "issues": [i.as_dict() for i in errors[:50]]})
        raise cls(first.message, path=first.path, context=context)

    def as_dict(self) -> dict:
        return {"document_kind": self.document_kind, "document_id": self.document_id,
                "revision": self.revision, "mode": self.mode, "ok": self.ok,
                "error_count": len(self.errors), "warning_count": len(self.warnings),
                "truncated": self.truncated, "checks": list(self.checks),
                "issues": [i.as_dict() for i in self.issues]}

    @classmethod
    def from_dict(cls, value: dict) -> "ValidationReport":
        if not isinstance(value, dict):
            raise ValidationError("invalid validation report")
        return cls(str(value.get("document_kind", "")), value.get("document_id"), value.get("revision"),
                   str(value.get("mode", "basic")),
                   tuple(Issue.from_dict(i) for i in value.get("issues", ())),
                   tuple(value.get("checks", ())), bool(value.get("truncated", False)))

    def summary(self) -> str:
        head = "OK" if self.ok else "INVALID"
        return (f"{head}: {self.document_kind} {self.document_id} r{self.revision} "
                f"[{self.mode}] {len(self.errors)} error(s), {len(self.warnings)} warning(s)")


class _Collector:
    """Accumulates issues up to a cap, remembering whether the cap was hit."""

    def __init__(self, max_issues: int):
        self.max_issues = max_issues
        self.issues: list[Issue] = []
        self.checks: list[str] = []
        self.truncated = False
        self._errors = 0

    def add(self, severity, code, message, path=None, **context):
        if len(self.issues) >= self.max_issues and severity != "error":
            self.truncated = True
            return
        if len(self.issues) >= self.max_issues * 2:  # hard stop even for errors
            self.truncated = True
            return
        if severity == "error":
            self._errors += 1
        self.issues.append(Issue(severity, code, message, path, context))

    def error_from(self, exc: SpatialFormatError, path=None, **context):
        merged = dict(exc.context or {})
        merged.update(context)
        message = exc.args[0] if exc.args else exc.default_message
        self.add("error", exc.code, str(message), exc.path or path, **merged)

    def guard(self, path, fn, *args, **context):
        """Run ``fn``; record a typed failure instead of propagating it."""
        try:
            return fn(*args)
        except SpatialFormatError as exc:
            self.error_from(exc, path, **context)
        return None

    @property
    def has_errors(self) -> bool:
        return self._errors > 0

    def ran(self, name):
        self.checks.append(name)


def report_from_exception(exc: SpatialFormatError, *, document_kind="unknown", mode="basic",
                          document_id=None, revision=None) -> ValidationReport:
    """Wrap any typed failure (e.g. from loading a file) as a one-issue report."""
    collector = _Collector(DEFAULT_MAX_ISSUES)
    collector.error_from(exc)
    return ValidationReport(document_kind, document_id, revision, mode, tuple(collector.issues))


def _check_mode(mode):
    if mode not in MODES:
        raise ValidationError("mode must be one of: " + ", ".join(MODES))


def _header(c: _Collector, model, label: str):
    c.ran("header")
    c.guard("/document_id", require_id, model.document_id)
    if type(model.revision) is not int or model.revision < 1:
        c.add("error", ValidationError.code, "revision must be a positive integer", "/revision")
    if not isinstance(model.name, str) or len(model.name) > 255:
        c.add("error", ValidationError.code, "invalid " + label + " name", "/name")
    c.guard("/metadata", require_metadata, model.metadata)


# ---------------------------------------------------------------------------
# SDRW
# ---------------------------------------------------------------------------

def check_drawing(drawing, *, mode: str = "strict", max_issues: int = DEFAULT_MAX_ISSUES) -> ValidationReport:
    """Validate a :class:`Drawing`, returning every finding instead of raising."""
    from ..sdrw.blocks import check_block_graph, validate_block_record
    from ..sdrw.drawing import Drawing, MAX_ENTITIES, MAX_LAYERS
    from ..sdrw.entities import validate_entity
    from ..sdrw.linetypes import validate_linetype
    _check_mode(mode)
    if not isinstance(drawing, Drawing):
        raise ValidationError("expected a Drawing")
    c = _Collector(max_issues)
    with drawing._mutation_lock:
        _header(c, drawing, "drawing")
        c.ran("linetypes")
        records = drawing.linetypes if isinstance(drawing.linetypes, list) else []
        lt_names: list[str] = []
        for index, record in enumerate(records):
            if c.guard(f"/linetypes/{index}", validate_linetype, record) is not None:
                lt_names.append(record["name"])
        for name, count in Counter(lt_names).items():
            if count > 1:
                c.add("error", ValidationError.code, "duplicate linetype name", "/linetypes", linetype=name, count=count)
        lt_set = set(lt_names)

        c.ran("layers")
        layers = drawing.layers
        names: list[str] = []
        if not isinstance(layers, list) or not 1 <= len(layers) <= MAX_LAYERS:
            c.add("error", ValidationError.code, "drawing requires between 1 and %d layers" % MAX_LAYERS, "/layers")
            layers = layers if isinstance(layers, list) else []
        for index, layer in enumerate(layers):
            path = f"/layers/{index}"
            if c.guard(path, Drawing._check_layer_record, layer, lt_set) is None and isinstance(layer, dict) \
                    and isinstance(layer.get("name"), str):
                names.append(layer["name"])
        for name, count in Counter(names).items():
            if count > 1:
                c.add("error", ValidationError.code, "duplicate drawing layer", "/layers", layer=name, count=count)
        known = set(names)

        c.ran("blocks")
        blocks = drawing.blocks if isinstance(drawing.blocks, list) else []
        block_names: list[str] = []
        for index, block in enumerate(blocks):
            if c.guard(f"/blocks/{index}", validate_block_record, block) is not None:
                block_names.append(block["name"])
        for name, count in Counter(block_names).items():
            if count > 1:
                c.add("error", ValidationError.code, "duplicate block name", "/blocks", block=name, count=count)
        block_set = set(block_names)
        seen_ids: Counter = Counter()
        inserted: Counter = Counter()

        def check_entity(entity, path, ident):
            if isinstance(ident, str):
                seen_ids[ident] += 1
                if seen_ids[ident] == 2:
                    c.add("error", ValidationError.code, "duplicate object identifier", path, id=ident)
            if isinstance(entity, dict) and entity.get("kind") == "insert" and isinstance(entity.get("geometry"), dict):
                inserted[entity["geometry"].get("block")] += 1
            c.guard(path, lambda e: validate_entity(e, known, linetypes=lt_set, blocks=block_set), entity, id=ident)

        for index, block in enumerate(blocks):
            if isinstance(block, dict) and isinstance(block.get("entities"), list):
                for position, entity in enumerate(block["entities"]):
                    check_entity(entity, f"/blocks/{index}/entities/{position}",
                                 entity.get("id") if isinstance(entity, dict) else None)
        if blocks and c.guard("/blocks", check_block_graph, blocks) is None:
            pass

        c.ran("entities")
        entities = drawing.entities
        if not isinstance(entities, list) or len(entities) > MAX_ENTITIES:
            c.add("error", ValidationError.code, "too many drawing entities (limit %d)" % MAX_ENTITIES, "/entities")
            entities = entities if isinstance(entities, list) else []
        used_layers: Counter = Counter()
        for index, entity in enumerate(entities):
            path = f"/entities/{index}"
            ident = entity.get("id") if isinstance(entity, dict) else None
            if isinstance(entity, dict) and isinstance(entity.get("layer", "0"), str):
                used_layers[entity.get("layer", "0")] += 1
            check_entity(entity, path, ident)

        # Advisory findings never affect `ok`.
        if not entities:
            c.add("info", "drawing.empty", "drawing has no entities", "/entities")
        for layer in layers:
            if isinstance(layer, dict) and isinstance(layer.get("name"), str):
                name = layer["name"]
                if name != "0" and not used_layers.get(name):
                    c.add("info", "layer.unused", "layer contains no entities", "/layers", layer=name)
                if layer.get("visible", True) is False and used_layers.get(name):
                    c.add("warning", "layer.hidden_content", "hidden layer contains entities", "/layers",
                          layer=name, entity_count=used_layers[name])
        for name in block_names:
            if not inserted.get(name):
                c.add("info", "block.unused", "block is never inserted", "/blocks", block=name)
        used_lt = {x for layer in layers if isinstance(layer, dict) for x in [layer.get("linetype")]}
        used_lt |= {(e.get("style") or {}).get("linetype") for e in entities if isinstance(e, dict)}
        for name in lt_names:
            if name not in used_lt:
                c.add("info", "linetype.unused", "linetype is not used by any layer or entity", "/linetypes",
                      linetype=name)
        if isinstance(entities, list) and len(entities) > 60_000:
            c.add("warning", "drawing.near_save_budget",
                  "very large drawing may exceed the JSON structural budget when saved", "/entities",
                  entity_count=len(entities))

        if mode == "strict":
            _strict(c, drawing, "slai.sdrw")
        return ValidationReport("drawing", _safe(drawing, "document_id"), _safe(drawing, "revision"), mode,
                                tuple(c.issues), tuple(c.checks), c.truncated)


# ---------------------------------------------------------------------------
# SGEO
# ---------------------------------------------------------------------------

def check_scene(scene, *, mode: str = "strict", max_issues: int = DEFAULT_MAX_ISSUES) -> ValidationReport:
    """Validate a :class:`Scene`, returning every finding instead of raising."""
    from ..sgeo.scene import Scene, MAX_OBJECTS, MAX_FAMILIES
    from ..sgeo.families import validate_family, generate_geometry
    from ..sgeo.analysis import analyze_mesh
    from ..common.transforms import IDENTITY, determinant, is_rigid
    _check_mode(mode)
    if not isinstance(scene, Scene):
        raise ValidationError("expected a Scene")
    c = _Collector(max_issues)
    with scene._mutation_lock:
        _header(c, scene, "scene")
        if scene.document_kind not in ("model", "family"):
            c.add("error", ValidationError.code, "invalid SGEO document kind", "/document_kind")

        c.ran("objects")
        objects = scene.objects
        if not isinstance(objects, list) or len(objects) > MAX_OBJECTS:
            c.add("error", ValidationError.code, "invalid scene object count (limit %d)" % MAX_OBJECTS, "/objects")
            objects = objects if isinstance(objects, list) else []
        ids: Counter = Counter()
        index: dict[str, dict] = {}
        for position, obj in enumerate(objects):
            path = f"/objects/{position}"
            ident = obj.get("id") if isinstance(obj, dict) else None
            ok_id = c.guard(path + "/id", require_id, ident) is not None
            if ok_id:
                ids[ident] += 1
                if ids[ident] == 2:
                    c.add("error", ValidationError.code, "duplicate object identifier", path, id=ident)
                index.setdefault(ident, obj)
            c.guard(path, Scene._check_object, obj, id=ident)

        c.ran("hierarchy")
        for position, obj in enumerate(objects):
            if not isinstance(obj, dict):
                continue
            parent = obj.get("parent")
            if parent is None:
                continue
            path = f"/objects/{position}/parent"
            if parent == obj.get("id"):
                c.add("error", ReferenceError.code, "object is its own parent", path, id=obj.get("id"))
            elif parent not in index:
                c.add("error", ReferenceError.code, "invalid scene parent reference", path,
                      id=obj.get("id"), parent=parent)
        reported: set[str] = set()
        state: dict[str, int] = {}
        for start in index:
            cursor, chain = start, []
            while cursor is not None and cursor in index and state.get(cursor) is None:
                state[cursor] = 1
                chain.append(cursor)
                nxt = index[cursor].get("parent")
                cursor = nxt if isinstance(nxt, str) else None
            if cursor is not None and state.get(cursor) == 1 and cursor not in reported:
                loop = chain[chain.index(cursor):] if cursor in chain else [cursor]
                reported.update(loop)
                c.add("error", ValidationError.code, "scene graph contains a parent cycle", "/objects",
                      cycle=loop[:20])
            for key in chain:
                state[key] = 2

        c.ran("families")
        families = scene.families
        if not isinstance(families, list) or len(families) > MAX_FAMILIES:
            c.add("error", ValidationError.code, "families must be a list of at most %d" % MAX_FAMILIES, "/families")
            families = families if isinstance(families, list) else []
        fam_ids: Counter = Counter()
        fam_index: dict[str, dict] = {}
        for position, family in enumerate(families):
            path = f"/families/{position}"
            ident = family.get("id") if isinstance(family, dict) else None
            if c.guard(path, validate_family, family, id=ident) is not None and isinstance(ident, str):
                fam_ids[ident] += 1
                if fam_ids[ident] == 2:
                    c.add("error", ValidationError.code, "duplicate object identifier", path, id=ident)
                fam_index.setdefault(ident, family)
        used_families: Counter = Counter()
        for position, obj in enumerate(objects):
            if not isinstance(obj, dict):
                continue
            path = f"/objects/{position}"
            if "family_id" in obj:
                fid = obj["family_id"]
                if not isinstance(fid, str) or fid not in fam_index:
                    c.add("error", ReferenceError.code, "unknown family reference", path + "/family_id",
                          id=obj.get("id"), family_id=fid if isinstance(fid, str) else None)
                    continue
                used_families[fid] += 1
                try:
                    expected = generate_geometry(fam_index[fid], obj.get("parameters", {}))
                except SpatialFormatError as exc:
                    c.error_from(exc, path + "/parameters", id=obj.get("id"))
                    continue
                if obj.get("geometry") != expected:
                    c.add("error", ValidationError.code,
                          "family instance geometry does not match its parameters", path + "/geometry",
                          id=obj.get("id"))
            elif "parameters" in obj:
                c.add("error", ValidationError.code, "parameters require a family reference",
                      path + "/parameters", id=obj.get("id"))

        c.ran("materials")
        from ..sgeo.materials import validate_registries
        materials = scene.materials if isinstance(scene.materials, list) else []
        images = scene.images if isinstance(scene.images, list) else []
        try:
            validate_registries(scene.materials, scene.images, image_data=scene.image_data or None)
        except SpatialFormatError as exc:
            c.error_from(exc, "/materials")
        if images and not scene.image_data:
            c.add("error", ValidationError.code,
                  "the bytes of the %d registered image(s) are not embedded; the scene cannot be saved" % len(images),
                  "/images")
        material_index = {m.get("id"): m for m in materials if isinstance(m, dict)}
        for position, obj in enumerate(objects):
            if isinstance(obj, dict) and "material" in obj and obj["material"] not in material_index:
                c.add("error", ReferenceError.code, "object references an unknown material",
                      f"/objects/{position}/material", id=obj.get("id"), material=obj["material"])
        used_materials = {o.get("material") for o in objects if isinstance(o, dict) and "material" in o}
        used_images = {spec.get("image") for m in material_index.values() if isinstance(m.get("textures"), dict)
                       for spec in m["textures"].values() if isinstance(spec, dict)}
        for position, m in enumerate(materials):
            if isinstance(m, dict) and m.get("id") not in used_materials:
                c.add("info", "material.unused", "material is not assigned to any object", f"/materials/{position}",
                      id=m.get("id"))
        for position, record in enumerate(images):
            if isinstance(record, dict) and record.get("id") not in used_images:
                c.add("info", "image.unused", "image is not used by any material texture", f"/images/{position}",
                      id=record.get("id"))

        # Advisory findings (only run on structurally sound objects).
        c.ran("advisories")
        children: Counter = Counter(o.get("parent") for o in objects if isinstance(o, dict) and o.get("parent"))
        if not objects:
            c.add("info", "scene.empty", "scene has no objects", "/objects")
        mesh_budget = 200  # bound the O(faces) diagnostics work
        brep_budget = 25   # tessellating a brep is much costlier than scanning a mesh
        for position, obj in enumerate(objects):
            if not isinstance(obj, dict) or not isinstance(obj.get("id"), str):
                continue
            path = f"/objects/{position}"
            geometry = obj.get("geometry")
            if geometry is None and not children.get(obj["id"]):
                c.add("info", "object.empty", "object has neither geometry nor children", path, id=obj["id"])
            transform = obj.get("transform", IDENTITY)
            try:
                if transform is not IDENTITY and len(transform) == 16:
                    if determinant(transform) < 0:
                        c.add("warning", "transform.mirrored",
                              "negative determinant mirrors geometry and flips mesh winding", path, id=obj["id"])
            except (TypeError, ValueError, SpatialFormatError):
                pass
            if isinstance(obj.get("material"), str) and obj["material"] in material_index and isinstance(geometry, dict):
                spec = material_index[obj["material"]]
                if spec.get("textures") and geometry.get("kind") == "mesh" and "uvs" not in geometry:
                    c.add("warning", "material.no_uvs",
                          "object has a textured material but its mesh has no UVs (the texture cannot be mapped)",
                          path, id=obj["id"], material=obj["material"])
            if isinstance(geometry, dict) and geometry.get("kind") == "brep" and brep_budget > 0:
                brep_budget -= 1
                from ..sgeo.brep_mesh import analyze_brep
                try:
                    facts = analyze_brep(geometry, radial_segments=16)
                except SpatialFormatError as exc:
                    c.error_from(exc, path + "/geometry", id=obj["id"])
                    continue
                if not facts["closed"]:
                    c.add("info", "brep.open", "brep is not a closed solid (a sheet has no volume)", path,
                          id=obj["id"])
                elif not facts["watertight"]:
                    c.add("warning", "brep.not_watertight",
                          "brep is topologically closed but its tessellation is not watertight", path, id=obj["id"])
                elif facts["volume"] is not None and facts["volume"] < 0:
                    c.add("warning", "brep.inverted", "closed brep has inward-facing faces (negative volume)", path,
                          id=obj["id"], volume=facts["volume"])
            if isinstance(geometry, dict) and geometry.get("kind") == "mesh" and mesh_budget > 0:
                mesh_budget -= 1
                try:
                    diag = analyze_mesh(geometry)
                except SpatialFormatError:
                    continue
                if diag.non_manifold_edge_count:
                    c.add("warning", "mesh.non_manifold", "mesh has non-manifold edges", path,
                          id=obj["id"], edges=diag.non_manifold_edge_count)
                if diag.inconsistent_winding_edge_count:
                    c.add("warning", "mesh.winding", "mesh has inconsistently oriented triangles", path,
                          id=obj["id"], edges=diag.inconsistent_winding_edge_count)
                if diag.boundary_edge_count:
                    c.add("info", "mesh.open", "mesh is not watertight", path,
                          id=obj["id"], boundary_edges=diag.boundary_edge_count)
                if diag.isolated_vertex_count:
                    c.add("warning", "mesh.isolated_vertices", "mesh has unreferenced vertices", path,
                          id=obj["id"], vertices=diag.isolated_vertex_count)
                if diag.is_closed_oriented_surface and diag.signed_volume is not None and diag.signed_volume < 0:
                    c.add("warning", "mesh.inverted", "closed mesh has inward-facing winding (negative volume)",
                          path, id=obj["id"], signed_volume=diag.signed_volume)
        for fid, family in fam_index.items():
            if not used_families.get(fid):
                c.add("info", "family.unused", "family has no instances", "/families", family_id=fid)
        if len(objects) > 60_000:
            c.add("warning", "scene.near_save_budget",
                  "very large scene may exceed the JSON structural budget when saved", "/objects",
                  object_count=len(objects))

        if mode == "strict":
            _strict(c, scene, "slai.sgeo")
        return ValidationReport("scene", _safe(scene, "document_id"), _safe(scene, "revision"), mode,
                                tuple(c.issues), tuple(c.checks), c.truncated)


def _safe(model, attribute):
    value = getattr(model, attribute, None)
    return value if isinstance(value, (str, int)) and not isinstance(value, bool) else None


def _strict(c: _Collector, model, format_id: str):
    """Schema, save budgets and deterministic round trip of the real bytes."""
    if c.has_errors:
        c.add("info", "strict.skipped", "schema and round-trip checks skipped: semantic errors present")
        return
    from ..container.writer import dumps_package
    from ..container.reader import loads_package
    from ..verification.invariants import validate_document
    try:
        manifest = model.manifest()
        document = model.as_dict()
    except SpatialFormatError as exc:  # pragma: no cover - guarded by basic checks
        c.error_from(exc)
        return
    c.ran("schema")
    try:
        validate_document({**manifest, "parts": [{"path": "drawing.json" if format_id == "slai.sdrw" else "scene.json",
                                                   "size": 0, "sha256": "0" * 64}]}, document)
    except SpatialFormatError as exc:
        c.error_from(exc)
        return
    c.ran("serialization")
    try:
        textures = dict(getattr(model, "image_data", None) or {}) or None
        first = dumps_package(manifest, document, textures=textures)
    except ResourceLimitError as exc:
        c.error_from(exc)
        return
    except SpatialFormatError as exc:
        c.error_from(exc)
        return
    c.ran("round-trip")
    try:
        package = loads_package(first)
        from ..sgeo.images import texture_path
        reloaded = {rec["id"]: package.assets[texture_path(rec["sha256"], rec["mime"])]
                    for rec in package.document.get("images", [])} or None
        again = dumps_package(package.manifest, package.document, textures=reloaded)
    except SpatialFormatError as exc:
        c.error_from(exc)
        return
    if package.document != document:
        c.add("error", SchemaError.code, "document changed during save/load round trip")
    c.ran("determinism")
    if again != first:
        c.add("error", SchemaError.code, "package serialization is not deterministic")
    c.add("info", "strict.package_size", "serialized package size", None,
          bytes=len(first))
