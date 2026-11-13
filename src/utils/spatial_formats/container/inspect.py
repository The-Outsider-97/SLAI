"""Metadata-only inspection of SDRW/SGEO packages.

``inspect_package`` answers "what is this file?" without hydrating geometry,
decoding binary buffers, building a Scene/Drawing or running the semantic
validator. It reuses the loader's bounded ZIP/manifest checks, so a hostile
archive is rejected exactly as it would be on a full load, but cost stays
proportional to the archive index (plus, optionally, one bounded JSON parse
for object counts) instead of to geometry size.

An inspection is *not* a validation: ``verified`` says which integrity checks
actually ran. Callers that will trust the content must still ``load_package``.
"""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from hashlib import sha256
from io import BytesIO
from zipfile import BadZipFile, ZipFile

from ..common.errors import IntegrityError, ResourceLimitError, ValidationError
from ..common.helpers import MAX_DOCUMENT_NODES, digest, json_safe, parse_json
from .reader import (MAX_ARCHIVE_SIZE, MAX_UNPACKED, PAYLOADS, _bounded_member, _limits,
                     _read_index, read_source_bytes)

__all__ = ["PartInfo", "PackageInfo", "inspect_package", "inspect_bytes"]

_VERIFY_LEVELS = ("index", "digests")


@dataclass(frozen=True)
class PartInfo:
    path: str
    size: int
    sha256: str
    compressed_size: int


@dataclass(frozen=True)
class PackageInfo:
    format_id: str
    format_version: str
    schema: str
    document_kind: str
    spatial_dimension: int
    document_id: str
    revision: int
    units: str
    requires: tuple[str, ...]
    parts: tuple[PartInfo, ...]
    archive_size: int
    archive_sha256: str
    unpacked_size: int
    verified: tuple[str, ...]
    summary: dict = field(default_factory=dict)
    axis_up: str | None = None
    handedness: str | None = None

    def as_dict(self) -> dict:
        return {
            "format_id": self.format_id, "format_version": self.format_version,
            "schema": self.schema, "document_kind": self.document_kind,
            "spatial_dimension": self.spatial_dimension, "document_id": self.document_id,
            "revision": self.revision, "units": self.units, "requires": list(self.requires),
            "axis_up": self.axis_up, "handedness": self.handedness,
            "archive_size": self.archive_size, "archive_sha256": self.archive_sha256,
            "unpacked_size": self.unpacked_size, "verified": list(self.verified),
            "parts": [{"path": p.path, "size": p.size, "sha256": p.sha256,
                       "compressed_size": p.compressed_size} for p in self.parts],
            "summary": json_safe(self.summary),
        }


def _summarize(format_id: str, document) -> dict:
    """Structural counts from an *unvalidated* payload; never raises on shape."""
    if not isinstance(document, dict):
        return {}
    if format_id == "slai.sdrw":
        entities = document.get("entities")
        layers = document.get("layers")
        kinds = Counter(e.get("kind") for e in entities if isinstance(e, dict)) if isinstance(entities, list) else Counter()
        return {
            "name": document.get("name") if isinstance(document.get("name"), str) else None,
            "layer_count": len(layers) if isinstance(layers, list) else None,
            "layers": [x.get("name") for x in layers if isinstance(x, dict)][:200] if isinstance(layers, list) else [],
            "entity_count": len(entities) if isinstance(entities, list) else None,
            "entity_kinds": {str(k): v for k, v in sorted(kinds.items(), key=lambda kv: str(kv[0]))},
        }
    objects = document.get("objects")
    families = document.get("families")
    kinds: Counter = Counter()
    triangles = vertices = textured_meshes = 0
    if isinstance(objects, list):
        for obj in objects:
            geometry = obj.get("geometry") if isinstance(obj, dict) else None
            if not isinstance(geometry, dict):
                continue
            kinds[geometry.get("kind")] += 1
            if geometry.get("kind") == "mesh_buffer":
                buffers = geometry.get("buffers")
                try:
                    vertices += int(buffers["vertices"]["shape"][0])
                    triangles += int(buffers["faces"]["shape"][0])
                except (KeyError, TypeError, IndexError, ValueError):
                    pass
            elif geometry.get("kind") == "mesh":
                vertices += len(geometry.get("vertices") or [])
                triangles += len(geometry.get("faces") or [])
            if geometry.get("kind") in ("mesh", "mesh_buffer"):
                buffers = geometry.get("buffers")
                if geometry.get("uvs") is not None or (isinstance(buffers, dict) and "uvs" in buffers):
                    textured_meshes += 1
    materials = document.get("materials")
    images = document.get("images")
    return {
        "material_count": len(materials) if isinstance(materials, list) else 0,
        "image_count": len(images) if isinstance(images, list) else 0,
        "image_bytes": sum(i.get("size", 0) for i in images if isinstance(i, dict) and isinstance(i.get("size"), int))
        if isinstance(images, list) else 0,
        "meshes_with_uvs": textured_meshes,
        "name": document.get("name") if isinstance(document.get("name"), str) else None,
        "object_count": len(objects) if isinstance(objects, list) else None,
        "root_count": sum(1 for o in objects if isinstance(o, dict) and o.get("parent") is None) if isinstance(objects, list) else None,
        "family_count": len(families) if isinstance(families, list) else None,
        "geometry_kinds": {str(k): v for k, v in sorted(kinds.items(), key=lambda kv: str(kv[0]))},
        "mesh_vertex_count": vertices,
        "mesh_triangle_count": triangles,
    }


def _inspect(data: bytes, *, verify: str, summary: bool, max_unpacked: int) -> PackageInfo:
    if verify not in _VERIFY_LEVELS:
        raise ValidationError("verify must be one of: " + ", ".join(_VERIFY_LEVELS))
    _, expanded_limit = _limits(MAX_ARCHIVE_SIZE, max_unpacked)
    try:
        with ZipFile(BytesIO(data), "r", allowZip64=False) as archive:
            infos, manifest, format_id, parts = _read_index(archive, expanded_limit)
            verified = ["zip-index", "manifest-structure", "member-limits"]
            by_name = {info.filename: info for info in infos}
            payload_name = PAYLOADS[format_id]
            for name, part in parts.items():
                if by_name[name].file_size != part["size"]:
                    raise IntegrityError("member size mismatch: " + name)
            verified.append("member-sizes")
            document = None
            if verify == "digests":
                for name, part in parts.items():
                    raw = _bounded_member(archive, by_name[name], min(expanded_limit, part["size"]))
                    if digest(raw) != part["sha256"]:
                        raise IntegrityError("member SHA-256 mismatch: " + name)
                    if name == payload_name and summary:
                        document = parse_json(raw, max_nodes=MAX_DOCUMENT_NODES)
                verified.append("member-digests")
            elif summary:
                raw = _bounded_member(archive, by_name[payload_name],
                                      min(expanded_limit, parts[payload_name]["size"]))
                document = parse_json(raw, max_nodes=MAX_DOCUMENT_NODES)
            part_infos = tuple(PartInfo(name, part["size"], part["sha256"], by_name[name].compress_size)
                               for name, part in sorted(parts.items()))
            return PackageInfo(
                format_id=format_id, format_version=manifest["format_version"],
                schema=manifest["schema"], document_kind=manifest["document_kind"],
                spatial_dimension=manifest["spatial_dimension"], document_id=manifest["document_id"],
                revision=manifest["revision"], units=manifest["units"],
                requires=tuple(manifest.get("requires", [])), parts=part_infos,
                archive_size=len(data), archive_sha256=sha256(data).hexdigest(),
                unpacked_size=sum(p.size for p in part_infos), verified=tuple(verified),
                summary=_summarize(format_id, document) if summary else {},
                axis_up=manifest.get("axis_up"), handedness=manifest.get("handedness"))
    except (BadZipFile, EOFError, RuntimeError) as exc:
        raise IntegrityError("invalid or corrupted spatial ZIP package") from exc


def inspect_bytes(data, *, verify: str = "index", summary: bool = True,
                  max_unpacked: int = MAX_UNPACKED) -> PackageInfo:
    """Inspect an in-memory package (see :func:`inspect_package`)."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValidationError("package data must be bytes")
    if len(data) > MAX_ARCHIVE_SIZE:
        raise ResourceLimitError("compressed spatial package too large")
    return _inspect(bytes(data), verify=verify, summary=summary, max_unpacked=max_unpacked)


def inspect_package(path, *, verify: str = "index", summary: bool = True,
                    max_archive_size: int = MAX_ARCHIVE_SIZE,
                    max_unpacked: int = MAX_UNPACKED) -> PackageInfo:
    """Describe a package file without loading its geometry.

    ``verify="index"`` (default) checks the ZIP index, manifest and declared
    sizes. ``verify="digests"`` additionally decompresses every member within
    the usual budgets and checks its SHA-256. ``summary=True`` adds structural
    counts (entities, objects, mesh triangles...) from one bounded JSON parse
    of the payload; counts come from the *unvalidated* payload.
    """
    return _inspect(read_source_bytes(path, max_archive_size=max_archive_size),
                    verify=verify, summary=summary, max_unpacked=max_unpacked)
