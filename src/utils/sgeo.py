"""
SGEO -- SLAI's native 3D spatial-model format (``.sgeo``).

SGEO = SLAI Geometry. Versatile enough for architectural BIM models, engineering
assemblies, scientific geometry and parametric families. Supports family
definitions, type parameters, instance overrides and deterministic regeneration.

This module is the small, stable public facade. The implementation lives in
``src.utils.spatial_formats`` (this module imports it, never the reverse).

Quick start::

    from src.utils.sgeo import SGEODocument

    scene = SGEODocument.new("Room", unit="m")
    wall = scene.add_box(width=3.0, depth=0.2, height=2.7, name="Wall A", position=(0, 0, 1.35))
    scene.attach_semantics(wall, {"category": "wall", "fire_rating": "unknown"}, namespace="aec")
    scene.validate().raise_for_errors()
    scene.save("room.sgeo")

    with SGEODocument.open("room.sgeo") as room:             # read-only by default
        info = room.inspect()
        walls = room.query_ids(type_name="box", limit=100)

Conventions: right-handed, **+Z up**, lengths in the document unit, angles in
radians (``degrees=True`` where offered), flat row-major 4x4 transforms, UUID
ids, metadata keys ``namespace:name`` or ``slai.*``. Meshes are stored as
content-addressed little-endian binary buffers inside the package and are
hydrated transparently. Every edit is atomic and bumps the revision once;
batches (``apply``) are one revision as a whole.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .spatial_formats.adapters import capabilities as _capabilities
from .spatial_formats.common.errors import (
    ConflictError, ConversionLossError, DependencyUnavailableError, GeometryValidationError,
    IntegrityError, MigrationError, ReadOnlyDocumentError, ReferenceError as SpatialReferenceError,
    ResourceLimitError, SchemaError, SecurityLimitError, SpatialFormatError, UnsupportedFeatureError,
    ValidationError, error_for_code)
from .spatial_formats.container import PackageInfo, inspect_package, loads_package, read_source_bytes
from .spatial_formats.container.migration import CURRENT_VERSION
from .spatial_formats.container.reader import MIMES
from .spatial_formats.documents import SaveResult
from .spatial_formats.sgeo.document import SceneDocument
from .spatial_formats.verification import ConversionReport, Issue, ValidationReport, report_from_exception

__version__ = "0.2.0"

FORMAT_ID = "slai.sgeo"
FORMAT_VERSION = CURRENT_VERSION  # What new documents are written as.
SCHEMA = FORMAT_ID + "/" + ".".join(FORMAT_VERSION.split(".")[:2])  # Value of ``manifest["schema"]``.
MIMETYPE = MIMES[FORMAT_ID].decode("ascii")
FILE_EXTENSION = ".sgeo"
AXIS_UP = "+Z"
HANDEDNESS = "right"


class SGEODocument(SceneDocument):
    """An editable, validated 3D scene (``document_kind`` ``model`` or ``family``).

    Construction: :meth:`new`, :meth:`open`, :meth:`from_bytes`, :meth:`from_glb`,
    :meth:`import_glb`, :meth:`from_stl`, :meth:`import_stl`, :meth:`import_ifc`.
    Objects: ``add_object/group/point/segment/box/sphere/cylinder/mesh/extrusion/revolution``, NURBS
    (``add_nurbs_curve``, ``add_nurbs_surface``) and exact B-reps (``add_brep``, ``add_brep_box/cylinder/
    sphere/torus/extrusion/sheet``). Materials and textures: ``add_material``, ``update_material``,
    ``set_texture``, ``set_material``, ``add_image``/``add_image_file``, ``images``, ``materials``.
    Edits (one revision each): ``remove``, ``remove_tree``, ``rename_object``, ``set_geometry``,
    ``set_transform``, ``move``, ``rotate``, ``reparent``, ``duplicate``, ``attach_semantics``,
    ``add_claim``, ``convert_unit``, families (``define_*_family``, ``instantiate``,
    ``set_parameters``...), and ``apply`` for atomic batches.
    Queries: ``get``, ``query_ids``, ``find``, ``children``, ``descendants``, ``ancestors``,
    ``bounds``, ``world_bounds``, ``volume``, ``surface_area``, ``analyze``, ``to_mesh``.
    Persistence: ``validate``, ``inspect``, ``save``, ``dumps``, ``content_digest``.
    Interchange: ``to_glb``/``export_glb``, ``to_stl``/``export_stl`` (each returns a
    ``ConversionReport``); agents: ``reference``, ``spatial_records``, ``to_agent_bundle``.
    """

    __slots__ = ()


@dataclass(frozen=True)
class SGEOPackage:
    """A loaded, fully validated package at the container level (no editing API)."""

    manifest: dict
    document: dict   # hydrated: meshes are ordinary indexed meshes, not buffer references
    sha256: str
    size: int
    path: str | None = None
    assets: dict | None = None   # embedded texture images: ``textures/sha256-<hex>.png|jpg`` -> bytes

    def to_document(self, *, mode: str = "write") -> SGEODocument:
        """Build an editable (or read-only) document from this package."""
        from .spatial_formats.container.reader import SpatialPackage
        return SGEODocument.from_package(SpatialPackage(self.manifest, self.document, dict(self.assets or {})),
                                         mode=mode)


def new(name: str = "Untitled", *, unit: str = "m", document_kind: str = "model", metadata=None,
        document_id=None) -> SGEODocument:
    """Create an empty scene (same as :meth:`SGEODocument.new`)."""
    return SGEODocument.new(name, unit=unit, document_kind=document_kind, metadata=metadata,
                            document_id=document_id)


def open_document(path, *, mode: str = "read") -> SGEODocument:
    """Open and validate a ``.sgeo`` file (``mode``: ``"read"`` or ``"write"``)."""
    return SGEODocument.open(path, mode=mode)


def read_package(path) -> SGEOPackage:
    """Load a ``.sgeo`` file as a plain validated package (manifest + hydrated payload + digest)."""
    from hashlib import sha256
    data = read_source_bytes(path)
    package = loads_package(data)
    if package.manifest["format_id"] != FORMAT_ID:
        raise UnsupportedFeatureError("expected a %s package but found %s" % (FORMAT_ID, package.manifest["format_id"]))
    textures = {name: raw for name, raw in package.assets.items() if name.startswith("textures/")}
    return SGEOPackage(package.manifest, package.document, sha256(data).hexdigest(), len(data), str(Path(path)),
                       textures)


def inspect(path, *, verify: str = "index", summary: bool = True) -> PackageInfo:
    """Describe a ``.sgeo`` file without hydrating geometry (identity, units, parts, counts)."""
    info = inspect_package(path, verify=verify, summary=summary)
    if info.format_id != FORMAT_ID:
        raise UnsupportedFeatureError("expected a %s package but found %s" % (FORMAT_ID, info.format_id))
    return info


def validate(path, *, mode: str = "strict", max_issues: int = 200) -> ValidationReport:
    """Validate a ``.sgeo`` file and return a :class:`ValidationReport`.

    Problems loading the file (corruption, integrity, limits, wrong format) are
    reported as errors in the report rather than raised; OS errors such as a
    missing file still propagate.
    """
    try:
        return SGEODocument.open(path, mode="read").validate(mode, max_issues=max_issues)
    except SpatialFormatError as exc:
        return report_from_exception(exc, document_kind="scene", mode=mode)


def capabilities(*, dwg_executable=None) -> dict:
    """Which interchange formats are available here, and their limits (no side effects)."""
    return _capabilities(dwg_executable=dwg_executable)


# The Spatial* names below are the stable exception taxonomy of the feasibility report.
SpatialSchemaError = SchemaError
SpatialIntegrityError = IntegrityError
SpatialUnsupportedFeature = UnsupportedFeatureError
SpatialSecurityLimitError = SecurityLimitError
SpatialGeometryValidationError = GeometryValidationError
SpatialConversionLossError = ConversionLossError

__all__ = [
    "__version__", "FORMAT_ID", "FORMAT_VERSION", "SCHEMA", "MIMETYPE", "FILE_EXTENSION", "AXIS_UP", "HANDEDNESS",
    "SGEODocument", "SGEOPackage", "SaveResult", "PackageInfo", "ValidationReport", "ConversionReport", "Issue",
    "new", "open_document", "read_package", "inspect", "validate", "capabilities",
    "SpatialFormatError", "ValidationError", "SchemaError", "SpatialSchemaError", "SpatialReferenceError",
    "GeometryValidationError", "SpatialGeometryValidationError", "IntegrityError", "SpatialIntegrityError",
    "ResourceLimitError", "SecurityLimitError", "SpatialSecurityLimitError", "UnsupportedFeatureError",
    "SpatialUnsupportedFeature", "DependencyUnavailableError", "ConflictError", "ReadOnlyDocumentError",
    "MigrationError", "ConversionLossError", "SpatialConversionLossError", "error_for_code",
]
