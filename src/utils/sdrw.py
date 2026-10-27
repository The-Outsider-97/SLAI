"""
SDRW -- SLAI's native 2D drawing format (``.sdrw``).

CAD drawings, plans, sections, dimensions, diagrams and other 2D elements:
points, lines, polylines, circles, arcs, layers, coordinates, units, stable
ids, semantic metadata, and validated reading/writing.

This module is the small, stable public facade. The implementation lives in
``src.utils.spatial_formats`` (this module imports it, never the reverse).

Quick start::

    from src.utils.sdrw import SDRWDocument

    drawing = SDRWDocument.new("Plan A", unit="mm")
    wall = drawing.add_line((0, 0), (1500, 0), layer="walls", create_layer=True)
    drawing.attach_semantics(wall, {"category": "wall"}, namespace="aec")
    report = drawing.validate()            # structured ValidationReport, strict by default
    drawing.save("plan.sdrw")              # atomic, verified, refuses to overwrite by default

    with SDRWDocument.open("plan.sdrw") as plan:          # read-only by default
        ids = plan.query_ids(kind="line", layer="walls")

Conventions: planar, right-handed, lengths in the document unit, angles in
radians (``degrees=True`` where offered), ids are UUID strings, metadata keys
are ``namespace:name`` or ``slai.*``. Every edit is atomic and bumps the
revision exactly once; batches (``apply``) are one revision as a whole.
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
from .spatial_formats.sdrw.document import DrawingDocument
from .spatial_formats.verification import ConversionReport, Issue, ValidationReport, report_from_exception

__version__ = "0.2.0"

FORMAT_ID = "slai.sdrw"
FORMAT_VERSION = CURRENT_VERSION  # What new documents are written as.
SCHEMA = FORMAT_ID + "/" + ".".join(FORMAT_VERSION.split(".")[:2])  # Value of ``manifest["schema"]``.
MIMETYPE = MIMES[FORMAT_ID].decode("ascii")
FILE_EXTENSION = ".sdrw"


class SDRWDocument(DrawingDocument):
    """An editable, validated 2D drawing.

    Construction: :meth:`new`, :meth:`open`, :meth:`from_bytes`, :meth:`from_svg`,
    :meth:`import_svg`, :meth:`import_dxf`, :meth:`import_dwg`.
    Entities: ``add_point/line/polyline/polygon/rectangle/circle/arc/arc_through``.
    Edits (one revision each): ``update``, ``remove``, ``move``, ``rotate``, ``scale``,
    ``transform``, ``duplicate``, layer methods, ``attach_semantics``, ``add_claim``,
    ``convert_unit``, and ``apply`` for atomic batches.
    Queries: ``get``, ``query_ids``, ``find``, ``count``, ``bounds``, ``length``, ``area``,
    ``measure``, ``check_constraints``.
    Persistence: ``validate``, ``inspect``, ``save``, ``dumps``, ``content_digest``.
    Interchange: ``to_svg``/``export_svg``, ``export_dxf``, ``export_dwg`` (each returns a
    ``ConversionReport``).
    """

    __slots__ = ()


@dataclass(frozen=True)
class SDRWPackage:
    """A loaded, fully validated package at the container level (no editing API)."""

    manifest: dict
    document: dict
    sha256: str
    size: int
    path: str | None = None

    def to_document(self, *, mode: str = "write") -> SDRWDocument:
        """Build an editable (or read-only) document from this package."""
        from .spatial_formats.container.reader import SpatialPackage
        return SDRWDocument.from_package(SpatialPackage(self.manifest, self.document, {}), mode=mode)


def new(name: str = "Untitled", *, unit: str = "mm", layers=None, metadata=None, document_id=None) -> SDRWDocument:
    """Create an empty drawing (same as :meth:`SDRWDocument.new`)."""
    return SDRWDocument.new(name, unit=unit, layers=layers, metadata=metadata, document_id=document_id)


def open_document(path, *, mode: str = "read") -> SDRWDocument:
    """Open and validate a ``.sdrw`` file (``mode``: ``"read"`` or ``"write"``)."""
    return SDRWDocument.open(path, mode=mode)


def read_package(path) -> SDRWPackage:
    """Load a ``.sdrw`` file as a plain validated package (manifest + payload + digest)."""
    from hashlib import sha256
    data = read_source_bytes(path)
    package = loads_package(data)
    if package.manifest["format_id"] != FORMAT_ID:
        raise UnsupportedFeatureError("expected a %s package but found %s" % (FORMAT_ID, package.manifest["format_id"]))
    return SDRWPackage(package.manifest, package.document, sha256(data).hexdigest(), len(data), str(Path(path)))


def inspect(path, *, verify: str = "index", summary: bool = True) -> PackageInfo:
    """Describe a ``.sdrw`` file without loading its geometry (identity, units, parts, counts)."""
    info = inspect_package(path, verify=verify, summary=summary)
    if info.format_id != FORMAT_ID:
        raise UnsupportedFeatureError("expected a %s package but found %s" % (FORMAT_ID, info.format_id))
    return info


def validate(path, *, mode: str = "strict", max_issues: int = 200) -> ValidationReport:
    """Validate a ``.sdrw`` file and return a :class:`ValidationReport`.

    Problems loading the file (corruption, integrity, limits, wrong format) are
    reported as errors in the report rather than raised; OS errors such as a
    missing file still propagate.
    """
    try:
        return SDRWDocument.open(path, mode="read").validate(mode, max_issues=max_issues)
    except SpatialFormatError as exc:
        return report_from_exception(exc, document_kind="drawing", mode=mode)


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
    "__version__", "FORMAT_ID", "FORMAT_VERSION", "SCHEMA", "MIMETYPE", "FILE_EXTENSION",
    "SDRWDocument", "SDRWPackage", "SaveResult", "PackageInfo", "ValidationReport", "ConversionReport", "Issue",
    "new", "open_document", "read_package", "inspect", "validate", "capabilities",
    "SpatialFormatError", "ValidationError", "SchemaError", "SpatialSchemaError", "SpatialReferenceError",
    "GeometryValidationError", "SpatialGeometryValidationError", "IntegrityError", "SpatialIntegrityError",
    "ResourceLimitError", "SecurityLimitError", "SpatialSecurityLimitError", "UnsupportedFeatureError",
    "SpatialUnsupportedFeature", "DependencyUnavailableError", "ConflictError", "ReadOnlyDocumentError",
    "MigrationError", "ConversionLossError", "SpatialConversionLossError", "error_for_code",
]
