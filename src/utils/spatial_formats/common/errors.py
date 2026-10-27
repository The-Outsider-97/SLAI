"""Typed, stable failure categories for native spatial documents.

Every error exposes a stable machine-readable :attr:`code` and optional
:attr:`path` / :attr:`context` diagnostics. Clients should dispatch on
``code``; human-readable messages are advisory and may be reworded in a
minor release.

Hierarchy::

    SpatialFormatError (ValueError)
      ├── ValidationError
      │     ├── SchemaError
      │     ├── ReferenceError
      │     └── GeometryValidationError
      ├── IntegrityError
      ├── ResourceLimitError
      │     └── SecurityLimitError
      ├── UnsupportedFeatureError
      │     └── DependencyUnavailableError
      ├── ConflictError
      │     └── ReadOnlyDocumentError
      ├── MigrationError
      └── ConversionLossError

Every class is also available under the ``Spatial*`` names used by the
SDRW/SGEO feasibility report (``SpatialSchemaError``, ``SpatialReferenceError``,
…). Prefer those names in application code: ``ReferenceError`` would shadow
the Python builtin of the same name. :func:`error_for_code` maps a stable
``code`` back to its class, which is how a serialized
:class:`~src.utils.spatial_formats.verification.report.ValidationReport` is
turned into a real exception again.

Pickling preserves ``args`` and the diagnostic attributes; equality
remains ``BaseException`` equality (identity based) — callers should
compare ``code`` and ``context`` explicitly, not exception objects.
"""


class SpatialFormatError(ValueError):
    """Base class for invalid or unsupported spatial artifacts."""

    code: str = "spatial"
    default_message: str = "invalid spatial artifact"

    def __init__(self, message=None, *, path=None, context=None):
        if message is None:
            message = self.default_message
        super().__init__(message)
        self.path = path
        self.context = context

    def __str__(self) -> str:
        message = self.args[0] if self.args else self.default_message
        if self.path:
            return f"{self.path}: {message}"
        return message

    def __reduce__(self):
        # BaseException.__reduce__ alone would lose path/context; keep both.
        return (type(self), self.args, dict(self.__dict__))

    def to_dict(self) -> dict:
        """Structured, JSON-safe diagnostics for logging or IPC."""
        return {
            "code": self.code,
            "message": self.args[0] if self.args else self.default_message,
            "path": self.path,
            "context": self.context,
        }

    def with_path(self, path: str) -> "SpatialFormatError":
        """Return a copy of this error bound to ``path``."""
        clone = type(self)(*(self.args or (self.default_message,)),
                           path=path, context=self.context)
        return clone

    def with_context(self, **context) -> "SpatialFormatError":
        """Return a copy of this error with merged structured context."""
        merged = dict(self.context or {})
        merged.update(context)
        return type(self)(*(self.args or (self.default_message,)),
                          path=self.path, context=merged)


class ValidationError(SpatialFormatError):
    """A document or geometry violates a mandatory invariant."""

    code = "spatial.validation"
    default_message = "spatial document failed validation"


class SchemaError(ValidationError):
    """A persisted document does not conform to its declared JSON schema.

    Distinct from :class:`ValidationError` only for dispatching; schema
    violations may be repairable by migration while semantic violations
    (reference, geometry) usually are not.
    """

    code = "spatial.schema"
    default_message = "spatial document violates schema"


class ReferenceError(ValidationError):
    """An object, family, layer, asset or claim references a missing ID.

    Named to match the feasibility report's ``SpatialReferenceError``;
    inherits ``ValidationError`` so existing ``except ValidationError``
    blocks continue to catch it.
    """

    code = "spatial.reference"
    default_message = "broken spatial reference"


class GeometryValidationError(ValidationError):
    """Geometry violates a numeric, topological or winding invariant."""

    code = "spatial.geometry"
    default_message = "invalid spatial geometry"


class IntegrityError(SpatialFormatError):
    """Container member metadata or content does not match."""

    code = "spatial.integrity"
    default_message = "spatial container integrity failure"


class ResourceLimitError(SpatialFormatError):
    """An artifact exceeds configured processing limits."""

    code = "spatial.limit"
    default_message = "spatial resource budget exceeded"


class SecurityLimitError(ResourceLimitError):
    """A limit was crossed specifically by adversarial or unsafe input.

    Examples: decompression bombs, symlink members, duplicate members,
    oversized header records. Inherits ``ResourceLimitError`` so trusted
    callers can downgrade to a warning without rewriting the ``except``
    clause, while security-aware callers can distinguish.
    """

    code = "spatial.security"
    default_message = "spatial input rejected for safety"


class UnsupportedFeatureError(SpatialFormatError):
    """A required capability or interchange construct is unsupported."""

    code = "spatial.unsupported"
    default_message = "unsupported spatial feature"


class DependencyUnavailableError(UnsupportedFeatureError):
    """A deliberately optional interchange backend is not installed."""

    code = "spatial.dependency"
    default_message = "optional spatial backend unavailable"


class ConflictError(SpatialFormatError):
    """A write conflicts with an existing artifact or revision.

    Suggested ``context`` keys: ``expected_revision``, ``actual_revision``,
    ``path`` (destination). Callers performing optimistic concurrency
    should compare against their own captured revision rather than parse
    the message.
    """

    code = "spatial.conflict"
    default_message = "spatial write conflict"


class ReadOnlyDocumentError(ConflictError):
    """A mutation was attempted on a document opened read-only or already closed.

    Inherits :class:`ConflictError` so code that treats "this write cannot
    happen right now" uniformly keeps working.
    """

    code = "spatial.read_only"
    default_message = "document is read-only or closed"


class MigrationError(SpatialFormatError):
    """A source document cannot be migrated without semantic loss.

    Suggested ``context`` keys: ``from_version``, ``to_version``, and
    optional ``losses`` (list of human-readable differences).
    """

    code = "spatial.migration"
    default_message = "spatial migration failed"


class ConversionLossError(SpatialFormatError):
    """A lossy import/export exceeded the caller's declared tolerance.

    Adapters that emit a ``ConversionReport`` raise this when the report's
    ``accuracy_class`` or ``dropped`` list is not accepted by the caller.
    """

    code = "spatial.loss"
    default_message = "spatial conversion exceeded permitted loss"


# ---------------------------------------------------------------------------
# Report-compatible names and code lookup
# ---------------------------------------------------------------------------

SpatialValidationError = ValidationError
SpatialSchemaError = SchemaError
SpatialReferenceError = ReferenceError
SpatialGeometryValidationError = GeometryValidationError
SpatialIntegrityError = IntegrityError
SpatialResourceLimitError = ResourceLimitError
SpatialSecurityLimitError = SecurityLimitError
SpatialUnsupportedFeature = UnsupportedFeatureError
SpatialDependencyError = DependencyUnavailableError
SpatialConflictError = ConflictError
SpatialReadOnlyError = ReadOnlyDocumentError
SpatialMigrationError = MigrationError
SpatialConversionLossError = ConversionLossError

_BY_CODE = {
    cls.code: cls
    for cls in (
        SpatialFormatError, ValidationError, SchemaError, ReferenceError,
        GeometryValidationError, IntegrityError, ResourceLimitError,
        SecurityLimitError, UnsupportedFeatureError, DependencyUnavailableError,
        ConflictError, ReadOnlyDocumentError, MigrationError, ConversionLossError,
    )
}


def error_for_code(code: str) -> type:
    """Return the exception class registered for a stable ``code``.

    Unknown codes fall back to :class:`SpatialFormatError`, so a report
    written by a newer release can still be raised by an older reader.
    """
    return _BY_CODE.get(code, SpatialFormatError)


__all__ = [
    "SpatialFormatError",
    "ValidationError",
    "SchemaError",
    "ReferenceError",
    "GeometryValidationError",
    "IntegrityError",
    "ResourceLimitError",
    "SecurityLimitError",
    "UnsupportedFeatureError",
    "DependencyUnavailableError",
    "ConflictError",
    "ReadOnlyDocumentError",
    "MigrationError",
    "ConversionLossError",
    "SpatialValidationError",
    "SpatialSchemaError",
    "SpatialReferenceError",
    "SpatialGeometryValidationError",
    "SpatialIntegrityError",
    "SpatialResourceLimitError",
    "SpatialSecurityLimitError",
    "SpatialUnsupportedFeature",
    "SpatialDependencyError",
    "SpatialConflictError",
    "SpatialReadOnlyError",
    "SpatialMigrationError",
    "SpatialConversionLossError",
    "error_for_code",
]
