"""Typed, stable failure categories for native spatial documents."""
class SpatialFormatError(ValueError):
    """Base class for invalid or unsupported spatial artifacts."""
class ValidationError(SpatialFormatError):
    """A document or geometry violates a mandatory invariant."""
class IntegrityError(SpatialFormatError):
    """Container member metadata or content does not match."""
class ResourceLimitError(SpatialFormatError):
    """An artifact exceeds configured processing limits."""
class UnsupportedFeatureError(SpatialFormatError):
    """A required capability or interchange construct is unsupported."""
class DependencyUnavailableError(UnsupportedFeatureError):
    """A deliberately optional interchange backend is not installed."""
class ConflictError(SpatialFormatError):
    """A write conflicts with an existing artifact or revision."""
class MigrationError(SpatialFormatError):
    """A source document cannot be migrated without semantic loss."""
