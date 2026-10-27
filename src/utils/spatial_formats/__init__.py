"""SLAI v2.4 native spatial formats: shared, CPU-only, versioned components.

Internal serialization and interchange infrastructure for .sgeo / .sdrw.
This package intentionally does not import the application-level public facades
(``src/utils/sdrw.py``, ``src/utils/sgeo.py``); those import this package.
"""
from .sdrw import Drawing
from .sgeo import Scene
from .sdrw.document import DrawingDocument
from .sgeo.document import SceneDocument
from .documents import SaveResult, SpatialDocument
from .container import (SpatialPackage, load_package, loads_package, write_package, write_document,
                        dumps_package, inspect_package, inspect_bytes, PackageInfo)
from .verification import (ConversionReport, ValidationReport, Issue, validate_document, check_drawing,
                           check_scene)
from .adapters import capabilities
from .common.errors import (SpatialFormatError, ValidationError, SchemaError, GeometryValidationError,
                            IntegrityError, ResourceLimitError, SecurityLimitError, UnsupportedFeatureError,
                            DependencyUnavailableError, ConflictError, ReadOnlyDocumentError, MigrationError,
                            ConversionLossError, error_for_code)

__all__ = ["Drawing", "Scene", "DrawingDocument", "SceneDocument", "SpatialDocument", "SaveResult",
           "SpatialPackage", "PackageInfo", "load_package", "loads_package", "write_package", "write_document",
           "dumps_package", "inspect_package", "inspect_bytes", "ConversionReport", "ValidationReport", "Issue",
           "validate_document", "check_drawing", "check_scene", "capabilities", "SpatialFormatError",
           "ValidationError", "SchemaError", "GeometryValidationError", "IntegrityError", "ResourceLimitError",
           "SecurityLimitError", "UnsupportedFeatureError", "DependencyUnavailableError", "ConflictError",
           "ReadOnlyDocumentError", "MigrationError", "ConversionLossError", "error_for_code"]
