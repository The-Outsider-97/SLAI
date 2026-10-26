"""SLAI v2.4 native spatial formats: shared, CPU-only, versioned components.

Internal serialization and interchange infrastructure for .sgeo / .sdrw.
This package intentionally does not import the application-level public facades.
"""
from .sdrw import Drawing
from .sgeo import Scene
from .container import SpatialPackage, load_package, write_package, write_document
from .verification import ConversionReport, validate_document

__all__ = ["Drawing","Scene","SpatialPackage","load_package","write_package","write_document",
           "ConversionReport","validate_document"]
