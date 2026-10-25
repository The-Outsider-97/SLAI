"""
SLAI's 3D spatial-model format
SGEO = SLAI Geometry
Versatile enough for architectural BIM models, engineering assemblies, scientific geometry, and parametric families.
Supports family definitions, type parameters, instance overrides and deterministic regeneration.
"""
from __future__ import annotations

__version__ = "1.0.0"

from dataclasses import dataclass


FORMAT_ID = "slai.sgeo"
FORMAT_VERSION = "1.0.0"  # What new documents are written as.
SCHEMA = "slai.sgeo/1"  # Value of ``document["schema"]`` in 1.x documents.


class SGEODocument:
    def __init__(self) -> None:
        pass


@dataclass
class SGEOPackage:
    """"""