"""
CAD drawings, plans, sections, dimensions, diagrams, and 2D elements.
Points, lines, polylines, circles, arcs, layers, coordinates, units, IDs, semantic metadata, and validated reading/writing.

"""
from __future__ import annotations

__version__ = "1.0.0"

from dataclasses import dataclass


FORMAT_ID = "slai.sdrw"
FORMAT_VERSION = "1.0.0"  # What new documents are written as.
SCHEMA = "slai.sdrw/1"  # Value of ``document["schema"]`` in 1.x documents.


class SDRWDocument:
    def __init__(self) -> None:
        pass


@dataclass
class SDRWPackage:
    """"""