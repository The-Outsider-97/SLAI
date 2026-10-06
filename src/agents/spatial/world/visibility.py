"""
sources:
- de Berg et al. has an explicit treatment of visibility graphs and the underlying computational geometry.

Visibility owns:
portals
frusta
occlusion structures
ray acceleration
view cells
"""
from __future__ import annotations

__version__ = "2.3.0"


from ..utils.spatial_errors import *
from ..utils.spatial_helpers import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Visibility")
printer = PrettyPrinter()


__all__ = []


if __name__ == "__main__":
    configure_logging()
    print("\n=== Running Visibility ===\n")
    printer.status("TEST", "Visibility initialized", "info")