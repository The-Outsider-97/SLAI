"""

"""
from __future__ import annotations

__version__ = "2.3.0"


from ..utils.spatial_errors import *
from ..utils.spatial_helpers import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Geometry")
printer = PrettyPrinter()


__all__ = []


if __name__ == "__main__":
    configure_logging()
    print("\n=== Running Geometry ===\n")
    printer.status("TEST", "Geometry initialized", "info")