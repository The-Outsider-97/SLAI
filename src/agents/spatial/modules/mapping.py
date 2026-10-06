"""
"""

from __future__ import annotations

from ..utils.spatial_errors import *
from ..utils.spatial_helpers import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Mapping")
printer = PrettyPrinter()


__all__ = []


if __name__ == "__main__":
    configure_logging()
    print("\n=== Running Mapping ===\n")
    printer.status("TEST", "Mapping initialized", "info")