from __future__ import annotations

from typing import Any, Optional

from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Provenance Types")
printer = PrettyPrinter()

class ProvenanceTypes:
    pass

__all__ = ["ProvenanceTypes"]

if __name__ == "__main__":
    pass