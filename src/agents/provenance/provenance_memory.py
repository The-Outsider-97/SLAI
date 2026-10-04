from __future__ import annotations

from typing import Any, Optional

from .utils.config_loader import load_global_config, get_config_section
from .utils.provenance_errors import *
from .utils.provenance_helpers import *
from .provenance_types import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Provenance Memory")
printer = PrettyPrinter()

class ProvenanceMemory:
    CHECKPOINT_PREFIX = "checkpoint"
    MANIFEST_FILE = "checkpoints_manifest.json"

    def __init__(self):
        self.config = load_global_config()
        self.memory_config = get_config_section('provenance_memory')



__all__ = ["ProvenanceMemory"]

if __name__ == "__main__":
    pass