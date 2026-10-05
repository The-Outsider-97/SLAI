"""


sources:
- Sandve et al. (2013), Ten Simple Rules for Reproducible Computational Research.
- Chirigati et al. (2016/2017), ReproZip: Computational Reproducibility With Ease.
- Vartak et al. (2016), ModelDB: A System for Machine Learning Model Management.

For SLAI, ProvenanceMemory should be concerned with:

checkpoint A
   │
   ├── agent/model version
   ├── configuration identity
   ├── source artifacts
   ├── dataset version
   ├── code/version identity
   └── parent checkpoint
not general conversational or semantic memory.
"""

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