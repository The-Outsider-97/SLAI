"""

Suitable ownership
- discrete algorithms;
- combinatorics;
- deterministic graph algorithms;
- recurrence evaluation;
- exact integer operations;
- algorithmic complexity metadata;
- numerical/computational primitives;
- deterministic transformations.

It should not absorb:
- source-code analysis;
- code generation;
- refactoring;
- repository manipulation;
because those belongs to the SoftwareAgent.

sources:
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). Introduction to Algorithms (4th ed.). MIT Press.
- Higham (2002) for finite-precision numerical algorithms.
- IEEE 754-2019 for floating-point execution semantics.
"""

from __future__ import annotations

__version__ = "2.3.0"

from typing import Any, Mapping, Optional, Dict

from ..base.modules.math_science import *
from .utils.config_loader import load_global_config, get_config_section
from .utils.stem_errors import *
from .utils.stem_helpers import *
from .units.unit_system import *
from .stem_memory import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("SLAI Computing")
printer = PrettyPrinter()


class Computing:
    """the deterministic computer-science computation layer"""
    def __init__(self, config: Mapping[str, Any] | None = None, memory: Optional[STEMMemory] = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.computing_config = dict(get_config_section("stem_computing", config=self.config) or {})
        if config:
            self.computing_config.update(dict(config))

        self.memory=memory

__all__ = ["Computing"]