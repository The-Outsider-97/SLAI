"""
This module owns optimization over discrete structured spaces rather than Planning's choice of actions.

sources:
- Land & Doig for branch-and-bound; 
- Gomory for cuts; 
- Rossi et al. for constraint-programming search; 
- Achterberg for practical hybrid discrete solver architecture.
"""
from __future__ import annotations

__version__ = "2.3.0"

import math
import uuid

from typing import Any, Dict, Mapping, Optional

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.optimization_errors import *
from ..utils.optimization_helpers import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Combinatorial Optimization Solver")
printer = PrettyPrinter()


class CombinatorialSolver:
    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.cs_config = dict(get_config_section("combinatorial_solver", config=self.config) or {})
        if config:
            self.cs_config.update(dict(config))


__all__ = ["CombinatorialSolver"]