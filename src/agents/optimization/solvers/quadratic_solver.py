"""

sources:
- Boyd & Vandenberghe (2004) should be the main reference for convex QP.
- Nocedal & Wright (2006) should support numerical QP and its role inside SQP/trust-region methods.

QP is treated as its own problem class even if the code eventually dispatches QP through a general continuous-solver adapter.
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

logger = get_logger("Quadratic Optimization Solver")
printer = PrettyPrinter()


class QuadraticSolver:
    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.quadratic_config = dict(get_config_section("quadratic_solver", config=self.config) or {})
        if config:
            self.quadratic_config.update(dict(config))


__all__ = ["QuadraticSolver"]