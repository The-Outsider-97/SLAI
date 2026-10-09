"""

sources:
- Nocedal & Wright (2006).
- Rossi, van Beek, & Walsh (2006), Handbook of Constraint Programming.

it should inform:
- variable domains
- constraint propagation
- domain filtering
- arc consistency
- global constraints
- backtracking
- branching
- constraint satisfaction
- constraint optimization
This should sit logically beside mathematical programming, not beneath Planning.
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

logger = get_logger("Non-Linear Optimization Solver")
printer = PrettyPrinter()


class NonLinearSolver:
    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.nls_config = dict(get_config_section("non_linear_solver", config=self.config) or {})
        if config:
            self.nls_config.update(dict(config))


__all__ = ["NonLinearSolver"]