"""

sources:
- Dantzig, G. B. (1963). Linear Programming and Extensions. Princeton University Press. - Foundation for simplex and classical LP.
- Karmarkar, N. (1984). A new polynomial-time algorithm for linear programming. Combinatorica, 4, 373–395. - Foundation for modern interior-point LP methodology.
- Boyd & Vandenberghe (2004). - Useful for primal/dual LP, convexity and interior-point context.
- Land, A. H., & Doig, A. G. (1960). An automatic method of solving discrete programming problems. Econometrica, 28(3), 497–520. - This is the foundational branch-and-bound reference.
- Gomory, R. E. (1958). Outline of an algorithm for integer solutions to linear programs. Bulletin of the American Mathematical Society, 64(5), 275–278. DOI 10.1090/S0002-9904-1958-10224-4.
- Achterberg, T. (2009). SCIP: solving constraint integer programs. Mathematical Programming Computation, 1, 1–41. DOI 10.1007/s12532-008-0001-1.

Together:
LP relaxation
     ↓
branching
     ↓
bounding
     ↓
cuts
     ↓
incumbent
     ↓
optimality gap
forms the conceptual basis of modern MILP.

linear solver is capable of abstracting:
    simplex
    dual simplex
    interior point
rather than implementing only one LP method.


"""
from __future__ import annotations

__version__ = "2.3.0"

import math
import uuid

from typing import Any, Dict, Mapping, Optional

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.optimization_errors import *
from ..utils.optimization_helpers import *
from .base_solver import BaseSolver
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Linear Optimization Solver")
printer = PrettyPrinter()


class LinearSolver(BaseSolver):
    """Linear Programming"""
    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        self.config: Dict[str, Any] = load_global_config()
        self.ls_config = dict(get_config_section("linear_solver", config=self.config) or {})
        if config:
            self.ls_config.update(dict(config))


__all__ = ["LinearSolver"]