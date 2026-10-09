"""
Black-box optimization needs a particularly strict boundary from src/tuning/.

sources:
- Conn, A. R., Scheinberg, K., & Vicente, L. N. (2009). Introduction to Derivative-Free Optimization. SIAM. DOI 10.1137/1.9780898718768.
- Audet, C., & Dennis, J. E., Jr. (2006). Mesh Adaptive Direct Search Algorithms for Constrained Optimization. SIAM Journal on Optimization.
- Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient Global Optimization of Expensive Black-Box Functions. Journal of Global Optimization, 13, 455–492.

ownership boundary:
    OptimizationAgent black-box optimization:
        x = arbitrary decision variables
        f(x) = arbitrary objective supplied by caller

    src/tuning:
        x = SLAI/model hyperparameters
        f(x) = evaluation metric for candidate configuration
The mathematical algorithm may be related; the domain contract is not.


"""
from __future__ import annotations

__version__ = "2.3.0"

import math
import uuid

from typing import Any, Dict, Mapping, Optional

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.optimization_errors import *
from ..utils.optimization_helpers import *
from src.tuning.tuner import * # type: ignore
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Black-Box Optimization Solver")
printer = PrettyPrinter()


class BlackBoxSolver:
    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.bbs_config = dict(get_config_section("black_box_solver", config=self.config) or {})
        if config:
            self.bbs_config.update(dict(config))


__all__ = ["BlackBoxSolver"]