"""

Ownership:
- symbolic expressions;
- polynomial arithmetic;
- exact rational arithmetic;
- symbolic simplification;
- equation manipulation;
- systems of algebraic equations;
- polynomial roots where symbolically tractable;
- Gröbner bases;
- symbolic factorization;
- symbolic substitution.

sources:
- Geddes, K. O., Czapor, S. R., & Labahn, G. (1992). Algorithms for Computer Algebra. Springer. DOI 10.1007/b102438.
- Bronstein, M. (2005). Symbolic Integration I: Transcendental Functions (2nd ed.). Springer. DOI 10.1007/b138171.

A key architectural distinction is:
algebra.py:
    Solve x² - 5x + 6 = 0

ReasoningAgent:
    Determine which equation should model the problem

OptimizationAgent:
    Find x minimizing some objective
"""

from __future__ import annotations

__version__ = "2.3.0"

from typing import Any, Dict, Mapping

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.stem_errors import *
from ..utils.stem_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("Algebra")
printer = PrettyPrinter()


class Algebra:
    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.algebra_config = dict(get_config_section("algebra", config=self.config) or {})
        if config:
            self.algebra_config.update(dict(config))

__all__ = ["Algebra"]