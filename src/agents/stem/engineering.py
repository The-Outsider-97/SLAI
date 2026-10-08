"""

Ownership
mechanics calculations;
structural quantities;
fluid/thermal calculations;
electrical engineering formulae;
engineering constitutive relations;
continuum-equation evaluation;
discretized engineering systems;
engineering-specific input/output transformations.

sources:
- Schäfer, M. (2022). Computational Engineering – Introduction to Numerical Methods (2nd ed.). Springer. DOI 10.1007/978-3-030-76027-4.
- Quarteroni, A., Sacco, R., & Saleri, F. (2007). Numerical Mathematics. Springer. DOI 10.1007/978-0-387-22750-4.
"""

from __future__ import annotations

from typing import Any, Mapping

__version__ = "2.3.0"

from ..base.modules.biology_constraints import *
from ..base.modules.chemistry_constraints import *
from ..base.modules.base_engineering import *
from ..base.modules.math_science import *
from .utils.config_loader import load_global_config, get_config_section
from .utils.stem_errors import *
from .utils.stem_helpers import *
from .units.unit_system import *
from .stem_memory import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("SLAI Engineering")
printer = PrettyPrinter()

@dataclass(frozen=True)
class EngineeringQuantity:
    """conforms to the BIPM SI system and ISO quantity/unit definitions."""


class Engineering(EngineeringEngine):
    """deterministic engineering equations and computational models."""
    def __init__(self, config: Mapping[str, Any] | None = None, memory: Optional[STEMMemory] = None) -> None:
        super().__init__(config)
        self.config: Dict[str, Any] = load_global_config()
        self.engineering_config = dict(get_config_section("stem_engineering", config=self.config) or {})
        if config:
            self.engineering_config.update(dict(config))

        self.memory=memory

__all__ = ["Engineering"]