"""

Ownership
This module should provide:

deterministic physical laws/equations;
- physical constants;
- kinematics;
- dynamics;
- energy/momentum calculations;
- field/electromagnetic formulas;
- thermodynamic relationships;
- waves/optics relationships;
- formula evaluation;
- physical model construction.
- It shouldn't decide why a physical event happened—that is Reasoning—and shouldn't evolve an entire physical world—that is Simulation.

sources:
- Landau, R. H., Páez, M. J., & Bordeianu, C. C. (2007). Computational Physics: Problem Solving with Computers. Wiley-VCH. DOI 10.1002/9783527618835.
- Mohr, P. J., Newell, D. B., Taylor, B. N., & Tiesinga, E. (2025). “CODATA Recommended Values of the Fundamental Physical Constants: 2022.” Reviews of Modern Physics.
- BIPM SI Brochure, 9th ed., updated 2026 should govern units/constants representation.
"""

from __future__ import annotations

__version__ = "2.3.0"

from typing import Any, Mapping

from ..base.modules.biology_constraints import *
from ..base.modules.chemistry_constraints import *
from ..base.modules.physics_constraints import *
from ..base.modules.math_science import *
from .utils.config_loader import load_global_config, get_config_section
from .utils.stem_errors import *
from .utils.stem_helpers import *
from .units.unit_system import *
from .stem_memory import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("SLAI Physics")
printer = PrettyPrinter()


class Physics(PhysicsEngine):
    def __init__(self, config: Mapping[str, Any] | None = None, memory: Optional[STEMMemory] = None):
        super().__init__(config)
        self.config: Dict[str, Any] = load_global_config()
        self.physics_config = dict(get_config_section("stem_physics", config=self.config) or {})
        if config:
            self.physics_config.update(dict(config))

        self.memory = memory

__all__ = []