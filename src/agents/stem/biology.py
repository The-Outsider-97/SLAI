"""

Biology is computational rather than interpretive, and it owns:
   - deterministic population-model equations;
   - biochemical reaction-rate equations;
   - enzyme kinetics;
   - compartment models;
   - equilibrium calculations;
   - growth/decay calculations;
   - deterministic gene/network equations;
   - model residuals;
   - rate/Jacobian evaluation;
   - biological parameter conversions.

It defines and evaluate an ODE/PDE biological model,
but the actual general ODE algorithm belongs to math/numerical_methods.py;
scenario evolution belongs to SimulationAgent.   

Sources:
- Edelstein-Keshet, L. (2005). Mathematical Models in Biology. SIAM. DOI 10.1137/1.9780898719147.
- Segel, L. A., & Edelstein-Keshet, L. (2013). A Primer on Mathematical Models in Biology. SIAM. DOI 10.1137/1.9781611972504.

Boundary:
Biology
    evaluates biological mathematics

Reasoning
    interprets biological evidence

Simulation
    evolves a biological scenario

Optimization
    searches for optimal parameters/interventions
"""

from __future__ import annotations

__version__ = "2.3.0"

from typing import Any, Mapping, Optional, Dict

from ..base.modules.biology_constraints import *
from .utils.config_loader import load_global_config, get_config_section
from .utils.stem_errors import *
from .utils.stem_helpers import *
from .math.numerical_methods import *
from .units.unit_system import *
from .stem_memory import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("SLAI Biology")
printer = PrettyPrinter()


class Biology:
    def __init__(self, config: Mapping[str, Any] | None = None, memory: Optional[STEMMemory] = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.biology_config = dict(get_config_section("stem_biology", config=self.config) or {})
        if config:
            self.biology_config.update(dict(config))

        self.memory=memory

    def biological_models(self):
        """"Mathematical Models in Biology""""


__all__ = ["Biology"]