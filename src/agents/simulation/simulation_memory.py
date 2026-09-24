"""
Simulation Memory preserves simulation-specific reproducibility information:
- run_id
- parent_run_id
- scenario_id
- model version
- initial state
- seed
- RNG state
- actions supplied
- interventions supplied
- parameters
- environment version
- termination condition
- trajectory references
- timestamps
 
Sources:
- Sandve, G. K., Nekrutenko, A., Taylor, J., & Hovig, E. (2013). Ten Simple Rules for Reproducible Computational Research. PLOS Computational Biology, 9(10), e1003285. DOI: 10.1371/journal.pcbi.1003285.
"""

from __future__ import annotations

__version__ = "2.3.0"

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from .utils.config_loader import *
from .utils.simulation_errors import *
from .utils.simulation_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Simulation Memory")
printer = PrettyPrinter()


class SimulationMemory:
    def __int__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = load_global_config()
        self.mem_config: Dict[str, Any] = dict(get_config_section("simulation_memory") or {})
        if config:
            self.mem_config.update(dict(config))


__all__ = [
    "SimulationMemory",
]