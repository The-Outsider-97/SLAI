"""
Memory module consumed by the stem submodules for scientific computation caching and reproducibility.

the caching identity should consider:
- function/model
- inputs
- units
- algorithm
- algorithm version
- precision
- tolerances
- boundary conditions
- constants version
- seed, if applicable

sources:
- Michie, D. (1968). “‘Memo’ Functions and Machine Learning.” Nature, 218, 19–22. DOI 10.1038/218019a0.
- Sandve, G. K., et al. (2013). “Ten Simple Rules for Reproducible Computational Research.” PLOS Computational Biology, 9(10), e1003285.
- Wilson et al. (2014) also supports reproducible and reliable scientific-software practice.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

__version__ = "2.3.0"

from .utils.config_loader import load_global_config, get_config_section
from .utils.stem_errors import *
from .utils.stem_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("STEM Memory")
printer = PrettyPrinter()


class STEMMemory:
    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.mem_config = dict(get_config_section("stem_memory", config=self.config) or {})
        if config:
            self.mem_config.update(dict(config))

__all__ = ["STEMMemory"]