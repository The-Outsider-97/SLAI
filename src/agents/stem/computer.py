"""

"""

from __future__ import annotations

__version__ = "2.3.0"

from typing import Any, Mapping, Optional, Dict

from ..base.modules.math_science import *
from .utils.config_loader import load_global_config, get_config_section
from .utils.stem_errors import *
from .utils.stem_helpers import *
from .stem_memory import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("SLAI Computing")
printer = PrettyPrinter()


class Computing:
    def __init__(self, config: Mapping[str, Any] | None = None, memory: Optional[STEMMemory] = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.computing_config = dict(get_config_section("stem_computing", config=self.config) or {})
        if config:
            self.computing_config.update(dict(config))

        self.memory=memory

__all__ = ["Computing"]