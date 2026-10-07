"""
Memory module consumed by the stem submodule  for caching, storage, and logic
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