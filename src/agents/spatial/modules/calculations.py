"""
"""

from __future__ import annotations


from ...base.modules.math_science import *
from ...base.modules.physics_constraints import *
from ..utils.config_loader import (
    load_global_config as load_config,
    get_config_section as get_section
    )
from ..utils.spatial_errors import *
from ..utils.spatial_helpers import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Calculations")
printer = PrettyPrinter()


class Calculations:
    def __init__(self) -> None:
        self.config = load_config()
        self.calc_config = get_section("calculations", config=self.config, default={})


__all__ = ["Calculations"]


if __name__ == "__main__":
    configure_logging()
    print("\n=== Running Calculations ===\n")
    printer.status("TEST", "Calculations initialized", "info")