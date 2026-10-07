"""

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
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("SLAI Physics")
printer = PrettyPrinter()


class Physics(PhysicsEngine):
    def __init__(self, config: Mapping[str, Any] | None = None):
        super().__init__(config)

__all__ = []