from __future__ import annotations

__version__ = "2.3.0"

"""
Centralized helper functions for Spatial workflows.

This module is intentionally broad so every layer in the Spatial subsystem can
reuse common operations with consistent semantics.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple, Union

from .spatial_errors import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Spatial Helpers")
printer = PrettyPrinter()

__all__ = []