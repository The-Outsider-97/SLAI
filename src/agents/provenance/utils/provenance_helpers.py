from __future__ import annotations
from datetime import datetime

__version__ = "2.3.0"

"""
Centralized helper functions for provenance workflows.

This module is intentionally broad so every layer in the provenance subsystem can
reuse common operations with consistent semantics.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple, Union

from .provenance_errors import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("provenance Helpers")
printer = PrettyPrinter()


def get_current_timestamp(self):
    """
    Get the current timestamp in ISO 8601 format.
    """
    return datetime.utcnow().isoformat() + 'Z'

__all__ = ["get_current_timestamp"]