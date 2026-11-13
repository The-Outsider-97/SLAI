"""Public graph layer of the SLAI Software subsystem."""

from __future__ import annotations

__version__ = "2.4.0"

from .base_graph import *
from .dependency_graph import *
from .call_graph import *


from .base_graph import __all__ as _base_graph_exports
from .dependency_graph import __all__ as _dependency_graph_exports
from .call_graph import __all__ as _call_graph_exports


__all__ = [
    *_base_graph_exports,
    *_dependency_graph_exports,
    *_call_graph_exports,
] # type: ignore
