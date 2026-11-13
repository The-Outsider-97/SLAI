"""Public semantic-analysis layer (symbols, hierarchy, search, compatibility)."""

from __future__ import annotations

__version__ = "2.4.0"

from .symbol_index import *
from .class_hierarchy import *
from .code_search import *
from .interface_compatibility import *


from .symbol_index import __all__ as _symbol_index_exports
from .class_hierarchy import __all__ as _class_hierarchy_exports
from .code_search import __all__ as _code_search_exports
from .interface_compatibility import __all__ as _interface_compatibility_exports


__all__ = [
    *_symbol_index_exports,
    *_class_hierarchy_exports,
    *_code_search_exports,
    *_interface_compatibility_exports,
] # type: ignore
