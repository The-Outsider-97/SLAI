"""Analysis and transformation algorithms of the SLAI Software subsystem."""

from __future__ import annotations

__version__ = "2.4.0"

from .parser import *
from .static_analysis import *
from .dead_code import *
from .change_impact import *
from .architecture import *
from .patch_builder import *
from .refactoring import *
from .api_migration import *


from .parser import __all__ as _parser_exports
from .static_analysis import __all__ as _static_analysis_exports
from .dead_code import __all__ as _dead_code_exports
from .change_impact import __all__ as _change_impact_exports
from .architecture import __all__ as _architecture_exports
from .patch_builder import __all__ as _patch_builder_exports
from .refactoring import __all__ as _refactoring_exports
from .api_migration import __all__ as _api_migration_exports


__all__ = [
    *_parser_exports,
    *_static_analysis_exports,
    *_dead_code_exports,
    *_change_impact_exports,
    *_architecture_exports,
    *_patch_builder_exports,
    *_refactoring_exports,
    *_api_migration_exports,
] # type: ignore
