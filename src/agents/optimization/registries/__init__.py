from .base_registry import *
from .capability_registry import *
from .scalarization_registry import *
from .solver_registry import *


from .base_registry import __all__ as _base_registry_exports
from .capability_registry import __all__ as _capability_registry_exports
from .scalarization_registry import __all__ as _scalarization_registry_exports
from .solver_registry import __all__ as _solver_registry_exports


__all__ = [
    *_base_registry_exports,
    *_capability_registry_exports,
    *_scalarization_registry_exports,
    *_solver_registry_exports,
] # type: ignore