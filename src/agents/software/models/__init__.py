"""Public program-model layer of the SLAI Software subsystem."""

from __future__ import annotations

__version__ = "2.4.0"

from .base_model import *
from .source_model import *
from .repository_model import *


from .base_model import __all__ as _base_model_exports
from .source_model import __all__ as _source_model_exports
from .repository_model import __all__ as _repository_model_exports


__all__ = [
    *_base_model_exports,
    *_source_model_exports,
    *_repository_model_exports,
] # type: ignore
