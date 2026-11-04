"""SLAI Software subsystem: program analysis and source-transformation proposals.

The public surface is the five central handlers, layered bottom-up:

``SoftwareModels`` → ``SoftwareGraphs`` → ``SoftwareSemantics`` →
``SoftwareAnalysis`` → ``SoftwareRefactoring``

plus the immutable data contracts in :mod:`software_types`.  The subsystem
reads and writes no files and starts no processes; its outputs are facts,
reports and *unapplied* patch proposals.
"""

from __future__ import annotations

__version__ = "2.4.0"

from .software_types import *
from .utils.software_errors import *
from .models import *
from .graphs import *
from .semantics import *
from .modules import *
from .software_models import *
from .software_graphs import *
from .software_semantics import *
from .software_analysis import *
from .software_refactoring import *


from .software_types import __all__ as _software_types_exports
from .utils.software_errors import __all__ as _software_error_exports
from .models import __all__ as _model_exports
from .graphs import __all__ as _graph_exports
from .semantics import __all__ as _semantic_exports
from .modules import __all__ as _module_exports
from .software_models import __all__ as _software_models_exports
from .software_graphs import __all__ as _software_graphs_exports
from .software_semantics import __all__ as _software_semantics_exports
from .software_analysis import __all__ as _software_analysis_exports
from .software_refactoring import __all__ as _software_refactoring_exports


__all__ = [
    *_software_types_exports,
    *_software_error_exports,
    *_model_exports,
    *_graph_exports,
    *_semantic_exports,
    *_module_exports,
    *_software_models_exports,
    *_software_graphs_exports,
    *_software_semantics_exports,
    *_software_analysis_exports,
    *_software_refactoring_exports,
] # type: ignore
