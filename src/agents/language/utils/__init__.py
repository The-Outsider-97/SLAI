from .config_loader import *
from .language_error import *
from .language_helpers import *


from .config_loader import __all__ as _config_loader_exports
from .language_error import __all__ as _language_error_exports
from .language_helpers import __all__ as _language_helpers_exports

# Spell checking needs lexical utilities, not PyTorch. Importing this package
# must not demand the neural model stack just to import SpellChecker. Preserve
# the legacy re-exports whenever the optional cache dependency is installed.
try:
    from .language_cache import *
    from .language_cache import __all__ as _language_cache_exports
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    _language_cache_exports = []


__all__ = [
    *_config_loader_exports,
    *_language_cache_exports,
    *_language_error_exports,
    *_language_helpers_exports,
] # type: ignore
