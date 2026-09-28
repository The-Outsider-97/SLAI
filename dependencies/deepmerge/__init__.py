"""SLAI-native hierarchical mapping merge semantics.

This package replaces the narrow ``deepmerge`` usage required by SLAI's
configuration loader. It intentionally does not reproduce the upstream
strategy-registration API because SLAI does not use it.
"""

from .mapping import merge_mappings

__all__ = ["merge_mappings"]
