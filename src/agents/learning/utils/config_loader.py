"""
Compatibility facade for the SLAI Learning subsystem configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from src.utils.configuration import DEFAULT_CACHE_TTL_SECONDS, bind_config


DEFAULT_CONFIG_PATH = (Path(__file__).resolve().parent.parent
    / "configs"
    / "learning_config.yaml")

# Retained for backward compatibility with imports from older learning modules.
FILE_WATCH_INTERVAL_SECONDS = 5
_BINDING = bind_config(DEFAULT_CONFIG_PATH)


def load_global_config(
    config_path: Optional[Union[str, Path]] = None,
    *,
    force_reload: bool = False,
    cache_ttl: float = DEFAULT_CACHE_TTL_SECONDS,
) -> Dict[str, Any]:
    """Load Learning configuration through SLAI's shared config repository."""
    return _BINDING.load(config_path, force_reload=force_reload, cache_ttl=cache_ttl)


def reload_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Force a reload of the Learning configuration."""
    return _BINDING.reload(config_path)


def clear_config_cache() -> None:
    """Clear cached Learning configuration entries."""
    _BINDING.clear()


def get_config_cache_info() -> Dict[str, Any]:
    """Return cache diagnostics for the active Learning config."""
    return _BINDING.cache_info()


def get_config_section(
    section_name: str,
    config: Optional[Mapping[str, Any]] = None,
    *,
    config_path: Optional[Union[str, Path]] = None,
    default: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a detached Learning configuration section."""
    return _BINDING.section(section_name, config=config, config_path=config_path, default=default)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "_BINDING",
    "load_global_config",
    "reload_config",
    "clear_config_cache",
    "get_config_cache_info",
    "get_config_section",
]
