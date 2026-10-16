"""
Perception subsystem configuration facade.

The configuration repository is maintained centrally by
src.utils.configuration.

This module preserves the legacy Perception configuration API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.utils.configuration import bind_config # type: ignore


DEFAULT_CONFIG_PATH = ("perception/configs/perception_config.yaml")
DEFAULT_CACHE_TTL_SECONDS = 60.0
FILE_WATCH_INTERVAL_SECONDS = 5.0


_BINDING = bind_config(
    Path(__file__).resolve().parents[2]
    / DEFAULT_CONFIG_PATH
)


def _resolve_config_path(config_path: str | Path | None = None) -> Path:
    if config_path is None:
        return _BINDING.default_path.resolve()

    return Path(config_path).expanduser().resolve()


def load_global_config(
    config_path: str | Path | None = None,
    *,
    force_reload: bool = False,
    cache_ttl: float = DEFAULT_CACHE_TTL_SECONDS,
) -> dict[str, Any]:
    return _BINDING.load(config_path, force_reload=force_reload, cache_ttl=cache_ttl)


def reload_config(config_path: str | Path | None = None) -> dict[str, Any]:
    return _BINDING.reload(config_path)


def clear_config_cache() -> None:
    _BINDING.clear()


def get_config_cache_info() -> dict[str, Any]:
    return _BINDING.cache_info()


def config_cache_info() -> dict[str, Any]:
    return get_config_cache_info()


def get_config_section(
    section_name: str,
    config: Mapping[str, Any] | None = None,
    *,
    config_path: str | Path | None = None,
    default: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _BINDING.section(section_name, config=config, config_path=config_path, default=default)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_CACHE_TTL_SECONDS",
    "FILE_WATCH_INTERVAL_SECONDS",
    "load_global_config",
    "reload_config",
    "clear_config_cache",
    "get_config_cache_info",
    "config_cache_info",
    "get_config_section",
]
