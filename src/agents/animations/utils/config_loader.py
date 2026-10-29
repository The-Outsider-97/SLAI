"""Animation configuration facade using SLAI's shared YAML infrastructure.

Keeps the historical Animation config helpers, but does not reimplement YAML
parsing, duplicate-key checks, caching, or lock management. The settings live
exclusively in ``animations/configs/animation_config.yaml``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.utils.configuration import DEFAULT_CACHE_TTL_SECONDS, bind_config # type: ignore
from logs.logger import get_logger # type: ignore

logger = get_logger("Animation Config Loader")


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "animation_config.yaml"
_CONFIG_BINDING = bind_config(DEFAULT_CONFIG_PATH)


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    selected = DEFAULT_CONFIG_PATH if config_path is None else Path(config_path)
    return _CONFIG_BINDING.repository.resolve(selected)


def load_global_config(
    config_path: str | Path | None = None,
    *,
    force_reload: bool = False,
    cache_ttl: float = DEFAULT_CACHE_TTL_SECONDS,
) -> dict[str, Any]:
    """Use the shared, thread-safe, duplicate-key-safe configuration loader."""
    resolved = resolve_config_path(config_path)
    data = _CONFIG_BINDING.load(
        resolved, force_reload=force_reload, cache_ttl=cache_ttl,
    )
    # Compatibility with the previous Animation loader's diagnostics field.
    # This is added to the returned copy, never to the shared repository cache.
    data["__config_path__"] = str(resolved)
    return data


def reload_config(config_path: str | Path | None = None) -> dict[str, Any]:
    resolved = resolve_config_path(config_path)
    data = _CONFIG_BINDING.reload(resolved)
    logger.info("animation configuration reloaded | path=%s", resolved)
    data["__config_path__"] = str(resolved)
    return data


def clear_config_cache() -> None:
    _CONFIG_BINDING.clear()
    logger.debug("animation configuration cache cleared")


def get_config_cache_info() -> dict[str, Any]:
    return _CONFIG_BINDING.cache_info()


def get_config_section(
    section_name: str,
    config: Mapping[str, Any] | None = None,
    *,
    config_path: str | Path | None = None,
    default: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(section_name, str) or not section_name.strip():
        raise ValueError("section_name must be a non-empty string")
    return _CONFIG_BINDING.section(
        section_name.strip(), config=config, config_path=config_path, default=default,
    )


__all__ = [
    "DEFAULT_CACHE_TTL_SECONDS", "DEFAULT_CONFIG_PATH", "resolve_config_path",
    "load_global_config", "reload_config", "clear_config_cache",
    "get_config_cache_info", "get_config_section",
]
