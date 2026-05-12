from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any, Dict, Tuple

import yaml # type: ignore

from .data_error import DataConfigError

CONFIG_PATH = "configs/data_config.yaml"

# Deterministic cache strategy:
# cache key is the config file identity (path + (mtime_ns, size)).
# If identity changes, the YAML is reloaded exactly once and the cache is updated.
# This keeps repeated reads fast while making reload behavior predictable.
_CACHE_LOCK = RLock()
_global_config: Dict[str, Any] | None = None
_global_config_signature: Tuple[int, int] | None = None
_global_config_resolved_path: Path | None = None


def _resolve_config_path() -> Path:
    return (Path(__file__).parent.parent / CONFIG_PATH).resolve()


def _build_signature(config_path: Path) -> Tuple[int, int]:
    stat_result = config_path.stat()
    return (stat_result.st_mtime_ns, stat_result.st_size)


def clear_global_config_cache() -> None:
    """Clear the in-memory configuration cache.

    Use this in tests or explicit hot-reload flows.
    """
    global _global_config, _global_config_signature, _global_config_resolved_path
    with _CACHE_LOCK:
        _global_config = None
        _global_config_signature = None
        _global_config_resolved_path = None


def load_global_config(*, force_reload: bool = False) -> Dict[str, Any]:
    global _global_config, _global_config_signature, _global_config_resolved_path

    config_path = _resolve_config_path()
    if not config_path.exists():
        raise DataConfigError(
            "Data config file not found",
            context={"config_path": str(config_path)},
        )

    with _CACHE_LOCK:
        current_signature = _build_signature(config_path)
        cache_is_stale = (
            _global_config is None
            or _global_config_signature != current_signature
            or _global_config_resolved_path != config_path
        )

        if force_reload or cache_is_stale:
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f) or {}
            except yaml.YAMLError as exc:
                raise DataConfigError(
                    "Failed to parse data config YAML",
                    context={"config_path": str(config_path), "error": str(exc)},
                ) from exc

            if not isinstance(loaded, dict):
                raise DataConfigError(
                    "Top-level YAML config must be a mapping",
                    context={"config_path": str(config_path), "type": type(loaded).__name__},
                )

            loaded["__config_path__"] = str(config_path)
            _global_config = loaded
            _global_config_signature = current_signature
            _global_config_resolved_path = config_path

        return _global_config


def get_config_section(section_name: str) -> Dict[str, Any]:
    config = load_global_config()
    section = config.get(section_name)
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise DataConfigError(
            "Requested config section is not a mapping",
            context={"section": section_name, "type": type(section).__name__},
        )
    return section
