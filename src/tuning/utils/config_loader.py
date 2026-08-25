"""Tuning configuration facade over SLAI's shared configuration repository.

This module owns only tuning-specific path resolution and error translation.
YAML parsing, duplicate-key detection, path-keyed caching, nanosecond file
change detection, monotonic TTLs, locking, and defensive deep copies remain in
``src.utils.configuration``.  There is therefore one configuration source of
truth for SLAI rather than a second cache inside the tuning package.
"""

from __future__ import annotations

import copy
import math

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .tuning_errors import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]
from src.utils.configuration import (  # pyright: ignore[reportMissingImports]
    DEFAULT_CACHE_TTL_SECONDS,
    ConfigBinding,
    DuplicateConfigKeyError,
    bind_config,
)


logger = get_logger("TuningConfig")
printer = PrettyPrinter()

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "hyperparam.yaml"
_CONFIG_BINDING: ConfigBinding = bind_config(DEFAULT_CONFIG_PATH)


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    """Return the absolute tuning config path without reading the file."""

    selected = DEFAULT_CONFIG_PATH if config_path is None else Path(config_path)
    return _CONFIG_BINDING.repository.resolve(selected)


def _validate_cache_ttl(cache_ttl: float) -> float:
    if isinstance(cache_ttl, bool) or not isinstance(cache_ttl, (int, float)):
        raise TuningConfigError(
            "cache_ttl must be a finite, non-negative number.",
            context=TuningErrorContext(component="TuningConfig", operation="validate_cache_ttl"),
            details={"cache_ttl": cache_ttl},
        )
    value = float(cache_ttl)
    if not math.isfinite(value) or value < 0:
        raise TuningConfigError(
            "cache_ttl must be a finite, non-negative number.",
            context=TuningErrorContext(component="TuningConfig", operation="validate_cache_ttl"),
            details={"cache_ttl": cache_ttl},
        )
    return value


def load_global_config(
    config_path: str | Path | None = None,
    *,
    force_reload: bool = False,
    cache_ttl: float = DEFAULT_CACHE_TTL_SECONDS,
) -> dict[str, Any]:
    """Load a deep-copied YAML mapping from the shared SLAI repository.

    Importing this module performs no file access.  The file is read only when
    this function (or another facade function that calls it) is invoked.
    """

    resolved = resolve_config_path(config_path)
    ttl = _validate_cache_ttl(cache_ttl)
    if not isinstance(force_reload, bool):
        raise TuningConfigError(
            "force_reload must be a bool.",
            context=TuningErrorContext(
                component="TuningConfig",
                operation="load",
                config_path=str(resolved),
            ),
            details={"actual_type": type(force_reload).__name__},
        )
    try:
        config = _CONFIG_BINDING.load(
            resolved,
            force_reload=force_reload,
            cache_ttl=ttl,
        )
    except TuningConfigError:
        raise
    except Exception as exc:
        details: dict[str, Any] = {
            "error_type": exc.__class__.__name__,
            "duplicate_key": isinstance(exc, DuplicateConfigKeyError),
        }
        raise TuningConfigError(
            f"Unable to load tuning configuration from {resolved}.",
            context=TuningErrorContext(
                component="TuningConfig",
                operation="load",
                config_path=str(resolved),
            ),
            details=details,
            cause=exc,
        ) from exc

    if not isinstance(config, dict):
        # The shared repository already enforces this.  Retaining the boundary
        # assertion protects callers if that implementation changes.
        raise TuningConfigError(
            "Tuning configuration root must be a mapping.",
            context=TuningErrorContext(
                component="TuningConfig",
                operation="load",
                config_path=str(resolved),
            ),
            details={"actual_type": type(config).__name__},
        )
    return config


def reload_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Force a reload of one tuning configuration path."""

    return load_global_config(config_path, force_reload=True)


def clear_config_cache(config_path: str | Path | None = None) -> None:
    """Evict tuning-owned entries from the shared repository cache.

    When a path is supplied, only that exact path is evicted.  With no path,
    every path previously loaded through this tuning binding is evicted; other
    SLAI subsystem bindings are unaffected.
    """

    if config_path is None:
        _CONFIG_BINDING.clear()
    else:
        _CONFIG_BINDING.repository.clear(resolve_config_path(config_path))


def get_config_cache_info(config_path: str | Path | None = None) -> dict[str, Any]:
    """Return cache diagnostics for the selected path."""

    if config_path is None:
        return _CONFIG_BINDING.cache_info()
    return _CONFIG_BINDING.repository.cache_info(
        resolve_config_path(config_path)
    ).to_dict()


def get_config_section(
    section_name: str,
    config: Mapping[str, Any] | None = None,
    *,
    config_path: str | Path | None = None,
    default: Mapping[str, Any] | None = None,
    required: bool = False,
) -> dict[str, Any]:
    """Return one top-level configuration section as a deep copy.

    Missing optional sections return ``default``.  A present non-mapping
    section is always an error; silently replacing malformed configuration
    with a default would conceal an operator mistake.
    """

    if not isinstance(section_name, str) or not section_name.strip():
        raise TuningConfigError(
            "section_name must be a non-empty string.",
            context=TuningErrorContext(component="TuningConfig", operation="section"),
        )
    name = section_name.strip()
    source = load_global_config(config_path) if config is None else config
    if not isinstance(source, Mapping):
        raise TuningConfigError(
            "Configuration supplied to get_config_section must be a mapping.",
            context=TuningErrorContext(
                component="TuningConfig",
                operation="section",
                config_path=str(resolve_config_path(config_path)) if config_path else None,
            ),
            details={"section": name, "actual_type": type(source).__name__},
        )

    section = source.get(name)
    if section is None:
        if required:
            raise TuningConfigError(
                f"Required tuning configuration section {name!r} is missing.",
                context=TuningErrorContext(
                    component="TuningConfig",
                    operation="section",
                    config_path=str(resolve_config_path(config_path)) if config_path else None,
                ),
                details={"section": name},
            )
        return copy.deepcopy(dict(default or {}))
    if not isinstance(section, Mapping):
        raise TuningConfigError(
            f"Tuning configuration section {name!r} must be a mapping.",
            context=TuningErrorContext(
                component="TuningConfig",
                operation="section",
                config_path=str(resolve_config_path(config_path)) if config_path else None,
            ),
            details={"section": name, "actual_type": type(section).__name__},
        )
    return copy.deepcopy(dict(section))


# Clearer alias for new code; the v2.2 name remains supported above.
load_tuning_config = load_global_config


__all__ = [
    "DEFAULT_CACHE_TTL_SECONDS",
    "DEFAULT_CONFIG_PATH",
    "clear_config_cache",
    "get_config_cache_info",
    "get_config_section",
    "load_global_config",
    "load_tuning_config",
    "reload_config",
    "resolve_config_path",
]
