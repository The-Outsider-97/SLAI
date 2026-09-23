from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from src.utils.configuration import DEFAULT_CACHE_TTL_SECONDS, ConfigBinding, bind_config # pyright: ignore[reportMissingImports]
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("Alignment Config Loader")
printer = PrettyPrinter()


# ---------------------------------------------------------------------------
# Canonical alignment configuration binding
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "alignment_config.yaml"
)

_CONFIG_BINDING: ConfigBinding = bind_config(DEFAULT_CONFIG_PATH)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_config_path(config_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve a alignment configuration path without reading the file.

    When ``config_path`` is omitted, the canonical
    ``alignment/configs/alignment_config.yaml`` path is returned.

    Args:
        config_path:
            Optional alternative YAML configuration path.

    Returns:
        Absolute resolved configuration path.
    """
    selected = (
        DEFAULT_CONFIG_PATH
        if config_path is None
        else Path(config_path)
    )

    return _CONFIG_BINDING.repository.resolve(selected)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_global_config(
    config_path: Optional[Union[str, Path]] = None,
    *,
    force_reload: bool = False,
    cache_ttl: float = DEFAULT_CACHE_TTL_SECONDS,
) -> Dict[str, Any]:
    """
    Load the alignment subsystem configuration.

    This function preserves the established alignment API while delegating
    parsing, caching, modification detection, locking, and defensive copying
    to SLAI's shared configuration repository.

    Args:
        config_path:
            Optional custom alignment configuration path. When omitted,
            ``alignment_config.yaml`` is used.

        force_reload:
            If ``True``, bypass the cached value and reload the YAML file.

        cache_ttl:
            Maximum cache age in seconds. ``0`` disables TTL expiration while
            retaining file-change invalidation.

    Returns:
        A defensive copy of the loaded configuration mapping.

    Raises:
        FileNotFoundError:
            If the selected configuration file does not exist.

        yaml.YAMLError:
            If YAML parsing fails.

        DuplicateConfigKeyError:
            If a YAML mapping contains duplicate keys.

        TypeError:
            If the YAML root is not a mapping.

        ValueError:
            If configuration infrastructure rejects invalid parameters.
    """
    resolved = resolve_config_path(config_path)

    config = _CONFIG_BINDING.load(
        resolved,
        force_reload=force_reload,
        cache_ttl=cache_ttl,
    )
    logger.debug("alignment configuration loaded | path=%s | force_reload=%s", resolved, force_reload)

    return config


def reload_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Force-reload the alignment configuration.

    Equivalent to::

        load_global_config(config_path, force_reload=True)
    """
    resolved = resolve_config_path(config_path)
    config = _CONFIG_BINDING.reload(resolved)

    logger.info("alignment configuration reloaded | path=%s", resolved)

    return config


def clear_config_cache() -> None:
    """
    Clear configuration entries loaded through this alignment binding.

    The shared configuration repository itself is not globally cleared.
    Configuration entries owned by other SLAI subsystems remain untouched.
    """
    _CONFIG_BINDING.clear()

    logger.debug("alignment configuration cache cleared")


def get_config_cache_info() -> Dict[str, Any]:
    """
    Return cache diagnostics for the alignment configuration binding.

    Returns:
        Mapping containing fields such as:

        - ``config_path``
        - ``mtime``
        - ``mtime_ns``
        - ``size``
        - ``loaded_at``
        - ``age_seconds``
        - ``has_data``
    """
    return _CONFIG_BINDING.cache_info()


def get_config_section(
    section_name: str,
    config: Optional[Mapping[str, Any]] = None,
    *,
    config_path: Optional[Union[str, Path]] = None,
    default: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Return one top-level alignment configuration section.

    When an already-loaded ``config`` mapping is supplied, no additional file
    access is performed.

    Args:
        section_name:
            Top-level YAML section name.

        config:
            Optional already-loaded alignment configuration.

        config_path:
            Optional configuration path used only when ``config`` is omitted.

        default:
            Mapping returned when the section is absent or is not a mapping.

    Returns:
        Defensive copy of the selected configuration section.
    """
    if not isinstance(section_name, str) or not section_name.strip():
        raise ValueError("section_name must be a non-empty string")

    return _CONFIG_BINDING.section(
        section_name.strip(),
        config=config,
        config_path=config_path,
        default=default,
    )


__all__ = [
    "DEFAULT_CACHE_TTL_SECONDS",
    "DEFAULT_CONFIG_PATH",
    "clear_config_cache",
    "get_config_cache_info",
    "get_config_section",
    "load_global_config",
    "reload_config",
    "resolve_config_path",
]