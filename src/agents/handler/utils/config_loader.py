import threading
import time
import yaml # type: ignore

from pathlib import Path
from typing import Any, Dict, Optional, Union

from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Config Loader")
printer = PrettyPrinter()

# ----------------------------------------------------------------------
# Defaults & constants
# ----------------------------------------------------------------------
DEFAULT_CONFIG_PATH = "handler/configs/handler_config.yaml"
DEFAULT_CACHE_TTL_SECONDS = 60          # 1 minute
FILE_WATCH_INTERVAL_SECONDS = 5         # check mtime at most every 5s (avoid stat storms)

# Global cache state (thread‑safe)
_config_cache: Dict[str, Any] = {
    "path": None,            # absolute path of the loaded config file
    "mtime": 0.0,            # last modification time (os.path.getmtime)
    "data": None,            # parsed YAML dict (or None if not loaded)
    "loaded_at": 0.0,        # timestamp when the file was last loaded
}

_lock = threading.RLock()


# ----------------------------------------------------------------------
# Helper functions (pure, no external error handling)
# ----------------------------------------------------------------------
def _resolve_config_path(config_path: Optional[Union[str, Path]] = None) -> Path:
    """Return an absolute Path to the config file, with default fallback."""
    if config_path is None:
        base = Path(__file__).parent.parent.parent  # goes up to project root
        path = base / DEFAULT_CONFIG_PATH
    else:
        path = Path(config_path)
    return path.resolve()


def _get_file_mtime(path: Path) -> float:
    """Return file modification time or 0.0 if the file does not exist."""
    try:
        return path.stat().st_mtime
    except (OSError, FileNotFoundError):
        return 0.0


def _should_reload(current_mtime: float, current_loaded_at: float,
                   ttl: float, force: bool) -> bool:
    """
    Deterministic cache freshness check.

    Returns True if the cache must be refreshed because:
    - force=True, or
    - file mtime changed, or
    - TTL has expired since last load.
    """
    if force:
        return True
    # mtime changed
    if current_mtime != 0.0 and current_mtime != _config_cache["mtime"]:
        return True
    # TTL expired
    if ttl > 0 and (time.time() - current_loaded_at) > ttl:
        return True
    return False


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    """
    Load and parse a YAML file in a deterministic way.
    Raises FileNotFoundError or yaml.YAMLError on failure.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # Ensure the result is a dictionary
    if not isinstance(data, dict):
        raise TypeError(f"Config file {path} must contain a YAML mapping, got {type(data).__name__}")
    return data


# ----------------------------------------------------------------------
# Public API (backward compatible + deterministic caching)
# ----------------------------------------------------------------------
def load_global_config(config_path: Optional[Union[str, Path]] = None, *, force_reload: bool = False,
                       cache_ttl: float = DEFAULT_CACHE_TTL_SECONDS) -> Dict[str, Any]:
    """
    Load the global configuration with deterministic caching.

    Deterministic means:
      - Same file content + same load parameters → same dict.
      - Cache decisions are based on file mtime and TTL.
      - The function is idempotent and thread‑safe.

    Args:
        config_path: Optional path to the config YAML file.
        force_reload: If True, ignore cache and reload from disk.
        cache_ttl: Maximum seconds to keep cached data without checking mtime.
                  Set to 0 to disable TTL (use only mtime changes).

    Returns:
        Dictionary with the merged configuration.
        The dictionary also contains the special key '__config_path__'
        with the absolute path of the loaded file.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
        TypeError: If the YAML root is not a mapping.
    """
    path = _resolve_config_path(config_path)
    abs_path_str = str(path)

    with _lock:
        # Initialise cache if this is the first call or the path changed
        if _config_cache["path"] != abs_path_str:
            _config_cache["path"] = abs_path_str
            _config_cache["mtime"] = 0.0
            _config_cache["data"] = None
            _config_cache["loaded_at"] = 0.0

        current_mtime = _get_file_mtime(path)
        # Decide whether we need to reload
        if _should_reload(
            current_mtime=current_mtime,
            current_loaded_at=_config_cache["loaded_at"],
            ttl=cache_ttl,
            force=force_reload,
        ):
            try:
                data = _load_yaml_file(path)
                # Update cache atomically
                _config_cache["data"] = data
                _config_cache["mtime"] = current_mtime
                _config_cache["loaded_at"] = time.time()
                # Store the resolved path inside the config for introspection
                data["__config_path__"] = abs_path_str
                logger.info(f"Configuration reloaded from {abs_path_str} (mtime={current_mtime})")
            except Exception as e:
                # Log error but re-raise – do not serve stale data on failure
                logger.error(f"Failed to load config {abs_path_str}: {e}")
                raise
        else:
            logger.debug(f"Using cached configuration from {abs_path_str}")

        # Return a copy to avoid accidental mutation of the cache
        return dict(_config_cache["data"] or {})


def reload_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Force a reload of the configuration, ignoring any cache.
    Equivalent to load_global_config(force_reload=True).
    """
    return load_global_config(config_path, force_reload=True)


def clear_config_cache() -> None:
    """Completely clear the in‑memory configuration cache."""
    with _lock:
        _config_cache["path"] = None
        _config_cache["mtime"] = 0.0
        _config_cache["data"] = None
        _config_cache["loaded_at"] = 0.0
        logger.info("Configuration cache cleared")


def get_config_cache_info() -> Dict[str, Any]:
    """
    Return diagnostic information about the current cache state.
    Useful for debugging deterministic caching behaviour.
    """
    with _lock:
        return {
            "config_path": _config_cache["path"],
            "mtime": _config_cache["mtime"],
            "loaded_at": _config_cache["loaded_at"],
            "age_seconds": time.time() - _config_cache["loaded_at"] if _config_cache["loaded_at"] else None,
            "has_data": _config_cache["data"] is not None,
        }


def get_config_section(section_name: str, config: Optional[Dict[str, Any]] = None, *,
    config_path: Optional[Union[str, Path]] = None, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Retrieve a specific section from the global configuration.

    Args:
        section_name: Name of the section (top‑level key in YAML).
        config: If provided, use this dict instead of loading the global config.
        config_path: Optional custom config file path (ignored if config is given).
        default: Dictionary returned when the section is missing.

    Returns:
        The configuration section (always a dict), or the default if missing.
    """
    if config is None:
        config = load_global_config(config_path)
    section = config.get(section_name)
    if section is None:
        return dict(default or {})
    if not isinstance(section, dict):
        logger.warning(f"Config section '{section_name}' is not a dictionary, returning default")
        return dict(default or {})
    return dict(section)
