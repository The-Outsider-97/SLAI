
from pathlib import Path
from typing import Any, Mapping

from src.utils.configuration import bind_config # type: ignore


_BINDING = bind_config(
    Path(__file__).resolve().parent.parent
    / "configs"
    / "agents_config.yaml"
)


def load_global_config(
    *,
    force_reload: bool = False,
    cache_ttl: float = 60.0,
) -> dict[str, Any]:
    return _BINDING.load(
        force_reload=force_reload,
        cache_ttl=cache_ttl,
    )


def reload_config() -> dict[str, Any]:
    return _BINDING.reload()


def get_config_section(
    section_name: str,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _BINDING.section(section_name, config=config)


def clear_config_cache() -> None:
    _BINDING.clear()


def config_cache_info() -> dict[str, Any]:
    return _BINDING.cache_info()
