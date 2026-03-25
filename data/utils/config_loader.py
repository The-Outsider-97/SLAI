from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from data.utils.data_error import DataConfigError

CONFIG_PATH = "configs/data_config.yaml"
_global_config: Dict[str, Any] | None = None


def load_global_config() -> Dict[str, Any]:
    global _global_config
    if _global_config is None:
        config_path = Path(__file__).parent.parent / CONFIG_PATH
        if not config_path.exists():
            raise DataConfigError(
                "Data config file not found",
                context={"config_path": str(config_path.resolve())},
            )
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                _global_config = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            raise DataConfigError(
                "Failed to parse data config YAML",
                context={"config_path": str(config_path.resolve()), "error": str(exc)},
            ) from exc
        _global_config["__config_path__"] = str(config_path.resolve())
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
