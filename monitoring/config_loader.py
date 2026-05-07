from __future__ import annotations

import yaml

from pathlib import Path

CONFIG_PATH = "monitoring/configs/monitor_config.yaml"

_global_config: dict | None = None


def load_global_config() -> dict:
    """
    Load and cache monitor_config.yaml.
    Subsequent calls return the same dict without re-reading the file.
    """
    global _global_config
    if _global_config is None:
        config_path = Path(__file__).parent.parent / CONFIG_PATH
        with open(config_path, "r", encoding="utf-8") as f:
            _global_config = yaml.safe_load(f)
        _global_config["__config_path__"] = str(config_path.resolve()) # type: ignore
    return _global_config # type: ignore


def get_config_section(section_name: str) -> dict:
    """Return a top-level section from monitor_config.yaml, or {} if absent."""
    config = load_global_config()
    return config.get(section_name, {})