import yaml

from pathlib import Path
from threading import RLock

CONFIG_PATH = "base/configs/agents_config.yaml"

_global_config = None
_config_lock = RLock()
_config_sections = {}

def load_global_config():
    global _global_config
    if _global_config is not None:
        return _global_config
    with _config_lock:
        if _global_config is None:
            config_path = Path(__file__).parent.parent.parent / CONFIG_PATH
            with open(config_path, "r", encoding="utf-8") as f:
                _global_config = yaml.safe_load(f)
            _global_config["__config_path__"] = str(config_path.resolve())
    return _global_config

def get_config_section(section_name: str) -> dict:
    cached = _config_sections.get(section_name)
    if cached is not None:
        return cached
    config = load_global_config()
    section = config.get(section_name, {})
    with _config_lock:
        _config_sections.setdefault(section_name, section)
    return section
