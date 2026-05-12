import yaml # type: ignore

from pathlib import Path

CONFIG_PATH = "reader/configs/reader_config.yaml"
_global_config = None


def load_reader_config() -> dict:
    global _global_config
    if _global_config is None:
        config_path = Path(__file__).parent.parent.parent / CONFIG_PATH
        with open(config_path, "r", encoding="utf-8") as f:
            _global_config = yaml.safe_load(f) or {}
        _global_config["__config_path__"] = str(config_path.resolve())
    return _global_config


def get_config_section(section_name: str) -> dict:
    config = load_reader_config()
    return config.get(section_name, {})
