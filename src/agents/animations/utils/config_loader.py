"""Animation subsystem configuration uses SLAI's shared configuration loader."""
from pathlib import Path
from src.utils.configuration import bind_config
_BINDING = bind_config(Path(__file__).resolve().parents[1] / "configs" / "animation_config.yaml")

def load_animation_config():
    return _BINDING.load()
