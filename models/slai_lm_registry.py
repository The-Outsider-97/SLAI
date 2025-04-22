from models.slai_lm import SLAILM
from typing import Optional, Dict
from threading import Lock


class SLAILMManager:
    _instances: Dict[str, SLAILM] = {}
    _lock = Lock()

    @classmethod
    def get_instance(cls, key: str = "default", shared_memory=None, agent_factory=None, config: Optional[dict] = None) -> SLAILM:
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = SLAILM(
                    shared_memory=shared_memory,
                    agent_factory=agent_factory,
                    custom_config=config
                )
            return cls._instances[key]

    @classmethod
    def reset(cls, key: Optional[str] = None):
        with cls._lock:
            if key:
                cls._instances.pop(key, None)
            else:
                cls._instances.clear()

    @classmethod
    def get_all(cls) -> Dict[str, SLAILM]:
        return cls._instances
