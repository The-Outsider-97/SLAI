
import yaml, json

from typing import Tuple, Dict, List
from dataclasses import dataclass
from typing import Tuple

from logs.logger import get_logger

logger = get_logger("Agent Meta Data")

CONFIG_PATH = "src/agents/factory/configs/factory_config.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_merged_config(user_config=None):
    base_config = load_config()
    if user_config: base_config.update(user_config)
    return base_config

@dataclass(slots=True)
class AgentMetaData:
#    __slots__ = ['name', 'class_name', 'module_path', 'required_params', 
#                 'version', 'description', 'author', 'dependencies', 'config']
    name: str
    class_name: str
    module_path: str
    required_params: Tuple[str] = ()
    version: str = None
    description: str = ""
    author: str = "Unknown"
    dependencies: List[str] = None
    config: Dict = None

    def __post_init__(self):
        meta_config = load_config().get("agent_meta")
        self.version = self.version or meta_config.get("default_version", "1.0.0")
        self.dependencies = self.dependencies or []
        self.config = self.config or {}
        self._validate(meta_config)

    def _validate(self, config: Dict):
        # Check required fields
        for field in config.get("required_fields", []):
            if not getattr(self, field):
                raise ValueError(f"Missing required field: {field}")

        # Validate name length
        max_length = config.get("validation_rules", {}).get("max_name_length", 50)
        if len(self.name) > max_length:
            raise ValueError(f"Name exceeds maximum length of {max_length} characters")

        # Check module path validity
        allowed_modules = config.get("validation_rules", {}).get("allowed_modules", [])
        if allowed_modules and not any(self.module_path.startswith(m) for m in allowed_modules):
            raise ValueError(f"Invalid module path: {self.module_path}")

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "class_name": self.class_name,
            "module_path": self.module_path,
            "version": self.version,
            "dependencies": self.dependencies,
            "config": self.config
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'AgentMetaData':
        return cls(**data)
    
# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Agent Meta Data ===\n")
    config = load_config()
    metadata = AgentMetaData(
        name="TestAgent", 
        class_name="TestClass", 
        module_path="src.agents.core"
    )
    metadata.__post_init__()

    print(f"\n{metadata.to_dict()}")

    print("\n* * * * * Phase 2 * * * * *\n")

    validate = metadata._validate(config)
    print(f"\n{validate}")

    print("\n* * * * * Phase 3 * * * * *\n")
#    agent_id1='dqn'
#    agent_id2='maml'
#    cross = factory._crossover(agent_id1, agent_id2)
#    print(f"\n{cross}")

    print("\n* * * * * Phase 4 * * * * *\n")
#    monitor = factory.monitor_architecture()
#    print(f"\n{monitor}")

    print("\n=== Successfully Ran Agent Meta Data ===\n")
