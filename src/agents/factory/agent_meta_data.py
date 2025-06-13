
import yaml, json

from typing import Tuple, Dict, List, Optional
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple

from src.agents.factory.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Agent Meta Data")
printer = PrettyPrinter

class DotDict(dict):
    def __getattr__(self, item):
        return self.get(item)

@dataclass(slots=True)
class AgentMetaData:
    name: str
    class_name: str
    module_path: str
    required_params: Tuple[str] = ()
    version: str = None
    description: str = ""
    author: str = "Unknown"
    dependencies: List[str] = None
    config: Dict = None
    meta_config: Dict = None
    required_fields: Dict = None
    validation_rules: Dict = None

    def __post_init__(self):
        self.config = load_global_config()
        self.meta_config = get_config_section('agent_meta')
        self.version = self.meta_config.get('default_version')
        self.required_fields = self.meta_config.get('required_fields')
        self.validation_rules = self.meta_config.get('validation_rules', {
            'max_name_length', 'allowed_modules'
        })

        self.validation_rules = DotDict(self.validation_rules)
        self.dependencies = self.dependencies or []
        self._validate()

    def _validate(self):
        # Check required fields
        for field in self.required_fields:
            if not getattr(self, field):
                raise ValueError(f"Missing required field: {field}")

        # Validate name length
        max_length = self.validation_rules.max_name_length
        if len(self.name) > max_length:
            raise ValueError(f"Name exceeds maximum length of {max_length} characters")

        # Check module path validity
        allowed_modules = self.validation_rules.allowed_modules
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


class AgentRegistry:
    def __init__(self):
        self.config = load_global_config()
        self.registry_config = get_config_section('agent_registry')
        #self.version = self.registry_config.get('')
        self.agents: Dict[str, AgentMetaData] = {}
        self.version_map = defaultdict(list)
        self.dependency_graph = defaultdict(set)

        self.registry = {}

    def register(self, metadata: AgentMetaData):
        if not isinstance(metadata, AgentMetaData):
            raise TypeError("Only AgentMetaData objects can be registered")
            
        # Update primary registry
        self.agents[metadata.name] = metadata
        
        # Update version index
        self.version_map[metadata.version].append(metadata.name)
        
        # Build dependency graph
        for dep in metadata.dependencies:
            self.dependency_graph[metadata.name].add(dep)
            
        logger.info(f"Registered agent: {metadata.name} v{metadata.version}")

    def get(self, agent_name: str, version: Optional[str] = None) -> AgentMetaData:
        """Retrieve agent metadata with optional version specifier"""
        if agent_name not in self.agents:
            raise KeyError(f"Agent '{agent_name}' not registered")
            
        if version and self.agents[agent_name].version != version:
            candidates = [m for m in self.version_map[version] if m == agent_name]
            if not candidates:
                raise ValueError(f"Version {version} not available for {agent_name}")
            return self.agents[candidates[0]]
            
        return self.agents[agent_name]
    
    def get_dependencies(self, agent_name: str) -> List[str]:
        """Get all dependencies for an agent"""
        if agent_name not in self.dependency_graph:
            return []
        return list(self.dependency_graph[agent_name])

    def resolve_dependency_tree(self, agent_name: str) -> List[str]:
        """Get full dependency tree in load order"""
        visited = set()
        load_order = []
        
        def resolve(name):
            if name in visited:
                return
            for dep in self.get_dependencies(name):
                resolve(dep)
            visited.add(name)
            load_order.append(name)
            
        resolve(agent_name)
        return load_order

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Agent Meta Data ===\n")
    metadata = AgentMetaData(
        name="TestAgent", 
        class_name="TestClass", 
        module_path="src.agents.core"
    )
    metadata.__post_init__()

    printer.pretty("DATA", metadata.to_dict(), "success")

    print("\n* * * * * Phase 2 * * * * *\n")

    validate = metadata._validate()
    print(f"\n{validate}")

    print("\n=== Successfully Ran Agent Meta Data ===\n")
