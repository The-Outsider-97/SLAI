from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.agents.factory.utils.config_loader import get_config_section, load_global_config
from logs.logger import PrettyPrinter, get_logger

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
    required_params: Tuple[str, ...] = ()
    version: Optional[str] = None
    description: str = ""
    author: str = "Unknown"
    dependencies: Optional[List[str]] = None
    config: Optional[Dict] = None
    meta_config: Optional[Dict] = None
    required_fields: Optional[List[str]] = None
    validation_rules: Optional[Dict] = None

    def __post_init__(self):
        self.config = load_global_config()
        self.meta_config = get_config_section("agent_meta")
        if self.version is None:
            self.version = self.meta_config.get("default_version")

        self.required_fields = self.meta_config.get(
            "required_fields", ["name", "class_name", "module_path", "version"]
        )
        self.validation_rules = DotDict(
            self.meta_config.get(
                "validation_rules",
                {"max_name_length": 64, "allowed_modules": ["agents.", "src.agents."]},
            )
        )

        self.dependencies = self.dependencies or []
        self._validate()

    def _validate(self):
        for field in self.required_fields:
            if not getattr(self, field, None):
                raise ValueError(f"Missing required field: {field}")

        max_length = self.validation_rules.max_name_length or 64
        if len(self.name) > max_length:
            raise ValueError(f"Name exceeds maximum length of {max_length} characters")

        allowed_modules = self.validation_rules.allowed_modules or []
        if allowed_modules and not any(self.module_path.startswith(m) for m in allowed_modules):
            raise ValueError(f"Invalid module path: {self.module_path}")

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "class_name": self.class_name,
            "module_path": self.module_path,
            "version": self.version,
            "dependencies": self.dependencies,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "AgentMetaData":
        return cls(**data)


class AgentRegistry:
    def __init__(self):
        self.config = load_global_config()
        self.registry_config = get_config_section("agent_registry")
        self.agents: Dict[str, AgentMetaData] = {}
        self.version_map = defaultdict(list)
        self.dependency_graph = defaultdict(set)

    def register(self, metadata: AgentMetaData):
        if not isinstance(metadata, AgentMetaData):
            raise TypeError("Only AgentMetaData objects can be registered")

        self.agents[metadata.name] = metadata
        self.version_map[metadata.version].append(metadata.name)

        for dep in metadata.dependencies:
            self.dependency_graph[metadata.name].add(dep)

        logger.info(f"Registered agent: {metadata.name} v{metadata.version}")

    def get(self, agent_name: str, version: Optional[str] = None) -> AgentMetaData:
        if agent_name not in self.agents:
            raise KeyError(f"Agent '{agent_name}' not registered")

        if version and self.agents[agent_name].version != version:
            candidates = [m for m in self.version_map[version] if m == agent_name]
            if not candidates:
                raise ValueError(f"Version {version} not available for {agent_name}")
            return self.agents[candidates[0]]

        return self.agents[agent_name]

    def get_dependencies(self, agent_name: str) -> List[str]:
        if agent_name not in self.dependency_graph:
            return []
        return list(self.dependency_graph[agent_name])

    def resolve_dependency_tree(self, agent_name: str) -> List[str]:
        visited = set()
        load_order: List[str] = []

        def resolve(name):
            if name in visited:
                return
            for dep in self.get_dependencies(name):
                resolve(dep)
            visited.add(name)
            load_order.append(name)

        resolve(agent_name)
        return load_order
