import importlib
import pkgutil
import inspect
import time

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

from src.agents.collaborative.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Agent Registry")
printer = PrettyPrinter

class AgentRegistry:
    """
    Enhanced registry system with extended capabilities including:
    - Dynamic agent discovery
    - Health monitoring
    - Capability-based routing
    - Versioned registrations
    """
    def __init__(self, shared_memory: Optional[Any] = None, auto_discover: bool = True):
        self.config = load_global_config()
        self.registry_config = get_config_section('registry')
        agent_discovery_config =  self.registry_config.get('agent_discovery', {})
        
        self._agents: Dict[str, Dict] = {}
        self.shared_memory = shared_memory
        self._version = 1.8
        self._health_check_interval =  self.registry_config.get('health_check_interval', 300)
        self.default_package = agent_discovery_config.get('default_package', 'src.agents')
        self.excluded_modules = agent_discovery_config.get('excluded_modules', [])
        
        # Initialize with dynamic discovery from config
        if auto_discover:
            self.discover_agents(self.default_package)

        logger.info(f"Agent Registry Version {self._version} successfully initialized with:\n{self._agents}")

    def discover_agents(self, agents_package: str = 'src.agents') -> None:
        """
        Dynamically discover and register all concrete agent implementations.
        
        Args:
            agents_package: Python package path to search for agents
            
        Raises:
            ImportError: If package cannot be imported
        """       
        try:
            package = importlib.import_module(agents_package)
            for _, module_name, _ in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
                lowered = module_name.lower()
                if any(excluded in lowered for excluded in self.excluded_modules):
                    logger.warning(f"Skipping {module_name} due to exclusion rule.")
                    continue
                if "agent" in lowered:
                    self._load_agent_module(module_name)
        except ImportError as e:
            logger.error(f"Failed to import agents package: {e}")
            raise

    def _load_agent_module(self, module_name: str) -> None:
        """Internal method to load and validate agent modules"""
        try:
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and
                    not inspect.isabstract(obj)):
                    try:
                        instance = obj()
                        caps = getattr(instance, "capabilities", [])
                        meta = {
                            "class": obj,
                            "instance": instance,
                            "capabilities": caps,
                            "version": self._version
                        }
                        self._register_agent(obj.__name__, meta)
                    except Exception as e:
                        logger.error(f"Failed to instantiate {name}: {e}")
        except Exception as e:
            logger.error(f"Failed to load module {module_name}: {e}")

    def _register_agent(self, name: str, meta: Dict) -> None:
        """Validate and register an agent with version control"""
        if name in self._agents:
            if self._agents[name]["version"] >= self._version:
                logger.warning(f"Skipping older version of {name}")
                return
            logger.info(f"Upgrading {name} to version {self._version}")

        required_attrs = ["execute", "capabilities"]
        if not all(hasattr(meta["instance"], attr) for attr in required_attrs):
            raise ValueError(f"Agent {name} missing required attributes")

        self._agents[name] = meta
        logger.info(f"Registered agent: {name} with capabilities: {meta['capabilities']}")

        if self.shared_memory:
            self.shared_memory.set(
                f"agent:{name}",
                {
                    "status": "active",
                    "capabilities": meta["capabilities"],
                    "version": self._version,
                    "last_seen": time.time()
                }
            )

    def get_agents_by_task(self, task_type: str) -> Dict[str, Dict]:
        """
        Find agents supporting a specific task type with health checks
        
        Args:
            task_type: Task identifier to match against capabilities
            
        Returns:
            Dictionary of qualified agents with their metadata
        """
        return {
            name: {
                "instance": agent["instance"],
                "capabilities": agent["capabilities"],
                "version": agent["version"]
            }
            for name, agent in self._agents.items()
            if task_type in agent["capabilities"] and 
            self._check_agent_health(name)
        }

    def _check_agent_health(self, name: str) -> bool:
        """Perform health check on registered agent"""
        agent = self._agents.get(name)
        if not agent:
            return False
            
        if self.shared_memory:
            record = self.shared_memory.get(f"agent:{name}") or {}
            if time.time() - record.get("last_seen", 0) > self._health_check_interval:
                logger.warning(f"Agent {name} appears unresponsive")
                return False
        return True

    def reload_agent(self, name: str) -> bool:
        """Reload an agent module and update registration"""
        agent = self._agents.get(name)
        if not agent:
            logger.error(f"Agent {name} not found for reload")
            return False

        try:
            module = inspect.getmodule(agent["class"])
            if module:
                importlib.reload(module)
                self.discover_agents(module.__name__)
                return True
        except Exception as e:
            logger.error(f"Failed to reload agent {name}: {e}")
        return False

    def batch_register(self, agents: List[Dict[str, Any]]) -> None:
        """Bulk register pre-configured agents"""
        for agent in agents:
            try:
                self._register_agent(agent["name"], agent["meta"])
            except KeyError as e:
                logger.error(f"Invalid agent configuration: {e}")

    # Existing methods with enhanced type hints and error handling
    def unregister(self, name: str) -> None:
        """Safely remove an agent from the registry"""
        if name not in self._agents:
            logger.warning(f"Attempted to unregister unknown agent: {name}")
            return

        del self._agents[name]
        if self.shared_memory:
            self.shared_memory.delete(f"agent:{name}")
        logger.info(f"Unregistered agent: {name}")

    def list_agents(self) -> Dict[str, List[str]]:
        """Get comprehensive agent list with capabilities"""
        return {name: info["capabilities"] for name, info in self._agents.items()}



if __name__ == "__main__":
    print("\n=== Running Agent Registry ===\n")

    class MockSharedMemory:
        def __init__(self):
            self._store = {}

        def set(self, key: str, value: Any) -> None:
            self._store[key] = value

        def get(self, key: str) -> Any:
            return self._store.get(key)

        def delete(self, key: str) -> None:
            if key in self._store:
                del self._store[key]

    class TranslationAgent(ABC):
        capabilities = ["translation", "language"]

        def execute(self, task_data: Dict) -> Any:
            return f"Translated: {task_data['text']}"

    class AnalysisAgent(ABC):
        capabilities = ["analysis", "data"]

        def execute(self, task_data: Dict) -> Any:
            return {"status": "analyzed", "result": 42}

    shared_memory = MockSharedMemory()
    registry = AgentRegistry(shared_memory, auto_discover=False)

    registry.batch_register([
        {"name": "Translator", "meta": {
            "class": TranslationAgent,
            "instance": TranslationAgent(),
            "capabilities": ["translation"],
            "version": 1.0,
        }},
        {"name": "Analyzer", "meta": {
            "class": AnalysisAgent,
            "instance": AnalysisAgent(),
            "capabilities": ["analysis"],
            "version": 1.0,
        }},
    ])

    agents = registry.list_agents()
    assert "Translator" in agents and "Analyzer" in agents

    translators = registry.get_agents_by_task("translation")
    assert "Translator" in translators
    assert translators["Translator"]["instance"].execute({"text": "Hello"}).startswith("Translated")

    assert registry._check_agent_health("Translator") is True
    registry.unregister("Analyzer")
    assert "Analyzer" not in registry.list_agents()

    print("All registry.py tests passed.\n")
