import importlib
import pkgutil
import inspect
import time

from typing import Dict, Any, List, Optional

from src.agents.base_agent import BaseAgent
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
    _module_failures: Dict[str, str] = {}

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
        self._discovered_packages = set()
        self._agent_init_failures: Dict[str, str] = {}
        
        
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
        if agents_package in self._discovered_packages:
            logger.debug("Agent package %s already discovered for this registry instance; skipping re-scan.", agents_package)
            return

        try:
            package = importlib.import_module(agents_package)
            for _, module_name, _ in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
                lowered = module_name.lower()
                if any(excluded in lowered for excluded in self.excluded_modules):
                    logger.warning(f"Skipping {module_name} due to exclusion rule.")
                    continue
                # Avoid recursive registry bootstraps caused by discovering collaborative_agent itself.
                if lowered.endswith(".collaborative_agent"):
                    logger.debug("Skipping %s to avoid recursive collaboration manager bootstrap.", module_name)
                    continue
                if "agent" in lowered:
                    self._load_agent_module(module_name)
            self._discovered_packages.add(agents_package)
        except ImportError as e:
            logger.error(f"Failed to import agents package: {e}")
            raise

    def _load_agent_module(self, module_name: str) -> None:
        """Internal method to load and validate agent modules"""
        if module_name in self._module_failures:
            logger.debug(
                "Skipping module %s after cached import failure: %s",
                module_name,
                self._module_failures[module_name],
            )
            return
        try:
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ != module.__name__:
                    continue
                if inspect.isabstract(obj) or not issubclass(obj, BaseAgent):
                    continue

                caps = list(getattr(obj, "capabilities", []))
                meta = {
                    "class": obj,
                    "instance": None,
                    "capabilities": caps,
                    "version": self._version,
                }
                self._register_agent(obj.__name__, meta)
        except Exception as e:
            self._module_failures[module_name] = f"{type(e).__name__}: {e}"
            logger.error(f"Failed to load module {module_name}: {e}")

    def _register_agent(self, name: str, meta: Dict) -> None:
        """Validate and register an agent with version control"""
        if name in self._agents:
            if self._agents[name]["version"] >= self._version:
                logger.warning(f"Skipping older version of {name}")
                return
            logger.info(f"Upgrading {name} to version {self._version}")

        agent_class = meta.get("class")
        required_attrs = ["execute", "capabilities"]
        if not agent_class or not all(hasattr(agent_class, attr) for attr in required_attrs):
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
        qualified: Dict[str, Dict] = {}
        for name, agent in self._agents.items():
            if task_type not in agent["capabilities"]:
                continue
            if not self._check_agent_health(name):
                continue
            instance = self._get_or_create_instance(name)
            if instance is None:
                continue
            qualified[name] = {
                "instance": instance,
                "capabilities": agent["capabilities"],
                "version": agent["version"],
            }
        return qualified

    def _get_or_create_instance(self, name: str):
        agent = self._agents.get(name)
        if not agent:
            return None
        if agent.get("instance") is not None:
            return agent["instance"]
        if name in self._agent_init_failures:
            logger.debug(
                "Skipping agent %s after cached initialization failure: %s",
                name,
                self._agent_init_failures[name],
            )
            return None

        try:
            cls = agent["class"]
            init_signature = inspect.signature(cls.__init__)
            kwargs = {}
            if "shared_memory" in init_signature.parameters:
                kwargs["shared_memory"] = self.shared_memory
            instance = cls(**kwargs)
            agent["instance"] = instance
            return instance
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._agent_init_failures[name] = error
            logger.warning("Agent %s is unavailable for execution: %s", name, error)
            return None

    def _check_agent_health(self, name: str) -> bool:
        """Perform health check on registered agent"""
        agent = self._agents.get(name)
        if not agent:
            return False
            
        if self.shared_memory:
            record = self.shared_memory.get(f"agent:{name}") or {}
            if time.time() - record.get("last_seen", 0) > self._health_check_interval:
                logger.warning(f"Agent {name} appears unresponsive")
                # Keep stale agents routable; router-level retries/failure accounting
                # provide a safer degradation path than hard de-registration.
                return True
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
