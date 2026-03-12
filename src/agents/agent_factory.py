import importlib
import inspect
from typing import Any, Dict, Optional, Type

from . import __version__ 

from .base.utils.main_config_loader import load_global_config
from .factory.agent_meta_data import AgentMetaData, AgentRegistry
from .factory.metrics_adapter import MetricsAdapter
from .factory.reasoner import BasicZeroReasoner
from .base_agent import BaseAgent
from logs.logger import get_logger, PrettyPrinter

from .evaluation_agent import EvaluationAgent
from .execution_agent import ExecutionAgent
from .alignment_agent import AlignmentAgent
from .knowledge_agent import KnowledgeAgent
from .language_agent import LanguageAgent
from .perception_agent import PerceptionAgent
from .learning_agent import LearningAgent
from .planning_agent import PlanningAgent
from .safety_agent import SafetyAgent
from .adaptive_agent import AdaptiveAgent
from .reasoning_agent import ReasoningAgent
from .handler_agent import HandlerAgent

logger = get_logger("Agent Factory")
printer = PrettyPrinter

class AgentFactory:
    """
    A dynamic, adaptive factory for creating and managing agents.
    It uses a metadata registry for dynamic agent loading and a metrics
    adapter for runtime configuration tuning.
    """
    _agent_classes: Dict[str, Type[BaseAgent]] = {
        'adaptive': AdaptiveAgent,
        'alignment': AlignmentAgent,
        'evaluation': EvaluationAgent,
        'execution': ExecutionAgent,
        'knowledge': KnowledgeAgent,
        'language': LanguageAgent,
        'learning': LearningAgent,
        'perception': PerceptionAgent,
        'planning': PlanningAgent,
        'reasoning': ReasoningAgent,
        'safety': SafetyAgent,
        'handler': HandlerAgent,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the AgentFactory with global and optional runtime configurations.

        Args:
            config (Optional[Dict[str, Any]]): A dictionary for runtime configuration
                                               overrides.
        """
        self.global_config = load_global_config()
        if config:
            self.global_config.update(config)

        self.metrics_adapter = MetricsAdapter()
        self.registry = AgentRegistry()

        self.active_agents: Dict[str, BaseAgent] = {}
        for name, cls in self._agent_classes.items():
            self.registry.register(AgentMetaData(
                name=name,
                module_path=cls.__module__,
                class_name=cls.__name__,
                version=__version__,
                dependencies=self._get_agent_dependencies(cls)
            ))

        logger.info("Agent Factory initialized with dynamic registry and metrics adapter.")

    def _get_agent_dependencies(self, cls) -> list[str]:
        """Stub for future dependency inspection."""
        return getattr(cls, "REQUIRES", [])
    
    def discover_agents(self):
        agents_module = importlib.import_module(__package__)

        for _, obj in inspect.getmembers(agents_module):
            if inspect.isclass(obj) and issubclass(obj, BaseAgent) and obj is not BaseAgent:
                self._agent_classes[obj.__name__.lower()] = obj

    def register_agent(self, metadata: AgentMetaData):
        """
        Registers an agent's metadata, making it available for creation.
        """
        if not isinstance(metadata, AgentMetaData):
            raise TypeError("Can only register objects of type AgentMetaData.")

        if metadata.name in self.registry.agents:
            logger.warning(f"Agent '{metadata.name}' is already registered. Overwriting metadata.")

        self.registry.register(metadata)
        logger.info(f"Registered agent: '{metadata.name}' (version {metadata.version})")

    def create(self, agent_type: str, shared_memory: Any, **kwargs: Any) -> BaseAgent:
        """
        Creates an instance of a specified agent.

        This method retrieves the appropriate configuration for the agent type,
        merges it with any runtime arguments, and instantiates the agent class.

        Args:
            agent_type (str): The type of agent to create (e.g., 'planning', 'learning').
            shared_memory (Any): The shared memory object to be used by the agent.
            **kwargs (Any): Additional keyword arguments to be passed to the agent's
                            constructor, which may include agent-specific dependencies
                            like 'env' for the LearningAgent.

        Returns:
            BaseAgent: An instance of the requested agent.

        Raises:
            ValueError: If the requested agent_type is unknown.
            TypeError: If the arguments provided do not match the agent's constructor signature.
        """
        printer.status("CREATE", f"Request to create agent of type: '{agent_type}'")

        # 1: Check the cache first. If agent already exists, return it.
        if agent_type in self.active_agents:
            logger.info(f"Returning cached instance of agent: '{agent_type}'")
            return self.active_agents[agent_type]

        if agent_type not in self.registry.agents:
            logger.error(f"Unknown agent type requested: '{agent_type}'. Ensure it is registered first.")
            raise ValueError(f"Unknown agent type requested: '{agent_type}'")

        # Resolve dependency order
        load_order = self.registry.resolve_dependency_tree(agent_type)
        logger.info(f"Dependency-aware creation order for '{agent_type}': {load_order}")

        # 2: Make the dependency loop functional by recursively calling create.
        for dep_name in load_order[:-1]: # All but the last one (the target agent)
            if dep_name not in self.active_agents:
                self.create(dep_name, shared_memory)

        metadata = self.registry.agents[agent_type]
        try:
            # Dynamically import the module and get the class
            module = importlib.import_module(metadata.module_path)
            agent_class = getattr(module, metadata.class_name)
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load agent class '{metadata.class_name}' from '{metadata.module_path}': {e}", exc_info=True)
            raise ImportError(f"Could not load agent class for '{agent_type}'.") from e

        # 3: Get the agent-specific configuration and merge with any runtime kwargs
        agent_config_key = f"{agent_type}_agent"
        agent_config = dict(self.global_config.get(agent_config_key, {}))
        agent_config.update(kwargs)

        try:
            constructor_params = inspect.signature(agent_class.__init__).parameters
            constructor_args = {
                "shared_memory": shared_memory,
                "agent_factory": self,
                "config": agent_config,
                **kwargs
            }
            for key, value in kwargs.items():
                if key in constructor_params:
                    constructor_args[key] = value
                else:
                    logger.debug(f"Skipping arg '{key}' for {agent_type} agent")
            
            agent_instance = agent_class(**constructor_args)
            logger.info(f"Successfully created instance of agent: '{agent_type}'")
            
            # Cache the newly created agent before returning it.
            self.active_agents[agent_type] = agent_instance

            if not any(hasattr(agent_instance, method) for method in ['predict', 'get_action', 'act']):
                raise TypeError(f"Agent {agent_type} must implement predict(), get_action(), or act() method")
                
            return agent_instance

        except TypeError as e:
            logger.error(
                f"Failed to create agent '{agent_type}' due to a TypeError. "
                f"Check if the constructor signature matches the provided arguments. Error: {e}"
            )
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while creating agent '{agent_type}': {e}", exc_info=True)
            raise

    def inspect_registered_agents(self) -> Dict[str, Dict[str, Any]]:
        """
        Inspects all registered agents for metadata consistency, import integrity, and instantiability.
    
        Returns:
            Dict[str, Dict[str, Any]]: A dictionary keyed by agent name with diagnostic info.
        """
        diagnostics = {}
    
        for name, metadata in self.registry.agents.items():
            info = {
                "status": "OK",
                "module_path": metadata.module_path,
                "class_name": metadata.class_name,
                "version": metadata.version,
                "issues": []
            }
    
            try:
                module = importlib.import_module(metadata.module_path)
                cls = getattr(module, metadata.class_name)
    
                if not issubclass(cls, BaseAgent):
                    info["status"] = "Warning"
                    info["issues"].append("Class is not a subclass of BaseAgent.")
    
                # Check constructor signature
                expected_args = {"shared_memory", "agent_factory", "config"}
                ctor_args = set(cls.__init__.__code__.co_varnames)
                missing = expected_args - ctor_args
    
                if missing:
                    info["status"] = "Warning"
                    info["issues"].append(f"Missing constructor args: {missing}")
    
            except (ImportError, AttributeError) as e:
                info["status"] = "Error"
                info["issues"].append(f"Import or attribute error: {e}")
    
            diagnostics[name] = info
    
        printer.pretty("Agent Diagnostics", diagnostics, "debug")
        return diagnostics

    def run_adaptation_cycle(self, metrics: Dict[str, Any], agent_types: list[str]):
        """
        Processes metrics through the adapter and applies adjustments to the global config.
        This affects the configuration of subsequently created agents.
        """
        logger.info("Running adaptation cycle based on new metrics...")
        
        # 1. Process metrics to get adjustments
        adjustments = self.metrics_adapter.process_metrics(metrics, agent_types)
        printer.pretty("Generated Adjustments", adjustments, "info")

        # 2. Apply adjustments to the factory's global configuration
        for key, adj_value_tensor in adjustments.items():
            adj_value = adj_value_tensor.item()
            
            # Example logic: "fairness_adjustment" -> adapt "risk_threshold"
            if "fairness_adjustment" in key:
                # Decrease risk threshold if fairness error is high (positive adjustment)
                target_param = 'risk_threshold'
                for agent_name in self.registry.agents.keys():
                    config_key = f"{agent_name}_agent"
                    if config_key in self.global_config and target_param in self.global_config[config_key]:
                        current_val = self.global_config[config_key][target_param]
                        # Apply adjustment defensively
                        new_val = max(0.01, current_val - adj_value * 0.1)
                        self.global_config[config_key][target_param] = new_val
                        logger.info(f"Adapted '{config_key}.{target_param}' from {current_val:.3f} to {new_val:.3f}")

            # Example logic: "performance_adjustment" -> adapt "learning_rate"
            if "performance_adjustment" in key:
                # Decrease learning rate if performance error is high (positive adjustment)
                target_param = 'learning_rate'
                for agent_name in self.registry.agents.keys():
                    config_key = f"{agent_name}_agent"
                    if config_key in self.global_config and target_param in self.global_config[config_key]:
                        current_val = self.global_config[config_key][target_param]
                        # Apply adjustment defensively
                        new_val = max(1e-6, current_val * (1 - adj_value * 0.05))
                        self.global_config[config_key][target_param] = new_val
                        logger.info(f"Adapted '{config_key}.{target_param}' from {current_val:.4f} to {new_val:.4f}")

        logger.info("Adaptation cycle complete. Global config updated.")

    def validate_with_azr(self, fact_tuple):
        self.bzr = BasicZeroReasoner()
        return 0.0
