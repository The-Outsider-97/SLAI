"""
Agent Factory for Dynamic Agent Creation

This implementation provides a robust factory pattern for creating all agent types
in the system with:
- Comprehensive error handling
- Lazy imports for better performance
- Configuration validation
- Automatic agent discovery
- Support for all agent types except BaseAgent

Academic References:
- Gamma et al. (1994) "Design Patterns: Factory Method"
- Fowler (2002) "Patterns of Enterprise Application Architecture"
"""

import importlib
from typing import Dict, Any, Type
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Agent registry mapping names to their module and class names
AGENT_REGISTRY = {
    "dqn": ("agents.dqn_agent", "NeuralNetwork"),
    "evolution": ("agents.evolution_agent", "EvolutionAgent"),
    "multitask": ("agents.multitask_rl", "MultiTaskRLAgent"),
    "maml": ("agents.maml_rl", "MAMLAgent"),
    "rsi": ("agents.rsi_agent", "RSI_Agent"),
    "rl_agent": ("agents.rl_agent", "RLAgent"),
    "safe_ai": ("agents.safe_ai_agent", "SafeAI_Agent"),
    "collaborative": ("agents.collaborative_agent", "CollaborativeAgent"),
    "nl_agent": ("agents.nl_agent", "LanguageAgent"),
    "reasoning": ("agents.reasoning_agent", "ReasoningAgent"),
    "planning": ("agents.planning_agent", "PlanningAgent"),
    "perception": ("agents.perception_agent", "PerceptionAgent"),
    "adaptive": ("agents.adaptive_agent", "AdaptiveAgent"),
    "evaluation": ("agents.eval_agent", "EvaluationAgent"),
    "execution": ("agents.execution_agent", "ExecutionAgent"),
    "knowledge": ("agents.knowledge_agent", "KnowledgeAgent")
}

def create_agent(agent_name: str, config: Dict[str, Any]) -> Any:
    """
    Factory method for creating agents by name with configuration.

    Args:
        agent_name: Name of the agent type to create (case-insensitive)
        config: Dictionary containing agent-specific configuration

    Returns:
        Instance of the requested agent

    Raises:
        ValueError: If agent name is unknown or required config is missing
        ImportError: If there's an issue importing the agent module
        TypeError: If agent initialization fails
    """
    agent_name = agent_name.lower()
    
    if agent_name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent type: {agent_name}. Available agents: {list(AGENT_REGISTRY.keys())}")

    module_name, class_name = AGENT_REGISTRY[agent_name]
    
    try:
        # Lazy import the required module
        module = importlib.import_module(module_name)
        agent_class = getattr(module, class_name)
        
        # Special initialization cases
        if agent_name == "evolutionary_dqn":
            return _create_evolutionary_dqn(agent_class, config)
        elif agent_name == "perception":
            return _create_perception_agent(agent_class, config)
        elif agent_name == "execution":
            return _create_execution_agent(agent_class, config)
            
        # Standard initialization
        return agent_class(**config)
        
    except ImportError as e:
        logger.error(f"Failed to import agent module {module_name}: {str(e)}")
        raise
    except AttributeError:
        logger.error(f"Agent class {class_name} not found in module {module_name}")
        raise
    except TypeError as e:
        logger.error(f"Agent initialization failed for {agent_name}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating agent {agent_name}: {str(e)}")
        raise

def _create_evolutionary_dqn(agent_class: Type, config: Dict[str, Any]) -> Any:
    """Special initialization for EvolutionaryDQNAgent."""
    import gym
    env = gym.make(config.get("env", "CartPole-v1"))
    return agent_class(
        env=env,
        state_size=config["state_size"],
        action_size=config["action_size"]
    )

def _create_perception_agent(agent_class: Type, config: Dict[str, Any]) -> Any:
    """Special initialization for PerceptionAgent."""
    return agent_class(
        device=config.get("device"),
        max_workers=config.get("max_workers")
    )

def _create_execution_agent(agent_class: Type, config: Dict[str, Any]) -> Any:
    """Special initialization for ExecutionAgent."""
    return agent_class(config=config.get("config"))

def discover_agents() -> Dict[str, Dict[str, str]]:
    """
    Discover available agents by scanning the agents directory.

    Returns:
        Dictionary mapping agent names to their metadata
    """
    agents_dir = Path(__file__).parent.parent / "agents"
    discovered = {}
    
    for agent_file in agents_dir.glob("*_agent.py"):
        agent_name = agent_file.stem.replace("_agent", "")
        if agent_name != "base":  # Skip base agent
            discovered[agent_name] = {
                "module": f"agents.{agent_file.stem}",
                "class": f"{agent_file.stem.replace('_', '').title()}"
            }
    
    return discovered

def validate_config(agent_name: str, config: Dict[str, Any]) -> bool:
    """
    Validate configuration for a specific agent type.

    Args:
        agent_name: Name of the agent type
        config: Configuration to validate

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If required parameters are missing
    """
    agent_name = agent_name.lower()
    
    # Common required parameters
    required_params = []
    
    # Agent-specific requirements
    if agent_name == "dqn":
        required_params = ["state_size", "action_size"]
    elif agent_name == "maml":
        required_params = ["state_size", "action_size", "hidden_size", 
                          "meta_lr", "inner_lr", "gamma"]
    elif agent_name == "rsi":
        required_params = ["state_size", "action_size", "shared_memory"]
    
    missing = [param for param in required_params if param not in config]
    if missing:
        raise ValueError(f"Missing required parameters for {agent_name}: {missing}")
    
    return True

def get_agent_documentation(agent_name: str) -> str:
    """
    Retrieve documentation for a specific agent type.

    Args:
        agent_name: Name of the agent type

    Returns:
        Agent class documentation string
    """
    agent_name = agent_name.lower()
    if agent_name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent type: {agent_name}")

    module_name, class_name = AGENT_REGISTRY[agent_name]
    
    try:
        module = importlib.import_module(module_name)
        agent_class = getattr(module, class_name)
        return agent_class.__doc__ or "No documentation available"
    except (ImportError, AttributeError):
        return "Documentation not available"

# Example usage
if __name__ == "__main__":
    # Example configuration for DQN agent
    dqn_config = {
        "state_size": 8,
        "action_size": 4,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995
    }
    
    try:
        agent = create_agent("dqn", dqn_config)
        print(f"Successfully created {agent.__class__.__name__} agent")
        print("Available agents:", list(AGENT_REGISTRY.keys()))
    except Exception as e:
        print(f"Agent creation failed: {str(e)}")
