from registry import AgentRegistry
from task_router import TaskRouter

self.router = TaskRouter(self.registry, shared_memory=self.shared_memory)

class CollaborationManager:
    """
    Central manager that registers agents and coordinates collaborative task execution
    using intelligent routing via TaskRouter and capability tracking in AgentRegistry.
    """

    def __init__(self, shared_memory=None):
        """
        Initialize the collaboration manager with a registry and a task router.

        Args:
            shared_memory (dict, optional): Shared in-memory structure for tracking stats.
        """
        self.shared_memory = shared_memory or {}
        self.registry = AgentRegistry()
        self.router = TaskRouter(self.registry, shared_memory=self.shared_memory)

    def register_agent(self, agent_name, agent_class, capabilities):
        """
        Register a new agent into the collaborative framework.

        Args:
            agent_name (str): Identifier for the agent.
            agent_class (object): Class or instance of the agent with an `execute()` method.
            capabilities (list[str]): List of task types this agent supports.
        """
        self.registry.register(agent_name, agent_class, capabilities)

    def run_task(self, task_type, task_data):
        """
        Route a task to the most suitable agent and return the result.

        Args:
            task_type (str): Type/category of task.
            task_data (dict): Payload needed for the task.

        Returns:
            Any: Result returned by the selected agent.
        """
        return self.router.route(task_type, task_data)

    def list_agents(self):
        """
        List all registered agents with their capabilities.

        Returns:
            dict: Mapping of agent names to their capability sets.
        """
        return self.registry.list_agents()

    def get_agent_stats(self):
        """
        Retrieve success/failure stats from shared memory.

        Returns:
            dict: Dictionary of agent performance metadata.
        """
        return self.shared_memory.get("agent_stats", {})

    def reset_agent_stats(self):
        """
        Clear all recorded agent performance data.
        """
        self.shared_memory["agent_stats"] = {}
