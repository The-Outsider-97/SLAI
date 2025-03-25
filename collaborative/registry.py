class AgentRegistry:
    """
    Registry system to track and manage agents and their supported capabilities.
    Each agent is associated with a set of task types it can handle.
    """

    def __init__(self):
        # Internal storage for registered agents
        # Format: {agent_name: {"class": agent_class, "capabilities": [task_type1, task_type2, ...]}}
        self._agents = {}

    def register(self, name, agent_class, capabilities):
        """
        Register a new agent.

        Args:
            name (str): Unique identifier for the agent.
            agent_class (object): Class or instance of the agent with an `execute()` method.
            capabilities (list[str]): Task types the agent can perform.

        Raises:
            ValueError: If the agent name is already registered or input is invalid.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Agent name must be a non-empty string.")
        if name in self._agents:
            raise ValueError(f"Agent '{name}' is already registered.")
        if not hasattr(agent_class, "execute") or not callable(agent_class.execute):
            raise ValueError("Agent class must implement an 'execute' method.")
        if not isinstance(capabilities, list) or not all(isinstance(c, str) for c in capabilities):
            raise ValueError("Capabilities must be a list of strings.")

        self._agents[name] = {
            "class": agent_class,
            "capabilities": capabilities
        }

    def get_agent_for_task(self, task_type):
        """
        Retrieve the first agent capable of handling the given task type.

        Args:
            task_type (str): Type of task to match.

        Returns:
            tuple[str, object]: Agent name and instance.

        Raises:
            LookupError: If no suitable agent is found.
        """
        for name, info in self._agents.items():
            if task_type in info["capabilities"]:
                return name, info["class"]
        raise LookupError(f"No agent found capable of handling task type: '{task_type}'")

    def list_agents(self):
        """
        Return a dictionary of all registered agents and their capabilities.

        Returns:
            dict[str, list[str]]: Mapping of agent names to their capability lists.
        """
        return {name: info["capabilities"] for name, info in self._agents.items()}

    def get_agent_class(self, name):
        """
        Get the agent class/instance for a given name.

        Args:
            name (str): Agent identifier.

        Returns:
            object: Registered agent instance or class.

        Raises:
            KeyError: If the agent name is not registered.
        """
        if name not in self._agents:
            raise KeyError(f"Agent '{name}' is not registered.")
        return self._agents[name]["class"]

    def unregister(self, name):
        """
        Remove an agent from the registry.

        Args:
            name (str): Agent identifier.

        Raises:
            KeyError: If the agent is not found.
        """
        if name not in self._agents:
            raise KeyError(f"Agent '{name}' not found.")
        del self._agents[name]
