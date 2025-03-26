from shared_memory import SharedMemory

class AgentRegistry:
    """
    Centralized registry for agents and their capabilities.
    Also integrates with SharedMemory for distributed coordination.
    """

    def __init__(self, shared_memory: SharedMemory = None):
        """
        Args:
            shared_memory (SharedMemory): Optional shared memory instance.
        """
        self._agents = {}
        self.shared_memory = shared_memory or SharedMemory()

    def register(self, name, agent_class, capabilities):
        """
        Register a new agent and update shared memory.

        Args:
            name (str): Unique agent identifier.
            agent_class (object): Agent instance or class with an `execute()` method.
            capabilities (list[str]): Task types this agent can perform.
        """
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

        # Store in shared memory for system-wide awareness
        self.shared_memory.set(f"agent:{name}", {
            "status": "active",
            "capabilities": capabilities
        })

    def unregister(self, name):
        """
        Remove an agent from the registry and shared memory.

        Args:
            name (str): Agent name.
        """
        if name not in self._agents:
            raise KeyError(f"Agent '{name}' is not registered.")

        del self._agents[name]
        self.shared_memory.delete(f"agent:{name}")

    def update_status(self, name, status: str):
        """
        Update the status of an agent in shared memory.

        Args:
            name (str): Agent identifier.
            status (str): Status to broadcast (e.g., 'busy', 'idle', 'offline').
        """
        if name not in self._agents:
            raise KeyError(f"Agent '{name}' not registered.")

        agent_key = f"agent:{name}"
        record = self.shared_memory.get(agent_key) or {}
        record["status"] = status
        self.shared_memory.set(agent_key, record)

    def get_status(self, name):
        """
        Get the current shared memory status of an agent.

        Args:
            name (str): Agent name.

        Returns:
            str: Status string (e.g., 'active', 'busy', etc.)
        """
        return self.shared_memory.get(f"agent:{name}").get("status", "unknown")

    def get_agents_by_task(self, task_type):
        """
        Return all agents that support a given task_type.

        Args:
            task_type (str): Task label to match against agent capabilities.

        Returns:
            dict[str, dict]: Dictionary of agent names and metadata that support the task_type.
        """
        return {
            name: agent for name, agent in self._agents.items()
            if task_type in agent.get("capabilities", [])
        }

    def list_agents(self):
        """
        Return a list of all registered agents with capabilities.

        Returns:
            dict[str, list[str]]
        """
        return {name: info["capabilities"] for name, info in self._agents.items()}

    def get_agent_class(self, name):
        """
        Get agent class instance by name.

        Args:
            name (str): Agent name.

        Returns:
            object: Registered agent class instance.
        """
        if name not in self._agents:
            raise KeyError(f"Agent '{name}' not found.")
        return self._agents[name]["class"]
