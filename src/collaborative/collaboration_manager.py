import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from src.collaborative.registry import AgentRegistry
from src.collaborative.task_router import TaskRouter

@property
def MAX_LOAD(self):
    """Dynamic threshold based on registered agents"""
    return min(
        self.MAX_CONCURRENT_TASKS,
        int(len(self.registry.list_agents()) * 5 * self.LOAD_FACTOR)  # 5 tasks/agent avg
    )

class CollaborationManager:
    """
    Central manager that registers agents and coordinates collaborative task execution
    using intelligent routing via TaskRouter and capability tracking in AgentRegistry.
    """

    # System-wide capacity configuration
    MAX_CONCURRENT_TASKS = 100  # Based on avg agent capacity
    LOAD_FACTOR = 0.75  # 75% of max capacity threshold

    def __init__(self, shared_memory):
        """
        Initialize the collaboration manager with a registry and a task router.

        Args:
            shared_memory (dict, optional): Shared in-memory structure for tracking stats.
        """
        self.shared_memory = shared_memory
        self.registry = AgentRegistry(shared_memory=self.shared_memory)
        self.router = TaskRouter(self.registry, shared_memory=self.shared_memory)
        self.registry.discover_agents()
        self.health_check_interval = 60  # seconds
        self._init_health_monitor()
        self.executor = ThreadPoolExecutor(max_workers=10)

    async def run_task_async(self, task_type, task_data):
        return await self.loop.run_in_executor(self.executor, self.run_task, task_type, task_data)

    def _init_health_monitor(self):
        def monitor():
            while True:
                self._check_agent_availability()
                time.sleep(self.health_check_interval)
        threading.Thread(target=monitor, daemon=True).start()

    def _check_agent_availability(self):
        for agent_name in self.registry.list_agents():
            agent_stats = self.shared_memory.get("agent_stats", {})
            agent_stats[agent_name]["last_heartbeat"] = time.time()
            self.shared_memory.set("agent_stats", agent_stats)

    def register_agent(self, agent_name, agent_class, capabilities):
        """
        Register a new agent into the collaborative framework.

        Args:
            agent_name (str): Identifier for the agent.
            agent_class (object): Class or instance of the agent with an `execute()` method.
            capabilities (list[str]): List of task types this agent supports.
        """
        self.registry.register(agent_name, agent_class, capabilities)

    def get_system_load(self):
        return sum(
            agent_meta.get("active_tasks", 0) 
            for agent_meta in self.shared_memory.get("agent_stats", {}).values()
        )

    def run_task(self, task_type, task_data, retries=3):
        current_load = self.get_system_load()
        
        if current_load >= self.MAX_LOAD:
            raise OverloadError(
                f"System load {current_load}/{self.MAX_LOAD} exceeded. "
                f"Available agents: {len(self.list_agents())}"
            )

        if self.get_system_load() > MAX_LOAD:
            raise OverloadError("System at capacity")

        for attempt in range(retries):
            try:
                return self.router.route(task_type, task_data)
            except Exception as e:
                if attempt == retries-1:
                    raise
                logging.warning(f"Retry {attempt+1}/{retries} for {task_type}")
        """
        Route the task to the best-fit agent based on capabilities and context.
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

    def export_stats_to_json(self, filename="report/agent_stats.json"):
        import json
        with open(filename, 'w') as f:
            json.dump(self.shared_memory.get("agent_stats", {}), f, indent=2)

class OverloadError(Exception):
    """Custom exception for system capacity limits"""
    def __init__(self, message="System at maximum capacity. Please retry later"):
        super().__init__(message)
        self.throttle_time = 30 
