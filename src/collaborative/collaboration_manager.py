
import os
import time
import json
import logging
import threading

from concurrent.futures import ThreadPoolExecutor

from src.collaborative.registry import AgentRegistry
from src.collaborative.task_router import TaskRouter
from logs.logger import get_logger

logger = get_logger("Collaboration Manager")

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
        self.registry.batch_register([{
            "name": agent_name,
            "meta": {
                "class": type(agent_class),
                "instance": agent_class,
                "capabilities": capabilities,
                "version": 1.0
            }
        }])

    def get_system_load(self):
        stats = self.shared_memory.get("agent_stats", default={})
        if not isinstance(stats, dict):
            return 0
        return sum(agent_meta.get("active_tasks", 0) for agent_meta in stats.values())

    @property
    def MAX_LOAD(self):
        """Dynamic threshold based on registered agents"""
        return min(
            self.MAX_CONCURRENT_TASKS,
            int(len(self.registry.list_agents()) * 5 * self.LOAD_FACTOR)  # 5 tasks/agent avg
        )

    def run_task(self, task_type, task_data, retries=3):
        current_load = self.get_system_load()
        
        if current_load >= self.MAX_LOAD:
            raise OverloadError(
                f"System load {current_load}/{self.MAX_LOAD} exceeded. "
                f"Available agents: {len(self.list_agents())}"
            )

        for attempt in range(retries):
            try:
                return self.router.route(task_type, task_data)
            except Exception as e:
                if attempt == retries-1:
                    raise
                logging.warning(f"Retry {attempt+1}/{retries} for {task_type}")

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
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.shared_memory.get("agent_stats", {}), f, indent=2)

class OverloadError(Exception):
    """Custom exception for system capacity limits"""
    def __init__(self, message="System at maximum capacity. Please retry later"):
        super().__init__(message)
        self.throttle_time = 30 

if __name__ == "__main__":
    import time
    from src.collaborative.shared_memory import SharedMemory
    from src.agents.collaborative_agent import CollaborativeAgent
    from src.utils.agent_factory import AgentFactory

    # Initialize components
    shared_memory = SharedMemory(config={
        'max_memory_mb': 100,
        'max_versions': 10,
        'ttl_check_interval': 30,
        'network_latency': 0.0
    })
    agent_factory = AgentFactory(
        config={},
        shared_resources={"shared_memory": shared_memory}
        )
    manager = CollaborationManager(shared_memory)

    # Define mock agents
    class TranslationAgent(CollaborativeAgent):
        capabilities = ["translation"]
        def __init__(self):
            super().__init__(shared_memory, agent_factory)
        def execute(self, task_data):
            return f"Translated: {task_data['text']}"

    class AnalysisAgent(CollaborativeAgent):
        capabilities = ["analysis"]
        def __init__(self):
            super().__init__(shared_memory, agent_factory)
        def execute(self, task_data):
            return {"status": "analyzed", "result": 42}

    class FailingAgent(CollaborativeAgent):
        capabilities = ["flaky_task"]
        def __init__(self):
            super().__init__(shared_memory, agent_factory)
        def execute(self, task_data):
            raise Exception("Intentional failure")

    class RetrySimpleTrainerAgent(CollaborativeAgent):
        capabilities = ["retry_simple_trainer"]
        def __init__(self):
            super().__init__(shared_memory, agent_factory)
        def execute(self, task_data):
            return "Fallback task succeeded"

    # Register agents
    manager.register_agent("Translator", TranslationAgent(), ["translation"])
    manager.register_agent("Analyzer", AnalysisAgent(), ["analysis"])

    print("--- Testing Valid Task ---")
    try:
        result = manager.run_task("translation", {"text": "Hello World"})
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Testing Overload ---")
    max_load = manager.MAX_LOAD
    print(f"MAX_LOAD: {max_load}")
    shared_memory.set("agent_stats", {
        "Translator": {"active_tasks": max_load, "successes": 0, "failures": 0}
    })
    try:
        manager.run_task("analysis", {"data": "test"})
    except OverloadError as e:
        print(f"Overload correctly handled: {e}")
    shared_memory.set("agent_stats", {})

    print("\n--- Testing Retries & Fallback ---")
    manager.register_agent("FlakyAgent", FailingAgent(), ["flaky_task"])
    manager.register_agent("RetryAgent", RetrySimpleTrainerAgent(), ["retry_simple_trainer"])
    try:
        result = manager.run_task("train_model", {"model": "CNN"})
        print(f"Fallback Result: {result}")
    except Exception as e:
        print(f"Task failed: {e}")

    print("\n--- Agent Stats ---")
    print(manager.get_agent_stats())

    manager.export_stats_to_json()
    print("\nTests completed.")
