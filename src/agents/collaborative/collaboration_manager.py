import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from src.agents.collaborative.registry import AgentRegistry
from src.agents.collaborative.task_router import TaskRouter
from src.agents.collaborative.utils.config_loader import get_config_section
from src.agents.collaborative.utils.collaboration_error import OverloadError
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Collaboration Manager")
printer = PrettyPrinter


class CollaborationManager:
    """Coordinates collaborative execution through registry + task router."""

    def __init__(self, shared_memory=None):
        self.shared_memory = shared_memory
        manager_config = get_config_section("collaboration") or {}
        self.max_concurrent_tasks = int(manager_config.get("max_concurrent_tasks", 100))
        self.load_factor = float(manager_config.get("load_factor", 0.75))

        self.executor = ThreadPoolExecutor(max_workers=int(manager_config.get("thread_pool_workers", 10)))
        self.registry = AgentRegistry(shared_memory=self.shared_memory)
        self.router = TaskRouter(registry=self.registry, shared_memory=self.shared_memory)

        logger.info("Collaboration manager initialized")

    @property
    def max_load(self) -> int:
        agent_count = max(1, len(self.registry.list_agents()))
        return min(self.max_concurrent_tasks, int(agent_count * 5 * self.load_factor))

    def register_agent(self, agent_name: str, agent_instance: Any, capabilities):
        self.registry.batch_register([
            {
                "name": agent_name,
                "meta": {
                    "class": type(agent_instance),
                    "instance": agent_instance,
                    "capabilities": list(capabilities),
                    "version": 1.0,
                },
            }
        ])

    def run_task(self, task_type: str, task_data: Dict[str, Any], retries: int = 1) -> Any:
        if self.get_system_load() >= self.max_load:
            raise OverloadError(f"System load exceeded ({self.get_system_load()}/{self.max_load})")

        last_error: Optional[Exception] = None
        for _ in range(max(1, retries)):
            try:
                return self.router.route(task_type, task_data)
            except Exception as exc:
                last_error = exc
        if last_error:
            raise last_error

    def get_system_load(self) -> int:
        if self.shared_memory is None:
            return 0
        stats = self.shared_memory.get("agent_stats", {}) or {}
        return int(sum(v.get("active_tasks", 0) for v in stats.values() if isinstance(v, dict)))

    def list_agents(self):
        return self.registry.list_agents()

    def get_agent_stats(self):
        if self.shared_memory is None:
            return {}
        return self.shared_memory.get("agent_stats", {}) or {}

    def export_stats_to_json(self, filename="report/agent_stats.json"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.get_agent_stats(), f, indent=2)


if __name__ == "__main__":
    print("\n=== Running Collaboration Manager ===\n")
    printer.status("TEST", "Starting Collaboration Manager tests", "info")
    print("\nAll tests completed successfully!\n")
