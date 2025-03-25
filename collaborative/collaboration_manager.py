from registry import AgentRegistry
from task_router import TaskRouter

self.router = TaskRouter(self.registry, shared_memory=self.shared_memory)

class CollaborationManager:
    def __init__(self):
        self.registry = AgentRegistry()
        self.router = TaskRouter(self.registry)

    def register_agent(self, agent_name, agent_class, capabilities):
        self.registry.register(agent_name, agent_class, capabilities)

    def run_task(self, task_type, task_data):
        result = self.router.route(task_type, task_data)
        return result

    def list_agents(self):
        return self.registry.list_agents()
