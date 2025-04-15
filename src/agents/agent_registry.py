
class AgentRegistry:
    def __init__(self):
        self.registry = {}

    def register(self, agent_name, agent_class):
        self.registry[agent_name] = {
            "class": agent_class,
            "instance": None
        }

    def get(self, agent_name: str, shared_memory, agent_factory=None):
        entry = self.registry.get(agent_name)
        if not entry:
            raise ValueError(f"Agent '{agent_name}' not registered.")
        if entry["instance"] is None:
            entry["instance"] = entry["class"](shared_memory, agent_factory)
        return entry["instance"]

    def discover_agents(self):
        import inspect
        from . import base_agent  # or a central agent module

        for name, obj in inspect.getmembers(base_agent):
            if inspect.isclass(obj) and name.endswith("Agent"):
                self.register(name.lower(), obj)
