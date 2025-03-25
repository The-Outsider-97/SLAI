class AgentRegistry:
    def __init__(self):
        self.registry = {}

    def register(self, agent_name, agent_class, capabilities=None):
        """
        Register an agent to the system.
        :param agent_name: str, unique identifier
        :param agent_class: class or instance of the agent
        :param capabilities: list of task types or strengths
        """
        self.registry[agent_name] = {
            'agent': agent_class,
            'capabilities': capabilities or []
        }

    def get_agent(self, agent_name):
        return self.registry.get(agent_name, {}).get('agent')

    def get_agents_by_task(self, task_type):
        """
        Return all agents capable of handling the given task type.
        """
        return {
            name: info['agent']
            for name, info in self.registry.items()
            if task_type in info['capabilities']
        }

    def list_agents(self):
        return list(self.registry.keys())
