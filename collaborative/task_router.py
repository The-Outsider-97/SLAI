from collaborative.registry import AgentRegistry

class TaskRouter:
    def __init__(self, registry: AgentRegistry):
        self.registry = registry

    def route(self, task_type, task_data):
        """
        Route task to best-suited agent(s)
        """
        eligible_agents = self.registry.get_agents_by_task(task_type)

        if not eligible_agents:
            raise Exception(f"No agents found for task type '{task_type}'")

        # Select an agent - for now, just pick the first one
        selected_agent_name, selected_agent = next(iter(eligible_agents.items()))

        print(f"Routing task '{task_type}' to agent: {selected_agent_name}")

        # Run the agent's execute method (assuming interface)
        result = selected_agent.execute(task_data)
        return result
