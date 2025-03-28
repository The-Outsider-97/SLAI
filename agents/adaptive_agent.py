class AdaptationAgent:
    def __init__(self):
        self.shared_memory = {}

    def update_memory(self, key: str, value):
        self.shared_memory[key] = value

    def retrieve_memory(self, key: str):
        return self.shared_memory.get(key, None)

    def route_message(self, message: str, routing_table: dict):
        for condition, agent in routing_table.items():
            if condition in message:
                return agent.handle_message(message)
        return "No suitable agent found."
