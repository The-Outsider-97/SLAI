from maml_rl import MAMLAgent
from rsi_agent import RSIAgent
from dqn_agent import DQNAgent

class LearningAgent:
    def __init__(self, strategy="maml"):
        if strategy == "maml":
            self.model = MAMLAgent()
        elif strategy == "rsi":
            self.model = RSIAgent()
        elif strategy == "dqn":
            self.model = DQNAgent()
        else:
            raise ValueError("Invalid strategy.")

    def train(self, data):
        return self.model.train(data)

    def act(self, state):
        return self.model.act(state)

    def evaluate(self):
        return self.model.evaluate()
