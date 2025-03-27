from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Abstract base class for all agents. Every agent should implement these methods.
    """

    def __init__(self):
        self.history = []

    @abstractmethod
    def build_model(self):
        """
        Build a new model (to be defined in subclasses).
        """
        pass

    @abstractmethod
    def select_action(self, state):
        """Selects an action based on the given state."""
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        """Trains the agent."""
        pass

    @abstractmethod
    def evaluate_model(self, model, val_loader):
        """
        Evaluate the model and return a performance metric.
        """
        pass
