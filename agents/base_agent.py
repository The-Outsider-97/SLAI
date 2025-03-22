# agents/base_agent.py

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
    def train_model(self, model, train_loader, val_loader):
        """
        Train the model on given data.
        """
        pass

    @abstractmethod
    def evaluate_model(self, model, val_loader):
        """
        Evaluate the model and return a performance metric.
        """
        pass
