# agents/evolution_agent.py

import torch.nn as nn
import torch.optim as optim
import torch
from agents.base_agent import BaseAgent
import random

class EvolutionAgent(BaseAgent):
    """
    Agent that evolves neural network architectures via random mutation.
    """

    def __init__(self, input_size=10, output_size=2):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

    def build_model(self):
        """
        Build a random feedforward neural network.
        """
        hidden_size = random.choice([16, 32, 64, 128])
        model = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.output_size)
        )
        return model

    def train_model(self, model, train_loader, val_loader):
        """
        Train the model with basic training loop.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(3):  # small number of epochs for testing
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def evaluate_model(self, model, val_loader):
        """
        Evaluate accuracy.
        """
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy
