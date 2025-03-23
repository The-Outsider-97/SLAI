# agents/evolution_agent.py

import torch.nn as nn
import torch.optim as optim
import torch
from agents.base_agent import BaseAgent
import random
import copy

class EvolutionAgent(BaseAgent):
    """
    Agent that evolves neural network architectures and hyperparameters.
    """

    def __init__(self, input_size=10, output_size=2, config=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.config = config or {}

        self.hidden_sizes = config.get('hidden_sizes', [16, 32, 64, 128])
        self.learning_rate = config.get('learning_rate', 0.001)
        self.population = []  # list of models
        self.population_size = config.get('population_size', 5)
        self.elite_fraction = config.get('elite_fraction', 0.4)

        if self.learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.learning_rate}")

        if self.population_size <= 0:
            raise ValueError(f"Population size must be positive, got {self.population_size}")

        if not isinstance(self.hidden_sizes, list) or not all(isinstance(x, int) and x > 0 for x in self.hidden_sizes):
            raise ValueError(f"hidden_sizes must be a list of positive integers, got {self.hidden_sizes}")

        if not (0 < self.elite_fraction <= 1):
            raise ValueError(f"elite_fraction must be between 0 and 1, got {self.elite_fraction}")

    def build_model(self, hidden_size=None):
        if not hidden_size:
            hidden_size = random.choice(self.hidden_sizes)

        model = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.output_size)
        )
        return model

    def initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            model = self.build_model()
            self.population.append({
                'model': model,
                'hidden_size': model[0].out_features,
                'learning_rate': self.learning_rate,
                'performance': 0
            })

    def mutate(self, parent):
        """
        Mutate a given model (architecture or hyperparameters).
        """
        child = copy.deepcopy(parent)

        mutation_type = random.choice(['hidden_size', 'learning_rate'])
        if mutation_type == 'hidden_size':
            new_hidden_size = random.choice(self.hidden_sizes)
            child['model'] = self.build_model(hidden_size=new_hidden_size)
            child['hidden_size'] = new_hidden_size

        elif mutation_type == 'learning_rate':
            factor = random.choice([0.5, 1.5])
            child['learning_rate'] *= factor
            child['learning_rate'] = max(1e-6, min(1.0, child['learning_rate']))  # bounds

        return child

    def evolve_population(self, evaluator, train_loader, val_loader):
        """
        Evolve population: train, evaluate, select, mutate.
        """
        logger = evaluator.logger if hasattr(evaluator, 'logger') else None

        # Train and evaluate current population
        for individual in self.population:
            self.train_model(individual, train_loader, val_loader)
            individual['performance'] = self.evaluate_model(individual, val_loader)

        # Sort by performance
        self.population.sort(key=lambda x: x['performance'], reverse=True)
        elites = self.population[:int(self.elite_fraction * self.population_size)]

        if logger:
            logger.info(f"Elites selected with top performance: {elites[0]['performance']:.2f}%")

        # Create new generation via mutation
        new_population = elites.copy()
        while len(new_population) < self.population_size:
            parent = random.choice(elites)
            child = self.mutate(parent)
            new_population.append(child)

        self.population = new_population

    def train_model(self, individual, train_loader, val_loader):
        """
        Train individual model using its learning rate.
        """
        model = individual['model']
        lr = individual['learning_rate']

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model.train()
        for epoch in range(3):  # limit epochs for speed
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def evaluate_model(self, individual, val_loader):
        """
        Evaluate accuracy.
        """
        model = individual['model']
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
