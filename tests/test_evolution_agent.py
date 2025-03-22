import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.evolution_agent import EvolutionAgent
from evaluators.performance_evaluator import PerformanceEvaluator
from utils.data_loader import generate_dummy_data

@pytest.fixture
def config():
    return {
        'hidden_sizes': [16, 32, 64],
        'learning_rate': 0.001,
        'population_size': 4,
        'elite_fraction': 0.5
    }

@pytest.fixture
def agent(config):
    return EvolutionAgent(input_size=10, output_size=2, config=config)

def test_initialize_population(agent):
    agent.initialize_population()
    assert len(agent.population) == agent.population_size
    for individual in agent.population:
        assert 'model' in individual
        assert 'learning_rate' in individual

def test_mutate(agent):
    agent.initialize_population()
    parent = agent.population[0]
    child = agent.mutate(parent)

    assert child != parent
    assert child['learning_rate'] != 0  # should always be > 0

def test_evolve_population(agent):
    evaluator = PerformanceEvaluator(threshold=70.0)
    train_loader = generate_dummy_data()
    val_loader = generate_dummy_data()

    agent.initialize_population()
    agent.evolve_population(evaluator, train_loader, val_loader)

    assert len(agent.population) == agent.population_size
    performances = [ind['performance'] for ind in agent.population]
    assert all(isinstance(p, float) for p in performances)
