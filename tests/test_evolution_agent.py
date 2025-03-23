import pytest
import sys
import os
import torch
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.evolution_agent import EvolutionAgent
from evaluators.performance_evaluator import PerformanceEvaluator
from utils.data_loader import generate_dummy_data

# =============================
# FIXTURES
# =============================

@pytest.fixture(scope="module")
def config():
    return {
        'hidden_sizes': [16, 32, 64],
        'learning_rate': 0.001,
        'population_size': 4,
        'elite_fraction': 0.5
    }

@pytest.fixture(scope="module")
def agent(config):
    return EvolutionAgent(input_size=10, output_size=2, config=config)

@pytest.fixture(scope="module")
def evaluator():
    return PerformanceEvaluator(threshold=70.0)

@pytest.fixture(scope="module")
def data_loaders():
    train_loader = generate_dummy_data(batch_size=32, num_batches=5)
    val_loader = generate_dummy_data(batch_size=32, num_batches=2)
    return train_loader, val_loader

# =============================
# TEST CASES
# =============================

def test_initialize_population(agent):
    agent.initialize_population()

    assert isinstance(agent.population, list), "Population must be a list"
    assert len(agent.population) == agent.population_size, "Population size mismatch"

    for ind in agent.population:
        assert 'model' in ind, "Individual missing 'model' key"
        assert 'learning_rate' in ind, "Individual missing 'learning_rate' key"
        assert isinstance(ind['learning_rate'], float), "Learning rate must be a float"

def test_mutate(agent):
    agent.initialize_population()

    parent = agent.population[0]
    child = agent.mutate(parent)

    assert isinstance(child, dict), "Mutated child must be a dict"
    assert 'learning_rate' in child, "Child missing 'learning_rate'"
    assert child['learning_rate'] != parent['learning_rate'], "Learning rate should change after mutation"
    assert 'model' in child, "Child missing 'model' after mutation"

def test_evolve_population(agent, evaluator, data_loaders):
    train_loader, val_loader = data_loaders

    agent.initialize_population()

    try:
        agent.evolve_population(evaluator, train_loader, val_loader)
    except Exception as e:
        pytest.fail(f"Evolve population failed with exception: {e}")

    assert len(agent.population) == agent.population_size, "Population size mismatch after evolution"
    performances = [ind.get('performance') for ind in agent.population]

    assert all(isinstance(p, float) for p in performances), "All performances should be floats"
    assert all(p >= 0 for p in performances), "Performances should be non-negative"
    print(f"Evolution performances: {performances}")

def test_elite_selection(agent, evaluator, data_loaders):
    train_loader, val_loader = data_loaders
    agent.initialize_population()

    agent.evolve_population(evaluator, train_loader, val_loader)

    sorted_population = sorted(agent.population, key=lambda x: x['performance'], reverse=True)
    elite_count = int(agent.elite_fraction * agent.population_size)

    assert elite_count > 0, "Elite count should be positive"
    assert sorted_population[0]['performance'] >= sorted_population[-1]['performance'], "Elite selection order invalid"

def test_invalid_population_config():
    # Edge case: negative population size
    invalid_config = {
        'hidden_sizes': [16, 32, 64],
        'learning_rate': 0.001,
        'population_size': -5,
        'elite_fraction': 0.5
    }

    with pytest.raises(ValueError):
        EvolutionAgent(input_size=10, output_size=2, config=invalid_config)

def test_mutate_with_invalid_individual(agent):
    # Edge case: invalid parent structure
    invalid_parent = {
        'learning_rate': "invalid_value",
        'model': None
    }

    with pytest.raises(Exception):
        agent.mutate(invalid_parent)

def test_model_performance(agent, evaluator, data_loaders):
    train_loader, val_loader = data_loaders
    model = agent.build_model()

    agent.train_model(model, train_loader, val_loader)
    accuracy = agent.evaluate_model(model, val_loader)

    assert isinstance(accuracy, float), "Accuracy must be a float"
    assert 0 <= accuracy <= 100, "Accuracy should be between 0 and 100"

    performance = evaluator.evaluate(accuracy)

    assert isinstance(performance, float), "Performance should be a float"
    assert performance >= 0, "Performance should be non-negative"
    print(f"Model performance: {performance}")

def test_population_after_multiple_generations(agent, evaluator, data_loaders):
    train_loader, val_loader = data_loaders
    generations = 5

    agent.initialize_population()

    for _ in range(generations):
        agent.evolve_population(evaluator, train_loader, val_loader)

    assert len(agent.population) == agent.population_size, "Population size mismatch after multiple generations"
    performances = [ind.get('performance') for ind in agent.population]

    assert all(isinstance(p, float) for p in performances), "Each individual should have float performance after evolution"
    print(f"Performances after {generations} generations: {performances}")

# =============================
# TEARDOWN / CLEANUP (Optional)
# =============================

@pytest.fixture(scope="module", autouse=True)
def cleanup():
    yield
    # Add cleanup tasks if needed
    print("Teardown complete")
