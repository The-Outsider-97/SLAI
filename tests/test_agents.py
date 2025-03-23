import pytest
import sys
import os
import torch
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.evolution_agent import EvolutionAgent
from utils.data_loader import generate_dummy_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================
# GLOBAL CONSTANTS
# =====================================
INPUT_SIZE = 10
OUTPUT_SIZE = 2
DEFAULT_CONFIG = {
    'hidden_sizes': [16, 32, 64],
    'learning_rate': 0.001,
    'population_size': 4,
    'elite_fraction': 0.5
}

# =====================================
# FIXTURES
# =====================================
@pytest.fixture(scope="module")
def config():
    return DEFAULT_CONFIG.copy()

@pytest.fixture(scope="function")
def agent(config):
    return EvolutionAgent(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, config=config)

@pytest.fixture(scope="function")
def data_loaders():
    train_loader = generate_dummy_data(batch_size=32, num_batches=5)
    val_loader = generate_dummy_data(batch_size=32, num_batches=2)
    return train_loader, val_loader

# =====================================
# CORE FUNCTIONALITY TESTS
# =====================================
def test_build_model(agent):
    model = agent.build_model()

    assert model is not None, "Model should not be None"
    assert hasattr(model, 'parameters'), "Model should have parameters() method"
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameter count: {param_count}")
    assert param_count > 0, "Model should have trainable parameters"

def test_train_and_evaluate(agent, data_loaders):
    train_loader, val_loader = data_loaders

    model = agent.build_model()
    assert model is not None, "Failed to build model"

    # Train model with expected valid data
    try:
        agent.train_model({'model': model, 'learning_rate': agent.learning_rate}, train_loader, val_loader)
    except Exception as e:
        pytest.fail(f"Training failed with exception: {e}")

    accuracy = agent.evaluate_model({'model': model}, val_loader)
    assert isinstance(accuracy, float), "Accuracy should be float"
    assert 0.0 <= accuracy <= 100.0, "Accuracy should be between 0 and 100"
    logger.info(f"Model evaluation accuracy: {accuracy}")

def test_population_initialization(agent):
    agent.initialize_population()
    
    assert hasattr(agent, 'population'), "Agent should have a population attribute"
    assert isinstance(agent.population, list), "Population should be a list"
    assert len(agent.population) == agent.population_size, "Population size mismatch"
    
    for individual in agent.population:
        assert 'model' in individual, "Each individual must have a model"
        assert isinstance(individual['learning_rate'], float), "Learning rate must be float"
        assert individual['hidden_size'] in agent.hidden_sizes, "Hidden size must be valid"

def test_model_forward_pass(agent):
    model = agent.build_model()

    dummy_input = torch.randn(1, INPUT_SIZE)
    try:
        output = model(dummy_input)
    except Exception as e:
        pytest.fail(f"Forward pass failed with exception: {e}")

    assert output is not None, "Output should not be None"
    assert output.shape[-1] == OUTPUT_SIZE, f"Expected output size {OUTPUT_SIZE}, got {output.shape[-1]}"

# =====================================
# MUTATION & EVOLUTION TESTS
# =====================================
def test_mutation_changes(agent):
    agent.initialize_population()
    parent = agent.population[0]
    mutation_happened = False

    for _ in range(20):  # Increase iterations to reduce randomness issues
        child = agent.mutate(parent)
        if (child['learning_rate'] != parent['learning_rate']) or (child['hidden_size'] != parent['hidden_size']):
            mutation_happened = True
            break
    
    assert mutation_happened, "Mutation should result in changes to learning_rate or hidden_size"

def test_mutate_invalid_individual(agent):
    invalid_parent = {'model': None, 'learning_rate': 'invalid'}
    with pytest.raises(ValueError):
        agent.mutate(invalid_parent)

def test_evolution_process(agent, data_loaders):
    train_loader, val_loader = data_loaders
    agent.initialize_population()
    
    initial_population_size = len(agent.population)
    agent.evolve_population(evaluator=None, train_loader=train_loader, val_loader=val_loader)

    assert len(agent.population) == initial_population_size, "Population size should remain constant after evolution"
    performances = [individual['performance'] for individual in agent.population]
    assert all(isinstance(p, float) for p in performances), "All performances should be float"

# =====================================
# EDGE CASES & INVALID CONFIGS
# =====================================
def test_zero_population_initialization():
    zero_pop_config = DEFAULT_CONFIG.copy()
    zero_pop_config['population_size'] = 0
    
    with pytest.raises(ValueError):
        EvolutionAgent(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, config=zero_pop_config)

def test_negative_learning_rate_initialization():
    bad_lr_config = DEFAULT_CONFIG.copy()
    bad_lr_config['learning_rate'] = -0.01
    
    with pytest.raises(ValueError):
        EvolutionAgent(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, config=bad_lr_config)

def test_invalid_hidden_sizes_initialization():
    bad_hidden_config = DEFAULT_CONFIG.copy()
    bad_hidden_config['hidden_sizes'] = [-10, 0, 16]

    with pytest.raises(ValueError):
        EvolutionAgent(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, config=bad_hidden_config)

# =====================================
# TEARDOWN / CLEANUP
# =====================================
@pytest.fixture(scope="module", autouse=True)
def teardown():
    yield
    logger.info("Teardown and cleanup complete.")
