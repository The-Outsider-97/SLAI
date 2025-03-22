import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.evolution_agent import EvolutionAgent
from utils.data_loader import generate_dummy_data

@pytest.fixture
def agent():
    return EvolutionAgent(input_size=10, output_size=2)

def test_build_model(agent):
    model = agent.build_model()
    assert model is not None
    assert hasattr(model, 'parameters')

def test_train_and_evaluate(agent):
    train_loader = generate_dummy_data()
    val_loader = generate_dummy_data()
    
    model = agent.build_model()
    agent.train_model(model, train_loader, val_loader)
    
    accuracy = agent.evaluate_model(model, val_loader)
    
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 100
