import pytest
import numpy as np
from datetime import datetime, timedelta
from agents.eval_agent import EvaluationAgent, EvaluationResult
from evaluators.performance_evaluator import PerformanceEvaluator

class MockEvaluator:
    def evaluate(self, agent_outputs, ground_truths):
        return 0.8  # Fixed score for testing

@pytest.fixture
def eval_agent():
    agent = EvaluationAgent({
        'performance': MockEvaluator(),
        'custom': MockEvaluator()
    })
    return agent

def test_initialization(eval_agent):
    assert len(eval_agent.evaluators) == 2
    assert 'performance' in eval_agent.evaluators
    assert 'custom' in eval_agent.evaluators

def test_evaluate_agent(eval_agent):
    results = eval_agent.evaluate_agent('agent1', {}, {})
    assert isinstance(results, dict)
    assert 'performance' in results
    assert results['performance'].score == 0.8

def test_pareto_ranking():
    # Setup agents with different score profiles
    agent = EvaluationAgent()
    
    # Mock scores
    agent.scores = {
        'agent1': {
            'performance': [EvaluationResult(0.9, 'performance')],
            'efficiency': [EvaluationResult(0.6, 'efficiency')]
        },
        'agent2': {
            'performance': [EvaluationResult(0.7, 'performance')],
            'efficiency': [EvaluationResult(0.8, 'efficiency')]
        }
    }
    
    ranked = agent._pareto_ranking()
    assert len(ranked) == 2
    # Depending on implementation, verify ordering
    assert isinstance(ranked[0][0], str)

def test_statistical_significance():
    agent = EvaluationAgent()
    
    # Create historical data
    base_time = datetime.now()
    for i in range(5):
        time = base_time + timedelta(days=i)
        agent.scores.setdefault('agent1', {}).setdefault('performance', []).append(
            EvaluationResult(0.8 + i*0.01, 'performance', timestamp=time)
        )
        agent.scores.setdefault('agent2', {}).setdefault('performance', []).append(
            EvaluationResult(0.7 - i*0.01, 'performance', timestamp=time)
        )
    
    results = agent.statistical_significance_test('agent1', 'agent2', 'performance')
    assert 'p_value' in results
    assert results['significant'] in [True, False]

def test_visualization(eval_agent):
    # Add some test data
    eval_agent.scores = {
        'agent1': {
            'performance': [EvaluationResult(0.9, 'performance')],
            'custom': [EvaluationResult(0.7, 'custom')]
        },
        'agent2': {
            'performance': [EvaluationResult(0.8, 'performance')],
            'custom': [EvaluationResult(0.8, 'custom')]
        }
    }
    
    fig = eval_agent.visualize_metrics(plot_type='bar')
    assert fig is not None

def test_tracking_over_time(eval_agent):
    # Initial evaluation
    eval_agent.track_evaluation_over_time('agent1', {}, {})
    
    # Second evaluation
    results = eval_agent.track_evaluation_over_time('agent1', {}, {})
    
    assert len(eval_agent.scores['agent1']['performance']) == 2
    assert isinstance(results['performance'], EvaluationResult)
