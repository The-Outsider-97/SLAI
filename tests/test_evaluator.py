import pytest
from evaluators.performance_evaluator import PerformanceEvaluator

@pytest.fixture
def evaluator():
    return PerformanceEvaluator(threshold=70.0)

def test_meets_threshold(evaluator):
    assert evaluator.meets_threshold(75.0) == True
    assert evaluator.meets_threshold(65.0) == False

def test_is_better(evaluator):
    assert evaluator.is_better(80.0, 70.0) == True
    assert evaluator.is_better(60.0, 70.0) == False
