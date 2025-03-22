# evaluators/performance_evaluator.py

class PerformanceEvaluator:
    """
    Evaluate performance metrics and decide if a model is worth keeping.
    """

    def __init__(self, threshold=70.0):
        self.threshold = threshold

    def is_better(self, performance, best_performance):
        """
        Return True if performance is better.
        """
        return performance > best_performance

    def meets_threshold(self, performance):
        """
        Check if model passes minimum performance.
        """
        return performance >= self.threshold
