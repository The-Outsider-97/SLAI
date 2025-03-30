class PerformanceEvaluator:
    def evaluate(self, outputs, ground_truths):
        correct = sum(o == g for o, g in zip(outputs, ground_truths))
        return correct / len(ground_truths) if ground_truths else 0.0
