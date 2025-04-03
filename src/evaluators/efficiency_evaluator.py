class EfficiencyEvaluator:
    def evaluate(self, outputs, ground_truths):
        # Example: shorter responses = more efficient (you can define your own logic)
        total_length = sum(len(str(o)) for o in outputs)
        return 1 / total_length if total_length > 0 else 0.0
