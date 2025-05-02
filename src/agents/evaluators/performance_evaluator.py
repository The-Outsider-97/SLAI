import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from typing import List, Any, Dict, Optional

class PerformanceEvaluator:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.metric_weights = self.config.get(
            'weights', 
            {'accuracy': 0.4, 'precision': 0.3, 'recall': 0.3}
        )
        self.classes = self.config.get('classes', None)
        self.average = self.config.get('average', 'macro')

    def evaluate(self, outputs: List[Any], ground_truths: List[Any]) -> Dict[str, float]:
        """Comprehensive performance assessment with configurable metrics"""
        if len(outputs) != len(ground_truths):
            raise ValueError("Outputs and ground truths must have same length")
            
        results = {
            'accuracy': self._calculate_accuracy(outputs, ground_truths),
            'precision': self._calculate_precision(outputs, ground_truths),
            'recall': self._calculate_recall(outputs, ground_truths),
            'f1': self._calculate_f1(outputs, ground_truths),
            'confusion_matrix': self._calculate_confusion_matrix(outputs, ground_truths)
        }
        
        if self.config.get('enable_composite_score', True):
            results['composite_score'] = self._calculate_composite_score(results)
            
        return results

    def _calculate_accuracy(self, outputs, truths):
        correct = sum(o == g for o, g in zip(outputs, truths))
        return correct / len(truths) if truths else 0.0

    def _calculate_precision(self, outputs, truths):
        return precision_score(truths, outputs, 
                             average=self.average, 
                             zero_division=0)

    def _calculate_recall(self, outputs, truths):
        return recall_score(truths, outputs,
                          average=self.average,
                          zero_division=0)

    def _calculate_f1(self, outputs, truths):
        return f1_score(truths, outputs,
                       average=self.average,
                       zero_division=0)

    def _calculate_confusion_matrix(self, outputs, truths):
        return confusion_matrix(truths, outputs,
                               labels=self.classes).tolist()

    def _calculate_composite_score(self, results):
        return sum(
            self.metric_weights.get(metric, 0) * value 
            for metric, value in results.items()
            if metric in self.metric_weights
        )
