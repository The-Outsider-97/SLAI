import numpy as np
import yaml, json

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from typing import List, Any, Dict, Optional
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QSize
from datetime import datetime

from src.agents.evaluators.evaluators_memory import EvaluatorsMemory
from logs.logger import get_logger

logger = get_logger("Performance Evaluator")

CONFIG_PATH = "src/agents/evaluators/configs/evaluator_config.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_merged_config(user_config=None):
    base_config = load_config()
    if user_config:
        base_config.update(user_config)
    return base_config

class PerformanceEvaluator:
    def __init__(self, config):
        config = load_config() or {}
        self.config = config.get('performance_evaluator', {})
        memory = EvaluatorsMemory(config)
        self.memory = memory
        self.metric_weights = self.config.get('weights', {})
        self.classes = self.config.get('classes', None)
        self.average = self.config.get('average', 'macro')

        if self.classes is not None and not isinstance(self.classes, list):
            logger.warning(f"`classes` config should be a list, got: {type(self.classes)}")

        logger.info(f"Performance Evaluator succesfully initialized")

    def evaluate(self, outputs: List[Any], ground_truths: List[Any], report: bool = False) -> Dict[str, Any]:
        """Comprehensive performance assessment with detailed metric breakdown and optional reporting"""
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
            results['weighted_breakdown'] = {
                metric: self.metric_weights.get(metric, 0) * value
                for metric, value in results.items()
                if metric in self.metric_weights
            }
    
        # Report output
        if report:
            results['report'] = self.generate_report(results)
    
        # Store in memory if configured
        if self.config.get('store_results', False):
            self.memory.add(
                entry=results,
                tags=["performance_eval"],
                priority="medium"
            )
    
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
        if self.classes is not None:
            return confusion_matrix(truths, outputs, labels=self.classes).tolist()
        return confusion_matrix(truths, outputs).tolist()

    def _calculate_composite_score(self, results):
        return sum(
            self.metric_weights.get(metric, 0) * value 
            for metric, value in results.items()
            if metric in self.metric_weights
        )

    def generate_report(self, metrics: Dict[str, float]) -> str:
        from src.agents.evaluators.utils.report import get_visualizer
        report = []

        visualizer = get_visualizer()
        cm = np.array(metrics['confusion_matrix'])
        # Extract TN, FP, FN, TP by flattening the matrix
        tn, fp, fn, tp = cm.ravel()
        visualizer.add_metrics('performance', {
            'successes': tp + tn,  # True positives + true negatives
            'failures': fp + fn    # False positives + false negatives
        })
        

        # Header Section
        report.append(f"\n# Performance Evaluator Report\n")
        report.append(f"**Generated**: {datetime.now().isoformat()}\n")
        chart = visualizer.render_temporal_chart(QSize(600, 400), 'success_rate')
        report.append(f"![Performance Chart](data:image/png;base64,{visualizer._chart_to_base64(chart)})")

        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            value = metrics.get(metric, 0)
            report.append(f"- **{metric.title()}**: {value:.3f}")
    
        if 'composite_score' in metrics:
            report.append(f"- **Composite Score**: {metrics['composite_score']:.3f}")
            report.append("\n### Metric Weight Impact")
            for metric, weighted in metrics.get('weighted_breakdown', {}).items():
                report.append(f"  - {metric}: {weighted:.3f}")

        report.append(f"\n---\n*Report generated by {self.__class__.__name__}*")   # Footer with system info

        return "\n".join(report)

    def disable_temporarily(self):
        """Temporarily disable performance testing during degraded mode"""
        self.test_cases = []
        logger.warning("Performance Evaluator temporarily disabled.")

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Performance Evaluator ===\n")
    import sys
    app = QApplication(sys.argv)
    config = load_config()

    performance = PerformanceEvaluator(config)
    logger.info(f"{performance}")
    print(f"\n* * * * * Phase 2 * * * * *\n")
    outputs = [1, 0, 1, 1]
    ground_truths = [1, 0, 0, 1]
    report = True

    results = performance.evaluate(outputs, ground_truths, report)
    logger.info(f"{results}")
    if 'report' in results:
        print(results['report'])
    print(f"\n* * * * * Phase 3 * * * * *\n")

    print("\n=== Successfully Ran Performance Evaluator ===\n")
