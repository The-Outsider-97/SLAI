import numpy as np
import yaml, json

from datetime import datetime
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QApplication
from typing import List, Any, Dict, Optional

from src.agents.evaluators.utils.config_loader import load_global_config, get_config_section
from src.agents.evaluators.utils.evaluators_calculations import EvaluatorsCalculations
from src.agents.evaluators.utils.evaluation_errors import (EvaluationError, ReportGenerationError,
                                                           ConfigLoadError, ValidationFailureError)
from src.agents.evaluators.utils.report import get_visualizer
from src.agents.evaluators.evaluators_memory import EvaluatorsMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Performance Evaluator")
printer = PrettyPrinter

class PerformanceEvaluator:
    def __init__(self):
        self.config = load_global_config()
        self.perform_config = get_config_section('performance_evaluator')
        self.classes = self.perform_config.get('classes')
        self.average = self.perform_config.get('average')
        self.enable_composite_score = self.perform_config.get('enable_composite_score')
        self.store_results = self.perform_config.get('store_results')
        self.threshold = self.perform_config.get('threshold')
        self.metric_weights = self.perform_config.get('weights', {})
        self.metric_params = self.perform_config.get('metric_params', {})
        self.custom_metrics = self.perform_config.get('custom_metrics', {})

        self.memory = EvaluatorsMemory()
        self.calculations = EvaluatorsCalculations()
        self._validate_config()

        self.num_classes = len(self.classes) if self.classes else 0

        if self.classes is not None and not isinstance(self.classes, list):
            logger.warning(f"`classes` config should be a list, got: {type(self.classes)}")

        logger.info(f"Performance Evaluator succesfully initialized")

    def _validate_config(self):
        valid_averages = ['micro', 'macro', 'weighted', 'samples', None]
        if self.average not in valid_averages:
            raise ConfigLoadError(
                config_path="performance_evaluator",
                section="average",
                error_details=f"Invalid value '{self.average}'. Valid options: {valid_averages}"
            )
        
        if self.metric_weights:
            total_weight = sum(self.metric_weights.values())
            if not (0.99 <= total_weight <= 1.01):
                logger.warning(f"Metric weights sum to {total_weight}, not 1.0")

    def compare_with_baseline(self, current_metrics, baseline_metrics):
        improvements = {}
        for metric in current_metrics:
            if metric in baseline_metrics:
                diff = current_metrics[metric] - baseline_metrics[metric]
                improvements[metric] = {
                    'difference': diff,
                    'improvement': diff > 0
                }
        return improvements

    def evaluate(self, outputs: List[Any], ground_truths: List[Any], report: bool = False) -> Dict[str, Any]:
        """Comprehensive performance assessment with detailed metric breakdown and optional reporting"""
        if len(outputs) != len(ground_truths):
            raise ValidationFailureError(
                rule_name="output_ground_truth_length_match",
                data={"outputs_len": len(outputs), "ground_truths_len": len(ground_truths)},
                expected="Equal lengths"
            )

        try:
            if self.threshold is not None:
                if all(isinstance(x, (float, int)) for x in outputs):
                    outputs_bin = [1 if x >= self.threshold else 0 for x in outputs]
                else:
                    logger.warning("Threshold ignored - outputs not numeric")
                    outputs_bin = outputs
            else:
                outputs_bin = outputs
    
            results = {
                'accuracy': self.calculations._calculate_accuracy(outputs_bin, ground_truths),
                'precision': self.calculations._calculate_precision(outputs_bin, ground_truths),
                'recall': self.calculations._calculate_recall(outputs_bin, ground_truths),
                'f1': self.calculations._calculate_f1(outputs_bin, ground_truths),
                'confusion_matrix': self.calculations._calculate_confusion_matrix(outputs_bin, ground_truths)
            }
        
            # Add custom metrics
            for metric in self.custom_metrics:
                metric_name = metric['name']
                try:
                    # Dynamically import metric function
                    module_path, func_name = metric['function'].rsplit('.', 1)
                    module = __import__(module_path, fromlist=[func_name])
                    metric_fn = getattr(module, func_name)
                    params = self.metric_params.get(metric_name, {})
                    results[metric_name] = metric_fn(ground_truths, outputs_bin, **params)
                except (ImportError, AttributeError, KeyError) as e:
                    logger.error(f"Custom metric {metric_name} failed: {str(e)}")
                    results[metric_name] = None
            
            # Composite score calculation
            if self.enable_composite_score:
                results['composite_score'] = self.calculations._calculate_composite_score(results)
                results['weighted_breakdown'] = {
                    metric: self.metric_weights.get(metric, 0) * value
                    for metric, value in results.items()
                    if metric in self.metric_weights
                }
        
            # Report output
            if report:
                results['report'] = self.generate_report(results)
            if self.store_results:    # Store in memory if configured
                self.memory.add(entry=results, tags=["performance_eval"], priority="medium")

            return results
        except EvaluationError as e:
            printer.status("FAILED", f"Evaluation failed: {str(e)}", "error")
            # Log full error details including forensic information
            logger.error(f"EvaluationError: {e.to_audit_dict()}")
            return {
                'error': str(e), 
                'status': 'failed',
                'error_type': e.error_type.value,
                'error_id': e.error_id,
                'forensic_hash': e.forensic_hash
            }
        except Exception as e:
            printer.status("FAILED", f"Evaluation failed: {str(e)}", "error")
            return {'error': str(e), 'status': 'failed'}

    def generate_report(self, metrics: Dict[str, float]) -> str:
        try:
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
        except Exception as e:
            raise ReportGenerationError(
                report_type="Performance",
                template="performance_report_template",
                error_details=f"Error during report generation: {str(e)}"
            )

    def disable_temporarily(self):
        """Temporarily disable performance testing during degraded mode"""
        self.test_cases = []
        logger.warning("Performance Evaluator temporarily disabled.")

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Performance Evaluator ===\n")
    import sys
    app = QApplication(sys.argv)

    performance = PerformanceEvaluator()
    logger.info(f"{performance}")
    print(f"\n* * * * * Phase 2 * * * * *\n")
    outputs = [1, 0, 1, 1]
    ground_truths = [1, 0, 0, 1]
    report = True

    results = performance.evaluate(outputs, ground_truths, report)
    printer.pretty("FINAL", results, "success" if results else "error")
    if 'report' in results:
        print(results['report'])
    print(f"\n* * * * * Phase 3 * * * * *\n")

    print("\n=== Successfully Ran Performance Evaluator ===\n")
