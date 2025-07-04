import time
import psutil
import hashlib
import yaml, json

from pathlib import Path
from typing import Dict, List, Any
from jsonschema import ValidationError, validate
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QSize
from datetime import datetime

from src.agents.evaluators.utils.config_loader import load_global_config, get_config_section
from src.agents.evaluators.utils.evaluators_calculations import EvaluatorsCalculations
from src.agents.evaluators.utils.evaluation_errors import (EvaluationError, ReportGenerationError,
                                                           ConfigLoadError, MetricCalculationError,
                                                           MemoryAccessError, VisualizationError)
from src.agents.evaluators.utils.report import get_visualizer
from src.agents.evaluators.evaluators_memory import EvaluatorsMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Resource Utilization Evaluator")
printer = PrettyPrinter

class ResourceUtilizationEvaluator:
    def __init__(self):
        self.config = load_global_config()
        self.rue_config = get_config_section('resource_utilization_evaluator')
        self.monitor_duration = self.rue_config.get('monitor_duration')
        self.enable_historical = self.rue_config.get('enable_historical')
        self.store_results = self.rue_config.get('store_results')
        self.thresholds = self.rue_config.get('thresholds', {})
        self.weights = self.rue_config.get('weights', {})

        # Validate thresholds configuration
        required_thresholds = ['cpu', 'memory', 'disk', 'network']
        for metric in required_thresholds:
            if metric not in self.thresholds:
                raise ConfigLoadError(
                    config_path="resource_utilization_evaluator",
                    section="thresholds",
                    error_details=f"Missing threshold for {metric}"
                )

        self.memory = EvaluatorsMemory()
        self.calculations = EvaluatorsCalculations()

        logger.info(f"Resource Utilization Evaluator succesfully initialized")

    def evaluate(self, report: bool = False) -> Dict[str, Any]:
        """Comprehensive system resource analysis with utilization scoring"""
        try:
            results = {
                'timestamp': datetime.now().isoformat(),
                'metrics': {},
                'health_status': {}
            }

            # Gather metrics with error handling
            try:
                results['metrics'] = self._gather_metrics()
            except Exception as e:
                raise MetricCalculationError(
                    metric_name="resource_metrics",
                    inputs={},
                    reason=f"Failed to gather metrics: {str(e)}"
                )

            # Calculate scores with error handling
            try:
                results['scores'] = self._calculate_scores(results['metrics'])
                results['composite_score'] = sum(
                    self.weights.get(metric, 0) * results['scores'].get(metric, 0)
                    for metric in self.weights
                )
            except Exception as e:
                raise MetricCalculationError(
                    metric_name="utilization_scores",
                    inputs=results['metrics'],
                    reason=f"Failed to calculate scores: {str(e)}"
                )

            # Generate health status indicators
            for metric in self.thresholds:
                value = results['metrics'].get(metric, 0)
                threshold = self.thresholds[metric]
                try:
                    results['health_status'][metric] = (
                        "CRITICAL" if value > threshold
                        else "WARNING" if value > threshold * 0.8
                        else "NORMAL"
                    )
                except Exception as e:
                    logger.error(f"Health status calculation failed for {metric}: {str(e)}")
                    results['health_status'][metric] = "UNKNOWN"

            # Store results if configured
            if self.store_results:
                try:
                    self.memory.add(
                        entry=results,
                        tags=["resource_analysis", f"eval_{datetime.now().date()}"],
                        priority="high" if "CRITICAL" in results['health_status'].values() else "medium"
                    )
                except Exception as e:
                    raise MemoryAccessError(
                        operation="add",
                        key="resource_metrics",
                        error_details=str(e)
                    )

            if report:
                try:
                    results['report'] = self.generate_report(results)
                except Exception as e:
                    logger.error(f"Report generation failed: {str(e)}")
                    # Return partial results even if report fails
                    results['report_error'] = str(e)

            return results
        except EvaluationError as e:
            logger.error(f"Resource evaluation failed: {e.to_audit_dict()}")
            return {
                'error': str(e),
                'error_type': type(e).__name__,
                'error_id': getattr(e, 'error_id', 'N/A'),
                'forensic_hash': getattr(e, 'forensic_hash', 'N/A')
            }
        except Exception as e:
            logger.error(f"Unexpected error during resource evaluation: {str(e)}")
            return {
                'error': str(e),
                'error_type': 'UnexpectedError'
            }

    def _gather_metrics(self) -> Dict[str, float]:
        """Collect system resource metrics with temporal monitoring"""
        metrics = {}
        
        # CPU measurement
        try:
            metrics['cpu'] = self._measure_cpu()
        except Exception as e:
            logger.error(f"CPU measurement failed: {str(e)}")
            metrics['cpu'] = -1.0  # Error indicator
            
        # Memory measurement
        try:
            metrics['memory'] = self._measure_memory()
        except Exception as e:
            logger.error(f"Memory measurement failed: {str(e)}")
            metrics['memory'] = -1.0
            
        # Disk measurement
        try:
            metrics['disk'] = self._measure_disk()
        except Exception as e:
            logger.error(f"Disk measurement failed: {str(e)}")
            metrics['disk'] = -1.0
            
        # Network measurement
        try:
            metrics['network'] = self._measure_network()
        except Exception as e:
            logger.error(f"Network measurement failed: {str(e)}")
            metrics['network'] = -1.0
        
        # Historical data
        if self.enable_historical:
            try:
                metrics['cpu_hist'] = self._get_historical('cpu')
                metrics['memory_hist'] = self._get_historical('memory')
            except Exception as e:
                logger.warning(f"Historical data retrieval failed: {str(e)}")
            
        return metrics

    def _measure_cpu(self) -> float:
        """Measure CPU usage with duration averaging"""
        try:
            start = psutil.cpu_percent(percpu=True)
            time.sleep(self.monitor_duration)
            end = psutil.cpu_percent(percpu=True)
            return round((sum(end) - sum(start)) / len(end), 2)
        except Exception as e:
            raise MetricCalculationError(
                metric_name="cpu_utilization",
                inputs={"monitor_duration": self.monitor_duration},
                reason=str(e)
            )

    def _measure_memory(self) -> float:
        """Measure system memory utilization"""
        try:
            return round(psutil.virtual_memory().percent, 2)
        except Exception as e:
            raise MetricCalculationError(
                metric_name="memory_utilization",
                inputs={},
                reason=str(e)
            )

    def _measure_disk(self) -> float:
        """Measure primary disk usage"""
        try:
            return round(psutil.disk_usage('/').percent, 2)
        except Exception as e:
            raise MetricCalculationError(
                metric_name="disk_utilization",
                inputs={"path": "/"},
                reason=str(e)
            )

    def _measure_network(self) -> float:
        """Measure network bandwidth utilization"""
        try:
            start = psutil.net_io_counters()
            time.sleep(1)
            end = psutil.net_io_counters()
            return round((end.bytes_sent + end.bytes_recv - 
                         start.bytes_sent - start.bytes_recv) / 1e6, 2)  # MB/s
        except Exception as e:
            raise MetricCalculationError(
                metric_name="network_utilization",
                inputs={"duration": 1},
                reason=str(e)
            )

    def _get_historical(self, metric: str) -> List[float]:
        """Retrieve historical data from memory"""
        try:
            return self.memory.query(
                tags=["resource_analysis"],
                filters=[f"metrics.{metric}:exists"],
                limit=10
            )
        except Exception as e:
            raise MemoryAccessError(
                operation="query",
                key=f"historical_{metric}",
                error_details=str(e)
            )
    
    def _calculate_scores(self, metrics: Dict) -> Dict[str, float]:
        """Calculate normalized resource efficiency scores"""
        scores = {}
        try:
            for metric in ['cpu', 'memory', 'disk', 'network']:
                value = metrics.get(metric, 0)
                threshold = self.thresholds.get(metric, 100)
                try:
                    # Handle error values
                    if value < 0:
                        scores[metric] = 0.0
                    else:
                        scores[metric] = max(0, 1 - value/threshold)
                except Exception as e:
                    logger.error(f"Score calculation failed for {metric}: {str(e)}")
                    scores[metric] = 0.0
        except Exception as e:
            raise MetricCalculationError(
                metric_name="resource_scores",
                inputs=metrics,
                reason=str(e)
            )
        return scores

    def generate_report(self, results: Dict) -> str:
        """Generate comprehensive resource utilization report"""
        try:
            report = [
                "# Resource Utilization Report",
                f"**Generated**: {results['timestamp']}\n",
                "## System Health Status\n"
            ]
            visualizer = get_visualizer()
            
            # Health status indicators
            for metric, status in results.get('health_status', {}).items():
                icon = "🟢" if status == "NORMAL" else "🟡" if status == "WARNING" else "🔴"
                report.append(f"- {icon} **{metric.upper()}**: {status}")

            # Detailed metrics
            report.append("\n## Resource Metrics\n")
            metrics = results.get('metrics', {})
            for metric in ['cpu', 'memory', 'disk', 'network']:
                value = metrics.get(metric, -1)
                if value < 0:  # Error indicator
                    report.append(f"- **{metric.title()}**: Measurement failed")
                else:
                    threshold = self.thresholds.get(metric, 'N/A')
                    report.append(f"- **{metric.title()}**: {value}% (Threshold: {threshold}%)")

            # Visualization
            try:
                chart = visualizer.render_temporal_chart(QSize(600, 400), 'resource_usage')
                report.append(f"![Resource Chart](data:image/png;base64,{visualizer._chart_to_base64(chart)})")
            except Exception as e:
                raise VisualizationError(
                    chart_type="temporal",
                    data=results.get('metrics', {}),
                    error_details=f"Chart rendering failed: {str(e)}"
                )

            # Utilization scores
            report.append("\n## Utilization Scores\n")
            scores = results.get('scores', {})
            for metric in ['cpu', 'memory', 'disk', 'network']:
                score = scores.get(metric, 0)
                report.append(f"- **{metric.title()} Efficiency**: {score:.2f}/1.0")

            # Composite score
            composite = results.get('composite_score', 0)
            report.append(f"\n---\n*Composite Score: {composite:.2f}/1.0*")
            report.append(f"*Report generated by {self.__class__.__name__}*")
            
            return "\n".join(report)
        except Exception as e:
            raise ReportGenerationError(
                report_type="Resource Utilization",
                template="resource_report",
                error_details=f"Error generating report: {str(e)}"
            )

    def disable_temporarily(self):
        """Temporarily disable resource testing during degraded mode"""
        self.test_cases = []
        logger.warning("Resource Utilization Evaluator temporarily disabled.")

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Resource Utilization Evaluator ===\n")
    import sys
    app = QApplication(sys.argv)
    
    try:
        evaluator = ResourceUtilizationEvaluator()
        results = evaluator.evaluate(report=True)
        
        if 'report' in results:
            print(results['report'])
        elif 'error' in results:
            printer.pretty("Evaluation failed", results, "error")
            
    except Exception as e:
        printer.pretty("Fatal error during evaluation", str(e), "error")
    
    print("\n=== Resource Utilization Evaluation Complete ===\n")
