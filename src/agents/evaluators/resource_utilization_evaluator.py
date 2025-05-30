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

from src.agents.evaluators.evaluators_memory import EvaluatorsMemory
from logs.logger import get_logger

logger = get_logger("Resource Utilization Evaluator")

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

class ResourceUtilizationEvaluator:
    def __init__(self, config):
        config = load_config() or {}
        self.config = config.get('resource_utilization_evaluator', {})
        memory = EvaluatorsMemory(config)
        self.memory = memory

        # Configuration parameters
        self.thresholds = self.config.get('thresholds', {})
        self.weights = self.config.get('weights', {})

        self.monitor_duration = self.config.get('monitor_duration', 5)  # Seconds
        self.enable_historical = self.config.get('enable_historical', True)

        logger.info(f"Resource Utilization Evaluator succesfully initialized")

    def evaluate(self, report: bool = False) -> Dict[str, Any]:
        """Comprehensive system resource analysis with utilization scoring"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'metrics': self._gather_metrics(),
            'health_status': {}
        }

        # Calculate utilization scores
        results['scores'] = self._calculate_scores(results['metrics'])
        results['composite_score'] = sum(
            self.weights[metric] * results['scores'][metric]
            for metric in self.weights
        )

        # Generate health status indicators
        for metric in self.thresholds:
            results['health_status'][metric] = (
                "CRITICAL" if results['metrics'][metric] > self.thresholds[metric]
                else "WARNING" if results['metrics'][metric] > self.thresholds[metric] * 0.8
                else "NORMAL"
            )

        # Store results if configured
        if self.config.get('store_results', False):
            self.memory.add(
                entry=results,
                tags=["resource_analysis", f"eval_{datetime.now().date()}"],
                priority="high" if "CRITICAL" in results['health_status'].values() else "medium"
            )

        if report:
            results['report'] = self.generate_report(results)

        return results

    def _gather_metrics(self) -> Dict[str, float]:
        """Collect system resource metrics with temporal monitoring"""
        metrics = {
            'cpu': self._measure_cpu(),
            'memory': self._measure_memory(),
            'disk': self._measure_disk(),
            'network': self._measure_network()
        }
        
        if self.enable_historical:
            metrics.update({
                'cpu_hist': self._get_historical('cpu'),
                'memory_hist': self._get_historical('memory')
            })
            
        return metrics

    def _measure_cpu(self) -> float:
        """Measure CPU usage with duration averaging"""
        start = psutil.cpu_percent(percpu=True)
        time.sleep(self.monitor_duration)
        end = psutil.cpu_percent(percpu=True)
        return round((sum(end) - sum(start)) / len(end), 2)

    def _measure_memory(self) -> float:
        """Measure system memory utilization"""
        return round(psutil.virtual_memory().percent, 2)

    def _measure_disk(self) -> float:
        """Measure primary disk usage"""
        return round(psutil.disk_usage('/').percent, 2)

    def _measure_network(self) -> float:
        """Measure network bandwidth utilization"""
        start = psutil.net_io_counters()
        time.sleep(1)
        end = psutil.net_io_counters()
        return round((end.bytes_sent + end.bytes_recv - 
                     start.bytes_sent - start.bytes_recv) / 1e6, 2)  # MB/s

    def _calculate_scores(self, metrics: Dict) -> Dict[str, float]:
        """Calculate normalized resource efficiency scores"""
        return {
            'cpu': max(0, 1 - metrics['cpu']/self.thresholds['cpu']),
            'memory': max(0, 1 - metrics['memory']/self.thresholds['memory']),
            'disk': max(0, 1 - metrics['disk']/self.thresholds['disk']),
            'network': max(0, 1 - metrics['network']/self.thresholds['network'])
        }

    def _get_historical(self, metric: str) -> List[float]:
        """Retrieve historical data from memory (simplified example)"""
        return self.memory.query(
            tags=["resource_analysis"],
            filters=[f"metrics.{metric}:exists"],
            limit=10
        )

    def generate_report(self, results: Dict) -> str:
        """Generate comprehensive resource utilization report"""
        from src.agents.evaluators.utils.report import get_visualizer
        report = [
            "# Resource Utilization Report",
            f"**Generated**: {results['timestamp']}\n",
            "## System Health Status\n"
        ]
        visualizer = get_visualizer()
        if 'descriptive_stats' in results:
            visualizer.add_metrics("descriptive_stats", results['descriptive_stats'])


        # Health status indicators
        for metric, status in results['health_status'].items():
            icon = "🟢" if status == "NORMAL" else "🟡" if status == "WARNING" else "🔴"
            report.append(f"- {icon} **{metric.upper()}**: {status}")

        # Detailed metrics
        report.append("\n## Resource Metrics\n")
        for metric, value in results['metrics'].items():
            if '_hist' not in metric:  # Exclude historical data from main metrics
                report.append(
                    f"- **{metric.title()}**: {value}% "
                    f"(Threshold: {self.thresholds.get(metric, 'N/A')}%)"
                )
        chart = visualizer.render_temporal_chart(QSize(600, 400), 'success_rate')
        report.append(f"![Statistical Chart](data:image/png;base64,{visualizer._chart_to_base64(chart)})")

        # Utilization scores
        report.append("\n## Utilization Scores\n")
        for metric, score in results['scores'].items():
            report.append(f"- **{metric.title()} Efficiency**: {score:.2f}/1.0")

        report.append(f"\n---\n*Composite Score: {results['composite_score']:.2f}/1.0*")
        report.append(f"*Report generated by {self.__class__.__name__}*")
        
        return "\n".join(report)

    def disable_temporarily(self):
        """Temporarily disable resource testing during degraded mode"""
        self.test_cases = []
        logger.warning("Resource Utilization Evaluator temporarily disabled.")

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Resource Utilization Evaluator ===\n")
    import sys
    app = QApplication(sys.argv)
    config = load_config()
    
    evaluator = ResourceUtilizationEvaluator(config)
    results = evaluator.evaluate(report=True)
    
    if 'report' in results:
        print(results['report'])
    print(f"\n* * * * * Phase 2 * * * * *\n")

    print(f"\n* * * * * Phase 3 * * * * *\n")

    print("\n=== Successfully Ran Resource Utilization Evaluator ===\n")
