
import numpy as np
import yaml, json
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from typing import Dict, List, Tuple, Any
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QBuffer, QSize
from datetime import datetime
from dataclasses import dataclass

from src.agents.evaluators.utils.config_loader import load_global_config, get_config_section
from src.agents.evaluators.utils.evaluators_calculations import EvaluatorsCalculations
from src.agents.evaluators.utils.evaluation_errors import (ReportGenerationError, ValidationFailureError, 
                                                          MetricCalculationError, VisualizationError)
from src.agents.evaluators.utils.report import get_visualizer
from src.agents.evaluators.evaluators_memory import EvaluatorsMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Autonomous Evaluator")
printer = PrettyPrinter

@dataclass
class TaskMetrics:
    completion_time: float
    path_length: float
    energy_consumed: float
    success: bool
    collisions: int
    deviation_from_optimal: float

class AutonomousEvaluator:
    """Evaluator for planning systems, robotics, and automation tasks"""
    def __init__(self):
        self.config = load_global_config()
        self.eval_config = get_config_section('autonomous_evaluator')
        self.store_results = self.eval_config.get('store_results', True)
        
        # Load specialized thresholds
        self.thresholds = self.eval_config.get('thresholds', {
            'max_path_deviation': 1.2,
            'max_energy_per_task': 500,
            'max_collisions': 0,
            'min_success_rate': 0.95
        })
        
        # Metrics configuration
        self.metric_weights = self.eval_config.get('weights', {
            'success_rate': 0.4,
            'path_efficiency': 0.3,
            'energy_efficiency': 0.2,
            'collision_penalty': -0.5
        })
        
        # Visualization settings
        self.viz_config = self.eval_config.get('visualization', {
            'node_size': 300,
            'font_size': 8,
            'show_weights': True
        })
        
        self.memory = EvaluatorsMemory()
        self.calculations = EvaluatorsCalculations()

        self.task_history = []
        self.plan_graphs = []

        logger.info(f"Autonomous Evaluator initialized with thresholds: {self.thresholds}")

    def evaluate_task(self, task: Dict) -> TaskMetrics:
        """
        Evaluate a single planning/robotics task
        Args:
            task: Dictionary containing task details and results
        Returns:
            TaskMetrics object with evaluation results
        """
        # Validate task structure
        required_keys = ['completion_time', 'path', 'optimal_path', 'energy_consumed', 'collisions']
        if not all(key in task for key in required_keys):
            raise ValidationFailureError(
                rule_name="task_structure",
                data=task,
                expected=f"Task must contain keys: {required_keys}"
            )
        
        # Validate task structure with defaults for missing keys
        defaults = {
            'completion_time': 0.0,
            'path': [],
            'optimal_path': [],
            'energy_consumed': 0.0,
            'collisions': 0,
            'success': False
        }
        
        # Apply defaults for missing keys
        for key in defaults:
            if key not in task:
                task[key] = defaults[key]
                logger.warning(f"Missing key '{key}' in task, using default value")
            
        # Calculate path metrics
        path_length = self._calculate_path_length(task['path'])
        optimal_length = self._calculate_path_length(task['optimal_path'])
        deviation = path_length / optimal_length if optimal_length > 0 else float('inf')
        
        # Create metrics object
        metrics = TaskMetrics(
            completion_time=task['completion_time'],
            path_length=path_length,
            energy_consumed=task['energy_consumed'],
            success=task.get('success', True),
            collisions=task['collisions'],
            deviation_from_optimal=deviation
        )
        
        # Store task graph if available
        if 'plan_graph' in task:
            self.plan_graphs.append((task['id'], task['plan_graph']))
        
        self.task_history.append(metrics)
        return metrics

    def evaluate_task_set(self, tasks: List[Dict]) -> Dict[str, Any]:
        """Validate task structure before processing"""
        validated_tasks = []
        for task in tasks:
            # Add missing keys with defaults
            task.setdefault('optimal_path', [])
            task.setdefault('path', [])
            task.setdefault('energy_consumed', 0.0)
            task.setdefault('collisions', 0)
            task.setdefault('success', False)
            validated_tasks.append(task)
        
        # Process validated tasks
        return self._evaluate_valid_tasks(validated_tasks)
    
    def _evaluate_valid_tasks(self, tasks: List[Dict]) -> Dict[str, Any]:
        """Actual evaluation logic"""
        """
        Evaluate a set of planning/robotics tasks
        Args:
            tasks: List of task dictionaries
        Returns:
            Aggregate metrics dictionary
        """
        results = {
            'total_tasks': len(tasks),
            'successful_tasks': 0,
            'total_collisions': 0,
            'total_energy': 0,
            'total_path_length': 0,
            'total_optimal_length': 0,
            'task_metrics': []
        }
        
        for task in tasks:
            try:
                metrics = self.evaluate_task(task)
                results['task_metrics'].append(metrics)
                
                # Aggregate metrics
                results['successful_tasks'] += int(metrics.success)
                results['total_collisions'] += metrics.collisions
                results['total_energy'] += metrics.energy_consumed
                results['total_path_length'] += metrics.path_length
                results['total_optimal_length'] += metrics.path_length / metrics.deviation_from_optimal
                
            except Exception as e:
                logger.error(f"Task evaluation failed for task {task.get('id', 'unknown')}: {str(e)}")
        
        # Calculate aggregate metrics
        results['success_rate'] = results['successful_tasks'] / len(tasks) if tasks else 0
        results['path_efficiency'] = (
            results['total_optimal_length'] / results['total_path_length']
        ) if results['total_path_length'] > 0 else 0
        results['energy_efficiency'] = (
            (len(tasks) * self.thresholds['max_energy_per_task']) / results['total_energy']
        ) if results['total_energy'] > 0 else 0
        results['collision_rate'] = results['total_collisions'] / len(tasks) if tasks else 0
        
        # Composite score
        results['composite_score'] = (
            self.metric_weights['success_rate'] * results['success_rate'] +
            self.metric_weights['path_efficiency'] * results['path_efficiency'] +
            self.metric_weights['energy_efficiency'] * results['energy_efficiency'] +
            self.metric_weights['collision_penalty'] * results['collision_rate']
        )

        min_success = self.thresholds.get('min_success_rate', 0.95)  # Default value
        
        # Store results with safe access
        if self.config.get('store_results', False):
            try:
                priority = "high" if results['success_rate'] < min_success else "medium"
                self.memory.add(
                    entry=results,
                    tags=["planning_evaluation", "robotics"],
                    priority=priority
                )
            except Exception as e:
                logger.error(f"Memory storage failed: {str(e)}")
                
        return results

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive planning/robotics evaluation report"""
        try:
            report = []
            visualizer = get_visualizer()
            
            # Header Section
            report.append(f"\n# Planning & Robotics Evaluation Report\n")
            report.append(f"**Generated**: {datetime.now().isoformat()}\n")
            
            # Summary Metrics
            report.append("## Executive Summary\n")
            report.append(f"- **Tasks Evaluated**: {results['total_tasks']}")
            report.append(f"- **Success Rate**: {results['success_rate']:.2%}")
            report.append(f"- **Path Efficiency**: {results['path_efficiency']:.2f}x optimal")
            report.append(f"- **Energy Efficiency**: {results['energy_efficiency']:.2f}")
            report.append(f"- **Collision Rate**: {results['collision_rate']:.2f} per task")
            report.append(f"- **Composite Score**: {results['composite_score']:.2f}/1.0\n")
            
            # Path Visualization
            if self.plan_graphs:
                report.append("\n## Plan Visualization\n")
                try:
                    graph_data = self._render_plan_graph(self.plan_graphs[0][1])
                    report.append(f"![Plan Graph](data:image/png;base64,{graph_data})")
                except Exception as e:
                    logger.error(f"Graph rendering failed: {str(e)}")
                    report.append("*Plan visualization unavailable*")
            
            # Detailed Task Analysis
            report.append("\n## Detailed Task Metrics\n")
            report.append("| Metric | Average | Min | Max |")
            report.append("|--------|---------|-----|-----|")
            
            metrics = [
                ('Completion Time', 'completion_time'),
                ('Path Length', 'path_length'),
                ('Energy Consumed', 'energy_consumed'),
                ('Deviation from Optimal', 'deviation_from_optimal')
            ]
            
            for name, key in metrics:
                values = [getattr(m, key) for m in results['task_metrics']]
                if values:
                    report.append(
                        f"| {name} | {np.mean(values):.2f} | "
                        f"{min(values):.2f} | {max(values):.2f} |"
                    )
            
            # Failure Analysis
            if results['success_rate'] < 1.0:
                report.append("\n## Failure Analysis\n")
                failed_tasks = [t for t in results['task_metrics'] if not t.success]
                failure_reasons = {
                    'collisions': sum(t.collisions > 0 for t in failed_tasks),
                    'timeout': sum(t.completion_time > 30 for t in failed_tasks),  # Example threshold
                    'deviation': sum(t.deviation_from_optimal > self.thresholds['max_path_deviation'] for t in failed_tasks)
                }
                
                for reason, count in failure_reasons.items():
                    if count > 0:
                        report.append(f"- **{reason.replace('_', ' ').title()}**: {count} failures")
            
            # Recommendation Engine
            report.append("\n## Optimization Recommendations\n")
            if results['collision_rate'] > 0:
                report.append("- Implement collision prediction system")
                report.append("- Increase obstacle detection resolution")
            if results['path_efficiency'] < 0.9:
                report.append("- Optimize path planning algorithms")
                report.append("- Consider alternative search heuristics")
            if results['energy_efficiency'] < 0.8:
                report.append("- Implement energy-aware motion planning")
                report.append("- Optimize actuator control parameters")
            
            report.append(f"\n---\n*Report generated by {self.__class__.__name__}*")
            return "\n".join(report)
            
        except Exception as e:
            raise ReportGenerationError(
                report_type="Planning & Robotics",
                template="planning_report_template",
                error_details=f"Error generating report: {str(e)}"
            )

    def _calculate_path_length(self, path: List[Tuple[float, float]]) -> float:
        """Calculate total path length from waypoints"""
        if len(path) < 2:
            return 0.0
            
        total = 0.0
        for i in range(1, len(path)):
            x1, y1 = path[i-1]
            x2, y2 = path[i]
            total += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return total

    def _render_plan_graph(self, graph_data: Dict) -> str:
        """Render planning graph as base64 encoded image"""
        try:
            G = nx.DiGraph()
            
            # Add nodes
            for node in graph_data['nodes']:
                G.add_node(node['id'], **node.get('properties', {}))
                
            # Add edges
            for edge in graph_data['edges']:
                G.add_edge(edge['source'], edge['target'], weight=edge.get('weight', 1.0))
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            pos = nx.spring_layout(G)
            
            # Draw nodes
            node_colors = []
            for node in G.nodes(data=True):
                if node[1].get('start', False):
                    node_colors.append('green')
                elif node[1].get('goal', False):
                    node_colors.append('red')
                else:
                    node_colors.append('skyblue')
                    
            nx.draw_networkx_nodes(
                G, pos, 
                node_size=self.viz_config['node_size'],
                node_color=node_colors
            )
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20)
            
            # Draw labels
            nx.draw_networkx_labels(
                G, pos, 
                font_size=self.viz_config['font_size']
            )
            
            # Draw edge weights
            if self.viz_config['show_weights']:
                edge_labels = nx.get_edge_attributes(G, 'weight')
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            
            # Convert to base64
            canvas = FigureCanvasQTAgg(fig)
            buffer = QBuffer()
            buffer.open(QBuffer.ReadWrite)
            canvas.print_png(buffer)
            return bytes(buffer.data().toBase64()).decode('utf-8')
            
        except Exception as e:
            raise VisualizationError(
                chart_type="plan_graph",
                data=graph_data,
                error_details=f"Graph rendering failed: {str(e)}"
            )

    def disable_temporarily(self):
        """Temporarily disable evaluator during degraded mode"""
        self.task_history = []
        logger.warning("Autonomous Evaluator temporarily disabled.")

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Autonomous Evaluator ===\n")
    import sys
    app = QApplication(sys.argv)
    
    # Sample tasks
    tasks = [
        {
            'id': 'nav_001',
            'type': 'navigation',
            'path': [(0,0), (1,1), (2,2), (3,3)],
            'optimal_path': [(0,0), (3,3)],
            'completion_time': 12.5,
            'energy_consumed': 120,
            'collisions': 0,
            'success': True,
            'plan_graph': {
                'nodes': [
                    {'id': 'A', 'properties': {'start': True}},
                    {'id': 'B'},
                    {'id': 'C', 'properties': {'goal': True}}
                ],
                'edges': [
                    {'source': 'A', 'target': 'B', 'weight': 1.2},
                    {'source': 'B', 'target': 'C', 'weight': 0.8}
                ]
            }
        },
        {
            'id': 'manip_002',
            'type': 'manipulation',
            'path': [(0,0), (1,0), (1,1), (2,1)],
            'optimal_path': [(0,0), (2,1)],
            'completion_time': 18.2,
            'energy_consumed': 210,
            'collisions': 1,
            'success': False
        }
    ]
    
    evaluator = AutonomousEvaluator()
    results = evaluator.evaluate_task_set(tasks)
    
    print(f"Results: {results}")
    print(f"\nReport:\n{evaluator.generate_report(results)}")
    
    print("\n=== Evaluation Complete ===\n")
