
import numpy as np
import yaml, json
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

logger = get_logger("Safety Evaluator")
printer = PrettyPrinter

@dataclass
class SafetyMetrics:
    risk_level: float
    hazard_detection_time: float
    emergency_stop_time: float
    safety_margin: float
    collision_avoidance: bool
    standards_compliance: float

class SafetyEvaluator:
    """Evaluator for safety-critical aspects of robotics and automation systems"""
    def __init__(self):
        self.config = load_global_config()
        self.eval_config = get_config_section('safety_evaluator')
        
        # Load safety thresholds
        self.thresholds = self.eval_config.get('thresholds', {
            'max_risk_level': 0.3,
            'max_hazard_detection_time': 0.5,
            'max_emergency_stop_time': 0.2,
            'min_safety_margin': 0.5,
            'min_standards_compliance': 0.95
        })
        
        # Safety standards configuration
        self.safety_standards = self.eval_config.get('safety_standards', [
            'ISO 13849', 'IEC 61508', 'ANSI/RIA R15.06'
        ])
        
        # Metrics configuration
        self.metric_weights = self.eval_config.get('weights', {
            'risk_level': 0.3,
            'hazard_detection': 0.25,
            'emergency_response': 0.25,
            'standards_compliance': 0.2
        })
        
        # Risk categories
        self.risk_categories = self.eval_config.get('risk_categories', [
            'collision', 'pinch_point', 'crush_hazard',
            'electrical', 'environmental', 'control_failure'
        ])
        
        self.memory = EvaluatorsMemory()
        self.calculations = EvaluatorsCalculations()

        self.safety_incidents = []
        self.hazard_data = []
        self.raw_incidents = []
        self.compliance_history = []
        self.rule_based_checks = True

        logger.info(f"SafetyEvaluator initialized with standards: {self.safety_standards}")

    def evaluate_incident(self, incident: Dict) -> SafetyMetrics:
        """
        Evaluate a safety incident or scenario
        Args:
            incident: Dictionary containing safety incident details
        Returns:
            SafetyMetrics object with evaluation results
        """
        # Validate incident structure
        required_keys = ['risk_assessment', 'hazard_detection_time', 
                         'emergency_stop_time', 'safety_margin', 'collision_avoided']
        if not all(key in incident for key in required_keys):
            raise ValidationFailureError(
                rule_name="incident_structure",
                data=incident,
                expected=f"Incident must contain keys: {required_keys}"
            )
            
        # Calculate risk level (weighted average of risk categories)
        risk_vector = np.array([incident['risk_assessment'].get(cat, 0) 
                              for cat in self.risk_categories])
        risk_level = np.mean(risk_vector)
        
        # Calculate standards compliance
        standards_vector = np.array([incident.get(std, False) 
                                   for std in self.safety_standards])
        standards_compliance = np.mean(standards_vector)
        
        # Create metrics object
        metrics = SafetyMetrics(
            risk_level=risk_level,
            hazard_detection_time=incident['hazard_detection_time'],
            emergency_stop_time=incident['emergency_stop_time'],
            safety_margin=incident['safety_margin'],
            collision_avoidance=incident['collision_avoided'],
            standards_compliance=standards_compliance
        )
        
        # Store detailed hazard data if available
        if 'hazard_details' in incident:
            self.hazard_data.append(incident['hazard_details'])
        
        self.safety_incidents.append(metrics)
        self.raw_incidents.append(incident)
        self.compliance_history.append(standards_compliance)
        return metrics

    def evaluate_operation(self, incidents: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate a set of safety incidents from an operational scenario
        Args:
            incidents: List of incident dictionaries
        Returns:
            Aggregate safety metrics dictionary
        """
        results = {
            'total_incidents': len(incidents),
            'critical_incidents': 0,
            'collisions_prevented': 0,
            'total_risk': 0,
            'total_detection_time': 0,
            'total_stop_time': 0,
            'total_safety_margin': 0,
            'incident_metrics': []
        }
        
        for incident in incidents:
            try:
                metrics = self.evaluate_incident(incident)
                results['incident_metrics'].append(metrics)
                
                # Aggregate metrics
                results['critical_incidents'] += int(metrics.risk_level > self.thresholds['max_risk_level'])
                results['collisions_prevented'] += int(metrics.collision_avoidance)
                results['total_risk'] += metrics.risk_level
                results['total_detection_time'] += metrics.hazard_detection_time
                results['total_stop_time'] += metrics.emergency_stop_time
                results['total_safety_margin'] += metrics.safety_margin
                
            except Exception as e:
                logger.error(f"Incident evaluation failed: {str(e)}")
        
        # Calculate aggregate metrics
        if incidents:
            results['avg_risk_level'] = results['total_risk'] / len(incidents)
            results['avg_detection_time'] = results['total_detection_time'] / len(incidents)
            results['avg_stop_time'] = results['total_stop_time'] / len(incidents)
            results['avg_safety_margin'] = results['total_safety_margin'] / len(incidents)
            results['compliance_rate'] = np.mean(self.compliance_history[-len(incidents):])
        else:
            results.update({
                'avg_risk_level': 0,
                'avg_detection_time': 0,
                'avg_stop_time': 0,
                'avg_safety_margin': 0,
                'compliance_rate': 0
            })
        
        # Safety effectiveness scores
        results['detection_score'] = max(0, 1 - (results['avg_detection_time'] / self.thresholds['max_hazard_detection_time']))
        results['response_score'] = max(0, 1 - (results['avg_stop_time'] / self.thresholds['max_emergency_stop_time']))
        results['safety_score'] = results['avg_safety_margin'] / self.thresholds['min_safety_margin']
        
        # Composite safety score
        results['composite_score'] = (
            self.metric_weights['risk_level'] * (1 - results['avg_risk_level']) +
            self.metric_weights['hazard_detection'] * results['detection_score'] +
            self.metric_weights['emergency_response'] * results['response_score'] +
            self.metric_weights['standards_compliance'] * results['compliance_rate']
        )
        
        # Risk categorization
        if self.hazard_data:
            results['risk_distribution'] = self.calculations._calculate_risk_distribution()
        
        # Store results
        if self.config.get('store_results', False):
            try:
                self.memory.add(
                    entry=results,
                    tags=["safety_evaluation", "robotics"],
                    priority="high" if results['avg_risk_level'] > self.thresholds['max_risk_level'] else "medium"
                )
            except Exception as e:
                logger.error(f"Memory storage failed: {str(e)}")
                
        return results

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive safety evaluation report"""
        try:
            report = []
            visualizer = get_visualizer()
            
            # Header Section
            report.append(f"\n# Safety Evaluation Report\n")
            report.append(f"**Generated**: {datetime.now().isoformat()}\n")
            
            # Summary Metrics
            report.append("## Executive Summary\n")
            report.append(f"- **Incidents Evaluated**: {results['total_incidents']}")
            report.append(f"- **Critical Incidents**: {results['critical_incidents']}")
            report.append(f"- **Collisions Prevented**: {results['collisions_prevented']}")
            report.append(f"- **Average Risk Level**: {results['avg_risk_level']:.2f}")
            report.append(f"- **Compliance Rate**: {results['compliance_rate']:.2%}")
            report.append(f"- **Safety Score**: {results['composite_score']:.2f}/1.0\n")
            
            # Risk Visualization
            if 'risk_distribution' in results:
                report.append("\n## Risk Distribution\n")
                try:
                    risk_chart = self._render_risk_distribution(results['risk_distribution'])
                    report.append(f"![Risk Distribution](data:image/png;base64,{risk_chart})")
                except Exception as e:
                    logger.error(f"Risk visualization failed: {str(e)}")
                    report.append("*Risk visualization unavailable*")
            
            # Standards Compliance
            report.append("\n## Standards Compliance\n")
            for standard in self.safety_standards:
                compliant_count = sum(incident.get(standard, False) for incident in self.raw_incidents)
                report.append(f"- **{standard}**: {compliant_count}/{results['total_incidents']} "
                              f"({compliant_count/results['total_incidents']:.1%})")
            
            # Timeline Analysis
            if self.compliance_history:
                report.append("\n## Compliance Trend\n")
                try:
                    timeline_chart = self._render_compliance_timeline()
                    report.append(f"![Compliance Trend](data:image/png;base64,{timeline_chart})")
                except Exception as e:
                    logger.error(f"Timeline rendering failed: {str(e)}")
            
            # Critical Incident Analysis
            if results['critical_incidents'] > 0:
                report.append("\n## Critical Incident Analysis\n")
                report.append("| Category | Count | Avg Risk |")
                report.append("|----------|-------|----------|")
                
                for category in self.risk_categories:
                    category_incidents = [inc for inc in self.safety_incidents 
                                        if inc.risk_level > self.thresholds['max_risk_level'] 
                                        and category in inc.risk_assessment]
                    if category_incidents:
                        avg_risk = np.mean([inc.risk_assessment[category] 
                                          for inc in category_incidents])
                        report.append(f"| {category.replace('_', ' ').title()} | "
                                    f"{len(category_incidents)} | {avg_risk:.2f} |")
            
            # Safety Recommendations
            report.append("\n## Safety Recommendations\n")
            if results['avg_detection_time'] > self.thresholds['max_hazard_detection_time']:
                report.append("- Upgrade hazard detection sensors")
                report.append("- Implement predictive hazard modeling")
            if results['avg_stop_time'] > self.thresholds['max_emergency_stop_time']:
                report.append("- Optimize emergency stop circuits")
                report.append("- Implement redundant stop systems")
            if results['compliance_rate'] < self.thresholds['min_standards_compliance']:
                report.append("- Conduct standards compliance audit")
                report.append("- Update safety procedures to match standards")
            if results['critical_incidents'] > 0:
                report.append("- Implement additional safety barriers for high-risk categories")
            
            report.append(f"\n---\n*Report generated by {self.__class__.__name__}*")
            return "\n".join(report)
            
        except Exception as e:
            raise ReportGenerationError(
                report_type="Safety Evaluation",
                template="safety_report_template",
                error_details=f"Error generating report: {str(e)}"
            )

    def _render_risk_distribution(self, distribution: Dict[str, float]) -> str:
        """Render risk distribution as a pie chart"""
        try:
            # Prepare data
            labels = [cat.replace('_', ' ').title() for cat in distribution.keys()]
            sizes = list(distribution.values())
            if not sizes or sum(sizes) == 0:
                raise VisualizationError(
                    chart_type="risk_distribution",
                    data=distribution,
                    error_details="No valid data to render pie chart"
                )
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures pie is circular
            
            # Convert to base64
            canvas = FigureCanvasQTAgg(fig)
            buffer = QBuffer()
            buffer.open(QBuffer.ReadWrite)
            canvas.print_png(buffer)
            return bytes(buffer.data().toBase64()).decode('utf-8')
            
        except Exception as e:
            if not distribution or sum(distribution.values()) == 0:
                raise VisualizationError(
                    chart_type="risk_distribution",
                    data=distribution,
                    error_details=f"Risk chart rendering failed: {str(e)}"
                )
            
    def _render_compliance_timeline(self) -> str:
        """Render compliance trend over time"""
        try:
            # Prepare data
            timeline = range(len(self.compliance_history))
            compliance = self.compliance_history
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(timeline, compliance, marker='o', linestyle='-')
            ax.axhline(y=self.thresholds['min_standards_compliance'], color='r', linestyle='--')
            ax.set_ylim(0, 1.1)
            ax.set_xlabel('Evaluation Sequence')
            ax.set_ylabel('Compliance Rate')
            ax.set_title('Safety Standards Compliance Trend')
            ax.grid(True)
            
            # Convert to base64
            canvas = FigureCanvasQTAgg(fig)
            buffer = QBuffer()
            buffer.open(QBuffer.ReadWrite)
            canvas.print_png(buffer)
            return bytes(buffer.data().toBase64()).decode('utf-8')
            
        except Exception as e:
            raise VisualizationError(
                chart_type="compliance_timeline",
                data=self.compliance_history,
                error_details=f"Timeline rendering failed: {str(e)}"
            )

    def disable_temporarily(self):
        """Temporarily disable evaluator during degraded mode"""
        self.safety_incidents = []
        logger.warning("SafetyEvaluator temporarily disabled.")

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Safety Evaluator ===\n")
    import sys
    app = QApplication(sys.argv)
    
    # Sample safety incidents
    incidents = [
        {
            'risk_assessment': {'collision': 0.4, 'pinch_point': 0.2},
            'hazard_detection_time': 0.3,
            'emergency_stop_time': 0.15,
            'safety_margin': 0.8,
            'collision_avoided': True,
            'ISO 13849': True,
            'IEC 61508': True,
            'hazard_details': {'collision': 0.6, 'pinch_point': 0.4}
        },
        {
            'risk_assessment': {'crush_hazard': 0.6, 'control_failure': 0.3},
            'hazard_detection_time': 0.7,
            'emergency_stop_time': 0.25,
            'safety_margin': 0.3,
            'collision_avoided': False,
            'ISO 13849': False,
            'IEC 61508': True,
            'hazard_details': {'crush_hazard': 0.8, 'control_failure': 0.2}
        }
    ]
    
    evaluator = SafetyEvaluator()
    results = evaluator.evaluate_operation(incidents)
    
    printer.pretty("Results:", results, "success" if results else "error")
    print(f"\nReport:\n{evaluator.generate_report(results)}")
    
    print("\n=== Safety Evaluation Complete ===\n")