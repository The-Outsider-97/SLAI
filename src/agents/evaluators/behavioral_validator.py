
import numpy as np
import yaml, json

from typing import Dict, List, Any
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QSize
from datetime import datetime

from src.agents.evaluators.evaluators_memory import EvaluatorsMemory
from logs.logger import get_logger

logger = get_logger("Behavioral Validator")

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

class BehavioralValidator:
    """
    Behavioral testing framework implementing
    the SUT: System Under Test paradigm from Ammann & Black (2014)
    """
    def __init__(self, config, test_cases: List[Dict] = None):
        config = load_config() or {}
        self.config = config.get('behavioral_evaluator', {})
        memory = EvaluatorsMemory(config)
        self.memory = memory
        self.test_cases = test_cases or []
        self.failure_modes = []
        self.requirement_coverage = set()
        self.historical_results = []
        
        # Configuration parameters
        self.thresholds = self.config.get('thresholds', {})
        self.weights = self.config.get('weights', {})
        self.enable_mutation = self.config.get('mutation_testing', False)

        logger.info(f"Behavioral Validator succesfully initialized with {self.requirement_coverage} & {self.historical_results}")

    def execute_test_suite(self, sut: callable) -> Dict[str, Any]:
        """Execute full test battery with enhanced tracking"""
        results = {
            'passed': 0, 
            'failed': 0, 
            'anomalies': [],
            'requirement_coverage': 0,
            'failure_modes': []
        }
        
        for test in self.test_cases:
            try:
                output = sut(test['scenario'])
                if test['oracle'](output):
                    results['passed'] += 1
                    if 'requirement_id' in test['scenario']:
                        self.requirement_coverage.add(test['scenario']['requirement_id'])
                else:
                    results['failed'] += 1
                    failure_analysis = self._analyze_failure(test, output)
                    results['failure_modes'].append(failure_analysis)
            except Exception as e:
                logger.error(f"Test execution error: {str(e)}")
                results['anomalies'].append({
                    'test': test,
                    'error': str(e)
                })
        
        results['requirement_coverage'] = len(self.requirement_coverage)
        results['pass_rate'] = results['passed'] / len(self.test_cases) if self.test_cases else 0
        results['test_coverage'] = len(self.test_cases)
        
        if self.config.get('store_results', False):
            self.memory.add(
                entry=results,
                tags=["behavioral_eval"],
                priority="high" if results['pass_rate'] < self.thresholds['pass_rate'] else "medium"
            )
        
        self.historical_results.append(results)
        return results

    def _analyze_failure(self, test: Dict, output: Any) -> Dict:
        """Enhanced failure analysis with FMEA scoring"""
        failure_mode = {
            'test_id': hash(str(test['scenario'])),
            'severity': test.get('severity', 'medium'),
            'output': output,
            'timestamp': datetime.now(),
            'detection_mechanism': test['scenario'].get('detection_method', 'oracle'),
            'criticality_score': 0
        }
        
        # Calculate criticality score
        severity_weights = {'low': 1, 'medium': 3, 'high': 5}
        failure_mode['criticality_score'] = severity_weights[failure_mode['severity']] * \
            (1 if failure_mode['detection_mechanism'] == 'automated' else 2)
        
        self.failure_modes.append(failure_mode)
        return failure_mode

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive behavioral validation report"""
        from src.agents.evaluators.utils.report import get_visualizer
        if not results:
            return "# Error: No test results available to generate report"
        report = []
        visualizer = get_visualizer()

        # Header Section
        report.append(f"\n# Behavioral Validation Report\n")
        report.append(f"**Generated**: {datetime.now().isoformat()}\n")
        
        # Summary Metrics
        report.append("## Executive Summary\n")
        report.append(f"- **Total Tests Executed**: {results['test_coverage']}")
        report.append(f"- **Pass Rate**: {results['pass_rate']:.2%}")
        report.append(f"- **Requirements Covered**: {results['requirement_coverage']}")
        report.append(f"- **Critical Failures**: {sum(1 for f in results['failure_modes'] if f['severity'] == 'high')}")

        # Failure Analysis
        report.append("\n## Failure Mode Analysis\n")
        if results['failure_modes']:
            top_failures = sorted(results['failure_modes'], 
                                key=lambda x: x['criticality_score'], 
                                reverse=True)[:self.config.get('max_failure_modes', 5)]
            for failure in top_failures:
                report.append(
                    f"- **{failure['test_id']}**: {failure['severity'].title()} severity "
                    f"(Criticality: {failure['criticality_score']}) - "
                    f"Detected via {failure['detection_mechanism']}"
                )
        else:
            report.append("✅ No critical failures detected")

        # Historical Trends
        if self.config.get('enable_historical', False):
            report.append("\n## Historical Performance\n")
            chart = visualizer.render_temporal_chart(
                QSize(600, 400), 
                'pass_rate',
                data=[res['pass_rate'] for res in self.historical_results]
            )
            report.append(f"![Pass Rate Trend](data:image/png;base64,{visualizer._chart_to_base64(chart)})")

        # Composite Score
        if self.weights:
            report.append("\n## Composite Validation Score\n")
            composite = sum(
                self.weights.get(metric, 0) * results.get(metric, 0)
                for metric in ['pass_rate', 'test_coverage', 'requirement_coverage']
            )
            report.append(f"**Overall Score**: {composite:.2f}/1.0")
            report.append("### Score Breakdown")
            for metric in ['pass_rate', 'test_coverage', 'requirement_coverage']:
                report.append(f"- {metric.replace('_', ' ').title()}: {self.weights.get(metric, 0) * results.get(metric, 0):.2f}")

        # Footer
        report.append(f"\n---\n*Report generated by {self.__class__.__name__}*")
        
        return "\n".join(report)

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Behavioral Validator ===\n")
    import sys
    app = QApplication(sys.argv)
    config = load_config()
    
    # Proper test case and SUT example
    test_cases = [{
        'scenario': {
            'input': "test_input",
            'requirement_id': "REQ-001",
            'detection_method': 'automated'
        },
        'oracle': lambda x: x == "expected_output"
    }]
    
    # Proper System Under Test implementation
    def sample_sut(scenario):
        return "actual_output"  # Should fail the test
    
    validator = BehavioralValidator(config, test_cases=test_cases)
    results = validator.execute_test_suite(sut=sample_sut)

    # Generate report with actual results
    print(validator.generate_report(results))

    print(f"\n* * * * * Phase 2 * * * * *\n")
    #results=None
    #validator.generate_report(results=results)

    #logger.info("Report:\n" + json.dumps(validator.generate_report(), indent=4))
    #print(json.dumps (validator.generate_report(), indent=4))
    print(f"\n* * * * * Phase 3 * * * * *\n")
    print("\n=== Successfully Ran Behavioral Validator ===\n")
