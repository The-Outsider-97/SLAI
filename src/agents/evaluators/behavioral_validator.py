
import time
import numpy as np
import yaml, json

from typing import Dict, List, Any
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QSize
from datetime import datetime

from src.agents.evaluators.utils.config_loader import load_global_config, get_config_section
from src.agents.evaluators.utils.evaluation_errors import ReportGenerationError, ValidationFailureError, MemoryAccessError
from src.agents.evaluators.utils.report import get_visualizer
from src.agents.evaluators.evaluators_memory import EvaluatorsMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Behavioral Validator")
printer = PrettyPrinter



class BehavioralValidator:
    """
    Behavioral testing framework implementing
    the SUT: System Under Test paradigm from Ammann & Black (2014)
    """
    def __init__(self, test_cases: List[Dict] = None):
        self.config = load_global_config()
        self.enable_historical = self.config.get('enable_historical')
        self.store_results = self.config.get('store_results')

        self.validator_config = get_config_section('behavioral_validator')
        self.mutation_testing = self.validator_config.get('mutation_testing')
        self.requirement_tags = self.validator_config.get('requirement_tags')
        self.max_failure_modes = self.validator_config.get('max_failure_modes')
        self.thresholds = self.validator_config.get('thresholds', {
            'pass_rate', 'failure_tolerance'
        })
        self.weights = self.validator_config.get('weights', {
            'test_coverage', 'pass_rate', 'requirement_coverage'
        })

        self.memory = EvaluatorsMemory()
        self.sut = self.default_sut()
        self.test_cases = test_cases or []
        self.failure_modes = []
        self.requirement_coverage = set()
        self.historical_results = []

        logger.info(f"Behavioral Validator succesfully initialized with {self.requirement_coverage} & {self.historical_results}")

    def default_sut(self, model=None):
        """
        Returns a callable `sut(scenario)` that executes the system under test.

        Parameters:
        -----------
        model : object, optional
            A model object with a .predict(input_data) method. If None, a dummy model is loaded.

        Returns:
        --------
        callable
            A function sut(scenario: dict) -> Any that can be used for certification testing.
        """
        try:
            if model is None:
                model = self.load_default_model()

            def sut(scenario):
                """
                Executes the model on a given scenario.

                Parameters:
                -----------
                scenario : dict
                    A dictionary containing at least the 'input' key.

                Returns:
                --------
                Any
                    The result of the model prediction.
                """
                try:
                    input_data = scenario.get("input")
                    if input_data is None:
                        raise ValueError("Scenario missing 'input' field.")

                    # For advanced usage, you could trace input metadata here
                    output = model.predict(input_data)

                    # Post-processing or logging can go here if needed
                    return output

                except Exception as e:
                    logger.warning(f"SUT execution failed for scenario {scenario.get('id', '<unknown>')}: {str(e)}")
                    return None

            return sut
        except Exception as e:
            logger.error(f"Failed to create SUT: {str(e)}")

    def load_default_model(self):
        """
        Loads or returns a default model for testing if none is provided.

        This can be extended to load from disk, registry, or even mock a full agent.

        Returns:
        --------
        object
            A model-like object with a .predict(input_data) method.
        """
        try:
            class DummyModel:
                def predict(self, input_data):
                    # Simple rule-based simulation for testing
                    if input_data == "test_input":
                        return "expected_output"
                    return "unexpected_output"

            logger.info("Using DummyModel as fallback SUT.")
            return DummyModel()
        except Exception as e:
            logger.error(f"Failed to load default model: {str(e)}")
            raise

    def execute_test_suite(self, agent: Any) -> Dict[str, Any]:
        """Execute full test battery with enhanced tracking"""
        # Create SUT from agent's predict method
        def sut(scenario):
            return agent.predict(scenario['input'])

        if not sut:
            raise ValidationFailureError(
                rule_name="sut_validation",
                data=sut,
                expected="Callable System Under Test"
            )
            
        if not self.test_cases:
            logger.warning("No test cases defined for behavioral validation")
            return {
                'passed': 0,
                'failed': 0,
                'predictions': [],
                'anomalies': [],
                'requirement_coverage': 0,
                'failure_modes': [],
                'expected_outputs': [],
                'pass_rate': 0.0,
                'test_coverage': 0
            }

        results = {
            'passed': 0,
            'failed': 0,
            'predictions': [],
            'anomalies': [],
            'requirement_coverage': 0,
            'failure_modes': [],
            'expected_outputs': []
        }
        
        for test in self.test_cases:
            try:
                if 'scenario' not in test or 'oracle' not in test:
                    raise ValidationFailureError(
                        rule_name="test_case_validation",
                        data=test,
                        expected="Test case must contain 'scenario' and 'oracle'"
                    )
                start_time = time.perf_counter()    
                output = sut(test['scenario'])
                gen_time = time.perf_counter() - start_time
                output = {'value': output, 'generation_time': gen_time}
                results['predictions'].append(output)
                results['expected_outputs'].append(test['scenario'].get('expected_output'))
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
            try:
                self.memory.add(
                    entry=results,
                    tags=["behavioral_eval"],
                    priority="high" if results['pass_rate'] < self.thresholds.get('pass_rate', 0.8) else "medium"
                )
            except Exception as e:
                raise MemoryAccessError(
                    operation="add",
                    key="behavioral_eval_results",
                    error_details=str(e)
                )
        
        self.historical_results.append(results)
        return results

    def _analyze_failure(self, test: Dict, output: Any) -> Dict:
        """Enhanced failure analysis with FMEA scoring"""
        try:
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
        except Exception as e:
            logger.error(f"Failure analysis error: {str(e)}")
            return {
                'error': str(e),
                'test': str(test)[:200],
                'output': str(output)[:200]
            }
    
    def execute_certification_suite(self, sut: callable, certification_requirements: List[Dict]) -> Dict[str, Any]:
        """
        Executes a formal test suite for certification purposes.

        This method is stricter than execute_test_suite. It requires explicit,
        traceable requirements and generates a formal evidence package.

        Args:
            sut (callable): The System Under Test.
            certification_requirements (List[Dict]): A list of requirements, where each
                is a dictionary containing keys like 'id', 'scenario', 'oracle', and 'severity'.

        Returns:
            Dict[str, Any]: A dictionary containing the certification results,
                            including status, traceability, and an evidence log.
        """
        if not sut:
            raise ValidationFailureError(
                rule_name="sut_validation",
                data=sut,
                expected="Callable System Under Test"
            )
            
        if not certification_requirements:
            raise ValidationFailureError(
                rule_name="certification_requirements",
                data=certification_requirements,
                expected="Non-empty list of requirements"
            )
            
        logger.info(f"Starting certification suite with {len(certification_requirements)} requirements.")
        
        traceability_matrix = []
        evidence_log = []
        overall_status = "PASSED"
        
        for req in certification_requirements:
            req_id = req.get('id', 'UNKNOWN_REQ')
            scenario = req.get('scenario')
            oracle = req.get('oracle') # The oracle function should be part of the requirement spec
            severity = req.get('severity', 'high')
            test_passed = False
            error_message = None

            # Validate requirement structure
            if not all(key in req for key in ['id', 'scenario', 'oracle']):
                raise ValidationFailureError(
                    rule_name="requirement_structure",
                    data=req,
                    expected="Requirement must contain 'id', 'scenario', and 'oracle'"
                )

            try:
                output = sut(scenario)
                if oracle(output):
                    test_passed = True
                else:
                    error_message = f"Oracle returned False for output: {str(output)[:100]}"
            except Exception as e:
                error_message = f"SUT raised an exception: {e}"
                logger.error(f"Certification test for requirement {req_id} failed with an exception: {e}")

            # Update status and logs
            status = "PASSED" if test_passed else "FAILED"
            traceability_matrix.append({
                "requirement_id": req_id,
                "test_scenario_summary": str(scenario)[:150],
                "status": status
            })
            evidence_log.append({
                "timestamp": datetime.now().isoformat(),
                "requirement_id": req_id,
                "status": status,
                "details": "Validation successful." if test_passed else error_message
            })

            # If a critical test fails, the entire certification fails immediately.
            if not test_passed and severity == 'critical':
                overall_status = "FAILED"
                logger.critical(f"Critical requirement {req_id} FAILED. Halting certification suite.")
                break # Stop further testing

        # If any non-critical test failed, the overall status is also failed.
        if any(item['status'] == 'FAILED' for item in traceability_matrix):
            overall_status = "FAILED"
            
        return {
            "overall_status": overall_status,
            "traceability_matrix": traceability_matrix,
            "evidence_log": evidence_log
        }

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive behavioral validation report"""
        if not results:
            return "# Error: No test results available to generate report"
        
        try:
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
            report.append(f"- **Critical Failures**: {sum(1 for f in results['failure_modes'] if f.get('severity') == 'high')}")

            # Failure Analysis
            report.append("\n## Failure Mode Analysis\n")
            if results.get('failure_modes'):
                top_failures = sorted(results['failure_modes'], 
                                    key=lambda x: x.get('criticality_score', 0), 
                                    reverse=True)[:self.config.get('max_failure_modes', 5)]
                for failure in top_failures:
                    report.append(
                        f"- **{failure.get('test_id', 'unknown')}**: {failure.get('severity', 'unknown').title()} severity "
                        f"(Criticality: {failure.get('criticality_score', 0)}) - "
                        f"Detected via {failure.get('detection_mechanism', 'unknown')}"
                    )
            else:
                report.append("✅ No critical failures detected")

            # Historical Trends
            if self.config.get('enable_historical', False) and self.historical_results:
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
        except Exception as e:
            raise ReportGenerationError(
                report_type="Behavioral Validation",
                template="behavioral_report_template",
                error_details=f"Error generating report: {str(e)}"
            )

    def disable_temporarily(self):
        """Temporarily disable behavioral testing during degraded mode"""
        self.test_cases = []
        logger.warning("Behavioral Validator temporarily disabled.")

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Behavioral Validator ===\n")
    import sys
    app = QApplication(sys.argv)

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

    validator = BehavioralValidator(test_cases=test_cases)
    results = validator.execute_test_suite(agent=sample_sut)

    # Generate report with actual results
    print(validator.generate_report(results))

    print(f"\n* * * * * Phase 2 * * * * *\n")
    sut = validator.default_sut()
    req=[
        {
            "id": "REQ-001",
            "scenario": {
                "input": "test_input",
                "expected_output": "expected_output"
            },
            "oracle": lambda output: output == "expected_output",
            "severity": "critical"
        },
        {
            "id": "REQ-002",
            "scenario": {
                "input": "other_input",
                "expected_output": "alt_output"
            },
            "oracle": lambda output: output == "alt_output",
            "severity": "medium"
        }
    ]
    suite = validator.execute_certification_suite(sut=sut, certification_requirements=req)

    printer.pretty("SUITE", suite, "success" if suite else "error")
    print(f"\n* * * * * Phase 3 * * * * *\n")
    print("\n=== Successfully Ran Behavioral Validator ===\n")
