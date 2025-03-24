import torch
import os
import logging
import subprocess

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class StaticAnalysis:
    """
    Runs static code analysis on AI agent codebases to detect risky patterns,
    security vulnerabilities, or violations of coding standards. Automatically
    triggers rollback or retrain when critical issues are detected.
    """

    def __init__(self, codebase_path, rollback_handler=None, hyperparam_tuner=None, thresholds=None):
        """
        Initializes the StaticAnalysis tool.

        Args:
            codebase_path (str): The root directory of the codebase to analyze.
            rollback_handler (object): Rollback handler for reverting versions if issues are found.
            hyperparam_tuner (object): Hyperparameter tuner for retraining if issues are found.
            thresholds (dict): Thresholds to decide rollback/retrain, e.g.:
                {
                    'max_warnings': 10,
                    'critical_issues': True
                }
        """
        self.codebase_path = codebase_path
        self.rollback_handler = rollback_handler
        self.hyperparam_tuner = hyperparam_tuner
        self.thresholds = thresholds or {
            'max_warnings': 10,
            'critical_issues': True
        }

    def run_static_analysis(self):
        """
        Runs static analysis using pylint and parses the results.
        """
        logger.info("Starting static analysis on codebase: %s", self.codebase_path)

        try:
            cmd = [
                'pylint',
                self.codebase_path,
                '--exit-zero',
                '--output-format=json'
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            output = result.stdout
            issues = self._parse_pylint_output(output)
        except subprocess.CalledProcessError as e:
            logger.error("Static analysis failed to execute: %s", e)
            issues = []

        logger.info("Static analysis found %d issues.", len(issues))
        self._evaluate_issues(issues)

        return issues

    def _parse_pylint_output(self, output):
        """
        Parses pylint JSON output into a list of issue dictionaries.
        """
        import json
        try:
            issues = json.loads(output)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse pylint output: %s", e)
            issues = []
        return issues

    def _evaluate_issues(self, issues):
        """
        Evaluates parsed issues against thresholds and triggers rollback/retrain if needed.
        """
        warnings_count = len([i for i in issues if i.get('type') == 'warning'])
        critical_issues = any(i for i in issues if i.get('type') == 'error' or i.get('type') == 'fatal')

        logger.info("Warnings count: %d", warnings_count)
        logger.info("Critical issues found: %s", critical_issues)

        rollback_required = critical_issues if self.thresholds['critical_issues'] else False
        retrain_required = warnings_count > self.thresholds['max_warnings']

        if rollback_required:
            logger.warning("Triggering rollback due to critical issues detected!")
            self._trigger_rollback("Critical issues found in static analysis.")

        if retrain_required:
            logger.warning("Triggering retrain due to excessive warnings detected!")
            self._trigger_retrain("Excessive warnings found in static analysis.")

    def _trigger_rollback(self, reason):
        """
        Triggers the rollback process via rollback handler.
        """
        logger.info("Rollback triggered due to: %s", reason)
        if self.rollback_handler:
            self.rollback_handler.rollback_model()
        else:
            logger.warning("No rollback handler provided. Manual rollback may be required.")

    def _trigger_retrain(self, reason):
        """
        Triggers retraining process via hyperparameter tuner.
        """
        logger.info("Retraining triggered due to: %s", reason)
        if self.hyperparam_tuner:
            self.hyperparam_tuner.run_tuning_pipeline()
        else:
            logger.warning("No hyperparameter tuner provided. Manual retraining may be required.")

if __name__ == "__main__":
    # Demonstration rollback and retrain handlers
    class RollbackHandler:
        def rollback_model(self):
            logger.info("[RollbackHandler] Rolling back model and codebase to last stable state...")
            print("[RollbackHandler] Rollback complete!")

    class HyperParamTuner:
        def run_tuning_pipeline(self):
            logger.info("[HyperParamTuner] Running hyperparameter tuning and retraining pipeline...")
            print("[HyperParamTuner] Retraining complete!")

    # Example static analysis usage
    static_analyzer = StaticAnalysis(
        codebase_path='src/',
        rollback_handler=RollbackHandler(),
        hyperparam_tuner=HyperParamTuner(),
        thresholds={
            'max_warnings': 5,
            'critical_issues': True
        }
    )

    static_analyzer.run_static_analysis()
