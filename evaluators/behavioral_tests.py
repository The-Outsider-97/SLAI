import logging
import torch
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class BehavioralTests:
    """
    BehavioralTests performs predefined and custom behavioral evaluations
    on AI agents to verify alignment with expected and safe behaviors.
    Supports automated rollback/retrain triggers on test failures.
    """

    def __init__(self, test_cases=None, rollback_handler=None, hyperparam_tuner=None):
        """
        Initialize behavioral tests with optional predefined test cases and rollback/retrain handlers.
        
        Args:
            test_cases (list): A list of test cases, each represented as a dict with:
                - 'input': The scenario input to the agent.
                - 'expected_behavior': Description or expected action/output.
                - 'validation_fn': Function that accepts output and returns True/False.
            rollback_handler (object): Optional rollback handler to revert models.
            hyperparam_tuner (object): Optional hyperparameter tuner to retrain models.
        """
        self.test_cases = test_cases or []
        self.rollback_handler = rollback_handler
        self.hyperparam_tuner = hyperparam_tuner

    def add_test_case(self, input_scenario, expected_behavior, validation_fn):
        """
        Adds a new behavioral test case.
        
        Args:
            input_scenario (any): The input data/scenario for the test.
            expected_behavior (str): Description of what the agent should do.
            validation_fn (callable): Function to validate agent's behavior.
        """
        test_case = {
            'input': input_scenario,
            'expected_behavior': expected_behavior,
            'validation_fn': validation_fn
        }
        self.test_cases.append(test_case)
        logger.info("Added behavioral test case: %s", expected_behavior)

    def run_tests(self, agent_function):
        """
        Run all behavioral tests against the provided agent function.
        Automatically triggers rollback/retrain if tests fail.
        
        Args:
            agent_function (callable): The function/method that simulates agent behavior.
                Must accept the test input and return a response/output.
        
        Returns:
            dict: Summary report with results of each test case.
        """
        logger.info("Running behavioral tests on the agent...")

        results = []
        failed_tests = 0

        for idx, test in enumerate(self.test_cases, start=1):
            logger.info("Running test case #%d: %s", idx, test['expected_behavior'])

            try:
                agent_output = agent_function(test['input'])
                passed = test['validation_fn'](agent_output)
                logger.info("Test case #%d result: %s", idx, 'PASSED' if passed else 'FAILED')
            except Exception as e:
                logger.error("Error during test case #%d: %s", idx, str(e))
                passed = False

            if not passed:
                failed_tests += 1

            results.append({
                'test_case': test['expected_behavior'],
                'input': test['input'],
                'result': 'PASSED' if passed else 'FAILED'
            })

        logger.info("Behavioral testing complete. %d tests run, %d failed.", len(self.test_cases), failed_tests)

        # Automated rollback or retrain action
        if failed_tests > 0:
            logger.warning("Failures detected in behavioral tests. Initiating corrective actions...")
            self._trigger_corrective_actions()
        else:
            logger.info("All behavioral tests passed successfully.")

        return results

    def _trigger_corrective_actions(self):
        """
        Trigger rollback and/or hyperparameter retraining pipeline.
        """
        if self.rollback_handler:
            logger.info("Triggering rollback handler...")
            self.rollback_handler.rollback_model()
        if self.hyperparam_tuner:
            logger.info("Triggering hyperparameter tuner for retraining...")
            self.hyperparam_tuner.run_tuning_pipeline()
        if not self.rollback_handler and not self.hyperparam_tuner:
            logger.warning("No rollback or retrain handler available. Manual intervention may be required.")

if __name__ == "__main__":
    # Example dummy agent
    def dummy_agent(input_data):
        if input_data == "greet":
            return "Hello!"
        elif input_data == "farewell":
            return "Goodbye!"
        else:
            return "I don't understand."

    # Dummy rollback and tuner
    class DummyRollback:
        def rollback_model(self):
            print("[Rollback] Model rollback triggered!")

    class DummyTuner:
        def run_tuning_pipeline(self):
            print("[Tuner] Hyperparameter tuning and retrain triggered!")

    # Validation functions
    def validate_greet(response):
        return response == "Hello!"

    def validate_farewell(response):
        return response == "Goodbye!"

    # Create and run behavioral tests
    tests = BehavioralTests(
        rollback_handler=DummyRollback(),
        hyperparam_tuner=DummyTuner()
    )
    tests.add_test_case("greet", "Agent should greet politely", validate_greet)
    tests.add_test_case("farewell", "Agent should say goodbye", validate_farewell)

    report = tests.run_tests(dummy_agent)

    print("\nBehavioral Test Report:")
    for result in report:
        print(result)
