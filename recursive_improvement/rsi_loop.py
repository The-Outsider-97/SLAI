import logging
import time
from recursive_improvement.codegen.codegen import generate_code
from recursive_improvement.unit_tests.test_generator import generate_unit_tests
from recursive_improvement.sandbox.runner import run_code_and_tests
from recursive_improvement.evaluators.static_analysis import static_analysis_bandit
from recursive_improvement.evaluators.behavioral_tests import behavioral_test

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def recursive_improvement_loop(task_description: str, iterations: int = 5):
    """
    Orchestrates the recursive self-improvement loop for a specific task.
    
    Parameters:
    - task_description (str): Description of the task to generate code for.
    - iterations (int): How many times to loop through recursive generation.
    """
    logger.info(f"Starting Recursive Self-Improvement Loop for task: {task_description}")
    
    for iteration in range(1, iterations + 1):
        logger.info(f"\n=== Iteration {iteration} ===")

        # Step 1: Generate Code
        code = generate_code(task_description)
        if not code:
            logger.warning("Code generation failed, moving to next iteration.")
            continue

        # Step 2: Generate Unit Tests for the Code
        unit_tests = generate_unit_tests(code)
        if not unit_tests:
            logger.warning("Unit test generation failed, moving to next iteration.")
            continue

        # Step 3: Run Code + Unit Tests inside a Sandbox
        tests_passed, test_output = run_code_and_tests(code, unit_tests)
        logger.info(f"Unit Test Output:\n{test_output}")

        if not tests_passed:
            logger.warning("Unit tests failed, moving to next iteration.")
            continue

        # Step 4: Static Analysis for Security and Code Quality
        static_analysis_results = static_analysis_bandit(code)
        if "HIGH" in static_analysis_results:
            logger.warning("High severity issues found in static analysis, skipping deployment.")
            logger.debug(f"Static Analysis Details: {static_analysis_results}")
            continue

        # Step 5: Behavioral Testing (Optional but recommended)
        behavior_passed = behavioral_test(code)
        if not behavior_passed:
            logger.warning("Behavioral tests failed. Skipping deployment.")
            continue

        # Step 6: If all tests pass, save, deploy, or evolve!
        logger.info(f"Iteration {iteration} successful. Code is ready for deployment or evolution.")
        
        # You can commit to Git, hot-swap modules, or evolve agents here
        _deploy_code(code)
        
        logger.info("Deployment complete. Loop terminated successfully.")
        break  # Exit after a successful improvement cycle

    else:
        logger.info("Recursive loop ended without successful deployment.")


def _deploy_code(code: str):
    """
    Placeholder function for deploying or integrating the generated code.

    Replace this with your GitOps, dynamic module loader, or deployment logic.
    """
    logger.info("Deploying code... (implement your deployment strategy here)")
    # For now, just log the code being 'deployed'
    logger.debug(f"Deployed Code:\n{code}")


if __name__ == "__main__":
    # Example task - you can change this
    example_task = (
        "Implement a Python class for an asynchronous multi-agent communication system. "
        "Agents should exchange JSON messages over asyncio queues, with error handling and message validation."
    )

    # Run the recursive self-improvement loop
    recursive_improvement_loop(example_task, iterations=10)
