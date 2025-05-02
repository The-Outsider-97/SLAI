import logging
import time
from src.rsi.codegen.codegen import generate_code
from src.rsi.unit_tests.test_generator import generate_unit_tests
from src.rsi.sandbox.runner import run_code_and_tests
from src.agents.evaluators.static_analysis import static_analysis_bandit
from src.agents.evaluators.behavioral_tests import behavioral_test
from src.agents.evaluators.reward_function import calculate_reward
from monitoring.dashboard import push_rsi_update
from deployment.git.rollback_handler import rollback_to_previous_release
from deployment.git.branch_manager import create_branch, merge_branches, delete_branch, auto_name_branch
from deployment.git.ci_cd_trigger import trigger_github_actions
from deployment.deployment_logger import log_event
from deployment.deployment_history import log_to_history, get_history
from logs.logger import get_logger

logger = get_logger(__name__)

log_to_history(
    event_type="merge",
    user="rsi_process",
    branch="rsi/agent_v3/iter-10",
    success=True,
    details={"conflicts": 0}
)

history = get_history()
print(json.dumps(history, indent=2))

# Successful deployment
log_event(
    event_type="deploy",
    user="rsi_process",
    branch="main",
    version="v1.3.0",
    success=True,
    details={
        "risk_level": "LOW",
        "reward_score": 82.5
    }
)

# Rollback example
log_event(
    event_type="rollback",
    user="rsi_process",
    branch="main",
    version="v1.3.0",
    success=True,
    details={
        "rollback_target": "v1.2.3"
    }
)

# After successful RSI iteration and deployment
trigger_github_actions(branch="main")

# Create RSI iteration branch
branch = auto_name_branch(task_desc="agent_evolution", iteration=5)
create_branch(branch)

# Merge back to main after success
merge_branches(source_branch=branch, target_branch="main", squash=True)

# Clean up
delete_branch(branch, remote=True)

try:
    # Assume RSI iteration logic...
    success = recursive_improvement_loop(task_description, iterations=5)

    if not success:
        logger.warning("RSI loop failed! Initiating rollback...")
        rollback_to_previous_release()

except Exception as e:
    logger.error(f"Unexpected failure in RSI loop: {e}")
    rollback_to_previous_release()

# After each iteration success:
push_rsi_update(
    iteration=iteration,
    reward=reward,
    risk_level=risk_level,
    details={
        "static_analysis": static_analysis_report,
        "behavioral_passed": behavior_passed
    }
)

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
        static_analysis_report = static_analysis_bandit(code)
        risk_level = static_analysis_report.get("risk_level", "CRITICAL")

        if risk_level in ["HIGH", "CRITICAL"]:
            logger.warning(f"Static analysis returned risk level '{risk_level}'. Skipping deployment.")
            logger.debug(f"Full Static Analysis Report: {static_analysis_report}")
            continue

        # Reward calculation (uses granular risk level)          
        reward = calculate_reward(
            tests_passed=tests_passed,
            static_analysis_result=static_analysis_results,
            behavioral_test_passed=behavior_passed,
            execution_time=5.8  # example from sandbox runner duration
        )

        # Step 5: Behavioral Testing (Optional but recommended)
        behavior_passed = behavioral_test(code)
        if not behavior_passed:
            logger.warning("Behavioral tests failed. Skipping deployment.")
            continue


        logger.info(f"Reward for this iteration: {reward}")
        
        # Step 6: If all tests pass, save, deploy, or evolve!
        logger.info(f"Iteration {iteration} successful. Code is ready for deployment or evolution.")
        
        from recursive_improvement.deployment.git.git_handler import commit_and_push, tag_release

        def _deploy_code(code: str):
            """
            Saves and deploys the successful code via Git.
            """
            logger.info("Deploying code to Git...")

            # Example path in repo where the improved module lives
            file_path = "agents/evolved_agent.py"  # Or wherever you want to save it

            commit_message = "RSI Auto-Generated Agent Improvement"
            commit_and_push(code_string=code, file_path=file_path, commit_message=commit_message)

            # Optional: Create a tag for versioning
            tag_release("Stable RSI-generated agent after evaluation.")

        
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
