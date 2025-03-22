import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

RISK_REWARD_MAPPING = {
    "LOW": 25,
    "MEDIUM": 10,
    "HIGH": -10,
    "CRITICAL": -50
}

def calculate_reward(tests_passed: bool,
                     static_analysis_result: str,
                     behavioral_test_passed: bool = True,
                     execution_time: float = None,
                     timeout_threshold: float = 10.0):
    """
    Calculates a reward score based on test results, static analysis, and behavior.

    Parameters:
    - tests_passed (bool): True if all unit tests pass.
    - static_analysis_result (str): Static analysis summary (expects Bandit result output).
    - behavioral_test_passed (bool): True if behavioral evaluations pass.
    - execution_time (float): Time in seconds taken to execute tests/code.
    - timeout_threshold (float): Time limit where performance becomes penalized.

    Returns:
    - reward (float): The total reward score for this iteration.
    """

    reward = 0.0
    logger.info("Calculating reward...")

    # Unit Tests Reward
    if tests_passed:
        reward += 50
        logger.debug("Tests passed: +50 reward")
    else:
        logger.debug("Tests failed: no reward added")

    # Static Analysis Reward
    risk_level = static_analysis_report.get("risk_level", "CRITICAL").upper()
    static_reward = RISK_REWARD_MAPPING.get(risk_level, -100)
    reward += static_reward
    logger.debug(f"Static analysis risk level '{risk_level}': {static_reward} reward")

    # Behavioral Test Reward
    if behavioral_test_passed:
        reward += 15
        logger.debug("Behavioral tests passed: +15 reward")
    else:
        logger.debug("Behavioral tests failed: no reward added")

    # Performance Reward (optional)
    if execution_time is not None:
        if execution_time < timeout_threshold:
            performance_reward = max(10, (timeout_threshold - execution_time))  # reward faster code
            reward += performance_reward
            logger.debug(f"Execution time {execution_time}s: +{performance_reward} reward")
        else:
            logger.debug("Execution time exceeded threshold: no performance reward")

    logger.info(f"Total reward: {reward}")
    return reward
