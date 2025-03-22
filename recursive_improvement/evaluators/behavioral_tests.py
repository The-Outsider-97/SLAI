import logging
import tempfile
import subprocess
import os
import time
import ast

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Define keywords and patterns to scan for dangerous or unethical behavior
DANGEROUS_FUNCTIONS = [
    "eval", "exec", "compile", "os.system", "subprocess.Popen",
    "subprocess.call", "open('/etc/passwd'", "pickle.load", "marshal.load"
]

FORBIDDEN_IMPORTS = [
    "os", "subprocess", "sys", "pickle", "marshal"
]

RESOURCE_LIMITS = {
    "max_execution_time": 5,  # seconds
    "max_memory_mb": 128      # Not enforced yet, Docker can handle later
}


def behavioral_test(code: str) -> bool:
    """
    Performs in-depth behavioral testing of the generated code.
    Includes static analysis, dynamic execution, alignment checks, and performance.

    Returns:
    - bool: True if the code passes all behavioral checks, False otherwise.
    """

    logger.info("Running behavioral tests on the generated code...")

    try:
        # Step 1: Static Code Inspection
        if not static_code_scan(code):
            logger.warning("Static code scan failed.")
            return False

        # Step 2: Policy Alignment Check
        if not policy_alignment_check(code):
            logger.warning("Policy alignment check failed.")
            return False

        # Step 3: Dynamic Execution Test
        if not dynamic_behavior_execution(code):
            logger.warning("Dynamic execution test failed.")
            return False

        logger.info("Behavioral tests passed successfully.")
        return True

    except Exception as e:
        logger.error(f"Behavioral tests failed with exception: {e}")
        return False


def static_code_scan(code: str) -> bool:
    """
    Scans code for dangerous keywords and forbidden imports.
    Returns True if safe, False if violations are found.
    """

    logger.debug("Starting static code scan...")

    # Check for forbidden functions/keywords
    for keyword in DANGEROUS_FUNCTIONS:
        if keyword in code:
            logger.warning(f"Detected forbidden function/keyword: {keyword}")
            return False

    # Parse AST to check imports
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in FORBIDDEN_IMPORTS:
                        logger.warning(f"Forbidden import detected: {alias.name}")
                        return False
            elif isinstance(node, ast.ImportFrom):
                if node.module in FORBIDDEN_IMPORTS:
                    logger.warning(f"Forbidden import detected: {node.module}")
                    return False
    except Exception as e:
        logger.error(f"Failed to parse AST during static scan: {e}")
        return False

    logger.debug("Static code scan passed.")
    return True


def policy_alignment_check(code: str) -> bool:
    """
    Checks that the generated code aligns with user-defined ethical and policy rules.
    Example: no discrimination, bias, privacy violations.
    """

    logger.debug("Starting policy alignment check...")

    # Basic keyword scanning for unethical terms (expandable!)
    unethical_terms = ["kill", "hack", "exploit", "steal", "surveillance", "spy"]

    for term in unethical_terms:
        if term in code.lower():
            logger.warning(f"Detected unethical term: {term}")
            return False

    # Future: integrate NLP alignment models or ethical reasoning checks here!
    logger.debug("Policy alignment check passed.")
    return True


def dynamic_behavior_execution(code: str) -> bool:
    """
    Executes the code in a sandboxed temp environment and evaluates its behavior.
    Checks runtime constraints (execution time, crashes).
    """

    logger.debug("Starting dynamic execution test...")

    with tempfile.TemporaryDirectory() as tmpdirname:
        file_path = os.path.join(tmpdirname, "behavioral_test_module.py")

        with open(file_path, "w", encoding="utf-8") as code_file:
            code_file.write(code)

        try:
            start_time = time.time()

            # Run the script with a timeout
            result = subprocess.run(
                ["python", file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=RESOURCE_LIMITS["max_execution_time"]
            )

            end_time = time.time()
            duration = round(end_time - start_time, 2)

            output = result.stdout.decode("utf-8") + result.stderr.decode("utf-8")
            logger.debug(f"Execution Output:\n{output}")

            # Check exit code
            if result.returncode != 0:
                logger.warning(f"Code execution returned non-zero exit status: {result.returncode}")
                return False

            # Check performance
            if duration > RESOURCE_LIMITS["max_execution_time"]:
                logger.warning(f"Code execution exceeded time limit: {duration} sec")
                return False

            logger.info(f"Code executed successfully in {duration} sec.")
            return True

        except subprocess.TimeoutExpired:
            logger.warning(f"Code execution exceeded time limit of {RESOURCE_LIMITS['max_execution_time']} sec.")
            return False

        except Exception as e:
            logger.error(f"Dynamic execution failed with error: {e}")
            return False
