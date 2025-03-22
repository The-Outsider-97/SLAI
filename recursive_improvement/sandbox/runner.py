import os
import tempfile
import subprocess
import logging
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def run_code_and_tests(code: str, tests: str, timeout: int = 10):
    """
    Executes the generated code and runs unit tests in an isolated temp directory.

    Parameters:
    - code (str): The generated Python code.
    - tests (str): The generated PyTest test cases for the code.
    - timeout (int): Time in seconds to allow for each subprocess to run.

    Returns:
    - (bool, str): Tuple indicating if tests passed and the output logs.
    """

    logger.info("Starting sandboxed execution of code and tests...")

    with tempfile.TemporaryDirectory() as tmpdirname:
        logger.debug(f"Created temporary directory: {tmpdirname}")

        # Paths for code and tests
        code_file_path = os.path.join(tmpdirname, "generated_module.py")
        test_file_path = os.path.join(tmpdirname, "test_generated_module.py")

        # Write the generated code
        with open(code_file_path, "w", encoding="utf-8") as code_file:
            code_file.write(code)
        logger.info(f"Saved generated code to {code_file_path}")

        # Write the generated tests
        with open(test_file_path, "w", encoding="utf-8") as test_file:
            test_file.write(tests)
        logger.info(f"Saved generated unit tests to {test_file_path}")

        try:
            # Run pytest on the test file
            logger.info("Running PyTest on generated code...")
            start_time = time.time()

            result = subprocess.run(
                ["pytest", "-q", test_file_path, "--disable-warnings", "--maxfail=3"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=tmpdirname,
                timeout=timeout
            )

            end_time = time.time()
            duration = round(end_time - start_time, 2)

            output = result.stdout.decode("utf-8") + result.stderr.decode("utf-8")
            logger.debug(f"PyTest output:\n{output}")
            logger.info(f"Test run completed in {duration} seconds with return code {result.returncode}")

            # Check if tests passed
            success = result.returncode == 0
            return success, output

        except subprocess.TimeoutExpired:
            logger.warning(f"Test execution exceeded timeout of {timeout} seconds.")
            return False, f"Timeout after {timeout} seconds."

        except Exception as e:
            logger.error(f"Error running code and tests: {e}")
            return False, str(e)
