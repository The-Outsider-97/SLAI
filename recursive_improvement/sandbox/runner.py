import os
import tempfile
import subprocess
import logging
import time
import shutil

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DOCKER_IMAGE = "python:3.11-slim"

def run_code_and_tests_docker(code: str, tests: str, timeout: int = 30):
    """
    Runs generated code and unit tests inside a Docker container for sandboxing.

    Parameters:
    - code (str): The generated code.
    - tests (str): The unit tests.
    - timeout (int): Time limit for execution (seconds).

    Returns:
    - (bool, str): Tuple with success flag and output logs.
    """

    logger.info("Starting Docker sandbox execution...")

    # Temp directory for code + tests
    with tempfile.TemporaryDirectory() as tmpdirname:
        logger.debug(f"Created temp dir: {tmpdirname}")

        code_file = os.path.join(tmpdirname, "generated_module.py")
        test_file = os.path.join(tmpdirname, "test_generated_module.py")

        # Write code + tests
        with open(code_file, "w", encoding="utf-8") as cf:
            cf.write(code)

        with open(test_file, "w", encoding="utf-8") as tf:
            tf.write(tests)

        logger.info(f"Files written to {tmpdirname}")

        container_name = f"sandbox_{int(time.time())}"

        try:
            # Spin up a Docker container
            logger.info(f"Starting Docker container: {container_name}")
            subprocess.run([
                "docker", "run", "--name", container_name,
                "-v", f"{tmpdirname}:/sandbox",
                "-w", "/sandbox",
                "--rm", DOCKER_IMAGE,
                "bash", "-c",
                "pip install pytest >/dev/null 2>&1 && pytest -q test_generated_module.py --disable-warnings --maxfail=3"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout
            )

            result = subprocess.run([
                "docker", "logs", container_name
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout
            )

            output = result.stdout.decode() + result.stderr.decode()
            logger.info(f"Container logs:\n{output}")

            # Exit code 0 = tests passed
            success = ("FAILED" not in output)

            return success, output

        except subprocess.TimeoutExpired:
            logger.warning(f"Docker sandbox timed out after {timeout} seconds.")
            return False, f"Timeout after {timeout} seconds."

        except Exception as e:
            logger.error(f"Docker sandbox failed: {e}")
            return False, str(e)

        finally:
            # Clean up just in case container lingers
            subprocess.run(["docker", "rm", "-f", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info("Docker container cleaned up.")
