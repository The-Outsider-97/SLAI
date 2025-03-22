import openai
import os
import time
import logging
from typing import Optional

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Set the API key (secure via environment variable)
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("OpenAI API key is not set. Please set OPENAI_API_KEY in your environment variables.")


def generate_unit_tests(code_snippet: str, language: str = "Python", max_tokens: int = 600, temperature: float = 0.2) -> Optional[str]:
    """
    Generates PyTest unit tests for the provided code snippet.

    Parameters:
    - code_snippet (str): The source code to write unit tests for.
    - language (str): Programming language for the tests (default is Python).
    - max_tokens (int): Max tokens to return from OpenAI.
    - temperature (float): Creativity level of the LLM.

    Returns:
    - str: Unit test code as a string, or None if generation failed.
    """

    prompt = (
        f"You are a senior software engineer specializing in writing unit tests.\n"
        f"Write comprehensive PyTest unit tests for the following {language} code. "
        f"Ensure the tests cover normal cases, edge cases, and invalid input handling. "
        f"Tests should be clear, maintainable, and adhere to best practices.\n\n"
        f"Code:\n{code_snippet}\n"
    )

    logger.info("Requesting unit test generation from OpenAI...")
    logger.debug(f"Prompt for unit test generation:\n{prompt}")

    retries = 3
    delay = 2  # seconds between retries
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "system", "content": "You are a senior Python developer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            test_code = response['choices'][0]['message']['content']
            logger.info("Unit test generation successful.")
            logger.debug(f"Generated Unit Tests:\n{test_code}")

            # Optional: Save generated tests for audit trail
            _save_test_output(code_snippet, test_code)

            return test_code.strip()

        except Exception as e:
            logger.warning(f"Unit test generation attempt {attempt + 1} failed: {str(e)}")
            time.sleep(delay)

    logger.error("All attempts to generate unit tests failed.")
    return None


def _save_test_output(code_snippet: str, test_code: str, directory: str = "./logs/tests/") -> None:
    """
    Saves the generated test output along with its source code for traceability.

    Parameters:
    - code_snippet (str): The original code.
    - test_code (str): The generated unit test code.
    - directory (str): Path to save logs.
    """

    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, f"unit_tests_{int(time.time())}.txt")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("Code Snippet:\n")
        f.write(code_snippet + "\n\n")
        f.write("Generated Unit Tests:\n")
        f.write(test_code)

    logger.info(f"Unit tests saved to {filename}")
