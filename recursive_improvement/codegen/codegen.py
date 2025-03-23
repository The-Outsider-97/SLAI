import os
import time
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

client = OpenAI(
  api_key="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"
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "write a haiku about ai"}
  ]
)

print(completion.choices[0].message);

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def generate_code(task_description: str, language: str = "Python", max_tokens: int = 800, temperature: float = 0.2) -> Optional[str]:
    """
    Generates code for a given task description using OpenAI GPT-4.
    
    Parameters:
    - task_description (str): Description of the code to generate.
    - language (str): Programming language (default is Python).
    - max_tokens (int): Maximum tokens in the response.
    - temperature (float): Creativity level of the model.

    Returns:
    - str: Generated code, or None if generation failed.
    """

    # Create the prompt structure
    prompt = (
        f"Generate a clean, efficient, and well-documented {language} code implementation "
        f"for the following task:\n\n"
        f"{task_description}\n\n"
        f"Include:\n"
        f"- Type hints\n"
        f"- Docstrings\n"
        f"- Adhere to best practices and safety guidelines\n"
        f"- Ensure the code can be easily tested\n"
    )

    logger.info("Sending request to OpenAI Codegen model...")
    logger.debug(f"Prompt: {prompt}")

    # Retry mechanism
    retries = 3
    delay = 2  # seconds between retries
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",  # Use gpt-4o or your preferred model
                messages=[
                    {"role": "system", "content": "You are a senior software engineer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            code_output = response.choices[0].message.content
            logger.info("Code generation successful.")
            logger.debug(f"Generated Code: {code_output}")

            # Optional: Save raw outputs for traceability
            _save_codegen_output(task_description, code_output)

            return code_output.strip()

        except Exception as e:
            logger.warning(f"Codegen attempt {attempt + 1} failed: {str(e)}")
            time.sleep(delay)

    logger.error("All code generation attempts failed.")
    return None


def _save_codegen_output(task_description: str, code_output: str, directory: str = "./logs/codegen/") -> None:
    """
    Saves the generated code output and its task description for record-keeping.

    Parameters:
    - task_description (str): The prompt that led to code generation.
    - code_output (str): The actual code generated.
    - directory (str): Path to save logs.
    """

    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, f"codegen_{int(time.time())}.txt")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("Task Description:\n")
        f.write(task_description + "\n\n")
        f.write("Generated Code:\n")
        f.write(code_output)

    logger.info(f"Codegen output saved to {filename}")
