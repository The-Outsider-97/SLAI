import os
import subprocess
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

REPO_PATH = os.path.abspath("./")

# Security validation inspired by secure repository patterns
def _sanitize_path(input_path: str) -> str:
    """Prevent path traversal attacks using canonicalization"""
    resolved = os.path.normpath(input_path)
    if not resolved.startswith(os.path.abspath(REPO_PATH)):
        raise ValueError("Invalid path outside repository boundary")
    return resolved

def commit_and_push(code_string: str, file_path: str, commit_message: str, branch: str = "main"):
    file_path = _sanitize_path(file_path)
    # Content validation from software artifact research
    if len(code_string) > 1_000_000:
        raise ValueError("File size exceeds academic recommended limit")

    full_path = os.path.join(REPO_PATH, file_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    with open(full_path, "w", encoding="utf-8") as f:
        f.write(code_string)

    try:
        subprocess.run(["git", "add", file_path], cwd=REPO_PATH, check=True)
        subprocess.run(["git", "commit", "-m", commit_message], cwd=REPO_PATH, check=True)
        subprocess.run(["git", "push", "origin", branch], cwd=REPO_PATH, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Git commit/push failed: {e}")
        raise
