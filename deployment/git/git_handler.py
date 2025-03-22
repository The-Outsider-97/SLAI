import os
import subprocess
import logging
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# This assumes you already did `git init` and have a remote set up (origin/main)
REPO_PATH = os.path.abspath("./")  # Root of your SLAI repo

def commit_and_push(code_string: str, file_path: str, commit_message: str = "Automated RSI Commit"):
    """
    Saves code to file, stages, commits, and pushes it to Git.
    
    Parameters:
    - code_string (str): Code to be saved and committed.
    - file_path (str): Relative path inside repo where code is stored.
    - commit_message (str): Git commit message.
    """
    full_file_path = os.path.join(REPO_PATH, file_path)

    # Ensure target dir exists
    os.makedirs(os.path.dirname(full_file_path), exist_ok=True)

    # Write new code to the file
    with open(full_file_path, "w", encoding="utf-8") as f:
        f.write(code_string)

    logger.info(f"Code saved to {full_file_path}")

    try:
        # Git add, commit, and push commands
        subprocess.run(["git", "add", file_path], cwd=REPO_PATH, check=True)
        subprocess.run(["git", "commit", "-m", commit_message], cwd=REPO_PATH, check=True)
        subprocess.run(["git", "push", "origin", "main"], cwd=REPO_PATH, check=True)

        logger.info(f"Code committed and pushed with message: '{commit_message}'")

    except subprocess.CalledProcessError as e:
        logger.error(f"Git operation failed: {e}")

def tag_release(tag_message: str):
    """
    Creates a git tag and pushes it to the remote.
    
    Parameters:
    - tag_message (str): Description of the tag (version notes, etc).
    """
    version_tag = f"rsi-v{int(time.time())}"

    try:
        subprocess.run(["git", "tag", "-a", version_tag, "-m", tag_message], cwd=REPO_PATH, check=True)
        subprocess.run(["git", "push", "origin", version_tag], cwd=REPO_PATH, check=True)

        logger.info(f"Git tag '{version_tag}' created and pushed successfully.")

    except subprocess.CalledProcessError as e:
        logger.error(f"Git tagging failed: {e}")
