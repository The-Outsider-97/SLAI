import subprocess
import logging
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_current_branch():
    """Returns the name of the current git branch."""
    try:
        result = subprocess.run(['git', 'branch', '--show-current'],
                                stdout=subprocess.PIPE, check=True)
        branch = result.stdout.decode().strip()
        logger.info(f"Current branch: {branch}")
        return branch
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get current branch: {e}")
        return None


def create_branch(branch_name: str, checkout: bool = True):
    """Creates a new branch and optionally checks it out."""
    try:
        subprocess.run(['git', 'checkout', '-b', branch_name], check=True)
        logger.info(f"Branch '{branch_name}' created and checked out.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create branch {branch_name}: {e}")
        raise

    if not checkout:
        subprocess.run(['git', 'checkout', get_current_branch()])


def switch_branch(branch_name: str):
    """Switch to an existing branch."""
    try:
        subprocess.run(['git', 'checkout', branch_name], check=True)
        logger.info(f"Switched to branch '{branch_name}'.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to switch to branch {branch_name}: {e}")
        raise


def merge_branches(source_branch: str, target_branch: str, squash: bool = False):
    """Merges source_branch into target_branch. Optionally squash commits."""
    try:
        logger.info(f"Merging branch '{source_branch}' into '{target_branch}' (squash={squash})")

        # Switch to target branch first
        switch_branch(target_branch)

        merge_cmd = ['git', 'merge']
        if squash:
            merge_cmd.append('--squash')

        merge_cmd.append(source_branch)

        subprocess.run(merge_cmd, check=True)
        logger.info(f"Merge completed successfully from '{source_branch}' to '{target_branch}'.")

    except subprocess.CalledProcessError as e:
        logger.error(f"Merge failed from '{source_branch}' to '{target_branch}': {e}")
        raise


def delete_branch(branch_name: str, remote: bool = False):
    """Deletes a local and optionally remote branch."""
    try:
        subprocess.run(['git', 'branch', '-d', branch_name], check=True)
        logger.info(f"Local branch '{branch_name}' deleted.")

        if remote:
            subprocess.run(['git', 'push', 'origin', '--delete', branch_name], check=True)
            logger.info(f"Remote branch '{branch_name}' deleted.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to delete branch '{branch_name}': {e}")


def auto_name_branch(prefix: str = "rsi", task_desc: str = "", iteration: int = 0):
    """Creates a standardized branch name for RSI iterations."""
    safe_task = re.sub(r'[^a-zA-Z0-9_\-]', '_', task_desc.lower())[:20]
    branch_name = f"{prefix}/{safe_task}/iter-{iteration}"
    logger.info(f"Generated branch name: {branch_name}")
    return branch_name
