import subprocess
import logging
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_current_branch():
    try:
        result = subprocess.run(['git', 'branch', '--show-current'],
                                stdout=subprocess.PIPE, check=True)
        return result.stdout.decode().strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get current branch: {e}")
        return None

def create_branch(branch_name: str, checkout: bool = True):
    try:
        subprocess.run(['git', 'checkout', '-b', branch_name], check=True)
        if not checkout:
            subprocess.run(['git', 'checkout', get_current_branch()])
    except subprocess.CalledProcessError as e:
        logger.error(f"Branch creation failed: {e}")
        raise

def switch_branch(branch_name: str):
    try:
        subprocess.run(['git', 'checkout', branch_name], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Branch switch failed: {e}")
        raise

def merge_branches(source_branch: str, target_branch: str, squash: bool = False):
    try:
        switch_branch(target_branch)
        cmd = ['git', 'merge']
        if squash:
            cmd.append('--squash')
        cmd.append(source_branch)
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Merge failed: {e}")
        raise

# Added branch validation and conflict prevention
def validate_branch_name(branch_name: str) -> bool:
    """Implements Git's branch naming rules from research on version control systems"""
    if re.search(r"\.\.", branch_name) or branch_name.strip() == "":
        return False
    return bool(re.match(r"^(?!\/|.*(?:\/\.|\/\/|@\{|\\))[^\040\177 ~^:?*[]+/[^\040\177 ~^:?*[]+(?<!\.lock)$", branch_name))

def safe_delete_branch(branch_name: str):
    """Implements safe deletion protocol from branching strategies literature"""
    current = get_current_branch()
    if current == branch_name:
        switch_branch("main")  # Prevent deletion of active branch
    delete_branch(branch_name)

def delete_branch(branch_name: str, remote: bool = False):
    try:
        subprocess.run(['git', 'branch', '-d', branch_name], check=True)
        if remote:
            subprocess.run(['git', 'push', 'origin', '--delete', branch_name], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Branch deletion failed: {e}")
        raise

def auto_name_branch(prefix: str = "task", task_desc: str = "", iteration: int = 0):
    safe = re.sub(r'[^a-zA-Z0-9_\-]', '_', task_desc.lower())[:20]
    return f"{prefix}/{safe}/iter-{iteration}"
