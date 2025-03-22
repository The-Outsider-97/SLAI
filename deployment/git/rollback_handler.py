import subprocess
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def reset_to_commit(commit_hash: str, hard_reset: bool = True):
    """
    Resets the git repository to a specific commit.

    Parameters:
    - commit_hash (str): Hash to reset to.
    - hard_reset (bool): Whether to perform a hard reset.
    """
    cmd = ['git', 'reset', '--hard' if hard_reset else '--soft', commit_hash]
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Successfully reset to commit {commit_hash}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to reset to commit {commit_hash}: {e}")

def delete_tag(tag: str, remote: bool = True):
    """
    Deletes a local and optionally remote git tag.

    Parameters:
    - tag (str): Tag to delete.
    - remote (bool): Whether to delete the tag from remote.
    """
    try:
        subprocess.run(['git', 'tag', '-d', tag], check=True)
        logger.info(f"Deleted local tag {tag}")

        if remote:
            subprocess.run(['git', 'push', 'origin', f':refs/tags/{tag}'], check=True)
            logger.info(f"Deleted remote tag {tag}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to delete tag {tag}: {e}")

def rollback_to_previous_release():
    """
    Rolls back to the previous git tag (semantic versioning assumed).
    """
    try:
        # Get list of tags, ordered by version
        result = subprocess.run(['git', 'tag', '--sort=-creatordate'], stdout=subprocess.PIPE, check=True)
        tags = result.stdout.decode().splitlines()

        if len(tags) < 2:
            logger.warning("No previous tag available for rollback.")
            return

        latest_tag = tags[0]
        previous_tag = tags[1]

        # Get commit hash of previous tag
        prev_commit = subprocess.run(['git', 'rev-list', '-n', '1', previous_tag], stdout=subprocess.PIPE, check=True)
        prev_commit_hash = prev_commit.stdout.decode().strip()

        logger.info(f"Rolling back to tag {previous_tag} at commit {prev_commit_hash}")

        # Reset to previous commit and delete the bad tag
        reset_to_commit(prev_commit_hash)
        delete_tag(latest_tag)

    except subprocess.CalledProcessError as e:
        logger.error(f"Rollback failed: {e}")
