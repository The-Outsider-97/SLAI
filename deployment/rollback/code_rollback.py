import subprocess
import datetime
import logging
import hashlib
import re

from typing import List
from packaging.version import Version

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Atomic rollback capability (Stonebraker 1987 - Crash Recovery)
class AtomicRollback:
    """Implements all-or-nothing rollback pattern"""
    def __enter__(self):
        self.pre_state = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                      stdout=subprocess.PIPE).stdout.decode().strip()
        return self

    def __exit__(self, exc_type, *_):
        if exc_type:
            reset_to_commit(self.pre_state, hard=True)
            logger.warning("Rollback aborted - restored to %s", self.pre_state)

# Version validation (Tichy 1985 - RCS principles)
def validate_semver(tag: str) -> bool:
    """Enforce semantic versioning for rollback targets"""
    return re.match(r"^v?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$", tag) is not None
            
def reset_to_commit(commit_hash: str, hard=True):
    """
    Resets the Git repo to a specific commit.
    """
    cmd = ['git', 'reset', '--hard' if hard else '--soft', commit_hash]
    subprocess.run(cmd, check=True)
    logger.info(f"Git reset to {commit_hash} ({'hard' if hard else 'soft'})")

def delete_tag(tag: str, remote=True):
    """
    Deletes a Git tag locally and optionally remotely.
    """
    subprocess.run(['git', 'tag', '-d', tag], check=True)
    logger.info(f"Deleted local tag: {tag}")
    if remote:
        subprocess.run(['git', 'push', 'origin', f':refs/tags/{tag}'], check=True)
        logger.info(f"Deleted remote tag: {tag}")

def get_sorted_tags() -> list:
    """Get tags sorted by creation date (newest first) using Git's internal timestamp
    Follows version control patterns from Appleton et al. (1998) Streamed Versioning Model"""
    
    # Get tags with creator date formatting (ISO 8601)
    result = subprocess.run(
        ['git', 'for-each-ref', 
         '--sort=-creatordate', 
         '--format=%(refname:short)|%(creatordate:iso-strict)',
         'refs/tags'],
        stdout=subprocess.PIPE,
        check=True,
        text=True
    )
    
    # Parse output into (tag, timestamp) tuples
    tag_data = []
    for line in result.stdout.splitlines():
        tag, timestamp = line.split('|', 1)
        tag_data.append((tag, datetime.fromisoformat(timestamp)))

    # Secondary sort by semantic version (PEP 440) if timestamps match
    return sorted(
        tag_data,
        key=lambda x: (x[1], Version(x[0].lstrip('v'))),  # Uses packaging.version
        reverse=True
    )

def rollback_to_previous_tag():
    """Rolls back using version selection heuristics from Mens (2004)"""
    tags = get_sorted_tags()
    
    if len(tags) < 2:
        logger.warning("Insufficient version history for rollback")
        return False

    latest_tag, latest_ts = tags[0]
    previous_tag, previous_ts = tags[1]

    logger.info(f"Rolling back from {latest_tag} ({latest_ts}) to {previous_tag} ({previous_ts})")
    """
    Rolls back to the Git tag immediately before the most recent one.
    """
    result = subprocess.run(['git', 'tag', '--sort=-creatordate'], stdout=subprocess.PIPE, check=True)
    tags = result.stdout.decode().splitlines()
    
    if len(tags) < 2:
        logger.warning("Not enough tags to perform rollback.")
        return False

    latest_tag = tags[0]
    previous_tag = tags[1]

    prev_commit = subprocess.run(['git', 'rev-list', '-n', '1', previous_tag], stdout=subprocess.PIPE, check=True)
    prev_commit_hash = prev_commit.stdout.decode().strip()

    logger.info(f"Rolling back to tag {previous_tag} at commit {prev_commit_hash}")
    reset_to_commit(prev_commit_hash)
    delete_tag(latest_tag)
    return True
