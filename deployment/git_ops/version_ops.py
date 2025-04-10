"""
Combined Git Handler and Version Control Module for SLAI
- Provides commit/push, semantic versioning, tagging, rollback utilities
- Environment-aware for multi-env pipelines
"""

import os
import subprocess
import time
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

REPO_PATH = os.path.abspath("./")  # Root of SLAI repo
SEMVER_REGEX = r'^v?(\d+)\.(\d+)\.(\d+)$'


def run_git_command(args, cwd=REPO_PATH, check=True):
    """Wrapper to run git commands with logging."""
    try:
        result = subprocess.run(['git'] + args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)
        stdout = result.stdout.decode().strip()
        stderr = result.stderr.decode().strip()
        if stderr:
            logger.warning(f"Git stderr: {stderr}")
        return stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Git command failed: {e.stderr.decode().strip()}")
        raise


def commit_and_push(file_path: str, commit_message: str = "Automated Commit", branch: str = "main"):
    """Stage, commit, and push a file to a target branch."""
    full_path = os.path.join(REPO_PATH, file_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    if not os.path.exists(full_path):
        logger.warning(f"File to commit does not exist: {full_path}")
        return False

    run_git_command(["add", file_path])
    run_git_command(["commit", "-m", commit_message])
    run_git_command(["push", "origin", branch])
    logger.info(f"Committed and pushed {file_path} to {branch}.")
    return True


def get_latest_git_tag() -> str:
    """Fetch the latest Git tag or return v0.0.0."""
    try:
        return run_git_command(["describe", "--tags", "--abbrev=0"])
    except Exception:
        return "v0.0.0"


def increment_version(version: str, level: str = "patch") -> str:
    """Increment a semantic version string."""
    match = re.match(SEMVER_REGEX, version)
    if not match:
        raise ValueError(f"Invalid version format: {version}")

    major, minor, patch = map(int, match.groups())

    if level == "major":
        major += 1
        minor = 0
        patch = 0
    elif level == "minor":
        minor += 1
        patch = 0
    elif level == "patch":
        patch += 1
    else:
        raise ValueError("Level must be one of: major, minor, patch")

    return f"v{major}.{minor}.{patch}"


def create_tag(version: str, message: str = "Release"):
    """Create and push annotated Git tag."""
    run_git_command(["tag", "-a", version, "-m", message])
    run_git_command(["push", "origin", version])
    logger.info(f"Created and pushed tag {version}.")


def bump_version_and_tag(level: str = "patch", message: str = "Auto bump") -> str:
    """Bump version and create/push tag."""
    latest = get_latest_git_tag()
    new_ver = increment_version(latest, level)
    create_tag(new_ver, message)
    return new_ver

# Added version validation inspired by semantic versioning research
def validate_semver(version: str) -> bool:
    """Strict validation based on SemVer 2.0 specification"""
    return bool(re.fullmatch(
        r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$", 
        version
    ))

def rollback_to_tag(tag: str):
    """Implements transactional rollback pattern with audit trail"""
    audit_hash = run_git_command(["rev-parse", "HEAD"])
    logger.info(f"Initiated rollback from {audit_hash} to {tag}")

    """Rollback to a given tag or previous release."""
    if not tag:
        tags = run_git_command(["tag", "--sort=-creatordate"]).splitlines()
        if len(tags) < 2:
            raise RuntimeError("No previous tag found for rollback.")
        tag = tags[1]  # second most recent

    commit = run_git_command(["rev-list", "-n", "1", tag])
    run_git_command(["reset", "--hard", commit])
    run_git_command(["tag", "-d", tag])
    run_git_command(["push", "origin", f":refs/tags/{tag}"])
    logger.info(f"Rolled back to tag {tag} and deleted it.")
    return tag
