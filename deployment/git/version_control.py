import subprocess
import re
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SEMVER_REGEX = r'^v?(\d+)\.(\d+)\.(\d+)$'

def get_latest_git_tag():
    """Fetches the latest git tag."""
    try:
        result = subprocess.run(
            ['git', 'describe', '--tags', '--abbrev=0'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        tag = result.stdout.decode().strip()
        logger.info(f"Latest tag found: {tag}")
        return tag
    except subprocess.CalledProcessError:
        logger.warning("No existing git tags found. Starting from v0.0.0")
        return "v0.0.0"

def increment_version(version: str, level: str = "patch"):
    """
    Increment version based on level: patch, minor, or major.

    Parameters:
    - version (str): Current version (e.g., v1.2.3)
    - level (str): Increment type ('patch', 'minor', 'major')

    Returns:
    - str: New incremented version.
    """
    match = re.match(SEMVER_REGEX, version)
    if not match:
        logger.error(f"Invalid version format: {version}")
        raise ValueError("Invalid SemVer format.")

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
        logger.error(f"Invalid increment level: {level}")
        raise ValueError("Invalid increment level (must be major, minor, patch).")

    new_version = f"v{major}.{minor}.{patch}"
    logger.info(f"Version incremented from {version} to {new_version}")
    return new_version

def create_and_push_tag(new_version: str, message: str = "Automated release"):
    """
    Create a git tag and push it to origin.

    Parameters:
    - new_version (str): Version to tag (e.g., v1.2.4)
    - message (str): Tag message.
    """
    try:
        subprocess.run(['git', 'tag', '-a', new_version, '-m', message], check=True)
        subprocess.run(['git', 'push', 'origin', new_version], check=True)
        logger.info(f"Tag {new_version} created and pushed.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create/push git tag: {e}")

def bump_version_and_tag(level="patch", message="Automated version bump"):
    """
    Bump the latest version and create a new git tag.

    Parameters:
    - level (str): 'patch', 'minor', 'major'
    - message (str): Commit message
    """
    latest_version = get_latest_git_tag()
    new_version = increment_version(latest_version, level=level)
    create_and_push_tag(new_version, message=message)
    return new_version
