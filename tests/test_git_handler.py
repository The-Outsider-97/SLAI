import pytest
import subprocess
from unittest.mock import patch, MagicMock
import logging

# Import the module you're testing
from deployment.git import git_handler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@pytest.fixture
def mock_subprocess_run():
    """Fixture to patch subprocess.run across all tests."""
    with patch("subprocess.run") as mock_run:
        yield mock_run


def test_commit_and_push_success(mock_subprocess_run):
    """Test that commit_and_push() runs expected subprocess commands."""
    mock_subprocess_run.return_value = MagicMock(returncode=0)

    # Example parameters for the test
    dummy_code = "print('Hello, World!')"
    dummy_file_path = "agents/test_agent.py"
    dummy_commit_message = "Test commit from RSI"

    # Run the function
    git_handler.commit_and_push(
        code_string=dummy_code,
        file_path=dummy_file_path,
        commit_message=dummy_commit_message
    )

    # Validate subprocess.run calls
    assert mock_subprocess_run.call_count >= 3, "Expected multiple subprocess calls for git commands"

    commands = [call.args[0] for call in mock_subprocess_run.call_args_list]

    assert any(["git", "add", dummy_file_path] == cmd for cmd in commands), "Expected git add command"
    assert any("git commit" in " ".join(cmd) for cmd in commands), "Expected git commit command"
    assert any("git push" in " ".join(cmd) for cmd in commands), "Expected git push command"


def test_commit_and_push_failure(mock_subprocess_run):
    """Test commit_and_push() handles subprocess failure."""
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd="git push")

    dummy_code = "print('Failure test')"
    dummy_file_path = "agents/broken_agent.py"
    dummy_commit_message = "Should fail"

    with pytest.raises(subprocess.CalledProcessError):
        git_handler.commit_and_push(
            code_string=dummy_code,
            file_path=dummy_file_path,
            commit_message=dummy_commit_message
        )


@pytest.mark.parametrize("tag_message", [
    "Automated tag release",
    "Patch release v1.0.1",
    "Rollback recovery release"
])
def test_tag_release_success(mock_subprocess_run, tag_message):
    """Test tag_release() runs expected subprocess commands with valid tags."""
    mock_subprocess_run.return_value = MagicMock(returncode=0)

    dummy_tag = "v1.0.0"

    git_handler.tag_release(
        dummy_tag,
        message=tag_message
    )

    assert mock_subprocess_run.call_count == 2, "Expected 2 subprocess calls (tag + push)"

    commands = [call.args[0] for call in mock_subprocess_run.call_args_list]

    assert any(dummy_tag in cmd for cmd in commands for arg in cmd), "Tag missing in git commands"
    assert any("git push" in " ".join(cmd) for cmd in commands), "Expected git push command"


def test_tag_release_failure(mock_subprocess_run):
    """Test tag_release() handles subprocess failure."""
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd="git tag")

    with pytest.raises(subprocess.CalledProcessError):
        git_handler.tag_release(
            "v1.0.0",
            message="Fail test"
        )


def test_commit_and_push_file_write(tmp_path, mock_subprocess_run):
    """Test that the file is written with correct code before git commands."""
    mock_subprocess_run.return_value = MagicMock(returncode=0)

    # Create a temporary file path
    dummy_file = tmp_path / "temp_agent.py"
    dummy_code = "print('Temporary code')"

    git_handler.commit_and_push(
        code_string=dummy_code,
        file_path=str(dummy_file),
        commit_message="Temp file test"
    )

    # Check file content
    with open(dummy_file, "r") as f:
        contents = f.read()

    assert contents == dummy_code, "File content doesn't match input code"


def test_commit_and_push_invalid_path(mock_subprocess_run):
    """Test commit_and_push() with an invalid file path."""
    mock_subprocess_run.return_value = MagicMock(returncode=0)

    dummy_code = "print('Invalid path test')"
    dummy_file_path = "/invalid/path/agent.py"

    with pytest.raises(FileNotFoundError):
        git_handler.commit_and_push(
            code_string=dummy_code,
            file_path=dummy_file_path,
            commit_message="Invalid path"
        )
