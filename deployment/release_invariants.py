import json
import os
import platform
import shutil
import subprocess

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

LOCKFILE_PATH = Path("uv.lock")
PROJECT_PATH = Path("pyproject.toml")
ENV_INVARIANTS_PATH = Path("deployment/environment_invariants.json")


@dataclass
class InvariantArtifacts:
    lockfile_path: Path = LOCKFILE_PATH
    environment_invariants_path: Path = ENV_INVARIANTS_PATH


def _uv_command(project_path: Path, *, check: bool) -> Sequence[str]:
    uv_executable = shutil.which("uv")
    if uv_executable is None:
        raise RuntimeError("uv is required to generate or verify the dependency lock")

    command = [uv_executable, "lock", "--project", str(project_path.parent)]
    if check:
        command.insert(2, "--check")
    return command


def _validate_lock_location(project_path: Path, lockfile_path: Path) -> None:
    expected_path = project_path.parent / "uv.lock"
    if lockfile_path.resolve() != expected_path.resolve():
        raise ValueError(f"uv lockfile must be located at {expected_path}")


def generate_lockfile(project_path: Optional[Path] = None, lockfile_path: Optional[Path] = None) -> Path:
    """Resolve the project dependencies through uv and create ``uv.lock``."""

    project_path = project_path or PROJECT_PATH
    lockfile_path = lockfile_path or LOCKFILE_PATH
    if not project_path.exists():
        raise FileNotFoundError(f"Missing project metadata: {project_path}")
    _validate_lock_location(project_path, lockfile_path)

    result = subprocess.run(
        _uv_command(project_path, check=False),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown uv error"
        raise RuntimeError(f"Failed to generate uv.lock: {detail}")
    if not lockfile_path.exists():
        raise RuntimeError(f"uv completed without creating {lockfile_path}")
    return lockfile_path


def verify_lockfile(project_path: Optional[Path] = None, lockfile_path: Optional[Path] = None) -> bool:
    """Return whether ``uv.lock`` exists and is current for ``pyproject.toml``."""

    project_path = project_path or PROJECT_PATH
    lockfile_path = lockfile_path or LOCKFILE_PATH
    if not project_path.exists() or not lockfile_path.exists():
        return False

    try:
        _validate_lock_location(project_path, lockfile_path)
        result = subprocess.run(
            _uv_command(project_path, check=True),
            capture_output=True,
            text=True,
            check=False,
        )
    except (RuntimeError, ValueError):
        return False
    return result.returncode == 0


def collect_environment_invariants(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    invariants = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "release_env": os.getenv("SLAI_DEPLOY_ENV", "unknown"),
    }
    if extra:
        invariants.update(extra)
    return invariants


def write_environment_invariants(output_path: Optional[Path] = None, extra: Optional[Dict[str, str]] = None) -> Path:
    output_path = output_path or ENV_INVARIANTS_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    invariants = collect_environment_invariants(extra=extra)
    output_path.write_text(json.dumps(invariants, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def verify_environment_invariants(
    expected_path: Optional[Path] = None, current: Optional[Dict[str, str]] = None
) -> bool:
    expected_path = expected_path or ENV_INVARIANTS_PATH
    if not expected_path.exists():
        return False

    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    current_data = current or collect_environment_invariants()

    for key, expected_value in expected.items():
        if current_data.get(key) != expected_value:
            return False
    return True


def ensure_release_invariants(strict: bool = True) -> InvariantArtifacts:
    if not LOCKFILE_PATH.exists():
        generate_lockfile(PROJECT_PATH, LOCKFILE_PATH)
    if not ENV_INVARIANTS_PATH.exists():
        write_environment_invariants(ENV_INVARIANTS_PATH)

    lock_ok = verify_lockfile(PROJECT_PATH, LOCKFILE_PATH)
    env_ok = verify_environment_invariants(ENV_INVARIANTS_PATH)

    if strict and (not lock_ok or not env_ok):
        failing = []
        if not lock_ok:
            failing.append("lockfile")
        if not env_ok:
            failing.append("environment invariants")
        raise RuntimeError(f"Release invariants verification failed: {', '.join(failing)}")

    return InvariantArtifacts(lockfile_path=LOCKFILE_PATH, environment_invariants_path=ENV_INVARIANTS_PATH)
