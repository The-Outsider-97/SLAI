import hashlib
import json
import os
import platform

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

LOCKFILE_PATH = Path("requirements.lock")
REQUIREMENTS_PATH = Path("requirements.txt")
ENV_INVARIANTS_PATH = Path("deployment/environment_invariants.json")


@dataclass
class InvariantArtifacts:
    lockfile_path: Path = LOCKFILE_PATH
    environment_invariants_path: Path = ENV_INVARIANTS_PATH


def _read_requirements(requirements_path: Optional[Path] = None) -> str:
    requirements_path = requirements_path or REQUIREMENTS_PATH
    if not requirements_path.exists():
        raise FileNotFoundError(f"Missing requirements file: {requirements_path}")

    raw_lines = requirements_path.read_text(encoding="utf-8").splitlines()
    normalized = [line.strip() for line in raw_lines if line.strip() and not line.strip().startswith("#")]
    return "\n".join(normalized) + "\n"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_lockfile(requirements_path: Optional[Path] = None, lockfile_path: Optional[Path] = None) -> Path:
    requirements_path = requirements_path or REQUIREMENTS_PATH
    lockfile_path = lockfile_path or LOCKFILE_PATH

    normalized_requirements = _read_requirements(requirements_path)
    content_hash = _hash_text(normalized_requirements)

    payload = {
        "schema_version": 1,
        "requirements_file": str(requirements_path),
        "requirements_sha256": content_hash,
        "requirements": normalized_requirements.splitlines(),
    }

    lockfile_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return lockfile_path


def verify_lockfile(requirements_path: Optional[Path] = None, lockfile_path: Optional[Path] = None) -> bool:
    requirements_path = requirements_path or REQUIREMENTS_PATH
    lockfile_path = lockfile_path or LOCKFILE_PATH

    if not lockfile_path.exists():
        return False

    lock_data = json.loads(lockfile_path.read_text(encoding="utf-8"))
    normalized_requirements = _read_requirements(requirements_path)
    expected_hash = _hash_text(normalized_requirements)
    return lock_data.get("requirements_sha256") == expected_hash


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


def verify_environment_invariants(expected_path: Optional[Path] = None, current: Optional[Dict[str, str]] = None) -> bool:
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
        generate_lockfile(REQUIREMENTS_PATH, LOCKFILE_PATH)
    if not ENV_INVARIANTS_PATH.exists():
        write_environment_invariants(ENV_INVARIANTS_PATH)

    lock_ok = verify_lockfile(REQUIREMENTS_PATH, LOCKFILE_PATH)
    env_ok = verify_environment_invariants(ENV_INVARIANTS_PATH)

    if strict and (not lock_ok or not env_ok):
        failing = []
        if not lock_ok:
            failing.append("lockfile")
        if not env_ok:
            failing.append("environment invariants")
        raise RuntimeError(f"Release invariants verification failed: {', '.join(failing)}")

    return InvariantArtifacts(lockfile_path=LOCKFILE_PATH, environment_invariants_path=ENV_INVARIANTS_PATH)
