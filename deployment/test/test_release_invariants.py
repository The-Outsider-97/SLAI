import json

from pathlib import Path

import deployment.release_invariants as release_invariants


def test_generate_and_verify_lockfile(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("requests==2.32.0\npytest==8.0.0\n", encoding="utf-8")
    lock = tmp_path / "requirements.lock"

    release_invariants.generate_lockfile(req, lock)
    assert release_invariants.verify_lockfile(req, lock) is True

    req.write_text("requests==2.31.0\n", encoding="utf-8")
    assert release_invariants.verify_lockfile(req, lock) is False


def test_write_and_verify_environment_invariants(tmp_path):
    inv = tmp_path / "environment_invariants.json"
    release_invariants.write_environment_invariants(inv, extra={"release_env": "test"})

    current = release_invariants.collect_environment_invariants(extra={"release_env": "test"})
    assert release_invariants.verify_environment_invariants(inv, current=current) is True

    wrong_current = dict(current)
    wrong_current["python_version"] = "0.0.0"
    assert release_invariants.verify_environment_invariants(inv, current=wrong_current) is False


def test_ensure_release_invariants_generates_missing_artifacts(tmp_path, monkeypatch):
    req = tmp_path / "requirements.txt"
    req.write_text("requests==2.32.0\n", encoding="utf-8")
    lock = tmp_path / "requirements.lock"
    env = tmp_path / "environment_invariants.json"

    monkeypatch.setattr(release_invariants, "REQUIREMENTS_PATH", req)
    monkeypatch.setattr(release_invariants, "LOCKFILE_PATH", lock)
    monkeypatch.setattr(release_invariants, "ENV_INVARIANTS_PATH", env)

    release_invariants.ensure_release_invariants(strict=True)

    assert lock.exists()
    assert env.exists()

    data = json.loads(lock.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1


def test_deployment_manager_calls_release_invariants():
    manager_source = Path("deployment/deployment_manager.py").read_text(encoding="utf-8")
    assert "ensure_release_invariants(strict=True)" in manager_source
