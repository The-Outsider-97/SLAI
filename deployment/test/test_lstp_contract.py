from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_lstp_lock_is_pinned_under_model_directory() -> None:
    lock = json.loads(
        (ROOT / "model" / "lstp.lock.json").read_text(encoding="utf-8")
    )
    assert lock["target"] == "model/LSTP"
    assert lock["protocol_version"] == "0.1"
    assert lock["package_version"] == "0.1.0a1"
    assert len(lock["commit"]) == 40


def test_lstp_integration_uses_installation_not_sys_path_mutation() -> None:
    for relative in ("setup_lstp.py", "run_lstp.py"):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "sys.path.insert" not in source
        assert "sys.path.append" not in source
        assert "sys.path.extend" not in source


def test_language_agent_lstp_projection_is_opt_in_and_non_authorizing() -> None:
    source = (ROOT / "src" / "agents" / "language_agent.py").read_text(
        encoding="utf-8"
    )
    config = (
        ROOT
        / "src"
        / "agents"
        / "language"
        / "configs"
        / "language_config.yaml"
    ).read_text(encoding="utf-8")

    assert "def _build_lstp_metadata" in source
    assert '"authority_inferred": False' in source
    assert 'cfg.get("lstp", {})' in source
    assert "lstp:" in config
    assert "enabled: false" in config


def test_pinned_clone_is_not_vendored_into_slai() -> None:
    ignored = (ROOT / "model" / ".gitignore").read_text(encoding="utf-8")
    assert "LSTP/" in ignored
