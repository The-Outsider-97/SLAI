from __future__ import annotations

import ast
import importlib
import importlib.util
import sys

from pathlib import Path

import pytest
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_replay_buffer_module():
    module_path = REPOSITORY_ROOT / "src/utils/buffer/replay_buffer.py"
    spec = importlib.util.spec_from_file_location("slai_uniform_replay_buffer", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_version_ops_module():
    module_path = REPOSITORY_ROOT / "deployment/git_ops/version_ops.py"
    spec = importlib.util.spec_from_file_location("slai_version_ops", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_planning_package_uses_lazy_relative_exports() -> None:
    init_path = REPOSITORY_ROOT / "src/agents/planning/__init__.py"
    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    planning = importlib.import_module("src.agents.planning")

    assert all(alias.name != "*" for node in imports for alias in node.names)
    assert "SafetyPlanning" in planning.__all__
    assert "DeadlineAwareScheduler" in planning.__all__
    assert all(module_name.startswith(".") for module_name, _ in planning._EXPORTS.values())


def test_autonomous_task_config_is_a_complete_task_mapping() -> None:
    config_path = REPOSITORY_ROOT / "src/agents/base/configs/agents_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tasks = config["evaluation_agent"]["autonomous_tasks"]

    assert len(tasks) == 1
    assert set(
        (
            "id",
            "type",
            "path",
            "optimal_path",
            "completion_time",
            "energy_consumed",
            "collisions",
            "success",
        )
    ).issubset(tasks[0])


def test_uniform_replay_buffer_is_bounded_and_samples_without_mutation() -> None:
    ReplayBuffer = _load_replay_buffer_module().ReplayBuffer
    buffer = ReplayBuffer(capacity=3, seed=7)

    for value in range(5):
        buffer.push((value, value + 1))

    assert len(buffer) == 3
    assert buffer.get_all() == [(2, 3), (3, 4), (4, 5)]

    before = buffer.get_all()
    sample = buffer.sample(2)
    assert len(sample) == 2
    assert len(set(sample)) == 2
    assert all(item in before for item in sample)
    assert buffer.get_all() == before


def test_replay_backend_names_are_normalized_consistently() -> None:
    replay = _load_replay_buffer_module()

    assert replay.normalize_replay_backend("uniform") == "uniform"
    assert replay.normalize_replay_backend("legacy_uniform") == "uniform"
    assert replay.normalize_replay_backend("legacy_per") == "prioritized"
    assert replay.normalize_replay_backend(None) == "distributed"
    with pytest.raises(ValueError, match="Unknown replay backend"):
        replay.normalize_replay_backend("unsupported")


def test_version_bump_uses_the_defined_tag_publisher(monkeypatch) -> None:
    version_ops = _load_version_ops_module()
    published = []
    monkeypatch.setattr(version_ops, "get_latest_git_tag", lambda: "v2.2.0")
    monkeypatch.setattr(
        version_ops,
        "create_and_push_tag",
        lambda version, message: published.append((version, message)),
    )

    assert version_ops.bump_version_and_tag("patch", "Foundation hardening") == "v2.2.1"
    assert published == [("v2.2.1", "Foundation hardening")]
