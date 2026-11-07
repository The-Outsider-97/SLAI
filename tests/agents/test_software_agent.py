"""SoftwareAgent facade: request contract, dispatch, SharedMemory publication and isolation."""

import json
import pathlib

import pytest

import src.agents as agents_package
from src.agents.agent_factory import AgentFactory
from src.agents.base.utils.base_errors import BaseConfigurationError, BaseStateError
from src.agents.base.utils.main_config_loader import get_config_section
from src.agents.collaborative.shared_memory import SharedMemory
from src.agents.software import HANDOFF_GATES, SoftwareError, SoftwareRequestError, UnsupportedSoftwareOperationError
from src.agents.software_agent import (
    SoftwareAgent,
    SoftwareAgentRequest,
    SoftwareOperation,
    SoftwareSource,
)

FILES = {
    "app/__init__.py": "from .core import compute\n__all__ = ['compute']\n",
    "app/core.py": (
        "import os, sys\n"
        "class Base:\n    def compute(self, x):\n        return x\n"
        "class Sub(Base):\n    def compute(self, x):\n        return super().compute(x) + 1\n"
        "def compute(a):\n    return Sub().compute(a)\n"
        "def user():\n    v = compute(2)\n    return v\n"
        "def _dead():\n    return 1\n"
    ),
    "app/other.py": "from app.core import compute as c\nx = c(1)\n",
    "tests/test_core.py": "from app.core import compute\n\n\ndef test_compute():\n    assert compute(1)\n",
}


class BrokenMemory:
    """A store that is reachable for reads but rejects every write (SharedMemory itself is a singleton)."""

    def get(self, key, default=None):
        return default

    def set(self, *args, **kwargs):
        raise RuntimeError("store down")


def make_agent(**config):
    memory = SharedMemory()
    return SoftwareAgent(shared_memory=memory, agent_factory=AgentFactory(), config=config), memory


def run(agent, operation, parameters=None, **payload):
    outcome = agent.perform_task({"operation": operation, "parameters": parameters or {}, **payload})
    assert outcome["execution_success"] is True
    assert outcome["proposal_applied"] is False
    return outcome


# --------------------------------------------------------------------------- registration and isolation

def test_agent_is_registered_in_factory_package_and_configuration():
    assert AgentFactory.DEFAULT_AGENT_SPECS["software"] == {"module_path": "src.agents.software_agent", "class_name": "SoftwareAgent"}
    assert AgentFactory.DEFAULT_ALIASES["software_agent"] == "software"
    assert AgentFactory.DEFAULT_DEPENDENCY_PROFILES["software"]["torch_required"] is False
    assert agents_package.SoftwareAgent is SoftwareAgent
    section = get_config_section("software_agent")
    assert section["enabled"] is True and section["publish_proposal_diffs"] is True


def test_agent_is_unaware_of_subsystem_memory_and_configuration():
    source = pathlib.Path(__import__("src.agents.software_agent", fromlist=["x"]).__file__).read_text(encoding="utf-8")
    for token in ("software_memory", "SoftwareMemory", "software_config", "utils.config_loader", "LRUCache"):
        assert token not in source, token
    assert "get_config_section(\"software_agent\")" in source


def test_capabilities_state_the_safety_posture():
    agent, _ = make_agent()
    caps = agent.capabilities()
    assert caps["proposals_only"] is True and caps["executes_commands"] is False
    assert caps["handoff_gates"] == HANDOFF_GATES
    assert set(caps["operations"]) == {item.value for item in SoftwareOperation}


# --------------------------------------------------------------------------- request contract

def test_request_normalization():
    request = SoftwareAgentRequest.from_mapping(
        {
            "operation": " Rename ",
            "files": FILES,
            "source_roots": ["src"],
            "baseline": {"snapshot": "abc"},
            "parameters": {"symbol": "a.b"},
            "metadata": {"ticket": 7},
            "prior_request_ids": ["x", "x", "y"],
        }
    )
    assert request.operation is SoftwareOperation.RENAME
    assert request.source.source_roots == ("src",)
    assert request.baseline.snapshot == "abc"
    assert request.prior_request_ids == ("x", "y")
    assert request.request_id


def test_malformed_requests_are_rejected_before_any_work():
    bad = [
        {},
        {"operation": "summary", "parameters": []},
        {"operation": "summary", "files": FILES, "snapshot": "abc"},
        {"operation": "summary", "changes": {"a.py": "x = 1\n"}},
        {"operation": "summary", "files": ["a.py"]},
        {"operation": "summary", "files": FILES, "metadata": {"nested": {"a": 1}}},
        {"operation": "summary", "files": FILES, "prior_request_ids": [""]},
    ]
    for payload in bad:
        with pytest.raises(SoftwareRequestError):
            SoftwareAgentRequest.from_mapping(payload)
    with pytest.raises(UnsupportedSoftwareOperationError):
        SoftwareAgentRequest.from_mapping({"operation": "execute_shell"})
    with pytest.raises(SoftwareRequestError):
        SoftwareSource()


# --------------------------------------------------------------------------- read-only operations

def test_every_read_operation_returns_its_report():
    agent, _ = make_agent(publish_shared_memory=False)
    cases = [
        ("summary", {}, {"snapshot", "graphs", "index"}),
        ("outline", {"paths": ["app/core.py"]}, {"files"}),
        ("symbol", {"symbol": "app.core.compute"}, {"qualified_name", "kind"}),
        ("search", {"text": "compute"}, {"count", "hits"}),
        ("search", {"text": "compute", "grep": True}, {"count", "hits"}),
        ("references", {"symbol": "app.core.compute"}, {"count", "references"}),
        ("dependency_graph", {}, {"cycles", "summary"}),
        ("dependency_graph", {"module": "app.other", "direction": "dependencies"}, {"modules"}),
        ("dependency_graph", {"module": "app.other", "target": "app.core"}, {"path"}),
        ("call_graph", {}, {"cycles", "summary"}),
        ("call_graph", {"symbol": "app.core.user", "transitive": True}, {"symbols"}),
        ("call_graph", {"symbol": "app.core.user", "target": "app.core.Sub.compute"}, {"path"}),
        ("class_hierarchy", {}, {"classes", "roots"}),
        ("static_analysis", {}, {"findings", "summary"}),
        ("dead_code", {}, {"items", "entry_points"}),
        ("change_impact", {"symbols": ["app.core.compute"]}, {"impacted_modules", "tests", "risk_level"}),
        ("architecture", {"forbidden": [{"from": "app.other*", "to": "app.core*", "reason": "r"}]}, {"layer_violations"}),
    ]
    for operation, parameters, keys in cases:
        outcome = run(agent, operation, parameters, files=FILES)
        assert keys <= set(outcome["result"]), (operation, sorted(outcome["result"]))
        assert outcome["patch_id"] is None and outcome["requires_gates"] == []
        assert outcome["snapshot_digest"]
    forbidden = run(agent, "architecture", cases[-1][1], files=FILES)["result"]["layer_violations"]
    assert [item["rule"] for item in forbidden] == ["r"]


def test_options_override_analyzer_thresholds_per_request():
    agent, _ = make_agent(publish_shared_memory=False)
    default = run(agent, "static_analysis", files={"m.py": "import os\n"})["result"]["findings"]
    muted = run(agent, "static_analysis", files={"m.py": "import os\n"}, options={"disabled_rules": ["SW102"]})["result"]["findings"]
    assert len(default) == 1 and muted == []


def test_equal_inputs_give_equal_results():
    agent, _ = make_agent(publish_shared_memory=False)
    first = run(agent, "static_analysis", files=FILES)
    second = run(agent, "static_analysis", files=FILES)
    assert first["snapshot_digest"] == second["snapshot_digest"]
    assert first["result"] == second["result"]


# --------------------------------------------------------------------------- patch proposals

def test_patch_chain_is_tracked_through_base_patch_id_and_never_applied():
    agent, _ = make_agent(publish_shared_memory=False)
    submitted = dict(FILES)
    renamed = run(agent, "rename", {"symbol": "app.core.user", "new_name": "run_user"}, files=submitted)
    assert submitted == FILES  # the caller's inputs are never mutated
    assert renamed["requires_gates"] == list(HANDOFF_GATES)
    assert "+def run_user" in renamed["result"]["diff"]
    first = renamed["patch_id"]

    cleaned = run(agent, "remove_unused_imports", base_patch_id=first)
    pruned = run(agent, "remove_dead_code", base_patch_id=cleaned["patch_id"])
    assert cleaned["result"]["proposal"]["parent_patch_id"] == first
    assert pruned["result"]["proposal"]["parent_patch_id"] == cleaned["patch_id"]

    fetched = run(agent, "proposal", {"patch_id": pruned["patch_id"], "include_reverse_diff": True})["result"]
    assert [entry["patch_id"] for entry in fetched["lineage"]] == [first, cleaned["patch_id"], pruned["patch_id"]]
    assert fetched["reverse_diff"] and fetched["diff"]


def test_diff_and_migration_proposals():
    agent, _ = make_agent(publish_shared_memory=False)
    made = run(agent, "rename", {"symbol": "app.core.user", "new_name": "run_user"}, files=FILES)
    replay = run(agent, "apply_diff", {"diff": made["result"]["diff"], "intent": "replay"}, files=FILES)
    assert replay["result"]["proposal"]["result_snapshot_digest"] == made["result"]["proposal"]["result_snapshot_digest"]
    migrated = run(
        agent,
        "migrate_api",
        {"rules": [{"kind": "rename_keyword", "old": "lib.f", "keyword": "old", "new": "renamed"}]},
        files={"m.py": "from lib import f\nf(old=1)\n"},
    )
    assert "+f(renamed=1)" in migrated["result"]["diff"]
    with pytest.raises(SoftwareRequestError):
        run(agent, "migrate_api", {"rules": []}, files=FILES)


def test_patch_review_operations_use_the_recorded_patch_or_a_baseline():
    agent, _ = make_agent(publish_shared_memory=False)
    base = {"files": FILES}
    broken = dict(FILES, **{"app/core.py": FILES["app/core.py"].replace("def compute(a):", "def compute(a, b):")})

    compat = run(agent, "compatibility", {}, files=broken, baseline=base)["result"]
    assert compat["recommended_bump"] == "major"
    impact = run(agent, "change_impact", {}, files=broken, baseline=base)["result"]
    assert "app.core" in impact["changed_modules"]

    patch = run(agent, "rename", {"symbol": "app.core.compute", "new_name": "calculate"}, files=FILES)["patch_id"]
    by_patch = run(agent, "compatibility", {"patch_id": patch})["result"]
    assert by_patch["recommended_bump"] == "major"
    with pytest.raises(SoftwareRequestError):
        run(agent, "compatibility", {"patch_id": patch}, files=FILES)
    with pytest.raises(SoftwareRequestError):
        run(agent, "compatibility", {}, files=FILES)


# --------------------------------------------------------------------------- operational errors

def test_operational_errors_raise_software_errors_and_are_published():
    agent, memory = make_agent()
    cases = [
        {"operation": "rename", "files": FILES, "parameters": {"symbol": "app.core.nothing", "new_name": "x"}, "request_id": "e1"},
        {"operation": "rename", "parameters": {"symbol": "a", "new_name": "b"}, "request_id": "e2"},
        {"operation": "rename", "files": FILES, "parameters": {"symbol": "app.core.user"}, "request_id": "e3"},
        {"operation": "proposal", "parameters": {"patch_id": "patch-missing"}, "request_id": "e4"},
    ]
    for payload in cases:
        with pytest.raises(SoftwareError):
            agent.perform_task(payload)
        record = memory.get(f"software_agent.error:{payload['request_id']}")
        assert record["execution_success"] is False and record["error_type"].startswith(("Software", "Unsupported"))


def test_unexpected_exceptions_are_wrapped():
    agent, _ = make_agent(publish_shared_memory=False)

    def boom(request):
        raise RuntimeError("boom")

    agent._operation_handlers[SoftwareOperation.SUMMARY] = boom
    with pytest.raises(SoftwareError) as caught:
        run(agent, "summary", files=FILES)
    assert isinstance(caught.value.__cause__, RuntimeError)


# --------------------------------------------------------------------------- SharedMemory

def test_results_summaries_proposals_and_events_are_published():
    agent, memory = make_agent()
    outcome = run(agent, "rename", {"symbol": "app.core.user", "new_name": "run_user"}, files=FILES, request_id="pub-1", metadata={"ticket": "T-1", "api_key": "sk-secret-123456"})
    patch_id = outcome["patch_id"]

    request = memory.get("software_agent.request:pub-1")
    assert request["operation"] == "rename" and request["source"]["files"] == len(FILES)
    assert "def compute" not in json.dumps(request)
    assert "sk-secret-123456" not in json.dumps(request)
    assert request["parameters"] == {"symbol": "app.core.user", "new_name": "run_user"}

    result = agent.get_software_result("pub-1")
    assert result["patch_id"] == patch_id and result["proposal_applied"] is False
    assert result["result"]["diff"] == outcome["result"]["diff"]

    summary = memory.get("software_agent.summary:pub-1")
    assert summary["patch_id"] == patch_id and summary["fingerprint"] == result["fingerprint"]
    assert memory.get("software_agent.latest")["request_id"] == "pub-1"

    proposal = memory.get(f"software_agent.proposal:{patch_id}")
    assert proposal["kind"] == "rename_symbol" and proposal["requires_gates"] == list(HANDOFF_GATES)
    assert "diff" not in proposal
    assert memory.get("software_agent.event:pub-1:completed")["patch_id"] == patch_id


def test_shared_diffs_are_bounded_or_withheld_by_configuration():
    agent, _ = make_agent(max_shared_diff_chars=40)
    run(agent, "rename", {"symbol": "app.core.user", "new_name": "run_user"}, files=FILES, request_id="cap")
    shared = agent.get_software_result("cap")["result"]
    assert len(shared["diff"]) == 40 and shared["diff_truncated_chars"] > 0

    withheld, _ = make_agent(publish_proposal_diffs=False)
    run(withheld, "rename", {"symbol": "app.core.user", "new_name": "run_user"}, files=FILES, request_id="off")
    shared = withheld.get_software_result("off")["result"]
    assert "diff" not in shared and shared["diff_omitted"] is True


def test_prior_references_are_resolved_from_summaries():
    agent, _ = make_agent()
    run(agent, "summary", files=FILES, request_id="first")
    outcome = run(agent, "summary", files=FILES, prior_request_ids=["first", "never-ran"])
    assert [item["request_id"] for item in outcome["prior_references"]] == ["first"]


def test_publication_can_be_disabled_or_degrade_gracefully():
    quiet, memory = make_agent(publish_shared_memory=False)
    run(quiet, "summary", files=FILES, request_id="quiet")
    assert memory.get("software_agent.summary:quiet") is None and quiet.get_software_result("quiet") is None

    degraded = SoftwareAgent(shared_memory=BrokenMemory(), agent_factory=AgentFactory())
    assert run(degraded, "summary", files=FILES)["result"]["snapshot"]
    with pytest.raises(BaseStateError):
        SoftwareAgent(shared_memory=BrokenMemory(), agent_factory=AgentFactory(), config={"fail_on_shared_memory_error": True})


# --------------------------------------------------------------------------- configuration

def test_allowed_operations_enabled_flag_and_invalid_configuration():
    narrow, _ = make_agent(allowed_operations=["summary", "Search"], publish_shared_memory=False)
    assert set(narrow.capabilities()["operations"]) == {"summary", "search"}
    run(narrow, "summary", files=FILES)
    with pytest.raises(UnsupportedSoftwareOperationError):
        run(narrow, "rename", {"symbol": "a", "new_name": "b"}, files=FILES)

    disabled, _ = make_agent(enabled=False, publish_shared_memory=False)
    with pytest.raises(BaseConfigurationError):
        run(disabled, "summary", files=FILES)

    for config in ({"allowed_operations": ["bogus"]}, {"allowed_operations": "summary"}, {"result_key_prefix": ""}):
        with pytest.raises(BaseConfigurationError):
            make_agent(**config)


# --------------------------------------------------------------------------- BaseAgent integration

def test_predict_act_and_metrics():
    agent, _ = make_agent(publish_shared_memory=False)
    assert agent.predict({"operation": "summary", "files": FILES})["operation"] == "summary"
    assert agent.act({"operation": "summary", "files": FILES}, context={"k": 1})["execution_success"] is True
    with pytest.raises(SoftwareRequestError):
        agent.predict("summary")
    with pytest.raises(SoftwareRequestError):
        agent.perform_task(["summary"])

    patch = run(agent, "rename", {"symbol": "app.core.user", "new_name": "run_user"}, files=FILES)
    metrics = agent.extract_performance_metrics(patch)
    assert metrics["software.proposal_emitted"] == 1.0 and 0.0 < metrics["software.proposal_confidence"] <= 1.0
    assert metrics["software.proposal_files"] == 1.0
    read = agent.extract_performance_metrics(run(agent, "static_analysis", files=FILES))
    assert read["software.proposal_emitted"] == 0.0 and read["software.findings"] >= 1.0
    assert agent.extract_performance_metrics("nope") == {}
