"""Evaluation Agent orchestration regressions (domain-neutral and dependency-light)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agents.evaluation_agent import EvaluationAgent


class FakeMemory:
    def __init__(self, percent=15.0):
        self.percent = percent
        self.data = {}
        self.history = []

    def get(self, key, default=None):
        return self.data.get(key, default)

    def metrics(self):
        return {"access_count": 0}

    def get_usage_stats(self):
        return {
            "memory_usage_percentage": self.percent,
            "available_memory_mb": 500,
            "item_count": len(self.data),
        }

    def set(self, key, val):
        self.data[key] = val

    def append(self, key, val):
        self.history.append((key, val))


def make_agent(*, flow=None, percent=15.0):
    agent = EvaluationAgent.__new__(EvaluationAgent)
    agent.shared_memory = FakeMemory(percent)
    agent.evaluators = {}
    agent.initialization_issues = []
    agent.agent_config = {}
    agent.protocol = SimpleNamespace(
        static_analysis={"enable": False},
        behavioral_testing={"test_types": []},
        get_enabled_evaluation_flow=lambda: flow or {
            "enable_performance": False, "enable_efficiency": False,
            "enable_resource": False, "enable_statistical": False,
        },
    )
    agent.memory_warning_threshold = 0.7
    agent.memory_critical_threshold = 0.9
    return agent


def test_failing_optional_stage_does_not_hide_evaluation_error():
    agent = make_agent()
    result = agent.execute_validation_cycle({
        "lightweight": True, "persist_results": False,
        "resource_result": {"status": "error", "error": "unavailable sensor"},
    })
    assert result["status"] == "critical"
    assert any("resource" in str(issue) for issue in result["critical_issues"])
    assert result["completed_at"]
    assert result["duration_ms"] >= 0


def test_incomplete_budget_and_no_contracts_are_never_reported_as_healthy():
    agent = make_agent()
    for value in ({"status": "incomplete", "summary": {"missing_evidence": 2}},
                  {"status": "no_contracts", "summary": {"missing_evidence": 0}}):
        warnings = agent._evaluate_validation_health(
            {"performance_budget": value}, {}
        )
        assert warnings["component_status"]["performance_budget"] == "warning"
        assert warnings["warnings"]


def test_unknown_resource_utilization_does_not_masquerade_as_normal():
    agent = make_agent(percent=float("nan"))
    agent.shared_memory.data["latest_metrics"] = {"accuracy": 0.5}
    status = agent.get_overall_system_health()
    assert status["status"] == "Unknown"
    assert status["shared_memory"]["status"] == "Unknown"


def test_run_stage_isolation_preserves_diagnostics_without_falsifying_success():
    agent = make_agent()
    def explode():
        raise ValueError("broken test dependency")
    result = agent._run_stage("behavioral", explode)
    assert result["status"] == "error"
    assert result["error_type"] == "ValueError"
    assert "broken test dependency" in result["error"]


def test_static_analysis_path_selection_respects_trusted_allowlist(tmp_path):
    agent = make_agent()
    target = tmp_path / "codebase"
    target.mkdir()
    with pytest.raises(PermissionError):
        agent._resolve_static_analysis_root({"codebase_path": str(target)})
    agent.agent_config["static_analysis_roots"] = [str(tmp_path)]
    assert agent._resolve_static_analysis_root({"codebase_path": str(target)}) == target.resolve()
    outside = tmp_path.parent / "outside"
    # No need to create any files in an untrusted path: a different parent is
    # rejected before static analysis is attempted.
    with pytest.raises((PermissionError, FileNotFoundError)):
        agent._resolve_static_analysis_root({"codebase_path": str(outside)})


def test_partial_static_analysis_is_reported_as_critical():
    agent = make_agent()
    report = agent._evaluate_validation_health({
        "static_analysis": {
            "status": "pass", "parse_failures": [{"file": "broken.py", "error": "SyntaxError"}],
            "security_metrics": {"critical_count": 0},
        }
    })
    assert report["component_status"]["static_analysis"] == "critical"


def test_fallback_sut_cannot_fabricate_a_passing_behavioral_result():
    from src.agents.evaluation_agent import FallbackEvaluatorAgent
    sut = FallbackEvaluatorAgent.__new__(FallbackEvaluatorAgent)
    result = sut.predict("test_input")
    assert result["status"] == "unavailable"
    assert result["execution_success"] is False
    assert result["confidence"] == 0.0
    assert result["prediction"] is None


def test_internal_certification_gates_require_real_evidence():
    from src.agents.evaluation_agent import AIValidationSuite
    protocol = SimpleNamespace(static_analysis={"security": {"max_critical": 0}})
    suite = AIValidationSuite(protocol)
    assert suite._generate_compliance_matrix() == {
        "static_analysis_internal_gate": False,
        "behavioral_internal_gate": False,
        "safety_case_present": False,
    }
    suite.artifacts["static_report"] = {
        "summary": {"files_analyzed": 2}, "security_metrics": {"critical_count": 0},
        "parse_failures": [],
    }
    suite.artifacts["behavioral_results"] = {
        "overall_status": "PASSED",
        "traceability_matrix": [{"requirement_id": "REQ1", "status": "PASSED"}],
    }
    gates = suite._generate_compliance_matrix()
    assert gates["static_analysis_internal_gate"] is True
    assert gates["behavioral_internal_gate"] is True
    suite.artifacts["static_report"]["parse_failures"] = [{"file": "x.py"}]
    assert suite._generate_compliance_matrix()["static_analysis_internal_gate"] is False
