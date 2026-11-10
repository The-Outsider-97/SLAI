"""Regression tests for SLAI Evaluation Agent and cross-evaluator correctness.

Tests use real production methods with isolated dependencies; no generated test
code is executed, no database connection is opened and no external network used.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.agents.evaluators.performance_budget_evaluator import (
    BudgetContract, PerformanceBudgetEvaluator,
)
from src.agents.evaluators.statistical_evaluator import StatisticalEvaluator
from src.agents.evaluators.modules.evaluators_calculations import PerformanceMetrics
from src.agents.evaluators.documentation.documentation import AuditBlock, AuditTrail
from src.agents.evaluators.documentation.certification_framework import (
    CertificationManager, CertificationLevel, CertificationRequirement,
)
from src.agents.evaluators.data.issue_db import FallbackIssueTracker
from src.agents.evaluators.behavioral_validator import BehavioralValidator
from src.agents.evaluators.safety_evaluator import SafetyEvaluator
from src.agents.evaluators.resource_utilization_evaluator import ResourceUtilizationEvaluator
from src.agents.evaluators.autonomous_evaluator import AutonomousEvaluator
from src.agents.evaluation_agent import EvaluationAgent


def budget():
    target = PerformanceBudgetEvaluator.__new__(PerformanceBudgetEvaluator)
    target.enabled = True
    target.strict_mode = True
    target.store_results = False
    target.config_path = "<test>"
    target.contracts = {
        "agent": BudgetContract(
            agent="agent", max_latency_ms=200.0, max_memory_mb=512.0,
            soft_latency_ms=150.0, soft_memory_mb=400.0,
        )
    }
    return target


def test_budget_does_not_pass_missing_nonfinite_or_negative_observations():
    evaluator = budget()
    for observed in ({}, {"latency_ms": float("nan"), "memory_mb": 20},
                     {"latency_ms": -1, "memory_mb": 20}):
        result = evaluator.evaluate({"agent": observed})
        assert result["status"] == "incomplete"
        assert result["summary"]["passed"] is False
        assert result["summary"]["missing_evidence"] > 0
    assert evaluator.evaluate({"agent": {"latency_ms": 190, "memory_mb": 30}})["status"] == "pass"
    failed = evaluator.evaluate({"agent": {"latency_ms": 210, "memory_mb": 30}})
    assert failed["status"] == "fail" and failed["summary"]["violations"] == 1


def test_budget_invalid_configuration_rejected_without_false_defaults():
    evaluator = budget()
    with pytest.raises(ValueError):
        evaluator._load_contracts([{"agent": "a", "max_latency_ms": float("nan")}])
    with pytest.raises(ValueError):
        evaluator._load_contracts([{"agent": "a"}, {"agent": "a"}])
    with pytest.raises(ValueError):
        evaluator._load_contracts([{"agent": "a", "soft_latency_ms": 2000}])


def test_calibration_uses_predicted_class_confidence_not_positive_probability():
    assert PerformanceMetrics.calibration_error([0, 1], [0.1, 0.9], bins=5) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        PerformanceMetrics.calibration_error([0], [float("nan")])
    with pytest.raises(ValueError):
        PerformanceMetrics.calibration_error([0], [-0.01])
    with pytest.raises(ValueError):
        PerformanceMetrics.calibration_error([0], [0.1], bins=0)


def test_paired_statistics_use_actual_pairwise_difference():
    evaluator = StatisticalEvaluator.__new__(StatisticalEvaluator)
    evaluator.significance_threshold = 0.05
    evaluator.calculations = SimpleNamespace(effect_size=lambda a, b: 0.0)
    paired = evaluator.statistical_analysis(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [0.9, 1.7, 2.8, 3.5, 4.6],
    )
    assert paired["paired_test"]["test"] == "paired_student_t"
    assert paired["paired_test"]["degrees_of_freedom"] == 4
    assert paired["paired_test"]["p_value"] is not None
    assert paired["paired_effect_size_dz"] is not None
    degenerate = evaluator.statistical_analysis([1, 2, 3], [0, 1, 2])
    assert degenerate["paired_test"]["significant"] is None
    with pytest.raises(Exception):
        evaluator.statistical_analysis([1, 2, 3], [1, float("inf"), 3])


def test_statistical_anova_reports_f_and_valid_p_value():
    evaluator = StatisticalEvaluator.__new__(StatisticalEvaluator)
    evaluator.significance_threshold = 0.05
    result = evaluator.anova([[1, 2, 3, 4], [4, 5, 6, 7]])
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["degrees_of_freedom"] == {"between": 1, "within": 6}
    assert result["status"] == "ok"
    assert evaluator.anova([[1, 1, 1], [2, 2, 2]])["p_value"] is None


def audit_trail():
    audit = AuditTrail.__new__(AuditTrail)
    audit.doc_config = {"system": "SLAI"}
    audit.hash_algorithm_name = "sha256"
    audit.difficulty = 4
    audit.chain = [audit._create_genesis_block()]
    return audit


def test_audit_default_unmined_entry_verifies_and_tampering_fails():
    ledger = audit_trail()
    ledger.add_entry({"result": "reviewed"}, mine=False)
    assert ledger.verify_chain() is True
    good = [block.to_dict() for block in ledger.chain]
    damaged = json.loads(json.dumps(good))
    damaged[1]["data"]["document"]["result"] = "tampered"
    with pytest.raises(Exception):
        ledger.load_chain(damaged)
    assert ledger.verify_chain() is True
    assert [block.to_dict() for block in ledger.chain] == good


def certification():
    manager = CertificationManager.__new__(CertificationManager)
    manager.system = "SLAI"
    manager.domain = "development"
    manager.current_level = CertificationLevel.DEVELOPMENT
    manager.requirements = {
        CertificationLevel.DEVELOPMENT: [
            CertificationRequirement(
                description="Independent quality gate",
                test_method="pytest",
                passing_condition="validated test pass",
                evidence_required=["test_report"],
                requirement_id="REQ-TEST-1",
            )
        ]
    }
    manager.evidence_registry = []
    return manager


def test_certificate_requires_requirement_specific_verified_evidence():
    manager = certification()
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "test_report",
        "content": {"passed": True},
        "source": "pytest",
        "metadata": {},
    }
    unverified = manager.submit_evidence(payload)
    assert manager.evaluate_certification_detailed().passed is False
    verified = dict(payload)
    verified["metadata"] = {
        "requirement_ids": ["REQ-TEST-1"],
        "passed": True,
        "verification_status": "verified",
    }
    record = manager.submit_evidence(verified)
    assert manager.evaluate_certification_detailed().passed is True
    record.content["passed"] = False
    assert manager.evaluate_certification_detailed().passed is False
    assert manager.generate_certificate()["independent_certification_granted"] is False


def test_fallback_issue_tracker_is_bounded_and_returns_copies():
    tracker = FallbackIssueTracker(max_entries=2)
    issue = {"id": "1", "context": {"nested": [1]}}
    for num in ("1", "2", "3"):
        issue["id"] = num
        tracker.log_issue(issue)
    result = tracker.get_issues()
    assert [item["id"] for item in result] == ["2", "3"]
    assert all("+00:00" in item["timestamp"] for item in result)
    result[0]["context"]["nested"].append(2)
    assert tracker.get_issues()[0]["context"]["nested"] == [1]
    tracker.close()
    tracker.close()


def test_behavioral_anomalies_count_as_errors_not_full_coverage():
    validator = BehavioralValidator.__new__(BehavioralValidator)
    validator.requirement_coverage = set()
    validator.failure_modes = []
    validator.thresholds = {"pass_rate": 1.0}
    validator.weights = {}
    validator._calculate_composite_score = lambda **kw: 0.0
    validator._build_statistical_summary = lambda *a: {}
    validator._evaluate_thresholds = lambda **kw: []
    validator._build_history_summary = lambda: {}
    now = datetime.now(timezone.utc)
    report = validator._build_results_payload(
        suite_name="smoke", execution_records=[],
        anomalies=[{"test_id": "broken", "error": "timeout"}],
        started_at=now, completed_at=now, source="test",
    )
    assert report["status"] == "fail"
    assert report["summary"]["total_cases"] == 1
    assert report["summary"]["errored"] == 1
    assert report["summary"]["test_coverage"] == 0.0
    assert report["thresholds"]["passed"] is False


def test_safety_requires_complete_explicit_risk_data():
    evaluator = SafetyEvaluator.__new__(SafetyEvaluator)
    evaluator.risk_categories = ["collision", "electrical"]
    evaluator.safety_standards = ["standard"]
    evaluator.raw_incidents = []
    minimal = {
        "risk_assessment": {"collision": 0.1}, "hazard_detection_time": 0.1,
        "emergency_stop_time": 0.1, "safety_margin": 1.0,
        "collision_avoided": True, "standard": True,
    }
    with pytest.raises(Exception):
        evaluator._normalize_incident(minimal)
    complete = dict(minimal)
    complete["risk_assessment"] = {"collision": 0.1, "electrical": 0.0}
    complete["collision_avoided"] = "False"
    with pytest.raises(Exception):
        evaluator._normalize_incident(complete)


def test_resource_network_counter_reset_is_not_a_valid_zero_observation():
    evaluator = ResourceUtilizationEvaluator.__new__(ResourceUtilizationEvaluator)
    first = SimpleNamespace(bytes_sent=1000, bytes_recv=500)
    second = SimpleNamespace(bytes_sent=900, bytes_recv=550)
    with pytest.raises(Exception):
        evaluator._calculate_network_mbps(first, second, 2.0)
    valid = SimpleNamespace(bytes_sent=2000, bytes_recv=1500)
    assert evaluator._calculate_network_mbps(first, valid, 2.0) == pytest.approx(0.008)


def test_explicit_autonomous_success_must_be_boolean():
    evaluator = AutonomousEvaluator.__new__(AutonomousEvaluator)
    with pytest.raises(Exception):
        evaluator._resolve_success({"success": "false"}, 0.0)
    assert evaluator._resolve_success({"success": False}, 0.0) is False


def test_evaluation_agent_redacts_sensitive_fields():
    agent = EvaluationAgent.__new__(EvaluationAgent)
    redacted = agent._sanitize_results({
        "status": "normal", "api_key": "secret-value",
        "nested": {"password": "top-secret"},
        "notes": "private note",
    })
    assert redacted["status"] == "normal"
    assert "secret-value" not in json.dumps(redacted)
    assert "top-secret" not in json.dumps(redacted)
    assert redacted["notes"] == "[REDACTED]"


def test_evaluation_memory_integrity_and_atomic_checkpoint(tmp_path, monkeypatch):
    from src.agents.evaluators import evaluators_memory as memory_module
    monkeypatch.setattr(memory_module, "load_global_config", lambda: {"__config_path__": str(tmp_path / "config.yml")})
    monkeypatch.setattr(memory_module, "get_config_section", lambda key: {
        "checkpoint_dir": str(tmp_path / "checkpoints"), "max_size": 5,
        "checkpoint_freq": 100, "auto_save": False,
    })
    store = memory_module.EvaluatorsMemory()
    original = {"nested": {"value": 1}}
    key = store.add(original)
    original["nested"]["value"] = 100
    assert store.get(key)["data"]["nested"]["value"] == 1
    retrieved = store.get(key)
    retrieved["data"]["nested"]["value"] = 999
    assert store.get(key)["data"]["nested"]["value"] == 1
    with pytest.raises(Exception):
        store.create_checkpoint("../escaped")
    path = store.create_checkpoint("valid")
    state = json.loads(open(path, encoding="utf-8").read())
    state["store"][0]["data"]["nested"]["value"] = 777
    with open(path, "w", encoding="utf-8") as output:
        json.dump(state, output)
    with pytest.raises(Exception):
        store.load_checkpoint(path)
    assert store.get(key)["data"]["nested"]["value"] == 1


def test_performance_prediction_mapping_alignment_and_nonfinite_inputs():
    from src.agents.evaluators.performance_evaluator import PerformanceEvaluator
    evaluator = PerformanceEvaluator.__new__(PerformanceEvaluator)
    evaluator.threshold = 0.5
    evaluator.classes = [0, 1]
    labels, scores = evaluator._extract_predictions_from_mappings([
        {"label": 1, "probabilities": [0.2, 0.8]},
        {"probabilities": [0.9, 0.1]},
    ], expected_len=2)
    assert labels == [1, 0]
    assert scores == [[0.2, 0.8], [0.9, 0.1]]
    with pytest.raises(Exception):
        evaluator._extract_predictions_from_mappings([
            {"probabilities": [0.2, 0.8]},
            {"label": 1},
        ], expected_len=2)
    for invalid in ([0.2, float("nan")], [[0.1, 0.9], [float("inf"), 0.2]], [[0.9], [0.5, 0.5]]):
        with pytest.raises(Exception):
            evaluator._coerce_probabilities(invalid, expected_len=2)


def test_infrastructure_refuses_fake_tuning_or_implicit_hazard_updates():
    from src.agents.evaluators.base_infra import InfrastructureManager
    manager = InfrastructureManager.__new__(InfrastructureManager)
    manager._evaluation_function = None
    manager.tuner = None
    manager.config = {"initial_hazard_rates": {"system_failure": 0.000001}}
    with pytest.raises(RuntimeError):
        manager._agent_evaluator({"param": 42})
    assert manager.automated_tuning_cycle()["status"] == "disabled"
    manager._evaluation_function = lambda params: 0.73
    assert manager._agent_evaluator({}) == pytest.approx(0.73)
    assert manager._update_risk_profile({"best_score": 0.82}) == pytest.approx(0.82)
    assert manager.config["initial_hazard_rates"] == {"system_failure": 0.000001}


def test_adaptive_risk_rejects_boolean_hazard_counts_and_exports_atomic(tmp_path):
    from src.agents.evaluators.adaptive_risk import RiskObservation, RiskAdaptation, HazardBayesianState
    with pytest.raises(Exception):
        RiskObservation(
            timestamp=datetime.now(timezone.utc).isoformat(),
            hazard_counts={"collision": True}, operational_time=10.0,
        )
    state = RiskAdaptation.__new__(RiskAdaptation)
    import threading
    from collections import OrderedDict
    state._lock = threading.RLock()
    state._state_directory = tmp_path
    state.learning_rate = 0.05
    state.uncertainty_window = 1000.0
    state.hazard_states = {
        "collision": HazardBayesianState(
            hazard_id="collision", prior_rate=0.01,
            base_alpha=1.0, base_beta=100.0,
            alpha=2.0, beta=120.0,
        )
    }
    state.observation_history = []
    state.safety_case_versions = OrderedDict()
    target = tmp_path / "risk.json"
    assert state.export_state(str(target)) == str(target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["hazard_states"]["collision"]["alpha"] == 2.0
    assert not list(tmp_path.glob("*.tmp"))


def test_efficiency_does_not_treat_evaluator_overhead_as_generation_energy():
    from src.agents.evaluators.efficiency_evaluator import EfficiencyEvaluator
    evaluator = EfficiencyEvaluator.__new__(EfficiencyEvaluator)
    evaluator.baselines = {"energy_usage": 120.0}
    # Five seconds spent evaluating output is NOT five seconds of production.
    assert evaluator._estimate_energy_usage([{"text": "output"}], runtime_seconds=5.0) == 0.0
    # Measured generation duration supports a transparently modeled estimate.
    assert evaluator._estimate_energy_usage(
        [{"text": "output", "generation_time": 2.0}], runtime_seconds=5.0
    ) == pytest.approx(240.0)
    # Direct observed energy takes precedence.
    assert evaluator._estimate_energy_usage(
        [{"energy_joules": 7.5}], runtime_seconds=10.0
    ) == pytest.approx(7.5)
    with pytest.raises(Exception):
        evaluator._estimate_energy_usage(
            [{"generation_time": float("nan")}], runtime_seconds=2.0
        )


def test_validation_protocol_rejects_nonfinite_and_boolean_thresholds():
    from src.agents.evaluators.modules.validation_protocol import ValidationProtocol
    protocol = ValidationProtocol.__new__(ValidationProtocol)
    for bad in (float("nan"), float("inf"), -float("inf"), True, False):
        assert protocol._is_number(bad) is False
    assert protocol._is_number(0.75) is True


def test_lazy_evaluator_module_exports_preserved():
    import src.agents.evaluators.modules as exports
    from src.agents.evaluators.modules.evaluators_calculations import EvaluatorsCalculations
    from src.agents.evaluators.modules.validation_protocol import ValidationProtocol
    assert exports.EvaluatorsCalculations is EvaluatorsCalculations
    assert exports.ValidationProtocol is ValidationProtocol
    assert "EvaluationTransformer" in exports.__all__
    assert callable(exports.__getattr__)


def test_autonomous_undefined_path_deviation_remains_json_serializable():
    from src.agents.evaluators.autonomous_evaluator import TaskMetrics
    evaluator = AutonomousEvaluator.__new__(AutonomousEvaluator)
    evaluator.thresholds = {
        "max_path_deviation": 1.2, "max_energy_per_task": 100.0,
        "max_collisions": 0,
    }
    metric = TaskMetrics(
        task_id="case", task_type="navigation",
        completion_time=1.0, path_length=5.0,
        optimal_path_length=0.0, energy_consumed=10.0,
        success=False, collisions=0, deviation_from_optimal=float("inf"),
        path_efficiency=0.0, energy_efficiency=0.5,
        waypoint_count=2, graph_available=False,
    )
    serialized = metric.to_dict()
    assert serialized["deviation_from_optimal"] is None
    json.dumps(serialized, allow_nan=False)
    violations = evaluator._evaluate_task_thresholds(metric)
    assert violations[0]["value"] is None
    json.dumps(violations, allow_nan=False)
