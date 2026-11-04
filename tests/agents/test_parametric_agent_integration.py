"""General-purpose, non-application-specific parametric analysis regressions."""

import pytest

from src.agents.software.software_analysis import SoftwareAnalysis
from src.agents.software_agent import SoftwareAgent
from src.agents.stem.parametric import ParametricMath
from src.agents.spatial.world.geometry import Polygon
from src.agents.spatial.world.topology import Topology
from src.agents.optimization.optimization_solver import OptimizationSolver


def test_software_parametric_analysis_is_read_only_and_impact_is_transitive():
    graph = {"width": [], "area": ["width"], "volume": ["area"], "unused": []}
    untouched = {name: list(deps) for name, deps in graph.items()}
    result = SoftwareAnalysis.parametric({"code": "a = width * height\nb = width * height", "graph": graph, "changed": ["width"]})
    assert result["repeated_expressions"] == ["width * height"]
    assert result["helper_proposals"][0]["proposed_code"] == "def compute_shared_1(height, width):\n    return width * height\n"
    assert result["graph"]["affected"] == ["area", "volume"]
    assert graph == untouched
    assert result["test_proposals"]


def test_software_parametric_diagnostics_detect_cycles_and_division():
    report = SoftwareAnalysis.parametric({"code": "x = value / 0", "graph": {"a": ["b"], "b": ["a"]}})
    assert {entry["rule_id"] for entry in report["diagnostics"]} == {"SWP002", "SWP004"}
    assert report["graph"]["cycles"]


def test_software_agent_routes_parametric_operation():
    agent = SoftwareAgent(None, None)
    outcome = agent.perform_task({"operation": "parametric_analysis", "parameters": {"code": "x = 1 / 0"}})
    assert outcome["execution_success"] is True
    assert outcome["result"]["diagnostics"][0]["rule_id"] == "SWP002"


def test_stem_expression_inspection_and_units():
    stem = ParametricMath()
    report = stem.inspect_expression("width * height", {"width": {"value": 2, "unit": "m"}, "height": {"value": 3, "unit": "m"}})
    assert report["dependencies"] == ["height", "width"]
    assert report["result"]["magnitude"] == pytest.approx(6)
    assert stem.inspect_expression("sin(pi/2)")["dependencies"] == []


def test_spatial_topology_diagnostics():
    square = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    assert Topology.validate_polygon(square)["valid"]
    crossed = Polygon([[0, 0], [1, 1], [0, 1], [1, 0]])
    result = Topology.validate_polygon(crossed)
    assert not result["valid"]
    assert any(item["kind"] == "self_intersection" for item in result["findings"])


def test_optimization_feasibility_interface_reuses_existing_checker(monkeypatch):
    from src.agents.optimization import optimization_math
    seen = []
    def evidence(problem, point, *, tolerance):
        seen.append((problem, point, tolerance))
        return 0.25, False
    monkeypatch.setattr(optimization_math, "feasibility_evidence", evidence)
    token = object()
    result = OptimizationSolver.inspect_candidate(token, {"x": 2}, tolerance=0.01)
    assert seen == [(token, {"x": 2}, 0.01)]
    assert result["feasible"] is False
    assert result["optimality"] == "not_evaluated"
