"""General-purpose, non-application-specific parametric analysis regressions."""

import pytest
from time import perf_counter

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


def test_interval_diagnostics_and_attested_integer_equivalence():
    result = SoftwareAnalysis.parametric({
        "code": "a = x / divisor\nb = math.sqrt(x - 2)\nc = w + h\nd = h + w",
        "ranges": {"x": [0, 1], "divisor": [-1, 1]},
        "integer_parameters": ["w", "h"],
    })
    rules = {issue["rule_id"] for issue in result["diagnostics"]}
    assert {"SWP010", "SWP012"}.issubset(rules)
    assert result["equivalence_proposals"][0]["expressions"] == ["h + w", "w + h"]


def test_integer_polynomial_equivalence_has_a_bounded_conditional_proof():
    report = SoftwareAnalysis.parametric({
        "code": "a = (x + y) * z\nb = z * x + z * y\nc = x / y\nd = y / x",
        "integer_parameters": ["x", "y", "z"],
    })
    assert len(report["equivalence_proposals"]) == 1
    proposal = report["equivalence_proposals"][0]
    assert proposal["proof_scope"] == "algebraic_equivalence_under_stated_assumptions"
    assert proposal["expressions"] == ["(x + y) * z", "z * x + z * y"]
    assert "built-in Python int" in proposal["assumptions"]

    unsupported = SoftwareAnalysis.parametric({"code": "a = x * y\nb = y * x"})
    assert unsupported["equivalence_proposals"] == []


def test_executable_test_generation_preserves_caller_oracles():
    result = SoftwareAnalysis.parametric({"test_samples": [
        {"module": "math", "function": "sqrt", "args": [16], "expected": 4.0},
        {"module": "math", "function": "sqrt", "args": [-1], "raises": "ValueError"},
    ]})
    generated = result["executable_test_proposal"]
    assert generated["tested"] is False
    compile(generated["code"], "<proposed-tests>", "exec")
    namespace = {}
    exec(compile(generated["code"], "<proposed-tests>", "exec"), namespace)
    namespace["test_generated_parametric_0"]()
    namespace["test_generated_parametric_1"]()
    assert "pytest.raises(ValueError)" in generated["code"]
    with pytest.raises(ValueError):
        SoftwareAnalysis.parametric({"test_samples": [{"module": "math;unsafe", "function": "sqrt", "expected": 1}]})


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


def test_optimization_feasibility_interface_and_parameter_search():
    spec = {"variables": {"x": {"type": "real", "lower": 0, "upper": 10}},
            "objective": {"linear": {"x": 1}},
            "constraints": [{"linear": {"x": 1}, "sense": ">=", "rhs": 3}]}
    problem = OptimizationSolver.parameter_problem(spec)
    result = OptimizationSolver.inspect_candidate(problem, {"x": 2}, tolerance=0.01)
    assert result["feasible"] is False
    assert result["constraints"]["constraint_0"] == pytest.approx(1.0)
    assert result["optimality"] == "not_evaluated"
    solution = OptimizationSolver().search_parameters(spec).solution
    assert solution.success
    assert solution.primal_solution["x"] == pytest.approx(3.0)


def test_checked_unit_aware_calculation_contract():
    from src.agents.stem.utils.stem_errors import STEMUnitError, STEMValidationError
    math_service = ParametricMath()
    report = math_service.evaluate_checked("2*width", {"width": {"value": 3, "unit": "m"}},
                                           output_unit="cm", allowed_range=(600, 600),
                                           assumptions=("Input width is measured",))
    assert report["result"]["magnitude"] == pytest.approx(600)
    assert report["engineering_code_compliance"] == "not_assessed"
    with pytest.raises(STEMUnitError):
        math_service.evaluate_checked("2*width", {"width": {"value": 3, "unit": "m"}}, output_unit="s")
    with pytest.raises(STEMValidationError):
        math_service.evaluate_checked("2*width", {"width": 3}, allowed_range=(7, 10))


def test_spatial_constraints_and_revision_aware_references():
    from src.agents.spatial.spatial_memory import SpatialMemory
    from src.agents.spatial.spatial_types import SpatialEntity
    points = {"a": [0, 0], "b": [1, 0], "c": [0, 1], "d": [1, 1]}
    outcome = Topology.validate_constraints(points, [
        {"kind": "distance", "entities": ["a", "b"], "target": 1},
        {"kind": "parallel", "entities": ["a", "b", "c", "d"]},
        {"kind": "perpendicular", "entities": ["a", "b", "a", "c"]},
        {"kind": "minimum_clearance", "entities": ["a", "b"], "target": 2},
        {"kind": "coincident", "entities": ["a", "missing"]},
    ])
    assert [x["status"] for x in outcome["constraints"]] == ["satisfied", "satisfied", "satisfied", "violated", "unresolved"]
    memory = SpatialMemory()
    entity = SpatialEntity("a", (0, 0), revision=4)
    memory.upsert_entity(entity)
    assert memory.resolve_reference("a", revision=4)["status"] == "resolved"
    assert memory.resolve_reference("a", revision=2)["status"] == "stale"
    assert memory.resolve_reference("unknown")["status"] == "missing"


def test_spatial_conflict_diagnostics_distinguish_proven_from_unresolved():
    points = {"start": [0, 0], "end": [1, 0]}
    conflicting = Topology.validate_constraints(points, [
        {"kind": "distance", "entities": ["start", "end"], "target": 1.0},
        {"kind": "distance", "entities": ["end", "start"], "target": 2.0},
        {"kind": "minimum_clearance", "entities": ["start", "end"], "target": 3.0},
    ])
    assert conflicting["consistency"] == "inconsistent"
    assert len(conflicting["conflicts"]) == 3
    assert not conflicting["satisfied"]
    independent = Topology.validate_constraints(points, [
        {"kind": "distance", "entities": ["start", "end"], "target": 1.0},
    ])
    assert independent["consistency"] == "not_proven"


def test_existing_agent_facades_accept_generic_parametric_workflow():
    from src.agents.stem_agent import STEMAgent
    from src.agents.spatial_agent import SpatialAgent
    from src.agents.optimization_agent import OptimizationAgent

    source = SoftwareAgent(None, None).perform_task({"operation": "parametric_analysis", "parameters": {
        "code": "area = width * height", "graph": {"width": [], "height": [], "area": ["width", "height"]},
        "changed": ["width"],
    }})
    assert source["result"]["graph"]["affected"] == ["area"]

    checked = STEMAgent(None, None).perform_task({"domain": "parametric", "operation": "evaluate_checked",
        "args": ["width*height", {"width": {"value": 2, "unit": "m"}, "height": {"value": 3, "unit": "m"}}],
        "kwargs": {"allowed_range": [6, 6]}, "cache": False, "publish": False})
    assert checked["status"] == "success"
    assert checked["result"]["result"]["magnitude"] == pytest.approx(6)

    geometry = SpatialAgent(None, None).perform_task({"operation": "validate_constraints",
        "entities": {"left": [0, 0], "right": [2, 0]},
        "constraints": [{"kind": "distance", "entities": ["left", "right"], "target": 2}]})
    assert geometry["result"]["satisfied"] is True

    spec = {"variables": {"x": {"lower": 0, "upper": 10}}, "objective": {"linear": {"x": 1}},
            "constraints": [{"linear": {"x": 1}, "sense": ">=", "rhs": 3}]}
    searched = OptimizationAgent(None, None).perform_task({"problem": spec, "publish": False})
    assert searched["status"] == "success"


def test_large_caller_owned_graph_is_bounded_and_incremental():
    graph = {"n0": []}
    for index in range(1, 2001):
        graph[f"n{index}"] = [f"n{index-1}"]
    started = perf_counter()
    report = SoftwareAnalysis.parametric({"graph": graph, "changed": ["n0"]})
    elapsed = perf_counter() - started
    assert report["graph"]["nodes"] == 2001
    assert len(report["graph"]["affected"]) == 2000
    assert not report["graph"]["cycles"]
    assert elapsed < 10.0  # broad regression budget, not a machine-specific speed promise


def test_shared_memory_preserves_generic_agent_handoff_context():
    from uuid import uuid4
    from src.agents.collaborative.shared_memory import SharedMemory
    memory = SharedMemory()
    key = f"parametric.integration.{uuid4().hex}"
    try:
        memory.put(key, {"task_id": "generic-parameter-analysis", "context_revision": 3,
                         "result_ref": "artifact:sample", "owner": "software"})
        context = memory.get(key)
        assert context["context_revision"] == 3
        software = SoftwareAgent(memory, None)
        answer = software.perform_task({"operation": "parametric_analysis", "parameters": {
            "code": "result = width * length",
            "graph": {"width": [], "length": [], "result": ["width", "length"]}, "changed": ["width"]}})
        assert answer["execution_success"]
        # The recipient receives only the explicit context reference; no app graph is mutated.
        memory.put(key, {**context, "owner": "verification", "analysis_request_id": answer["request_id"]})
        assert memory.get(key)["analysis_request_id"] == answer["request_id"]
    finally:
        memory.delete(key)
