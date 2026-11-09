"""Typed parameters, the extended formula language, the dependency graph and incremental evaluation (SPADRA P0-01/02/06)."""
import math
import random

import pytest

from src.utils.spatial_formats.common.errors import ValidationError, ResourceLimitError
from src.utils.spatial_formats.common.units import to_si, from_si, units_for
from src.utils.spatial_formats.sgeo.parameters import (
    ParameterEvaluator, ParameterGraph, Quantity, check_formula, dimension_name, evaluate_expression,
    evaluate_parameters, expression_dependencies, function_catalog,
)

L = (1, 0, 0, 0, 0)


def ev(expr, **variables):
    return evaluate_expression(expr, {k: (v if isinstance(v, Quantity) else Quantity(v)) for k, v in variables.items()})


# --------------------------------------------------------------------------------------------- the report's worked formulas
def test_door_clear_width_and_conditional_frame():
    specs = {"width": {"type": "length", "default": 0.9}, "frame_thickness": {"type": "length", "default": 0.035},
             "clear_width": {"type": "length", "expression": "width - 2 * frame_thickness"},
             "frame_depth": {"type": "length", "expression": "if(width > length(1.2), length(0.08), length(0.06))"}}
    v = evaluate_parameters(specs)
    assert v["clear_width"] == pytest.approx(0.83) and v["frame_depth"] == 0.06
    assert evaluate_parameters(specs, {"width": 1.5})["frame_depth"] == 0.08


def test_roof_rise_adaptive_panels_and_visibility():
    specs = {"horizontal_run": {"type": "length", "default": 4.0}, "roof_pitch": {"type": "angle", "default": math.radians(30)},
             "roof_rise": {"type": "length", "expression": "horizontal_run * sin(roof_pitch) / cos(roof_pitch)"},
             "facade_width": {"type": "length", "default": 12.0}, "joint_width": {"type": "length", "default": 0.02},
             "panel_count": {"type": "integer", "default": 8},
             "panel_width": {"type": "length", "expression": "(facade_width - (panel_count - 1) * joint_width) / panel_count"},
             "show_intermediate_support": {"type": "boolean", "expression": "panel_count > 2"}}
    v = evaluate_parameters(specs)
    assert v["roof_rise"] == pytest.approx(4.0 * math.tan(math.radians(30)))
    assert v["panel_width"] == pytest.approx(1.4825)               # the report's 8-panel example (equation 8)
    assert v["show_intermediate_support"] is True
    assert evaluate_parameters(specs, {"panel_count": 2})["show_intermediate_support"] is False


def test_second_moment_of_area_is_dimensionally_checked():
    specs = {"b": {"type": "length", "default": 0.2}, "h": {"type": "length", "default": 0.4},
             "I": {"type": "second_moment", "expression": "b * h ** 3 / 12"},
             "A": {"type": "area", "expression": "b * h"}}
    v = evaluate_parameters(specs)
    assert v["I"] == pytest.approx(0.2 * 0.4 ** 3 / 12) and v["A"] == pytest.approx(0.08)
    bad = dict(specs, I={"type": "area", "expression": "b * h ** 3 / 12"})
    with pytest.raises(ValidationError, match="dimension mismatch") as exc:
        evaluate_parameters(bad)
    assert exc.value.context["parameter"] == "I" and exc.value.context["expected"] == "area" and exc.value.context["actual"] == "second_moment"


# ----------------------------------------------------------------------------------------------------------- new functions
@pytest.mark.parametrize("expr,expected", [
    ("tan(angle(0.0))", 0.0), ("atan(1)", math.pi / 4), ("asin(1)", math.pi / 2), ("acos(1)", 0.0),
    ("exp(0)", 1.0), ("ln(1)", 0.0), ("log10(1000)", 3.0), ("round(2.5)", 3.0), ("round(-2.5)", -3.0),
    ("roundup(2.1)", 3.0), ("rounddown(2.9)", 2.0), ("ceil(2.0)", 2.0), ("floor(-0.5)", -1.0),
    ("mod(7, 3)", 1.0), ("mod(-1, 3)", 2.0), ("7 // 2", 3.0), ("7 % 4", 3.0), ("pi", math.pi), ("2 * pi", math.tau),
    ("round(2.4, 0.5)", 2.5), ("roundup(2.1, 0.5)", 2.5),
])
def test_new_functions(expr, expected):
    assert ev(expr).value == pytest.approx(expected)


def test_ceil_floor_use_a_tolerance_so_exact_quotients_survive():
    assert ev("floor(0.3 / 0.1)").value == 3.0          # 0.3 / 0.1 == 2.9999999999999996 in binary floating point
    assert ev("ceil(0.6 / 0.2)").value == 3.0           # 0.6 / 0.2 == 3.0000000000000004
    assert ev("floor(2.5)").value == 2.0 and ev("ceil(2.5)").value == 3.0


def test_rounding_keeps_units_and_steps_must_match():
    q = ev("round(w, step)", w=Quantity(1.234, L), step=Quantity(0.05, L))
    assert q.dimension == L and q.value == pytest.approx(1.25)
    with pytest.raises(ValidationError, match="incompatible"):
        ev("round(w, 0.5)", w=Quantity(1.0, L))
    with pytest.raises(ValidationError, match="step"):
        ev("round(1, 0)")


@pytest.mark.parametrize("expr", ["asin(2)", "acos(-1.5)", "ln(0)", "ln(-1)", "log10(0)", "exp(100)", "mod(1, 0)", "1 // 0", "atan2(0, 0)",
                                  "tan(angle(1.5707963267948966))", "sqrt(-4)", "1 / 0", "0 ** -1", "round()", "mod(1)"])
def test_domain_errors_are_rejected_not_defaulted(expr):
    with pytest.raises(ValidationError):
        ev(expr)


def test_trig_and_log_need_the_right_dimensions():
    with pytest.raises(ValidationError):
        ev("sin(w)", w=Quantity(1.0, L))
    with pytest.raises(ValidationError):
        ev("ln(w)", w=Quantity(2.0, L))
    assert ev("atan2(y, x)", y=Quantity(1.0, L), x=Quantity(1.0, L)).dimension == (0, 1, 0, 0, 0)
    assert dimension_name(ev("asin(0.5)").dimension) == "angle"


def test_boolean_equality_and_logic():
    assert ev("a == b", a=Quantity(True), b=Quantity(True)).value is True
    assert ev("a != b", a=Quantity(True), b=Quantity(False)).value is True
    with pytest.raises(ValidationError):
        ev("a < b", a=Quantity(True), b=Quantity(False))
    with pytest.raises(ValidationError):
        ev("a == 1", a=Quantity(True))


def test_constants_are_not_dependencies_but_a_parameter_may_shadow_them():
    assert expression_dependencies("2 * pi * r") == {"r"}
    specs = {"pi": {"type": "number", "default": 3.0}, "x": {"type": "number", "expression": "pi * 2"}}
    assert evaluate_parameters(specs)["x"] == 6.0
    assert ParameterGraph(specs).order == ("pi", "x")


def test_a_parameter_may_share_a_name_with_a_function():
    specs = {"area": {"type": "area", "default": 2.0}, "length": {"type": "length", "default": 3.0},
             "double": {"type": "area", "expression": "area * 2"}, "side": {"type": "length", "expression": "sqrt(area) + length(0)"}}
    assert evaluate_parameters(specs)["double"] == 4.0


# ------------------------------------------------------------------------------------------------------ physical quantities
def test_engineering_quantities_combine_dimensionally():
    specs = {"F": {"type": "force", "default": 5000.0}, "A": {"type": "area", "default": 0.01},
             "sigma": {"type": "pressure", "expression": "F / A"}, "m": {"type": "mass", "default": 10.0},
             "g": {"type": "acceleration", "default": 9.81}, "W": {"type": "force", "expression": "m * g"},
             "E": {"type": "energy", "expression": "W * length(2)"}, "t": {"type": "time", "default": 4.0},
             "P": {"type": "power", "expression": "E / t"}, "rho": {"type": "density", "expression": "m / volume(0.5)"}}
    v = evaluate_parameters(specs)
    assert v["sigma"] == pytest.approx(5e5) and v["W"] == pytest.approx(98.1) and v["P"] == pytest.approx(49.05) and v["rho"] == pytest.approx(20.0)
    with pytest.raises(ValidationError, match="dimension"):
        evaluate_parameters(dict(specs, sigma={"type": "force", "expression": "F / A"}))
    with pytest.raises(ValidationError, match="incompatible"):
        evaluate_parameters(dict(specs, bad={"type": "force", "expression": "F + m"}))


def test_absolute_and_differential_temperature_are_different_quantities():
    specs = {"T1": {"type": "temperature", "default": 293.15}, "T2": {"type": "temperature", "default": 283.15},
             "dT": {"type": "temperature_difference", "expression": "T1 - T2"},
             "T3": {"type": "temperature", "expression": "T1 + dT"},
             "warm": {"type": "boolean", "expression": "T1 > T2"}}
    v = evaluate_parameters(specs)
    assert v["dT"] == pytest.approx(10.0) and v["T3"] == pytest.approx(303.15) and v["warm"] is True
    for bad in ("T1 + T2", "2 * T1", "T1 / 2", "-T1", "abs(T1)", "T1 * dT"):
        with pytest.raises(ValidationError):
            evaluate_parameters(dict(specs, x={"type": "temperature", "expression": bad}))
    with pytest.raises(ValidationError):
        evaluate_parameters(dict(specs, x={"type": "temperature_difference", "expression": "T1 + dT"}))   # kind mismatch
    with pytest.raises(ValidationError):
        ev("t > d", t=Quantity(300.0, (0, 0, 0, 0, 1), True), d=Quantity(5.0, (0, 0, 0, 0, 1)))


def test_unit_conversion_goes_through_one_controlled_interface():
    assert to_si("length", 900, "mm") == pytest.approx(0.9) and from_si("length", 0.9, "mm") == pytest.approx(900)
    assert to_si("angle", 180, "deg") == pytest.approx(math.pi)
    assert to_si("pressure", 2, "MPa") == 2e6 and to_si("force", 3, "kN") == 3000
    assert to_si("temperature", 20, "C") == pytest.approx(293.15) and from_si("temperature", 293.15, "C") == pytest.approx(20)
    assert to_si("temperature_difference", 10, "C") == 10 and to_si("temperature_difference", 9, "F") == pytest.approx(5)
    assert units_for("length")[0] == "mm" and "psi" in units_for("pressure")
    with pytest.raises(ValidationError, match="unsupported length unit"):
        to_si("length", 1, "parsec")
    with pytest.raises(ValidationError):
        to_si("length", float("nan"), "mm")
    with pytest.raises(ValidationError):
        units_for("string")


# ------------------------------------------------------------------------------------------------------------ data kinds
def test_data_kinds_are_validated_not_arithmetic():
    ref = "7f3c1c9e-3b1a-4c53-9a58-1f2d4d5e6f70"
    specs = {"style": {"type": "enumeration", "choices": ["single", "double"], "default": "single"},
             "label": {"type": "string", "default": "D-01"}, "mat": {"type": "material_ref", "default": None},
             "origin": {"type": "coordinate", "default": [0, 0, 0]}, "axis": {"type": "vector", "size": 2, "default": [1, 0]},
             "host": {"type": "element_ref", "default": ref}}
    v = evaluate_parameters(specs, {"style": "double", "origin": [1, 2, 3]})
    assert v["style"] == "double" and v["origin"] == [1.0, 2.0, 3.0] and v["mat"] is None and v["host"] == ref
    for bad in ({"style": "triple"}, {"origin": [1, 2]}, {"axis": [1, True]}, {"label": 5}, {"host": "not an id"}):
        with pytest.raises(ValidationError):
            evaluate_parameters(specs, bad)
    with pytest.raises(ValidationError, match="cannot read"):
        evaluate_parameters(dict(specs, n={"type": "number", "expression": "label + 1"}))
    with pytest.raises(ValidationError):
        evaluate_parameters({"s": {"type": "string", "expression": "1"}})
    with pytest.raises(ValidationError):
        evaluate_parameters({"c": {"type": "enumeration", "choices": [], "default": "a"}})
    with pytest.raises(ValidationError):
        evaluate_parameters({"c": {"type": "coordinate", "size": 2, "default": [0, 0]}})


# ----------------------------------------------------------------------------------------------- diagnostics name the parameter
def test_errors_name_the_parameter_and_formula():
    specs = {"w": {"type": "length", "default": 1.0}, "z": {"type": "number", "default": 0.0}, "q": {"type": "length", "expression": "w / z"}}
    with pytest.raises(ValidationError, match=r"division by zero \(parameter 'q' = w / z\)") as exc:
        evaluate_parameters(specs)
    assert exc.value.context["parameter"] == "q" and exc.value.context["expression"] == "w / z"
    with pytest.raises(ValidationError, match="unknown expression dependencies: nope") as exc:
        evaluate_parameters({"a": {"type": "number", "expression": "nope + 1"}})
    assert exc.value.context["unknown"] == ["nope"]
    with pytest.raises(ValidationError, match="outside bounds") as exc:
        evaluate_parameters({"a": {"type": "length", "default": 5.0, "max": 2.0}})
    assert exc.value.context["parameter"] == "a" and exc.value.context["max"] == 2.0


def test_cycle_path_is_reported_with_its_participants():
    specs = {"width": {"type": "length", "expression": "height"}, "height": {"type": "length", "expression": "depth"},
             "depth": {"type": "length", "expression": "width"}, "other": {"type": "length", "default": 1.0}}
    with pytest.raises(ValidationError, match="cyclic parameter dependencies: depth -> width -> height -> depth|cyclic parameter dependencies") as exc:
        evaluate_parameters(specs)
    cycle = exc.value.context["cycle"]
    assert cycle[0] == cycle[-1] and set(cycle) == {"width", "height", "depth"}
    with pytest.raises(ValidationError, match="cyclic") as exc:
        evaluate_parameters({"a": {"type": "number", "expression": "a + 1"}})
    assert exc.value.context["cycle"] == ["a", "a"]


def test_complexity_limits_still_apply():
    with pytest.raises(ResourceLimitError):
        evaluate_parameters({"a": {"type": "number", "expression": "+".join(["1"] * 200)}})
    with pytest.raises(ValidationError, match="too long|length"):
        ev("1" + "+1" * 300)


# ---------------------------------------------------------------------------------------------------------------- the graph
CHAIN = {"building_width": {"type": "length", "default": 12.0},
         "usable_width": {"type": "length", "expression": "building_width - 2 * length(0.3)"},
         "panel_count": {"type": "integer", "expression": "floor(usable_width / length(1.5))", "min": 1},
         "panel_width": {"type": "length", "expression": "usable_width / panel_count"},
         "unrelated": {"type": "number", "default": 3.0}, "doubled": {"type": "number", "expression": "unrelated * 2"}}


def test_graph_queries_and_deterministic_order():
    g = ParameterGraph(CHAIN)
    assert g.order.index("building_width") < g.order.index("usable_width") < g.order.index("panel_count") < g.order.index("panel_width")
    assert g.transitive_dependents({"building_width"}) == {"usable_width", "panel_count", "panel_width"}
    assert g.transitive_dependencies({"panel_width"}) == {"usable_width", "panel_count", "building_width"}
    assert g.dependencies["panel_width"] == {"usable_width", "panel_count"}
    assert g.impact({"usable_width"}) == tuple(n for n in g.order if n in {"usable_width", "panel_count", "panel_width"})
    assert ("panel_width", "panel_count") in g.edges()
    assert ParameterGraph(dict(reversed(list(CHAIN.items())))).order == g.order       # insertion order does not matter
    with pytest.raises(ValidationError, match="unknown parameter"):
        g.impact({"ghost"})


def test_incremental_evaluation_recomputes_only_descendants():
    ev_ = ParameterEvaluator()
    full = ev_.evaluate(CHAIN)
    assert set(ev_.recomputed) == set(CHAIN) and ev_.full_evaluations == 1
    again = ev_.evaluate(CHAIN)
    assert again == full and ev_.recomputed == () and ev_.cache_hits == 1
    out = ev_.evaluate(CHAIN, {"building_width": 15.0})
    assert set(ev_.recomputed) == {"building_width", "usable_width", "panel_count", "panel_width"}      # not 'unrelated' or 'doubled'
    assert out == evaluate_parameters(CHAIN, {"building_width": 15.0}) and out["doubled"] == 6.0
    ev_.evaluate(CHAIN, {"building_width": 15.0, "unrelated": 4.0})
    assert set(ev_.recomputed) == {"unrelated", "doubled"}
    # a changed definition invalidates the whole cache, even though the overrides are the same
    edited = dict(CHAIN, usable_width={"type": "length", "expression": "building_width - 2 * length(0.5)"})
    out2 = ev_.evaluate(edited, {"building_width": 15.0, "unrelated": 4.0})
    assert set(ev_.recomputed) == set(CHAIN) and out2 == evaluate_parameters(edited, {"building_width": 15.0, "unrelated": 4.0})


def test_failed_incremental_edit_leaves_the_cache_usable():
    ev_ = ParameterEvaluator()
    good = ev_.evaluate(CHAIN)
    with pytest.raises(ValidationError, match="usable_width"):
        ev_.evaluate(CHAIN, {"building_width": 0.5})              # usable_width would be negative -> bounds failure downstream
    assert ev_.evaluate(CHAIN) == good


@pytest.mark.parametrize("seed", range(25))
def test_random_graphs_incremental_equals_full(seed):
    rnd = random.Random(seed)
    n = rnd.randint(4, 40)
    names = ["p%d" % i for i in range(n)]
    specs = {}
    for i, name in enumerate(names):
        if i < 3 or rnd.random() < 0.3:
            specs[name] = {"type": "number", "default": float(rnd.randint(1, 9))}
        else:
            a, b = rnd.sample(names[:i], 2)
            op = rnd.choice(["{a} + {b}", "{a} * {b} / 7", "max({a}, {b}) - min({a}, {b})", "if({a} > {b}, {a}, {b})", "abs({a} - {b}) + 1"])
            specs[name] = {"type": "number", "expression": op.format(a=a, b=b)}
    g = ParameterGraph(specs)
    cache = ParameterEvaluator()
    overrides = {}
    base = [m for m in names if "expression" not in specs[m]]
    for _ in range(8):
        for m in rnd.sample(base, rnd.randint(0, min(3, len(base)))):
            overrides[m] = float(rnd.randint(1, 9))
        got = cache.evaluate(specs, overrides)
        assert got == evaluate_parameters(specs, overrides)
        if cache.incremental_evaluations:
            dirty = set(cache.recomputed)
            assert dirty == set(g.impact(dirty & set(base)) if dirty & set(base) else ())     # exactly the changed + descendants


# ------------------------------------------------------------------------------------------------------- formula editor API
def test_check_formula_previews_value_dimension_and_graph():
    specs = {"width": {"type": "length", "default": 0.9}, "frame": {"type": "length", "default": 0.035}}
    ok = check_formula(specs, "clear_width", "width - 2 * frame", type="length")
    assert ok["ok"] and ok["value"] == pytest.approx(0.83) and ok["dimension"] == "length" and ok["dependencies"] == ["frame", "width"]
    assert specs == {"width": {"type": "length", "default": 0.9}, "frame": {"type": "length", "default": 0.035}}     # nothing changed
    bad = check_formula(specs, "clear_width", "width - 2 * missing", type="length")
    assert not bad["ok"] and "missing" in bad["error"]["message"] and bad["error"]["parameter"] == "clear_width"
    wrong = check_formula(specs, "clear_width", "width * frame", type="length")
    assert not wrong["ok"] and wrong["error"]["expected"] == "length" and wrong["error"]["actual"] == "area"
    cyc = check_formula({**specs, "a": {"type": "length", "expression": "width"}}, "width", "a + length(1)")
    assert not cyc["ok"] and cyc["error"]["code"] and set(cyc["error"]["cycle"]) == {"width", "a"}
    existing = check_formula({**specs, "x": {"type": "length", "default": 1.0}, "y": {"type": "length", "expression": "x"}}, "x", "frame * 2")
    assert existing["ok"] and existing["dependents"] == ["y"]


def test_function_catalog_documents_every_callable():
    names = {r["name"] for r in function_catalog()}
    assert {"if", "sin", "atan2", "floor", "roundup", "length", "pressure", "temperature_difference", "pi"} <= names
    assert all(r["signature"] and r["description"] for r in function_catalog())
