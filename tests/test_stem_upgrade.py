"""Run from the SLAI repository root AFTER applying the STEM upgrade:
    py -m pytest tests/test_stem_upgrade.py -q
"""
import math
import pytest

from src.agents.stem import ParametricMath, EngineeringAnalysis, Quantity, UnitSystem
from src.agents.stem.utils.stem_errors import STEMEquationError, STEMUnitError


@pytest.fixture(scope="module")
def parametric():
    return ParametricMath()


def test_conversions_and_derived_dimension(parametric):
    u = parametric.units
    assert u.convert(30, "cm", "m") == pytest.approx(.3)
    assert u.convert(0, "°C", "K") == pytest.approx(273.15)
    assert u.convert(32, "°F", "°C") == pytest.approx(0)
    assert parametric.evaluate_expression("5*m + 30*cm").unit.dimension == u.lookup("m").dimension
    assert parametric.evaluate_expression("5*m + 30*cm").unit.to_base(
        parametric.evaluate_expression("5*m + 30*cm").magnitude) == pytest.approx(5.3)


def test_incompatible_units_are_rejected(parametric):
    with pytest.raises(STEMUnitError):
        parametric.evaluate_expression("5*m + 3*s")


def test_temperature_semantics(parametric):
    a = parametric.evaluate_expression("25*deg")  # angle, not temperature
    assert a.unit.to_base(a.magnitude) == pytest.approx(math.radians(25))
    c = parametric._quantity({"value": 20, "unit": "°C"})
    f = parametric._quantity({"value": 68, "unit": "°F"})
    delta = c - f
    assert delta.unit.symbol == "K"
    assert delta.magnitude == pytest.approx(0, abs=1e-10)
    with pytest.raises(STEMUnitError):
        _ = c + f
    with pytest.raises(STEMUnitError):
        _ = c * 2


def test_quantity_uncertainty_scalar_propagation(parametric):
    from src.agents.stem import UncertaintyValue
    q = Quantity(4.0, parametric.units.lookup("m"), UncertaintyValue(0.2))
    scaled = q * 3
    assert scaled.magnitude == pytest.approx(12)
    assert scaled.uncertainty.value == pytest.approx(.6)


def test_parametric_dependency(parametric):
    r = parametric.evaluate({"height": "0.5*width + 0.2*m", "area": "width*height"},
                            {"width": {"value": 4, "unit": "m"}})
    assert r["height"].unit.to_base(r["height"].magnitude) == pytest.approx(2.2)
    assert r["area"].unit.to_base(r["area"].magnitude) == pytest.approx(8.8)


def test_dependency_cycle(parametric):
    with pytest.raises(STEMEquationError):
        parametric.evaluate({"a": "b+1", "b": "c+1", "c": "a+1"})


@pytest.mark.parametrize("expression", [
    '__import__("os").system("echo unsafe")',
    '(1).__class__',
    '[x for x in (1,2)]',
    '10**1000000',
    'log(-1)',
    '1/0',
])
def test_unsafe_or_invalid_expressions(parametric, expression):
    with pytest.raises(Exception):
        parametric.evaluate_expression(expression)


def test_time_dependent_function_and_derivative(parametric):
    expr = '(pi/2)*rad * sin(pi*t/(2*T))**2'
    args = {"t": {"value": 0, "unit": "s"}, "T": {"value": 2, "unit": "s"}}
    first = parametric.evaluate_expression(expr, args)
    assert first.unit.to_base(first.magnitude) == pytest.approx(0, abs=1e-10)
    args["t"] = {"value": 2, "unit": "s"}
    last = parametric.evaluate_expression(expr, args)
    assert last.unit.to_base(last.magnitude) == pytest.approx(math.pi/2)
    args["t"] = {"value": 1, "unit": "s"}
    angular_velocity = parametric.derivative(expr, args, "t")
    assert angular_velocity["value"] == pytest.approx(math.pi**2 / 8, rel=1e-5)
    assert angular_velocity["status"] == "approximate"


def test_engineering_baselines():
    m = EngineeringAnalysis()
    r = m.axial_stress({"value": 5, "unit": "kN"}, {"value": 100, "unit": "cm**2"})
    assert r.value == pytest.approx(500000)
    t = m.heat_conduction_1d({"value": 0.5, "unit": "W/(m*K)"},
                              {"value": 10, "unit": "m**2"},
                              {"value": 20, "unit": "K"},
                              {"value": .25, "unit": "m"})
    assert t.value == pytest.approx(400)
    with pytest.raises(STEMUnitError):
        m.axial_stress({"value": 5, "unit": "s"}, {"value": 100, "unit": "cm**2"})


def test_beam_uses_existing_base_engine():
    analysis = EngineeringAnalysis()
    result = analysis.beam_response(
        span={"value": 4, "unit": "m"},
        load={"value": 5, "unit": "kN"},
        second_moment={"value": 8.5e-6, "unit": "m**4"},
        youngs_modulus={"value": 210, "unit": "GPa"},
        support="simply_supported", load_type="point_center",
    )
    expected = 5000*4**3/(48*210e9*8.5e-6)
    assert abs(result["outputs"]["max_deflection_m"]["value"]) == pytest.approx(expected, rel=1e-6)


def test_equation_no_eval():
    from src.agents.stem import Equation
    q = Equation("controlled", "sin(pi/2) + x", variables=("x",))
    assert q.evaluate(2) == pytest.approx(3)
