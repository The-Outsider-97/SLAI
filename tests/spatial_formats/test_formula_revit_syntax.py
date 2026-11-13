"""Revit-style formula spellings (report section 4.2, Tables 3-4) and the checks that must stay strict.

Internal values are SI (metres, radians), so a unit literal such as ``800 mm`` is the dimensioned constant 0.8 m; a plain ``800`` is a number and is
refused next to a length. Nothing here is a Python-only feature: each spelling is rewritten to the same checked subset before evaluation.
"""
import math

import pytest

from src.utils.spatial_formats.common.errors import ValidationError
from src.utils.spatial_formats.sgeo import parameters as P

BASE = {
    "Width": {"type": "length", "default": 0.9}, "Height": {"type": "length", "default": 2.1}, "Frame": {"type": "length", "default": 0.035},
    "Radius": {"type": "length", "default": 0.1}, "Spacing": {"type": "length", "default": 0.3}, "Panels": {"type": "integer", "default": 3},
    "ShowFrame": {"type": "boolean", "default": True}, "Angle": {"type": "angle", "default": math.radians(30)}, "Span": {"type": "length", "default": 4.0},
    "Finish": {"type": "material_ref"}, "Label": {"type": "string", "default": "A"},
}


def run(expression, kind, **overrides):
    specs = dict(BASE)
    specs["X"] = {"type": kind, "expression": expression}
    return P.evaluate_parameters(specs, overrides or None)["X"]


@pytest.mark.parametrize("expression, kind, expected", [
    ("Width - 2 * Frame", "length", 0.83),                                    # arithmetic
    ("Width * Height", "area", 0.9 * 2.1),                                    # area
    ("pi * Radius ^ 2", "area", math.pi * 0.01),                              # Revit exponent operator
    ("pi * Radius ** 2", "area", math.pi * 0.01),                             # the Python spelling still works
    ("if(Width < 800 mm, 25 mm, 35 mm)", "length", 0.035),                    # threshold, branches are lengths
    ("if(Width < 1000 mm, 25 mm, 35 mm)", "length", 0.025),
    ("Panels > 1", "boolean", True),                                          # visibility
    ("roundup(Width / Spacing)", "integer", 3),                               # rounding of a dimensionless ratio
    ("rounddown(Width / Spacing)", "integer", 3),
    ("round(Width / Spacing)", "integer", 3),
    ("Span * cos(Angle)", "length", 4.0 * math.cos(math.radians(30))),        # angle
    ("and(Panels > 1, ShowFrame)", "boolean", True),                          # logical combination as a function
    ("or(Panels > 5, not(ShowFrame))", "boolean", False),
    ("and(Panels > 1, ShowFrame, Width > 0.5 m)", "boolean", True),
    ("Panels > 1 and (ShowFrame)", "boolean", True),                          # the infix form with a parenthesised operand is not a function call
    ("log(100)", "number", 2.0),                                              # Revit log is base 10
    ("ln(pi)", "number", math.log(math.pi)),
    ("sqrt(Width * Width + Height * Height)", "length", math.hypot(0.9, 2.1)),
    ("abs(Width - Height)", "length", 1.2),
    ("min(Width, 500 mm)", "length", 0.5),
    ("90 deg", "angle", math.pi / 2),
    ("30 GPa", "pressure", 3e10),
    ("5 kN", "force", 5000.0),
    ("2 m2", "area", 2.0),
    ("1e3 mm", "length", 1.0),
])
def test_report_table_4_forms(expression, kind, expected):
    got = run(expression, kind)
    if isinstance(expected, bool) or isinstance(expected, int):
        assert got == expected and type(got) is type(expected)
    else:
        assert got == pytest.approx(expected, rel=1e-12)


def test_single_equals_means_equality():
    assert run("Width = 900 mm", "boolean") is True
    assert run("Width = 800 mm", "boolean") is False
    assert run("Width != 900 mm", "boolean") is False


def test_comparisons_tolerate_rounding_noise_but_not_real_differences():
    assert run("0.1 + 0.2 = 0.3", "boolean") is True
    assert run("0.1 + 0.2 <= 0.3", "boolean") is True
    assert run("0.1 + 0.2 >= 0.3", "boolean") is True
    assert run("0.1 + 0.2 < 0.3", "boolean") is False
    assert run("0.1 + 0.2 > 0.3", "boolean") is False
    assert run("Width < 900.001 mm", "boolean") is True
    assert run("Width = 900.001 mm", "boolean") is False


@pytest.mark.parametrize("expression, kind, fragment", [
    ("Width < 800", "boolean", "dimensions"),                  # a plain number is not a length, however it is compared
    ("Width + Panels", "length", "dimensions"),
    ("Width - Width * Height", "length", "dimensions"),
    ("if(Width < 800 mm, 25 mm, 35)", "length", "dimensions"),
    ("5 kN", "length", "dimension"),                           # a unit literal carries its own dimension
    ("5 parsec", "length", "syntax"),
    ("and(ShowFrame)", "boolean", "two arguments"),
    ("Finish + 1", "number", "cannot read"),                   # references are bound, never computed with
    ("Label = 1", "boolean", "cannot read"),
    ("Width > Height > ", "boolean", "syntax"),
])
def test_what_must_still_be_refused(expression, kind, fragment):
    with pytest.raises(ValidationError) as caught:
        run(expression, kind)
    assert fragment in str(caught.value)


def test_unit_suffix_is_not_taken_for_a_function_call():
    # ``2 m(`` has never been valid; it must stay an error rather than be read as 2 metres followed by a call
    with pytest.raises(ValidationError):
        run("2 m(3)", "length")


def test_dependencies_ignore_unit_symbols_and_boolean_function_names():
    deps = P.expression_dependencies("and(Panels > 1, Width > 800 mm) and ShowFrame")
    assert deps == {"Panels", "Width", "ShowFrame"}


def test_catalog_lists_the_revit_spellings():
    names = {row["name"] for row in P.function_catalog()}
    assert {"and", "or", "log", "roundup", "rounddown", "round", "if"} <= names


def test_if_branches_must_agree_even_when_the_other_branch_is_taken():
    # Width is 0.9 m, so the second branch is the one taken; the first would be a plain number next to a length.
    with pytest.raises(ValidationError) as caught:
        run("if(Width > 800 mm, 35, 25 mm)", "length")
    assert "dimensions" in str(caught.value)
    with pytest.raises(ValidationError):
        run("if(Width > 800 mm, 1, ShowFrame)", "number")


def test_a_guarded_domain_error_in_the_untaken_branch_is_still_allowed():
    assert run("if(Panels = 0, 0, 1 / Panels)", "number", Panels=0) == 0
    assert run("if(Panels = 0, 0, 1 / Panels)", "number", Panels=4) == 0.25
    assert run("if(Frame > 0 mm, 1, ln(0))", "number") == 1
