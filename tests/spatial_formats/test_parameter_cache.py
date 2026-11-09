"""The memo in ``evaluate_parameters``: resolved values are remembered for a table and a set of overrides (scene validation resolves every instance on every edit),
and remembering must be invisible - same values, no shared state between callers, no failure ever served from it."""
import copy

import pytest

from src.utils.spatial_formats.common.errors import ValidationError
from src.utils.spatial_formats.sgeo import parameters as SP

SPECS = {"width": {"type": "length", "default": 0.9, "min": 0.6, "max": 1.8}, "frame": {"type": "length", "default": 0.035},
         "clear": {"type": "length", "expression": "width - 2 * frame"}, "pos": {"type": "vector", "default": [1.0, 2.0, 3.0], "dimension": "length"}}


def fresh(specs, overrides=None):
    """What the engine gives with every cache bypassed."""
    return SP.ParameterGraph(copy.deepcopy(specs)).evaluate(overrides)


def test_cached_results_equal_uncached_ones_for_many_override_sets():
    for w in (0.6, 0.75, 0.9, 1.2, 1.8):
        for f in (0.02, 0.035, 0.08):
            overrides = {"width": w, "frame": f}
            assert SP.evaluate_parameters(SPECS, overrides) == fresh(SPECS, overrides)
            assert SP.evaluate_parameters(SPECS, overrides) == fresh(SPECS, overrides)       # and again, now from the memo


def test_a_caller_changing_its_result_cannot_change_what_the_next_caller_gets():
    first = SP.evaluate_parameters(SPECS, {"width": 1.0})
    first["clear"] = -1
    first["pos"].append(99.0)
    second = SP.evaluate_parameters(SPECS, {"width": 1.0})
    assert second == fresh(SPECS, {"width": 1.0})
    assert second["pos"] == [1.0, 2.0, 3.0]


def test_different_tables_and_overrides_never_share_an_entry():
    a = SP.evaluate_parameters(SPECS, {"width": 1.0})
    changed = copy.deepcopy(SPECS)
    changed["clear"]["expression"] = "width - frame"
    b = SP.evaluate_parameters(changed, {"width": 1.0})
    assert a["clear"] != b["clear"]
    assert SP.evaluate_parameters(SPECS, {"width": 1.1})["clear"] != a["clear"]
    assert SP.evaluate_parameters(SPECS, None) == SP.evaluate_parameters(SPECS, {}) == fresh(SPECS)


def test_a_failure_is_raised_every_time_and_never_remembered():
    for _ in range(3):
        with pytest.raises(ValidationError):
            SP.evaluate_parameters(SPECS, {"width": 99.0})                                   # above its maximum
    assert SP.evaluate_parameters(SPECS, {"width": 1.0}) == fresh(SPECS, {"width": 1.0})      # the engine still works after


def test_the_memo_is_bounded():
    for i in range(SP._RESULT_CACHE_SIZE + 50):
        SP.evaluate_parameters({"x": {"type": "number", "default": 1.0}}, {"x": float(i)})
    assert len(SP._RESULT_CACHE) <= SP._RESULT_CACHE_SIZE
