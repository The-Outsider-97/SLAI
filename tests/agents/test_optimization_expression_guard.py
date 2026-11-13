"""Regression checks for the restricted optimization expression evaluator.

Integer exponentiation used to be unbounded: ``9**9**9`` was accepted by the AST
whitelist and then never finished. Powers are now evaluated on floats, so an
unrepresentable result fails fast with an evaluation error.
"""
from __future__ import annotations

import time

import pytest

from src.agents.optimization.utils.optimization_errors import (
    OptimizationEvaluationError,
    OptimizationValidationError,
)
from src.agents.optimization.utils.optimization_helpers import compile_expression


def _eval(source: str, **values: float) -> float:
    evaluate, _ = compile_expression(source)
    return evaluate(values)


def test_power_semantics_are_preserved() -> None:
    assert _eval("x**2 + 1", x=3) == 10.0
    assert _eval("2**-1") == pytest.approx(0.5)
    assert _eval("-x**2", x=3) == -9.0  # unary minus binds looser than **
    assert _eval("2**3**2") == 512.0  # right associative


@pytest.mark.parametrize("source", ["9**9**9**2", "9**9**9 + x", "10**1000**2"])
def test_unbounded_integer_power_fails_fast(source: str) -> None:
    evaluate, _ = compile_expression(source)
    started = time.monotonic()
    with pytest.raises(OptimizationEvaluationError):
        evaluate({"x": 1.0})
    assert time.monotonic() - started < 1.0


def test_internal_rewrite_name_is_reserved() -> None:
    with pytest.raises(OptimizationValidationError):
        compile_expression("__guarded_pow__(1, 2)")
