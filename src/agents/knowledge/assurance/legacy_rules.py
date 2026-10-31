"""Bounded parser/interpreter for a conservative subset of legacy SLAI rules.

This replaces ``exec(rule_code, ...)`` when the P0 hardening script is applied.
It does not use exec/eval/compile/ast.literal_eval/pickle/subprocess.
Unsupported Python constructs are rejected, NOT executed. This is an interim
compatibility bridge: all professional rules should migrate to data-only DSL.
"""
from __future__ import annotations

import ast
import math
from typing import Any, Mapping

from .contracts import ValidationFailure, json_object


class UnsafeLegacyRule(ValidationFailure):
    pass


_MAX_SCRIPT = 8000
_MAX_NODES = 180
_MAX_STEPS = 300
_MAX_FACTS = 32
_MAX_STRING = 1024


def _reject(message: str) -> None:
    raise UnsafeLegacyRule(f"Unsupported legacy rule: {message}")


def _safe_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _reject("numeric expression expected")
    if not math.isfinite(float(value)) or abs(float(value)) > 1e12:
        _reject("non-finite or oversized numeric expression")
    return float(value)


def _expr(node: ast.AST, kb: Mapping[str, Any], steps: list[int], depth: int = 0) -> Any:
    steps[0] += 1
    if steps[0] > _MAX_STEPS or depth > 25:
        _reject("execution budget exceeded")
    if isinstance(node, ast.Constant):
        val = node.value
        if not isinstance(val, (str, bool, int, float, type(None))):
            _reject("constant type")
        if isinstance(val, str) and len(val) > _MAX_STRING:
            _reject("string size")
        return val
    if isinstance(node, ast.Name):
        if node.id == "kb":
            return kb
        _reject("name access")
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "kb":
        key = _expr(node.slice, kb, steps, depth + 1)
        if not isinstance(key, str):
            _reject("knowledge key type")
        return kb.get(key)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "kb"
            and node.func.attr == "get" and not node.keywords and len(node.args) in (1, 2)):
        key = _expr(node.args[0], kb, steps, depth + 1)
        if not isinstance(key, str):
            _reject("knowledge key type")
        default = _expr(node.args[1], kb, steps, depth + 1) if len(node.args) == 2 else None
        return kb.get(key, default)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        if not node.values or len(node.values) > 20:
            _reject("boolean arity")
        # Python short-circuit semantics, but bounded and data-only.
        if isinstance(node.op, ast.And):
            result: Any = True
            for child in node.values:
                result = _expr(child, kb, steps, depth + 1)
                if not result:
                    break
            return result
        result = False
        for child in node.values:
            result = _expr(child, kb, steps, depth + 1)
            if result:
                break
        return result
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not bool(_expr(node.operand, kb, steps, depth + 1))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        a = _safe_number(_expr(node.operand, kb, steps, depth + 1))
        return a if isinstance(node.op, ast.UAdd) else -a
    if isinstance(node, ast.IfExp):
        branch = node.body if bool(_expr(node.test, kb, steps, depth + 1)) else node.orelse
        return _expr(branch, kb, steps, depth + 1)
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
        lhs = _expr(node.left, kb, steps, depth + 1)
        rhs = _expr(node.comparators[0], kb, steps, depth + 1)
        op = node.ops[0]
        try:
            if isinstance(op, ast.Eq): return lhs == rhs
            if isinstance(op, ast.NotEq): return lhs != rhs
            if isinstance(op, ast.Lt): return lhs < rhs
            if isinstance(op, ast.LtE): return lhs <= rhs
            if isinstance(op, ast.Gt): return lhs > rhs
            if isinstance(op, ast.GtE): return lhs >= rhs
            if isinstance(op, ast.Is): return lhs is rhs
            if isinstance(op, ast.IsNot): return lhs is not rhs
            if isinstance(op, ast.In): return lhs in rhs
            if isinstance(op, ast.NotIn): return lhs not in rhs
        except (TypeError, ValueError) as exc:
            raise UnsafeLegacyRule("incompatible comparison") from exc
        _reject("comparison operator")
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod)):
        left = _safe_number(_expr(node.left, kb, steps, depth + 1))
        right = _safe_number(_expr(node.right, kb, steps, depth + 1))
        if isinstance(node.op, ast.Add): result = left + right
        elif isinstance(node.op, ast.Sub): result = left - right
        elif isinstance(node.op, ast.Mult): result = left * right
        elif isinstance(node.op, ast.Div): result = left / right
        else: result = left % right
        return _safe_number(result)
    if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) <= 16:
        arr = [_expr(x, kb, steps, depth + 1) for x in node.elts]
        return arr if isinstance(node, ast.List) else tuple(arr)
    _reject(type(node).__name__)


def _statements(body: list[ast.stmt], kb: Mapping[str, Any],
                inferred: dict[str, float], steps: list[int], depth: int = 0) -> None:
    if depth > 18:
        _reject("statement nesting")
    for stmt in body:
        steps[0] += 1
        if steps[0] > _MAX_STEPS:
            _reject("execution step budget")
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.If):
            condition = bool(_expr(stmt.test, kb, steps, depth + 1))
            _statements(stmt.body if condition else stmt.orelse, kb, inferred, steps, depth + 1)
            continue
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if (not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name)
                    or target.value.id != "inferred"):
                _reject("assignment target")
            key = _expr(target.slice, kb, steps, depth + 1)
            if not isinstance(key, str) or not key or len(key) > 128:
                _reject("fact identifier")
            value = _safe_number(_expr(stmt.value, kb, steps, depth + 1))
            if not 0 <= value <= 1:
                _reject("confidence outside [0, 1]")
            if len(inferred) >= _MAX_FACTS and key not in inferred:
                _reject("fact count limit")
            inferred[key] = value
            continue
        _reject(type(stmt).__name__)


def interpret_legacy_rule_code(rule_code: str, knowledge_base: Mapping[Any, Any]) -> dict[str, float]:
    """Interpret a limited, non-executable subset of legacy inference snippets.

    No IO, imports, arbitrary attributes, loops, comprehensions, function calls,
    dynamic code, or user-defined objects are permitted. Unsupported snippets
    fail closed. This does not magically make authored rules trustworthy.
    """
    if not isinstance(rule_code, str) or len(rule_code) > _MAX_SCRIPT:
        _reject("script is missing or oversized")
    if not rule_code.strip():
        return {}
    if not isinstance(knowledge_base, Mapping):
        _reject("knowledge input must be an object")
    kb = json_object(dict(knowledge_base), "knowledge_base", max_bytes=262144)
    try:
        tree = ast.parse(rule_code, mode="exec")
    except SyntaxError as exc:
        raise UnsafeLegacyRule("malformed syntax") from exc
    if sum(1 for _ in ast.walk(tree)) > _MAX_NODES:
        _reject("too many AST nodes")
    inferred: dict[str, float] = {}
    _statements(tree.body, kb, inferred, [0])
    return inferred
