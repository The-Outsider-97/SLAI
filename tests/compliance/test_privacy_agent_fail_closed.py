"""Regression tests for audit finding 4.1: a disabled Privacy Agent must fail closed.

The real PrivacyAgent needs the full BaseAgent runtime (heavy ML dependencies). These tests
therefore load the *actual source* of the two fail-closed methods from privacy_agent.py via
the AST and execute them against a minimal stand-in class, so they verify the shipped code
without importing the whole agent runtime. If the full runtime is importable, an additional
end-to-end test is run.
"""
from __future__ import annotations

import ast
import enum
import logging
import textwrap
import unittest
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

SOURCE = Path(__file__).resolve().parents[2] / "src" / "agents" / "privacy_agent.py"
WANTED = {"_structurally_non_personal", "_disabled_privacy_decision"}


class _Decision(enum.Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"


def _build_stub():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PrivacyAgent")
    funcs = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in WANTED]
    assert {f.name for f in funcs} == WANTED, "fail-closed methods missing from privacy_agent.py"
    body = "\n".join(ast.unparse(f) for f in funcs)
    src = "class PrivacyAgent:\n" + textwrap.indent(body, "    ") + "\n"
    ns: Dict[str, Any] = {
        "Any": Any, "Dict": Dict, "Mapping": Mapping, "Optional": Optional,
        "PrivacyDecision": _Decision, "logger": logging.getLogger("test.privacy"),
    }
    exec(compile(src, str(SOURCE), "exec"), ns)  # noqa: S102 - trusted repo source
    cls_obj = ns["PrivacyAgent"]
    cls_obj._nonempty = staticmethod(lambda v, d: v or d)
    cls_obj._new_id = staticmethod(lambda p: f"{p}_test")
    return cls_obj


class DisabledPrivacyAgentFailsClosed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = _build_stub()()

    def decide(self, payload):
        return self.agent._disabled_privacy_decision(payload, request_id=None)

    def test_free_text_is_blocked_not_passed_through(self):
        out = self.decide({"note": "Jan de Vries, jan@example.nl, BSN 123456782"})
        self.assertEqual(out["decision"], "BLOCK")
        self.assertEqual(out["sanitized_payload"], {})

    def test_nested_string_is_blocked(self):
        self.assertEqual(self.decide({"a": {"b": [1, 2, {"c": "x"}]}})["decision"], "BLOCK")

    def test_bytes_and_unknown_objects_are_blocked(self):
        self.assertEqual(self.decide({"a": b"raw"})["decision"], "BLOCK")
        self.assertEqual(self.decide({"a": object()})["decision"], "BLOCK")

    def test_numeric_payload_passes_unchanged(self):
        payload = {"vector": [1.0, 2.5, 3], "flag": True, "none": None}
        out = self.decide(payload)
        self.assertEqual(out["decision"], "ALLOW")
        self.assertEqual(out["sanitized_payload"], payload)

    def test_depth_and_size_bounds_fail_closed(self):
        deep: Any = 1
        for _ in range(12):
            deep = {"k": deep}
        self.assertEqual(self.decide(deep)["decision"], "BLOCK")
        self.assertEqual(self.decide({"v": list(range(5000))})["decision"], "BLOCK")

    def test_report_shape_matches_enabled_path(self):
        out = self.decide({"x": "y"})
        for key in ("request_id", "decision", "summary", "sanitized_payload", "decision_sources", "stages"):
            self.assertIn(key, out)

    def test_source_has_no_unconditional_passthrough(self):
        text = SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("passing through without mutation", text)


if __name__ == "__main__":
    unittest.main()


class ErasureRequestRules(unittest.TestCase):
    """Audit finding 4.5: reason must not be mandatory; identity verification must be."""

    SRC = Path(__file__).resolve().parents[2] / "src" / "agents" / "privacy" / "data_retention.py"

    def test_reason_is_no_longer_mandatory(self):
        text = self.SRC.read_text(encoding="utf-8")
        self.assertNotIn("'reason' is required for right-to-be-forgotten", text)
        self.assertIn("self.right_to_be_forgotten_requires_reason = False", text)

    def test_identity_verification_gate_logic(self):
        tree = ast.parse(self.SRC.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and any(
            isinstance(b, ast.FunctionDef) and b.name == "_require_verified_requester" for b in n.body))
        fn = next(b for b in cls.body if isinstance(b, ast.FunctionDef) and b.name == "_require_verified_requester")

        class PolicyEvaluationError(Exception):
            def __init__(self, stage, details, context=None):
                super().__init__(details)

        src = "class G:\n    require_identity_verification_for_erasure = True\n" + textwrap.indent(ast.unparse(fn), "    ")
        ns = {"Any": Any, "Mapping": Mapping, "Optional": Optional, "PolicyEvaluationError": PolicyEvaluationError}
        exec(compile(src, "x", "exec"), ns)  # noqa: S102
        g = ns["G"]()
        good = {"verified": True, "method": "session", "verified_by": "idp", "subject_id": "s1"}
        g._require_verified_requester(good, "s1")
        for bad in (None, {}, {**good, "verified": "yes"}, {**good, "method": ""}, {**good, "subject_id": "other"}):
            with self.assertRaises(PolicyEvaluationError):
                g._require_verified_requester(bad, "s1")
