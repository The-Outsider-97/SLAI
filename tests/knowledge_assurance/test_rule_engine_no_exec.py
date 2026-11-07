"""RuleEngine must never execute rule text as Python.

Before the Phase A P0 patch, ``_run_rule_code`` used ``exec`` with an allow-listed
``__builtins__``. Walking ``().__class__.__base__.__subclasses__()`` recovered the
real ``__import__``, giving a rule full OS access. The rule below is that escape.
"""
from __future__ import annotations

import inspect
import unittest

from src.agents.knowledge.modules import rule_engine

ESCAPE_RULE = """
c=[x for x in ().__class__.__base__.__subclasses__() if x.__name__=='catch_warnings'][0]
imp=c()._module.__builtins__['__import__']
inferred={'ESCAPED': 1.0}
"""


class RuleEngineNoExecTests(unittest.TestCase):
    def test_sandbox_escape_rule_is_not_executed(self):
        try:
            result = rule_engine._run_rule_code(ESCAPE_RULE, {})
        except Exception:
            return  # rejected: expected outcome
        self.assertNotIn("ESCAPED", result)

    def test_worker_does_not_call_exec_or_eval(self):
        source = inspect.getsource(rule_engine._run_rule_code)
        self.assertNotIn("exec(", source)
        self.assertNotIn("eval(", source)


if __name__ == "__main__":
    unittest.main()
