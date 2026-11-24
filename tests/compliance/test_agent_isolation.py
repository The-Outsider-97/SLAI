"""Audit finding 4.6: delegation must only reach public, class-defined, authorised actions."""
from __future__ import annotations

import unittest

from src.agents.governance.delegation_guard import (
    DelegationDenied, check_delegable, install_authorizer,
)


class Target:
    def analyse(self, x=1):
        return x * 2

    def _secret(self):
        return "no"

    def shutdown(self):
        return "no"

    @property
    def prop(self):
        return lambda: "no"

    @staticmethod
    def static_ok():
        return "ok"


class DelegationGuardTests(unittest.TestCase):
    def tearDown(self):
        install_authorizer(None)

    def call(self, action, target=None):
        return check_delegable(target or Target(), target_name="t", action=action, caller="c")

    def test_public_class_method_allowed(self):
        self.assertEqual(self.call("analyse")(x=4), 8)
        self.assertEqual(self.call("static_ok")(), "ok")

    def test_private_dunder_and_invalid_names_denied(self):
        for bad in ("_secret", "__init__", "__class__", "", "a b", "a;b", "x" * 100, None, 5):
            with self.assertRaises(DelegationDenied, msg=repr(bad)):
                self.call(bad)

    def test_lifecycle_and_property_denied(self):
        for bad in ("shutdown", "prop", "missing"):
            with self.assertRaises(DelegationDenied):
                self.call(bad)

    def test_injected_instance_callable_denied(self):
        t = Target()
        t.evil = lambda: "pwned"
        with self.assertRaises(DelegationDenied):
            self.call("evil", t)

    def test_authorizer_can_deny_and_failure_denies(self):
        def deny(caller, target, action):
            raise DelegationDenied("registry says no")

        install_authorizer(deny)
        with self.assertRaises(DelegationDenied):
            self.call("analyse")

        def broken(caller, target, action):
            raise RuntimeError("boom")

        install_authorizer(broken)
        with self.assertRaises(DelegationDenied):
            self.call("analyse")

    def test_authorizer_allow(self):
        seen = []
        install_authorizer(lambda c, t, a: seen.append((c, t, a)))
        self.call("analyse")
        self.assertEqual(seen, [("c", "t", "analyse")])

    def test_is_subclass_of_permission_error(self):
        self.assertTrue(issubclass(DelegationDenied, PermissionError))


if __name__ == "__main__":
    unittest.main()
