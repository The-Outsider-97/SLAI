"""Behavioral regression checks for the opt-in command execution scope."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('execution_guard_under_test', ROOT/'src/agents/execution/execution_guard.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GuardTests(unittest.TestCase):
    def test_direct_action_denied(self):
        with self.assertRaises(PermissionError):
            module.require_authorized_action('move_to')

    def test_authorization_scoped_by_registered_operation(self):
        with module.authorized_action('execution.action.move_to', 'command-1'):
            module.require_authorized_action('move_to')
            with self.assertRaises(PermissionError):
                module.require_authorized_action('pick_object')
        with self.assertRaises(PermissionError):
            module.require_authorized_action('move_to')

    def test_scope_restored_on_failure(self):
        with self.assertRaises(ValueError):
            with module.authorized_action('execution.action.move_to', 'one'):
                raise ValueError('failure')
        with self.assertRaises(PermissionError):
            module.require_authorized_action('move_to')

    def test_nested_scope_restores_outer(self):
        with module.authorized_action('execution.action.move_to', 'one'):
            with module.authorized_action('execution.action.pick_object', 'two'):
                module.require_authorized_action('pick_object')
            module.require_authorized_action('move_to')

    def test_invalid_scope_is_denied(self):
        for op, cmd in (('', 'x'), ('x','')):
            with self.assertRaises(PermissionError):
                with module.authorized_action(op, cmd):
                    pass

    def test_robot_adapter_blocks_direct_commands_and_reads(self):
        class Raw:
            calls = 0
            def stop(self):
                self.calls += 1
                return True
            def get_pose(self):
                return (0, 0, 0)
        raw = Raw()
        proxy = module.GuardedRobotAdapter(raw)
        self.assertTrue(proxy.supports_capability('stop'))
        with self.assertRaises(PermissionError):
            proxy.stop()
        with self.assertRaises(PermissionError):
            proxy.get_pose()
        self.assertEqual(raw.calls, 0)
        with module.authorized_action('execution.action.stop', 'one'):
            self.assertTrue(proxy.stop())
        self.assertEqual(raw.calls, 1)

    def test_robot_adapter_does_not_expose_private_attributes(self):
        class Raw:
            def stop(self):
                return True
        proxy = module.GuardedRobotAdapter(Raw())
        with self.assertRaises(AttributeError):
            proxy._hidden


if __name__ == '__main__':
    unittest.main()