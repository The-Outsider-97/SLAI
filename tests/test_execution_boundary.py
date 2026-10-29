import importlib.util
import sys
from pathlib import Path
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'src' / 'agents' / 'execution'
for name in ['test_execution', 'test_execution.command_gateway', 'test_execution.execution_guard', 'test_execution.execution_boundary']:
    sys.modules.pop(name, None)
module = types.ModuleType('test_execution')
module.__path__ = [str(ROOT)]
sys.modules['test_execution'] = module

def get_module(short):
    name = 'test_execution.' + short
    spec = importlib.util.spec_from_file_location(name, ROOT / (short+'.py'))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

gw = get_module('command_gateway')
guard = get_module('execution_guard')
boundary = get_module('execution_boundary')

class Auth:
    def __init__(self): self.allowed = True
    def authorize(self, ctx, command, cap):
        return gw.Decision(bool(self.allowed and ctx == 'trusted'), 'trusted-principal')

class Action:
    calls = 0
    retry_attempts = 4
    def __init__(self, context): self.context = context
    def execute(self):
        guard.require_authorized_action('run')
        assert self.retry_attempts == 0
        type(self).calls += 1
        self.context['count'] = type(self).calls
        return True

class BreakAction:
    calls = 0
    def __init__(self, context): self.context = context
    def execute(self):
        guard.require_authorized_action('break')
        type(self).calls += 1
        raise RuntimeError('effect then lost reply')

class Tests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.auth = Auth()
        self.gateway = gw.CommandGateway(Path(self.dir.name)/'log.sqlite', self.auth)
        self.seen = []
        Action.calls=0; BreakAction.calls=0
        self.b = boundary.ExecutionBoundary(self.gateway, {'run':Action,'break':BreakAction},
                                             apply_action=lambda x: self.seen.append(x.context))
    def tearDown(self):
        self.gateway.close();self.dir.cleanup()
    def test_no_identity_no_dispatch(self):
        with self.assertRaises(PermissionError): self.b.start_task('id',request_context=None)
        with self.assertRaises(PermissionError): self.b.dispatch('run', {})
    def test_authorized_single_dispatch(self):
        self.b.start_task('id',request_context='trusted')
        self.b.dispatch('run',{})
        self.assertEqual(Action.calls,1)
        self.assertEqual(self.seen,[{'count':1,'robot_ready':False}])
        self.b.finish_task()
    def test_denied_no_side_effect(self):
        self.b.start_task('id',request_context='forged')
        with self.assertRaises(gw.CommandRejected): self.b.dispatch('run',{})
        self.assertEqual(Action.calls,0)
    def test_duplicate_after_restart_not_reexecuted(self):
        self.b.start_task('id',request_context='trusted')
        self.b.dispatch('run',{});self.b.finish_task()
        self.b.start_task('id',request_context='trusted')
        with self.assertRaises(gw.CommandRejected): self.b.dispatch('run',{})
        self.assertEqual(Action.calls,1)
    def test_exception_after_side_effect_is_not_retried(self):
        self.b.start_task('id',request_context='trusted')
        with self.assertRaises(gw.CommandRejected): self.b.dispatch('break',{})
        with self.assertRaises(gw.CommandRejected): self.b.dispatch('break',{})
        self.assertEqual(BreakAction.calls,1)
    def test_agent_cannot_inject_raw_robot(self):
        class Raw:
            calls=0
            def stop(self): self.calls+=1;return True
        raw=Raw()
        self.b.start_task('id',request_context='trusted')
        self.b.dispatch('run',{'robot':raw})
        self.assertNotIn('robot',self.seen[0])
        self.assertEqual(raw.calls,0)
    def test_unregistered_action_rejected(self):
        self.b.start_task('id',request_context='trusted')
        with self.assertRaises(gw.CommandRejected): self.b.dispatch('other',{})

if __name__=='__main__':unittest.main()