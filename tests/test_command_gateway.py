"""Isolated, standard-library regression tests for the general command gateway."""
import importlib.util
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'src/agents/execution/command_gateway.py'
spec = importlib.util.spec_from_file_location('command_gateway', SOURCE)
module = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Command, Capability = module.Command, module.Capability
CommandGateway, Decision, Outcome = module.CommandGateway, module.Decision, module.Outcome


class Authority:
    def __init__(self):
        self.permitted = True
        self.principal = 'trusted-principal'
        self.allow_reconcile = False
    def authorize(self, ctx, command, capability):
        return Decision(self.permitted and ctx == 'trusted-session', self.principal)
    def authorize_reconciliation(self, principal, resource, key, outcome):
        return self.allow_reconcile


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / 'journal.sqlite'
        self.authority = Authority()
        self.gateway = CommandGateway(self.path, self.authority)
        self.effects = []
        self.gateway.register(Capability('write', lambda c: self.effects.append(c.parameters['value']) or {'ok':True}))

    def tearDown(self):
        self.gateway.close()
        self.dir.cleanup()

    def cmd(self, i='k1', *, value=7, op='write', deadline=None, expected_version=None):
        return Command(command_id='cmd-'+i,idempotency_key=i,operation=op,
                       resource='resource-a',parameters={'value':value}, deadline=deadline, expected_version=expected_version)

    def run_cmd(self,c,session='trusted-session'):
        return self.gateway.execute(c, request_context=session)

    def test_authorized_effect(self):
        r = self.run_cmd(self.cmd())
        self.assertEqual(r.status, Outcome.SUCCEEDED)
        self.assertEqual(self.effects, [7])

    def test_untrusted_session_denied_no_effect(self):
        self.assertEqual(self.run_cmd(self.cmd(), 'forged').status, Outcome.DENIED)
        self.assertEqual(self.effects, [])

    def test_revoked_session_cannot_replay(self):
        self.run_cmd(self.cmd())
        self.authority.permitted=False
        self.assertEqual(self.run_cmd(self.cmd()).status,Outcome.DENIED)

    def test_idempotency_duplicate_suppressed(self):
        first=self.run_cmd(self.cmd())
        later=self.run_cmd(self.cmd())
        self.assertEqual((first.status,later.status,later.replayed), (Outcome.SUCCEEDED,Outcome.SUCCEEDED,True))
        self.assertEqual(self.effects,[7])

    def test_idempotency_conflict(self):
        self.run_cmd(self.cmd())
        self.assertEqual(self.run_cmd(self.cmd(value=8)).status,Outcome.REJECTED)
        self.assertEqual(self.effects,[7])

    def test_missing_capability(self):
        self.assertEqual(self.run_cmd(self.cmd(op='not_registered')).status,Outcome.REJECTED)

    def test_expired_deadline(self):
        self.assertEqual(self.run_cmd(self.cmd(deadline=time.time()-1)).status,Outcome.REJECTED)
        self.assertEqual(self.effects,[])

    def test_version_unsupported(self):
        self.assertEqual(self.run_cmd(self.cmd(expected_version='3')).status,Outcome.REJECTED)

    def test_version_conflict(self):
        self.gateway.register(Capability('versioned',lambda c:self.effects.append('changed'), version_reader=lambda _: '4'))
        self.assertEqual(self.run_cmd(self.cmd(op='versioned',expected_version='3')).status,Outcome.REJECTED)
        self.assertEqual(self.effects,[])

    def test_verifier_failed_ambiguous_result(self):
        self.gateway.register(Capability('unverified',lambda c:self.effects.append('changed'),verifier=lambda c,res:False))
        self.assertEqual(self.run_cmd(self.cmd(op='unverified')).status,Outcome.INDETERMINATE)
        self.assertEqual(self.effects,['changed'])

    def test_exception_after_external_effect_marks_uncertain(self):
        def fail(c):
            self.effects.append('did-send')
            raise OSError('lost acknowledgement')
        self.gateway.register(Capability('mail', fail))
        self.assertEqual(self.run_cmd(self.cmd(op='mail')).status,Outcome.INDETERMINATE)
        self.assertEqual(self.run_cmd(self.cmd(op='mail')).status,Outcome.INDETERMINATE)
        self.assertEqual(self.effects,['did-send'])

    def test_read_failure_not_external_side_effect(self):
        def fail(c): raise OSError('read offline')
        self.gateway.register(Capability('read', fail, effect='read'))
        self.assertEqual(self.run_cmd(self.cmd(op='read')).status,Outcome.FAILED)

    def test_restart_replay_does_not_repeat(self):
        self.run_cmd(self.cmd())
        self.gateway.close()
        self.gateway=CommandGateway(self.path,self.authority)
        self.gateway.register(Capability('write',lambda c:self.effects.append('repeat')))
        self.assertEqual(self.run_cmd(self.cmd()).status,Outcome.SUCCEEDED)
        self.assertEqual(self.effects,[7])

    def test_crash_window_in_progress_does_not_retry(self):
        c=self.cmd()
        fp=c.fingerprint()
        with sqlite3.connect(self.path) as db:
            db.execute('''INSERT INTO command_journal VALUES(?,?,?,?,?,?,?)''',
                       ('trusted-principal','resource-a','k1',fp,'cmd-k1','IN_PROGRESS',time.time()))
        r=self.run_cmd(c)
        self.assertEqual(r.status,Outcome.IN_PROGRESS)
        self.assertFalse(self.effects)

    def test_parallel_duplicate(self):
        barrier=threading.Barrier(8)
        def handler(c):
            self.effects.append('once')
            time.sleep(.04)
        self.gateway.register(Capability('parallel',handler))
        results=[]
        def worker():
            barrier.wait()
            results.append(self.run_cmd(self.cmd(op='parallel')).status)
        threads=[threading.Thread(target=worker) for _ in range(8)]
        for t in threads:t.start()
        for t in threads:t.join()
        self.assertEqual(self.effects,['once'])
        self.assertTrue(all(x in (Outcome.IN_PROGRESS,Outcome.SUCCEEDED) for x in results))

    def test_sqlite_transaction_atomic_rollback_and_commit(self):
        with self.gateway._db:
            self.gateway._db.execute('CREATE TABLE items (id TEXT PRIMARY KEY, val INTEGER)')
        def good(c, con):
            con.execute('INSERT INTO items VALUES (?,?)', ('a', c.parameters['value']))
            return 'applied'
        def bad(c, con):
            con.execute('INSERT INTO items VALUES (?,?)', ('b', c.parameters['value']))
            raise RuntimeError('after SQL write')
        self.gateway.register(Capability('txgood',good,effect='transactional'))
        self.gateway.register(Capability('txbad',bad,effect='transactional'))
        self.assertEqual(self.run_cmd(self.cmd(op='txgood')).status, Outcome.COMMITTED)
        self.assertEqual(self.run_cmd(self.cmd(i='k2',op='txbad')).status, Outcome.FAILED)
        rows=self.gateway._db.execute('SELECT id FROM items').fetchall()
        self.assertEqual(rows,[('a',)])
        self.assertEqual(self.run_cmd(self.cmd(op='txgood')).status, Outcome.COMMITTED)
        self.assertEqual(self.gateway._db.execute('SELECT count(*) FROM items').fetchone()[0],1)

    def test_malformed_parameters_rejected(self):
        with self.assertRaises(module.CommandRejected):
            Command('x','key','write','resource',{'notjson':object()})
        with self.assertRaises(module.CommandRejected):
            Command('x','key','write','resource',{'nan':float('nan')})

    def test_authorized_reconciliation_with_evidence(self):
        def fail(c): raise OSError('ack lost')
        self.gateway.register(Capability('mail',fail))
        self.run_cmd(self.cmd(op='mail'))
        self.authority.allow_reconcile = True
        self.assertFalse(self.gateway.reconcile('trusted-principal','resource-a','missing',observed_outcome=Outcome.COMMITTED,authoritative_evidence=True))
        self.authority.allow_reconcile = False
        with self.assertRaises(module.CommandRejected):
            self.gateway.reconcile('trusted-principal','resource-a','k1',observed_outcome=Outcome.COMMITTED)
        self.authority.allow_reconcile = True
        self.assertTrue(self.gateway.reconcile('trusted-principal','resource-a','k1',observed_outcome=Outcome.COMMITTED,authoritative_evidence=True))
        self.assertEqual(self.run_cmd(self.cmd(op='mail')).status,Outcome.COMMITTED)


if __name__=='__main__':unittest.main()
