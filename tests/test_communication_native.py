"""Native SLDC release regressions; no cryptographic test doubles."""
import multiprocessing
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.communication.communication_security import _read_private, read_encrypted
from src.communication.communication_storage import CommunicationJournal
from src.communication.communication_types import new_message
from src.utils import sldc


def _append_in_child(directory, pin, record, start, results):
    """Spawn-safe worker: independent process and native SLDC journal."""
    try:
        journal = CommunicationJournal(Path(directory), pin)
        if not start.wait(60):
            raise TimeoutError('Writer start timed out')
        journal.append(record)
    except FileExistsError:
        results.put('duplicate')
    except Exception as exc:
        results.put(type(exc).__name__)
    else:
        results.put('committed')


class NativeCommunicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / 'service'
        self.pin = CommunicationJournal.initialize(self.directory)
        self.journal = CommunicationJournal(self.directory, self.pin)
        self.password = 'fixture-reader-passphrase-01'
        self.scope = dict(agents=['sender'], sessions=['session'],
                          kinds=['event'], classifications=['internal'])
        self.journal.add_user('reader', self.password, self.scope)

    def record(self, **kwargs):
        params = dict(session_id='session', classification='internal')
        params.update(kwargs)
        return new_message('sender', 'recipient', 'event', {'value': 'fixture'}, **params)

    def test_native_scope_rotation_restart_and_revocation(self):
        record = self.record()
        path = self.journal.append(record)
        self.assertEqual(sldc.inspect_package(path)['container'], 'encrypted')
        with self.assertRaises(sldc.SLDCDecryptionError):
            sldc.read_package(path)
        with self.assertRaises(PermissionError):
            self.journal.view('reader', 'incorrect-fixture-password')
        self.assertEqual(self.journal.view('reader', self.password)['records'], [record])
        self.journal.append(self.record(classification='restricted'))
        self.journal.append(self.record(session_id='other'))
        self.assertEqual(self.journal.view('reader', self.password)['records'], [record])
        with self.assertRaises(PermissionError):
            self.journal.view('reader', self.password, session='other')
        with self.assertRaises(FileExistsError):
            self.journal.append(record)
        old_epoch = self.journal.status()['current_epoch']
        self.journal.rotate_keys()
        restarted = CommunicationJournal(self.directory, self.pin)
        self.assertEqual(restarted.view('reader', self.password)['records'], [record])
        with self.assertRaises(FileExistsError):
            restarted.append(record)
        restarted.revoke_epoch(old_epoch)
        self.assertEqual(self.journal.view('reader', self.password)['records'], [])
        restarted.revoke_user('reader')
        with self.assertRaises(PermissionError):
            self.journal.view('reader', self.password)

    def test_audit_failure_prevents_disclosure(self):
        self.journal.append(self.record())
        original = self.journal._audit
        def fail_final(state, actor, action, status, **kwargs):
            if action == 'view' and status == 'allowed':
                raise OSError('injected final audit persistence failure')
            return original(state, actor, action, status, **kwargs)
        with patch.object(self.journal, '_audit', side_effect=fail_final):
            with self.assertRaises(OSError):
                self.journal.view('reader', self.password)
        events = self.journal.read_audit(limit=100)['events']
        self.assertFalse(any(e['action'] == 'view' and e['status'] == 'allowed' for e in events))

    def test_explicit_filters_only_narrow_authorized_records(self):
        record = self.record(task_id='task', correlation_id='correlation')
        self.journal.append(record)
        self.journal.append(self.record(task_id='other'))
        view = lambda **filters: self.journal.view('reader', self.password, **filters)['records']
        self.assertEqual(view(sender='sender', recipient='recipient', task='task',
                              correlation='correlation', classification='internal',
                              record_id=record['id']), [record])
        self.assertEqual(view(sender='recipient'), [])
        with self.assertRaises(PermissionError):
            view(classification='restricted')

    def test_admin_actor_bound_to_os(self):
        from src.communication.communication_security import operator_principal
        with self.assertRaises(PermissionError):
            CommunicationJournal(self.directory, self.pin, operator_id='impersonated')
        self.journal.rotate_keys()
        events = self.journal.read_audit()['events']
        self.assertEqual({e['actor'] for e in events if e['action'] == 'rotate_keys'},
                         {operator_principal()})

    @unittest.skipUnless(os.name == 'posix', 'POSIX directory sync')
    def test_directory_sync_error_prevents_disclosure(self):
        import stat
        real_sync = os.fsync
        def fail_directory(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError('directory sync failed')
            return real_sync(fd)
        self.journal.append(self.record())
        with patch('os.fsync', side_effect=fail_directory), self.assertRaises(OSError):
            self.journal.view('reader', self.password)

    @unittest.skipUnless(hasattr(__import__('socket'), 'SO_PEERCRED') and os.name == 'posix',
                         'Linux Unix peer credentials')
    def test_local_reader_policy_and_restart_backoff(self):
        import json
        import socket
        from threading import Thread
        from src.communication.communication_service import CommunicationReaderService

        def request(policy, body, accounts=None):
            journal = CommunicationJournal(self.directory, self.pin)
            service = CommunicationReaderService(journal, {os.geteuid(): 'reader'} if accounts is None
                                                 else accounts, policy)
            client, server = socket.socketpair()
            thread = Thread(target=service.serve_connection, args=(server,))
            thread.start()
            try:
                with client, client.makefile('rb') as response:
                    client.settimeout(10)
                    client.sendall(json.dumps(body).encode() + bytes([10]))
                    return json.loads(response.readline())
            finally:
                thread.join(10)
                self.assertFalse(thread.is_alive())

        record = self.record()
        self.journal.append(record)
        body = {'password': self.password}
        allow = lambda user, message: user == 'reader'
        self.assertEqual(request(allow, body)['records'], [record])
        self.assertEqual(request(lambda user, message: False, body)['records'], [])
        self.assertEqual(request(allow, body, accounts={}), {'ok': False})
        self.assertEqual(request(allow, dict(body, user='admin')), {'ok': False})
        with patch('src.communication.communication_security.time.time', return_value=2000000000):
            self.assertEqual(request(allow, {'password': 'wrong-test-password'}), {'ok': False})
            self.assertEqual(request(allow, body), {'ok': False})
        with patch('src.communication.communication_security.time.time', return_value=2000000002):
            self.assertEqual(request(allow, body)['records'], [record])
        self.journal.revoke_user('reader')
        self.assertEqual(request(allow, body), {'ok': False})

    def test_wrong_root_and_corrupt_archive(self):
        record = self.record()
        path = self.journal.append(record)
        wrong_pin = '0' * 64 if self.pin != '0' * 64 else '1' * 64
        with self.assertRaises(PermissionError):
            CommunicationJournal(self.directory, wrong_pin)
        path.write_bytes(b'not an SLDC container')
        with self.assertRaises(ValueError):
            self.journal.view('reader', self.password)

    def test_unencrypted_and_wrong_key_rejected(self):
        path = self.directory / 'plain.sldc'
        sldc.write_package(path, sldc.new_package('test fixture'))
        with self.assertRaises(ValueError):
            read_encrypted(path)
        encrypted = self.journal.append(self.record())
        wrong_private, _ = sldc.x25519_generate()
        with self.assertRaises(sldc.SLDCDecryptionError):
            read_encrypted(encrypted, private_key=wrong_private)

    @unittest.skipUnless(os.name == 'posix', 'POSIX inode/mode checks; not NTFS ACL evidence')
    def test_private_file_rejects_unsafe_inode(self):
        target = self.directory / 'secret'
        target.touch(mode=0o600)
        target.write_bytes(b'fixture')
        self.assertEqual(_read_private(target, 16), b'fixture')
        with self.assertRaises(ValueError):
            _read_private(target, 2)
        alias = self.directory / 'alias'
        alias.symlink_to(target)
        with self.assertRaises(OSError):
            _read_private(alias, 16)
        alias.unlink()
        os.link(target, alias)
        with self.assertRaises(PermissionError):
            _read_private(target, 16)
        alias.unlink()
        target.chmod(0o644)
        with self.assertRaises(PermissionError):
            _read_private(target, 16)
        fifo = self.directory / 'fifo'
        os.mkfifo(fifo, 0o600)
        with self.assertRaises(PermissionError):
            _read_private(fifo, 16)

    @unittest.skipUnless(os.name == 'posix', 'POSIX symlink and mode regression')
    def test_managed_directory_substitution_rejected(self):
        alias = self.directory.parent / 'alias'
        alias.symlink_to(self.directory, target_is_directory=True)
        with self.assertRaises(PermissionError):
            CommunicationJournal(alias, self.pin)

        accounts = self.directory / 'accounts'
        moved = self.directory / 'accounts-old'
        accounts.rename(moved)
        accounts.symlink_to(moved, target_is_directory=True)
        with self.assertRaises(PermissionError):
            self.journal.view('reader', self.password)
        accounts.unlink()
        moved.rename(accounts)

        record = self.record()
        path = self.journal.append(record)
        epoch_dir = path.parent
        moved_epoch = epoch_dir.with_name('substituted')
        epoch_dir.rename(moved_epoch)
        epoch_dir.symlink_to(moved_epoch, target_is_directory=True)
        with self.assertRaises(PermissionError):
            self.journal.view('reader', self.password)
        with self.assertRaises(PermissionError):
            self.journal.append(self.record())
        epoch_dir.unlink()
        moved_epoch.rename(epoch_dir)
        epoch_dir.chmod(0o755)
        with self.assertRaises(PermissionError):
            self.journal.view('reader', self.password)
        epoch_dir.chmod(0o700)
        self.assertEqual(self.journal.view('reader', self.password)['records'], [record])

    def test_two_processes_cannot_commit_same_message(self):
        ctx = multiprocessing.get_context('spawn')
        start = ctx.Event()
        results = ctx.Queue()
        record = self.record()
        writers = [ctx.Process(target=_append_in_child,
                               args=(str(self.directory), self.pin, record, start, results))
                   for _ in range(2)]
        started = []
        try:
            for writer in writers:
                writer.start()
                started.append(writer)
            start.set()
            outcomes = [results.get(timeout=90) for _ in writers]
            self.assertCountEqual(outcomes, ['committed', 'duplicate'])
            for writer in started:
                writer.join(15)
                self.assertEqual(writer.exitcode, 0)
            self.assertEqual(self.journal.view('reader', self.password)['records'], [record])
        finally:
            for writer in started:
                if writer.is_alive():
                    writer.terminate()
                writer.join(15)
            results.close()
            results.join_thread()


if __name__ == '__main__':
    unittest.main()
