"""Native SLDC release regressions; no cryptographic test doubles."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.communication.communication_security import _read_private, read_encrypted
from src.communication.communication_storage import CommunicationJournal
from src.communication.communication_types import new_message
from src.utils import sldc


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


if __name__ == '__main__':
    unittest.main()
