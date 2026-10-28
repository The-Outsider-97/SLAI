"""Storage and security properties, including tamper and session checks."""
from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path

from applications.tools.errors import BadRequest, NotFound, StorageError, Unauthorized
from applications.secure_folder.vault import CHUNK_SIZE, SESSION_IDLE_SECONDS, Vault

PASSWORD = "I have a strong 2026 secure vault passphrase."


class VaultTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "private"
        self.vault = Vault(self.path)
        self.vault.setup(PASSWORD, PASSWORD)
        self.token = self.vault.unlock(PASSWORD)["sessionToken"]

    def test_setup_requires_strong_password(self) -> None:
        with self.assertRaises(BadRequest):
            Vault(Path(self.temporary.name) / "other").setup("1234", "1234")

    def test_password_never_persisted(self) -> None:
        contents = b"".join(p.read_bytes() for p in self.path.rglob("*") if p.is_file())
        self.assertNotIn(PASSWORD.encode(), contents)
        self.assertNotIn(b"entries", contents)

    def test_wrong_password_and_locked_access(self) -> None:
        with self.assertRaises(Unauthorized):
            self.vault.unlock("wrong-passphrase")
        with self.assertRaises(Unauthorized):
            self.vault.snapshot("made-up-token")
        self.vault.lock(self.token)
        with self.assertRaises(Unauthorized):
            self.vault.snapshot(self.token)
        self.assertTrue(self.vault.unlock(PASSWORD)["sessionToken"])

    def test_import_read_reboot_and_integrity(self) -> None:
        secret = b"PRIVATE DOCUMENT\x00" * (CHUNK_SIZE // 15 + 2)
        ticket = self.vault.start_upload(self.token, "secret.txt", None, "text/plain", len(secret))
        for index, offset in enumerate(range(0, len(secret), CHUNK_SIZE)):
            encoded = base64.b64encode(secret[offset:offset + CHUNK_SIZE]).decode("ascii")
            self.vault.upload_chunk(self.token, ticket["uploadId"], index, encoded)
        item = self.vault.finish_upload(self.token, ticket["uploadId"])
        self.assertNotIn(b"PRIVATE DOCUMENT", b"".join(p.read_bytes() for p in self.vault.blobs_dir.rglob("*.enc")))
        read = b"".join(self.vault.read_chunk(self.token, item["id"], i) for i in range(item["chunks"]))
        self.assertEqual(read, secret)
        self.vault.lock_all()
        fresh = Vault(self.path)
        new_token = fresh.unlock(PASSWORD)["sessionToken"]
        self.assertEqual(fresh.snapshot(new_token)["entries"][0]["name"], "secret.txt")
        path = fresh._blob_path(item["blob"], 0)
        modified = bytearray(path.read_bytes())
        modified[-1] ^= 1
        path.write_bytes(modified)
        with self.assertRaises(StorageError):
            fresh.read_chunk(new_token, item["id"], 0)

    def test_folder_note_edit_trash_restore_and_purge(self) -> None:
        folder = self.vault.folder(self.token, "Personal")
        note = self.vault.create_note(self.token, "My diary", "Secret first note", folder["id"])
        self.assertEqual(self.vault.note_text(self.token, note["id"])["text"], "Secret first note")
        self.vault.update_note(self.token, note["id"], "Second private note")
        self.assertEqual(self.vault.note_text(self.token, note["id"])["text"], "Second private note")
        with self.assertRaises(BadRequest):
            self.vault.edit(self.token, folder["id"], {"parent": folder["id"]})
        self.vault.trash(self.token, folder["id"])
        self.assertTrue(next(e for e in self.vault.snapshot(self.token)["entries"] if e["id"] == note["id"])["trashed"])
        with self.assertRaises(NotFound):
            self.vault.note_text(self.token, note["id"])
        self.vault.trash(self.token, folder["id"], restore=True)
        self.assertEqual(self.vault.note_text(self.token, note["id"])["text"], "Second private note")
        self.vault.trash(self.token, folder["id"])
        with self.assertRaises(Unauthorized):
            self.vault.purge(self.token, folder["id"], "wrong")
        self.vault.purge(self.token, folder["id"], PASSWORD)
        self.assertEqual(self.vault.snapshot(self.token)["entries"], [])

    def test_changing_password_revokes_sessions(self) -> None:
        second = self.vault.unlock(PASSWORD)["sessionToken"]
        next_password = "Another stronger 2026 passphrase for my vault."
        self.vault.change_password(self.token, PASSWORD, next_password)
        for token in (self.token, second):
            with self.assertRaises(Unauthorized):
                self.vault.snapshot(token)
        with self.assertRaises(Unauthorized):
            self.vault.unlock(PASSWORD)
        self.assertTrue(self.vault.unlock(next_password)["sessionToken"])

    def test_session_timeout_and_independent_tabs(self) -> None:
        second = self.vault.unlock(PASSWORD)["sessionToken"]
        self.vault.lock(self.token)
        self.assertEqual(self.vault.snapshot(second)["entries"], [])
        self.vault._sessions[second].touched -= SESSION_IDLE_SECONDS + 1
        with self.assertRaises(Unauthorized):
            self.vault.snapshot(second)

    def test_invalid_name_and_chunk_order(self) -> None:
        with self.assertRaises(BadRequest):
            self.vault.folder(self.token, "../secret")
        ticket = self.vault.start_upload(self.token, "a.bin", None, "application/octet-stream", 2)
        with self.assertRaises(BadRequest):
            self.vault.upload_chunk(self.token, ticket["uploadId"], 1, base64.b64encode(b"xx").decode())
        self.vault.cancel_upload(self.token, ticket["uploadId"])


if __name__ == "__main__":
    unittest.main()
