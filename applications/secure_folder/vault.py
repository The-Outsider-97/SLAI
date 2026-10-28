"""Encrypted vault storage and short-lived, independent unlock sessions.

On-disk paths and filenames contain random IDs, never user-supplied names.
The index, file contents, notes and all user metadata are authenticated and
encrypted. No endpoint may read any of them without a live vault session.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..tools.errors import BadRequest, Conflict, NotFound, StorageError, Unauthorized
from ..tools.storage import atomic_write, atomic_write_json
from .crypto import derive_key, open_json, open_sealed, seal, seal_json, unwrap_key, wrap_key

CHUNK_SIZE = 512 * 1024
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024
NOTE_LIMIT = 200 * 1024
SESSION_IDLE_SECONDS = 5 * 60
SESSION_ABSOLUTE_SECONDS = 30 * 60
ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _id() -> str:
    return secrets.token_hex(16)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 160 or any(ord(c) < 32 for c in value):
        raise BadRequest("Name must contain 1–160 printable characters")
    if value in (".", "..") or "/" in value or "\\" in value:
        raise BadRequest("Name cannot contain path separators")
    return value.strip()


def _strong_password(value: Any) -> str:
    if not isinstance(value, str) or len(value) < 12 or len(value) > 1024:
        raise BadRequest("Use a password or passphrase between 12 and 1024 characters")
    if value.isdigit() or value.isalpha() and len(value) < 20:
        raise BadRequest("Choose a less predictable passphrase; short PINs are not supported")
    return value


@dataclass
class Upload:
    blob: str
    name: str
    parent: str | None
    mime: str
    size: int
    received: int = 0
    chunks: int = 0
    digest: Any = field(default_factory=hashlib.sha256)


@dataclass
class Session:
    key: bytearray
    started: float
    touched: float
    uploads: dict[str, Upload] = field(default_factory=dict)


class Vault:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.header_path = self.data_dir / "vault.json"
        self.index_path = self.data_dir / "index.enc"
        self.blobs_dir = self.data_dir / "blobs"
        self._lock = threading.RLock()
        self._sessions: dict[str, Session] = {}
        self._failures = 0
        self._next_attempt = 0.0
        self._instance_lock: Any = None

    def acquire_instance_lock(self) -> None:
        """Prevent two local servers from writing the same encrypted index concurrently."""
        if self._instance_lock is not None:
            return
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._protect_path(self.data_dir)
        handle = open(self.data_dir / ".server.lock", "a+b")
        try:
            self._protect_path(self.data_dir / ".server.lock")
            if os.name == "nt":
                import msvcrt
                if handle.seek(0, 2) == 0:
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise Conflict("Another Secure Folder instance is already using this vault") from exc
        self._instance_lock = handle

    def close(self) -> None:
        self.lock_all()
        handle = self._instance_lock
        self._instance_lock = None
        if handle is not None:
            handle.close()  # closing releases OS-held advisory lock

    @property
    def initialized(self) -> bool:
        return self.header_path.is_file()

    def status(self) -> dict[str, Any]:
        return {"initialized": self.initialized, "locked": True,
                "cipher": "AES-256-GCM", "kdf": "scrypt", "pinSupported": False,
                "idleSeconds": SESSION_IDLE_SECONDS, "absoluteSeconds": SESSION_ABSOLUTE_SECONDS,
                "platform": "local-browser", "onlineSync": False}

    def _header(self) -> dict[str, Any]:
        try:
            h = json.loads(self.header_path.read_text(encoding="utf-8"))
            if h.get("version") != 1 or h.get("kdf") != "scrypt-v1":
                raise ValueError("unknown version")
            return h
        except (OSError, ValueError, AttributeError) as exc:
            raise StorageError("The vault header cannot be read") from exc

    def _read_master(self, password: str) -> bytes:
        h = self._header()
        try:
            return unwrap_key(password, base64.b64decode(h["salt"], validate=True),
                              base64.b64decode(h["wrapped"], validate=True))
        except (ValueError, KeyError, binascii.Error) as exc:
            raise StorageError("The vault header is damaged") from exc

    @staticmethod
    def _protect_path(path: Path) -> None:
        if os.name != "nt":
            path.chmod(0o700 if path.is_dir() else 0o600)

    def setup(self, password: str, confirmation: str) -> dict[str, Any]:
        password = _strong_password(password)
        if password != confirmation:
            raise BadRequest("The two passphrases do not match")
        with self._lock:
            if self.initialized:
                raise Conflict("A vault already exists")
            self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.blobs_dir.mkdir(exist_ok=True, mode=0o700)
            self._protect_path(self.data_dir)
            self._protect_path(self.blobs_dir)
            key = bytearray(secrets.token_bytes(32))
            try:
                salt = secrets.token_bytes(16)
                wrapped = wrap_key(password, salt, bytes(key))
                atomic_write(self.index_path, seal_json(key, {"version": 1, "entries": {}}))
                self._protect_path(self.index_path)
                atomic_write_json(self.header_path, {"version": 1, "kdf": "scrypt-v1",
                                                          "salt": base64.b64encode(salt).decode("ascii"),
                                                          "wrapped": base64.b64encode(wrapped).decode("ascii")})
                self._protect_path(self.header_path)
            finally:
                for i in range(len(key)):
                    key[i] = 0
            return {"initialized": True, "locked": True}

    def _expire(self) -> None:
        now = time.monotonic()
        for token, s in list(self._sessions.items()):
            if now - s.touched > SESSION_IDLE_SECONDS or now - s.started > SESSION_ABSOLUTE_SECONDS:
                self._revoke(token)

    def _revoke(self, token: str) -> None:
        s = self._sessions.pop(token, None)
        if s:
            for i in range(len(s.key)):
                s.key[i] = 0
            s.uploads.clear()

    def unlock(self, password: str) -> dict[str, Any]:
        with self._lock:
            self._expire()
            if not self.initialized:
                raise Conflict("Set up your vault before unlocking")
            if time.monotonic() < self._next_attempt:
                raise Unauthorized("Too many attempts. Wait before trying again")
            try:
                master = self._read_master(password)
                self._load(master)  # detect corrupted index before granting access
            except Unauthorized:
                self._failures += 1
                if self._failures >= 5:
                    self._next_attempt = time.monotonic() + min(30.0, 2 ** min(self._failures - 5, 5))
                raise
            self._failures = 0
            self._next_attempt = 0.0
            token = secrets.token_urlsafe(32)
            now = time.monotonic()
            self._sessions[token] = Session(bytearray(master), now, now)
            return {"sessionToken": token, "idleSeconds": SESSION_IDLE_SECONDS,
                    "absoluteSeconds": SESSION_ABSOLUTE_SECONDS}

    def lock(self, token: str) -> dict[str, bool]:
        with self._lock:
            self._revoke(token)
            return {"locked": True}

    def lock_all(self) -> None:
        with self._lock:
            for token in list(self._sessions):
                self._revoke(token)

    def reauthenticate(self, token: str, password: str) -> dict[str, bool]:
        """Step-up validation for explicit plaintext exports; never stores the password."""
        with self._lock:
            s = self._session(token)
            if not hmac.compare_digest(self._read_master(password), bytes(s.key)):
                raise Unauthorized("Your passphrase was not accepted")
            return {"verified": True}

    def _session(self, token: str | None) -> Session:
        self._expire()
        if not token or token not in self._sessions:
            raise Unauthorized("Your vault is locked. Enter your passphrase to unlock it")
        session = self._sessions[token]
        session.touched = time.monotonic()
        return session

    def _load(self, key: bytes | bytearray) -> dict[str, Any]:
        try:
            return open_json(key, self.index_path.read_bytes())
        except OSError as exc:
            raise StorageError("The encrypted vault index cannot be read") from exc

    def _save(self, key: bytearray, manifest: dict[str, Any]) -> None:
        atomic_write(self.index_path, seal_json(key, manifest))
        self._protect_path(self.index_path)

    @staticmethod
    def _entry(manifest: dict[str, Any], entry_id: str) -> dict[str, Any]:
        entry = manifest["entries"].get(entry_id)
        if not isinstance(entry, dict):
            raise NotFound("That item was not found")
        return entry

    @staticmethod
    def _parent(manifest: dict[str, Any], parent: Any) -> str | None:
        if parent in (None, "", "root"):
            return None
        if not isinstance(parent, str):
            raise BadRequest("The parent folder is invalid")
        item = Vault._entry(manifest, parent)
        if item["kind"] != "folder" or item["trashed"]:
            raise BadRequest("The destination must be an active folder")
        return parent

    def snapshot(self, token: str) -> dict[str, Any]:
        with self._lock:
            session = self._session(token)
            m = self._load(session.key)
            return {"entries": list(m["entries"].values()), "now": _now()}

    def folder(self, token: str, name: str, parent: Any = None) -> dict[str, Any]:
        with self._lock:
            s = self._session(token)
            m = self._load(s.key)
            now = _now()
            entry = {"id": _id(), "kind": "folder", "parent": self._parent(m, parent), "name": _name(name),
                     "size": 0, "mime": "", "favorite": False, "trashed": False, "created": now, "updated": now}
            m["entries"][entry["id"]] = entry
            self._save(s.key, m)
            return entry

    def start_upload(self, token: str, name: str, parent: Any, mime: str, size: int) -> dict[str, Any]:
        with self._lock:
            s = self._session(token)
            m = self._load(s.key)
            if type(size) is not int or size < 0 or size > MAX_FILE_SIZE:
                raise BadRequest("File size is not supported")
            if len(s.uploads) >= 4:
                raise Conflict("Finish an existing upload before starting another")
            upload = Upload(_id(), _name(name), self._parent(m, parent),
                            mime[:120] if isinstance(mime, str) else "application/octet-stream", size)
            upload_id = _id()
            s.uploads[upload_id] = upload
            return {"uploadId": upload_id, "chunkSize": CHUNK_SIZE}

    def _blob_path(self, blob: str, index: int) -> Path:
        if not ID_RE.fullmatch(blob) or type(index) is not int or index < 0 or index > 65536:
            raise BadRequest("The encrypted file reference is invalid")
        return self.blobs_dir / blob / f"{index:08d}.enc"

    def upload_chunk(self, token: str, upload_id: str, index: int, encoded: str) -> dict[str, int]:
        with self._lock:
            s = self._session(token)
            u = s.uploads.get(upload_id)
            if not u or type(index) is not int or index != u.chunks:
                raise BadRequest("Upload chunks must arrive in order")
            if not isinstance(encoded, str):
                raise BadRequest("Missing file data")
            try:
                raw = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise BadRequest("Invalid file data") from exc
            if len(raw) == 0 or len(raw) > CHUNK_SIZE or u.received + len(raw) > u.size:
                raise BadRequest("The upload chunk has an invalid length")
            folder = self.blobs_dir / u.blob
            folder.mkdir(mode=0o700, parents=True, exist_ok=True)
            self._protect_path(folder)
            aad = f"SLAI-SecureFolder/chunk/v1/{u.blob}/{index}".encode("ascii")
            path = self._blob_path(u.blob, index)
            atomic_write(path, seal(s.key, raw, aad))
            self._protect_path(path)
            u.digest.update(raw)
            u.received += len(raw)
            u.chunks += 1
            return {"received": u.received, "size": u.size}

    def finish_upload(self, token: str, upload_id: str) -> dict[str, Any]:
        with self._lock:
            s = self._session(token)
            u = s.uploads.get(upload_id)
            if not u or u.received != u.size:
                raise BadRequest("The upload is incomplete")
            m = self._load(s.key)
            self._parent(m, u.parent)
            now = _now()
            entry = {"id": _id(), "kind": "file", "parent": u.parent, "name": u.name, "size": u.size,
                     "mime": u.mime, "blob": u.blob, "chunks": u.chunks, "sha256": u.digest.hexdigest(),
                     "favorite": False, "trashed": False, "created": now, "updated": now}
            m["entries"][entry["id"]] = entry
            self._save(s.key, m)
            del s.uploads[upload_id]
            return entry

    def cancel_upload(self, token: str, upload_id: str) -> dict[str, bool]:
        with self._lock:
            s = self._session(token)
            u = s.uploads.pop(upload_id, None)
            if u:
                shutil.rmtree(self.blobs_dir / u.blob, ignore_errors=True)
            return {"cancelled": True}

    def read_chunk(self, token: str, entry_id: str, index: int) -> bytes:
        with self._lock:
            s = self._session(token)
            m = self._load(s.key)
            e = self._entry(m, entry_id)
            if e["kind"] not in ("file", "note") or e["trashed"] or type(index) is not int or index < 0 or index >= e["chunks"]:
                raise NotFound("That file chunk is unavailable")
            blob = e["blob"]
            aad = f"SLAI-SecureFolder/chunk/v1/{blob}/{index}".encode("ascii")
            try:
                return open_sealed(s.key, self._blob_path(blob, index).read_bytes(), aad)
            except OSError as exc:
                raise StorageError("An encrypted file chunk could not be read") from exc

    def create_note(self, token: str, name: str, content: str, parent: Any = None) -> dict[str, Any]:
        if not isinstance(content, str) or len(content.encode("utf-8")) > NOTE_LIMIT:
            raise BadRequest("The note is too large")
        with self._lock:
            s = self._session(token)
            m = self._load(s.key)
            safe_parent = self._parent(m, parent)
            safe_name = _name(name)
            blob = _id()
            body = content.encode("utf-8")
            path = self._blob_path(blob, 0)
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            self._protect_path(path.parent)
            atomic_write(path, seal(s.key, body, f"SLAI-SecureFolder/chunk/v1/{blob}/0".encode("ascii")))
            self._protect_path(path)
            now = _now()
            e = {"id": _id(), "kind": "note", "parent": safe_parent, "name": safe_name,
                 "size": len(body), "mime": "text/plain", "blob": blob, "chunks": 1,
                 "sha256": hashlib.sha256(body).hexdigest(), "favorite": False, "trashed": False,
                 "created": now, "updated": now}
            m["entries"][e["id"]] = e
            self._save(s.key, m)
            return e

    def note_text(self, token: str, entry_id: str) -> dict[str, str]:
        with self._lock:
            s = self._session(token)
            e = self._entry(self._load(s.key), entry_id)
            if e["kind"] != "note" or e["trashed"]:
                raise NotFound("The note was not found")
            return {"text": self.read_chunk(token, entry_id, 0).decode("utf-8")}

    def update_note(self, token: str, entry_id: str, content: str) -> dict[str, Any]:
        if not isinstance(content, str) or len(content.encode("utf-8")) > NOTE_LIMIT:
            raise BadRequest("The note is too large")
        with self._lock:
            s = self._session(token)
            m = self._load(s.key)
            e = self._entry(m, entry_id)
            if e["kind"] != "note" or e["trashed"]:
                raise NotFound("The note was not found")
            old_blob, new_blob = e["blob"], _id()
            body = content.encode("utf-8")
            path = self._blob_path(new_blob, 0)
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            self._protect_path(path.parent)
            atomic_write(path, seal(s.key, body, f"SLAI-SecureFolder/chunk/v1/{new_blob}/0".encode("ascii")))
            self._protect_path(path)
            e.update(blob=new_blob, size=len(body), sha256=hashlib.sha256(body).hexdigest(), updated=_now())
            self._save(s.key, m)
            shutil.rmtree(self.blobs_dir / old_blob, ignore_errors=True)
            return e

    def edit(self, token: str, entry_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        if not changes or any(k not in ("name", "parent", "favorite") for k in changes):
            raise BadRequest("Unsupported item changes")
        with self._lock:
            s = self._session(token)
            m = self._load(s.key)
            e = self._entry(m, entry_id)
            if e["trashed"]:
                raise BadRequest("Restore the item before editing")
            if "name" in changes:
                e["name"] = _name(changes["name"])
            if "favorite" in changes:
                if type(changes["favorite"]) is not bool:
                    raise BadRequest("Favorite must be true or false")
                e["favorite"] = changes["favorite"]
            if "parent" in changes:
                parent = self._parent(m, changes["parent"])
                ancestor = parent
                while ancestor:
                    if ancestor == entry_id:
                        raise BadRequest("A folder cannot be moved inside itself")
                    ancestor = self._entry(m, ancestor)["parent"]
                e["parent"] = parent
            e["updated"] = _now()
            self._save(s.key, m)
            return e

    @staticmethod
    def _subtree(entries: dict[str, dict[str, Any]], entry_id: str) -> set[str]:
        result = {entry_id}
        changed = True
        while changed:
            old = len(result)
            result.update(key for key, e in entries.items() if e.get("parent") in result)
            changed = len(result) != old
        return result

    def trash(self, token: str, entry_id: str, *, restore: bool = False) -> dict[str, bool]:
        with self._lock:
            s = self._session(token)
            m = self._load(s.key)
            selected = self._entry(m, entry_id)
            for key in self._subtree(m["entries"], entry_id):
                m["entries"][key]["trashed"] = not restore
            if restore:
                ancestor = selected["parent"]
                while ancestor:
                    parent_entry = self._entry(m, ancestor)
                    parent_entry["trashed"] = False
                    ancestor = parent_entry["parent"]
            self._save(s.key, m)
            return {"trashed": not restore}

    def purge(self, token: str, entry_id: str, password: str) -> dict[str, bool]:
        with self._lock:
            s = self._session(token)
            if not hmac.compare_digest(self._read_master(password), bytes(s.key)):
                raise Unauthorized("Your passphrase was not accepted")
            m = self._load(s.key)
            e = self._entry(m, entry_id)
            if not e["trashed"]:
                raise BadRequest("Move the item to Trash before permanently deleting it")
            affected = self._subtree(m["entries"], entry_id)
            blobs = [m["entries"][key]["blob"] for key in affected if "blob" in m["entries"][key]]
            for key in affected:
                del m["entries"][key]
            self._save(s.key, m)
            for blob in blobs:
                shutil.rmtree(self.blobs_dir / blob, ignore_errors=True)
            return {"deleted": True}

    def change_password(self, token: str, old_password: str, new_password: str) -> dict[str, bool]:
        new_password = _strong_password(new_password)
        with self._lock:
            s = self._session(token)
            if not hmac.compare_digest(self._read_master(old_password), bytes(s.key)):
                raise Unauthorized("Your current passphrase is incorrect")
            salt = secrets.token_bytes(16)
            wrapped = wrap_key(new_password, salt, bytes(s.key))
            atomic_write_json(self.header_path, {"version": 1, "kdf": "scrypt-v1",
                                                      "salt": base64.b64encode(salt).decode("ascii"),
                                                      "wrapped": base64.b64encode(wrapped).decode("ascii")})
            self._protect_path(self.header_path)
            self.lock_all()  # all browsers must enter the new passphrase
            return {"locked": True}
