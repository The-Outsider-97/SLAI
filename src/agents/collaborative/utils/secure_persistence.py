"""Authenticated (and, when available, encrypted) persistence for SharedMemory (audit 4.8).

Problems with the previous ``pickle.load`` of an arbitrary file:
* loading a tampered file executes attacker-controlled code (deserialisation RCE),
* personal data was written to disk in clear text with no integrity protection,
* there was no schema version or migration path.

This module provides:
* a versioned container ``SLAIMEM`` v2: ``magic | flags | HMAC-SHA256 | body``
  - the HMAC covers magic, flags and body and is verified BEFORE any deserialisation,
  - with the optional ``cryptography`` package the body is AES-256-GCM encrypted (flag bit 0),
    otherwise it is integrity-protected only and the caller is told via :func:`encryption_available`,
* a restricted unpickler that only reconstructs an explicit allowlist of plain-data types,
* keys derived (domain separated) from ``SLAI_SHARED_MEMORY_KEY`` or ``SLAI_GOVERNANCE_MASTER_KEY``.

No key => no persistence (fail closed). Legacy raw-pickle files are NEVER loaded unless the
caller opts in to a one-off migration, and even then only through the restricted unpickler.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import os
import pickle
from collections import OrderedDict, defaultdict, deque
from typing import Any, Optional

MAGIC = b"SLAIMEM\x02"
FLAG_ENCRYPTED = 0x01
_MAC_LEN = 32
_NONCE_LEN = 12
MAX_BYTES = 512 * 1024 * 1024

try:  # optional dependency: present in most deployments, never required
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore
except Exception:  # pragma: no cover - depends on environment
    AESGCM = None  # type: ignore


class PersistenceSecurityError(Exception):
    """Integrity, authenticity, format or key failure. Never contains payload content."""


def encryption_available() -> bool:
    return AESGCM is not None


def _derive(master: bytes, label: str) -> bytes:
    return hmac.new(master, b"slai-shared-memory-v1:" + label.encode("utf-8"), hashlib.sha256).digest()


def load_keys(explicit: Optional[bytes] = None) -> tuple[bytes, bytes]:
    """Return (mac_key, enc_key). Raises when no adequate key material is configured."""
    raw = explicit
    if raw is None:
        text = os.environ.get("SLAI_SHARED_MEMORY_KEY") or os.environ.get("SLAI_GOVERNANCE_MASTER_KEY")
        raw = text.encode("utf-8") if text else None
    if not raw or len(raw) < 32:
        raise PersistenceSecurityError(
            "No persistence key: set SLAI_SHARED_MEMORY_KEY (>= 32 characters). Refusing to persist or load unprotected state."
        )
    return _derive(raw, "mac"), _derive(raw, "enc")


_SAFE_BUILTINS = {
    ("builtins", n) for n in (
        "dict", "list", "tuple", "set", "frozenset", "str", "bytes", "bytearray", "int", "float",
        "complex", "bool", "NoneType", "slice", "range",
    )
}
_SAFE_OTHER = {
    ("collections", "OrderedDict"): OrderedDict,
    ("collections", "defaultdict"): defaultdict,
    ("collections", "deque"): deque,
    ("datetime", "datetime"): None,
    ("datetime", "date"): None,
    ("datetime", "timedelta"): None,
    ("datetime", "timezone"): None,
}


class RestrictedUnpickler(pickle.Unpickler):
    """Only plain data containers and ``VersionedItem`` may be reconstructed."""

    def __init__(self, file, *, versioned_item: Any) -> None:
        super().__init__(file)
        self._versioned_item = versioned_item

    def find_class(self, module: str, name: str):  # noqa: D401
        if (module, name) in _SAFE_BUILTINS:
            import builtins

            return getattr(builtins, name)
        if (module, name) in _SAFE_OTHER:
            if module == "datetime":
                import datetime

                return getattr(datetime, name)
            return _SAFE_OTHER[(module, name)]
        if name == "VersionedItem" and module.endswith("collaborative.shared_memory"):
            return self._versioned_item
        raise PersistenceSecurityError(f"Disallowed type in persisted state: {module}.{name}")


def restricted_loads(data: bytes, *, versioned_item: Any) -> Any:
    try:
        return RestrictedUnpickler(io.BytesIO(data), versioned_item=versioned_item).load()
    except PersistenceSecurityError:
        raise
    except Exception as exc:
        raise PersistenceSecurityError(f"Persisted state could not be decoded ({type(exc).__name__})") from exc


def seal(payload: Any, *, key: Optional[bytes] = None, require_encryption: bool = False) -> bytes:
    mac_key, enc_key = load_keys(key)
    body = pickle.dumps(payload, protocol=4)
    flags = 0
    if encryption_available():
        nonce = os.urandom(_NONCE_LEN)
        flags |= FLAG_ENCRYPTED
        body = nonce + AESGCM(enc_key).encrypt(nonce, body, MAGIC + bytes([flags]))  # type: ignore[misc]
    elif require_encryption:
        raise PersistenceSecurityError("Encryption required but the 'cryptography' package is not installed.")
    header = MAGIC + bytes([flags])
    mac = hmac.new(mac_key, header + body, hashlib.sha256).digest()
    return header + mac + body


def unseal(blob: bytes, *, versioned_item: Any, key: Optional[bytes] = None) -> Any:
    if len(blob) > MAX_BYTES:
        raise PersistenceSecurityError("Persisted state exceeds the maximum permitted size")
    if not blob.startswith(MAGIC):
        raise PersistenceSecurityError("Unrecognised persistence format (legacy/untrusted files are not loaded)")
    if len(blob) < len(MAGIC) + 1 + _MAC_LEN:
        raise PersistenceSecurityError("Truncated persisted state")
    mac_key, enc_key = load_keys(key)
    flags = blob[len(MAGIC)]
    mac = blob[len(MAGIC) + 1: len(MAGIC) + 1 + _MAC_LEN]
    body = blob[len(MAGIC) + 1 + _MAC_LEN:]
    expected = hmac.new(mac_key, MAGIC + bytes([flags]) + body, hashlib.sha256).digest()
    if not hmac.compare_digest(mac, expected):
        raise PersistenceSecurityError("Integrity/authenticity check failed (tampered file or wrong key)")
    if flags & FLAG_ENCRYPTED:
        if not encryption_available():
            raise PersistenceSecurityError("State is encrypted but the 'cryptography' package is not installed")
        nonce, ct = body[:_NONCE_LEN], body[_NONCE_LEN:]
        try:
            body = AESGCM(enc_key).decrypt(nonce, ct, MAGIC + bytes([flags]))  # type: ignore[misc]
        except Exception as exc:
            raise PersistenceSecurityError("Decryption failed") from exc
    return restricted_loads(body, versioned_item=versioned_item)


def migrate_legacy_pickle(blob: bytes, *, versioned_item: Any) -> Any:
    """One-off, explicitly requested migration of a pre-2.0 raw pickle via the restricted unpickler."""
    return restricted_loads(blob, versioned_item=versioned_item)
