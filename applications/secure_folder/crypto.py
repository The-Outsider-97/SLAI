"""Small, versioned cryptographic boundary for Secure Folder.

Uses cryptography's audited AES-GCM and memory-hard scrypt primitives. There is
no home-grown encryption, no persistent plaintext key, and no recovery bypass.
"""
from __future__ import annotations

import json
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from ..tools.errors import StorageError, Unauthorized

KDF_N = 1 << 15
KDF_R = 8
KDF_P = 1
KEY_LENGTH = 32
NONCE_LENGTH = 12
WRAP_AAD = b"SLAI-SecureFolder/keywrap/v1"
MANIFEST_AAD = b"SLAI-SecureFolder/manifest/v1"


def derive_key(password: str, salt: bytes) -> bytes:
    if not isinstance(password, str) or not password:
        raise Unauthorized("The password or passphrase is incorrect")
    return Scrypt(salt=salt, length=KEY_LENGTH, n=KDF_N, r=KDF_R, p=KDF_P).derive(password.encode("utf-8"))


def seal(key: bytes | bytearray, value: bytes, aad: bytes) -> bytes:
    nonce = os.urandom(NONCE_LENGTH)
    return nonce + AESGCM(bytes(key)).encrypt(nonce, value, aad)


def open_sealed(key: bytes | bytearray, blob: bytes, aad: bytes) -> bytes:
    if len(blob) < NONCE_LENGTH + 16:
        raise StorageError("The encrypted vault data is incomplete")
    try:
        return AESGCM(bytes(key)).decrypt(blob[:NONCE_LENGTH], blob[NONCE_LENGTH:], aad)
    except InvalidTag as exc:
        raise StorageError("Encrypted data failed its integrity check") from exc


def wrap_key(password: str, salt: bytes, master: bytes) -> bytes:
    return seal(derive_key(password, salt), master, WRAP_AAD)


def unwrap_key(password: str, salt: bytes, wrapped: bytes) -> bytes:
    try:
        return open_sealed(derive_key(password, salt), wrapped, WRAP_AAD)
    except StorageError as exc:
        raise Unauthorized("The password or passphrase is incorrect") from exc


def seal_json(key: bytes | bytearray, value: dict) -> bytes:
    return seal(key, json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), MANIFEST_AAD)


def open_json(key: bytes | bytearray, blob: bytes) -> dict:
    try:
        value = json.loads(open_sealed(key, blob, MANIFEST_AAD).decode("utf-8"))
    except (UnicodeError, ValueError, TypeError) as exc:
        raise StorageError("The vault index is corrupted") from exc
    if not isinstance(value, dict) or value.get("version") != 1 or not isinstance(value.get("entries"), dict):
        raise StorageError("The vault index has an unsupported format")
    return value
