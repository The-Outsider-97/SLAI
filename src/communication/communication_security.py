"""Trusted-service credentials, scoped accounts and archive key lifecycle.

Uses native SLDC envelopes/signatures; no custom encryption. Human credentials
never contain archive decryption keys. The service directory is an OS trust boundary.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import stat
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from src.utils import sldc
from .communication_types import KINDS, CLASSIFICATIONS, identifier

STATE_EXTENSION = 'slai.communication-state'
VAULT_EXTENSION = 'slai.communication-vault'
_USER = re.compile(r'[a-z][a-z0-9_-]{0,47}\Z')
_HEX = re.compile(r'[0-9a-f]{64}\Z')
_EPOCH = re.compile(r'[0-9a-f]{32}\Z')
_SCOPE_FIELDS = ('agents', 'sessions', 'kinds', 'classifications')


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def username(value: str) -> str:
    if not isinstance(value, str) or not _USER.fullmatch(value):
        raise ValueError('Username must be lowercase ASCII, 1–48 characters')
    return value


def encode_key(value: bytes) -> str:
    return base64.b64encode(value).decode('ascii')


def decode_key(value: str) -> bytes:
    raw = base64.b64decode(value, validate=True)
    if len(raw) != 32:
        raise ValueError('Invalid key length')
    return raw


def require_trusted(package: Any, key_id: str) -> None:
    results = sldc.verify_package(package, trusted_keys=[key_id])
    if len(results) != 1 or results[0]['state'] != 'valid_trusted':
        raise ValueError('SLDC signature is absent, invalid or untrusted')


def _private_fd(path: Path, flags: int, mode: int = 0o600) -> int:
    """Open a private regular file. POSIX checks apply to the opened inode.

    Windows reparse-point checks are best effort; deployment must enforce NTFS
    ACLs and prevent untrusted modification of the service directory.
    """
    if os.name == 'posix':
        flags |= os.O_NOFOLLOW | os.O_NONBLOCK
    else:
        try:
            info = path.lstat()
        except FileNotFoundError:
            if not flags & os.O_CREAT:
                raise
        else:
            if (not stat.S_ISREG(info.st_mode) or
                    getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
                raise PermissionError('Service file must be a regular, non-reparse file')
    fd = os.open(path, flags | getattr(os, 'O_BINARY', 0), mode)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise PermissionError('Service file must be regular')
        if os.name == 'posix' and (info.st_uid != os.geteuid() or
                                    info.st_mode & 0o077 or info.st_nlink != 1):
            raise PermissionError('Service file must be owner-only and not hard-linked')
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read_private(path: Path, limit: int) -> bytes:
    with os.fdopen(_private_fd(path, os.O_RDONLY), 'rb') as handle:
        size = os.fstat(handle.fileno()).st_size
        if size > limit:
            raise ValueError('Service file exceeds size limit')
        data = handle.read(size + 1)
        if len(data) != size:
            raise ValueError('Service file changed during read')
        return data


def read_encrypted(path: Path, **credentials: Any) -> Any:
    data = _read_private(path, sldc.MAX_PACKAGE_BYTES)
    if sldc.inspect_package(data)['container'] != 'encrypted':
        raise ValueError('Expected encrypted SLDC envelope')
    return sldc.loads_package(data, **credentials)


def write_signed(path: Path, title: str, extension: str, value: dict, signer: Any,
                 recipient: Any, *, overwrite: bool = False) -> Path:
    package = sldc.new_package(title, extensions={extension: value})
    package = sldc.sign_package(package, signer, purpose='authorship')
    return sldc.write_package(path, package, recipients=[recipient], overwrite=overwrite)


def validate_scope(scope: dict) -> dict:
    """All four dimensions required. [] denies; ['*'] explicitly permits all."""
    if type(scope) is not dict or set(scope) != set(_SCOPE_FIELDS):
        raise ValueError('All four scope dimensions must be explicitly specified')
    result = {}
    for key in _SCOPE_FIELDS:
        values = scope[key]
        if type(values) is not list or len(values) > 1024 or any(type(v) is not str for v in values):
            raise ValueError('Invalid scope values')
        if '*' in values and values != ['*']:
            raise ValueError('Wildcard must be the sole value')
        if values != ['*']:
            for value in values:
                identifier(value)
                if key == 'kinds' and value not in KINDS:
                    raise ValueError('Unknown message kind')
                if key == 'classifications' and value not in CLASSIFICATIONS:
                    raise ValueError('Unknown classification')
        result[key] = sorted(set(values))
    return result


def scope_allows(scope: dict, record: dict) -> bool:
    def match(field: str, *values: str) -> bool:
        allowed = scope[field]
        return allowed == ['*'] or any(value in allowed for value in values)
    return (match('agents', record['sender'], record['recipient'])
            and match('sessions', record['session_id'])
            and match('kinds', record['kind'])
            and match('classifications', record['classification']))


def _token_hash(token: bytes) -> str:
    return hashlib.sha256(token).hexdigest()


def _password(password: str) -> None:
    if not isinstance(password, str) or len(password) < 14 or len(password) > 1024:
        raise ValueError('Use a unique passphrase of 14–1024 characters')
    sldc.password_recipient(password)


def _new_epoch() -> tuple[str, dict]:
    private, public = sldc.x25519_generate()
    audit_private, audit_public = sldc.x25519_generate()
    seed = os.urandom(32)
    signer = sldc.Ed25519Signer.from_seed(seed)
    return uuid4().hex, dict(status='active', created_at=utc(), private_key=encode_key(private),
                            public_key=encode_key(public), audit_private_key=encode_key(audit_private),
                            audit_public_key=encode_key(audit_public), signing_seed=encode_key(seed),
                            signer_key_id=signer.key_id, ended_at=None)


class KeyStore:
    """Service-only state. Every operation must acquire locked() and reload state.

    Cross-process advisory lock (all cooperating writers/readers use this class).
    Root credentials remain in a protected machine file; deployment must isolate OS identities.
    """
    def __init__(self, directory: Path, trusted_root: str):
        self.directory = Path(directory).resolve()
        if not isinstance(trusted_root, str) or not _HEX.fullmatch(trusted_root):
            raise ValueError('An independent root fingerprint is required')
        if not self.directory.is_dir():
            raise ValueError('Service directory does not exist')
        if os.name == 'posix':
            info = self.directory.stat()
            if info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise PermissionError('Service directory must be owned by the service and OS-private')
        path = self.directory / 'service.env'
        expected = {'SLAI_COMM_ROOT_SEED', 'SLAI_COMM_STATE_PRIVATE', 'SLAI_COMM_STATE_PUBLIC'}
        values = {}
        for line in _read_private(path, 4096).decode('ascii').splitlines():
            key, sep, value = line.partition('=')
            if not sep or key not in expected or key in values:
                raise ValueError('Invalid service credentials')
            values[key] = decode_key(value)
        if set(values) != expected:
            raise ValueError('Missing service credentials')
        self.root_signer = sldc.Ed25519Signer.from_seed(values['SLAI_COMM_ROOT_SEED'])
        if not hmac.compare_digest(self.root_signer.key_id, trusted_root):
            raise PermissionError('Root fingerprint mismatch')
        self.root_pin = trusted_root
        self._state_private = values['SLAI_COMM_STATE_PRIVATE']
        self._state_recipient = sldc.x25519_recipient(values['SLAI_COMM_STATE_PUBLIC'])

    @classmethod
    def initialize(cls, directory: Path) -> str:
        directory = Path(directory).resolve()
        directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        seed = os.urandom(32)
        private, public = sldc.x25519_generate()
        pin = sldc.Ed25519Signer.from_seed(seed).key_id
        fd = os.open(directory / 'service.env', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w', encoding='ascii') as handle:
            handle.write(f'SLAI_COMM_ROOT_SEED={encode_key(seed)}\n'
                         f'SLAI_COMM_STATE_PRIVATE={encode_key(private)}\n'
                         f'SLAI_COMM_STATE_PUBLIC={encode_key(public)}\n')
            handle.flush()
            os.fsync(handle.fileno())
        for name in ('accounts', 'journal', 'audit'):
            (directory / name).mkdir(mode=0o700)
        store = cls(directory, pin)
        epoch, entry = _new_epoch()
        store.save(dict(schema='slai.communication-state.v2', revision=0,
                        current_epoch=epoch, epochs={epoch: entry}, accounts={}), initial=True)
        return pin

    @contextmanager
    def locked(self) -> Iterator[None]:
        fd = _private_fd(self.directory / '.lock', os.O_CREAT | os.O_RDWR)
        with os.fdopen(fd, 'r+b', buffering=0) as handle:
            if os.name == 'nt':
                import msvcrt
                # Lock a byte region, including an empty file; never truncate/replace the lock file.
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            elif os.name == 'posix':
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            else:
                raise OSError('Unsupported locking platform')

    def load(self) -> dict:
        package = read_encrypted(self.directory / 'state.sldc', private_key=self._state_private)
        require_trusted(package, self.root_pin)
        state = package.document['extensions'][STATE_EXTENSION]
        if state['schema'] != 'slai.communication-state.v2' or type(state['revision']) is not int or state['revision'] < 0:
            raise ValueError('Unsupported key state')
        active = []
        for epoch, entry in state['epochs'].items():
            if not _EPOCH.fullmatch(epoch) or entry['status'] not in ('active', 'retired', 'revoked'):
                raise ValueError('Invalid epoch state')
            decode_key(entry['private_key']); decode_key(entry['public_key'])
            decode_key(entry['audit_private_key']); decode_key(entry['audit_public_key'])
            if not _HEX.fullmatch(entry['signer_key_id']):
                raise ValueError('Invalid epoch signer')
            if entry['status'] == 'active':
                active.append(epoch)
                if sldc.Ed25519Signer.from_seed(decode_key(entry['signing_seed'])).key_id != entry['signer_key_id']:
                    raise ValueError('Active signer mismatch')
            elif entry['signing_seed'] is not None:
                raise ValueError('Inactive signing seed must not be retained')
        if active != [state['current_epoch']]:
            raise ValueError('Exactly one current epoch required')
        for user, account in state['accounts'].items():
            username(user)
            if account['status'] not in ('active', 'revoked') or not _EPOCH.fullmatch(account['generation']) or not _HEX.fullmatch(account['token_hash']):
                raise ValueError('Invalid account state')
            validate_scope(account['scope'])
        return state

    def save(self, state: dict, *, initial: bool = False) -> None:
        write_signed(self.directory / 'state.sldc', 'SLAI communication service state',
                     STATE_EXTENSION, state, self.root_signer, self._state_recipient,
                     overwrite=not initial)

    def issue(self, state: dict, user: str, password: str, scope: dict) -> None:
        """Prepare immutable password vault before committing state. Old generations become invalid."""
        user = username(user); _password(password); scope = validate_scope(scope)
        token = os.urandom(32); generation = uuid4().hex
        value = dict(schema='slai.communication-account.v2', username=user,
                     generation=generation, token=encode_key(token))
        write_signed(self.directory / 'accounts' / f'{user}-{generation}.sldc',
                     'SLAI communication account', VAULT_EXTENSION, value, self.root_signer,
                     sldc.password_recipient(password))
        state['accounts'][user] = dict(status='active', generation=generation,
                                       token_hash=_token_hash(token), scope=scope)

    def authenticate(self, state: dict, user: str, password: str) -> dict:
        try:
            user = username(user)
            _password(password)
            account = state['accounts'][user]
            if account['status'] != 'active':
                raise ValueError('Inactive account')
            generation = account['generation']
            package = read_encrypted(self.directory / 'accounts' / f'{user}-{generation}.sldc', password=password)
            require_trusted(package, self.root_pin)
            value = package.document['extensions'][VAULT_EXTENSION]
            if (value['schema'] != 'slai.communication-account.v2' or value['username'] != user
                    or value['generation'] != generation or not hmac.compare_digest(
                        _token_hash(decode_key(value['token'])), account['token_hash'])):
                raise ValueError('Account credential mismatch')
            return validate_scope(account['scope'])
        except (OSError, ValueError, KeyError, TypeError):
            raise PermissionError('Communication authentication failed') from None

    def rotate(self, state: dict, *, revoke_current: bool = False) -> str:
        previous = state['epochs'][state['current_epoch']]
        previous.update(status='revoked' if revoke_current else 'retired', signing_seed=None, ended_at=utc())
        epoch, entry = _new_epoch()
        state['epochs'][epoch] = entry
        state['current_epoch'] = epoch
        return epoch

    def revoke_epoch(self, state: dict, epoch: str) -> None:
        if epoch == state['current_epoch']:
            self.rotate(state, revoke_current=True)
        else:
            entry = state['epochs'][epoch]
            if entry['status'] == 'revoked':
                raise ValueError('Epoch already revoked')
            entry.update(status='revoked', signing_seed=None, ended_at=utc())
