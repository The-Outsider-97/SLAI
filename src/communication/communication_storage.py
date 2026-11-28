"""Trusted communication service: native encrypted journal, scoped reads and access audit."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from src.utils import sldc
from .communication_security import (KeyStore, _private_directory, decode_key, read_encrypted, require_trusted,
                                     operator_principal, scope_allows, sync_directory, username, utc, validate_scope, write_signed)
from .communication_types import CLASSIFICATIONS, EXTENSION, KINDS, identifier, message_id, validate_message

AUDIT_EXTENSION = 'slai.communication-audit'
EPOCH_EXTENSION = 'slai.communication-epoch'
MAX_RESULTS = 1000


class CommunicationJournal:
    """Trusted-process facade. Never give this object or filesystem access to untrusted users.

    All operations share the KeyStore lock. An authenticated service adapter must expose only
    view() to human clients; administrative methods require trusted OS/operator identity.
    """
    def __init__(self, directory: Path, trusted_root: str, *, operator_id: str | None = None):
        self._operator_id = operator_principal()
        if operator_id is not None and operator_id != self._operator_id:
            raise PermissionError('Operator must match the effective OS identity')
        self.store = KeyStore(directory, trusted_root)
        self.directory = self.store.directory

    @classmethod
    def initialize(cls, directory: Path) -> str:
        return KeyStore.initialize(directory)

    def _folder(self, area: str, epoch: str, *, create: bool = True) -> Path:
        if area not in ('journal', 'audit'):
            raise ValueError('Invalid communication storage area')
        message_id(epoch)
        parent = self.directory / area
        _private_directory(parent)
        path = parent / epoch
        if create:
            path.mkdir(mode=0o700, exist_ok=True)
            sync_directory(parent)
        try:
            _private_directory(path)
        except FileNotFoundError:
            if create:
                raise
            # Epoch directories are lazily created. Missing-history detection
            # requires a separate integrity anchor; do not infer completeness.
        return path

    def _packages(self, area: str, epoch: str) -> list[Path]:
        folder = self._folder(area, epoch, create=False)
        try:
            return sorted(path for path in folder.iterdir() if path.suffix == '.sldc')
        except FileNotFoundError:
            # Epoch folders are created on first write under the service lock.
            return []

    def _audit(self, state: dict, actor: str, action: str, status: str, *,
               reason: str = '', details: dict | None = None, operation_id: str | None = None) -> None:
        """Mandatory durable append before disclosure. No passwords, payloads or raw queries."""
        epoch = state['current_epoch']; entry = state['epochs'][epoch]
        event_id = uuid4().hex
        event = dict(schema='slai.communication-audit.v2', id=event_id, timestamp=utc(),
                     actor=actor, action=action, status=status, reason=reason,
                     operation_id=operation_id or uuid4().hex, revision=state['revision'],
                     epoch=epoch, details=details or {})
        write_signed(self._folder('audit', epoch) / f'{event_id}.sldc', 'SLAI access audit',
                     AUDIT_EXTENSION, event, self.store.root_signer,
                     sldc.x25519_recipient(decode_key(entry['audit_public_key'])))

    def _change(self, action: str, target: str, mutate: Callable[[dict], Any]) -> Any:
        with self.store.locked():
            state = self.store.load(); op = uuid4().hex
            self._audit(state, self._operator_id, action, 'intent', details={'target': target}, operation_id=op)
            try:
                result = mutate(state)
            except Exception:
                # Reload committed state rather than recording with uncommitted keys.
                self._audit(self.store.load(), self._operator_id, action, 'failed',
                            reason='change_rejected', details={'target': target}, operation_id=op)
                raise
            state['revision'] += 1
            # If save or final audit raises, intent remains. Inspect committed status before retrying.
            try:
                self.store.save(state)
            except Exception:
                # Native atomic replacement may have succeeded before an I/O error was reported.
                # Record uncertainty using whichever state is actually readable; never retry implicitly.
                try:
                    self._audit(self.store.load(), self._operator_id, action, 'uncertain',
                                reason='state_commit_failed_or_uncertain', details={'target': target}, operation_id=op)
                except Exception:
                    pass  # The durable intent remains; host must investigate.
                raise
            self._audit(state, self._operator_id, action, 'committed',
                        details={'target': target}, operation_id=op)
            return result

    def add_user(self, user: str, password: str, scope: dict) -> None:
        user = username(user); scope = validate_scope(scope)
        def mutate(state: dict) -> None:
            if user in state['accounts']:
                raise ValueError('Account exists; explicitly rotate credentials to reactivate')
            self.store.issue(state, user, password, scope)
        self._change('add_user', user, mutate)

    def rotate_password(self, user: str, password: str, *, reactivate: bool = False) -> None:
        user = username(user)
        def mutate(state: dict) -> None:
            account = state['accounts'][user]
            if account['status'] != 'active' and not reactivate:
                raise PermissionError('Explicit reactivation required')
            self.store.issue(state, user, password, account['scope'])
        self._change('reactivate_user' if reactivate else 'rotate_password', user, mutate)

    def revoke_user(self, user: str) -> None:
        user = username(user)
        def mutate(state: dict) -> None:
            account = state['accounts'][user]
            if account['status'] != 'active':
                raise ValueError('Account already revoked')
            account['status'] = 'revoked'
        self._change('revoke_user', user, mutate)

    def set_scope(self, user: str, scope: dict) -> None:
        user = username(user); scope = validate_scope(scope)
        def mutate(state: dict) -> None:
            state['accounts'][user]['scope'] = scope
        self._change('set_scope', user, mutate)

    def rotate_keys(self) -> str:
        return self._change('rotate_keys', 'current', self.store.rotate)

    def revoke_epoch(self, epoch: str) -> None:
        message_id(epoch)
        self._change('revoke_epoch', epoch, lambda state: self.store.revoke_epoch(state, epoch))

    def status(self) -> dict:
        """Operator-only; never returns private keys, tokens or password vaults."""
        with self.store.locked():
            state = self.store.load()
            result = dict(revision=state['revision'], current_epoch=state['current_epoch'],
                          epochs={k: {f: v[f] for f in ('status', 'created_at', 'ended_at', 'signer_key_id')}
                                  for k,v in state['epochs'].items()},
                          accounts={k: {f: v[f] for f in ('status', 'scope')} for k,v in state['accounts'].items()})
            self._audit(state, self._operator_id, 'status', 'allowed')
            return result

    def append(self, record: dict[str, Any]) -> Path:
        record = validate_message(record)
        with self.store.locked():
            state = self.store.load(); epoch = state['current_epoch']; entry = state['epochs'][epoch]
            if any((self._folder('journal', old, create=False) / f"{record['id']}.sldc").exists()
                   for old in state['epochs']):
                raise FileExistsError('Duplicate communication ID')
            signer = sldc.Ed25519Signer.from_seed(decode_key(entry['signing_seed']))
            package = sldc.new_package('SLAI communication record', extensions={EXTENSION: record,
                                        EPOCH_EXTENSION: {'epoch': epoch}})
            package = sldc.sign_package(package, signer, purpose='authorship')
            path = sldc.write_package(self._folder('journal', epoch) / f"{record['id']}.sldc", package,
                                      recipients=[sldc.x25519_recipient(decode_key(entry['public_key']))],
                                      overwrite=False)
            sync_directory(path.parent)
            return path

    def _read_record(self, path: Path, epoch: str, entry: dict) -> dict:
        message_id(path.stem)
        if entry['status'] == 'revoked':
            raise PermissionError('Epoch revoked')
        package = read_encrypted(path, private_key=decode_key(entry['private_key']))
        require_trusted(package, entry['signer_key_id'])
        ext = package.document['extensions']
        record = validate_message(ext[EXTENSION])
        if record['id'] != path.stem or ext[EPOCH_EXTENSION] != {'epoch': epoch}:
            raise ValueError('Record identity/epoch mismatch')
        return record

    def view(self, user: str, password: str, *, agent: str | None = None,
             session: str | None = None, kind: str | None = None,
             sender: str | None = None, recipient: str | None = None,
             task: str | None = None, correlation: str | None = None,
             classification: str | None = None, record_id: str | None = None,
             limit: int = 100, after: str | None = None,
             local_reader: bool = False,
             authorize_read: Callable[[str, dict], bool] | None = None) -> dict:
        """Authenticate on every request; scope AND query AND optional governance policy.

        Return at most limit verified records after successful audit. The cursor is an opaque
        epoch/id position, not chronological. No denied-record counts are exposed.
        An adapter may supply a policy callback; it receives only already-scoped records.
        """
        with self.store.locked():
            state = self.store.load(); op = uuid4().hex
            try:
                actor = username(user)
            except ValueError:
                actor = '<invalid>'
            try:
                authenticate = self.store.authenticate_local_reader if local_reader else self.store.authenticate
                scope = authenticate(state, user, password)
            except PermissionError:
                self._audit(state, actor, 'view', 'denied', reason='authentication_failed', operation_id=op)
                raise
            self._audit(state, actor, 'view', 'authenticated', operation_id=op)
            try:
                exact = dict(sender=sender, recipient=recipient, task_id=task,
                             correlation_id=correlation, classification=classification, id=record_id)
                filters = {key: value for key, value in exact.items() if value is not None}
                for value in filters.values():
                    identifier(value)
                if record_id is not None:
                    message_id(record_id)
                if classification is not None and classification not in CLASSIFICATIONS:
                    raise ValueError('Invalid classification')
                if agent is not None: identifier(agent)
                if session is not None: identifier(session)
                if kind is not None and kind not in KINDS: raise ValueError('Invalid kind')
                if type(limit) is not int or not 1 <= limit <= MAX_RESULTS: raise ValueError('Invalid limit')
                if after is not None:
                    parts = after.split('/')
                    if len(parts) != 2: raise ValueError('Invalid cursor')
                    message_id(parts[0]); message_id(parts[1])
                # Disallow explicitly out-of-scope filters, even if the journal is empty.
                for field, query in [('agents', agent), ('sessions', session), ('kinds', kind),
                                     ('classifications', classification)]:
                    if query is not None and scope[field] != ['*'] and query not in scope[field]:
                        raise PermissionError('Requested filter outside scope')
                if any(not values for values in scope.values()):
                    raise PermissionError('Account has no read scope')
                records = []; cursor = None; more = False; scope_filtered = 0; policy_filtered = 0
                for epoch, entry in sorted(state['epochs'].items()):
                    if entry['status'] == 'revoked': continue
                    for path in self._packages('journal', epoch):
                        position = f'{epoch}/{path.stem}'
                        if after is not None and position <= after: continue
                        record = self._read_record(path, epoch, entry)
                        if not scope_allows(scope, record):
                            scope_filtered += 1
                            continue
                        if agent is not None and agent not in (record['sender'], record['recipient']): continue
                        if session is not None and record['session_id'] != session: continue
                        if kind is not None and record['kind'] != kind: continue
                        if any(record[key] != value for key, value in filters.items()): continue
                        if authorize_read is not None:
                            from copy import deepcopy
                            if authorize_read(actor, deepcopy(record)) is not True:
                                policy_filtered += 1
                                continue
                        if len(records) == limit:
                            more = True; break
                        records.append(record); cursor = position
                    if more: break
            except PermissionError:
                self._audit(state, actor, 'view', 'denied', reason='scope_or_policy_denied', operation_id=op)
                raise
            except Exception:
                self._audit(state, actor, 'view', 'failed', reason='query_or_integrity_failure', operation_id=op)
                raise
            # Audit the exact returned record IDs, not payloads. If persistence fails, nothing returns.
            self._audit(state, actor, 'view', 'allowed', details={'count': len(records),
                        'record_ids': [r['id'] for r in records], 'scope_filtered': scope_filtered,
                        'policy_filtered': policy_filtered}, operation_id=op)
            return {'records': records, 'next_cursor': cursor if more else None}

    def read_audit(self, *, limit: int = 100, after: str | None = None) -> dict:
        """Operator only; audit reads are themselves audited before returning."""
        if type(limit) is not int or not 1 <= limit <= MAX_RESULTS:
            raise ValueError('Invalid limit')
        if after is not None:
            parts = after.split('/')
            if len(parts) != 2: raise ValueError('Invalid cursor')
            message_id(parts[0]); message_id(parts[1])
        with self.store.locked():
            state = self.store.load(); events = []; cursor = None; more = False
            try:
                for epoch, entry in sorted(state['epochs'].items()):
                    for path in self._packages('audit', epoch):
                        position = f'{epoch}/{path.stem}'
                        if after is not None and position <= after: continue
                        if len(events) == limit: more = True; break
                        message_id(path.stem)
                        package = read_encrypted(path, private_key=decode_key(entry['audit_private_key']))
                        require_trusted(package, self.store.root_pin)
                        event = package.document['extensions'][AUDIT_EXTENSION]
                        if event['schema'] != 'slai.communication-audit.v2' or event['id'] != path.stem or event['epoch'] != epoch:
                            raise ValueError('Audit identity mismatch')
                        events.append(event); cursor = position
                    if more: break
            except Exception:
                self._audit(state, self._operator_id, 'read_audit', 'failed', reason='integrity_failure')
                raise
            self._audit(state, self._operator_id, 'read_audit', 'allowed', details={'count': len(events),
                        'event_ids': [e['id'] for e in events]})
            return {'events': events, 'next_cursor': cursor if more else None}
