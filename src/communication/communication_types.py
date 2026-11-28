"""Bounded, versioned SLDC communication record schema (not a new file format)."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

SCHEMA = 'slai.communication.v1'
EXTENSION = 'slai.communication'
MAX_PAYLOAD_BYTES = 64 * 1024
KINDS = frozenset({'request', 'response', 'event', 'task', 'audit', 'provenance',
                   'governance', 'security', 'learning', 'transaction', 'diagnostic'})
CLASSIFICATIONS = frozenset({'internal', 'confidential', 'restricted'})
_IDENTIFIER = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}\Z')
_SECRET_FIELDS = frozenset({'password', 'passphrase', 'api_key', 'apikey', 'secret',
                            'token', 'access_token', 'refresh_token', 'authorization',
                            'private_key', 'encryption_key', 'credentials', 'chain_of_thought'})


def identifier(value: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError('Invalid communication identifier')
    return value


def message_id(value: str) -> str:
    if not isinstance(value, str) or UUID(value).hex != value:
        raise ValueError('Expected canonical UUID hex')
    return value


def _check_payload(value: Any, depth: int = 0) -> None:
    if depth > 16:
        raise ValueError('Payload nesting exceeds limit')
    if type(value) is dict:
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError('Payload keys must be strings')
            if key.lower().replace('-', '_') in _SECRET_FIELDS:
                raise ValueError('Forbidden sensitive payload field')
            _check_payload(child, depth + 1)
    elif type(value) is list:
        for child in value:
            _check_payload(child, depth + 1)
    elif type(value) not in (str, int, float, bool, type(None)):
        raise ValueError('Payload must contain only structured primitive data')


def payload_copy(payload: dict[str, Any]) -> dict[str, Any]:
    if type(payload) is not dict:
        raise ValueError('Payload must be an object')
    _check_payload(payload)
    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
    if len(raw) > MAX_PAYLOAD_BYTES:
        raise ValueError('Payload too large; send an authorized artifact reference instead')
    return json.loads(raw)


def new_message(sender: str, recipient: str, kind: str, payload: dict[str, Any], *,
                session_id: str, task_id: str | None = None, correlation_id: str | None = None,
                classification: str = 'internal') -> dict[str, Any]:
    record = dict(schema=SCHEMA, id=uuid4().hex, timestamp=datetime.now(timezone.utc).isoformat(),
                  sender=sender, recipient=recipient, kind=kind, payload=payload_copy(payload),
                  session_id=session_id, task_id=task_id, correlation_id=correlation_id,
                  classification=classification)
    return validate_message(record)


def validate_message(record: dict[str, Any]) -> dict[str, Any]:
    keys = {'schema', 'id', 'timestamp', 'sender', 'recipient', 'kind', 'payload',
            'session_id', 'task_id', 'correlation_id', 'classification'}
    if type(record) is not dict or set(record) != keys or record['schema'] != SCHEMA:
        raise ValueError('Unsupported communication schema')
    message_id(record['id'])
    for key in ('sender', 'recipient', 'session_id'):
        identifier(record[key])
    for key in ('task_id', 'correlation_id'):
        if record[key] is not None:
            identifier(record[key])
    if record['kind'] not in KINDS or record['classification'] not in CLASSIFICATIONS:
        raise ValueError('Unsupported message kind or classification')
    stamp = datetime.fromisoformat(record['timestamp'])
    if stamp.utcoffset() != timezone.utc.utcoffset(None):
        raise ValueError('Timestamp must be UTC')
    return {**record, 'payload': payload_copy(record['payload'])}
