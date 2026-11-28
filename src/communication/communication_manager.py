"""Small facade. Runtime policy, identity binding and delivery remain SLAI responsibilities."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .communication_storage import CommunicationJournal
from .communication_types import identifier, new_message, validate_message

Record = dict[str, Any]


class DeliveryError(RuntimeError):
    """The message was durably recorded, but delivery did not complete successfully."""
    def __init__(self, message_id: str):
        super().__init__(f'Delivery failed or is uncertain for recorded message {message_id}')
        self.message_id = message_id


class AgentCommunication:
    """Sender-bound interface issued by a trusted runtime composition root."""
    def __init__(self, sender: str, journal: CommunicationJournal,
                 authorize: Callable[[Record], bool], sanitize: Callable[[Record], Record],
                 deliver: Callable[[str, Record], None]):
        self._sender = identifier(sender)
        self._journal = journal
        self._authorize = authorize
        self._sanitize = sanitize
        self._deliver = deliver

    def send(self, recipient: str, kind: str, payload: dict[str, Any], *, session_id: str,
             task_id: str | None = None, correlation_id: str | None = None,
             classification: str = 'internal') -> str:
        """Authorize the exact sanitized record, commit, then invoke existing runtime delivery.

        The host must bind identity, classify/redact data and route to an authenticated recipient.
        No permissive policy or direct SharedMemory broadcast is supplied here.
        """
        original = new_message(self._sender, recipient, kind, payload, session_id=session_id,
                               task_id=task_id, correlation_id=correlation_id,
                               classification=classification)
        sanitized = validate_message(self._sanitize(deepcopy(original)))
        immutable = set(original) - {'payload', 'classification'}
        if any(sanitized[key] != original[key] for key in immutable):
            raise PermissionError('Sanitizer changed communication identity or routing')
        if self._authorize(deepcopy(sanitized)) is not True:
            raise PermissionError('Communication denied by runtime policy')
        self._journal.append(sanitized)
        try:
            self._deliver(recipient, deepcopy(sanitized))
        except Exception as exc:
            raise DeliveryError(sanitized['id']) from exc
        return sanitized['id']
