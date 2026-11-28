"""Read-only Linux IPC for a trusted host with a protected Unix-domain socket."""
from __future__ import annotations

import json
import socket
import struct
import time
from collections.abc import Callable, Mapping

from .communication_security import username
from .communication_storage import CommunicationJournal


class CommunicationReaderService:
    """Serve one connected stream at a time. The host owns socket ACLs and admission."""

    _filters = frozenset({'agent', 'sender', 'recipient', 'session', 'kind', 'task',
                          'correlation', 'classification', 'record_id', 'after', 'limit'})

    def __init__(self, journal: CommunicationJournal, accounts: Mapping[int, str],
                 authorize_read: Callable[[str, dict], bool]):
        if not callable(authorize_read):
            raise TypeError('A trusted read policy is required')
        if any(type(uid) is not int or uid < 0 for uid in accounts):
            raise ValueError('Invalid OS account mapping')
        self._accounts = {uid: username(name) for uid, name in accounts.items()}
        if len(set(self._accounts.values())) != len(self._accounts):
            raise ValueError('Each reader account requires its own OS principal')
        self._journal = journal
        self._authorize = authorize_read

    def serve_connection(self, connection: socket.socket) -> None:
        """Handle one bounded newline-delimited JSON request and close the connection."""
        with connection:
            if (connection.family != socket.AF_UNIX or
                    connection.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_STREAM
                    or not hasattr(socket, 'SO_PEERCRED')):
                raise PermissionError('Local authenticated stream required')
            connection.settimeout(5)
            try:
                credentials = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED,
                                                   struct.calcsize('3i'))
                _, uid, _ = struct.unpack('3i', credentials)
                user = self._accounts[uid]
                # One bounded read with an absolute deadline, including slow clients.
                deadline = time.monotonic() + 5
                raw = bytearray()
                while len(raw) <= 8192 and b'\n' not in raw:
                    connection.settimeout(max(0, deadline - time.monotonic()))
                    part = connection.recv(min(8193 - len(raw), 4096))
                    if not part:
                        break
                    raw.extend(part)
                if len(raw) > 8192 or not raw.endswith(b'\n'):
                    raise ValueError('Invalid request framing')
                request = json.loads(raw)
                if type(request) is not dict or set(request) - (self._filters | {'password'}):
                    raise ValueError('Invalid read request')
                password = request.pop('password')
                limit = request.pop('limit', 25)
                if type(limit) is not int or not 1 <= limit <= 25:
                    raise ValueError('Invalid page size')
                result = self._journal.view(user, password, limit=limit, local_reader=True,
                                            authorize_read=self._authorize, **request)
                response = {'ok': True, **result}
            except Exception:
                # Credential, policy, storage and protocol failures disclose no details.
                response = {'ok': False}
            connection.settimeout(5)
            connection.sendall(json.dumps(response, ensure_ascii=True).encode('ascii') + b'\n')
