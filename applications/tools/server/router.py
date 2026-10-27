"""A small router and the request/response types of the HTTP layer (kept apart from the server so handlers are testable).

A handler is ``handler(app, request)``. It returns a :class:`Response`, or any JSON-serialisable value, which is wrapped as
``{"ok": true, "data": value}`` - the one envelope every SLAI client understands. Errors are raised as
:class:`~applications.tools.errors.AppError` and become ``{"ok": false, "error": {"code", "message", ...}}``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Callable

from ..errors import BadRequest, LimitExceeded

MAX_JSON_BYTES = 8 * 1024 * 1024
_PARAM = r"(?P<\1>[A-Za-z0-9_.:-]{1,128})"


@dataclass
class Request:
    method: str
    path: str
    query: dict[str, str]
    headers: dict[str, str]
    stream: BinaryIO | None = None
    length: int | None = None
    params: dict[str, str] = field(default_factory=dict)
    max_body: int = MAX_JSON_BYTES
    _json: Any = None
    _read: bool = False

    def json(self, limit: int | None = None) -> dict[str, Any]:
        """The JSON object in the body (``{}`` when there is none). Read at most ``limit`` bytes; never evaluated."""
        if not self._read:
            self._read = True
            cap = min(limit or self.max_body, self.max_body, MAX_JSON_BYTES)
            if self.stream is None or not self.length:
                self._json = {}
            else:
                if self.length > cap:
                    raise LimitExceeded("The request is too large")
                raw = self.stream.read(self.length)
                try:
                    self._json = json.loads(raw.decode("utf-8")) if raw.strip() else {}
                except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
                    raise BadRequest(f"The request is not valid JSON: {exc}") from exc
        if not isinstance(self._json, dict):
            raise BadRequest("The request body must be a JSON object")
        return self._json

    def str_query(self, name: str, default: str = "", max_len: int = 200) -> str:
        value = self.query.get(name, default)
        if len(value) > max_len:
            raise BadRequest(f"'{name}' is too long")
        return value

    def int_query(self, name: str, default: int, low: int, high: int) -> int:
        raw = self.query.get(name)
        if raw is None or raw == "":
            return default
        try:
            value = int(raw)
        except ValueError as exc:
            raise BadRequest(f"'{name}' must be a whole number") from exc
        return max(low, min(high, value))


@dataclass
class Response:
    status: int = 200
    body: bytes = b""
    content_type: str = "application/json; charset=utf-8"
    headers: dict[str, str] = field(default_factory=dict)
    download_name: str | None = None


Handler = Callable[[Any, Request], Any]


class Router:
    def __init__(self) -> None:
        self.routes: list[tuple[str, re.Pattern[str], Handler]] = []

    def add(self, method: str, pattern: str, handler: Handler) -> None:
        regex = re.compile("^" + re.sub(r"\{(\w+)\}", _PARAM, pattern) + "$")
        self.routes.append((method, regex, handler))

    def route(self, method: str, pattern: str) -> Callable[[Handler], Handler]:
        def decorate(fn: Handler) -> Handler:
            self.add(method, pattern, fn)
            return fn
        return decorate

    def match(self, method: str, path: str) -> tuple[Handler, dict[str, str]] | None | str:
        """``(handler, params)``, ``"method"`` when the path exists for other methods only, or ``None``."""
        allowed = False
        for route_method, regex, handler in self.routes:
            found = regex.match(path)
            if found:
                if route_method == method:
                    return handler, found.groupdict()
                allowed = True
        return "method" if allowed else None


def dumps(payload: Any) -> bytes:
    return json.dumps(payload, default=str, ensure_ascii=False, allow_nan=False).encode("utf-8")


def ok(data: Any = None, *, status: int = 200, **extra: Any) -> Response:
    payload = {"ok": True, "data": data}
    payload.update(extra)
    return Response(status, dumps(payload))


def created(data: Any = None, **extra: Any) -> Response:
    return ok(data, status=201, **extra)


def file_response(body: bytes, name: str, content_type: str) -> Response:
    """A download: the browser is told to save it under ``name``."""
    return Response(200, body, content_type, {"Cache-Control": "no-store"}, download_name=name)
