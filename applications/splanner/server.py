"""Loopback-only SPlanner HTTP API and static client.

No remote users, external networking or background OS service are implied.
"""
from __future__ import annotations

import hmac
import json
import logging
import secrets
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .config import Settings
from .store import Conflict, Missing, PlannerError, PlannerStore

LOG = logging.getLogger("slai.splanner")
WEB_DIR = Path(__file__).with_name("web")


def make_handler(store: PlannerStore, token: str, settings: Settings):
    """Build the request handler without global mutable application state."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "SPlanner/0.1"

        def log_message(self, format_string: str, *args: object) -> None:
            LOG.info("http %s - %s", self.address_string(), format_string % args)

        def _headers(self, status: HTTPStatus, content_type: str, count: int) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(count))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
                "base-uri 'none'; frame-ancestors 'none'",
            )
            self.end_headers()

        def _bytes(self, status: HTTPStatus, data: bytes, content_type: str) -> None:
            self._headers(status, content_type, len(data))
            self.wfile.write(data)

        def _json(self, status: HTTPStatus, value: object) -> None:
            self._bytes(status, json.dumps(value, ensure_ascii=False).encode("utf-8"),
                        "application/json; charset=utf-8")

        def _authorized(self, *, page: bool = False) -> bool:
            if page:
                supplied = parse_qs(urlsplit(self.path).query).get("token", [""])[0]
            else:
                supplied = self.headers.get("X-SPlanner-Token", "")
            return hmac.compare_digest(supplied, token)

        def _origin_ok(self) -> bool:
            supplied = self.headers.get("Origin")
            if not supplied:
                return True
            return supplied in (f"http://127.0.0.1:{settings.port}",
                                f"http://localhost:{settings.port}")

        def _body(self) -> dict:
            try:
                count = int(self.headers.get("Content-Length", "-1"))
            except ValueError as exc:
                raise PlannerError("Invalid request length") from exc
            if count < 0 or count > settings.max_request_bytes:
                raise PlannerError("Request is empty or exceeds size limit")
            raw = self.rfile.read(count)
            try:
                value = json.loads(raw)
            except (ValueError, UnicodeDecodeError) as exc:
                raise PlannerError("Invalid JSON body") from exc
            if not isinstance(value, dict):
                raise PlannerError("Expected a JSON object")
            return value

        def _handle(self, callback) -> None:
            try:
                status, output = callback()
            except Conflict as exc:
                self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
            except Missing as exc:
                self._json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            except PlannerError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except (OSError, ValueError) as exc:
                LOG.exception("Storage or format operation failed")
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR,
                           {"error": "Storage operation failed; your previous data was retained where possible"})
            except Exception:
                LOG.exception("Unexpected SPlanner request failure")
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR,
                           {"error": "Unexpected server error"})
            else:
                self._json(status, output)

        def do_GET(self) -> None:
            route = urlsplit(self.path).path
            if route == "/":
                if not self._authorized(page=True):
                    self._json(HTTPStatus.UNAUTHORIZED, {"error": "An application access token is required"})
                    return
                page = (WEB_DIR / "index.html").read_text(encoding="utf-8")
                page = page.replace("__TOKEN__", token)
                self._bytes(HTTPStatus.OK, page.encode("utf-8"), "text/html; charset=utf-8")
                return
            assets = {"/app.js": ("app.js", "text/javascript; charset=utf-8"),
                      "/styles.css": ("styles.css", "text/css; charset=utf-8")}
            if route in assets:
                filename, content_type = assets[route]
                self._bytes(HTTPStatus.OK, (WEB_DIR / filename).read_bytes(), content_type)
                return
            if not route.startswith("/api/"):
                self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return
            if not self._authorized():
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "Invalid access token"})
                return
            def get():
                if route == "/api/bootstrap":
                    return HTTPStatus.OK, {
                        **store.snapshot(), "reminderPollSeconds": settings.reminder_poll_seconds,
                        "capabilities": {"sharing": False, "recurrence": False,
                                         "agentSuggestions": False, "nativeAlerts": False},
                    }
                if route == "/api/reminders/due":
                    return HTTPStatus.OK, store.due_reminders()
                if route.startswith("/api/notes/"):
                    return HTTPStatus.OK, store.get_note(route.rsplit("/", 1)[-1])
                raise Missing("Endpoint not found")
            self._handle(get)

        def _mutation(self, method: str) -> None:
            if not self._authorized() or not self._origin_ok():
                self._json(HTTPStatus.FORBIDDEN, {"error": "Access denied"})
                return
            route = urlsplit(self.path).path
            def mutate():
                data = {} if method == "DELETE" else self._body()
                if method == "POST":
                    creation = {"/api/events": store.create_event,
                                "/api/tasks": store.create_task,
                                "/api/notes": store.create_note,
                                "/api/reminders": store.create_reminder,
                                "/api/stickies": store.create_sticky}
                    if route in creation:
                        return HTTPStatus.CREATED, creation[route](data)
                parts = route.strip("/").split("/")
                if len(parts) == 3 and parts[0] == "api":
                    kind, item_id = parts[1:]
                    if method == "PATCH":
                        edits = {"events": store.update_event, "tasks": store.update_task,
                                 "notes": store.update_note, "reminders": store.change_reminder,
                                 "stickies": store.update_sticky}
                        if kind in edits:
                            return HTTPStatus.OK, edits[kind](item_id, data)
                    if method == "DELETE" and kind in ("events", "tasks"):
                        store.delete_item(kind, item_id)
                        return HTTPStatus.OK, {"deleted": item_id}
                raise Missing("Endpoint not found")
            self._handle(mutate)

        def do_POST(self) -> None:
            self._mutation("POST")

        def do_PATCH(self) -> None:
            self._mutation("PATCH")

        def do_DELETE(self) -> None:
            self._mutation("DELETE")

    return Handler


def serve(settings: Settings, *, open_browser: bool = True) -> None:
    """Launch localhost SPlanner; callers own and can close the server lifecycle."""
    import webbrowser

    token = secrets.token_urlsafe(32)
    store = PlannerStore(settings.data_dir)
    try:
        handler = make_handler(store, token, settings)
        with ThreadingHTTPServer(("127.0.0.1", settings.port), handler) as server:
            url = f"http://127.0.0.1:{settings.port}/?token={token}"
            LOG.info("SPlanner listening on 127.0.0.1:%d", settings.port)
            if open_browser:
                webbrowser.open(url)
            try:
                server.serve_forever(poll_interval=0.3)
            except KeyboardInterrupt:
                LOG.info("SPlanner shutting down")
    finally:
        store.close()
