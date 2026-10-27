"""The local web server. It listens on the loopback interface only and treats every request as untrusted.

* **Who may ask**: a random token, shown only in the link printed at start-up, is exchanged for an HttpOnly, SameSite=Strict cookie.
  Every other request needs that cookie. Another program on this computer, or a web page open in the browser, cannot read the token.
* **Which page may ask**: the ``Host`` header must be this server (defeats DNS rebinding); changes (POST/PUT/PATCH/DELETE) also need
  the ``X-SLAI-<App>`` header and, when the browser sends an ``Origin``, that origin must be this server.
* **What the page may do**: a strict Content-Security-Policy (own scripts and styles only, no inline script, no plugins, no framing).
* **What it may be sent**: request bodies are read only up to a limit; JSON is parsed, never evaluated.
* **What it may read**: the application's built client, the shared UI kit (``/tools/``) and the asset folders (``/assets/``), by a
  strict file-name pattern and never outside those folders.

This is the one implementation for every SLAI application; an application supplies an :class:`Identity`, a :class:`Router` and an
``app`` object (which its handlers receive).
"""
from __future__ import annotations

import hmac
import http.cookies
import json
import mimetypes
import re
import sqlite3
import sys
import traceback
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

from ..errors import AppError, BadRequest, MethodNotAllowed, NotFound, Unauthorized, classify_oserror, classify_sqlite
from .router import Request, Response, Router, dumps, ok

CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; font-src 'self'; media-src 'self'; "
       "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
SECURITY_HEADERS = {
    "Content-Security-Policy": CSP, "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY",
    "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "geolocation=(), camera=(), microphone=()",
}
TOOLS_FILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}\.(?:css|woff2|svg|png)$")
ASSET_FILE = re.compile(r"^(?:[A-Za-z0-9_-]{1,24}/)?[A-Za-z0-9][A-Za-z0-9 _.-]{0,80}\.(?:png|svg|ico|jpg|jpeg|webp|mp3|ogg|wav)$")
MIME_FIX = {".js": "text/javascript", ".mjs": "text/javascript", ".svg": "image/svg+xml", ".json": "application/json", ".css": "text/css",
            ".mp3": "audio/mpeg", ".ogg": "audio/ogg", ".wav": "audio/wav", ".map": "application/json"}


@dataclass(frozen=True)
class Identity:
    """Who the application is: ``Identity("splanner", "SPlanner")``."""

    slug: str
    title: str

    @property
    def cookie(self) -> str:
        return f"slai_{self.slug}_token"

    @property
    def marker(self) -> str:
        """The request header every change must carry, in lower case (``x-slai-splanner``)."""
        return f"x-slai-{self.slug}"

    @property
    def locked_page(self) -> bytes:
        title = self.title.replace("<", "").replace("&", "")
        return (f"<!doctype html><meta charset=utf-8><title>SLAI {title}</title><body style=\"font:16px system-ui;margin:3rem\">"
                f"<h1>SLAI {title}</h1><p>This page needs the link that {title} printed when it started (it contains a one-time "
                f"access token). Open that link, or start {title} again.</p></body>").encode("utf-8")


@dataclass
class ServerOptions:
    token: str
    client_dir: Path
    host: str = "127.0.0.1"
    port: int = 0
    tools_web_dir: Path | None = None                 # served at /tools/ (the shared UI kit: slai-ui.css)
    asset_dirs: tuple[Path, ...] = ()                 # served at /assets/, first match wins
    max_body_bytes: int = 1024 * 1024
    extra_headers: dict[str, str] = field(default_factory=dict)


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, app: Any, identity: Identity, router: Router, options: ServerOptions) -> None:
        self.app = app
        self.identity = identity
        self.router = router
        self.options = options
        self.client_dir = options.client_dir.resolve()
        super().__init__((options.host, options.port), Handler)
        self.port = self.server_address[1]
        self.allowed_hosts = {f"127.0.0.1:{self.port}", f"localhost:{self.port}", f"[::1]:{self.port}"}

    def handle_error(self, request: Any, client_address: Any) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
            return                                    # the browser went away mid-response: not an error
        super().handle_error(request, client_address)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


class Handler(BaseHTTPRequestHandler):
    server: LocalServer
    protocol_version = "HTTP/1.1"
    server_version = "SLAI"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:    # quiet: what a person plans or writes does not belong in a log
        return

    # ------------------------------------------------------------------ plumbing
    def _send(self, resp: Response) -> None:
        self.send_response(resp.status)
        self.send_header("Content-Type", resp.content_type)
        for key, value in SECURITY_HEADERS.items():
            self.send_header(key, value)
        for key, value in self.server.options.extra_headers.items():
            self.send_header(key, value)
        for key, value in resp.headers.items():
            self.send_header(key, value)
        if resp.download_name:
            name = re.sub(r"[^A-Za-z0-9._ -]", "_", resp.download_name)
            self.send_header("Content-Disposition", f'attachment; filename="{name}"')
        self.send_header("Content-Length", str(len(resp.body)))
        if self.close_connection:
            self.send_header("Connection", "close")           # tell the client, so it opens a new connection instead of reusing this
        if "Cache-Control" not in resp.headers:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(resp.body)

    @staticmethod
    def _error(exc: AppError) -> Response:
        return Response(exc.status, dumps({"ok": False, "error": exc.to_dict()}))

    def _authorised(self) -> bool:
        raw = self.headers.get("Cookie")
        if not raw:
            return False
        jar = http.cookies.SimpleCookie()
        try:
            jar.load(raw)
        except http.cookies.CookieError:
            return False
        morsel = jar.get(self.server.identity.cookie)
        return morsel is not None and hmac.compare_digest(morsel.value.encode(), self.server.options.token.encode())

    def _host_ok(self) -> bool:
        return (self.headers.get("Host") or "").lower() in self.server.allowed_hosts

    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        return origin is None or origin.lower() in {f"http://{host}" for host in self.server.allowed_hosts}

    # ------------------------------------------------------------------ dispatch
    def _handle(self) -> None:
        parts = urlsplit(self.path)
        path = unquote(parts.path)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        identity, options = self.server.identity, self.server.options
        try:
            if not self._host_ok():
                return self._send(self._error(Unauthorized("This address is not allowed")))
            if path == "/" and "token" in query:                  # token exchange: /?token=...  ->  cookie, then a clean URL
                if hmac.compare_digest(query["token"].encode(), options.token.encode()):
                    cookie = f"{identity.cookie}={options.token}; Path=/; HttpOnly; SameSite=Strict"
                    return self._send(Response(302, b"", "text/plain", {"Location": "/", "Set-Cookie": cookie}))
                return self._send(Response(403, identity.locked_page, "text/html; charset=utf-8"))
            if not self._authorised():
                if path.startswith("/api/"):
                    return self._send(self._error(Unauthorized(f"Not signed in. Open {identity.title} from the link it printed at start-up.")))
                return self._send(Response(403, identity.locked_page, "text/html; charset=utf-8"))
            if path.startswith("/api/"):
                return self._send(self._api(path, query))
            if self.command not in ("GET", "HEAD"):
                return self._send(self._error(MethodNotAllowed("Method not allowed")))
            if path.startswith("/tools/") and options.tools_web_dir is not None:
                return self._send(self._tools(path[len("/tools/"):]))
            if path.startswith("/assets/"):
                return self._send(self._asset(path[len("/assets/"):]))
            return self._send(self._static(path))
        except AppError as exc:
            self._send(self._error(exc))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # The browser closed its socket while a response was being written (WinError 10053 is ConnectionAbortedError). That is a
            # transport cancellation, not a failure of the application; never retry on a stream whose framing may be incomplete.
            self.close_connection = True
        except sqlite3.Error as exc:
            storage = classify_sqlite(exc)
            self._log_storage(storage)
            self._send(self._error(storage))
        except OSError as exc:
            storage = classify_oserror(exc)
            self._log_storage(storage)
            self._send(self._error(storage))
        except Exception:  # noqa: BLE001 - the person gets a plain message; the trace goes to the console
            traceback.print_exc(file=sys.stderr)
            try:
                self._send(self._error(AppError(f"Something went wrong inside {identity.title}. The details are in its console window.")))
            except Exception:  # noqa: BLE001
                pass

    def _log_storage(self, error: AppError) -> None:
        """One console line with the technical facts (no content, no tokens), so a failed save can be diagnosed."""
        try:
            print(f"[{self.server.identity.slug}] storage error {error.code}: {json.dumps(error.details, default=str)}",
                  file=sys.stderr, flush=True)
        except Exception:  # noqa: BLE001 - logging must never mask the real failure
            pass

    def _api(self, path: str, query: dict[str, str]) -> Response:
        mutating = self.command in ("POST", "PUT", "PATCH", "DELETE")
        if mutating:
            if self.server.identity.marker not in {key.lower() for key in self.headers.keys()}:
                raise Unauthorized("Missing request marker")
            if not self._origin_ok():
                raise Unauthorized("Cross-origin requests are not allowed")
        found = self.server.router.match(self.command if self.command != "HEAD" else "GET", path)
        if found is None:
            raise NotFound("There is no such function")
        if isinstance(found, str):
            raise MethodNotAllowed("Method not allowed for this address")
        handler, params = found
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as exc:
            raise BadRequest("Bad Content-Length") from exc
        if length < 0:
            raise BadRequest("Bad Content-Length")
        request = Request(self.command, path, query, {key.lower(): value for key, value in self.headers.items()},
                          self.rfile, length or None, params, max_body=self.server.options.max_body_bytes)
        try:
            out = handler(self.server.app, request)
        finally:
            if mutating and not request._read and length:
                self.close_connection = True              # an unread body would corrupt the next request on this connection
        return out if isinstance(out, Response) else ok(out)

    # ------------------------------------------------------------------ static files
    def _file(self, root: Path, rel: str, cache: str, *, ranged: bool = False) -> Response:
        root = root.resolve()
        target = (root / rel).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            raise NotFound("Not found") from None
        if not target.is_file():
            raise NotFound("Not found")
        mime = MIME_FIX.get(target.suffix.lower()) or mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if mime.startswith("text/") or mime == "application/json":
            mime += "; charset=utf-8"
        body = target.read_bytes()
        return self._ranged(body, mime, cache) if ranged else Response(200, body, mime, {"Cache-Control": cache})

    def _ranged(self, body: bytes, mime: str, cache: str) -> Response:
        """Media elements ask for byte ranges; answer them (206) so audio and video start and seek in every browser."""
        size = len(body)
        header = self.headers.get("Range")
        base = {"Cache-Control": cache, "Accept-Ranges": "bytes"}
        if header and header.startswith("bytes=") and "," not in header:
            first, _, last = header[len("bytes="):].partition("-")
            try:
                if first == "":
                    start, end = max(0, size - int(last)), size - 1
                else:
                    start, end = int(first), (int(last) if last else size - 1)
            except ValueError:
                return Response(200, body, mime, base)
            end = min(end, size - 1)
            if start < 0 or start > end:
                return Response(416, b"", "text/plain", {**base, "Content-Range": f"bytes */{size}"})
            return Response(206, body[start:end + 1], mime, {**base, "Content-Range": f"bytes {start}-{end}/{size}"})
        return Response(200, body, mime, base)

    def _static(self, path: str) -> Response:
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        return self._file(self.server.client_dir, rel, "no-cache")

    def _tools(self, rel: str) -> Response:
        """The shared UI kit (``applications/tools/web``): one stylesheet for every application."""
        if not TOOLS_FILE.match(rel) or self.server.options.tools_web_dir is None:
            raise NotFound("Not found")
        return self._file(self.server.options.tools_web_dir, rel, "no-cache")

    def _asset(self, rel: str) -> Response:
        """Icons, logos and sounds live once, in an asset folder; they are served from there and never copied into the client."""
        if not ASSET_FILE.match(rel):
            raise NotFound("Not found")
        for folder in self.server.options.asset_dirs:
            if (folder / rel).is_file():
                return self._file(folder, rel, "public, max-age=3600", ranged=True)
        raise NotFound("Not found")

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = _handle

    def do_OPTIONS(self) -> None:       # no cross-origin pre-flight is ever approved
        self._send(Response(405, b"", "text/plain"))
