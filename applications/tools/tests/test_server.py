"""Tests for the shared loopback HTTP server, against a real socket and a toy application."""
from __future__ import annotations

import http.client
import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from applications.tools.errors import BadRequest, Conflict, NotFound
from applications.tools.server import Identity, LocalServer, Router, ServerOptions, created, ok

TOKEN = "test-token-123"
IDENTITY = Identity("toyapp", "ToyApp")
router = Router()


@router.route("GET", "/api/hello")
def hello(app, req):
    return {"hello": "world", "count": app["count"]}


@router.route("POST", "/api/echo")
def echo(app, req):
    return created(req.json())


@router.route("GET", "/api/items/{id}")
def item(app, req):
    if req.params["id"] == "missing":
        raise NotFound("no such item")
    return ok({"id": req.params["id"], "q": req.str_query("q"), "n": req.int_query("n", 5, 1, 9)})


@router.route("PUT", "/api/conflict")
def conflict(app, req):
    raise Conflict("version changed")


@router.route("GET", "/api/boom")
def boom(app, req):
    raise RuntimeError("secret internals")


@router.route("GET", "/api/db")
def db_failure(app, req):
    raise sqlite3.OperationalError("database is locked")


@router.route("GET", "/api/disk")
def disk_failure(app, req):
    raise OSError(28, "No space left on device")


@router.route("POST", "/api/lazy")
def lazy(app, req):
    return {"ignored": "body"}                       # never reads the body


class Base(unittest.TestCase):
    server: LocalServer
    tmp: tempfile.TemporaryDirectory

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        (root / "client").mkdir()
        (root / "client" / "index.html").write_text("<!doctype html><title>toy</title>", encoding="utf-8")
        (root / "client" / "app.js").write_text("export {};", encoding="utf-8")
        (root / "web").mkdir()
        (root / "web" / "slai-ui.css").write_text("body{}", encoding="utf-8")
        (root / "web" / "secret.ts").write_text("nope", encoding="utf-8")
        (root / "assets" / "audio").mkdir(parents=True)
        (root / "assets" / "audio" / "ding.mp3").write_bytes(bytes(range(256)) * 4)
        (root / "assets" / "logo.svg").write_text("<svg/>", encoding="utf-8")
        (root.parent / "outside.txt").write_text("outside", encoding="utf-8")
        options = ServerOptions(token=TOKEN, client_dir=root / "client", port=0, tools_web_dir=root / "web", asset_dirs=(root / "assets",),
                                max_body_bytes=2048)
        cls.server = LocalServer({"count": 3}, IDENTITY, router, options)
        cls.thread = threading.Thread(target=cls.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.tmp.cleanup()

    def request(self, method: str, path: str, body: object = None, *, headers: dict[str, str] | None = None, cookie: bool = True,
                marker: bool | None = None, host: str | None = None, raw: bytes | None = None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.port, timeout=5)
        send = dict(headers or {})
        if cookie:
            send["Cookie"] = f"{IDENTITY.cookie}={TOKEN}"
        if marker is None:
            marker = method in ("POST", "PUT", "PATCH", "DELETE")
        if marker:
            send[IDENTITY.marker] = "1"
        if host is not None:
            send["Host"] = host
        payload = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        if payload is not None:
            send["Content-Type"] = "application/json"
        conn.request(method, path, body=payload, headers=send)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response, data

    def json(self, *args, **kwargs):
        response, data = self.request(*args, **kwargs)
        return response.status, json.loads(data)


class AuthTests(Base):
    def test_token_exchange_sets_a_strict_cookie_then_redirects_to_a_clean_url(self) -> None:
        response, _ = self.request("GET", f"/?token={TOKEN}", cookie=False)
        self.assertEqual((response.status, response.getheader("Location")), (302, "/"))
        cookie = response.getheader("Set-Cookie") or ""
        self.assertIn(f"{IDENTITY.cookie}={TOKEN}", cookie)
        for flag in ("HttpOnly", "SameSite=Strict", "Path=/"):
            self.assertIn(flag, cookie)

    def test_wrong_or_missing_token_gets_the_locked_page_and_no_cookie(self) -> None:
        response, data = self.request("GET", "/?token=wrong", cookie=False)
        self.assertEqual(response.status, 403)
        self.assertIsNone(response.getheader("Set-Cookie"))
        self.assertIn(b"ToyApp", data)
        self.assertEqual(self.request("GET", "/", cookie=False)[0].status, 403)
        self.assertEqual(self.request("GET", "/app.js", cookie=False)[0].status, 403)

    def test_api_needs_the_cookie(self) -> None:
        status, body = self.json("GET", "/api/hello", cookie=False)
        self.assertEqual((status, body["ok"], body["error"]["code"]), (401, False, "unauthorized"))
        status, body = self.json("GET", "/api/hello", headers={"Cookie": f"{IDENTITY.cookie}=nope"}, cookie=False)
        self.assertEqual(status, 401)

    def test_host_header_must_be_this_server(self) -> None:
        for host in ("evil.example", "127.0.0.1:1", f"attacker.test:{self.server.port}"):
            self.assertEqual(self.json("GET", "/api/hello", host=host)[0], 401, host)
        self.assertEqual(self.json("GET", "/api/hello", host=f"localhost:{self.server.port}")[0], 200)

    def test_changes_need_the_marker_and_a_local_origin(self) -> None:
        self.assertEqual(self.json("POST", "/api/echo", {"a": 1}, marker=False)[0], 401)
        self.assertEqual(self.json("POST", "/api/echo", {"a": 1}, headers={"Origin": "http://evil.example"})[0], 401)
        self.assertEqual(self.json("POST", "/api/echo", {"a": 1}, headers={"Origin": f"http://127.0.0.1:{self.server.port}"})[0], 201)
        self.assertEqual(self.json("POST", "/api/echo", {"a": 1})[0], 201)             # no Origin header (not a browser): marker suffices

    def test_options_is_never_approved(self) -> None:
        self.assertEqual(self.request("OPTIONS", "/api/hello", marker=False)[0].status, 405)


class ApiTests(Base):
    def test_envelope_params_and_queries(self) -> None:
        self.assertEqual(self.json("GET", "/api/hello"), (200, {"ok": True, "data": {"hello": "world", "count": 3}}))
        status, body = self.json("GET", "/api/items/abc-1?q=hi&n=99")
        self.assertEqual((status, body["data"]), (200, {"id": "abc-1", "q": "hi", "n": 9}))
        status, body = self.json("GET", "/api/items/abc?n=x")
        self.assertEqual((status, body["error"]["code"]), (400, "bad_request"))

    def test_errors_map_to_status_and_code(self) -> None:
        self.assertEqual(self.json("GET", "/api/items/missing")[0], 404)
        self.assertEqual(self.json("PUT", "/api/conflict", {})[1]["error"]["code"], "conflict")
        self.assertEqual(self.json("GET", "/api/nothing-here")[0], 404)
        self.assertEqual(self.json("POST", "/api/hello", {})[0], 405)               # path exists, method does not

    def test_unexpected_errors_do_not_leak_internals(self) -> None:
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()):
            status, body = self.json("GET", "/api/boom")
        self.assertEqual(status, 500)
        self.assertNotIn("secret internals", json.dumps(body))

    def test_storage_failures_are_classified_by_cause(self) -> None:
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()):
            status, body = self.json("GET", "/api/db")
            self.assertEqual((status, body["error"]["code"]), (423, "storage_locked"))
            status, body = self.json("GET", "/api/disk")
            self.assertEqual((status, body["error"]["code"]), (507, "storage_disk_full"))

    def test_bodies_are_validated_and_capped(self) -> None:
        self.assertEqual(self.json("POST", "/api/echo", raw=b"{broken")[0], 400)
        self.assertEqual(self.json("POST", "/api/echo", raw=b"[1,2]")[0], 400)
        self.assertEqual(self.json("POST", "/api/echo", {"pad": "x" * 4000})[0], 413)
        self.assertEqual(self.json("POST", "/api/echo", {"ok": True})[1]["data"], {"ok": True})

    def test_an_unread_body_does_not_corrupt_the_next_request(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.server.port, timeout=5)
        headers = {"Cookie": f"{IDENTITY.cookie}={TOKEN}", IDENTITY.marker: "1", "Content-Type": "application/json"}
        conn.request("POST", "/api/lazy", body=b'{"x": 1}', headers=headers)
        first = conn.getresponse()
        first.read()
        self.assertEqual(first.status, 200)
        conn.request("GET", "/api/hello", headers={"Cookie": headers["Cookie"]})
        second = conn.getresponse()
        self.assertEqual((second.status, json.loads(second.read())["ok"]), (200, True))
        conn.close()


class StaticTests(Base):
    def test_client_files_and_security_headers(self) -> None:
        response, data = self.request("GET", "/")
        self.assertEqual((response.status, data), (200, b"<!doctype html><title>toy</title>"))
        csp = response.getheader("Content-Security-Policy") or ""
        for part in ("default-src 'none'", "script-src 'self'", "style-src 'self'", "media-src 'self'", "frame-ancestors 'none'"):
            self.assertIn(part, csp)
        self.assertNotIn("unsafe-inline", csp)
        self.assertEqual(response.getheader("X-Content-Type-Options"), "nosniff")
        self.assertEqual(response.getheader("Referrer-Policy"), "no-referrer")
        self.assertEqual(response.getheader("Cache-Control"), "no-cache")
        response, _ = self.request("GET", "/app.js")
        self.assertEqual(response.getheader("Content-Type"), "text/javascript; charset=utf-8")
        self.assertEqual(self.request("HEAD", "/")[1], b"")

    def test_paths_cannot_leave_the_client_folder(self) -> None:
        for path in ("/../outside.txt", "/%2e%2e/outside.txt", "/..%2foutside.txt", "/a/../../outside.txt"):
            self.assertIn(self.request("GET", path)[0].status, (403, 404), path)

    def test_shared_ui_kit_serves_only_allowed_files(self) -> None:
        response, data = self.request("GET", "/tools/slai-ui.css")
        self.assertEqual((response.status, data, response.getheader("Content-Type")), (200, b"body{}", "text/css; charset=utf-8"))
        for path in ("/tools/secret.ts", "/tools/../client/index.html", "/tools/sub/slai-ui.css"):
            self.assertEqual(self.request("GET", path)[0].status, 404, path)

    def test_assets_by_pattern_with_byte_ranges(self) -> None:
        response, data = self.request("GET", "/assets/audio/ding.mp3")
        self.assertEqual((response.status, len(data), response.getheader("Content-Type"), response.getheader("Accept-Ranges")),
                         (200, 1024, "audio/mpeg", "bytes"))
        response, data = self.request("GET", "/assets/audio/ding.mp3", headers={"Range": "bytes=10-19"})
        self.assertEqual((response.status, data, response.getheader("Content-Range")), (206, bytes(range(10, 20)), "bytes 10-19/1024"))
        response, data = self.request("GET", "/assets/audio/ding.mp3", headers={"Range": "bytes=-4"})
        self.assertEqual((response.status, data), (206, bytes([252, 253, 254, 255])))
        self.assertEqual(self.request("GET", "/assets/audio/ding.mp3", headers={"Range": "bytes=5000-"})[0].status, 416)
        self.assertEqual(self.request("GET", "/assets/logo.svg")[0].getheader("Content-Type"), "image/svg+xml")
        for path in ("/assets/../client/index.html", "/assets/audio/../../client/app.js", "/assets/audio/nope.mp3", "/assets/a/b/c.png",
                     "/assets/logo.exe", "/assets/.hidden.png"):
            self.assertEqual(self.request("GET", path)[0].status, 404, path)


if __name__ == "__main__":
    unittest.main()
