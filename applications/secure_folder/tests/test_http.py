"""End-to-end local HTTP checks for browser and vault authorization boundaries."""
from __future__ import annotations

import base64
import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path

from applications.secure_folder.app import HERE, build_server

PASSWORD = "Strong passphrase for SLAI Secure Folder 2026!"


class SecureFolderHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.server, cls.vault = build_server(data_dir=Path(cls.tmp.name) / "vault", token="only-from-launch-url",
                                              client_dir=HERE / "dist", port=0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.vault.close()
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3)
        cls.tmp.cleanup()

    def send(self, method: str, path: str, body: dict | None = None, cookie: str = "", token: str = "",
             origin: str | None = None) -> tuple[int, dict[str, str], bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.server.port, timeout=5)
        headers = {"Host": f"127.0.0.1:{self.server.port}"}
        if cookie:
            headers["Cookie"] = cookie
        if token:
            headers["X-Vault-Session"] = token
        if body is not None:
            headers["Content-Type"] = "application/json"
            headers["X-SLAI-Secure_folder"] = "1"
        if origin is not None:
            headers["Origin"] = origin
        conn.request(method, path, body=json.dumps(body).encode() if body is not None else None, headers=headers)
        resp = conn.getresponse()
        content = resp.read()
        headers = dict(resp.getheaders())
        conn.close()
        return resp.status, headers, content

    def test_local_authentication_and_encrypted_file(self) -> None:
        code, headers, _ = self.send("GET", "/api/status")
        self.assertEqual(code, 401)
        code, headers, _ = self.send("GET", "/?token=only-from-launch-url")
        self.assertEqual(code, 302)
        cookie = headers["Set-Cookie"].split(";", 1)[0]
        self.assertIn("HttpOnly", headers["Set-Cookie"])
        code, _, data = self.send("GET", "/api/status", cookie=cookie)
        self.assertEqual(code, 200)
        self.assertFalse(json.loads(data)["data"]["initialized"])
        code, _, _ = self.send("POST", "/api/setup", {"password": PASSWORD, "confirmation": PASSWORD}, cookie)
        self.assertEqual(code, 200)
        code, _, _ = self.send("GET", "/api/items", cookie=cookie)
        self.assertEqual(code, 401)
        code, _, data = self.send("POST", "/api/unlock", {"password": PASSWORD}, cookie)
        self.assertEqual(code, 200)
        token = json.loads(data)["data"]["sessionToken"]
        code, _, _ = self.send("POST", "/api/folders", {"name": "Private"}, cookie, token, origin="https://evil.example")
        self.assertEqual(code, 401)
        code, _, data = self.send("POST", "/api/uploads", {"name": "secret.txt", "size": 7, "mime": "text/plain"}, cookie, token)
        self.assertEqual(code, 201)
        ticket = json.loads(data)["data"]
        code, _, _ = self.send("POST", f"/api/uploads/{ticket['uploadId']}/chunks",
                                {"index": 0, "data": base64.b64encode(b"SECRET!").decode()}, cookie, token)
        self.assertEqual(code, 200)
        code, _, data = self.send("POST", f"/api/uploads/{ticket['uploadId']}/finish", {}, cookie, token)
        self.assertEqual(code, 201)
        entry_id = json.loads(data)["data"]["id"]
        code, headers, data = self.send("GET", f"/api/files/{entry_id}/chunks/0", cookie=cookie, token=token)
        self.assertEqual(code, 200)
        self.assertEqual(data, b"SECRET!")
        self.assertEqual(headers["Cache-Control"], "no-store")
        code, _, _ = self.send("POST", "/api/lock", {}, cookie, token)
        self.assertEqual(code, 200)
        code, _, _ = self.send("GET", f"/api/files/{entry_id}/chunks/0", cookie=cookie, token=token)
        self.assertEqual(code, 401)


if __name__ == "__main__":
    unittest.main()
