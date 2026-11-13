"""Launch each installed app, check its published startup assets and API, then shut it down."""
from __future__ import annotations

import http.cookiejar
import io
import json
import os
import queue
import re
import socket
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
APPS = {
    "Documaster": ("slai", "health"),
    "Reader": ("slai_reader", "prefs"),
    "scical": ("scical", "status"),
    "CADCon": ("applications.CADCon", "system"),
    "slai_writer": ("applications.slai_writer", "bootstrap"),
    "splanner": ("applications.splanner", "bootstrap"),
    "SPADRA": ("applications.SPADRA", "bootstrap"),
}


class Scripts(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts = []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.scripts.append(dict(attrs))


def check(app, module, endpoint, *, dom=False):
    cwd = ROOT if module.startswith("applications.") else ROOT / "applications" / app
    with tempfile.TemporaryDirectory() as data, socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
        reservation.close()
        process = subprocess.Popen(
            [sys.executable, "-m", module, "--port", str(port), "--data-dir", data, "--no-browser"],
            cwd=cwd, env={**os.environ, "PYTHONPATH": str(ROOT), "SLAI_ROOT": str(ROOT)},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        urls = queue.Queue()

        def capture():
            for line in process.stdout:
                match = re.search(r"http://127\.0\.0\.1:\d+/\?token=\S+", line)
                if match:
                    urls.put(match.group())
            urls.put(None)

        reader = threading.Thread(target=capture, daemon=True)
        reader.start()
        try:
            url = urls.get(timeout=60)
            if not url:
                raise RuntimeError(f"{app} exited before printing its launch URL")
            jar = http.cookiejar.CookieJar()
            session = urllib.request.build_opener(
                urllib.request.ProxyHandler({}), urllib.request.HTTPCookieProcessor(jar)
            )

            def get(relative):
                with session.open(urllib.parse.urljoin(url, relative), timeout=20) as response:
                    return response.read()

            page = Scripts()
            page.feed(get(url).decode())
            loader = next(script for script in page.scripts if "data-logo" in script)
            for filename in ("loader.js", "loader.css"):
                assert get("startup/" + filename) == (ROOT / "applications/tools/web/startup" / filename).read_bytes()
            get(next(script["src"] for script in page.scripts if script.get("type") == "module"))
            try:
                logo = get(loader["data-logo"])
            except urllib.error.HTTPError as error:
                if error.code != 404 or "data-fallback" not in loader:
                    raise
                logo = get(loader["data-fallback"])
            with Image.open(io.BytesIO(logo)) as image:
                image.load()
            assert json.loads(get("/api/" + endpoint))["ok"]
            if dom:
                runner = Path(__file__).with_name("startup_dom.mjs")
                credentials = json.dumps({"url": url, "cookie": "; ".join(f"{c.name}={c.value}" for c in jar)})
                for scenario in ("success", "failure"):
                    subprocess.run(
                        ["node", str(runner), app, scenario], cwd=ROOT, text=True,
                        input=credentials, check=True, timeout=35,
                    )
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise
            reader.join(timeout=2)
            process.stdout.close()


if __name__ == "__main__":
    for name, (module_name, api_endpoint) in APPS.items():
        check(name, module_name, api_endpoint, dom="--dom" in sys.argv[1:])
        print(f"{name}: passed")
