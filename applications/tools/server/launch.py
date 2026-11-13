"""Start-up scaffolding shared by every application: common command-line options, the loopback rule, the start-up banner and a
clean shutdown on Ctrl+C, SIGTERM or Ctrl+Break.

An application's ``__main__`` adds its own options to the parser, builds its app object and server, then calls :func:`serve`.
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading
import webbrowser
from typing import Callable, Sequence

LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def add_server_arguments(parser: argparse.ArgumentParser, *, title: str, default_port: int) -> None:
    """The options every SLAI application understands."""
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on (loopback only)")
    parser.add_argument("--port", type=int, default=None, help=f"port to listen on (default {default_port}; "
                        "Documaster 8766, Reader 8767, SciCal 8768, SPlanner 8769)")
    parser.add_argument("--data-dir", help=f"where {title} keeps its data")
    parser.add_argument("--slai-root", help="path to the SLAI checkout (default: found automatically)")
    parser.add_argument("--config", help="YAML configuration file")
    parser.add_argument("--token", help="fixed access token (for tests; random by default)")
    parser.add_argument("--client-dir", help="folder with the built web client")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser window")


def refuse_remote_host(host: str, title: str, what: str = "your data") -> bool:
    """``True`` (after saying why on stderr) when ``host`` is not a loopback address."""
    if host in LOOPBACK_HOSTS:
        return False
    print(f"Refusing to listen on a non-loopback address: {title} is a local app and holds {what}.", file=sys.stderr)
    return True


def serve(server, *, title: str, version: str, url: str, details: Sequence[tuple[str, str]] = (), notes: Sequence[str] = (),
          open_browser: bool = True, on_close: Callable[[], None] | None = None) -> int:
    """Print the banner, open the browser, serve until stopped, then close the server and the application."""
    print(f"{title} {version}", flush=True)
    for label, value in details:
        print(f"  {label:<10}: {value}", flush=True)
    print(f"  {'Open':<10}: {url}", flush=True)
    for note in notes:
        print(f"  {'notice':<10}: {note}", flush=True)
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    def stop(*_: object) -> None:           # a polite "please stop" (Ctrl+Break, taskkill without /f, a service manager) ends cleanly
        threading.Thread(target=server.shutdown, daemon=True).start()

    for name in ("SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            try:
                signal.signal(getattr(signal, name), stop)
            except (ValueError, OSError):
                pass                         # not the main thread, or not supported here
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if on_close is not None:
            on_close()
    return 0
