"""Secure Folder application and its SLAI local-server entry point."""
from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

from ..tools import slai
from ..tools.errors import AppError
from ..tools.server import Identity, LocalServer, ServerOptions, add_server_arguments, refuse_remote_host, serve
from ..tools.storage import default_data_dir
from . import __version__
from .vault import Vault

HERE = Path(__file__).resolve().parent
TITLE = "Secure Folder"
SLUG = "secure_folder"


def build_server(*, data_dir: Path, token: str, client_dir: Path | None = None, port: int = 0,
                 host: str = "127.0.0.1") -> tuple[LocalServer, Vault]:
    from .routes import router

    vault = Vault(data_dir)
    vault.acquire_instance_lock()
    options = ServerOptions(token=token, client_dir=client_dir or HERE / "dist", host=host, port=port,
                            tools_web_dir=slai.applications_dir() / "tools" / "web",
                            asset_dirs=(HERE / "assets", slai.shared_assets_dir()), max_body_bytes=1024 * 1024,
                            extra_headers={"Cross-Origin-Embedder-Policy": "require-corp"})
    try:
        return LocalServer(vault, Identity(SLUG, TITLE), router, options), vault
    except BaseException:
        vault.close()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m applications.secure_folder", description="SLAI's encrypted local vault")
    add_server_arguments(parser, title=TITLE, default_port=0)
    args = parser.parse_args(argv)
    if refuse_remote_host(args.host, TITLE, "private encrypted files"):
        return 2
    slai.activate(slai.find_slai_root(args.slai_root))
    client_dir = Path(args.client_dir) if args.client_dir else HERE / "dist"
    if not (client_dir / "index.html").is_file():
        print(f"Build the client first: `npm run build` from {HERE}.", file=sys.stderr)
        return 2
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir(SLUG, TITLE)
    token = args.token or secrets.token_urlsafe(32)
    try:
        server, vault = build_server(data_dir=data_dir, token=token, client_dir=client_dir,
                                     port=args.port, host=args.host)
    except (OSError, AppError) as exc:
        print(f"Secure Folder could not start: {exc}", file=sys.stderr)
        return 1
    return serve(server, title=TITLE, version=__version__,
                 url=f"http://127.0.0.1:{server.port}/?token={token}",
                 details=[("Data", str(data_dir)), ("Protection", "Local-only; vault requires a separate passphrase")],
                 notes=["The launch URL is not your vault password. Unlock the vault inside the app."],
                 open_browser=not args.no_browser, on_close=vault.close)
