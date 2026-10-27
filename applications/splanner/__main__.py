"""Run using: python -m applications.splanner [--data-dir PATH] [--no-browser]."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import Settings
from .server import serve


def main() -> None:
    parser = argparse.ArgumentParser(description="SPlanner local-first planning workspace")
    parser.add_argument("--data-dir", type=Path, help="Directory for notes and planner database")
    parser.add_argument("--config", type=Path, help="YAML configuration file")
    parser.add_argument("--port", type=int, help="Local HTTP port")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open a browser")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings.load(config_path=args.config, data_dir=args.data_dir, port=args.port)
    serve(settings, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
