"""Clone, pin, install, and verify LSTP for SLAI v2.3.

This script never mutates sys.path. The pinned LSTP repository is cloned to
SLAI/model/LSTP and installed into the active Python environment.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
LOCK_PATH = ROOT / "model" / "lstp.lock.json"


def _load_lock() -> dict[str, Any]:
    data = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    required = {
        "repository",
        "commit",
        "package_version",
        "protocol_version",
        "target",
    }
    missing = required - set(data)
    if missing:
        raise SystemExit(f"LSTP lock missing fields: {sorted(missing)!r}")
    commit = str(data["commit"])
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise SystemExit("LSTP lock commit must be a lowercase 40-character SHA")
    return data


def _run(command: list[str], *, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )


def _target(lock: dict[str, Any]) -> Path:
    target = (ROOT / str(lock["target"])).resolve()
    model_root = (ROOT / "model").resolve()
    if model_root != target and model_root not in target.parents:
        raise SystemExit("LSTP target must stay under SLAI/model")
    return target


def _git_head(target: Path) -> str:
    return _run(["git", "-C", str(target), "rev-parse", "HEAD"]).stdout.strip()


def _verify_import(lock: dict[str, Any]) -> None:
    code = (
        "import json, lstp; "
        "print(json.dumps({'package_version': lstp.__version__}))"
    )
    completed = _run([sys.executable, "-c", code])
    payload = json.loads(completed.stdout)
    if payload.get("package_version") != lock["package_version"]:
        raise SystemExit(
            "Installed LSTP version mismatch: "
            f"{payload.get('package_version')!r} != {lock['package_version']!r}"
        )


def install(*, reinstall: bool = False, no_install: bool = False) -> None:
    lock = _load_lock()
    target = _target(lock)
    git = shutil.which("git")
    if git is None:
        raise SystemExit("git is required to provision SLAI/model/LSTP")

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not (target / ".git").is_dir():
        if any(target.iterdir()):
            raise SystemExit(
                f"{target} exists but is not an LSTP git clone; refusing to overwrite"
            )
        target.rmdir()

    if not target.exists():
        _run([git, "clone", str(lock["repository"]), str(target)])

    _run([git, "-C", str(target), "fetch", "origin"])
    _run([git, "-C", str(target), "checkout", "--detach", str(lock["commit"])])
    if _git_head(target) != lock["commit"]:
        raise SystemExit("LSTP checkout does not match the pinned commit")

    if not no_install:
        command = [sys.executable, "-m", "pip", "install"]
        if reinstall:
            command.append("--force-reinstall")
        command.extend(["--no-deps", "-e", str(target)])
        _run(command)
        _verify_import(lock)

    print(
        json.dumps(
            {
                "status": "ready",
                "target": str(target.relative_to(ROOT)),
                "commit": lock["commit"],
                "package_version": lock["package_version"],
                "protocol_version": lock["protocol_version"],
                "installed": not no_install,
            },
            sort_keys=True,
        )
    )


def check() -> None:
    lock = _load_lock()
    target = _target(lock)
    if not (target / ".git").is_dir():
        raise SystemExit(
            "Pinned LSTP clone is missing. Run: python setup_lstp.py"
        )
    head = _git_head(target)
    if head != lock["commit"]:
        raise SystemExit(
            f"LSTP checkout drift: {head} != pinned {lock['commit']}"
        )
    _verify_import(lock)
    print(
        json.dumps(
            {
                "status": "ready",
                "target": str(target.relative_to(ROOT)),
                "commit": head,
                "package_version": lock["package_version"],
                "protocol_version": lock["protocol_version"],
            },
            sort_keys=True,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--reinstall", action="store_true")
    parser.add_argument("--no-install", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        check()
    else:
        install(reinstall=args.reinstall, no_install=args.no_install)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
