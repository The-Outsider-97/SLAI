"""SLAI-root launcher for the pinned LSTP v0.1 integration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent


def _read(path: str, *, binary: bool = False) -> str | bytes:
    if path == "-":
        return sys.stdin.buffer.read() if binary else sys.stdin.read()
    target = Path(path)
    return target.read_bytes() if binary else target.read_text(encoding="utf-8")


def _require_lstp() -> Any:
    try:
        import lstp
    except ImportError as exc:
        raise SystemExit(
            "LSTP is not installed. Run python setup_lstp.py from the SLAI root."
        ) from exc
    if getattr(lstp, "__version__", None) != "0.1.0a1":
        raise SystemExit(
            f"Unsupported installed LSTP version: {getattr(lstp, '__version__', None)!r}"
        )
    return lstp


def _emit_packet(packet: Any) -> None:
    lstp = _require_lstp()
    sys.stdout.buffer.write(lstp.canonical_dumps(packet) + b"\n")


def _doctor() -> None:
    from setup_lstp import _git_head, _load_lock, _target

    lock = _load_lock()
    target = _target(lock)
    if not (target / ".git").is_dir():
        raise SystemExit(
            "Pinned clone missing at model/LSTP. Run: python setup_lstp.py"
        )
    head = _git_head(target)
    if head != lock["commit"]:
        raise SystemExit(f"LSTP checkout drift: {head} != {lock['commit']}")

    lstp = _require_lstp()
    print(
        json.dumps(
            {
                "status": "ready",
                "slai_root": str(ROOT),
                "target": str(target.relative_to(ROOT)),
                "commit": head,
                "package_version": lstp.__version__,
                "protocol_version": lock["protocol_version"],
                "authority_boundary": "host",
            },
            sort_keys=True,
        )
    )


def _frame_from_json(data: Any) -> SimpleNamespace:
    if not isinstance(data, dict):
        raise SystemExit("frame JSON must be an object")
    act_name = str(data.get("act_type", "ASSERTIVE")).strip().upper()
    return SimpleNamespace(
        intent=str(data.get("intent", "unknown")),
        entities=data.get("entities", {}),
        confidence=float(data.get("confidence", 0.0)),
        act_type=SimpleNamespace(name=act_name),
        propositional_content=data.get("propositional_content"),
        illocutionary_force=data.get("illocutionary_force"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("doctor", help="Verify pinned clone and installed LSTP")

    json_cmd = commands.add_parser(
        "validate-json",
        help="Validate canonical LSTP JSON and emit canonical bytes",
    )
    json_cmd.add_argument("input", help="UTF-8 file or - for stdin")
    json_cmd.add_argument("--require-canonical-bytes", action="store_true")

    canonical = commands.add_parser(
        "canonical-lattice",
        help="Validate one canonical Lattice packet and add SLAI envelope metadata",
    )
    canonical.add_argument("input", help="UTF-8 file or - for stdin")
    canonical.add_argument("--packet-id", required=True)

    compact = commands.add_parser(
        "compact-lattice",
        help="Compile compact Lattice to canonical LSTP",
    )
    compact.add_argument("input", help="UTF-8 file or - for stdin")
    compact.add_argument("--packet-id", required=True)
    compact.add_argument("--thread-id", required=True)

    frame = commands.add_parser(
        "frame-json",
        help="Convert an existing SLAI semantic frame JSON object to LSTP",
    )
    frame.add_argument("input", help="UTF-8 JSON file or - for stdin")
    frame.add_argument("--source-text", default="")
    frame.add_argument("--packet-id", required=True)
    frame.add_argument("--thread-id", required=True)

    args = parser.parse_args(argv)
    _require_lstp()

    from src.integrations.lstp_adapter import (
        canonical_lattice_to_packet,
        compact_lattice_to_packet,
        packet_from_frame,
    )

    if args.command == "doctor":
        _doctor()
        return 0
    if args.command == "validate-json":
        packet = _require_lstp().canonical_loads(
            _read(args.input, binary=True),
            require_canonical_bytes=args.require_canonical_bytes,
        )
        _emit_packet(packet)
        return 0
    if args.command == "canonical-lattice":
        packet = canonical_lattice_to_packet(
            str(_read(args.input)),
            packet_id=args.packet_id,
        )
        _emit_packet(packet)
        return 0
    if args.command == "compact-lattice":
        packet = compact_lattice_to_packet(
            str(_read(args.input)),
            packet_id=args.packet_id,
            thread_id=args.thread_id,
        )
        _emit_packet(packet)
        return 0
    if args.command == "frame-json":
        packet = packet_from_frame(
            _frame_from_json(json.loads(str(_read(args.input)))),
            source_text=args.source_text,
            packet_id=args.packet_id,
            thread_id=args.thread_id,
        )
        _emit_packet(packet)
        return 0

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
