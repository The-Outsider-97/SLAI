"""Command-line interface for SLAI spatial documents (``.sdrw`` / ``.sgeo``).

    python -m src.utils.spatial_formats inspect  plan.sdrw [--verify digests]
    python -m src.utils.spatial_formats validate room.sgeo [--mode strict|basic] [--max-issues N]
    python -m src.utils.spatial_formats convert  drawing.dxf plan.sdrw [--policy P] [--overwrite]
    python -m src.utils.spatial_formats convert  room.sgeo room.glb
    python -m src.utils.spatial_formats capabilities [--dwg-executable PATH]

Every command prints exactly one JSON document on stdout. Exit codes:
0 success (``validate``: document is valid), 1 the operation failed or the
document is invalid (the JSON carries a stable ``code``), 2 usage error.

``convert`` picks the direction from the file suffixes:

* into SDRW: ``.svg``, ``.dxf``, ``.dwg``        * out of SDRW: ``.svg``, ``.dxf``, ``.dwg``
* into SGEO: ``.glb``, ``.stl``, ``.ifc``        * out of SGEO: ``.glb``, ``.stl``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .common.errors import SpatialFormatError, ValidationError

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2

_POLICIES = ("strict", "lossless_where_supported", "allow_approximation", "best_effort")
_INTO_SDRW = {".svg", ".dxf", ".dwg"}
_INTO_SGEO = {".glb", ".stl", ".ifc"}
_OUT_SDRW = {".svg", ".dxf", ".dwg"}
_OUT_SGEO = {".glb", ".stl"}
_NATIVE = {".sdrw", ".sgeo"}


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _failure(exc: BaseException) -> dict:
    if isinstance(exc, SpatialFormatError):
        return {"ok": False, "code": exc.code, "error": str(exc), "path": getattr(exc, "path", None)}
    return {"ok": False, "code": "os.error" if isinstance(exc, OSError) else "internal.error",
            "error": "%s: %s" % (type(exc).__name__, exc), "path": None}


def _document_class(path):
    """Class for a native file, chosen from its *content* (not only its suffix)."""
    from .container import inspect_package
    from .sdrw.document import DrawingDocument
    from .sgeo.document import SceneDocument
    info = inspect_package(path, verify="index", summary=False)
    return DrawingDocument if info.format_id == "slai.sdrw" else SceneDocument


def _cmd_inspect(args) -> int:
    from .container import inspect_package
    info = inspect_package(args.path, verify=args.verify, summary=not args.no_summary)
    _emit({"ok": True, **info.as_dict()})
    return EXIT_OK


def _cmd_validate(args) -> int:
    from .verification import report_from_exception
    try:
        report = _document_class(args.path).open(args.path, mode="read").validate(
            args.mode, max_issues=args.max_issues)
    except SpatialFormatError as exc:
        report = report_from_exception(exc, mode=args.mode)
    _emit(report.as_dict())
    return EXIT_OK if report.ok else EXIT_FAILED


def _cmd_capabilities(args) -> int:
    from .adapters import capabilities
    _emit({"ok": True, "formats": capabilities(dwg_executable=args.dwg_executable)})
    return EXIT_OK


def _require_dwg_executable(args, suffix):
    if suffix == ".dwg" and not args.dwg_executable:
        raise ValidationError("DWG conversion requires --dwg-executable (an ODA File Converter path)")
    return args.dwg_executable


def _convert(src: Path, dst: Path, args):
    """Return ``(document_or_None, ConversionReport_or_None, output_path)``."""
    from .sdrw.document import DrawingDocument
    from .sgeo.document import SceneDocument
    s, d = src.suffix.lower(), dst.suffix.lower()
    if s in _NATIVE:
        document = _document_class(src).open(src, mode="read")
        if isinstance(document, DrawingDocument) and d in _OUT_SDRW:
            if d == ".svg":
                return document.export_svg(dst, overwrite=args.overwrite, policy=args.policy)
            if d == ".dxf":
                return document.export_dxf(dst, overwrite=args.overwrite, policy=args.policy)
            return document.export_dwg(dst, executable=_require_dwg_executable(args, d),
                                       overwrite=args.overwrite, policy=args.policy)
        if isinstance(document, SceneDocument) and d in _OUT_SGEO:
            export = document.export_glb if d == ".glb" else document.export_stl
            return export(dst, overwrite=args.overwrite, policy=args.policy)
        raise ValidationError("cannot convert %s to %s" % (s, d))
    if d == ".sdrw" and s in _INTO_SDRW:
        if s == ".svg":
            document, report = DrawingDocument.import_svg(src, unit=args.unit or "mm", policy=args.policy)
        elif s == ".dxf":
            document, report = DrawingDocument.import_dxf(src, unit=args.unit, policy=args.policy,
                                                          skip_unsupported=args.skip_unsupported)
        else:
            document, report = DrawingDocument.import_dwg(
                src, executable=_require_dwg_executable(args, s), unit=args.unit or "mm", policy=args.policy)
    elif d == ".sgeo" and s in _INTO_SGEO:
        if s == ".glb":
            document, report = SceneDocument.import_glb(src, policy=args.policy)
        elif s == ".stl":
            document, report = SceneDocument.import_stl(src, unit=args.unit or "mm", policy=args.policy)
        else:
            document, report = SceneDocument.import_ifc(src, policy=args.policy)
    else:
        raise ValidationError("unsupported conversion %s -> %s" % (s or "?", d or "?"))
    document.save(dst, overwrite=args.overwrite)
    return report


def _cmd_convert(args) -> int:
    src, dst = Path(args.source), Path(args.destination)
    if not src.is_file():
        raise FileNotFoundError("source file not found: %s" % src)
    report = _convert(src, dst, args)
    _emit({"ok": True, "source": str(src), "destination": str(dst), "report": report.as_dict()})
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spatial_formats", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("inspect", help="describe a .sdrw/.sgeo file without loading its geometry")
    p.add_argument("path")
    p.add_argument("--verify", choices=("index", "digests"), default="index",
                   help="'digests' also hashes every member (default: ZIP index only)")
    p.add_argument("--no-summary", action="store_true", help="skip entity/object counts")
    p.set_defaults(handler=_cmd_inspect)

    p = commands.add_parser("validate", help="validate a file; exit 1 when it is invalid")
    p.add_argument("path")
    p.add_argument("--mode", choices=("strict", "basic"), default="strict")
    p.add_argument("--max-issues", type=int, default=200)
    p.set_defaults(handler=_cmd_validate)

    p = commands.add_parser("convert", help="convert to or from SVG/DXF/DWG (2D) and GLB/STL/IFC (3D)")
    p.add_argument("source")
    p.add_argument("destination")
    p.add_argument("--policy", choices=_POLICIES, default="best_effort")
    p.add_argument("--unit", help="declare the source unit when the format carries none (default mm)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing destination")
    p.add_argument("--skip-unsupported", action="store_true",
                   help="DXF import: omit unsupported entities (listed in the report) instead of refusing")
    p.add_argument("--dwg-executable", help="absolute path of an ODA File Converter (DWG only)")
    p.set_defaults(handler=_cmd_convert)

    p = commands.add_parser("capabilities", help="show which exchange formats are available")
    p.add_argument("--dwg-executable")
    p.set_defaults(handler=_cmd_capabilities)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse: 2 for usage errors, 0 for --help
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE
    try:
        return args.handler(args)
    except (SpatialFormatError, OSError) as exc:
        _emit(_failure(exc))
        return EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
