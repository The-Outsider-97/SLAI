"""SLDC minor-version and page-band feature compatibility regressions.

The reader preserves the version recorded in a validated archive. Feature
requirements are derived from document semantics, including section page bands.
"""
from __future__ import annotations

from io import BytesIO
import json
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import pytest

from src.utils import sldc


def _package_with_page_bands():
    pkg = sldc.new_package(
        "Page bands",
        [sldc.make_block("paragraph", "Content")],
        sections=[
            {
                "id": "main",
                "page": {"size": "A4", "header_mm": 12.0, "footer_mm": 15.0},
            }
        ],
    )
    pkg.version = "1.1.0"
    return pkg


def _reseal_manifest(data: bytes, *, changes: dict) -> bytes:
    """Change manifest metadata without modifying the content or digest."""
    with ZipFile(BytesIO(data)) as source:
        manifest = json.loads(source.read("manifest.json"))
        manifest.update(changes)
        members = [
            (name, source.read(name))
            for name in source.namelist()
            if name not in {"mimetype", "manifest.json"}
        ]
        mimetype = source.read("mimetype")
    out = BytesIO()
    with ZipFile(out, "w", allowZip64=False) as dest:
        dest.writestr("mimetype", mimetype, compress_type=ZIP_STORED)
        dest.writestr(
            "manifest.json",
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(),
            compress_type=ZIP_DEFLATED,
        )
        for name, payload in members:
            dest.writestr(name, payload, compress_type=ZIP_DEFLATED)
    return out.getvalue()


def test_page_bands_advertised_and_preserved_on_roundtrip():
    package = _package_with_page_bands()
    required = sldc._required_features(package.document)
    assert "page-bands" in required
    data = sldc.dumps_package(package)
    report = sldc.inspect_package(data)
    assert "page-bands" in report["requires"]
    assert report["unsupported_requires"] == []
    decoded = sldc.loads_package(data)
    assert decoded.version == "1.1.0"
    assert decoded.document["sections"] == package.document["sections"]


def test_page_bands_rejected_for_v1_0_write():
    package = _package_with_page_bands()
    package.version = sldc.FORMAT_VERSION
    with pytest.raises(sldc.SLDCUnsupportedError):
        sldc.dumps_package(package)


def test_missing_page_bands_capability_is_rejected_on_read():
    package = _package_with_page_bands()
    data = sldc.dumps_package(package)
    missing = _reseal_manifest(
        data,
        changes={"requires": [
            name for name in sldc.inspect_package(data)["requires"]
            if name != "page-bands"
        ]},
    )
    with pytest.raises(sldc.SLDCValidationError, match="omits a feature"):
        sldc.loads_package(missing)


def test_newer_minor_version_is_preserved_not_rewritten():
    package = sldc.new_package("Version preservation", [sldc.make_block("paragraph", "Hello")])
    data = sldc.dumps_package(package)
    newer = _reseal_manifest(data, changes={"version": "1.7.2"})
    decoded = sldc.loads_package(newer)
    assert decoded.version == "1.7.2"
    assert sldc.inspect_package(newer)["newer_than_known"]
    with pytest.raises(sldc.SLDCUnsupportedError):
        sldc.dumps_package(decoded)
