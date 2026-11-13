"""Phase 2 native SLDC technical-sheet, revision and export regressions."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from shutil import which
from subprocess import run
from tempfile import TemporaryDirectory

import pytest

from src.utils import sldc
from src.utils.spatial_formats.sgeo.document import SceneDocument as SGEODocument


def sheet():
    return {
        "id": "A101", "title": "Ground-floor plan", "page": {"size": "A3", "orientation": "landscape"},
        "elements": [
            {"id": "wall", "kind": "rect", "geometry": {"x": 10, "y": 20, "width": 40, "height": 5},
             "style": {"stroke": "#000000", "fill": "#cccccc"}},
            {"id": "line", "kind": "line", "geometry": {"start": [10, 30], "end": [80, 30]}},
            {"id": "circle", "kind": "ellipse", "geometry": {"center": [100, 50],
                                                                   "radius_x": 10, "radius_y": 12}},
            {"id": "curve", "kind": "arc", "geometry": {"center": [140, 40], "radius": 10,
                                                          "start_deg": 15, "sweep_deg": 120}},
            {"id": "outline", "kind": "polyline", "geometry": {"points": [[20, 80], [30, 90],
                                                                       [40, 80]], "closed": False}},
            {"id": "label", "kind": "text", "geometry": {"position": [40, 110],
                                                          "text": "Wall A&B", "font_size_pt": 10}},
            {"id": "dim", "kind": "dimension", "geometry": {"start": [20, 100],
                                                                  "end": [50, 100], "offset_mm": -4}},
        ],
    }


def prepared():
    base = sldc.new_package("Technical sheets", [sldc.make_block("paragraph", "Engineering notes")])
    return sldc.add_technical_sheet(base, sheet())


def test_version_gate_roundtrip_and_native_edit():
    original = sldc.new_package("Legacy authoring")
    old_bytes = sldc.dumps_package(original)
    package = sldc.add_technical_sheet(original, sheet(), author="Designer")
    assert original.version == "1.0.0" and sldc.dumps_package(original) == old_bytes
    assert package.version == "1.1.0"
    payload = sldc.dumps_package(package)
    info = sldc.inspect_package(payload)
    assert "technical-sheets" in info["requires"] and not info["unsupported_requires"]
    restored = sldc.loads_package(payload)
    assert sldc.sheets_of(restored) == [sheet()]
    assert sldc.dumps_package(restored) == payload
    old = sldc.revision_id(package)
    updated = sldc.edit_sheet_element(package, "A101", "wall", geometry={"x": 10, "y": 20,
                                   "width": 50, "height": 5}, author="Designer")
    assert sldc.revision_id(updated) != old
    assert updated.document["revision"]["number"] == package.document["revision"]["number"] + 1
    assert updated.document["revision"]["parents"] == [old]
    assert sldc.sheets_of(package)[0]["elements"][0]["geometry"]["width"] == 40
    assert sldc.sheets_of(updated)[0]["elements"][0]["geometry"]["width"] == 50


def test_full_transactional_sheet_editing_and_optimistic_conflicts():
    package = prepared()
    before = sldc.dumps_package(package)
    expected = sldc.revision_id(package)
    new_element = {"id": "detail", "kind": "line", "geometry": {"start": [2, 2], "end": [8, 2]}}
    changed = sldc.apply_sheet_operations(package, "A101", [
        {"op": "add_element", "value": new_element},
        {"op": "update_element", "id": "wall", "changes": {"visible": False, "z": 3}},
        {"op": "set_title", "title": "Revised floor plan"},
        {"op": "reorder_elements", "ids": ["detail", "wall", "line", "circle", "curve", "outline", "label", "dim"]},
    ], expected_revision_id=expected)
    updated = sldc.sheets_of(changed)[0]
    assert updated["title"] == "Revised floor plan"
    assert updated["elements"][0] == {**new_element, "z": 0}
    assert updated["elements"][1]["visible"] is False
    assert sldc.dumps_package(package) == before
    with pytest.raises(sldc.SLDCValidationError, match="conflicts"):
        sldc.apply_sheet_operations(changed, "A101", [{"op": "remove_element", "id": "detail"}],
                                    expected_revision_id=expected)
    broken = [{"op": "remove_element", "id": "detail"},
              {"op": "update_element", "id": "missing", "changes": {"visible": False}}]
    snapshot = sldc.dumps_package(changed)
    with pytest.raises(sldc.SLDCValidationError, match="missing"):
        sldc.apply_sheet_operations(changed, "A101", broken)
    assert sldc.dumps_package(changed) == snapshot
    trimmed = sldc.apply_sheet_operations(changed, "A101", [{"op": "remove_element", "id": "detail"}])
    assert len(sldc.sheets_of(trimmed)[0]["elements"]) == 7
    updated_page = sldc.apply_sheet_operations(trimmed, "A101", [{"op": "set_page", "page": {"size": "A3"}}])
    assert sldc.sheets_of(updated_page)[0]["page"] == {"size": "A3"}
    removed = sldc.remove_technical_sheet(updated_page, "A101")
    assert not sldc.sheets_of(removed)
    with pytest.raises(sldc.SLDCValidationError, match="at least one"):
        sldc.render_sheets_pdf(removed)


def test_bad_geometry_and_features_fail_closed():
    package = prepared()
    for mutate in (
        lambda s: s["elements"][0]["geometry"].update(width=10000),
        lambda s: s["elements"][0].update(id="line"),
        lambda s: s["elements"][0]["style"].update(stroke="url(javascript:evil)"),
        lambda s: s["elements"][0].update(kind="script"),
        lambda s: s["elements"][1]["geometry"].update(start=[float("nan"), 0]),
        lambda s: s["elements"][0].update(source={"format": "https://example.org", "revision": 1}),
    ):
        bad = package.copy()
        mutate(bad.document["extensions"]["slai.technical_sheets"]["sheets"][0])
        with pytest.raises(sldc.SLDCValidationError):
            sldc.validate_package(bad)
    legacy = package.copy()
    legacy.version = "1.0.0"
    with pytest.raises(sldc.SLDCUnsupportedError):
        sldc.dumps_package(legacy)


def test_incremental_cache_only_rebuilds_changed_sheet():
    package = prepared()
    second = deepcopy(sheet())
    second["id"] = "A102"
    package = sldc.add_technical_sheet(package, second)
    pages, cache, dirty = sldc.incremental_sheet_render(package)
    assert dirty == ("A101", "A102") and len(pages) == 2
    assert b"Wall A&amp;B" in pages["A101"]
    newer, cached, dirty = sldc.incremental_sheet_render(package, cache)
    assert dirty == () and newer == pages
    assert cached["A101"][1] is pages["A101"]
    changed = sldc.edit_sheet_element(package, "A101", "wall", geometry={"x": 10, "y": 20,
                                      "width": 41, "height": 5})
    after, _, dirty = sldc.incremental_sheet_render(changed, cache)
    assert dirty == ("A101",) and after["A102"] is pages["A102"]
    assert after["A101"] != pages["A101"]


def test_revision_aware_spatial_links_never_auto_resolve():
    doc = SGEODocument.new("Referenced model")
    item = doc.add_box(size=(2, 2, 2))
    record = doc.get(item)
    ref = {"format": "slai.sgeo", "document_id": doc.document_id, "item_id": item,
           "revision": doc.revision, "sha256": sha256(sldc._canonical_json(record)).hexdigest()}
    example = sheet()
    example["elements"][0]["source"] = ref
    package = sldc.add_technical_sheet(sldc.new_package("Linked"), example)
    assert sldc.resolve_sheet_references(package, {})[0]["status"] == "unavailable"
    assert sldc.resolve_sheet_references(package, {doc.document_id: doc})[0]["status"] == "resolved"
    tampered = package.copy()
    tampered.document["extensions"]["slai.technical_sheets"]["sheets"][0]["elements"][0]["source"]["sha256"] = "0"*64
    assert sldc.resolve_sheet_references(tampered, {doc.document_id: doc})[0]["status"] == "changed"
    doc.add_box(size=(1, 1, 1))
    assert sldc.resolve_sheet_references(package, {doc.document_id: doc})[0]["status"] == "stale"


def test_pdf_and_svg_export_keep_native_authority():
    package = prepared()
    before = sldc.dumps_package(package)
    svg = sldc.render_sheet_svg(sldc.sheets_of(package)[0])
    assert svg.startswith(b'<svg xmlns="http://www.w3.org/2000/svg"')
    assert b"<rect" in svg and b"<ellipse" in svg and b"<path" in svg
    try:
        pdf = sldc.render_sheets_pdf(package)
    except sldc.SLDCDependencyError:
        pytest.skip("Optional reportlab PDF backend is not installed")
    assert pdf.startswith(b"%PDF-") and len(pdf) > 200
    if which("pdftotext"):
        extraction = run(["pdftotext", "-layout", "-", "-"], input=pdf, capture_output=True, check=False)
        assert extraction.returncode == 0
        assert b"Wall A&B" in extraction.stdout and b"30 mm" in extraction.stdout
    assert sldc.dumps_package(package) == before


def test_pdf_has_correct_independent_page_dimensions_and_count():
    pypdf = pytest.importorskip("pypdf")
    first = prepared()
    second = sheet()
    second["id"] = "A102"
    second["page"] = {"size": "A4"}
    second["elements"] = second["elements"][:1]
    package = sldc.add_technical_sheet(first, second)
    pdf = sldc.render_sheets_pdf(package)
    pages = pypdf.PdfReader(BytesIO(pdf)).pages
    assert len(pages) == 2
    assert abs(float(pages[0].mediabox.width)-420*72/25.4) < 0.01
    assert abs(float(pages[0].mediabox.height)-297*72/25.4) < 0.01
    assert abs(float(pages[1].mediabox.width)-210*72/25.4) < 0.01
    assert abs(float(pages[1].mediabox.height)-297*72/25.4) < 0.01


def test_locked_objects_unicode_and_signature_revision_separation():
    package = prepared()
    from src.utils.sldc import Ed25519Signer
    signer = Ed25519Signer.generate()
    approved = sldc.sign_package(package, signer)
    assert len(approved.signatures) == 1
    edited = sldc.edit_sheet_element(approved, "A101", "wall", geometry={
        "x": 12, "y": 20, "width": 40, "height": 5})
    assert not edited.signatures and len(approved.signatures) == 1
    locked = package.copy()
    locked.document["extensions"]["slai.technical_sheets"]["sheets"][0]["elements"][0]["locked"] = True
    with pytest.raises(sldc.SLDCValidationError, match="Locked"):
        sldc.edit_sheet_element(locked, "A101", "wall", geometry={
            "x": 13, "y": 20, "width": 40, "height": 5})
    with pytest.raises(sldc.SLDCValidationError, match="Locked"):
        sldc.apply_sheet_operations(locked, "A101", [{"op": "reorder_elements", "ids": [
            "dim", "label", "outline", "curve", "circle", "line", "wall"]}])
    with pytest.raises(sldc.SLDCValidationError, match="locked"):
        sldc.remove_technical_sheet(locked, "A101")
    unlocked = sldc.apply_sheet_operations(locked, "A101", [{"op": "unlock_element", "id": "wall"}])
    assert not sldc.sheets_of(unlocked)[0]["elements"][0]["locked"]
    unicode_page = package.copy()
    unicode_page.document["extensions"]["slai.technical_sheets"]["sheets"][0]["elements"][5]["geometry"]["text"] = "Maatvoering – Ö"
    sldc.validate_package(unicode_page)
    assert "Maatvoering" in sldc.render_sheet_svg(sldc.sheets_of(unicode_page)[0]).decode()
    with pytest.raises(sldc.SLDCUnsupportedError, match="embedded Unicode"):
        sldc.render_sheets_pdf(unicode_page)


@pytest.mark.parametrize("shape_id", ["wall", "line", "circle", "curve", "outline"])
def test_independent_pdf_svg_raster_parity_of_vector_shapes(shape_id):
    if not which("pdftoppm"):
        pytest.skip("Independent Poppler PDF rasterizer unavailable")
    try:
        from cairosvg import svg2png
        from PIL import Image, ImageChops, ImageFilter
    except ImportError:
        pytest.skip("Optional independent SVG rasterization tools unavailable")
    page = sheet()
    page["elements"] = [item for item in page["elements"] if item["id"] == shape_id]
    package = sldc.add_technical_sheet(sldc.new_package("Parity"), page)
    try:
        pdf = sldc.render_sheets_pdf(package)
    except sldc.SLDCDependencyError:
        pytest.skip("Optional reportlab backend unavailable")
    with TemporaryDirectory() as tmp:
        file = Path(tmp, "page.pdf")
        file.write_bytes(pdf)
        rendered = run(["pdftoppm", "-f", "1", "-singlefile", "-r", "96", "-png",
                        str(file), str(Path(tmp, "pdf"))], capture_output=True, check=False)
        assert rendered.returncode == 0, rendered.stderr.decode(errors="replace")
        raster_pdf = Image.open(Path(tmp, "pdf.png")).convert("L")
        svg_bytes = svg2png(bytestring=sldc.render_sheet_svg(page),
                            output_width=raster_pdf.width, output_height=raster_pdf.height,
                            background_color="white")
        raster_svg = Image.open(BytesIO(svg_bytes)).convert("L")
        pdf_ink = raster_pdf.point(lambda v: 255 if v < 235 else 0)
        svg_ink = raster_svg.point(lambda v: 255 if v < 235 else 0)
        assert pdf_ink.getbbox() and svg_ink.getbbox()
        # Compare independently rasterized geometry within one pixel because
        # Poppler and Cairo do not share antialiasing/edge-coverage rules.
        for a, b in zip(pdf_ink.getbbox(), svg_ink.getbbox()):
            assert abs(a-b) <= 2, (shape_id, pdf_ink.getbbox(), svg_ink.getbbox())
        near_pdf = pdf_ink.filter(ImageFilter.MaxFilter(3))
        near_svg = svg_ink.filter(ImageFilter.MaxFilter(3))
        pd = pdf_ink.histogram()[255]
        sv = svg_ink.histogram()[255]
        pdf_covered = ImageChops.multiply(pdf_ink, near_svg).histogram()[255]
        svg_covered = ImageChops.multiply(svg_ink, near_pdf).histogram()[255]
        assert pdf_covered/pd > 0.98 and svg_covered/sv > 0.98, (shape_id, pdf_covered/pd, svg_covered/sv)


def test_incremental_page_plan_handles_columns_sections_and_long_paragraphs():
    blocks = [sldc.make_block("paragraph", "An engineering note. "*550),
              sldc.make_block("heading", "Chapter two", attributes={"page_break_before": True}),
              sldc.make_block("paragraph", "Body after explicit page break")]
    doc = sldc.new_package("Flow", blocks, sections=[{
        "id": "engineering", "page": {"size": "A4", "margins_mm": {
            "left": 15, "right": 15, "top": 25, "bottom": 20},
            "header_mm": 12, "footer_mm": 16}, "columns": 2, "column_gap_mm": 7,
    }])
    doc.version = "1.1.0"  # Native page-band/column features require SLDC 1.1.
    doc = sldc.add_technical_sheet(doc, sheet())
    pages, cache, dirty, measured = sldc.plan_document_pages(doc)
    assert len(pages) >= 3 and pages[-1]["kind"] == "technical-sheet"
    assert pages[-1]["width_mm"] == 420 and pages[-1]["height_mm"] == 297
    assert len(measured) == 3 and dirty == tuple(page["id"] for page in pages)
    assert all(item["y_mm"] >= 25 and item["y_mm"] + item["height_mm"] <= 277.00001
               for page in pages if page["kind"] == "flow" for item in page["blocks"])
    assert all(item["column"] in (0, 1) for page in pages if page["kind"] == "flow"
               for item in page["blocks"])
    unchanged, cache2, dirty2, measured2 = sldc.plan_document_pages(doc, cache)
    assert unchanged == pages and dirty2 == () and measured2 == () and cache2 == cache
    altered = doc.copy()
    altered.document["blocks"][-1]["text"] += " changed"
    revised, _, dirty3, measured3 = sldc.plan_document_pages(altered, cache)
    assert measured3 == (doc.document["blocks"][-1]["id"],) and "sheet-A101" not in dirty3
    assert revised[-1] == pages[-1]


def test_existing_sldc_11_feature_advertisements_are_readable():
    # Several previously validated 1.1 features were omitted from the reader's
    # supported-feature registry, causing the writer to reject its own output.
    pkg = sldc.new_package("Rich page band", [sldc.make_block("paragraph", "Body")],
        sections=[{"id": "main", "page": {"size": "A4"},
                   "header": {"left": "Project {page}", "runs": {"left": [
                       {"text": "Project ", "format": {"bold": True}},
                       {"text": "{page}"}]}}}])
    pkg.version = "1.1.0"
    encoded = sldc.dumps_package(pkg)
    assert "rich-page-bands" in sldc.inspect_package(encoded)["requires"]
    assert not sldc.inspect_package(encoded)["unsupported_requires"]
    assert sldc.loads_package(encoded).document == pkg.document
