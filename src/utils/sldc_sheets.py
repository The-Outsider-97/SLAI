"""SLDC 1.1 native technical-sheet geometry, rendering and revision coordination.

The extension is *inert data*: reading an SLDC never evaluates a linked model,
opens a file, or executes a drawing command. SVG and PDF are explicit exports.
The native sheet remains editable independently of either derived rendition.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from html import escape
from io import BytesIO
from math import ceil, cos, radians, sin

SHEETS_EXTENSION = "slai.technical_sheets"
MAX_SHEETS = 256
MAX_SHEET_ELEMENTS = 8192
MAX_TOTAL_ELEMENTS = 50_000
MAX_PLANNED_PAGES = 10_000
_KINDS = frozenset({"line", "rect", "ellipse", "polyline", "arc", "text", "dimension"})
_PAGES = {"A3": (297.0, 420.0), "A4": (210.0, 297.0), "A5": (148.0, 210.0),
          "Letter": (215.9, 279.4), "Tabloid": (279.4, 431.8)}


def _sldc():
    from . import sldc
    return sldc


def _number(value, label, low=-5000, high=5000):
    return _sldc()._bounded_number(value, low, high, label)


def _page_size(page):
    s = _sldc()
    s._validate_page(page, "technical sheet page")
    if "size" in page:
        width, height = s._PAGE_SIZES_MM[page["size"]]
    else:
        width, height = page["width_mm"], page["height_mm"]
    if page.get("orientation") == "landscape":
        width, height = height, width
    return float(width), float(height)


def _point(pt, label, width, height):
    if not isinstance(pt, list) or len(pt) != 2:
        raise _sldc().SLDCValidationError(f"{label} must be an [x, y] coordinate")
    x = _number(pt[0], label + ".x", 0, width)
    y = _number(pt[1], label + ".y", 0, height)
    return x, y


def _reference(ref, label):
    s = _sldc()
    if not isinstance(ref, dict):
        raise s.SLDCValidationError(f"{label} must be a reference object")
    s._check_keys(ref, {"format", "document_id", "item_id", "revision", "sha256"}, label)
    if ref.get("format") not in ("slai.sgeo", "slai.sdrw"):
        raise s.SLDCValidationError(f"{label} must reference SGEO or SDRW")
    for key in ("document_id", "item_id"):
        s._id_field(ref.get(key), f"{label}.{key}")
    s._bounded_int(ref.get("revision"), 1, 2**31-1, f"{label}.revision")
    if "sha256" in ref and (not isinstance(ref["sha256"], str)
                            or not s._HEX_SHA256.fullmatch(ref["sha256"])):
        raise s.SLDCValidationError(f"{label}.sha256 is not a SHA-256 digest")


def _validate_element(item, label, width, height):
    s = _sldc()
    if not isinstance(item, dict):
        raise s.SLDCValidationError(f"{label} must be an object")
    s._check_keys(item, {"id", "kind", "geometry", "style", "z", "visible", "locked", "source"}, label)
    s._id_field(item.get("id"), label + ".id")
    kind = item.get("kind")
    if not isinstance(kind, str) or kind not in _KINDS:
        raise s.SLDCValidationError(f"{label} has an unsupported vector kind")
    if "source" in item:
        _reference(item["source"], label + ".source")
    if "z" in item:
        s._bounded_int(item["z"], -10000, 10000, label + ".z")
    for key in ("visible", "locked"):
        if key in item and type(item[key]) is not bool:
            raise s.SLDCValidationError(f"{label}.{key} must be Boolean")
    style = item.get("style", {})
    if not isinstance(style, dict):
        raise s.SLDCValidationError(f"{label}.style must be an object")
    s._check_keys(style, {"stroke", "fill", "stroke_width_mm", "dash_mm"}, label + ".style")
    for key in ("stroke", "fill"):
        if key in style and style[key] is not None and (not isinstance(style[key], str)
                or not s._COLOR.fullmatch(style[key])):
            raise s.SLDCValidationError(f"{label}.style.{key} must be a hex color or null")
    if "stroke_width_mm" in style:
        _number(style["stroke_width_mm"], label + ".style.stroke_width_mm", 0.01, 10)
    if "dash_mm" in style:
        dash = style["dash_mm"]
        if not isinstance(dash, list) or not 1 <= len(dash) <= 8:
            raise s.SLDCValidationError(f"{label}.style.dash_mm must have 1-8 lengths")
        for length in dash:
            _number(length, label + ".style.dash_mm", 0.01, 100)
    geo = item.get("geometry")
    if not isinstance(geo, dict):
        raise s.SLDCValidationError(f"{label}.geometry must be an object")
    fields = {
        "line": {"start", "end"}, "dimension": {"start", "end", "offset_mm"},
        "rect": {"x", "y", "width", "height"},
        "ellipse": {"center", "radius_x", "radius_y"},
        "arc": {"center", "radius", "start_deg", "sweep_deg"},
        "polyline": {"points", "closed"}, "text": {"position", "text", "font_size_pt"},
    }[kind]
    if set(geo) != fields:
        raise s.SLDCValidationError(f"{label}.geometry fields do not match {kind}")
    if (kind in {"line", "dimension", "arc"} or
            (kind == "polyline" and geo["closed"] is False)) and style.get("fill") is not None:
        raise s.SLDCValidationError(f"{label} cannot fill open linework")
    if kind == "text" and (style.get("stroke") is not None or "dash_mm" in style):
        raise s.SLDCValidationError(f"{label} vector text supports fill, not outlined/dashed glyphs")
    if kind in {"line", "dimension"}:
        a, b = _point(geo["start"], label, width, height), _point(geo["end"], label, width, height)
        if a == b:
            raise s.SLDCValidationError(f"{label} has zero length")
        if kind == "dimension":
            distance = _number(geo["offset_mm"], label + ".offset_mm", -500, 500)
            dx, dy = b[0]-a[0], b[1]-a[1]
            length = (dx*dx+dy*dy)**0.5
            shift = (-dy*distance/length, dx*distance/length)
            for pt in (a, b):
                _point([pt[0]+shift[0], pt[1]+shift[1]], label + ".offset", width, height)
    elif kind == "rect":
        x = _number(geo["x"], label + ".x", 0, width)
        y = _number(geo["y"], label + ".y", 0, height)
        rw = _number(geo["width"], label + ".width", 0.001, width)
        rh = _number(geo["height"], label + ".height", 0.001, height)
        if x + rw > width or y + rh > height:
            raise s.SLDCValidationError(f"{label} rectangle exceeds its page")
    elif kind in {"ellipse", "arc"}:
        x, y = _point(geo["center"], label, width, height)
        rx = _number(geo["radius_x" if kind == "ellipse" else "radius"], label, 0.001, 5000)
        ry = _number(geo["radius_y"], label, 0.001, 5000) if kind == "ellipse" else rx
        if x-rx < 0 or x+rx > width or y-ry < 0 or y+ry > height:
            raise s.SLDCValidationError(f"{label} ellipse/arc exceeds its page")
        if kind == "arc":
            _number(geo["start_deg"], label, -360, 360)
            _number(geo["sweep_deg"], label, -360, 360)
            if geo["sweep_deg"] == 0:
                raise s.SLDCValidationError(f"{label} arc needs nonzero sweep")
    elif kind == "polyline":
        points = geo["points"]
        if (not isinstance(points, list) or not 2 <= len(points) <= 10000
                or type(geo["closed"]) is not bool):
            raise s.SLDCValidationError(f"{label} invalid polyline")
        for pt in points:
            _point(pt, label, width, height)
        if geo["closed"] and len(points) < 3:
            raise s.SLDCValidationError(f"{label} closed polyline needs three points")
        if len({tuple(pt) for pt in points}) < (3 if geo["closed"] else 2):
            raise s.SLDCValidationError(f"{label} polyline is geometrically degenerate")
    else:
        _point(geo["position"], label, width, height)
        text = geo["text"]
        if not isinstance(text, str) or len(text) > 2000 or any(ord(ch) < 32 and ch not in "\t" for ch in text):
            raise s.SLDCValidationError(f"{label} invalid vector text")
        _number(geo["font_size_pt"], label, 4, 144)


def validate_technical_sheets(value):
    """Validate bounded editable geometry without resolving external references."""
    s = _sldc()
    if not isinstance(value, dict) or set(value) != {"version", "sheets"} or type(value["version"]) is not int or value["version"] != 1:
        raise s.SLDCValidationError("Unsupported technical sheet contract")
    sheets = value["sheets"]
    if not isinstance(sheets, list) or len(sheets) > MAX_SHEETS:
        raise s.SLDCValidationError("Technical sheet count exceeds 256")
    seen_sheets, total = set(), 0
    for index, sheet in enumerate(sheets):
        label = f"technical sheet {index}"
        if not isinstance(sheet, dict):
            raise s.SLDCValidationError(f"{label} must be an object")
        s._check_keys(sheet, {"id", "title", "page", "elements"}, label)
        sid = s._id_field(sheet.get("id"), label + ".id")
        if sid in seen_sheets:
            raise s.SLDCValidationError("Duplicate technical sheet id")
        seen_sheets.add(sid)
        s._text_field(sheet, "title", label, 256, required=True)
        width, height = _page_size(sheet.get("page"))
        items = sheet.get("elements")
        if not isinstance(items, list) or len(items) > MAX_SHEET_ELEMENTS:
            raise s.SLDCValidationError("Technical sheet element count exceeds 8192")
        total += len(items)
        if total > MAX_TOTAL_ELEMENTS:
            raise s.SLDCValidationError("Technical sheet element count exceeds global budget")
        seen = set()
        for j, item in enumerate(items):
            _validate_element(item, f"{label}.elements[{j}]", width, height)
            if item["id"] in seen:
                raise s.SLDCValidationError(f"{label} has duplicate element IDs")
            seen.add(item["id"])


def sheets_of(package):
    return deepcopy(package.document.get("extensions", {}).get(SHEETS_EXTENSION,
                   {"version": 1, "sheets": []})["sheets"])


def add_technical_sheet(package, sheet, *, author=None):
    """Add an independently editable, bounded vector page as a new revision."""
    s = _sldc()
    s.validate_package(package)
    validate_technical_sheets({"version": 1, "sheets": [sheet]})
    if s._parse_version(package.version)[0] != 1 or s._parse_version(package.version)[1] > 1:
        raise s.SLDCUnsupportedError("Native technical sheets require editable SLDC 1.x")
    upgraded = package.copy()
    upgraded.version = "1.1.0"
    result = s.next_revision(upgraded, author=author, note="Add native technical sheet")
    ext = result.document.setdefault("extensions", {})
    data = ext.setdefault(SHEETS_EXTENSION, {"version": 1, "sheets": []})
    data["sheets"].append(deepcopy(sheet))
    s.validate_package(result)
    return result


def edit_sheet_element(package, sheet_id, element_id, *, geometry=None, style=None, author=None):
    """Copy-on-write native edit; signed history remains in the previous revision."""
    changes = {}
    if geometry is not None:
        changes["geometry"] = geometry
    if style is not None:
        changes["style"] = style
    if not changes:
        raise _sldc().SLDCValidationError("Sheet element edit needs a geometry or style change")
    return apply_sheet_operations(package, sheet_id, [{"op": "update_element", "id": element_id,
                                                       "changes": changes}], author=author)


def apply_sheet_operations(package, sheet_id, operations, *, author=None, expected_revision_id=None):
    """Transactional native shape and sheet CRUD with optimistic revision checking.

    Supported operations: add/update/remove/unlock/reorder elements, set page,
    set title. The entire batch validates before it becomes a new revision;
    errors cannot partially change the existing package.
    """
    s = _sldc()
    s.validate_package(package)
    if package.version != "1.1.0":
        raise s.SLDCUnsupportedError("Native technical-sheet edits require SLDC 1.1.0")
    if expected_revision_id is not None and s.revision_id(package) != expected_revision_id:
        raise s.SLDCValidationError("Technical-sheet edit conflicts with the expected revision")
    if not isinstance(operations, list) or not 1 <= len(operations) <= 1000:
        raise s.SLDCValidationError("Sheet edit requires 1 to 1000 operations")
    result = s.next_revision(package, author=author, note="Edit native technical sheet")
    target = next((sheet for sheet in result.document["extensions"][SHEETS_EXTENSION]["sheets"]
                   if sheet["id"] == sheet_id), None)
    if target is None:
        raise s.SLDCValidationError("Technical sheet not found")
    for operation in operations:
        if not isinstance(operation, dict):
            raise s.SLDCValidationError("Sheet operation must be an object")
        kind = operation.get("op")
        if kind == "add_element" and set(operation) == {"op", "value"}:
            if not isinstance(operation["value"], dict):
                raise s.SLDCValidationError("New sheet element must be an object")
            target["elements"].append(deepcopy(operation["value"]))
        elif kind in {"update_element", "remove_element", "unlock_element"}:
            required = {"op", "id", "changes"} if kind == "update_element" else {"op", "id"}
            if set(operation) != required:
                raise s.SLDCValidationError("Invalid sheet element operation fields")
            ident = s._id_field(operation["id"], "sheet operation id")
            index = next((i for i, item in enumerate(target["elements"]) if item["id"] == ident), None)
            if index is None:
                raise s.SLDCValidationError("Sheet operation references a missing element")
            current = target["elements"][index]
            if kind == "unlock_element":
                current["locked"] = False
                continue
            if current.get("locked", False):
                raise s.SLDCValidationError("Locked technical-sheet element cannot be edited")
            if kind == "remove_element":
                target["elements"].pop(index)
            else:
                changes = operation["changes"]
                if (not isinstance(changes, dict) or not changes
                        or set(changes) - {"geometry", "style", "z", "visible", "locked", "source"}):
                    raise s.SLDCValidationError("Unsupported sheet element update fields")
                current.update(deepcopy(changes))
        elif kind == "reorder_elements" and set(operation) == {"op", "ids"}:
            ids = operation["ids"]
            by_id = {item["id"]: item for item in target["elements"]}
            if (not isinstance(ids, list) or len(ids) != len(by_id)
                    or not all(isinstance(ident, str) for ident in ids)
                    or len(set(ids)) != len(ids) or set(ids) != set(by_id)):
                raise s.SLDCValidationError("Sheet reorder must list every element once")
            if any(item.get("locked", False) for item in target["elements"]):
                raise s.SLDCValidationError("Locked technical-sheet elements cannot be reordered")
            target["elements"] = [by_id[ident] for ident in ids]
            for item in target["elements"]:
                item["z"] = 0  # Document order becomes the unambiguous draw order.
        elif kind == "set_page" and set(operation) == {"op", "page"}:
            target["page"] = deepcopy(operation["page"])
        elif kind == "set_title" and set(operation) == {"op", "title"}:
            target["title"] = operation["title"]
        else:
            raise s.SLDCValidationError("Unknown or malformed sheet editing operation")
    s.validate_package(result)
    return result


def remove_technical_sheet(package, sheet_id, *, author=None, expected_revision_id=None):
    """Remove one entire native sheet in a new revision, preserving the old file."""
    s = _sldc()
    s.validate_package(package)
    if package.version != "1.1.0":
        raise s.SLDCUnsupportedError("Native technical-sheet edits require SLDC 1.1.0")
    if expected_revision_id is not None and s.revision_id(package) != expected_revision_id:
        raise s.SLDCValidationError("Technical-sheet removal conflicts with the expected revision")
    result = s.next_revision(package, author=author, note="Remove native technical sheet")
    sheets = result.document.get("extensions", {}).get(SHEETS_EXTENSION, {}).get("sheets", [])
    original_count = len(sheets)
    if any(item.get("locked", False) for sheet in sheets if sheet["id"] == sheet_id
           for item in sheet["elements"]):
        raise s.SLDCValidationError("Sheet with locked elements cannot be removed")
    sheets[:] = [sheet for sheet in sheets if sheet["id"] != sheet_id]
    if len(sheets) == original_count:
        raise s.SLDCValidationError("Technical sheet not found")
    s.validate_package(result)
    return result


def resolve_sheet_references(package, sources):
    """Resolve only against explicitly supplied local documents; no automatic I/O."""
    s = _sldc()
    from .spatial_formats.common.errors import ReferenceError as SpatialReferenceError
    s.validate_package(package)
    if not isinstance(sources, Mapping):
        raise s.SLDCValidationError("Source resolver must be a mapping of document IDs")
    report = []
    for sheet in sheets_of(package):
        for item in sheet["elements"]:
            ref = item.get("source")
            if ref is None:
                continue
            source = sources.get(ref["document_id"])
            status = "unavailable"
            if source is not None:
                if getattr(source, "FORMAT_ID", None) != ref["format"] or source.document_id != ref["document_id"]:
                    status = "mismatch"
                elif source.revision != ref["revision"]:
                    status = "stale"
                else:
                    try:
                        record = source.get(ref["item_id"])
                    except (KeyError, SpatialReferenceError):
                        record = None
                    if record is None:
                        status = "missing"
                    elif "sha256" in ref and s._sha256(s._canonical_json(record)) != ref["sha256"]:
                        status = "changed"
                    else:
                        status = "resolved"
            report.append({"sheet_id": sheet["id"], "element_id": item["id"], "status": status})
    return report


def _rgb(value):
    return tuple(int(value[k:k+2], 16)/255 for k in (1, 3, 5))


def _items(sheet):
    return [item for _, item in sorted(enumerate(sheet["elements"]),
                                      key=lambda pair: (pair[1].get("z", 0), pair[0]))
            if item.get("visible", True)]


def render_sheet_svg(sheet):
    """SVG export of editable native geometry; SVG is never accepted as SLDC input."""
    validate_technical_sheets({"version": 1, "sheets": [sheet]})
    w, h = _page_size(sheet["page"])
    out = [(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:g}mm" height="{h:g}mm" '
            f'viewBox="0 0 {w:g} {h:g}">')]
    def p(x):
        return f"{x:g}"
    for item in _items(sheet):
        geo, kind, style = item["geometry"], item["kind"], item.get("style", {})
        stroke = style.get("stroke", "#000000")
        fill = style.get("fill")
        attrs = (f'stroke="{stroke or "none"}" fill="{fill or "none"}" '
                 f'stroke-width="{p(style.get("stroke_width_mm", 0.25))}"')
        if "dash_mm" in style:
            attrs += ' stroke-dasharray="' + ",".join(p(v) for v in style["dash_mm"]) + '"'
        if kind in ("line", "dimension"):
            a, b = geo["start"], geo["end"]
            if kind == "dimension":
                dx, dy = b[0]-a[0], b[1]-a[1]
                length = (dx*dx+dy*dy)**0.5
                shift = [-dy*geo["offset_mm"]/length, dx*geo["offset_mm"]/length]
                a, b = [a[i]+shift[i] for i in range(2)], [b[i]+shift[i] for i in range(2)]
            out.append(f'<line x1="{p(a[0])}" y1="{p(a[1])}" x2="{p(b[0])}" y2="{p(b[1])}" {attrs}/>')
            if kind == "dimension":
                mid = [(a[i]+b[i])/2 for i in range(2)]
                out.append(f'<text x="{p(mid[0])}" y="{p(mid[1]-1)}" font-size="3" '
                           f'fill="{stroke or "#000000"}" stroke="none" text-anchor="middle">'
                           f'{p(length)} mm</text>')
        elif kind == "rect":
            out.append(f'<rect x="{p(geo["x"])}" y="{p(geo["y"])}" '
                       f'width="{p(geo["width"])}" height="{p(geo["height"])}" {attrs}/>')
        elif kind == "ellipse":
            out.append(f'<ellipse cx="{p(geo["center"][0])}" cy="{p(geo["center"][1])}" '
                       f'rx="{p(geo["radius_x"])}" ry="{p(geo["radius_y"])}" {attrs}/>')
        elif kind == "polyline":
            points = " ".join(f'{p(x)},{p(y)}' for x, y in geo["points"])
            tag = "polygon" if geo["closed"] else "polyline"
            out.append(f'<{tag} points="{points}" {attrs}/>')
        elif kind == "arc":
            cx, cy = geo["center"]
            angle = radians(geo["start_deg"])
            end = radians(geo["start_deg"]+geo["sweep_deg"])
            radius = geo["radius"]
            a = (cx+radius*cos(angle), cy+radius*sin(angle))
            b = (cx+radius*cos(end), cy+radius*sin(end))
            large = int(abs(geo["sweep_deg"]) > 180)
            sweep = int(geo["sweep_deg"] > 0)
            if abs(geo["sweep_deg"]) == 360:
                out.append(f'<circle cx="{p(cx)}" cy="{p(cy)}" r="{p(radius)}" {attrs}/>')
            else:
                out.append(f'<path d="M {p(a[0])},{p(a[1])} A {p(radius)},{p(radius)} 0 '
                           f'{large},{sweep} {p(b[0])},{p(b[1])}" {attrs}/>')
        elif kind == "text":
            x, y = geo["position"]
            out.append(f'<text x="{p(x)}" y="{p(y)}" font-size="{p(geo["font_size_pt"])}pt" '
                       f'fill="{style.get("fill") or "#000000"}" stroke="none">'
                       f'{escape(geo["text"])}</text>')
    out.append("</svg>")
    return "\n".join(out).encode("utf-8")


def render_sheets_pdf(package):
    """Explicit PDF export of native sheets, using reportlab when installed.

    The PDF is a read-only derivative; all editable primitives remain in SLDC.
    This renderer does not purport to typeset arbitrary rich-text SLDC blocks.
    """
    s = _sldc()
    s.validate_package(package)
    try:
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise s.SLDCDependencyError("Technical sheet PDF export requires reportlab") from exc
    sheets = sheets_of(package)
    if not sheets:
        raise s.SLDCValidationError("PDF technical-sheet export requires at least one native sheet")
    output = BytesIO()
    pdf = canvas.Canvas(output, pagesize=(595, 842), pageCompression=0, invariant=1)
    scale = 72/25.4
    for sheet in sheets:
        width, height = _page_size(sheet["page"])
        pdf.setPageSize((width*scale, height*scale))
        def xy(pt, page_height=height):
            return pt[0]*scale, (page_height-pt[1])*scale
        for item in _items(sheet):
            geo, kind, style = item["geometry"], item["kind"], item.get("style", {})
            pdf.saveState()
            if style.get("stroke", "#000000") is not None:
                pdf.setStrokeColorRGB(*_rgb(style.get("stroke", "#000000")))
            if style.get("fill"):
                pdf.setFillColorRGB(*_rgb(style["fill"]))
            pdf.setLineWidth(style.get("stroke_width_mm", 0.25)*scale)
            if "dash_mm" in style:
                pdf.setDash([v*scale for v in style["dash_mm"]])
            outline = int(style.get("stroke", "#000000") is not None)
            filled = int(bool(style.get("fill")))
            if kind in ("line", "dimension"):
                a, b = geo["start"], geo["end"]
                if kind == "dimension":
                    dx, dy = b[0]-a[0], b[1]-a[1]
                    length = (dx*dx+dy*dy)**0.5
                    shift = [-dy*geo["offset_mm"]/length, dx*geo["offset_mm"]/length]
                    a, b = [a[i]+shift[i] for i in range(2)], [b[i]+shift[i] for i in range(2)]
                if outline:
                    pdf.line(*xy(a), *xy(b))
                if kind == "dimension":
                    m = [(a[i]+b[i])/2 for i in range(2)]
                    pdf.setFont("Helvetica", 8.5)
                    pdf.drawCentredString(*xy([m[0], m[1]-1]), f"{length:g} mm")
            elif kind == "rect":
                pdf.rect(geo["x"]*scale, (height-geo["y"]-geo["height"])*scale,
                         geo["width"]*scale, geo["height"]*scale, stroke=outline, fill=filled)
            elif kind == "ellipse":
                cx, cy = geo["center"]
                rx, ry = geo["radius_x"], geo["radius_y"]
                pdf.ellipse((cx-rx)*scale, (height-cy-ry)*scale,
                            (cx+rx)*scale, (height-cy+ry)*scale, stroke=outline, fill=filled)
            elif kind == "polyline":
                path = pdf.beginPath()
                path.moveTo(*xy(geo["points"][0]))
                for pt in geo["points"][1:]:
                    path.lineTo(*xy(pt))
                if geo["closed"]:
                    path.close()
                pdf.drawPath(path, stroke=outline, fill=filled and geo["closed"])
            elif kind == "arc":
                cx, cy = geo["center"]
                radius = geo["radius"]
                if outline:
                    pdf.arc((cx-radius)*scale, (height-cy-radius)*scale,
                            (cx+radius)*scale, (height-cy+radius)*scale,
                            startAng=-geo["start_deg"], extent=-geo["sweep_deg"])
            elif kind == "text":
                # Standard PDF base fonts cannot safely encode arbitrary Unicode.
                if any(ord(ch) > 126 or ord(ch) < 32 for ch in geo["text"]):
                    raise s.SLDCUnsupportedError("PDF vector text needs an embedded Unicode font")
                pdf.setFont("Helvetica", geo["font_size_pt"])
                if style.get("fill"):
                    pdf.setFillColorRGB(*_rgb(style["fill"]))
                pdf.drawString(*xy(geo["position"]), geo["text"])
            pdf.restoreState()
        pdf.showPage()
    pdf.save()
    return output.getvalue()


def incremental_sheet_render(package, cache=None):
    """Return sheet SVG bytes, new content-addressed cache and dirty sheet IDs.

    Cache is caller-owned and disposable; never part of the signed SLDC file.
    Unchanged sheet renderings retain their byte-for-byte data identity.
    """
    s = _sldc()
    s.validate_package(package)
    if cache is not None and not isinstance(cache, Mapping):
        raise s.SLDCValidationError("Sheet rendering cache must be a mapping")
    old = cache or {}
    pages, updated, changed = {}, {}, []
    for sheet in sheets_of(package):
        key = s._sha256(s._canonical_json(sheet))
        entry = old.get(sheet["id"])
        if isinstance(entry, tuple) and len(entry) == 2 and entry[0] == key and isinstance(entry[1], bytes):
            page = entry[1]
        else:
            page = render_sheet_svg(sheet)
            changed.append(sheet["id"])
        pages[sheet["id"]] = page
        updated[sheet["id"]] = (key, page)
    changed.extend(sorted(set(old) - set(updated)))
    return pages, updated, tuple(changed)


def plan_document_pages(package, cache=None):
    """Conservative *estimated* pagination for rich-text flow plus exact sheet pages.

    Never substitutes for a shaping engine's final glyph/line-break decisions.
    It respects native physical page sizes, margins, bands, columns, section
    boundaries, explicit page breaks and unsplittable content. Cached block
    height estimates survive unrelated edits; downstream pages always reflow.

    Returns ``(pages, cache, changed_page_ids, remeasured_block_ids)``.
    """
    s = _sldc()
    s.validate_package(package)
    if s._parse_version(package.version)[0] != 1:
        raise s.SLDCUnsupportedError("Page planning requires SLDC 1.x")
    doc = package.document
    sections = doc.get("sections") or [{"id": "default", "page": {"size": "A4",
                          "margins_mm": {"top": 20, "right": 20, "bottom": 20, "left": 20}}}]
    sections = {section["id"]: section for section in sections}
    first_section = next(iter(sections))
    old = cache if isinstance(cache, dict) else {}
    saved = old.get("measurements", {})
    measurements = {}
    remeasured = []
    pages = []
    current = None
    section_id = None
    column = 0
    y = 0.0

    def page_parameters(section):
        w, h = _page_size(section.get("page", {"size": "A4"}))
        page = section.get("page", {})
        margins = page.get("margins_mm", {})
        left, right = margins.get("left", 20), margins.get("right", 20)
        top = max(margins.get("top", 20), page.get("header_mm", 0))
        bottom = max(margins.get("bottom", 20), page.get("footer_mm", 0))
        count = section.get("columns", 1)
        gap = section.get("column_gap_mm", 5)
        width = (w-left-right-gap*(count-1))/count
        if width < 10 or h-top-bottom < 20:
            raise s.SLDCValidationError("Section has insufficient usable page/column space")
        return w, h, left, top, width, h-bottom, gap, count

    def new_page():
        nonlocal current, column, y
        if len(pages) >= MAX_PLANNED_PAGES:
            raise s.SLDCValidationError("Estimated pagination exceeds the 10,000-page budget")
        page = sections[section_id]
        w, h, _, top, _, _, _, _ = page_parameters(page)
        current = {"id": f"flow-{len(pages)+1}", "kind": "flow", "section_id": section_id,
                   "width_mm": w, "height_mm": h, "estimated": True, "blocks": []}
        pages.append(current)
        column, y = 0, top

    def advance():
        nonlocal column, y
        _, _, _, top, _, _, _, count = page_parameters(sections[section_id])
        if column+1 == count:
            new_page()
        else:
            column += 1
            y = top

    def style_of(block):
        props = {}
        group = doc.get("styles", {}).get("paragraph", {})
        name = block.get("style")
        names = []
        while name is not None:
            names.append(name)
            name = group[name].get("based_on")
        for name in reversed(names):
            props.update(group[name])
        props.update(block.get("attributes", {}))
        return props

    def measured(block, width):
        props = style_of(block)
        signature = s._sha256(s._canonical_json({"block": block, "width_mm": width,
                                                 "style": props}))
        cached = saved.get(block["id"])
        if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == signature:
            measurement = cached[1]
        else:
            font = props.get("font_size_pt", 16 if block["type"] == "heading" else 11)
            line = font*props.get("line_height", 1.25)*25.4/72
            max_chars = max(1, int(width / (font*0.53*25.4/72)))
            count = sum(max(1, ceil(len(part)/max_chars)) for part in block["text"].split("\n"))
            kind = block["type"]
            if kind == "table":
                count = max(1, len(block["table"]["rows"]))
                line = max(line, 7)
            elif kind == "figure":
                count = 1
                line = block["figure"].get("height_mm", 45)
            elif kind == "equation":
                count, line = 1, 14
            elif kind == "embed":
                count, line = 1, 20
            before = props.get("space_before_pt", 0)*25.4/72
            after = props.get("space_after_pt", 0)*25.4/72
            measurement = {"height_mm": count*line + before+after,
                           "space_before_mm": before, "space_after_mm": after,
                           "line_count": count, "line_height_mm": line,
                           "splittable": kind in ("paragraph", "heading", "list_item", "quote", "code")}
            remeasured.append(block["id"])
        measurements[block["id"]] = (signature, measurement)
        return measurement

    for block in doc["blocks"]:
        section = block.get("section", section_id or first_section)
        if section != section_id:
            section_id = section
            new_page()
        elif block.get("attributes", {}).get("page_break_before"):
            new_page()
        sec = sections[section_id]
        _, _, left, top, width, bottom, gap, _ = page_parameters(sec)
        measure = measured(block, width)
        full = measure["height_mm"]
        line_height = measure["line_height_mm"]
        remaining = max(0, measure["line_count"])
        split = measure["splittable"]
        if not split and full > bottom-top:
            raise s.SLDCUnsupportedError(f"Block {block['id']} cannot fit inside a page column")
        before = measure["space_before_mm"]
        if before+line_height > bottom-top:
            raise s.SLDCUnsupportedError(f"Block {block['id']} has an excessive top spacing")
        if y+before+line_height > bottom:
            advance()
        y += before
        while remaining:
            if y+min(full, line_height) > bottom+1e-8:
                advance()
            available = bottom-y
            if split:
                lines = min(remaining, max(1, int(available / line_height)))
                height = min(available, lines*line_height)
            else:
                if remaining*line_height + measure["space_after_mm"] > available:
                    advance()
                    y += before
                    continue
                lines, height = remaining, remaining*line_height
            consumed = measure["line_count"] - remaining
            current["blocks"].append({"block_id": block["id"], "column": column,
                                      "x_mm": left + column*(width+gap), "y_mm": round(y, 6),
                                      "width_mm": width, "height_mm": round(height, 6),
                                      "line_start": consumed, "line_end": consumed+lines})
            y += height
            remaining -= lines
            full = max(0, full-height)
        y = min(bottom, y+measure["space_after_mm"])

    sheet_records = sheets_of(package)
    sheets_by_id = {sheet["id"]: sheet for sheet in sheet_records}
    for sheet in sheet_records:
        w, h = _page_size(sheet["page"])
        pages.append({"id": "sheet-"+sheet["id"], "kind": "technical-sheet",
                      "sheet_id": sheet["id"], "width_mm": w, "height_mm": h,
                      "estimated": False, "elements": len(sheet["elements"])})
    previous_pages = old.get("page_fingerprints", {})
    page_fingerprints = {}
    changed = []
    for page in pages:
        # Block content contributes to a page fingerprint even when its size
        # did not change, so re-rendering invalidates correctly.
        contents = ([s._sha256(s._canonical_json(sections[page["section_id"]])),
                     *[measurements[item["block_id"]][0] for item in page["blocks"]]]
                    if page["kind"] == "flow" else
                    [s._sha256(s._canonical_json(sheets_by_id[page["sheet_id"]]))])
        fingerprint = s._sha256(s._canonical_json({"page": page, "contents": contents}))
        page_fingerprints[page["id"]] = fingerprint
        if previous_pages.get(page["id"]) != fingerprint:
            changed.append(page["id"])
    changed.extend(sorted(set(previous_pages) - set(page_fingerprints)))
    return pages, {"measurements": measurements, "page_fingerprints": page_fingerprints}, tuple(changed), tuple(remeasured)
