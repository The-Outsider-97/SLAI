"""Explicit conversion-fidelity contracts for supported STL/SVG transport.

These wrappers report losses rather than implying preservation of CAD
semantics through presentation/triangle-only interchange formats.
"""
from __future__ import annotations
from ..common.helpers import canonical_json, digest
from ..verification.loss_report import ConversionReport
from .stl import export_binary_stl, import_binary_stl
from .svg import export_svg, import_svg

def mesh_to_stl(geometry):
    """Export a triangle mesh and a machine-readable lossy conversion report."""
    result=export_binary_stl(geometry)
    report=ConversionReport(source_digest=digest(canonical_json(geometry)),
        target_digest=digest(result), adapter="binary-stl",adapter_version="1",
        accuracy_class="approximate",
        dropped=["semantic object identifiers","parametric family information",
                 "scene hierarchy","units","materials","unreferenced vertices"],
        approximations=["vertex coordinates encoded as float32"],
        warnings=["STL has no standard physical unit declaration"])
    return result, report

def stl_to_mesh(data):
    geometry=import_binary_stl(data)
    report=ConversionReport(source_digest=digest(data),
        target_digest=digest(canonical_json(geometry)),adapter="binary-stl",
        adapter_version="1",accuracy_class="approximate",
        dropped=["STL face attributes","original indexed-vertex topology"],
        assumptions=["triangle coordinates use the caller-selected length unit"],
        warnings=["triangle vertices are not welded during import"])
    return geometry,report

def drawing_to_svg(drawing):
    markup=export_svg(drawing)
    report=ConversionReport(source_digest=digest(canonical_json(drawing.as_dict())),
        target_digest=digest(markup.encode("utf-8")),adapter="svg-subset",
        adapter_version="1",accuracy_class="approximate",
        dropped=["document revision","stable drawing identities on reimport",
                 "metadata not represented by SVG subset","source entity styles"],
        warnings=["SVG display coordinates may use a different y-axis orientation"])
    return markup,report

def svg_to_drawing(data,*,unit="mm"):
    drawing=import_svg(data,unit=unit)
    raw=data.encode("utf-8") if isinstance(data,str) else data
    report=ConversionReport(source_digest=digest(raw),
        target_digest=digest(canonical_json(drawing.as_dict())),
        adapter="svg-subset",adapter_version="1",accuracy_class="approximate",
        assumptions=[f"SVG numbers interpreted as {unit}"],
        dropped=["source SVG styles and unknown semantic metadata"],
        warnings=["new SLAI entity identifiers were allocated"])
    return drawing,report
