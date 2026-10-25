"""Contract tests for explicit SVG/STL conversion-loss reporting."""
from src.utils.spatial_formats.adapters.exchange import (
    mesh_to_stl, stl_to_mesh, drawing_to_svg, svg_to_drawing
)
from src.utils.spatial_formats.sdrw import Drawing

def test_stl_conversion_reports_its_losses():
    mesh={"kind":"mesh","vertices":[[0,0,0],[1,0,0],[0,1,0]],"faces":[[0,1,2]]}
    data, outgoing=mesh_to_stl(mesh)
    result, incoming=stl_to_mesh(data)
    assert result["kind"]=="mesh"
    assert outgoing.target_digest==incoming.source_digest
    assert outgoing.accuracy_class=="approximate"
    assert any("units" in dropped for dropped in outgoing.dropped)
    assert incoming.warnings

def test_svg_conversion_reports_identifier_loss():
    drawing=Drawing()
    drawing.add("line",{"start":[0,0],"end":[1,1]})
    svg, report=drawing_to_svg(drawing)
    imported, incoming=svg_to_drawing(svg)
    assert imported.entities[0]["geometry"]==drawing.entities[0]["geometry"]
    assert report.target_digest==incoming.source_digest
    assert imported.entities[0]["id"]!=drawing.entities[0]["id"]
    assert report.accuracy_class=="approximate"
