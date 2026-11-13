"""Extended CPU-only geometric correctness and serialization regressions."""
from math import pi
import pytest

from src.utils.spatial_formats.common import ValidationError
from src.utils.spatial_formats.sdrw.entities import make_entity, entity_bounds
from src.utils.spatial_formats.sdrw.operations import entity_length, enclosed_area, transform_entity
from src.utils.spatial_formats.sgeo.geometry import local_bounds
from src.utils.spatial_formats.sgeo.analysis import analyze_mesh, transformed_bounds
from src.utils.spatial_formats.sgeo.tessellation import tessellate


def test_arc_bounds_are_tight():
    arc=make_entity("arc",{"center":[0,0],"radius":2,"start_angle":0,"end_angle":pi/2})
    lower,upper=entity_bounds(arc)
    assert lower==pytest.approx((0,0),abs=1e-10)
    assert upper==pytest.approx((2,2))
    assert entity_length(arc)==pytest.approx(pi)


def test_closed_area_measurements():
    rect=make_entity("polyline",{"points":[[0,0],[4,0],[4,2],[0,2]],"closed":True})
    assert enclosed_area(rect)==pytest.approx(8)
    assert entity_length(rect)==pytest.approx(12)
    circle=make_entity("circle",{"center":[2,3],"radius":2})
    assert enclosed_area(circle)==pytest.approx(4*pi)


def test_open_or_crossing_area_refused():
    with pytest.raises(ValidationError):
        enclosed_area(make_entity("line",{"start":[0,0],"end":[1,0]}))
    bowtie=make_entity("polyline",{"points":[[0,0],[2,2],[0,2],[2,0]],"closed":True})
    with pytest.raises(ValidationError):
        enclosed_area(bowtie)


def test_similarity_transform_retains_analytic_circle():
    circle=make_entity("circle",{"center":[1,2],"radius":3})
    moved=transform_entity(circle,[0,-2,10, 2,0,20, 0,0,1])
    assert moved["geometry"]["center"]==pytest.approx([6,22])
    assert moved["geometry"]["radius"]==pytest.approx(6)
    assert moved["id"]!=circle["id"]
    with pytest.raises(ValidationError):
        transform_entity(circle,[2,0,0, 0,1,0, 0,0,1])


@pytest.mark.parametrize("primitive",[
    {"kind":"box","center":[0,0,0],"size":[2,2,2]},
    {"kind":"cylinder","center":[0,0,0],"radius":1,"height":2},
    {"kind":"sphere","center":[0,0,0],"radius":1},
])
def test_primitives_tessellate_to_closed_oriented_surfaces(primitive):
    mesh=tessellate(primitive,radial_segments=12,latitude_bands=8)
    diagnostics=analyze_mesh(mesh)
    assert diagnostics.is_closed_oriented_surface
    assert diagnostics.boundary_edge_count==0
    assert diagnostics.inconsistent_winding_edge_count==0
    assert diagnostics.connected_components==1
    assert diagnostics.signed_volume>0


def test_open_mesh_diagnostics_do_not_claim_solid_volume():
    mesh={"kind":"mesh","vertices":[[0,0,0],[1,0,0],[0,1,0]],"faces":[[0,1,2]]}
    diagnostics=analyze_mesh(mesh)
    assert diagnostics.boundary_edge_count==3
    assert diagnostics.signed_volume is None
    assert diagnostics.surface_area==pytest.approx(0.5)


def test_analytic_bounds_under_translation():
    box={"kind":"box","center":[0,0,0],"size":[2,4,6]}
    original=local_bounds(box)
    output=transformed_bounds(original,[1,0,0,5,0,1,0,4,0,0,1,3,0,0,0,1])
    assert output==((-1+5,-2+4,-3+3),(1+5,2+4,3+3))


def test_tessellation_rejects_unsafe_resolution():
    with pytest.raises(ValidationError):
        tessellate({"kind":"sphere","center":[0,0,0],"radius":1},radial_segments=1)
