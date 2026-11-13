"""Deterministic bounded convex extrusion and revolution regression tests."""
from math import pi
import pytest
from src.utils.spatial_formats.sgeo.solids import extrude, revolve
from src.utils.spatial_formats.sgeo.analysis import analyze_mesh
from src.utils.spatial_formats.common.errors import ValidationError

RECTANGLE=[(0,0),(2,0),(2,1),(0,1)]
PROFILE=[(1,0),(2,0),(2,1),(1,1)]

def test_extruded_prism_is_closed_oriented_and_has_expected_volume():
    prism=extrude(RECTANGLE,3)
    info=analyze_mesh(prism)
    assert info.is_closed_oriented_surface
    assert info.signed_volume==pytest.approx(6.0)
    assert info.surface_area==pytest.approx(22.0)

def test_extrusion_tilt_preserves_z_volume_component():
    prism=extrude(RECTANGLE,5,direction=(3,0,4))
    info=analyze_mesh(prism)
    assert info.is_closed_oriented_surface
    assert info.signed_volume==pytest.approx(8.0)

@pytest.mark.parametrize("angle", [pi,2*pi])
def test_revolution_caps_or_closes_without_open_edges(angle):
    result=revolve(PROFILE,sweep=angle,segments=64)
    info=analyze_mesh(result)
    assert info.is_closed_oriented_surface
    assert info.signed_volume>0
    expected=3*pi*(angle/(2*pi))
    assert info.signed_volume==pytest.approx(expected,rel=0.01)

@pytest.mark.parametrize("profile", [
    [(0,0),(1,0),(1,1),(0,1)],
    [(0,0),(2,2),(0,2),(2,0)],
    [(0,0),(1,0),(2,0),(0,1)],
])
def test_invalid_revolution_profiles_fail_closed(profile):
    with pytest.raises(ValidationError):
        revolve(profile)

def test_horizontal_extrusion_is_rejected():
    with pytest.raises(ValidationError):
        extrude(RECTANGLE,2,direction=(1,0,0))
