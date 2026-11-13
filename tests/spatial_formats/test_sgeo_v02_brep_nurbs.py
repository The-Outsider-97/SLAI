"""SGEO 0.2 NURBS curves/surfaces and the native B-rep solid model: validation, builders, tessellation accuracy."""
from __future__ import annotations

import copy
import math

import pytest

from src.utils.sgeo import SGEODocument
from src.utils.spatial_formats.common.errors import (
    GeometryValidationError, ResourceLimitError, SpatialFormatError, ValidationError)
from src.utils.spatial_formats.sgeo import brep_build as build
from src.utils.spatial_formats.sgeo import nurbs
from src.utils.spatial_formats.sgeo.analysis import analyze_mesh
from src.utils.spatial_formats.sgeo.brep import brep_info, scale_brep, validate_brep
from src.utils.spatial_formats.sgeo.brep_mesh import analyze_brep, brep_bounds, tessellate_brep
from src.utils.spatial_formats.sgeo.geometry import validate_geometry

CLAMPED2 = [0, 0, 0, 1, 1, 1]
ARCH = [[[0, 0, 0], [1, 0, 1], [2, 0, 0]], [[0, 1, 1], [1, 1, 3], [2, 1, 1]], [[0, 2, 0], [1, 2, 1], [2, 2, 0]]]


def curve(**changes):
    value = {"kind": "nurbs_curve", "control_points": [[0, 0, 0], [1, 1, 0], [2, 0, 0]], "degree": 2,
             "knots": list(CLAMPED2)}
    value.update(changes)
    return value


def surface(**changes):
    value = {"kind": "nurbs_surface", "degree_u": 2, "degree_v": 2, "knots_u": list(CLAMPED2),
             "knots_v": list(CLAMPED2), "control_points": copy.deepcopy(ARCH)}
    value.update(changes)
    return value


# --------------------------------------------------------------------------- NURBS
def test_valid_nurbs_curve_and_surface_pass():
    assert validate_geometry(curve())["degree"] == 2
    assert validate_geometry(surface())["degree_u"] == 2


@pytest.mark.parametrize("bad", [
    {"degree": 3},                                       # fewer than degree + 1 control points
    {"knots": [0, 0, 0, 1, 0.5, 1]},                     # decreasing
    {"knots": [0, 1, 2, 3, 4, 5]},                       # not clamped
    {"knots": [0, 0, 1, 1]},                             # wrong length
    {"weights": [1, 0, 1]},                              # non-positive weight
    {"weights": [1, 1]},                                 # wrong length
    {"control_points": [[0, 0, 0], [1, float("nan"), 0], [2, 0, 0]]},
    {"degree": 0},
    {"degree": 13},                                      # beyond the supported degree
])
def test_invalid_nurbs_curves_are_rejected(bad):
    with pytest.raises(SpatialFormatError):
        validate_geometry(curve(**bad))


def test_invalid_nurbs_surfaces_are_rejected():
    ragged = copy.deepcopy(ARCH)
    ragged[1].pop()
    for bad in ({"control_points": ragged}, {"knots_u": [0, 0, 1, 1]}, {"degree_v": 0},
                {"weights": [[1, 1, 1], [1, 1, 1]]}):
        with pytest.raises(SpatialFormatError):
            validate_geometry(surface(**bad))


def test_rational_quadratic_arc_is_an_exact_circle():
    w = math.sqrt(2) / 2
    arc = validate_geometry(curve(control_points=[[1, 0, 0], [1, 1, 0], [0, 1, 0]], weights=[1, w, 1]))
    for t in (0.0, 0.25, 0.5, 0.8, 1.0):
        x, y, z = nurbs.curve_point(arc, t)
        assert math.hypot(x, y) == pytest.approx(1.0, abs=1e-12) and z == 0
    assert nurbs.curve_point(arc, 0.0) == pytest.approx((1, 0, 0))
    assert nurbs.curve_point(arc, 1.0) == pytest.approx((0, 1, 0))


def test_surface_evaluation_matches_a_bilinear_patch():
    patch = validate_geometry({"kind": "nurbs_surface", "degree_u": 1, "degree_v": 1, "knots_u": [0, 0, 1, 1],
                               "knots_v": [0, 0, 1, 1],
                               "control_points": [[[0, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 1]]]})
    assert nurbs.surface_point(patch, 0.5, 0.5) == pytest.approx((0.5, 0.5, 0.25))
    assert nurbs.surface_point(patch, 0, 0) == pytest.approx((0, 0, 0))
    assert nurbs.surface_point(patch, 1, 1) == pytest.approx((1, 1, 1))


def test_curve_bounds_enclose_the_curve():
    lo, hi = nurbs.curve_bounds(validate_geometry(curve()))
    assert lo[0] <= 0 and hi[0] >= 2 and hi[1] >= 0.5       # a quadratic reaches y = 0.5 at its peak


def test_nurbs_documents_fill_default_knots():
    d = SGEODocument.new("n")
    c = d.add_nurbs_curve([(0, 0, 0), (1, 2, 0), (2, -1, 0), (3, 1, 0)])
    s = d.add_nurbs_surface(ARCH, degree_u=2, degree_v=2)
    assert d.get(c)["geometry"]["knots"] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert len(d.get(s)["geometry"]["knots_u"]) == 6
    assert d.surface_area(s) == pytest.approx(5.87, abs=0.05)
    lo, hi = d.world_bounds(s)
    assert lo == pytest.approx((0, 0, 0)) and hi[0] == pytest.approx(2) and hi[2] > 1
    with pytest.raises(ValidationError):
        d.surface_area(c)                                    # a curve has no area


def test_a_nurbs_surface_tessellates_and_reports_diagnostics():
    d = SGEODocument.new("n")
    s = d.add_nurbs_surface(ARCH, degree_u=2, degree_v=2)
    mesh = d.tessellate(s, radial_segments=24, uvs=True)
    assert len(mesh["uvs"]) == len(mesh["vertices"])
    info = d.analyze(s)
    assert info["closed"] is False and info["volume"] is None
    assert info["area"] == pytest.approx(5.87, abs=0.05)


# --------------------------------------------------------------------------- builders
SOLIDS = {
    "box": (lambda: build.brep_box((1, 2, 3)), 6.0, 22.0, 2, 1e-12),
    "cylinder": (lambda: build.brep_cylinder(1, 2), 2 * math.pi, 6 * math.pi, 2, 0.02),
    "sphere": (lambda: build.brep_sphere(1), 4 / 3 * math.pi, 4 * math.pi, 2, 0.02),
    "torus": (lambda: build.brep_torus(3, 1), 2 * math.pi ** 2 * 3, 4 * math.pi ** 2 * 3, 0, 0.03),
    "cone": (lambda: build.brep_cylinder(2, 3, radius_top=1), math.pi * 3 * (4 + 2 + 1) / 3, None, 2, 0.02),
}


@pytest.mark.parametrize("name", sorted(SOLIDS))
def test_builders_make_closed_watertight_outward_solids(name):
    make, volume, area, euler, tolerance = SOLIDS[name]
    brep = make()
    info = validate_brep(brep)
    assert info["closed"] and info["euler"] == euler
    facts = analyze_brep(brep, radial_segments=64)
    assert facts["closed"] and facts["watertight"] and facts["outward"]
    assert facts["volume"] == pytest.approx(volume, rel=max(tolerance, 1e-9))
    if area is not None:
        assert facts["area"] == pytest.approx(area, rel=max(tolerance, 1e-9))


def test_accuracy_converges_with_resolution():
    brep = build.brep_sphere(1)
    exact = 4 / 3 * math.pi
    errors = [abs(analyze_brep(brep, radial_segments=n)["volume"] - exact) for n in (16, 32, 64, 128)]
    assert errors[0] > errors[1] > errors[2] > errors[3]
    assert errors[-1] / exact < 0.005


def test_tessellated_breps_are_closed_oriented_meshes():
    for name in ("box", "cylinder", "sphere", "torus"):
        mesh = tessellate_brep(SOLIDS[name][0](), radial_segments=32)
        info = analyze_mesh(mesh)
        assert info.is_closed_oriented_surface and info.signed_volume > 0, name


def test_tessellation_is_deterministic_and_bounded():
    brep = build.brep_sphere(1)
    assert tessellate_brep(brep, radial_segments=24) == tessellate_brep(brep, radial_segments=24)
    for bad in (0, 3, 129, 2.5, True):
        with pytest.raises(SpatialFormatError):
            tessellate_brep(brep, radial_segments=bad)


def test_brep_uvs_cover_each_face():
    mesh = tessellate_brep(build.brep_sphere(1), radial_segments=24, uvs=True)
    flat = [c for uv in mesh["uvs"] for c in uv]
    assert len(mesh["uvs"]) == len(mesh["vertices"]) and 0.0 <= min(flat) and max(flat) <= 1.0 + 1e-9
    assert "uvs" not in tessellate_brep(build.brep_sphere(1), radial_segments=24)


def test_scaling_a_brep_scales_every_measure():
    brep = build.brep_cylinder(1, 2)
    big = scale_brep(brep, 10)
    facts = analyze_brep(big)
    assert facts["volume"] == pytest.approx(2 * math.pi * 1000, rel=0.01)
    assert facts["area"] == pytest.approx(6 * math.pi * 100, rel=0.01)
    assert validate_brep(big)["closed"]


def test_bounds_are_tight_for_flat_solids_and_enclose_curved_ones():
    lo, hi = brep_bounds(build.brep_box((1, 2, 3), center=(1, 1, 1)))
    assert lo == pytest.approx((0.5, 0, -0.5)) and hi == pytest.approx((1.5, 2, 2.5))
    lo, hi = brep_bounds(build.brep_sphere(1))
    assert all(v <= -0.99 for v in lo) and all(v >= 0.99 for v in hi)


# --------------------------------------------------------------------------- trimmed faces
def test_extrusion_with_a_hole_has_exact_measures():
    outer = [(0, 0), (4, 0), (4, 4), (0, 4)]
    hole = [(1, 1), (1, 2), (2, 2), (2, 1)]                  # either winding is accepted
    brep = build.brep_extrusion(outer, 2, holes=[hole])
    facts = analyze_brep(brep)
    # the annular top and bottom faces count once each: V - E + F = 16 - 24 + 10 (genus 1 once the 2 inner loops count)
    assert facts["closed"] and facts["watertight"] and facts["euler_characteristic"] == 2
    assert facts["volume"] == pytest.approx((16 - 1) * 2)
    assert facts["area"] == pytest.approx(2 * 15 + 2 * (16 + 4))


def test_concave_profile_extrusion_is_exact():
    profile = [(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)]          # an L shape
    facts = analyze_brep(build.brep_extrusion(profile, 1))
    assert facts["watertight"] and facts["volume"] == pytest.approx(7.0) and facts["area"] == pytest.approx(30.0)


def test_extrusion_rejects_bad_profiles():
    square = [(0, 0), (4, 0), (4, 4), (0, 4)]
    with pytest.raises(SpatialFormatError):
        build.brep_extrusion([(0, 0), (2, 2), (2, 0), (0, 2)], 1)           # self-intersecting
    with pytest.raises(SpatialFormatError):
        build.brep_extrusion(square, 1, holes=[[(5, 5), (6, 5), (6, 6)]])   # hole outside
    with pytest.raises(SpatialFormatError):
        build.brep_extrusion(square, 1, holes=[[(1, 1), (3, 1), (3, 3)], [(2, 1.5), (3.5, 1.5), (3.5, 3.5)]])
    with pytest.raises(SpatialFormatError):
        build.brep_extrusion(square, 0)
    with pytest.raises(SpatialFormatError):
        build.brep_extrusion(square[:2], 1)


def test_a_nurbs_sheet_becomes_an_open_brep():
    sheet = build.brep_from_nurbs_surface(surface())
    info = validate_brep(sheet)
    assert info["closed"] is False
    facts = analyze_brep(sheet, radial_segments=24)
    assert facts["closed"] is False and facts["volume"] is None and facts["area"] == pytest.approx(5.87, abs=0.05)


# --------------------------------------------------------------------------- B-rep validation
def mutated(change):
    brep = build.brep_box((1, 2, 3))
    change(brep)
    return brep


@pytest.mark.parametrize("change", [
    lambda b: b["faces"].pop(),                                              # leaves an unused surface
    lambda b: b["faces"][0].__setitem__("surface", 99),
    lambda b: b["faces"][0]["loops"][0][0].__setitem__("edge", 99),
    lambda b: b["faces"][0]["loops"][0][0].__setitem__("forward", not b["faces"][0]["loops"][0][0]["forward"]),
    lambda b: b.__setitem__("extra", 1),
    lambda b: b["vertices"].__setitem__(0, [9, 9, 9]),                       # edge no longer meets its vertex
    lambda b: b["vertices"].__setitem__(0, [float("nan"), 0, 0]),
    lambda b: b["faces"][0].__setitem__("loops", []),
    lambda b: b["edges"][0].__setitem__("range", [2.0, 0.0]),
])
def test_corrupt_breps_are_rejected(change):
    with pytest.raises(SpatialFormatError):
        validate_brep(mutated(change))
    with pytest.raises(SpatialFormatError):
        validate_geometry(mutated(change))


def test_an_open_shell_is_valid_but_not_a_solid():
    brep = build.brep_box((1, 1, 1))
    # keep the topology consistent: drop the last face and the surface only it used
    last = brep["faces"].pop()
    unused = last["surface"]
    brep["surfaces"].pop(unused)
    for face in brep["faces"]:
        if face["surface"] > unused:
            face["surface"] -= 1
    info = validate_brep(brep)
    assert info["closed"] is False
    assert analyze_brep(brep)["volume"] is None


def test_an_inverted_solid_is_reported_not_silently_accepted():
    brep = build.brep_box((1, 1, 1))
    for face in brep["faces"]:
        face["reversed"] = not face.get("reversed", False)
        face["loops"] = [[{**c, "forward": not c["forward"]} for c in reversed(loop)] for loop in face["loops"]]
    facts = analyze_brep(brep)
    assert facts["outward"] is False and facts["volume"] < 0
    d = SGEODocument.new("x")
    inverted = d.add_brep(brep)
    with pytest.raises(ValidationError):
        d.volume(inverted)                                   # volume demands an outward solid
    assert "brep.inverted" in {i.code for i in d.validate().issues}


def test_resource_limits_are_enforced():
    brep = build.brep_box((1, 1, 1))
    brep["vertices"] = brep["vertices"] * 20000
    with pytest.raises((ResourceLimitError, GeometryValidationError)):
        validate_brep(brep)


# --------------------------------------------------------------------------- document API
def test_brep_objects_in_a_scene():
    d = SGEODocument.new("b")
    box = d.add_brep_box(width=1, depth=2, height=3, name="box")
    cyl = d.add_brep_cylinder(1, 2, position=(5, 0, 0))
    sph = d.add_brep_sphere(1, position=(0, 5, 0))
    tor = d.add_brep_torus(3, 1, position=(0, 0, 9))
    ext = d.add_brep_extrusion([(0, 0), (2, 0), (2, 2), (0, 2)], 1, holes=[[(0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5)]])
    assert d.volume(box) == pytest.approx(6.0) and d.surface_area(box) == pytest.approx(22.0)
    assert d.volume(ext) == pytest.approx(3.0)
    assert d.volume(sph, resolution=96) == pytest.approx(4 / 3 * math.pi, rel=0.01)
    assert d.analyze(tor)["euler_characteristic"] == 0
    assert d.inspect()["geometry_kinds"]["brep"] == 5
    lo, hi = d.world_bounds(cyl)
    assert 3.9 < lo[0] <= 4 and 6 <= hi[0] < 6.1                  # a conservative box: encloses, within a few percent
    assert d.validate().ok and len(d.find(type_name="brep")) == 5 and cyl in d


def test_transforms_apply_to_brep_measures():
    d = SGEODocument.new("b", unit="m")
    box = d.add_brep_box(width=1, depth=1, height=1)
    d.set_transform(box, [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1])
    assert d.volume(box) == pytest.approx(8.0) and d.surface_area(box) == pytest.approx(24.0)
    d.convert_unit("mm")
    assert d.volume(box, unit="m") == pytest.approx(8.0)


def test_brep_survives_save_open_and_scene_copy(tmp_path):
    d = SGEODocument.new("b")
    sph = d.add_brep_sphere(1, name="ball")
    box = d.add_brep_box(width=1, depth=2, height=3, name="box")
    d.save(tmp_path / "b.sgeo")
    with SGEODocument.open(tmp_path / "b.sgeo") as back:
        assert back.get(sph)["geometry"] == d.get(sph)["geometry"]
        assert back.volume(box) == pytest.approx(6.0)
        assert back.content_digest() == d.content_digest()
        assert "brep" in back.manifest()["requires"]
    from src.utils.spatial_formats import inspect_package
    assert {"brep"} <= set(inspect_package(tmp_path / "b.sgeo").requires)


def test_failed_brep_edits_leave_the_scene_untouched():
    d = SGEODocument.new("b")
    box = d.add_brep_box(width=1, depth=1, height=1)
    before = d.content_digest()
    broken = build.brep_box((1, 1, 1))
    broken["faces"].pop()
    with pytest.raises(SpatialFormatError):
        d.add_brep(broken)
    with pytest.raises(SpatialFormatError):
        d.set_geometry(box, {"kind": "brep", **broken})
    assert d.content_digest() == before


def test_stl_export_of_a_brep_is_a_closed_solid(tmp_path):
    d = SGEODocument.new("b", unit="mm")
    d.add_brep_cylinder(5, 10)
    report = d.export_stl(tmp_path / "c.stl")
    assert (tmp_path / "c.stl").stat().st_size > 84
    assert report.accuracy_class == "approximate"
    from src.utils.spatial_formats.adapters.exchange import stl_to_mesh
    mesh, _ = stl_to_mesh((tmp_path / "c.stl").read_bytes())
    from src.utils.spatial_formats.sgeo.meshing import weld_vertices
    welded, _ = weld_vertices(mesh)                        # STL carries no shared vertices
    assert analyze_mesh(welded).is_closed_oriented_surface


def test_brep_info_reports_topology():
    info = brep_info(build.brep_box((1, 1, 1)))
    assert (info["vertices"], info["edges"], info["faces"], info["euler"]) == (8, 12, 6, 2)
