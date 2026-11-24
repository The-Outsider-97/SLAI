"""Exact oblique splitting of boxes (sgeo.box_split) and the way a missing capping dependency is reported.

The reference for every kept volume is *not* the code under test: it is a dense point-sampling of the half-space (numpy), and the two halves of every
cut must add up to the whole box. Closedness is measured by SGEO's own mesh analysis, which is separate code from the splitter's self-check."""
from __future__ import annotations

import random
import unittest
from copy import deepcopy

import numpy as np

from src.utils.spatial_formats.common.errors import DependencyUnavailableError, GeometryValidationError
from src.utils.spatial_formats.sgeo.analysis import analyze_mesh
from src.utils.spatial_formats.sgeo.box_split import split_box
from src.utils.spatial_formats.sgeo.modifier_ops import _split

BOX = {"kind": "box", "center": [0.5, -1.0, 2.0], "size": [2.0, 3.0, 1.5]}
UNIT = {"kind": "box", "center": [0.5, 0.5, 0.5], "size": [1.0, 1.0, 1.0]}


def signed_volume(mesh):
    v = np.asarray(mesh["vertices"], dtype=float)
    f = np.asarray(mesh["faces"], dtype=int)
    v = v - v[0]                                                                                    # relative to a vertex, as the code under test does
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def sampled_fraction(box, normal, origin, n=140):
    """Fraction of the box on the positive side of the plane, from an n^3 grid of cell centres (error of order 1/n, no splitter code involved)."""
    c, s = np.asarray(box["center"]), np.asarray(box["size"])
    axes = [c[i] - s[i] / 2 + (np.arange(n) + 0.5) * s[i] / n for i in range(3)]
    nn = np.asarray(normal, dtype=float)
    nn /= np.linalg.norm(nn)
    o = np.asarray(origin, dtype=float)
    inside = 0
    for x in axes[0]:
        yy, zz = np.meshgrid(axes[1], axes[2], indexing="ij")
        d = nn[0] * (x - o[0]) + nn[1] * (yy - o[1]) + nn[2] * (zz - o[2])
        inside += int((d >= 0).sum())
    return inside / n ** 3


def box_volume(box):
    return box["size"][0] * box["size"][1] * box["size"][2]


class ObliqueBoxSplit(unittest.TestCase):
    def check(self, box, normal, origin, *, expect=None):
        kept, outcome, info = split_box(box, normal, origin)
        opposite, outcome2, _ = split_box(box, [-x for x in normal], origin)
        total = box_volume(box)
        v1 = signed_volume(kept) if kept is not None and kept["kind"] == "mesh" else (total if outcome == "unchanged" else 0.0)
        v2 = signed_volume(opposite) if opposite is not None and opposite["kind"] == "mesh" else (total if outcome2 == "unchanged" else 0.0)
        self.assertAlmostEqual(v1 + v2, total, delta=1e-9 * total, msg=(normal, origin))            # the two halves make the whole
        self.assertAlmostEqual(v1 / total, sampled_fraction(box, normal, origin), delta=0.02, msg=(normal, origin))
        if expect:
            self.assertEqual(outcome, expect)
        if outcome == "cut":
            report = analyze_mesh(kept)                                                              # SGEO's own analysis: closed, oriented, one piece
            self.assertEqual((report.boundary_edge_count, report.non_manifold_edge_count, report.inconsistent_winding_edge_count, report.connected_components, report.is_closed_oriented_surface),
                             (0, 0, 0, 1, True), (normal, origin))
            n = np.asarray(normal, float) / np.linalg.norm(normal)
            d = (np.asarray(kept["vertices"]) - np.asarray(origin)) @ n
            self.assertGreaterEqual(float(d.min()), -1e-9)                                           # nothing on the removed side
            lo = np.asarray(box["center"]) - np.asarray(box["size"]) / 2 - 1e-9
            hi = np.asarray(box["center"]) + np.asarray(box["size"]) / 2 + 1e-9
            self.assertTrue(((np.asarray(kept["vertices"]) >= lo) & (np.asarray(kept["vertices"]) <= hi)).all())
        return outcome, v1

    def test_thirty_random_oblique_planes_through_the_box(self):
        rng = random.Random(20261122)
        cuts = 0
        for _ in range(30):
            normal = [rng.uniform(-1, 1) for _ in range(3)]
            origin = [BOX["center"][i] + rng.uniform(-0.45, 0.45) * BOX["size"][i] for i in range(3)]
            outcome, _ = self.check(BOX, normal, origin)
            cuts += outcome == "cut"
        self.assertEqual(cuts, 30)                                                                   # every plane passed through the box: none refused, none empty

    def test_special_planes_have_known_volumes(self):
        # the diagonal plane x + y = 1 through opposite edges halves the unit cube; the plane x + y + z = 1.5 through its centre does too
        for normal, origin in (([1, 1, 0], [0.5, 0.5, 0]), ([1, 1, 1], [0.5, 0.5, 0.5]), ([1, -1, 0], [0.5, 0.5, 0.5]), ([0.3, -2, 5], [0.5, 0.5, 0.5])):
            _, v = self.check(UNIT, normal, origin)
            self.assertAlmostEqual(v, 0.5, places=12, msg=normal)
        _, v = self.check(UNIT, [1, 1, 1], [1, 0, 0])                                                # through three corners: the corner pyramid is on the other side
        self.assertAlmostEqual(v, 1 - 1 / 6, places=12)

    def test_planes_that_touch_or_miss_leave_the_box_or_remove_it(self):
        self.assertEqual(self.check(UNIT, [1, 1, 1], [3, 0, 0])[0], "empty")                          # the box is entirely on the negative side
        self.assertEqual(self.check(UNIT, [1, 1, 1], [-1, 0, 0])[0], "unchanged")                    # entirely on the positive side
        self.assertEqual(self.check(UNIT, [1, 1, 1], [3, 0, 0])[1], 0.0)
        self.assertEqual(self.check(UNIT, [1, 1, 1], [0, 0, 0], expect="unchanged")[0], "unchanged")  # touches only at the corner (0,0,0), box on the positive side
        self.assertEqual(self.check(UNIT, [-1, -1, -1], [0, 0, 0], expect="empty")[0], "empty")       # the same plane, other side: only the corner touches
        self.assertEqual(self.check(UNIT, [0, 0, 1], [0, 0, 1], expect="empty")[0], "empty")          # lies on the top face; the box is below it

    def test_a_plane_within_tolerance_of_a_face_does_not_make_a_sliver(self):
        self.assertEqual(split_box(UNIT, [0, 0, 1], [0, 0, 1 - 1e-12])[1], "empty")
        self.assertEqual(split_box(UNIT, [1e-10, 0, 1], [0, 0, 1e-12])[1], "unchanged")
        outcome, v = self.check(UNIT, [1e-6, 0, 1], [0, 0, 0.5])                                      # a genuine, nearly horizontal cut is still cut
        self.assertEqual(outcome, "cut")
        self.assertAlmostEqual(v, 0.5, delta=1e-6)

    def test_extreme_aspect_ratios_and_scales(self):
        for box in ({"kind": "box", "center": [0, 0, 0], "size": [1000.0, 0.01, 2.0]}, {"kind": "box", "center": [1e4, 2e4, 3e4], "size": [1e-3, 2e-3, 3e-3]}):
            for normal in ([1, 2, 3], [0.2, 1, -0.7], [-1, 0.001, 0.4]):
                self.check(box, normal, box["center"])

    def test_the_modifier_uses_the_exact_path_and_says_so(self):
        modifier = {"parameters": {"normal": [1, 1, 0.5], "origin": [0.5, 0.5, 0.5], "side": "positive"}, "evaluation_policy": {"mode": "exact"}}
        geometry, details = _split({"geometry": deepcopy(UNIT)}, modifier)
        self.assertEqual((details["backend"], details["approximate"], details["empty"], geometry["kind"]), ("analytic-box-plane-split", False, False, "mesh"))
        modifier["parameters"]["side"] = "negative"
        other, _ = _split({"geometry": deepcopy(UNIT)}, modifier)
        self.assertAlmostEqual(signed_volume(geometry) + signed_volume(other), 1.0, places=12)

    def test_a_zero_normal_is_refused(self):
        with self.assertRaises(GeometryValidationError):
            split_box(UNIT, [0, 0, 0], [0, 0, 0])


class MissingCappingDependency(unittest.TestCase):
    def test_a_missing_module_is_reported_as_a_dependency_not_as_invalid_geometry(self):
        try:
            import trimesh
        except ImportError:
            self.skipTest("trimesh is not installed here")
        cube = {"kind": "mesh", "vertices": [[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)],
                "faces": [[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]]}
        modifier = {"parameters": {"normal": [1, 1, 1], "origin": [0.5, 0.5, 0.5], "side": "positive"}, "evaluation_policy": {"mode": "tolerance_controlled"}}
        original = trimesh.Trimesh.slice_plane

        def missing(self, *a, **k):
            raise ModuleNotFoundError("No module named 'shapely'", name="shapely")

        trimesh.Trimesh.slice_plane = missing
        try:
            with self.assertRaises(DependencyUnavailableError) as raised:
                _split({"geometry": cube}, modifier)
        finally:
            trimesh.Trimesh.slice_plane = original
        self.assertEqual(raised.exception.context, {"module": "shapely"})


if __name__ == "__main__":
    unittest.main()
