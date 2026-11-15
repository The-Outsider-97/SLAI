"""Regression tests for persistent, selectable SGEO face topology."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from .common.errors import GeometryValidationError, ReferenceError, ValidationError
from .sgeo.analysis import analyze_mesh
from .sgeo.document import SceneDocument


class ParametricFaceModifierTests(unittest.TestCase):
    def setUp(self):
        self.scene = SceneDocument.new(unit="m")
        self.wall = self.scene.add_box(size=(6, .3, 3))
        self.front = next(ident for ident, face in self.scene.face_references(self.wall).items()
                          if face["axis"] == 1 and face["sign"] == -1)

    def add_grid(self, *, rows=2, columns=3):
        return self.scene.add_modifier(
            self.wall, kind="sgeo.face_subdivide",
            target_subelement={"kind": "face", "id": self.front},
            parameters={"rows": rows, "columns": columns})

    def add_extrusion(self, face_id, *, depth=.1, mode="add", direction="normal"):
        return self.scene.add_modifier(
            self.wall, kind="sgeo.face_extrude",
            target_subelement={"kind": "face", "id": face_id},
            parameters={"depth": depth, "mode": mode, "direction": direction,
                        "thickness_limit": .3},
            evaluation_policy={"mode": "tolerance_controlled", "absolute": 1e-7})

    def add_bevel(self, face_id):
        return self.scene.add_modifier(
            self.wall, kind="sgeo.face_bevel",
            target_subelement={"kind": "face", "id": face_id},
            parameters={"depth": .03, "width": .02, "edge": "u_min", "mode": "inline",
                        "thickness_limit": .3},
            evaluation_policy={"mode": "tolerance_controlled", "absolute": 1e-7})

    def assert_solid(self, geometry):
        result = analyze_mesh(geometry)
        self.assertTrue(result.is_closed_oriented_surface)
        self.assertGreater(result.signed_volume, 0)

    def test_real_grid_regions_and_conforming_closed_mesh(self):
        self.add_grid()
        evaluated = self.scene.evaluate_modifiers()[-1]
        cells = evaluated["diagnostics"]["subface_ids"]
        self.assertEqual(len(cells), 6)
        self.assertEqual(len(set(cells)), 6)
        self.assertEqual(len(self.scene.face_references(self.wall)), 12)
        self.assertAlmostEqual(self.scene.face_references(self.wall)[self.front]["physical_thickness"], .3)
        self.assert_solid(evaluated["result"]["geometry"])
        self.assertEqual(len(evaluated["result"]["geometry"]["faces"]), 44)
        # Actual surface hit resolves to the narrowest subface, not a triangle index.
        self.assertIn(self.scene.pick_face(self.wall, (-2, -.15, -1), (0, -1, 0)), cells)
        self.assertEqual(self.scene.pick_face(self.wall, (0, 0, 0), (1, 0, 0)), None)

    def test_subface_ids_survive_recalculation_and_save_reopen(self):
        self.add_grid()
        before = self.scene.evaluate_modifiers()[-1]["diagnostics"]["subface_ids"]
        self.assertTrue(self.scene.evaluate_modifiers()[-1]["cache_hit"])
        with tempfile.TemporaryDirectory() as folder:
            location = Path(folder) / "faces.sgeo"
            self.scene.save(location)
            reopened = SceneDocument.open(location, mode="write")
            self.assertEqual(before, reopened.evaluate_modifiers()[-1]["diagnostics"]["subface_ids"])
            self.assert_solid(reopened.evaluate_modifiers()[-1]["result"]["geometry"])

    def test_subdivide_extrude_and_bevel_remain_one_host(self):
        self.add_grid()
        cell = self.scene.evaluate_modifiers()[-1]["diagnostics"]["subface_ids"][0]
        self.add_extrusion(cell)
        self.assert_solid(self.scene.evaluate_modifiers()[-1]["result"]["geometry"])
        self.add_bevel(cell)
        final = self.scene.evaluate_modifiers()[-1]
        self.assert_solid(final["result"]["geometry"])
        self.assertEqual(final["result"]["id"], self.wall)
        self.assertEqual(len(self.scene.model.objects), 1)
        self.assertEqual(self.scene.model.objects[0]["geometry"]["kind"], "box")
        from .sgeo.face_modifiers import _face_region
        info = final["result"]["metadata"]["slai.sgeo.face_lineage.v1"]
        center, _ = _face_region(info["base_box"], info["faces"][cell])
        self.assertEqual(self.scene.pick_face(self.wall, center, (0, -1, 0)), cell)

    def test_recess_subtracts_material_and_preserves_host(self):
        self.add_grid()
        cell = self.scene.evaluate_modifiers()[-1]["diagnostics"]["subface_ids"][0]
        self.add_extrusion(cell, mode="recess", direction="reverse")
        final = self.scene.evaluate_modifiers()[-1]["result"]
        self.assert_solid(final["geometry"])
        self.assertLess(analyze_mesh(final["geometry"]).signed_volume, 5.4)

    def test_inclined_affine_wall_uses_actual_face_normal_and_thickness(self):
        from .sgeo.tessellation import tessellate
        mesh = tessellate({"kind": "box", "center": [0, 0, 0], "size": [6, .3, 3]})
        for vertex in mesh["vertices"]:
            vertex[1] += .4 * vertex[2]
        inclined = self.scene.add_mesh(mesh["vertices"], mesh["faces"])
        front = next(ident for ident, face in self.scene.face_references(inclined).items()
                     if face["axis"] == 1 and face["sign"] == -1)
        self.scene.add_modifier(inclined, kind="sgeo.face_subdivide",
                                target_subelement={"kind": "face", "id": front},
                                parameters={"rows": 2, "columns": 3})
        first = self.scene.evaluate_modifiers()[-1]
        self.assert_solid(first["result"]["geometry"])
        cell = first["diagnostics"]["subface_ids"][0]
        self.scene.add_modifier(inclined, kind="sgeo.face_extrude",
                                target_subelement={"kind": "face", "id": cell},
                                parameters={"depth": .15, "mode": "add", "direction": "normal",
                                            "thickness_limit": .3},
                                evaluation_policy={"mode": "tolerance_controlled", "absolute": 1e-7})
        second = self.scene.evaluate_modifiers()[-1]
        self.assert_solid(second["result"]["geometry"])
        self.assertEqual(second["result"]["id"], inclined)
        # Physical thickness is 0.3 / sqrt(1+0.4**2), not the nominal 0.3.
        mid = self.scene.modifiers()[-1]["id"]
        self.scene.update_modifier(mid, parameters={"depth": .29, "mode": "add", "direction": "normal",
                                                    "thickness_limit": .3})
        with self.assertRaises(GeometryValidationError):
            self.scene.evaluate_modifiers()

    def test_missing_face_is_reported_not_remapped_after_count_edit(self):
        subdivision = self.add_grid()
        cell = self.scene.evaluate_modifiers()[-1]["diagnostics"]["subface_ids"][0]
        self.add_extrusion(cell)
        self.scene.update_modifier(subdivision, parameters={"rows": 3, "columns": 3})
        with self.assertRaises(ReferenceError):
            self.scene.evaluate_modifiers()

    def test_wrong_target_unsupported_surface_and_thickness_rejection(self):
        with self.assertRaises(ValidationError):
            self.scene.add_modifier(self.wall, kind="sgeo.face_subdivide",
                                    parameters={"rows": 2, "columns": 3})
        with self.assertRaises(ValidationError):
            self.add_extrusion(self.front, depth=.31)
        mesh = self.scene.evaluate_modifiers()
        self.assertEqual(mesh, [])
        from .sgeo.tessellation import tessellate
        source = tessellate({"kind": "box", "center": [0, 0, 0], "size": [1, 1, 1]})
        source["vertices"][6][0] += .2  # valid closed mesh, but not an affine box
        second = self.scene.add_mesh(source["vertices"], source["faces"])
        self.assertEqual(self.scene.face_references(second), {})
        from .common.ids import new_id
        self.scene.add_modifier(second, kind="sgeo.face_subdivide",
                                target_subelement={"kind": "face", "id": new_id()},
                                parameters={"rows": 2, "columns": 2})
        with self.assertRaises(ReferenceError):
            self.scene.evaluate_modifiers()


if __name__ == "__main__":
    unittest.main()
