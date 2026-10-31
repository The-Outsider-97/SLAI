"""SLDC 1.1 regression coverage for optional font runs and managed TOC annotations."""
import unittest

from src.utils import sldc


class TocTypographyTests(unittest.TestCase):
    def test_inline_font_round_trip_requires_1_1(self):
        span = {"type": "format", "start": 0, "end": 5, "font_family": "Georgia", "font_size_pt": 12}
        p = sldc.new_package("Fonts", [sldc.make_block("paragraph", "Hello", spans=[span])])
        with self.assertRaises(sldc.SLDCUnsupportedError):
            sldc.validate_package(p)
        p.version = "1.1.0"
        sldc.validate_package(p)
        restored = sldc.loads_package(sldc.dumps_package(p))
        self.assertEqual(restored.document["blocks"][0]["spans"][0], span)
        self.assertEqual(restored.version, "1.1.0")

    def test_inline_font_validation_rejects_malformed_values(self):
        for changes in ({"font_family": "Bad; CSS"}, {"font_size_pt": 0}, {"font_size_pt": True}):
            span = {"type": "format", "start": 0, "end": 5, **changes}
            p = sldc.new_package("Fonts", [sldc.make_block("paragraph", "Hello", spans=[span])])
            p.version = "1.1.0"
            with self.assertRaises(sldc.SLDCValidationError):
                sldc.validate_package(p)

    def test_toc_extension_rejects_dangling_heading(self):
        metadata = {"slai.sldc.toc": {"id": "a" * 24, "role": "entry", "heading": "missing"}}
        p = sldc.new_package("TOC", [sldc.make_block("list_item", "Intro", extensions=metadata)])
        p.version = "1.1.0"
        with self.assertRaises(sldc.SLDCValidationError):
            sldc.validate_package(p)


if __name__ == "__main__":
    unittest.main()
