"""SLDC 1.1 paragraph-layout feature gate, validation and save/reopen regression."""
import unittest

from src.utils import sldc


class ParagraphLayout(unittest.TestCase):
    def make(self, attrs):
        block = sldc.make_block("paragraph", "A paragraph", block_id="p1")
        block["attributes"] = attrs
        return sldc.new_package("Paragraph layout", [block])

    def test_feature_is_versioned_and_survives_round_trip(self):
        values = {
            "line_height": 1.5,
            "space_before_pt": 8,
            "space_after_pt": 12,
            "indent_mm": 5,
            "first_line_indent_mm": -2,
        }
        package = self.make(values)
        self.assertIn("paragraph-layout", sldc._required_features(package.document))
        with self.assertRaises(sldc.SLDCUnsupportedError):
            sldc.dumps_package(package)
        package.version = "1.1.0"
        reopened = sldc.loads_package(sldc.dumps_package(package))
        self.assertEqual(reopened.document["blocks"][0]["attributes"], values)

    def test_invalid_direct_paragraph_values_are_rejected(self):
        for key, values in {
            "line_height": (-1, 0.49, 5.1, True, float("nan")),
            "space_before_pt": (-1, 1001, "10", float("inf")),
            "space_after_pt": (-1, 1001, False),
            "indent_mm": (-1, 1001, "2"),
            "first_line_indent_mm": (-101, 1001, float("-inf")),
        }.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    package = self.make({key: value})
                    package.version = "1.1.0"
                    with self.assertRaises(sldc.SLDCValidationError):
                        sldc.validate_package(package)

    def test_plain_document_requires_no_paragraph_layout_feature(self):
        package = self.make({"align": "center"})
        self.assertNotIn("paragraph-layout", sldc._required_features(package.document))
        sldc.dumps_package(package)


if __name__ == "__main__":
    unittest.main()
