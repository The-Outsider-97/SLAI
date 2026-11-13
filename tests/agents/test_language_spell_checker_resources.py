"""Regression: SLAI's large English wordlist must load from any application CWD."""
from __future__ import annotations

import os
import tempfile
import unittest


class SpellCheckerResourceTests(unittest.TestCase):
    def test_writer_working_directory_loads_real_dictionary(self) -> None:
        from src.agents.language.modules.spell_checker import SpellChecker

        original = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as outside_slai:
                os.chdir(outside_slai)
                checker = SpellChecker()
                self.assertGreaterEqual(checker.stats().word_count, 300_000)
                for word in ("The", "component", "execution", "diagram", "diagrams", "metadata", "lifecycle", "workflow"):
                    self.assertTrue(checker.check(word).is_correct, word)
                correction = checker.check("diagramm")
                self.assertFalse(correction.is_correct)
                self.assertTrue(correction.suggestions)
                self.assertEqual(correction.suggestions[0].word, "diagram")
        finally:
            os.chdir(original)


if __name__ == "__main__":
    unittest.main()
