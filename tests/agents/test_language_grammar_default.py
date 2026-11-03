"""Regression checks for the Language Agent's non-model grammar baseline."""

import unittest

from src.agents.language.grammar_processor import GrammarProcessor
from src.agents.language.nlp_engine import NLPEngine


class GrammarDefaultIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nlp = NLPEngine()
        cls.grammar = GrammarProcessor(rule_engine=cls.nlp.rule_engine)

    def test_nlp_and_grammar_reuse_the_same_rule_index(self) -> None:
        self.assertIs(self.grammar.rule_engine, self.nlp.rule_engine)

    def issues(self, text: str):
        analyzed = self.nlp.analyze_text(text)
        return self.grammar.analyze_tokens(
            analyzed.tokens,
            dependencies=analyzed.dependencies,
            full_text_snippet=text,
        ).issues

    def test_passive_agreement_replaces_only_auxiliary(self) -> None:
        text = "The components is connected."
        issues = [issue for issue in self.issues(text) if issue.replacement == "are"]
        self.assertEqual(len(issues), 1)
        start, end_inclusive = issues[0].source_text_char_span
        self.assertEqual(text[start:end_inclusive + 1], "is")
        self.assertEqual(self.issues("The components are connected."), [])

    def test_pronoun_verb_agreement_and_config_person_normalization(self) -> None:
        text = "She go to school."
        issues = [issue for issue in self.issues(text) if issue.replacement == "goes"]
        self.assertEqual(len(issues), 1)
        start, end_inclusive = issues[0].source_text_char_span
        self.assertEqual(text[start:end_inclusive + 1], "go")
        self.assertEqual(self.issues("She goes to school."), [])
        self.assertEqual(self.issues("They go to school."), [])


if __name__ == "__main__":
    unittest.main()
