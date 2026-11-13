"""Regression tests for the v2.4 Language Agent spelling suggestions.

The spell checker itself requires only its language helpers, YAML, and Metaphone.
In lightweight test environments where torch is not installed, skip the
unrelated eager language-cache import while loading the module under test.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types

import pytest


if importlib.util.find_spec("torch") is None:
    package = "src.agents.language.utils"
    namespace = types.ModuleType(package)
    namespace.__path__ = ["src/agents/language/utils"]
    sys.modules.setdefault(package, namespace)

from src.agents.language.modules import spell_checker as spell_module  # noqa: E402
from src.agents.language.modules.spell_checker import SpellChecker  # noqa: E402


@pytest.fixture(scope="module")
def checker() -> SpellChecker:
    return SpellChecker()


@pytest.mark.parametrize(
    ("misspelling", "expected"),
    [
        ("diagramm", "diagram"),
        ("adress", "address"),
        ("recieve", "receive"),
        ("teh", "the"),
        ("algoritm", "algorithm"),
        ("enviroment", "environment"),
        ("comunication", "communication"),
        ("writting", "writing"),
        ("fonetic", "phonetic"),
        ("definately", "definitely"),
        ("Diagramm", "Diagram"),
        ("DIAGRAMM", "DIAGRAM"),
    ],
)
def test_best_correction(checker: SpellChecker, misspelling: str, expected: str) -> None:
    result = checker.check(misspelling)
    assert not result.is_correct
    assert result.best is not None
    assert result.best.word == expected
    assert len(result.suggestions) <= checker.max_suggestions
    assert len({item.word for item in result.suggestions}) == len(result.suggestions)
    assert all(item.edit_distance <= checker.max_edit_distance for item in result.suggestions)


@pytest.mark.parametrize(
    "word",
    [
        "diagram", "diagrams", "communication", "environment", "metadata",
        "lifecycle", "workflow", "configuration", "interoperability", "SLAI",
        "LANTRA", "BaseAgent", "SharedMemory", "SGEO", "SDRW",
    ],
)
def test_valid_vocabulary_is_not_flagged(checker: SpellChecker, word: str) -> None:
    result = checker.check(word)
    assert result.is_correct
    assert result.suggestions == ()


def test_structured_text_preserves_original_spans(checker: SpellChecker) -> None:
    source = "The diagramm, shows metadata."
    result = checker.check_text(source, auto_correct=False)
    typo = next(item for item in result.checks if item.original == "diagramm")
    assert typo.span is not None
    assert source[typo.span[0] : typo.span[1]] == "diagramm"
    assert typo.best is not None and typo.best.word == "diagram"
    assert result.original_text == result.corrected_text == source
    assert checker.check_text(source, auto_correct=True, threshold=0.0).corrected_text == (
        "The diagram, shows metadata."
    )


def test_personal_dictionary_is_persisted(checker: SpellChecker, tmp_path) -> None:
    old_path = checker.user_dictionary_path
    old_personal = checker.personal_words.copy()
    path = tmp_path / "user_words.json"
    try:
        checker.user_dictionary_path = path
        checker.add_words(["Zyphoria"])
        assert checker.is_correct("Zyphoria")
        assert json.loads(path.read_text(encoding="utf-8")) == ["zyphoria"]
        checker.remove_words(["Zyphoria"])
        assert not checker.is_correct("Zyphoria")
        assert json.loads(path.read_text(encoding="utf-8")) == []
    finally:
        checker.user_dictionary_path = old_path
        checker.personal_words = old_personal


def test_candidate_selection_protects_close_matches(checker: SpellChecker) -> None:
    pool = checker._candidate_pool("diagramm")
    assert "diagram" in pool
    assert len(pool) <= checker.max_candidates
    assert checker.suggest("diagramm")[0] == "diagram"


def test_configured_personal_and_domain_dictionaries_reload(monkeypatch, tmp_path) -> None:
    core = tmp_path / "core.json"
    domain = tmp_path / "domain.json"
    personal = tmp_path / "personal.json"
    core.write_text(json.dumps({"diagram": 1}), encoding="utf-8")
    domain.write_text(json.dumps({"words": {"domainword": {"pos": ["noun"]}}}), encoding="utf-8")
    personal.write_text(json.dumps(["customterm"]), encoding="utf-8")
    settings = {
        "structured_wordlist_path": str(core),
        "main_wordlist_path": None,
        "wordlist_path": None,
        "domain_dictionary_paths": [str(domain)],
        "user_dictionary_path": str(personal),
        "enable_phonetic": False,
    }
    monkeypatch.setattr(spell_module, "get_config_section", lambda _section: settings)
    fresh = SpellChecker()
    assert fresh.is_correct("diagram")
    assert fresh.is_correct("domainword")
    assert fresh.is_correct("customterm")
    fresh.add_words(["newterm"])
    another = SpellChecker()
    assert another.is_correct("newterm")
    assert json.loads(personal.read_text(encoding="utf-8")) == ["customterm", "newterm"]


def test_repeated_calls_are_deterministic(checker: SpellChecker) -> None:
    first = checker.suggest_with_scores("diagramm")
    assert first == checker.suggest_with_scores("diagramm")
