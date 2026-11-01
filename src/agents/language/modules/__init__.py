"""Lazy exports for language implementation modules.

Directly importing the BPE tokenizer must not import the transformer, rules, and
spell checker as an incidental side effect.
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Tokenizer
    "BPEToken": (".language_tokenizer", "BPEToken"),
    "PreToken": (".language_tokenizer", "PreToken"),
    "BPETrainingSummary": (".language_tokenizer", "BPETrainingSummary"),
    "LanguageTokenizerStats": (".language_tokenizer", "LanguageTokenizerStats"),
    "TokenizationResult": (".language_tokenizer", "TokenizationResult"),
    "LanguageTokenizer": (".language_tokenizer", "LanguageTokenizer"),
    # Transformer
    "BeamCandidate": (".language_transformer", "BeamCandidate"),
    "BeamSearchOutput": (".language_transformer", "BeamSearchOutput"),
    "SequenceScore": (".language_transformer", "SequenceScore"),
    "EmbeddingOutput": (".language_transformer", "EmbeddingOutput"),
    "TaskAdaptationResult": (".language_transformer", "TaskAdaptationResult"),
    "LanguageTransformerStats": (".language_transformer", "LanguageTransformerStats"),
    "LanguageTransformer": (".language_transformer", "LanguageTransformer"),
    # Rules
    "RuleConfidence": (".rules", "RuleConfidence"),
    "LexicalEntry": (".rules", "LexicalEntry"),
    "VerbInflection": (".rules", "VerbInflection"),
    "RuleToken": (".rules", "RuleToken"),
    "DependencyRelation": (".rules", "DependencyRelation"),
    "RuleApplicationResult": (".rules", "RuleApplicationResult"),
    "LanguageRulesStats": (".rules", "LanguageRulesStats"),
    "Rules": (".rules", "Rules"),
    # Spell checker
    "WordEntry": (".spell_checker", "WordEntry"),
    "SpellSuggestion": (".spell_checker", "SpellSuggestion"),
    "SpellCheckResult": (".spell_checker", "SpellCheckResult"),
    "TextSpellCheckResult": (".spell_checker", "TextSpellCheckResult"),
    "SpellCheckerStats": (".spell_checker", "SpellCheckerStats"),
    "SpellChecker": (".spell_checker", "SpellChecker"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
